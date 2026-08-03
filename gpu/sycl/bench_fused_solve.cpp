/* bench_fused_solve.cpp
 *
 * Full batched AX=B solve (geqrf -> apply Q^T -> trsm): the primitives+fusion
 * strategy (ONE kernel) vs running the operations separately. Answers "is
 * fusing the whole workflow faster?". Three variants of the SAME solve:
 *   fused          : one parallel_for, all three primitives, data resident
 *   split-resident : three parallel_for, data stays on device (isolates the
 *                    cost of extra kernel launches)
 *   split-copies   : three ops each bracketed by host<->device copies of its
 *                    buffers -- the naive batched-call pattern the C launchers
 *                    use (input/output on the host)
 *
 * IMPORTANT: with no Intel GPU present this runs on the OpenCL CPU device, where
 * kernel launches are cheap and the "copies" are host memcpys -- so it
 * UNDERSTATES the fusion win relative to a real GPU, whose kernel-dispatch
 * latency and PCIe transfers make separate launches and per-call copies far more
 * expensive, precisely in the small-matrix regime compact batching targets.
 *
 * Self-checking (the fused solve must recover the known X); non-zero exit on
 * failure. Build: icpx -fsycl -O3 -march=native -std=c++17 -I ../../src \
 * -I . bench_fused_solve.cpp -o bench (the -I../../src is unused here; only the
 * primitives header is needed).
 *
 * Assisted-by: Claude:claude-opus-4.8
 */
#include <sycl/sycl.hpp>

#include "cqr_compact_sycl.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using clk = std::chrono::steady_clock;
static double secs(clk::duration d) { return std::chrono::duration<double>(d).count(); }
constexpr int V = 8;

int main()
{
    sycl::queue q{sycl::default_selector_v};
    auto sgs = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
    std::fprintf(stderr, "device: %s\n",
                 q.get_device().get_info<sycl::info::device::name>().c_str());
    if (std::find(sgs.begin(), sgs.end(), (size_t)V) == sgs.end()) {
        std::printf("sub-group size %d unsupported on this device; skipping\n", V);
        return 77;
    }

    const int nm = 512, nrhs = 4;
    const int ns[] = {16, 24, 32, 48, 64, 96, 128};
    std::mt19937_64 rng(1);
    std::uniform_real_distribution<double> dist(-1, 1);

    std::printf("full batched AX=B solve (geqrf -> Q^T -> trsm), nm=%d nrhs=%d V=%d\n", nm,
                nrhs, V);
    std::printf("(OpenCL CPU device: launches cheap, copies are host memcpys -- a lower "
                "bound on the GPU fusion win)\n");
    std::printf("   n | fused us | split-resid us | split-copies us | fused/resid | "
                "fused/copies | err\n");
    std::printf("-----+----------+----------------+-----------------+-------------+---------"
                "-----+-----\n");

    double worst = 0;
    for (int n : ns) {
        const int ng = (nm + V - 1) / V, k = n;
        const size_t na = (size_t)ng * n * n * V, nt = (size_t)ng * n * V,
                     nb = (size_t)ng * n * nrhs * V;

        /* host pristine packed A and B = A X, with X(:,c) = c+1 */
        std::vector<double> ap0(na, 0), bp0(nb, 0);
        for (int g = 0; g < ng; ++g)
            for (int v = 0; v < V; ++v) {
                std::vector<double> A((size_t)n * n);
                for (auto &x : A) x = dist(rng);
                for (int i = 0; i < n; ++i) A[i + (size_t)i * n] += 2.0 * n;
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < n; ++i)
                        ap0[(size_t)g * n * n * V + ((size_t)j * n + i) * V + v] =
                            A[i + (size_t)j * n];
                for (int c = 0; c < nrhs; ++c)
                    for (int i = 0; i < n; ++i) {
                        double s = 0;
                        for (int l = 0; l < n; ++l) s += A[i + (size_t)l * n];
                        bp0[(size_t)g * n * nrhs * V + ((size_t)c * n + i) * V + v] =
                            (c + 1) * s;
                    }
            }

        double *ap = sycl::malloc_device<double>(na, q), *tp = sycl::malloc_device<double>(nt, q),
               *bp = sycl::malloc_device<double>(nb, q);
        double *ap0d = sycl::malloc_device<double>(na, q),
               *bp0d = sycl::malloc_device<double>(nb, q);
        q.memcpy(ap0d, ap0.data(), na * 8).wait();
        q.memcpy(bp0d, bp0.data(), nb * 8).wait();
        std::vector<double> ap_h(na), tp_h(nt), bp_h(nb);

        auto nd = sycl::nd_range<1>(sycl::range<1>((size_t)ng * V), sycl::range<1>(V));
        auto geqrf = [=](sycl::handler &h) {
            h.parallel_for(nd, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(V)]] {
                cqr::gpu::geqrf_slot<double, V>((int)it.get_group(0),
                                                (int)it.get_local_id(0), ap, n, tp, n, n);
            });
        };
        auto ormqr = [=](sycl::handler &h) {
            h.parallel_for(nd, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(V)]] {
                cqr::gpu::ormqr_slot<double, V>((int)it.get_group(0),
                                                (int)it.get_local_id(0), ap, n, tp, bp, n,
                                                n, nrhs, k, true);
            });
        };
        auto trsm = [=](sycl::handler &h) {
            h.parallel_for(nd, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(V)]] {
                cqr::gpu::trsm_upper_slot<double, V>((int)it.get_group(0),
                                                     (int)it.get_local_id(0), ap, n, bp, n,
                                                     n, nrhs);
            });
        };

        auto reset_dev = [&] {
            q.memcpy(ap, ap0d, na * 8);
            q.memcpy(bp, bp0d, nb * 8);
            q.wait();
        };
        auto fused = [&] {
            q.submit([=](sycl::handler &h) {
                 h.parallel_for(nd, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(V)]] {
                     int g = it.get_group(0), v = it.get_local_id(0);
                     cqr::gpu::geqrf_slot<double, V>(g, v, ap, n, tp, n, n);
                     cqr::gpu::ormqr_slot<double, V>(g, v, ap, n, tp, bp, n, n, nrhs, k, true);
                     cqr::gpu::trsm_upper_slot<double, V>(g, v, ap, n, bp, n, n, nrhs);
                 });
             }).wait();
        };
        auto split_res = [&] {
            q.submit(geqrf).wait();
            q.submit(ormqr).wait();
            q.submit(trsm).wait();
        };
        auto split_cpy = [&] { /* each op copies its buffers host<->device */
            q.memcpy(ap, ap_h.data(), na * 8).wait();
            q.submit(geqrf).wait();
            q.memcpy(ap_h.data(), ap, na * 8).wait();
            q.memcpy(tp_h.data(), tp, nt * 8).wait();
            q.memcpy(ap, ap_h.data(), na * 8).wait();
            q.memcpy(tp, tp_h.data(), nt * 8).wait();
            q.memcpy(bp, bp_h.data(), nb * 8).wait();
            q.submit(ormqr).wait();
            q.memcpy(bp_h.data(), bp, nb * 8).wait();
            q.memcpy(ap, ap_h.data(), na * 8).wait();
            q.memcpy(bp, bp_h.data(), nb * 8).wait();
            q.submit(trsm).wait();
            q.memcpy(bp_h.data(), bp, nb * 8).wait();
        };

        reset_dev();
        fused();
        std::vector<double> out(nb);
        q.memcpy(out.data(), bp, nb * 8).wait();
        double err = 0;
        for (int g = 0; g < ng; ++g)
            for (int v = 0; v < V && g * V + v < nm; ++v)
                for (int c = 0; c < nrhs; ++c)
                    for (int i = 0; i < n; ++i)
                        err = std::max(err, std::fabs(
                            out[(size_t)g * n * nrhs * V + ((size_t)c * n + i) * V + v] -
                            (c + 1)));
        worst = std::max(worst, err);

        auto best = [&](auto &&reset, auto &&op) {
            double b = 1e30;
            reset();
            op();
            for (int s = 0; s < 40; ++s) {
                reset();
                auto t0 = clk::now();
                op();
                b = std::min(b, secs(clk::now() - t0));
            }
            return b;
        };
        double tf = best(reset_dev, fused);
        double tr = best(reset_dev, split_res);
        double tc = best([&] { ap_h = ap0; bp_h = bp0; }, split_cpy);

        std::printf("%4d | %8.1f | %14.1f | %15.1f | %10.2fx | %11.2fx | %.0e\n", n,
                    tf * 1e6, tr * 1e6, tc * 1e6, tr / tf, tc / tf, err);

        sycl::free(ap, q);
        sycl::free(tp, q);
        sycl::free(bp, q);
        sycl::free(ap0d, q);
        sycl::free(bp0d, q);
    }

    const double tol = 1e-9;
    std::printf("\nworst fused-solve error: %.1e (tol %.0e) -> %s\n", worst, tol,
                worst <= tol ? "OK" : "FAIL");
    return worst <= tol ? 0 : 1;
}
