/* bench_ormqr_sycl_vs_vec.cpp
 *
 * ormqr (apply Q^T) throughput: the SYCL kernel vs the vector-types CPU kernel
 * (cqr::detail::ormqr_compact_group, GNU vector_size), same compact layout and
 * same batch, all cores for both.
 *
 * IMPORTANT: with no Intel GPU present this runs the SYCL kernel on the OpenCL
 * CPU device -- so it is a CPU-vs-CPU codegen comparison on one Xeon, NOT the
 * Xe story and NOT predictive of GPU throughput. Its value is to quantify one
 * thing the design doc calls out: pinning the sub-group to the interleave width
 * V so the runtime vectorizes V work-items into one SIMD op (batch-in-SIMD,
 * exactly what the vector-types kernel does by hand). Three kernels are timed:
 *   vec          : GNU-vector kernel, OpenMP over the V-matrix groups
 *   SG-implicit  : SYCL nd_range, reqd_sub_group_size(V), per-lane indexing
 *   SG-explicit  : SYCL nd_range, reqd_sub_group_size(V), group_load/group_store
 *                  block ops (the Intel explicit-SIMD variant, ormqr_slot_sg)
 * Kernel-only timing (buffers resident; no per-op host<->device copy).
 *
 * Build (needs a SYCL compiler + OpenMP; V below is the AVX-512 double width):
 *   icpx -fsycl -O3 -march=native -qopenmp -std=c++17 -I ../../src \
 *        bench_ormqr_sycl_vs_vec.cpp -o bench
 * Returns non-zero if any SYCL result disagrees with the vec reference, so it
 * doubles as a cross-check.
 *
 * Assisted-by: Claude:claude-opus-4.8
 */
#include <sycl/sycl.hpp>
#include "cqr_compact.hpp"      // vector-types kernel
#include "cqr_compact_sycl.hpp" // explicit sub-group primitive (ormqr_slot_sg)
#include <omp.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>

using clk = std::chrono::steady_clock;
static double secs(clk::duration d) { return std::chrono::duration<double>(d).count(); }

constexpr int V = 8; // AVX-512 double interleave width (a valid Xe sub-group size too)

// SYCL ormqr (trans='T', side='L', column-major) on resident device pointers,
// calling the header's ormqr_slot primitive. `sg` selects the nd_range +
// reqd_sub_group_size(V) launch; otherwise a plain range parallel_for -- the
// launch shape is the only thing this wrapper varies.
template <bool sg>
static sycl::event sycl_ormqr(sycl::queue &q, int n, int nrhs, int nm, const double *ap,
                              const double *tp, double *bp)
{
    const int k = n, m = n, ldap = n, ldbp = n, ng = (nm + V - 1) / V;
    // Call the SHIPPED primitive rather than a copy of it. This lambda used to
    // hand-inline ormqr_slot's JB=4 body; a benchmark that re-implements the
    // kernel it measures silently stops measuring the real one the moment
    // either is tuned, so the "SG-implicit" column could drift away from the
    // code it advertises without any test failing.
    auto body = [=](int g, int v) {
        cqr::gpu::ormqr_slot<double, V>(g, v, ap, ldap, tp, bp, ldbp, m, nrhs, k,
                                        true); // trans: apply Q^T
    };
    if constexpr (sg)
        return q.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t)ng * V), sycl::range<1>(V)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(V)]] {
                body((int)it.get_group(0), (int)it.get_local_id(0));
            });
    else
        return q.parallel_for(sycl::range<1>((size_t)ng * V), [=](sycl::id<1> id) {
            body((int)id[0] / V, (int)id[0] % V);
        });
}

// EXPLICIT sub-group variant: block group_load/group_store (header primitive
// ormqr_slot_sg), nd_range + reqd_sub_group_size(V).
static sycl::event sycl_ormqr_sg_explicit(sycl::queue &q, int n, int nrhs, int nm,
                                          const double *ap, const double *tp, double *bp)
{
    const int ng = (nm + V - 1) / V;
    return q.parallel_for(
        sycl::nd_range<1>(sycl::range<1>((size_t)ng * V), sycl::range<1>(V)),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(V)]] {
            cqr::gpu::ormqr_slot_sg<double, V>(it.get_sub_group(), (int)it.get_group(0),
                                               ap, n, tp, bp, n, n, nrhs, n, true);
        });
}

// vector-types path: OpenMP over the V-matrix groups, tuned contiguous kernel.
static void vec_ormqr(int n, int nrhs, int nm, const double *ap, const double *tp,
                      double *bp)
{
    const int ng = (nm + V - 1) / V;
    const size_t sa = (size_t)n * n * V, st = (size_t)n * V, sb = (size_t)n * nrhs * V;
#pragma omp parallel for schedule(static)
    for (int g = 0; g < ng; ++g)
        cqr::detail::ormqr_compact_group<double, V>(cqr::detail::Direction::Forward, n,
                                                    nrhs, n, ap + (size_t)g * sa, n,
                                                    tp + (size_t)g * st, bp + (size_t)g * sb, n);
}

template <class F> static double best_per_op(F &&f, double target = 0.15, int passes = 3)
{
    auto run = [&](int it) {
        auto t0 = clk::now();
        for (int i = 0; i < it; ++i) f();
        return secs(clk::now() - t0);
    };
    run(1);
    double t1 = std::max(run(1), 1e-9);
    int it = std::max(3, (int)(target / t1));
    double best = 1e30;
    for (int p = 0; p < passes; ++p) best = std::min(best, run(it) / it);
    return best;
}

int main()
{
    sycl::queue q{sycl::default_selector_v};
    auto sizes = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
    const bool sg_ok = std::find(sizes.begin(), sizes.end(), (size_t)V) != sizes.end();
    std::fprintf(stderr, "device: %s | omp threads: %d | sub-group %d supported: %s\n",
                 q.get_device().get_info<sycl::info::device::name>().c_str(),
                 omp_get_max_threads(), V, sg_ok ? "yes" : "NO (SG column skipped)");

    const int nm = 512;
    const int ns[] = {10, 20, 30, 50, 75, 100, 125, 150};
    const int nrhss[] = {1, 4, 8};
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> d(-1, 1);
    double worst_rel = 0;

    for (int nrhs : nrhss) {
        std::printf("\n=== nrhs=%d, batch=%d, V=%d (AVX-512 double), %d cores ===\n", nrhs,
                    nm, V, omp_get_max_threads());
        std::printf("   n | vec GF/s | SG-implicit GF/s | SG-explicit GF/s | expl/impl | "
                    "expl/vec | xcheck\n");
        std::printf("-----+----------+------------------+------------------+-----------+"
                    "----------+-------\n");
        for (int n : ns) {
            const int ng = (nm + V - 1) / V;
            std::vector<double> ap((size_t)ng * n * n * V), tp((size_t)ng * n * V),
                bp0((size_t)ng * n * nrhs * V);
            for (auto &x : ap) x = d(rng) * 0.1;
            for (auto &x : tp) x = d(rng) * 0.1 + 1.0;
            for (auto &x : bp0) x = d(rng);
            double fl = 0;
            for (int kk = 0; kk < n; ++kk) fl += 4.0 * (n - kk) * nrhs;
            const double work = fl * nm;

            std::vector<double> bpv = bp0;
            double tv = best_per_op([&] { vec_ormqr(n, nrhs, nm, ap.data(), tp.data(), bpv.data()); });

            double *apd = sycl::malloc_device<double>(ap.size(), q);
            double *tpd = sycl::malloc_device<double>(tp.size(), q);
            double *bpd = sycl::malloc_device<double>(bp0.size(), q);
            q.memcpy(apd, ap.data(), ap.size() * 8).wait();
            q.memcpy(tpd, tp.data(), tp.size() * 8).wait();
            q.memcpy(bpd, bp0.data(), bp0.size() * 8).wait();
            double tsg = sg_ok
                ? best_per_op([&] { sycl_ormqr<true>(q, n, nrhs, nm, apd, tpd, bpd).wait(); })
                : 0.0;
            double tse = sg_ok
                ? best_per_op([&] { sycl_ormqr_sg_explicit(q, n, nrhs, nm, apd, tpd, bpd).wait(); })
                : 0.0;

            // cross-check the explicit-SG kernel against the vec reference
            // (implicit-SG is validated separately by the unit suites)
            std::vector<double> bpa = bp0, bpb(bp0.size());
            vec_ormqr(n, nrhs, nm, ap.data(), tp.data(), bpa.data());
            q.memcpy(bpd, bp0.data(), bp0.size() * 8).wait();
            (sg_ok ? sycl_ormqr_sg_explicit(q, n, nrhs, nm, apd, tpd, bpd)
                   : sycl_ormqr<false>(q, n, nrhs, nm, apd, tpd, bpd)).wait();
            q.memcpy(bpb.data(), bpd, bp0.size() * 8).wait();
            double err = 0, scale = 0;
            for (size_t i = 0; i < bpa.size(); ++i) {
                err = std::max(err, std::fabs(bpa[i] - bpb[i]));
                scale = std::max(scale, std::fabs(bpa[i]));
            }
            double rel = err / (scale > 0 ? scale : 1);
            worst_rel = std::max(worst_rel, rel);
            sycl::free(apd, q);
            sycl::free(tpd, q);
            sycl::free(bpd, q);

            if (sg_ok)
                std::printf("%4d | %8.2f | %16.2f | %16.2f | %8.2fx | %7.2fx | %.0e\n", n,
                            work / tv / 1e9, work / tsg / 1e9, work / tse / 1e9, tsg / tse,
                            tv / tse, rel);
            else
                std::printf("%4d | %8.2f | %16s | %16s | %9s | %8s | %.0e\n", n,
                            work / tv / 1e9, "n/a", "n/a", "n/a", "n/a", rel);
        }
    }
    const double tol = 1e-10;
    std::printf("\nworst cross-check rel err: %.1e (tol %.0e) -> %s\n", worst_rel, tol,
                worst_rel <= tol ? "OK" : "FAIL");
    return worst_rel <= tol ? 0 : 1;
}
