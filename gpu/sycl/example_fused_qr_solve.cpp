/* example_fused_qr_solve.cpp
 *
 * The realistic usage pattern for the SYCL primitives: the USER owns a single
 * parallel_for over the batch and FUSES every step into one kernel --
 *   fill A + build B   ->   geqrf   ->   apply Q^T   ->   triangular solve
 *   ->   unpack the result
 * -- with the cqr::gpu building blocks (cqr_compact_sycl.hpp). No pre-pack pass,
 * no per-step kernels, and the only host<->device transfer is copying the final
 * solution out: A and B are generated on the device and consumed in place.
 *
 * Contrast with the batched C launchers (dgeqrf_compact / dormqr_compact), which
 * each own their own parallel_for and copy in/out per call -- three launches and
 * six copies for the same solve. Fusing is what makes a batch of many small
 * matrices worthwhile on a GPU.
 *
 * Self-checking: each A_v is strongly diagonally dominant with a KNOWN solution
 * X(:,c) = c+1 (so B_v = A_v X), and the recovered X is checked to working
 * precision. Returns non-zero on failure, so it doubles as a test.
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cqr_compact_sycl.hpp"

namespace {

constexpr int V = 8; /* interleave width = sub-group size (AVX-512 / Xe SIMD8) */

/* Deterministic per-element value in [-1,1) -- a device-side stand-in for a
 * host-filled matrix, so nothing about A is copied from the host. */
inline double gen(int idx, int i, int j)
{
    unsigned h = static_cast<unsigned>(idx) * 2654435761u ^
                 static_cast<unsigned>(i) * 40503u ^ static_cast<unsigned>(j) * 77557u;
    h ^= h >> 13;
    h *= 0x5bd1e995u;
    h ^= h >> 15;
    return (static_cast<int>(h & 0xffffu) - 32768) / 32768.0;
}

/* Solve nm systems A_v X = B_v, everything fused in one kernel; return the max
 * deviation of the recovered X from the known X(:,c) = c+1. */
double fused_solve(sycl::queue &q, int n, int nrhs, int nm)
{
    const int ng = (nm + V - 1) / V;
    double *ap = sycl::malloc_device<double>((size_t)ng * n * n * V, q);
    double *tp = sycl::malloc_device<double>((size_t)ng * n * V, q);
    double *bp = sycl::malloc_device<double>((size_t)ng * n * nrhs * V, q);
    double *xout = sycl::malloc_device<double>((size_t)nm * n * nrhs, q);

    q.parallel_for(
         sycl::nd_range<1>(sycl::range<1>((size_t)ng * V), sycl::range<1>(V)),
         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(V)]] {
             const int g = it.get_group(0), v = it.get_local_id(0);
             const int idx = g * V + v;

             auto A = cqr::gpu::col_major<double, V>(ap, g, v, n, n);
             auto B = cqr::gpu::col_major<double, V>(bp, g, v, n, nrhs);

             /* (1) FILL A directly into the compact slot (fused pack, no copy),
              *     diagonally dominant so the solve is well conditioned. */
             for (int j = 0; j < n; ++j)
                 for (int i = 0; i < n; ++i)
                     A(i, j) = gen(idx, i, j);
             for (int i = 0; i < n; ++i)
                 A(i, i) += 2.0 * n;

             /* (2) BUILD B = A X for the known X(:,c) = c+1 (before geqrf
              *     overwrites A). B(i,c) = (c+1) * sum_l A(i,l). */
             for (int c = 0; c < nrhs; ++c)
                 for (int i = 0; i < n; ++i) {
                     double s = 0.0;
                     for (int l = 0; l < n; ++l)
                         s += A(i, l);
                     B(i, c) = (c + 1) * s;
                 }

             /* (3) FACTOR, (4) APPLY Q^T, (5) SOLVE -- three primitives, no
              *     intervening launches or copies; A/B stay in place. */
             cqr::gpu::geqrf_slot<double, V>(g, v, ap, n, tp, n, n);
             cqr::gpu::ormqr_slot<double, V>(g, v, ap, n, tp, bp, n, n, nrhs, n, true);
             cqr::gpu::trsm_upper_slot<double, V>(g, v, ap, n, bp, n, n, nrhs);

             /* (6) UNPACK to a plain per-matrix result buffer (fused unpack). */
             if (idx < nm)
                 for (int c = 0; c < nrhs; ++c)
                     for (int i = 0; i < n; ++i)
                         xout[(size_t)idx * n * nrhs + (size_t)c * n + i] = B(i, c);
         })
        .wait();

    std::vector<double> x((size_t)nm * n * nrhs);
    q.memcpy(x.data(), xout, x.size() * sizeof(double)).wait();

    double err = 0.0;
    for (int idx = 0; idx < nm; ++idx)
        for (int c = 0; c < nrhs; ++c)
            for (int i = 0; i < n; ++i)
                err = std::max(err, std::fabs(x[(size_t)idx * n * nrhs + (size_t)c * n + i] -
                                              (c + 1)));

    sycl::free(ap, q);
    sycl::free(tp, q);
    sycl::free(bp, q);
    sycl::free(xout, q);
    return err;
}

} /* namespace */

int main()
{
    sycl::queue q{sycl::default_selector_v};
    auto sgs = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
    std::fprintf(stderr, "device: %s\n",
                 q.get_device().get_info<sycl::info::device::name>().c_str());
    if (std::find(sgs.begin(), sgs.end(), (size_t)V) == sgs.end()) {
        std::printf("sub-group size %d unsupported on this device; skipping\n", V);
        return 77; /* CTest "skipped" */
    }

    const double eps = 2.220446049250313e-16;
    int fails = 0;
    std::printf("fused batched AX=B (one kernel: fill -> geqrf -> Q^T -> trsm -> unpack)\n");
    for (int n : {16, 40, 64}) {
        const int nrhs = 4, nm = 512;
        const double err = fused_solve(q, n, nrhs, nm);
        const double tol = 1e5 * eps * n;
        const bool ok = err <= tol;
        std::printf("  n=%-3d nrhs=%d nm=%d | max|X-Xexact| = %.2e (tol %.1e) %s\n", n, nrhs,
                    nm, err, tol, ok ? "OK" : "FAIL");
        fails += !ok;
    }
    if (fails) {
        std::printf("\n%d CHECK(S) FAILED\n", fails);
        return 1;
    }
    std::printf("\nall checks passed\n");
    return 0;
}
