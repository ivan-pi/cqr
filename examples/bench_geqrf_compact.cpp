/* bench_geqrf_compact.cpp
 *
 * Throughput benchmark of the QR *factorization* over pools of many small
 * matrices, comparing three implementations of the same LAPACK ?geqrf math:
 *
 *   cqr-compact  cqr_mkl_dgeqrf_compact   (this project's batched SIMD kernel)
 *   mkl-compact  mkl_dgeqrf_compact       (Intel MKL's batched compact kernel)
 *   per-matrix   LAPACKE_dgeqrf           (conventional one-matrix-at-a-time)
 *
 * It pits the compact batched factorization against the standard per-matrix
 * layout, with MKL's own compact kernel as a second yardstick (all three
 * benchmarks are documented in examples/BENCHMARKS.md). To measure the
 * factorization kernels rather than data movement, the pool is packed into
 * compact form once, up front; only the factorization is timed, and the
 * destroyed input is restored (untimed) before each pass. The cqr path is one
 * call on the whole pool -- the routine threads its own loop over groups. MKL's
 * compact kernel is not threaded here (sequential MKL; its threading is pinned
 * to 1 in any case), so it and the per-matrix LAPACK path are driven from an
 * OpenMP loop of the same thread count, group by group / matrix by matrix. The
 * factorization is checked (untimed) against per-matrix LAPACK, so the
 * benchmark doubles as an integration test.
 *
 * Usage:  bench_geqrf_compact [--size-sweep=nmin:nmax[:stride]] [--simdlen=2|4|8]
 *         [nmat] [reps]      (defaults: 512 matrices, 3 reps)
 *
 * With no --size-sweep it runs the 3-way comparison (cqr vs mkl_dgeqrf_compact
 * vs per-matrix LAPACK); with it, a cqr-only throughput scan over the size range.
 * --simdlen forces the interleave width (2/4/8) instead of the host's widest.
 *
 * Build: needs Intel MKL plus this repo's cqr_mkl_ext; wired up by CMakeLists.txt
 * as the `bench_geqrf_compact` target. OpenMP is used when available.
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#include <mkl.h>
#include <mkl_compact.h>

#include "cqr_mkl_alloc.h"
#include "bench_util.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>

namespace {

using namespace cqr::bench;

/* Standard LAPACK ?geqrf flop count (m >= n), in GFLOP. */
double geqrf_gflop(int m, int n)
{
    return (2.0 * m * n * (double)n - (2.0 / 3.0) * n * (double)n * n) * 1e-9;
}

/* A pool of `nmat` random m x n matrices, well conditioned (diagonal-boosted). */
MatrixPool make_pool(int m, int n, int nmat)
{
    MatrixPool P(nmat, m, n);
    std::mt19937_64 rng(2025);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int v = 0; v < nmat; ++v) {
        const auto A = P.matrix(v);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i)
                A(i, j) = dist(rng);
        for (int d = 0; d < std::min(m, n); ++d)
            A(d, d) += 2.0 * n;
    }
    return P;
}

/* Factor a pre-packed compact pool of nmat matrices in place. cqr: one call on
 * the whole pool, threaded inside the library. MKL: an OpenMP loop over the
 * groups of V (its compact kernel is not threaded in this build), so both paths
 * run on the same thread count. */
void factor_compact(bool use_cqr, double *ap, double *taup, int m, int n, int nmat, int V,
                    MKL_COMPACT_PACK fmt, MKL_INT lwork)
{
    if (use_cqr) {
        std::vector<double> work((size_t)std::max<MKL_INT>(lwork, 1));
        MKL_INT info;
        cqr_mkl_dgeqrf_compact(MKL_COL_MAJOR, m, n, ap, m, taup, work.data(), lwork,
                               &info, fmt, nmat);
        return;
    }
    const int k = std::min(m, n), ngroups = (nmat + V - 1) / V;
#pragma omp parallel
    {
        std::vector<double> work((size_t)std::max<MKL_INT>(lwork, 1));
        MKL_INT info;
#pragma omp for schedule(static)
        for (int g = 0; g < ngroups; ++g) {
            double *apg = ap + (size_t)g * m * n * V; /* group stride m*n*V */
            double *taupg = taup + (size_t)g * k * V;
            mkl_dgeqrf_compact(MKL_COL_MAJOR, m, n, apg, m, taupg, work.data(), lwork,
                               &info, fmt, V);
        }
    }
}

/* Per-matrix LAPACK factorization of a standard-layout pool copy in place. */
void factor_unbatched(double *a, int m, int n, int nmat)
{
    const int k = std::min(m, n);
#pragma omp parallel
    {
        std::vector<double> tau(k);
#pragma omp for schedule(static)
        for (int v = 0; v < nmat; ++v)
            LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, a + (size_t)v * m * n, m, tau.data());
    }
}

/* Relative factor error of the compact path vs per-matrix LAPACK: unpack the
 * compact (H, tau) and compare elementwise to a fresh LAPACKE_dgeqrf, scaled by
 * the matrix L1 norm. Untimed correctness gate. */
double factor_error(const MatrixPool &P, MKL_COMPACT_PACK fmt, int V)
{
    const int m = P.rows, n = P.cols, nmat = P.nmat, k = std::min(m, n);
    const size_t sA = P.stride();

    MKL_INT sz_a = mkl_dget_size_compact(m, n, fmt, nmat);
    MKL_INT sz_t = mkl_dget_size_compact(k, 1, fmt, nmat);
    auto ap = cqr::detail::mkl_alloc_bytes<double>(sz_a);
    auto tp = cqr::detail::mkl_alloc_bytes<double>(sz_t);

    auto Ap = P.base_ptrs();
    mkl_dgepack_compact(MKL_COL_MAJOR, m, n, Ap.data(), m, ap.get(), m, fmt, nmat);

    MKL_INT lwork = -1, info;
    double wq;
    cqr_mkl_dgeqrf_compact(MKL_COL_MAJOR, m, n, ap.get(), m, tp.get(), &wq, -1, &info,
                           fmt, V);
    lwork = (MKL_INT)wq;
    factor_compact(true, ap.get(), tp.get(), m, n, nmat, V, fmt, lwork);

    std::vector<double> H(nmat * sA), tau(nmat * (size_t)k);
    std::vector<double *> Hp(nmat), Tp(nmat);
    for (int v = 0; v < nmat; ++v) {
        Hp[v] = H.data() + v * sA;
        Tp[v] = tau.data() + v * (size_t)k;
    }
    mkl_dgeunpack_compact(MKL_COL_MAJOR, m, n, Hp.data(), m, ap.get(), m, fmt, nmat);
    mkl_dgeunpack_compact(MKL_COL_MAJOR, k, 1, Tp.data(), k, tp.get(), k, fmt, nmat);

    double worst = 0;
    std::vector<double> Href(sA), tref(k);
    for (int v = 0; v < nmat; ++v) {
        const double *Av = P.matrix(v).data;
        std::copy(Av, Av + sA, Href.begin());
        LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, Href.data(), m, tref.data());
        double num = 0, den = 0;
        for (size_t i = 0; i < sA; ++i) {
            num = std::max(num, std::abs(H[v * sA + i] - Href[i]));
            den = std::max(den, std::abs(Href[i]));
        }
        for (int i = 0; i < k; ++i)
            num = std::max(num, std::abs(tau[v * (size_t)k + i] - tref[i]));
        worst = std::max(worst, num / std::max(den, 1e-300));
    }
    return worst;
}

/* Single-kernel size sweep: factor a pre-packed pool with cqr at each n in
 * [nmin, nmax] (step stride) and print throughput only -- no MKL/LAPACK
 * cross-check, so it stays cheap and isolates the kernel. The point is the
 * staircase: n that is / is not a multiple of the interleave width V. The raw
 * best-pass time is printed next to the derived rates: it is the quantity they
 * come from (rate = nmat / time), so a total near the timer granularity flags a
 * noisy row -- raise nmat until it is comfortably above the clock resolution. */
void run_sweep(int nmat, int reps, int nmin, int nmax, int stride, MKL_COMPACT_PACK fmt,
               int V, int nthreads)
{
    std::printf("QR factorization size sweep: cqr_mkl_dgeqrf_compact only "
                "(throughput, no cross-check)\n");
    std::printf("matrices=%d  reps=%d  simdlen=%d (%s)  OpenMP threads=%d  (square, "
                "pre-packed)\n\n",
                nmat, reps, V, compact_format_name(fmt), nthreads);
    std::printf("   n |  total (s) | cqr GFLOP/s |   cqr mat/s\n");
    std::printf("-----+------------+-------------+-------------\n");

    for (int n = nmin; n <= nmax; n += stride) {
        const int m = n, k = n;
        const MatrixPool P = make_pool(m, n, nmat);

        MKL_INT sz_a = mkl_dget_size_compact(m, n, fmt, nmat);
        MKL_INT sz_t = mkl_dget_size_compact(k, 1, fmt, nmat);
        auto pristine = cqr::detail::mkl_alloc_bytes<double>(sz_a);
        auto work_ap = cqr::detail::mkl_alloc_bytes<double>(sz_a);
        auto taup = cqr::detail::mkl_alloc_bytes<double>(sz_t);
        auto Ap = P.base_ptrs();
        mkl_dgepack_compact(MKL_COL_MAJOR, m, n, Ap.data(), m, pristine.get(), m, fmt,
                            nmat);

        double wq;
        MKL_INT info;
        cqr_mkl_dgeqrf_compact(MKL_COL_MAJOR, m, n, work_ap.get(), m, taup.get(), &wq, -1,
                               &info, fmt, V);
        const MKL_INT lwork = (MKL_INT)wq;

        auto restore = [&] { std::memcpy(work_ap.get(), pristine.get(), sz_a); };
        const double t = best_time(reps, restore, [&] {
            factor_compact(true, work_ap.get(), taup.get(), m, n, nmat, V, fmt, lwork);
        });
        std::printf("%4d | %10.3e | %11.2f | %11.2e\n", n, t,
                    nmat * geqrf_gflop(m, n) / t, nmat / t);
    }
    std::printf("-----+------------+-------------+-------------\n");
}

} /* anonymous namespace */

int main(int argc, char **argv)
{
    const CmdArgs args(argc, argv, "bench_geqrf_compact");
    const int nmat = args.nmat, reps = args.reps, V = args.V;
    const MKL_COMPACT_PACK fmt = args.fmt;

    /* Pin MKL's internal threading: the OpenMP outer loop is the only
     * parallelism. LAPACKE NaN-checking off so the per-matrix path is timed clean. */
    mkl_set_num_threads(1);
    LAPACKE_set_nancheck(0);
    const int nthreads = omp_threads();

    if (args.sweep) {
        run_sweep(nmat, reps, args.sweep_min, args.sweep_max, args.sweep_step, fmt, V,
                  nthreads);
        return 0;
    }

    /* Square sizes spanning the target range, dense below 170. Deliberately mixes
     * sizes that are not multiples of the SIMD width V -- 30, 45, 60, 105, 168,
     * from 2-D/3-D RBF-FD stencils -- with the round powers, so the remainder
     * handling (the staircase SIMD effect) is visible; then a few larger sizes for
     * the crossover. Use --size-sweep for a finer cqr-only scan. */
    constexpr std::array sizes = {8,  16,  24,  30,  32,  45,  48,  60, 64,
                                  96, 105, 128, 168, 170, 256, 384, 500};

    std::printf("QR factorization throughput: cqr_mkl_dgeqrf_compact vs "
                "mkl_dgeqrf_compact vs per-matrix LAPACKE_dgeqrf\n");
    std::printf("matrices=%d  reps=%d  simdlen=%d (%s)  OpenMP threads=%d  (square, "
                "pre-packed)\n\n",
                nmat, reps, V, compact_format_name(fmt), nthreads);
    /* Throughput as matrices/second (scientific) so it stays legible across the
     * whole size range; three speedup ratios show where the wins come from. The
     * error column is elementwise (H, tau) vs per-matrix LAPACKE_dgeqrf. */
    std::printf("   n | cqr GFLOP/s |   cqr mat/s |   mkl mat/s | lapack mat/s | "
                "cqr/lap | mkl/lap | cqr/mkl | relerr(vs LAPACK)\n");
    std::printf(
        "-----+-------------+-------------+-------------+--------------+---------+"
        "---------+---------+------------------\n");

    double log_speed_vs_lapack = 0.0;
    for (int n : sizes) {
        const int m = n, k = n;
        const MatrixPool P = make_pool(m, n, nmat);

        /* pristine packed buffer + two working copies (cqr, mkl) */
        MKL_INT sz_a = mkl_dget_size_compact(m, n, fmt, nmat);
        MKL_INT sz_t = mkl_dget_size_compact(k, 1, fmt, nmat);
        auto pristine = cqr::detail::mkl_alloc_bytes<double>(sz_a);
        auto work_ap = cqr::detail::mkl_alloc_bytes<double>(sz_a);
        auto taup = cqr::detail::mkl_alloc_bytes<double>(sz_t);
        auto Ap = P.base_ptrs();
        mkl_dgepack_compact(MKL_COL_MAJOR, m, n, Ap.data(), m, pristine.get(), m, fmt,
                            nmat);

        /* Each routine reports its own optimal lwork (MKL's compact geqrf needs
         * real scratch; ours needs none). Query both and size per path. */
        double wq;
        MKL_INT info;
        cqr_mkl_dgeqrf_compact(MKL_COL_MAJOR, m, n, work_ap.get(), m, taup.get(), &wq, -1,
                               &info, fmt, V);
        const MKL_INT lwork_cqr = (MKL_INT)wq;
        mkl_dgeqrf_compact(MKL_COL_MAJOR, m, n, work_ap.get(), m, taup.get(), &wq, -1,
                           &info, fmt, V);
        const MKL_INT lwork_mkl = (MKL_INT)wq;

        aligned_vector<double> pool_work; /* standard-layout copy (aligned like pool) */

        /* both compact paths factor in place, so restore the packed input
         * (untimed) before each timed pass */
        auto restore = [&] { std::memcpy(work_ap.get(), pristine.get(), sz_a); };

        double t_cqr = best_time(reps, restore, [&] {
            factor_compact(true, work_ap.get(), taup.get(), m, n, nmat, V, fmt,
                           lwork_cqr);
        });
        double t_mkl = best_time(reps, restore, [&] {
            factor_compact(false, work_ap.get(), taup.get(), m, n, nmat, V, fmt,
                           lwork_mkl);
        });
        double t_lap = best_time(
            reps, [&] { pool_work = P.storage; },
            [&] { factor_unbatched(pool_work.data(), m, n, nmat); });

        const double rel = factor_error(P, fmt, V);
        check(rel <= 1e-9, "compact factorization matches LAPACK");

        const double sp_lap = t_lap / t_cqr;     /* cqr speedup over LAPACK */
        const double sp_mkl_lap = t_lap / t_mkl; /* MKL speedup over LAPACK */
        const double sp_mkl = t_mkl / t_cqr;     /* cqr speedup over MKL    */
        const double gflops_cqr = nmat * geqrf_gflop(m, n) / t_cqr;
        log_speed_vs_lapack += std::log(sp_lap);
        std::printf("%4d | %11.2f | %11.2e | %11.2e | %12.2e | %6.2fx | %6.2fx | "
                    "%6.2fx | %.2e\n",
                    n, gflops_cqr, nmat / t_cqr, nmat / t_mkl, nmat / t_lap, sp_lap,
                    sp_mkl_lap, sp_mkl, rel);
    }

    std::printf(
        "-----+-------------+-------------+-------------+--------------+---------+"
        "---------+---------+------------------\n");
    std::printf("geometric-mean speedup (cqr compact vs per-matrix LAPACK): %.2fx\n",
                std::exp(log_speed_vs_lapack / sizes.size()));
    return 0;
}
