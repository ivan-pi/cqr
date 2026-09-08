/* bench_potrf_compact.cpp
 *
 * Throughput benchmark of the Cholesky *factorization* over pools of many small
 * symmetric positive-definite (SPD) matrices, comparing three implementations of
 * the same LAPACK ?potrf math:
 *
 *   cqr-compact  cqr_mkl_dpotrf_compact   (this project's batched SIMD kernel)
 *   mkl-compact  mkl_dpotrf_compact       (Intel MKL's batched compact kernel)
 *   per-matrix   LAPACKE_dpotrf           (conventional one-matrix-at-a-time)
 *
 * This is the potrf analogue of bench_geqrf_compact (all three benchmarks are
 * documented in examples/BENCHMARKS.md): the compact batched factorization
 * against the standard per-matrix layout, with
 * MKL's own compact kernel as a second yardstick. Every matrix is factored on its
 * tuned path -- column-major, lower triangle (A = L L^T) -- which is the natural
 * Cholesky data flow. To measure the factorization kernels rather than data
 * movement, the pool is packed into compact form once, up front; only the
 * factorization is timed, and the destroyed input is restored (untimed) before
 * each pass. The cqr path is one call on the whole pool -- the routine threads
 * its own loop over groups. MKL's compact kernel is not threaded here
 * (sequential MKL; its threading is pinned to 1 in any case), so it and the
 * per-matrix LAPACK path are driven from an OpenMP loop of the same thread
 * count. The factorization is checked (untimed) against per-matrix LAPACK, so
 * the benchmark doubles as an integration test.
 *
 * Unlike ?geqrf, ?potrf needs no workspace, so there is no lwork query and no
 * per-thread work array (as in mkl_?potrf_compact / LAPACKE_dpotrf).
 *
 * Usage:  bench_potrf_compact [--size-sweep=nmin:nmax[:stride]] [--simdlen=2|4|8]
 *         [nmat] [reps]      (defaults: 512 matrices, 3 reps)
 *
 * With no --size-sweep it runs the 3-way comparison (cqr vs mkl_dpotrf_compact
 * vs per-matrix LAPACK); with it, a cqr-only throughput scan over the size range.
 * --simdlen forces the interleave width (2/4/8) instead of the host's widest.
 *
 * Build: needs Intel MKL plus this repo's cqr_mkl_ext; wired up by CMakeLists.txt
 * as the `bench_potrf_compact` target. OpenMP is used when available. For a fair
 * cqr-vs-MKL comparison, build with host-tuned flags (e.g.
 * `-DCMAKE_CXX_FLAGS="-O3 -march=native"`) so the open compact kernel emits the
 * full vector width, matching MKL's AVX-512 runtime dispatch.
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

/* Standard LAPACK ?potrf flop count in GFLOP: n^3/3 + n^2/2 + n/6 (adds +
 * mults, the classic LAWN 41 count; the n square roots are not counted, as in
 * LAPACK's own timing). */
double potrf_gflop(int n)
{
    const double dn = n;
    return (dn * dn * dn / 3.0 + dn * dn / 2.0 + dn / 6.0) * 1e-9;
}

/* A pool of `nmat` n x n symmetric positive-definite matrices. Each is built
 * symmetric with random off-diagonals in [-1,1] and a diagonal of 2n, so it is
 * strictly diagonally dominant (row off-diagonal magnitudes sum to at most
 * n-1 < 2n) and therefore SPD and well conditioned -- the O(n^2) analogue of
 * the geqrf pool's diagonal boost, without an O(n^3) M^T M product. The full
 * matrix is stored (both triangles) so the per-matrix LAPACK path and the
 * compact pack see identical symmetric input; each routine reads only the
 * lower triangle. */
MatrixPool make_pool(int n, int nmat)
{
    MatrixPool P(nmat, n, n);
    std::mt19937_64 rng(2025);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int v = 0; v < nmat; ++v) {
        const auto A = P.matrix(v);
        for (int j = 0; j < n; ++j) {
            for (int i = j + 1; i < n; ++i) {
                double x = dist(rng);
                A(i, j) = x; /* lower */
                A(j, i) = x; /* mirror to upper (symmetric) */
            }
            A(j, j) = 2.0 * n; /* diagonal dominant -> SPD */
        }
    }
    return P;
}

/* Factor a pre-packed compact pool of nmat matrices in place, column-major
 * lower (A = L L^T). cqr: one call on the whole pool, threaded inside the
 * library. MKL: an OpenMP loop over the groups of V (its compact kernel is not
 * threaded in this build), so both paths run on the same thread count. */
void factor_compact(bool use_cqr, double *ap, int n, int nmat, int V,
                    MKL_COMPACT_PACK fmt)
{
    MKL_INT info;
    if (use_cqr) {
        cqr_mkl_dpotrf_compact(MKL_COL_MAJOR, MKL_LOWER, n, ap, n, &info, fmt, nmat);
        return;
    }
    const int ngroups = (nmat + V - 1) / V;
#pragma omp parallel for schedule(static) private(info)
    for (int g = 0; g < ngroups; ++g) {
        double *apg = ap + (size_t)g * n * n * V; /* group stride n*n*V */
        mkl_dpotrf_compact(MKL_COL_MAJOR, MKL_LOWER, n, apg, n, &info, fmt, V);
    }
}

/* Per-matrix LAPACK Cholesky of a standard-layout pool copy in place (lower). */
void factor_unbatched(double *a, int n, int nmat)
{
#pragma omp parallel for schedule(static)
    for (int v = 0; v < nmat; ++v)
        LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', n, a + (size_t)v * n * n, n);
}

/* Relative factor error of the compact path vs per-matrix LAPACK: unpack the
 * compact factor and compare its lower triangle elementwise to a fresh
 * LAPACKE_dpotrf, scaled by the factor's L1 norm. The SPD Cholesky factor is
 * unique (positive diagonal), so this elementwise difference is a sharp signal.
 * Untimed correctness gate. */
double factor_error(const MatrixPool &P, MKL_COMPACT_PACK fmt, int V)
{
    const int n = P.rows, nmat = P.nmat;
    const size_t sA = P.stride();

    MKL_INT sz_a = mkl_dget_size_compact(n, n, fmt, nmat);
    auto ap = cqr::detail::mkl_alloc_bytes<double>(sz_a);

    auto Ap = P.base_ptrs();
    mkl_dgepack_compact(MKL_COL_MAJOR, n, n, Ap.data(), n, ap.get(), n, fmt, nmat);

    factor_compact(true, ap.get(), n, nmat, V, fmt);

    std::vector<double> H(nmat * sA);
    std::vector<double *> Hp(nmat);
    for (int v = 0; v < nmat; ++v)
        Hp[v] = H.data() + v * sA;
    mkl_dgeunpack_compact(MKL_COL_MAJOR, n, n, Hp.data(), n, ap.get(), n, fmt, nmat);

    double worst = 0;
    std::vector<double> Href(sA);
    for (int v = 0; v < nmat; ++v) {
        const double *Av = P.matrix(v).data;
        std::copy(Av, Av + sA, Href.begin());
        LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', n, Href.data(), n);
        /* compare only the lower triangle (i >= j): the factor L, uniquely
         * defined, vs LAPACK's; the strict upper triangle is untouched by both. */
        double num = 0, den = 0;
        for (int j = 0; j < n; ++j)
            for (int i = j; i < n; ++i) {
                const size_t off = i + (size_t)j * n;
                num = std::max(num, std::abs(H[v * sA + off] - Href[off]));
                den = std::max(den, std::abs(Href[off]));
            }
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
    std::printf("Cholesky factorization size sweep: cqr_mkl_dpotrf_compact only "
                "(throughput, no cross-check)\n");
    std::printf("matrices=%d  reps=%d  simdlen=%d (%s)  OpenMP threads=%d  (SPD, "
                "col-major lower, pre-packed)\n\n",
                nmat, reps, V, compact_format_name(fmt), nthreads);
    std::printf("   n |  total (s) | cqr GFLOP/s |   cqr mat/s\n");
    std::printf("-----+------------+-------------+-------------\n");

    for (int n = nmin; n <= nmax; n += stride) {
        const MatrixPool P = make_pool(n, nmat);

        MKL_INT sz_a = mkl_dget_size_compact(n, n, fmt, nmat);
        auto pristine = cqr::detail::mkl_alloc_bytes<double>(sz_a);
        auto work_ap = cqr::detail::mkl_alloc_bytes<double>(sz_a);
        auto Ap = P.base_ptrs();
        mkl_dgepack_compact(MKL_COL_MAJOR, n, n, Ap.data(), n, pristine.get(), n, fmt,
                            nmat);

        auto restore = [&] { std::memcpy(work_ap.get(), pristine.get(), sz_a); };
        const double t = best_time(
            reps, restore, [&] { factor_compact(true, work_ap.get(), n, nmat, V, fmt); });
        std::printf("%4d | %10.3e | %11.2f | %11.2e\n", n, t, nmat * potrf_gflop(n) / t,
                    nmat / t);
    }
    std::printf("-----+------------+-------------+-------------\n");
}

} /* anonymous namespace */

int main(int argc, char **argv)
{
    const CmdArgs args(argc, argv, "bench_potrf_compact");
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

    /* Square sizes spanning the target range (order 3..500, emphasis below 170).
     * Deliberately mixes sizes that are not multiples of the SIMD width V -- 30,
     * 45, 60, 105, 168, from 2-D/3-D RBF-FD stencils -- with the round powers, so
     * the remainder handling (the staircase SIMD effect) is visible; then a few
     * larger sizes for the crossover. Use --size-sweep for a finer cqr-only scan. */
    constexpr std::array sizes = {8,  16,  24,  30,  32,  45,  48,  60, 64,
                                  96, 105, 128, 168, 170, 256, 384, 500};

    std::printf("Cholesky factorization throughput: cqr_mkl_dpotrf_compact vs "
                "mkl_dpotrf_compact vs per-matrix LAPACKE_dpotrf\n");
    std::printf("matrices=%d  reps=%d  simdlen=%d (%s)  OpenMP threads=%d  (SPD, "
                "col-major lower, pre-packed)\n\n",
                nmat, reps, V, compact_format_name(fmt), nthreads);
    /* Throughput as matrices/second (scientific) so it stays legible across the
     * whole size range; three speedup ratios show where the wins come from. The
     * error column is elementwise (lower triangle) vs per-matrix LAPACKE_dpotrf. */
    std::printf("   n | cqr GFLOP/s |   cqr mat/s |   mkl mat/s | lapack mat/s | "
                "cqr/lap | mkl/lap | cqr/mkl | relerr(vs LAPACK)\n");
    std::printf(
        "-----+-------------+-------------+-------------+--------------+---------+"
        "---------+---------+------------------\n");

    double log_speed_vs_lapack = 0.0;
    for (int n : sizes) {
        const MatrixPool P = make_pool(n, nmat);

        /* pristine packed buffer + two working copies (cqr, mkl) */
        MKL_INT sz_a = mkl_dget_size_compact(n, n, fmt, nmat);
        auto pristine = cqr::detail::mkl_alloc_bytes<double>(sz_a);
        auto work_ap = cqr::detail::mkl_alloc_bytes<double>(sz_a);
        auto Ap = P.base_ptrs();
        mkl_dgepack_compact(MKL_COL_MAJOR, n, n, Ap.data(), n, pristine.get(), n, fmt,
                            nmat);

        aligned_vector<double> pool_work; /* standard-layout copy (aligned like pool) */

        /* both compact paths factor in place, so restore the packed input
         * (untimed) before each timed pass */
        auto restore = [&] { std::memcpy(work_ap.get(), pristine.get(), sz_a); };

        double t_cqr = best_time(
            reps, restore, [&] { factor_compact(true, work_ap.get(), n, nmat, V, fmt); });
        double t_mkl = best_time(reps, restore, [&] {
            factor_compact(false, work_ap.get(), n, nmat, V, fmt);
        });
        double t_lap = best_time(
            reps, [&] { pool_work = P.storage; },
            [&] { factor_unbatched(pool_work.data(), n, nmat); });

        const double rel = factor_error(P, fmt, V);
        check(rel <= 1e-9, "compact factorization matches LAPACK");

        const double sp_lap = t_lap / t_cqr;     /* cqr speedup over LAPACK */
        const double sp_mkl_lap = t_lap / t_mkl; /* MKL speedup over LAPACK */
        const double sp_mkl = t_mkl / t_cqr;     /* cqr speedup over MKL    */
        const double gflops_cqr = nmat * potrf_gflop(n) / t_cqr;
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
