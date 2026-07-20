/* bench_geqrf_compact.cpp
 *
 * Throughput benchmark of the QR *factorization* over pools of many small
 * matrices, comparing three implementations of the same LAPACK ?geqrf math:
 *
 *   cqr-compact  cqr_mkl_dgeqrf_compact   (this project's batched SIMD kernel)
 *   mkl-compact  mkl_dgeqrf_compact       (Intel MKL's batched compact kernel)
 *   per-matrix   LAPACKE_dgeqrf           (conventional one-matrix-at-a-time)
 *
 * This is benchmark 1 of the design document (section 9): the compact batched
 * factorization against the standard per-matrix layout, with MKL's own compact
 * kernel as a second yardstick. To measure the factorization kernels rather than
 * data movement, the pool is packed into compact form once, up front; only the
 * factorization is timed, and the destroyed input is restored (untimed) before
 * each pass. The two compact paths are driven from an OpenMP outer loop over the
 * groups of V interleaved matrices -- the intended "outer multi-threaded loop"
 * usage -- with MKL's own threading pinned to 1; the per-matrix path parallelizes
 * over matrices the same way. The factorization is checked (untimed) against
 * per-matrix LAPACK, so the benchmark doubles as an integration test.
 *
 * Usage:  bench_geqrf_compact [nmat] [reps]      (defaults: 512 matrices, 3 reps)
 *
 * Build: needs Intel MKL plus this repo's cqr_mkl_ext; wired up by CMakeLists.txt
 * as the `bench_geqrf_compact` target. OpenMP is used when available.
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#include <mkl.h>
#include <mkl_compact.h>

#include "cqr_mkl_ext.h"
#include "cqr_mkl_alloc.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using clk = std::chrono::steady_clock;
using cqr::detail::vlen_for_format;

void check(bool cond, const char *what)
{
    if (!cond) {
        std::printf("FAILED: %s\n", what);
        std::exit(1);
    }
}

/* Standard LAPACK ?geqrf flop count (m >= n), in GFLOP. */
double geqrf_gflop(int m, int n)
{
    return (2.0 * m * n * (double)n - (2.0 / 3.0) * n * (double)n * n) * 1e-9;
}

const char *compact_format_name(MKL_COMPACT_PACK format)
{
    switch (format) {
    case MKL_COMPACT_SSE: return "SSE";
    case MKL_COMPACT_AVX: return "AVX";
    case MKL_COMPACT_AVX512: return "AVX512";
    default: return "unknown";
    }
}

/* A pool of `nmat` dense column-major m x n matrices, back to back in `a`
 * (m*n per matrix), well conditioned (diagonal-boosted). */
struct Pool {
    int m, n, nmat;
    std::vector<double> a; /* nmat * m*n */

    Pool(int m_, int n_, int nmat_)
        : m(m_), n(n_), nmat(nmat_), a((size_t)nmat_ * m_ * n_)
    {
        std::mt19937_64 rng(2025);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (int v = 0; v < nmat; ++v) {
            double *A = a.data() + (size_t)v * m * n;
            for (int i = 0; i < m * n; ++i)
                A[i] = dist(rng);
            for (int i = 0; i < std::min(m, n); ++i)
                A[i + (size_t)i * m] += 2.0 * n;
        }
    }
};

/* Best (minimum) wall time over `reps` timed passes, in seconds; `reset` runs
 * untimed before every pass to restore the input the factorization destroys. */
template <typename Reset, typename Timed>
double best_time(int reps, Reset &&reset, Timed &&timed)
{
    reset();
    timed(); /* warm-up (untimed) */
    double best = std::numeric_limits<double>::infinity();
    for (int r = 0; r < reps; ++r) {
        reset();
        auto t0 = clk::now();
        timed();
        best = std::min(best, std::chrono::duration<double>(clk::now() - t0).count());
    }
    return best;
}

/* Factor a pre-packed compact pool in place, one group of V per OpenMP
 * iteration (the intended outer-loop usage). `use_cqr` selects our kernel or
 * MKL's; both see the identical compact buffer. */
void factor_compact(bool use_cqr, double *ap, double *taup, int m, int n, int ngroups,
                    int V, MKL_COMPACT_PACK fmt, MKL_INT lwork)
{
    const int k = std::min(m, n);
#pragma omp parallel
    {
        std::vector<double> work((size_t)std::max<MKL_INT>(lwork, 1));
        MKL_INT info;
#pragma omp for schedule(static)
        for (int g = 0; g < ngroups; ++g) {
            double *apg = ap + (size_t)g * m * n * V; /* group stride m*n*V */
            double *taupg = taup + (size_t)g * k * V;
            if (use_cqr)
                cqr_mkl_dgeqrf_compact(MKL_COL_MAJOR, m, n, apg, m, taupg, work.data(),
                                       lwork, &info, fmt, V);
            else
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
double factor_error(const Pool &P, MKL_COMPACT_PACK fmt, int V)
{
    const int m = P.m, n = P.n, nmat = P.nmat, k = std::min(m, n);
    const size_t sA = (size_t)m * n;

    MKL_INT sz_a = mkl_dget_size_compact(m, n, fmt, nmat);
    MKL_INT sz_t = mkl_dget_size_compact(k, 1, fmt, nmat);
    auto ap = cqr::detail::mkl_alloc_bytes<double>(sz_a);
    auto tp = cqr::detail::mkl_alloc_bytes<double>(sz_t);

    std::vector<double *> Ap(nmat);
    for (int v = 0; v < nmat; ++v)
        Ap[v] = const_cast<double *>(P.a.data()) + v * sA;
    mkl_dgepack_compact(MKL_COL_MAJOR, m, n, Ap.data(), m, ap.get(), m, fmt, nmat);

    const int ngroups = (nmat + V - 1) / V;
    MKL_INT lwork = -1, info;
    double wq;
    cqr_mkl_dgeqrf_compact(MKL_COL_MAJOR, m, n, ap.get(), m, tp.get(), &wq, -1, &info,
                           fmt, V);
    lwork = (MKL_INT)wq;
    factor_compact(true, ap.get(), tp.get(), m, n, ngroups, V, fmt, lwork);

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
        const double *Av = P.a.data() + v * sA;
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

} /* anonymous namespace */

int main(int argc, char **argv)
{
    const int nmat = (argc > 1) ? std::atoi(argv[1]) : 512;
    const int reps = (argc > 2) ? std::atoi(argv[2]) : 3;
    check(nmat > 0 && reps > 0, "usage: bench_geqrf_compact [nmat>0] [reps>0]");

    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = vlen_for_format<double>(fmt);
    mkl_set_num_threads(1);
    LAPACKE_set_nancheck(0);

    int nthreads = 1;
#ifdef _OPENMP
#pragma omp parallel
#pragma omp single
    nthreads = omp_get_num_threads();
#endif

    /* Square sizes spanning the target range (dense in the emphasized region
     * below 170, then a few larger to show where the crossover happens). */
    constexpr std::array sizes = {8, 16, 24, 32, 48, 64, 96, 128, 170, 256, 384, 500};

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
        Pool P(m, n, nmat);
        const int ngroups = (nmat + V - 1) / V;

        /* pristine packed buffer + two working copies (cqr, mkl) */
        MKL_INT sz_a = mkl_dget_size_compact(m, n, fmt, nmat);
        MKL_INT sz_t = mkl_dget_size_compact(k, 1, fmt, nmat);
        auto pristine = cqr::detail::mkl_alloc_bytes<double>(sz_a);
        auto work_ap = cqr::detail::mkl_alloc_bytes<double>(sz_a);
        auto taup = cqr::detail::mkl_alloc_bytes<double>(sz_t);
        std::vector<double *> Ap(nmat);
        for (int v = 0; v < nmat; ++v)
            Ap[v] = P.a.data() + (size_t)v * m * n;
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

        std::vector<double> pool_work; /* standard-layout copy */

        /* both compact paths factor in place, so restore the packed input
         * (untimed) before each timed pass */
        auto restore = [&] { std::memcpy(work_ap.get(), pristine.get(), sz_a); };

        double t_cqr = best_time(reps, restore, [&] {
            factor_compact(true, work_ap.get(), taup.get(), m, n, ngroups, V, fmt,
                           lwork_cqr);
        });
        double t_mkl = best_time(reps, restore, [&] {
            factor_compact(false, work_ap.get(), taup.get(), m, n, ngroups, V, fmt,
                           lwork_mkl);
        });
        double t_lap = best_time(
            reps, [&] { pool_work = P.a; },
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
