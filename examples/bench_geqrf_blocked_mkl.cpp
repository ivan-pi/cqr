/* bench_geqrf_blocked_mkl.cpp
 *
 * The blocked (Level-3, 2-D register-tiled larfb) compact QR factorization vs the
 * two yardsticks: Intel's own mkl_dgeqrf_compact and a per-matrix LAPACKE_dgeqrf
 * loop, across the square-size range, single-thread. It answers "how much faster
 * is the blocked compact geqrf than MKL's compact kernel, and than non-batched
 * LAPACK?" (cqr_geqrf_compact_cache_blocking.md).
 *
 * GFLOP/s = nmat*(4/3)n^3 / t. The blocked time is the best over NB in {8,16,32}
 * (a small tuning sweep). The batch size is capped per n so the working set stays
 * bounded (five ~n^2*nmat buffers live). The blocked factor is checked against
 * per-matrix LAPACK (norm-scaled elementwise); a relative error above 1e-9 aborts
 * non-zero, so this doubles as a correctness gate.
 *
 * Usage:  bench_geqrf_blocked_mkl [nmat] [reps]      (defaults: 512 matrices, 4 reps)
 * Build with -O3 -march=native. Needs Intel MKL (compact API + LAPACKE).
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#include <mkl.h>
#include <mkl_compact.h>

#include "cqr_mkl_ext.h"
#include "cqr_geqrf_compact.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>
#include <algorithm>

using clk = std::chrono::steady_clock;
using cqr::detail::geqrf_blocked_compact;
using cqr::detail::vlen_for_format;

static double gflop(int n)
{
    return (4.0 / 3.0) * n * (double)n * n * 1e-9;
}

static void run_blocked(int V, MKL_INT n, double *ap, double *tp, MKL_INT nmat, int NB,
                        int NC)
{
    switch (V) {
    case 2:
        geqrf_blocked_compact<double, 2, MKL_INT>(n, n, ap, n, tp, nmat, NB, NC);
        break;
    case 4:
        geqrf_blocked_compact<double, 4, MKL_INT>(n, n, ap, n, tp, nmat, NB, NC);
        break;
    default:
        geqrf_blocked_compact<double, 8, MKL_INT>(n, n, ap, n, tp, nmat, NB, NC);
        break;
    }
}

template <typename Reset, typename Timed>
static double best_time(int reps, Reset &&reset, Timed &&timed)
{
    reset();
    timed();
    double best = std::numeric_limits<double>::infinity();
    for (int r = 0; r < reps; ++r) {
        reset();
        auto t0 = clk::now();
        timed();
        best = std::min(best, std::chrono::duration<double>(clk::now() - t0).count());
    }
    return best;
}

int main(int argc, char **argv)
{
    const int NMAT_MAX = argc > 1 ? std::atoi(argv[1]) : 512;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 4;
    mkl_set_num_threads(1);
    LAPACKE_set_nancheck(0);
    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = vlen_for_format<double>(fmt);
    const int NC = 64;

    std::printf(
        "Blocked compact QR (2-D tiled larfb) vs mkl_dgeqrf_compact vs per-matrix "
        "LAPACKE_dgeqrf\n");
    std::printf("single-thread, nmat<=%d (capped by memory), reps=%d, V=%d; GFLOP/s and "
                "speedups\n\n",
                NMAT_MAX, reps, V);
    std::printf(
        "   n | nmat | cqr-blk GFLOP/s | mkl GFLOP/s | lapack GFLOP/s | blk/mkl | "
        "blk/lapack | NB | relerr\n");
    std::printf(
        "-----+------+-----------------+-------------+----------------+---------+--"
        "----------+----+--------\n");

    const int sizes[] = {16, 24, 32, 48, 64, 96, 128, 160, 192, 256, 320, 384, 500};
    double lg_mkl = 0, lg_lap = 0, gworst = 0;
    int ns = 0;

    for (int n : sizes) {
        const int m = n, k = n;
        /* cap the batch so each ~n^2*nmat buffer stays under ~150 MB (5 live) */
        const int nmat =
            std::min(NMAT_MAX, std::max(64, (int)(1.5e8 / ((double)n * n * 8))));
        const size_t sA = (size_t)m * n;
        std::vector<double> pool((size_t)nmat * sA);
        std::mt19937_64 rng(2025);
        std::uniform_real_distribution<double> d(-1, 1);
        for (int v = 0; v < nmat; ++v) {
            double *A = pool.data() + (size_t)v * sA;
            for (size_t i = 0; i < sA; ++i)
                A[i] = d(rng);
            for (int i = 0; i < n; ++i)
                A[i + (size_t)i * m] += 2.0 * n;
        }

        MKL_INT sza = mkl_dget_size_compact(m, n, fmt, nmat);
        MKL_INT szt = mkl_dget_size_compact(k, 1, fmt, nmat);
        std::vector<double> pristine(sza), work_ap(sza), tp(szt);
        std::vector<double *> Ap(nmat);
        for (int v = 0; v < nmat; ++v)
            Ap[v] = pool.data() + (size_t)v * sA;
        mkl_dgepack_compact(MKL_COL_MAJOR, m, n, Ap.data(), m, pristine.data(), m, fmt,
                            nmat);
        auto restore = [&] {
            std::memcpy(work_ap.data(), pristine.data(), sza * sizeof(double));
        };

        /* blocked cqr: best over NB */
        int bestNB = 8;
        double t_blk = std::numeric_limits<double>::infinity();
        for (int NB : {8, 16, 32}) {
            double t = best_time(reps, restore, [&] {
                run_blocked(V, n, work_ap.data(), tp.data(), nmat, NB, NC);
            });
            if (t < t_blk) {
                t_blk = t;
                bestNB = NB;
            }
        }

        /* mkl compact */
        MKL_INT lwork = -1, info;
        double wq;
        mkl_dgeqrf_compact(MKL_COL_MAJOR, m, n, work_ap.data(), m, tp.data(), &wq, -1,
                           &info, fmt, nmat);
        lwork = (MKL_INT)wq;
        std::vector<double> mwork(std::max<MKL_INT>(lwork, 1));
        double t_mkl = best_time(reps, restore, [&] {
            mkl_dgeqrf_compact(MKL_COL_MAJOR, m, n, work_ap.data(), m, tp.data(),
                               mwork.data(), lwork, &info, fmt, nmat);
        });

        /* per-matrix LAPACK */
        std::vector<double> lap(pool.size());
        std::vector<double> tau(k);
        double t_lap = best_time(
            reps, [&] { lap = pool; },
            [&] {
                for (int v = 0; v < nmat; ++v)
                    LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, lap.data() + (size_t)v * sA, m,
                                   tau.data());
            });

        /* correctness: blocked factor vs per-matrix LAPACK (norm-scaled) */
        restore();
        run_blocked(V, n, work_ap.data(), tp.data(), nmat, bestNB, NC);
        std::vector<double> H((size_t)nmat * sA), tt((size_t)nmat * k);
        std::vector<double *> Hp(nmat), Tp(nmat);
        for (int v = 0; v < nmat; ++v) {
            Hp[v] = H.data() + (size_t)v * sA;
            Tp[v] = tt.data() + (size_t)v * k;
        }
        mkl_dgeunpack_compact(MKL_COL_MAJOR, m, n, Hp.data(), m, work_ap.data(), m, fmt,
                              nmat);
        mkl_dgeunpack_compact(MKL_COL_MAJOR, k, 1, Tp.data(), k, tp.data(), k, fmt, nmat);
        double worst = 0;
        std::vector<double> Href(sA), tref(k);
        for (int v = 0; v < nmat; ++v) {
            std::copy(Ap[v], Ap[v] + sA, Href.begin());
            LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, Href.data(), m, tref.data());
            double num = 0, den = 0;
            for (size_t i = 0; i < sA; ++i) {
                num = std::max(num, std::fabs(H[(size_t)v * sA + i] - Href[i]));
                den = std::max(den, std::fabs(Href[i]));
            }
            for (int i = 0; i < k; ++i)
                num = std::max(num, std::fabs(tt[(size_t)v * k + i] - tref[i]));
            worst = std::max(worst, num / std::max(den, 1e-300));
        }

        double g_blk = nmat * gflop(n) / t_blk, g_mkl = nmat * gflop(n) / t_mkl,
               g_lap = nmat * gflop(n) / t_lap;
        lg_mkl += std::log(t_mkl / t_blk);
        lg_lap += std::log(t_lap / t_blk);
        ++ns;
        gworst = std::max(gworst, worst);
        std::printf(
            "%4d | %4d | %15.1f | %11.1f | %14.1f | %6.2fx | %9.2fx | %2d | %.1e\n", n,
            nmat, g_blk, g_mkl, g_lap, t_mkl / t_blk, t_lap / t_blk, bestNB, worst);
    }
    std::printf(
        "-----+------+-----------------+-------------+----------------+---------+--"
        "----------+----+--------\n");
    std::printf("geomean speedup: blocked cqr vs mkl_dgeqrf_compact = %.2fx ; vs "
                "per-matrix LAPACK = %.2fx\n",
                std::exp(lg_mkl / ns), std::exp(lg_lap / ns));
    if (gworst > 1e-9) {
        std::printf("FAILED: blocked factor diverged from LAPACK (relerr %.2e)\n",
                    gworst);
        return 1;
    }
    return 0;
}
