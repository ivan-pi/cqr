/* bench_qr_compact.cpp
 *
 * Throughput benchmark: solving many small square systems A_v X_v = B_v with
 * the QR pipeline, comparing the Intel MKL Compact (interleaved, batched) path
 * against the conventional per-matrix LAPACK path.
 *
 *   batched      mkl_dgeqrf_compact -> ext_mkl_dormqr_compact -> mkl_dtrsm_compact
 *   non-batched  LAPACKE_dgeqrf     -> LAPACKE_dormqr          -> cblas_dtrsm
 *
 * Both paths run the identical math (A = Q R; X = R^{-1} Q^T B); only the data
 * layout differs. The compact path packs `V` matrices (the SIMD vector length
 * of the active compact format) into interleaved storage and factors them with
 * one SIMD sweep; the conventional path walks one dense matrix at a time. This
 * mirrors the intended application, where matrices arrive on the fly and are
 * processed in groups of the vector length -- except that here a fixed pool of
 * matrices is built up front so the two paths factor exactly the same data.
 *
 * Methodology
 *   * A pool of `nmat` matrices is generated once per size (square, diagonally
 *     dominant so the solve is well conditioned), with a known exact solution
 *     X == 1 so B_v = A_v 1; each path must recover 1 (a correctness gate runs
 *     alongside the timing).
 *   * The batched path compacts/uncompacts each group of `V` on the fly inside
 *     the timed loop (the pool itself stays dense), matching the application.
 *   * The non-batched path factors in place (the application does not reuse the
 *     matrix), so no per-matrix copy is timed; a destroyable working copy of
 *     the pool is refreshed before each pass, outside the timed region. It also
 *     turns LAPACKE NaN-checking off (LAPACKE_set_nancheck) so the per-matrix
 *     driver is timed without that overhead.
 *   * The outer loop over the pool is parallelised with OpenMP; MKL's own
 *     threading is pinned to 1 so the outer loop is the only parallelism.
 *   * Each size is timed `reps` times and the best (minimum) wall time is kept.
 *   * A geometric mean of the per-size speedups is reported at the end -- the
 *     scale-invariant average that does not let one size dominate.
 *
 * Usage:  bench_qr_compact [nmat] [reps]      (defaults: 1000 matrices, 3 reps)
 *
 * Build: needs Intel MKL (the compact API is an MKL extension) plus this repo's
 * ext_mkl_ormqr_compact; wired up by CMakeLists.txt as the `bench_qr_compact`
 * target. OpenMP is used when available.
 */

#include <mkl.h>
#include <mkl_compact.h>

#include "ext_mkl_ormqr_compact.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using clk = std::chrono::steady_clock;

/* Report and abort on the spot if cond is false. */
void check(bool cond, const char *what)
{
    if (!cond) { std::printf("FAILED: %s\n", what); std::exit(1); }
}

/* Interleave width V for the active compact format (doubles): MKL packs
 * V = (SIMD register bytes) / sizeof(double). Mirrors vlen_for_format() in
 * ext_mkl_ormqr_compact.cpp. */
int vlen_for_format(MKL_COMPACT_PACK fmt)
{
    switch (fmt) {
    case MKL_COMPACT_SSE:    return 16 / (int)sizeof(double);
    case MKL_COMPACT_AVX:    return 32 / (int)sizeof(double);
    case MKL_COMPACT_AVX512: return 64 / (int)sizeof(double);
    default:                 return 0;
    }
}

/* A pool of `nmat` dense column-major square matrices of order n, each stored
 * back to back in `a` (n*n per matrix), with the matching right-hand sides in
 * `b` (n per matrix; single RHS). The exact solution is X == 1, so each RHS is
 * the row sum b_v = A_v 1, and a correct solve returns all ones. */
struct Pool {
    int n, nmat;
    std::vector<double> a;   /* nmat * n*n */
    std::vector<double> b;   /* nmat * n   */

    Pool(int n_, int nmat_) : n(n_), nmat(nmat_),
        a((size_t)nmat_ * n_ * n_), b((size_t)nmat_ * n_)
    {
        std::mt19937_64 rng(42);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (int v = 0; v < nmat; ++v) {
            double *A = a.data() + (size_t)v * n * n;
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) A[i + (size_t)j * n] = dist(rng);
            for (int i = 0; i < n; ++i) A[i + (size_t)i * n] += 2.0 * n; /* diag dominant */
            double *B = b.data() + (size_t)v * n;
            for (int i = 0; i < n; ++i) {                 /* B = A * ones */
                double s = 0.0;
                for (int j = 0; j < n; ++j) s += A[i + (size_t)j * n];
                B[i] = s;
            }
        }
    }
};

/* Largest deviation of a computed solution (expected all ones) from 1. */
double sol_error(const double *x, int n)
{
    double e = 0.0;
    for (int i = 0; i < n; ++i) e = std::max(e, std::fabs(x[i] - 1.0));
    return e;
}

/* ===== batched path: compact group-of-V pipeline ======================= *
 * Process the pool in groups of V, packing/factoring/solving/unpacking each
 * group inside the timed region. Returns the max solution error. */
double run_batched(const Pool &P, MKL_COMPACT_PACK fmt, int V)
{
    const int n = P.n, nmat = P.nmat, nrhs = 1;
    const int ngroups = (nmat + V - 1) / V;
    double maxerr = 0.0;

#pragma omp parallel reduction(max : maxerr)
    {
        /* Per-thread compact buffers, sized for a full group of V. */
        const int align = 64;
        double *ap   = (double *)mkl_malloc(mkl_dget_size_compact(n, n,    fmt, V), align);
        double *taup = (double *)mkl_malloc(mkl_dget_size_compact(n, 1,    fmt, V), align);
        double *bp   = (double *)mkl_malloc(mkl_dget_size_compact(n, nrhs, fmt, V), align);

        std::vector<MKL_INT> info(V);
        double wq;
        mkl_dgeqrf_compact(MKL_COL_MAJOR, n, n, ap, n, taup, &wq, -1, info.data(), fmt, V);
        const MKL_INT lwork = (MKL_INT)wq;
        std::vector<double> work((size_t)std::max<MKL_INT>(lwork, 1));

        std::vector<double *> Aptr(V), Bptr(V);   /* per-matrix base pointers */
        std::vector<double>   xout((size_t)V * n);  /* unpacked solutions      */
        std::vector<double *> Xptr(V);

#pragma omp for schedule(static)
        for (int g = 0; g < ngroups; ++g) {
            const int base = g * V;
            const MKL_INT cnt = std::min(V, nmat - base);   /* last group may be short */

            for (int s = 0; s < cnt; ++s) {
                Aptr[s] = const_cast<double *>(P.a.data()) + (size_t)(base + s) * n * n;
                Bptr[s] = const_cast<double *>(P.b.data()) + (size_t)(base + s) * n;
                Xptr[s] = xout.data() + (size_t)s * n;
            }

            mkl_dgepack_compact(MKL_COL_MAJOR, n, n,    Aptr.data(), n, ap, n, fmt, cnt);
            mkl_dgepack_compact(MKL_COL_MAJOR, n, nrhs, Bptr.data(), n, bp, n, fmt, cnt);

            mkl_dgeqrf_compact(MKL_COL_MAJOR, n, n, ap, n, taup,
                               work.data(), lwork, info.data(), fmt, cnt);
            double dummy;
            ext_mkl_dormqr_compact(MKL_COL_MAJOR, 'L', 'T', n, nrhs, n,
                                   ap, n, taup, bp, n, &dummy, 1, info.data(), fmt, cnt);
            mkl_dtrsm_compact(MKL_COL_MAJOR, MKL_LEFT, MKL_UPPER, MKL_NOTRANS, MKL_NONUNIT,
                              n, nrhs, 1.0, ap, n, bp, n, fmt, cnt);

            mkl_dgeunpack_compact(MKL_COL_MAJOR, n, nrhs, Xptr.data(), n, bp, n, fmt, cnt);
            for (int s = 0; s < cnt; ++s)
                maxerr = std::max(maxerr, sol_error(Xptr[s], n));
        }

        mkl_free(ap); mkl_free(taup); mkl_free(bp);
    }
    return maxerr;
}

/* ===== non-batched path: conventional per-matrix LAPACK ================ *
 * One dense matrix at a time: dgeqrf -> dormqr -> dtrsm, factoring in place
 * (no per-matrix copy in the hot loop -- the application does not reuse the
 * matrix afterwards). The caller refreshes `a`/`b` from the pristine pool
 * outside the timed region, since the factorization destroys them. `a` holds
 * the matrices (n*n each), `b` the right-hand sides (n each, overwritten with
 * the solutions). Returns the max solution error. */
double run_unbatched(int n, int nmat, double *a, double *b)
{
    const int nrhs = 1;
    double maxerr = 0.0;

#pragma omp parallel reduction(max : maxerr)
    {
        std::vector<double> tau(n);

#pragma omp for schedule(static)
        for (int v = 0; v < nmat; ++v) {
            double *A = a + (size_t)v * n * n;     /* dgeqrf overwrites A */
            double *B = b + (size_t)v * n;         /* dormqr/dtrsm overwrite B */

            LAPACKE_dgeqrf(LAPACK_COL_MAJOR, n, n, A, n, tau.data());
            LAPACKE_dormqr(LAPACK_COL_MAJOR, 'L', 'T', n, nrhs, n,
                           A, n, tau.data(), B, n);
            cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                        n, nrhs, 1.0, A, n, B, n);

            maxerr = std::max(maxerr, sol_error(B, n));
        }
    }
    return maxerr;
}

/* Best (minimum) wall time over `reps` timed passes, in seconds. `reset` runs
 * untimed before every pass (e.g. to restore input the timed work destroys);
 * only `timed` is clocked. */
template <typename Reset, typename Timed>
double best_time(int reps, Reset &&reset, Timed &&timed)
{
    reset(); timed();                         /* warm-up (untimed) */
    double best = std::numeric_limits<double>::infinity();
    for (int r = 0; r < reps; ++r) {
        reset();                              /* not clocked */
        auto t0 = clk::now();
        timed();
        const std::chrono::duration<double> elapsed = clk::now() - t0;  /* seconds */
        best = std::min(best, elapsed.count());
    }
    return best;
}

} /* anonymous namespace */

int main(int argc, char **argv)
{
    const int nmat = (argc > 1) ? std::atoi(argv[1]) : 1000;
    const int reps = (argc > 2) ? std::atoi(argv[2]) : 3;
    check(nmat > 0 && reps > 0, "usage: bench_qr_compact [nmat>0] [reps>0]");

    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = vlen_for_format(fmt);
    check(V > 0, "unrecognised compact format");

    /* Pin MKL's internal threading: the OpenMP outer loop is the only
     * parallelism, so per-call MKL threads would just oversubscribe. */
    mkl_set_num_threads(1);
    /* Turn LAPACKE NaN-checking off so the per-matrix path is timed clean. */
    LAPACKE_set_nancheck(0);

    int nthreads = 1;
#ifdef _OPENMP
#pragma omp parallel
#pragma omp single
    nthreads = omp_get_num_threads();
#endif

    const int sizes[] = {20, 40, 60, 80, 100};
    const int nsizes = (int)(sizeof(sizes) / sizeof(sizes[0]));
    const double eps = std::numeric_limits<double>::epsilon();

    std::printf("QR solve throughput: compact batched (mkl_dgeqrf_compact -> "
                "ext_mkl_dormqr_compact -> mkl_dtrsm_compact)\n");
    std::printf("            vs per-matrix (LAPACKE_dgeqrf -> LAPACKE_dormqr -> cblas_dtrsm)\n");
    std::printf("matrices=%d  reps=%d  compact V=%d  OpenMP threads=%d\n\n",
                nmat, reps, V, nthreads);
    std::printf("   n |  batched (s)  Mmat/s | unbatched (s)  Mmat/s | speedup |  max fwd err\n");
    std::printf("-----+----------------------+----------------------+---------+-------------\n");

    double log_speedup_sum = 0.0;
    for (int si = 0; si < nsizes; ++si) {
        const int n = sizes[si];
        Pool P(n, nmat);

        /* The batched path reads the pool read-only (pack copies into the
         * interleaved buffers), so it needs no reset. The non-batched path
         * factors in place, so refresh a destroyable working copy of the pool
         * before each pass -- untimed, mirroring an application that consumes
         * the matrix rather than copying it inside the solve. */
        std::vector<double> wa, wb;   /* filled by the reset step below */

        double err_b = 0.0, err_u = 0.0;
        const double tb = best_time(reps,
            [] {},
            [&] { err_b = run_batched(P, fmt, V); });
        const double tu = best_time(reps,
            [&] { wa = P.a; wb = P.b; },
            [&] { err_u = run_unbatched(n, nmat, wa.data(), wb.data()); });

        const double rtol  = 100.0 * n * eps;
        const double maxerr = std::max(err_b, err_u);
        check(maxerr <= rtol, "solve accuracy within rtol");

        const double speedup = tu / tb;
        log_speedup_sum += std::log(speedup);
        std::printf("%4d | %11.4f  %6.2f | %11.4f  %6.2f | %6.2fx | %.2e (rtol %.1e)\n",
                    n, tb, nmat / tb / 1e6, tu, nmat / tu / 1e6,
                    speedup, maxerr, rtol);
    }

    const double geomean = std::exp(log_speedup_sum / nsizes);
    std::printf("-----+----------------------+----------------------+---------+-------------\n");
    std::printf("geometric-mean speedup (batched vs unbatched) across sizes: %.2fx\n", geomean);
    return 0;
}
