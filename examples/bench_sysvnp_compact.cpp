/* bench_sysvnp_compact.cpp
 *
 * Throughput benchmark of the end-to-end symmetric *solve* A X = B over pools of
 * many small symmetric indefinite matrices, comparing the fused compact solver
 * with the conventional per-matrix LAPACK driver:
 *
 *   cqr-compact  cqr_mkl_dsysvnp_compact  (this project's fused unpivoted LDL^T
 *                                          factor + solve, one call on the pool)
 *   per-matrix   LAPACKE_dsysv            (Bunch-Kaufman LDL^T factor + solve,
 *                                          one matrix at a time)
 *
 * There is no MKL compact yardstick here: MKL ships no compact sytrf/sysv of any
 * kind (the reason these routines exist). The two paths do not run the same
 * arithmetic either -- LAPACK pivots, the compact solver does not -- so this is
 * a comparison of the two ways of solving the batch, not of two implementations
 * of one algorithm; the pool is built so that the unpivoted factorization is
 * safe (diagonally dominant), and both paths are checked against the known
 * solution.
 *
 * To measure the solvers rather than data movement, the pool is packed into
 * compact form once, up front (A and B); only the solve is timed, and the
 * destroyed input is restored (untimed) before each pass. The cqr path is one
 * call on the whole pool -- the routine threads its own loop over groups, and
 * factors and solves each group while its factor is cache-resident. The
 * per-matrix LAPACK path is driven from an OpenMP loop of the same thread
 * count, with a per-thread ipiv (LAPACKE_dsysv allocates its own workspace).
 *
 * Usage:  bench_sysvnp_compact [--nrhs=k] [--size-sweep=nmin:nmax[:stride]]
 *                              [--simdlen=2|4|8] [nmat] [reps]
 *         (defaults: 1 right-hand side, 512 matrices, 3 reps)
 *
 * With no --size-sweep it runs the cqr-vs-LAPACK comparison; with it, a
 * cqr-only throughput scan over the size range. --simdlen forces the interleave
 * width (2/4/8) instead of the host's widest.
 *
 * Build: needs Intel MKL plus this repo's cqr_mkl_ext; wired up by CMakeLists.txt
 * as the `bench_sysvnp_compact` target. OpenMP is used when available. Build with
 * host-tuned flags (e.g. `-DCMAKE_CXX_FLAGS="-O3 -march=native"`) so the open
 * compact kernel emits the full vector width.
 *
 * Assisted-by: Claude
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

/* Flop count of the unpivoted LDL^T solve in GFLOP: the n^3/3 + n^2/2 + n/6
 * of the factorization (the same count as Cholesky, LAWN 41; the n reciprocals
 * uncounted, as LAPACK leaves the square roots uncounted) plus the 2 n^2 nrhs
 * of the two triangular sweeps and the n nrhs diagonal scaling. */
double sysv_gflop(int n, int nrhs)
{
    const double dn = n, dr = nrhs;
    return (dn * dn * dn / 3.0 + dn * dn / 2.0 + dn / 6.0 + 2.0 * dn * dn * dr +
            dn * dr) *
           1e-9;
}

/* A pool of `nmat` dense column-major n x n symmetric *indefinite* matrices,
 * back to back in `a`, each with the right-hand sides B = A X (n x nrhs,
 * column-major) back to back in `b`, for the known X(:,j) = j + 1.
 *
 * Each A is symmetric with random off-diagonals in [-1,1] and a diagonal of
 * magnitude 2n with alternating sign, so it is strictly diagonally dominant
 * (row off-diagonal magnitudes sum to at most n-1 < 2n), hence every leading
 * principal minor is nonsingular and the unpivoted LDL^T exists with bounded
 * element growth -- the class of input the unpivoted solver is for -- while the
 * mixed diagonal signs make it genuinely indefinite (Cholesky would fail; LAPACK
 * needs ?sysv, not ?posv). O(n^2) to build, no O(n^3) product. The full matrix
 * is stored (both triangles) so the per-matrix LAPACK path and the compact pack
 * see identical symmetric input; each routine reads only the lower triangle. */
struct Pool {
    int n, nmat, nrhs;
    aligned_vector<double> a; /* nmat * n*n,    64 B-aligned */
    aligned_vector<double> b; /* nmat * n*nrhs, 64 B-aligned */

    Pool(int n_, int nmat_, int nrhs_)
        : n(n_), nmat(nmat_), nrhs(nrhs_), a((size_t)nmat_ * n_ * n_),
          b((size_t)nmat_ * n_ * nrhs_)
    {
        std::mt19937_64 rng(2025);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        std::vector<double> X((size_t)n * nrhs);
        for (int j = 0; j < nrhs; ++j)
            for (int i = 0; i < n; ++i)
                X[i + (size_t)j * n] = j + 1;
        for (int v = 0; v < nmat; ++v) {
            double *A = a.data() + (size_t)v * n * n;
            for (int j = 0; j < n; ++j) {
                for (int i = j + 1; i < n; ++i) {
                    double x = dist(rng);
                    A[i + (size_t)j * n] = x; /* lower */
                    A[j + (size_t)i * n] = x; /* mirror to upper (symmetric) */
                }
                A[j + (size_t)j * n] =
                    (j % 2 ? -2.0 : 2.0) * n; /* dominant, mixed sign */
            }
            /* B = A X for this matrix */
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, nrhs, n, 1.0, A, n,
                        X.data(), n, 0.0, b.data() + (size_t)v * n * nrhs, n);
        }
    }
};

/* The fused compact solve of a pre-packed pool in place, column-major lower:
 * one call, threaded inside the library. */
void solve_compact(double *ap, double *bp, int n, int nrhs, int nmat,
                   MKL_COMPACT_PACK fmt)
{
    MKL_INT info;
    cqr_mkl_dsysvnp_compact(MKL_COL_MAJOR, MKL_LOWER, n, nrhs, ap, n, bp, n, &info, fmt,
                            nmat);
}

/* Per-matrix LAPACK ?sysv (Bunch-Kaufman) of a standard-layout pool copy in
 * place (lower): A is overwritten with its pivoted factor, B with X. ipiv is
 * one n-vector per OpenMP thread. */
void solve_unbatched(double *a, double *b, int n, int nrhs, int nmat, MKL_INT *ipiv)
{
#pragma omp parallel for schedule(static)
    for (int v = 0; v < nmat; ++v) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        LAPACKE_dsysv(LAPACK_COL_MAJOR, 'L', n, nrhs, a + (size_t)v * n * n, n,
                      ipiv + (size_t)tid * n, b + (size_t)v * n * nrhs, n);
    }
}

/* Worst relative forward error max|X_v - X| / max|X| over the pool, X_v the
 * solution in a dense pool-layout buffer (nmat * n*nrhs). */
double forward_error(const double *x, int n, int nrhs, int nmat)
{
    const size_t sB = (size_t)n * nrhs;
    double worst = 0;
    for (int v = 0; v < nmat; ++v)
        for (int j = 0; j < nrhs; ++j)
            for (int i = 0; i < n; ++i)
                worst =
                    std::max(worst, std::abs(x[v * sB + i + (size_t)j * n] - (j + 1)));
    return worst / nrhs; /* max|X| = nrhs */
}

/* Pack the pool's A and B into fresh compact buffers (the pristine copies the
 * timed passes are restored from). */
struct Packed {
    MKL_INT sz_a, sz_b;
    cqr::detail::mkl_buffer<double> ap, bp;

    Packed(const Pool &P, MKL_COMPACT_PACK fmt)
        : sz_a(mkl_dget_size_compact(P.n, P.n, fmt, P.nmat)),
          sz_b(mkl_dget_size_compact(P.n, P.nrhs, fmt, P.nmat)),
          ap(cqr::detail::mkl_alloc_bytes<double>(sz_a)),
          bp(cqr::detail::mkl_alloc_bytes<double>(sz_b))
    {
        const int n = P.n, nmat = P.nmat, nrhs = P.nrhs;
        std::vector<const double *> Ap(nmat), Bp(nmat);
        for (int v = 0; v < nmat; ++v) {
            Ap[v] = P.a.data() + (size_t)v * n * n;
            Bp[v] = P.b.data() + (size_t)v * n * nrhs;
        }
        mkl_dgepack_compact(MKL_COL_MAJOR, n, n, Ap.data(), n, ap.get(), n, fmt, nmat);
        mkl_dgepack_compact(MKL_COL_MAJOR, n, nrhs, Bp.data(), n, bp.get(), n, fmt, nmat);
    }
};

/* Forward error of the compact path: solve fresh copies of the packed pool,
 * unpack X, compare to the known solution. Untimed correctness gate. */
double compact_error(const Pool &P, const Packed &pk, MKL_COMPACT_PACK fmt)
{
    const int n = P.n, nmat = P.nmat, nrhs = P.nrhs;
    auto ap = cqr::detail::mkl_alloc_bytes<double>(pk.sz_a);
    auto bp = cqr::detail::mkl_alloc_bytes<double>(pk.sz_b);
    std::memcpy(ap.get(), pk.ap.get(), pk.sz_a);
    std::memcpy(bp.get(), pk.bp.get(), pk.sz_b);
    solve_compact(ap.get(), bp.get(), n, nrhs, nmat, fmt);

    std::vector<double> X((size_t)nmat * n * nrhs);
    std::vector<double *> Xp(nmat);
    for (int v = 0; v < nmat; ++v)
        Xp[v] = X.data() + (size_t)v * n * nrhs;
    mkl_dgeunpack_compact(MKL_COL_MAJOR, n, nrhs, Xp.data(), n, bp.get(), n, fmt, nmat);
    return forward_error(X.data(), n, nrhs, nmat);
}

/* Single-solver size sweep: the fused compact solve of a pre-packed pool at
 * each n in [nmin, nmax] (step stride), throughput only -- no LAPACK
 * comparison, so it stays cheap and isolates the kernel. The raw best-pass
 * time is printed next to the derived rates (rate = nmat / time): a total near
 * the timer granularity flags a noisy row -- raise nmat until it is
 * comfortably above the clock resolution. */
void run_sweep(int nmat, int reps, int nrhs, int nmin, int nmax, int stride,
               MKL_COMPACT_PACK fmt, int V, int nthreads)
{
    std::printf("Symmetric solve size sweep: cqr_mkl_dsysvnp_compact only (throughput, "
                "no cross-check)\n");
    std::printf("matrices=%d  nrhs=%d  reps=%d  simdlen=%d (%s)  OpenMP threads=%d  "
                "(indefinite, col-major lower, pre-packed)\n\n",
                nmat, nrhs, reps, V, compact_format_name(fmt), nthreads);
    std::printf("   n |  total (s) | cqr GFLOP/s |   cqr mat/s\n");
    std::printf("-----+------------+-------------+-------------\n");

    for (int n = nmin; n <= nmax; n += stride) {
        Pool P(n, nmat, nrhs);
        Packed pk(P, fmt);
        auto ap = cqr::detail::mkl_alloc_bytes<double>(pk.sz_a);
        auto bp = cqr::detail::mkl_alloc_bytes<double>(pk.sz_b);
        auto restore = [&] {
            std::memcpy(ap.get(), pk.ap.get(), pk.sz_a);
            std::memcpy(bp.get(), pk.bp.get(), pk.sz_b);
        };
        const double t = best_time(reps, restore, [&] {
            solve_compact(ap.get(), bp.get(), n, nrhs, nmat, fmt);
        });
        std::printf("%4d | %10.3e | %11.2f | %11.2e\n", n, t,
                    nmat * sysv_gflop(n, nrhs) / t, nmat / t);
    }
    std::printf("-----+------------+-------------+-------------\n");
}

} /* anonymous namespace */

int main(int argc, char **argv)
{
    /* --nrhs=k is this benchmark's own flag; the rest is the shared command line */
    int nrhs = 1;
    std::vector<char *> rest;
    for (int i = 0; i < argc; ++i) {
        if (i > 0 && std::strncmp(argv[i], "--nrhs=", 7) == 0)
            nrhs = std::atoi(argv[i] + 7);
        else
            rest.push_back(argv[i]);
    }
    check(nrhs > 0, "usage: --nrhs=k needs k > 0");
    const CmdArgs args((int)rest.size(), rest.data(), "bench_sysvnp_compact");
    const int nmat = args.nmat, reps = args.reps, V = args.V;
    const MKL_COMPACT_PACK fmt = args.fmt;

    /* Pin MKL's internal threading: the OpenMP outer loop is the only
     * parallelism. LAPACKE NaN-checking off so the per-matrix path is timed clean. */
    mkl_set_num_threads(1);
    LAPACKE_set_nancheck(0);
    const int nthreads = omp_threads();

    if (args.sweep) {
        run_sweep(nmat, reps, nrhs, args.sweep_min, args.sweep_max, args.sweep_step, fmt,
                  V, nthreads);
        return 0;
    }

    /* Square sizes spanning the target range (order 3..500, emphasis below 170),
     * mixing sizes that are not multiples of the SIMD width V with the round
     * powers, as in the factorization benchmarks. */
    constexpr std::array sizes = {8,  16,  24,  30,  32,  45,  48,  60, 64,
                                  96, 105, 128, 168, 170, 256, 384, 500};

    std::printf("Symmetric solve throughput: cqr_mkl_dsysvnp_compact (fused unpivoted "
                "LDL^T) vs per-matrix LAPACKE_dsysv (Bunch-Kaufman)\n");
    std::printf("matrices=%d  nrhs=%d  reps=%d  simdlen=%d (%s)  OpenMP threads=%d  "
                "(indefinite, col-major lower, pre-packed)\n\n",
                nmat, nrhs, reps, V, compact_format_name(fmt), nthreads);
    /* Throughput as matrices/second (scientific) so it stays legible across the
     * whole size range. The error columns are forward errors against the known
     * solution, one per path. */
    std::printf("   n | cqr GFLOP/s |   cqr mat/s | lapack mat/s | cqr/lap | "
                "fwderr(cqr) | fwderr(lapack)\n");
    std::printf("-----+-------------+-------------+--------------+---------+"
                "-------------+---------------\n");

    /* one ipiv per thread, sized for the largest order in the list */
    std::vector<MKL_INT> ipiv((size_t)nthreads *
                              *std::max_element(sizes.begin(), sizes.end()));
    double log_speed = 0.0;
    for (int n : sizes) {
        Pool P(n, nmat, nrhs);
        Packed pk(P, fmt);

        /* working copies: compact (cqr) and standard layout (LAPACK) */
        auto ap = cqr::detail::mkl_alloc_bytes<double>(pk.sz_a);
        auto bp = cqr::detail::mkl_alloc_bytes<double>(pk.sz_b);
        aligned_vector<double> a_work, b_work;

        /* both paths destroy A and B, so restore the input (untimed) before
         * each timed pass */
        auto restore_compact = [&] {
            std::memcpy(ap.get(), pk.ap.get(), pk.sz_a);
            std::memcpy(bp.get(), pk.bp.get(), pk.sz_b);
        };
        auto restore_dense = [&] {
            a_work = P.a;
            b_work = P.b;
        };

        double t_cqr = best_time(reps, restore_compact, [&] {
            solve_compact(ap.get(), bp.get(), n, nrhs, nmat, fmt);
        });
        double t_lap = best_time(reps, restore_dense, [&] {
            solve_unbatched(a_work.data(), b_work.data(), n, nrhs, nmat, ipiv.data());
        });

        /* correctness gates: both paths vs the known solution (LAPACK's from the
         * last timed pass; the compact one from a fresh untimed solve) */
        const double err_cqr = compact_error(P, pk, fmt);
        const double err_lap = forward_error(b_work.data(), n, nrhs, nmat);
        check(err_cqr <= 1e-9, "compact solve recovers the known solution");
        check(err_lap <= 1e-9, "LAPACK solve recovers the known solution");

        const double sp = t_lap / t_cqr;
        const double gflops_cqr = nmat * sysv_gflop(n, nrhs) / t_cqr;
        log_speed += std::log(sp);
        std::printf("%4d | %11.2f | %11.2e | %12.2e | %6.2fx | %11.2e | %13.2e\n", n,
                    gflops_cqr, nmat / t_cqr, nmat / t_lap, sp, err_cqr, err_lap);
    }

    std::printf("-----+-------------+-------------+--------------+---------+"
                "-------------+---------------\n");
    std::printf("geometric-mean speedup (cqr fused compact solve vs per-matrix "
                "LAPACKE_dsysv): %.2fx\n",
                std::exp(log_speed / sizes.size()));
    return 0;
}
