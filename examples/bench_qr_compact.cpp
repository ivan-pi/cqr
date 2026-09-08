/* bench_qr_compact.cpp
 *
 * Throughput benchmark: solving many small square systems A_v X_v = B_v with
 * the QR pipeline (X = R^-1 Q^T B), comparing three ways to run the same math:
 *
 *   MKL batched  mkl_dgeqrf_compact     -> cqr_mkl_dormqr_compact -> mkl_dtrsm_compact
 *   cqr batched  cqr_mkl_dgeqrf_compact -> cqr_mkl_dormqr_compact -> cqr_mkl_dtrsm_compact
 *   unbatched    LAPACKE_dgeqrf         -> LAPACKE_dormqr          -> cblas_dtrsm
 *
 * The two batched paths run the same compact pipeline from different libraries:
 * the MKL path uses MKL's own `mkl_?geqrf_compact` and `mkl_?trsm_compact`, the
 * cqr path this repo's open `cqr_mkl_?geqrf_compact` and `cqr_mkl_?trsm_compact`,
 * so the cqr path runs the whole solve with no MKL compute kernel (MKL only packs
 * and unpacks) and their ratio is the end-to-end MKL-vs-open comparison. MKL has
 * no compact `ormqr`, so `cqr_mkl_dormqr_compact` is shared by both. The
 * unbatched path is the conventional per-matrix LAPACK baseline.
 *
 * For each size, a pool of `nmat` well-conditioned matrices with known solution
 * X == 1 is built once, and each path solves it -- the batched paths packing
 * each group of `V` (the compact SIMD width) on the fly, the per-matrix path
 * factoring in place. The outer loop over the pool runs under OpenMP (MKL's own
 * threading pinned to 1); each size is timed `reps` times keeping the best, and
 * geometric-mean speedups across sizes are printed at the end. (Details on the
 * timing harness and the in-place working copy are at best_time() and
 * run_unbatched().)
 *
 * The batched pipeline is deliberately driven group by group from the caller's
 * loop, not as three whole-pool calls (which the routines would thread
 * internally): per group, pack -> geqrf -> ormqr -> trsm -> unpack all touch one
 * group's buffers, a few tens of KB that stay in L1/L2 across the five steps,
 * whereas whole-pool calls stream the entire pool through five separate passes.
 * Measured on 4 cores, the whole-pool variant was 15-55% slower over n = 10..100.
 * (For a single factorization the two are equivalent; see bench_geqrf_compact.) A single right-hand side per system (nrhs = 1); every path
 * is checked against the known solution X == 1, so the reported error is a
 * forward error, not a comparison to LAPACK.
 *
 * Usage:  bench_qr_compact [nmat] [reps]      (defaults: 1000 matrices, 3 reps)
 *
 * Build: needs Intel MKL plus this repo's cqr_mkl_ormqr_compact and
 * cqr_mkl_trsm_compact; wired up by CMakeLists.txt as the `bench_qr_compact`
 * target. OpenMP is used when available. For a fair cqr-vs-MKL comparison, build
 * with host-tuned flags (e.g. `-DCMAKE_CXX_FLAGS="-O3 -march=native"`) so the
 * open compact kernels emit the full vector width, matching MKL's AVX-512 runtime
 * dispatch; without it the geqrf-dominated cqr path runs the V-wide packs on the
 * baseline ISA and is unfairly slow (cqr/MKL well below 1).
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#include <mkl.h>
#include <mkl_compact.h>

#include "cqr_mkl_alloc.h" /* mkl_alloc_bytes (calls mkl_malloc; links MKL) */
#include "bench_util.hpp"  /* check, best_time, omp_threads, format helpers */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>
#include <algorithm>

namespace {

using namespace cqr::bench;

/* The batch of systems: `nmat` square matrices of order n in `a`, and their
 * matching right-hand sides in `b` (one RHS per system). The exact solution is
 * X == 1, so each RHS is the row sum b_v = A_v 1, and a correct solve returns
 * all ones. */
struct Systems {
    MatrixPool a, b;

    Systems(int n, int nmat) : a(nmat, n, n), b(nmat, n, 1)
    {
        std::mt19937_64 rng(42);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (int v = 0; v < nmat; ++v) {
            const auto A = a.matrix(v), B = b.matrix(v);
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    A(i, j) = dist(rng);
            for (int d = 0; d < n; ++d)
                A(d, d) += 2.0 * n;       /* diag dominant */
            for (int i = 0; i < n; ++i) { /* B = A * ones */
                double s = 0.0;
                for (int j = 0; j < n; ++j)
                    s += A(i, j);
                B(i, 0) = s;
            }
        }
    }
};

/* Largest deviation of a computed solution (expected all ones) from 1. */
double sol_error(const double *x, int n)
{
    double e = 0.0;
    for (int i = 0; i < n; ++i)
        e = std::max(e, std::fabs(x[i] - 1.0));
    return e;
}

/* Which library backs the batched compute pipeline -- the QR factorization and
 * the closing triangular solve. MKL has no compact ormqr, so cqr_mkl_dormqr is
 * shared by both; every other compute kernel comes from the selected library. */
enum class Backend { Mkl, Cqr };

/* ===== batched path: compact group-of-V pipeline ======================= *
 * Process the pool in groups of V, packing/factoring/solving/unpacking each
 * group inside the timed region. `impl` picks the backend for the geqrf and the
 * trsm (MKL's compact kernels, or this repo's open drop-ins); the shared
 * cqr_mkl_dormqr and the pack/unpack around them are identical for both. Returns
 * the max solution error. */
double run_batched(const Systems &P, MKL_COMPACT_PACK fmt, int V, Backend impl)
{
    const int n = P.a.rows, nmat = P.a.nmat, nrhs = 1;
    const int ngroups = (nmat + V - 1) / V;
    double maxerr = 0.0;

#pragma omp parallel reduction(max : maxerr)
    {
        /* Per-thread compact buffers, sized for a full group of V. */
        const int align = 64;
        auto ap_buf = cqr::detail::mkl_alloc_bytes<double>(
            mkl_dget_size_compact(n, n, fmt, V), align);
        auto taup_buf = cqr::detail::mkl_alloc_bytes<double>(
            mkl_dget_size_compact(n, 1, fmt, V), align);
        auto bp_buf = cqr::detail::mkl_alloc_bytes<double>(
            mkl_dget_size_compact(n, nrhs, fmt, V), align);
        double *ap = ap_buf.get(), *taup = taup_buf.get(), *bp = bp_buf.get();

        /* Select the batched backend once: MKL's own compact kernels, or this
         * repo's open drop-ins (byte-identical signatures). The ormqr below is
         * always cqr's -- MKL ships no compact ormqr. */
        const auto geqrf_compact =
            (impl == Backend::Cqr) ? cqr_mkl_dgeqrf_compact : mkl_dgeqrf_compact;
        const auto trsm_compact =
            (impl == Backend::Cqr) ? cqr_mkl_dtrsm_compact : mkl_dtrsm_compact;

        MKL_INT info[1]; /* compact status: a single scalar (MKL convention) */
        double wq;
        geqrf_compact(MKL_COL_MAJOR, n, n, ap, n, taup, &wq, -1, info, fmt, V);
        const MKL_INT lwork = (MKL_INT)wq;
        std::vector<double> work((size_t)std::max<MKL_INT>(lwork, 1));

        std::vector<const double *> Aptr(V), Bptr(V); /* per-matrix base pointers */
        std::vector<double> xout((size_t)V * n);      /* unpacked solutions      */
        std::vector<double *> Xptr(V);

#pragma omp for schedule(static)
        for (int g = 0; g < ngroups; ++g) {
            const int base = g * V;
            const MKL_INT cnt = std::min(V, nmat - base); /* last group may be short */

            for (int s = 0; s < cnt; ++s) {
                Aptr[s] = P.a.matrix(base + s).data;
                Bptr[s] = P.b.matrix(base + s).data;
                Xptr[s] = xout.data() + (size_t)s * n;
            }

            mkl_dgepack_compact(MKL_COL_MAJOR, n, n, Aptr.data(), n, ap, n, fmt, cnt);
            mkl_dgepack_compact(MKL_COL_MAJOR, n, nrhs, Bptr.data(), n, bp, n, fmt, cnt);

            geqrf_compact(MKL_COL_MAJOR, n, n, ap, n, taup, work.data(), lwork, info, fmt,
                          cnt);
            double dummy;
            cqr_mkl_dormqr_compact(MKL_COL_MAJOR, 'L', 'T', n, nrhs, n, ap, n, taup, bp,
                                   n, &dummy, 1, info, fmt, cnt);
            trsm_compact(MKL_COL_MAJOR, MKL_LEFT, MKL_UPPER, MKL_NOTRANS, MKL_NONUNIT, n,
                         nrhs, 1.0, ap, n, bp, n, fmt, cnt);

            mkl_dgeunpack_compact(MKL_COL_MAJOR, n, nrhs, Xptr.data(), n, bp, n, fmt,
                                  cnt);
            for (int s = 0; s < cnt; ++s)
                maxerr = std::max(maxerr, sol_error(Xptr[s], n));
        }
        /* ap/taup/bp freed by their RAII owners at end of the parallel region. */
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
            double *A = a + (size_t)v * n * n; /* dgeqrf overwrites A */
            double *B = b + (size_t)v * n;     /* dormqr/dtrsm overwrite B */

            LAPACKE_dgeqrf(LAPACK_COL_MAJOR, n, n, A, n, tau.data());
            LAPACKE_dormqr(LAPACK_COL_MAJOR, 'L', 'T', n, nrhs, n, A, n, tau.data(), B,
                           n);
            cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                        n, nrhs, 1.0, A, n, B, n);

            maxerr = std::max(maxerr, sol_error(B, n));
        }
    }
    return maxerr;
}

} /* anonymous namespace */

int main(int argc, char **argv)
{
    const int nmat = (argc > 1) ? std::atoi(argv[1]) : 1000;
    const int reps = (argc > 2) ? std::atoi(argv[2]) : 3;
    check(nmat > 0 && reps > 0, "usage: bench_qr_compact [nmat>0] [reps>0]");

    /* mkl_get_format_compact() returns the architecture's optimal packing
     * format -- always one of SSE/AVX/AVX512 -- so V is one of 2/4/8. */
    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = vlen_for_format<double>(fmt);

    /* Pin MKL's internal threading: the OpenMP outer loop is the only
     * parallelism, so per-call MKL threads would just oversubscribe. */
    mkl_set_num_threads(1);
    /* Turn LAPACKE NaN-checking off so the per-matrix path is timed clean. */
    LAPACKE_set_nancheck(0);

    const int nthreads = omp_threads();

    const int sizes[] = {10, 20, 30, 40, 50, 60, 80, 100};
    const int nsizes = (int)(sizeof(sizes) / sizeof(sizes[0]));
    const int nrhs = 1; /* single RHS per system (see Systems / run_batched) */
    const double eps = std::numeric_limits<double>::epsilon();

    std::printf("QR solve throughput (matrices/second), three paths:\n");
    std::printf("  MKL-batch  mkl_dgeqrf_compact     -> cqr_mkl_dormqr_compact -> "
                "mkl_dtrsm_compact\n");
    std::printf("  cqr-batch  cqr_mkl_dgeqrf_compact -> cqr_mkl_dormqr_compact -> "
                "cqr_mkl_dtrsm_compact\n");
    std::printf("  unbatched  LAPACKE_dgeqrf         -> LAPACKE_dormqr          -> "
                "cblas_dtrsm\n");
    std::printf("matrices=%d  reps=%d  rhs=%d  simdlen=%d (%s)  OpenMP threads=%d\n\n",
                nmat, reps, nrhs, V, compact_format_name(fmt), nthreads);
    /* Throughput as matrices/second (scientific) for each path, then two speedups:
     * cqr-batch over the per-matrix baseline (the headline batched win), and
     * cqr-batch over MKL-batch (the two batched paths differ in the geqrf and the
     * trsm -- ormqr is shared -- so this is the end-to-end effect of the fully open
     * compact pipeline vs MKL's). The error is the forward error vs the known
     * solution X == 1, not vs LAPACK. */
    std::printf("   n | MKL-batch   cqr-batch   unbatched  | cqr/unbat | cqr/MKL | max "
                "fwd err (vs X=1)\n");
    std::printf("     |  (mat/s)     (mat/s)     (mat/s)    |           |         |\n");
    std::printf("-----+-------------------------------------+-----------+---------+------"
                "-------------\n");

    double log_cqr_vs_unbat = 0.0, log_cqr_vs_mkl = 0.0;
    for (const int n : sizes) {
        const Systems P(n, nmat);

        /* The batched paths read the pool read-only (pack copies into the
         * interleaved buffers), so they need no reset. The non-batched path
         * factors in place, so refresh a destroyable working copy of the pool
         * before each pass -- untimed, mirroring an application that consumes
         * the matrix rather than copying it inside the solve. */
        aligned_vector<double> wa, wb; /* filled by the reset step below */

        double err_m = 0.0, err_c = 0.0, err_u = 0.0;
        const double tb_mkl =
            best_time(reps, [] {}, [&] { err_m = run_batched(P, fmt, V, Backend::Mkl); });
        const double tb_cqr =
            best_time(reps, [] {}, [&] { err_c = run_batched(P, fmt, V, Backend::Cqr); });
        const double tu = best_time(
            reps,
            [&] {
                wa = P.a.storage;
                wb = P.b.storage;
            },
            [&] { err_u = run_unbatched(n, nmat, wa.data(), wb.data()); });

        const double rtol = 100.0 * n * eps;
        const double maxerr = std::max({err_m, err_c, err_u});
        check(maxerr <= rtol, "solve accuracy within rtol");

        const double cqr_vs_unbat = tu / tb_cqr;
        const double cqr_vs_mkl = tb_mkl / tb_cqr;
        log_cqr_vs_unbat += std::log(cqr_vs_unbat);
        log_cqr_vs_mkl += std::log(cqr_vs_mkl);
        std::printf("%4d | %10.2e  %10.2e  %10.2e | %8.2fx | %6.2fx | %.2e (rtol %.1e)\n",
                    n, nmat / tb_mkl, nmat / tb_cqr, nmat / tu, cqr_vs_unbat, cqr_vs_mkl,
                    maxerr, rtol);
    }

    std::printf("-----+-------------------------------------+-----------+---------+------"
                "-------------\n");
    std::printf("geometric-mean speedup across sizes:  cqr-batch vs unbatched %.2fx"
                "   |   cqr-batch vs MKL-batch %.2fx\n",
                std::exp(log_cqr_vs_unbat / nsizes), std::exp(log_cqr_vs_mkl / nsizes));
    return 0;
}
