/* bench_cqr_ispc.cpp
 *
 * Throughput benchmark of the ISPC compact-QR prototype against the alternatives
 * it is meant to be measured against, on the batched square solve A_v X_v = B_v:
 *
 *   factor        apply Q^T            solve
 *   -----------   ------------------   ----------------
 *   ISPC          cqr_ispc_dgeqrf   -> cqr_ispc_dormqr -> cqr_ispc_dtrsm   (all ISPC)
 *   GNU vectors   dgeqrf_compact    -> dormqr_compact  -> mkl_dtrsm_compact (repo path)
 *   MKL geqrf     mkl_dgeqrf_compact-> dormqr_compact  -> mkl_dtrsm_compact
 *   per-matrix    LAPACKE_dgeqrf    -> LAPACKE_dormqr  -> cblas_dtrsm       (scalar QR)
 *
 * Two views are reported:
 *   * Per-kernel micro-benchmarks -- the cleanest ISPC-vs-alternatives read:
 *       geqrf: ISPC vs GNU vs MKL     ormqr: ISPC vs GNU     trsm: ISPC vs MKL
 *   * The full three-stage solve pipeline, ISPC vs GNU vs MKL-geqrf vs per-matrix.
 *
 * The compact paths run on pre-packed interleaved buffers (pack/unpack is common
 * to all three and excluded from the timed region), so what is measured is the
 * kernels themselves -- the SIMD work ISPC and the GNU vector_size kernels each
 * express differently over the identical AVX-512 V=8 layout. Single-threaded, to
 * isolate per-core SIMD efficiency (thread scaling is orthogonal). Every method
 * is accuracy-gated against the known solution X == 1.
 *
 * Usage: bench_cqr_ispc [nmat] [reps] [nrhs]   (defaults: 1024, 3, 4)
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#include <mkl.h>
#include <mkl_compact.h>

#include "cqr_ispc.h"    /* ISPC prototype              */
#include "cqr_compact.h" /* reference GNU-vector kernels */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace {

using clk = std::chrono::steady_clock;

void die(const char *what)
{
    std::printf("FATAL: %s\n", what);
    std::exit(1);
}

/* 64-byte-aligned double buffer. */
double *aligned(size_t n)
{
    size_t bytes = ((n * sizeof(double) + 63) / 64) * 64;
    double *p = static_cast<double *>(std::aligned_alloc(64, bytes));
    if (!p) die("aligned_alloc");
    std::memset(p, 0, bytes);
    return p;
}

/* Best (minimum) wall time in seconds over `reps` timed passes; `reset` runs
 * untimed before each (restoring inputs the kernels overwrite). */
template <typename Reset, typename Timed>
double best_time(int reps, Reset &&reset, Timed &&timed)
{
    reset();
    timed(); /* warm-up */
    double best = std::numeric_limits<double>::infinity();
    for (int r = 0; r < reps; ++r) {
        reset();
        auto t0 = clk::now();
        timed();
        std::chrono::duration<double> dt = clk::now() - t0;
        best = std::min(best, dt.count());
    }
    return best;
}

/* Dense pool: nmat column-major matrices of order n (n*n each, back to back) and
 * nrhs right-hand sides per matrix (n*nrhs each). Exact solution X == 1, so each
 * RHS column is the row sums of A_v and a correct solve returns all ones. */
struct Pool {
    int n, nrhs, nmat;
    std::vector<double> a, b;
    Pool(int n_, int nrhs_, int nmat_)
        : n(n_), nrhs(nrhs_), nmat(nmat_), a((size_t)nmat_ * n_ * n_),
          b((size_t)nmat_ * n_ * nrhs_)
    {
        std::mt19937_64 rng(2024);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (int v = 0; v < nmat; ++v) {
            double *A = a.data() + (size_t)v * n * n;
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    A[i + (size_t)j * n] = dist(rng);
            for (int i = 0; i < n; ++i)
                A[i + (size_t)i * n] += 2.0 * n; /* diagonally dominant */
            double *B = b.data() + (size_t)v * n * nrhs;
            for (int j = 0; j < nrhs; ++j)
                for (int i = 0; i < n; ++i) {
                    double s = 0.0;
                    for (int l = 0; l < n; ++l)
                        s += A[i + (size_t)l * n];
                    B[i + (size_t)j * n] = s;
                }
        }
    }
};

double maxdev_from_one(const double *x, size_t n)
{
    double e = 0.0;
    for (size_t i = 0; i < n; ++i)
        e = std::max(e, std::fabs(x[i] - 1.0));
    return e;
}

const double EPS = std::numeric_limits<double>::epsilon();

/* ------------------------------------------------------------------------- */
struct Row {
    double t_ispc = 0, t_gnu = 0, t_mkl = 0, t_lap = 0; /* seconds, best-of-reps */
};

void print_kernel_header(const char *title)
{
    std::printf("\n%s   (throughput in GFLOP/s)\n", title);
    std::printf("   n |   ISPC          GNU            MKL            | ISPC vs GNU  "
                "ISPC vs MKL\n");
    std::printf("-----+------------------------------------------------+---------------"
                "-----------\n");
}

} /* namespace */

int main(int argc, char **argv)
{
    const int nmat = (argc > 1) ? std::atoi(argv[1]) : 1024;
    const int reps = (argc > 2) ? std::atoi(argv[2]) : 3;
    const int nrhs = (argc > 3) ? std::atoi(argv[3]) : 4;
    if (nmat <= 0 || reps <= 0 || nrhs <= 0)
        die("usage: bench_cqr_ispc [nmat>0] [reps>0] [nrhs>0]");

    /* Sequential only, no threading. Pin MKL's sequential layer before the first
     * MKL call so the single-DLL runtime loads libmkl_sequential (this also
     * avoids the iomp5/OpenMP dependency) -- no MKL_THREADING_LAYER env var
     * needed. Isolates per-core SIMD efficiency; thread scaling is orthogonal. */
    mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL);
    mkl_set_num_threads(1);

    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = (fmt == MKL_COMPACT_SSE ? 16 : fmt == MKL_COMPACT_AVX ? 32 : 64) / 8;
    LAPACKE_set_nancheck(0);

    if (V != 8) {
        std::printf("ISPC prototype needs V=8 (avx512skx-x8); host MKL format is V=%d.\n",
                    V);
        return 77;
    }

    const int sizes[] = {10, 20, 30, 40, 50, 60, 80, 100, 120, 150};
    const int nsizes = (int)(sizeof(sizes) / sizeof(sizes[0]));

    std::printf("ISPC compact-QR prototype benchmark\n");
    std::printf("matrices=%d  reps=%d  nrhs=%d  V=%d (AVX512)  sequential (1 thread)\n",
                nmat, reps, nrhs, V);
    std::printf("per-kernel and full-solve throughput in GFLOP/s; higher is better\n");

    /* accumulate geomean speedups for the full pipeline */
    double log_ig = 0, log_im = 0, log_il = 0;

    Row geqrf[16], ormqr[16], trsm[16], solve[16];

    for (int si = 0; si < nsizes; ++si) {
        const int n = sizes[si];
        Pool P(n, nrhs, nmat);
        const int ngroups = (nmat + V - 1) / V;
        (void)ngroups;

        /* Pristine compact buffers (all nmat, padded last group). */
        const size_t szA = (size_t)mkl_dget_size_compact(n, n, fmt, nmat);
        const size_t szT = (size_t)mkl_dget_size_compact(n, 1, fmt, nmat);
        const size_t szB = (size_t)mkl_dget_size_compact(n, nrhs, fmt, nmat);

        double *ap0 = aligned(szA), *bp0 = aligned(szB); /* pristine packed A, B */
        {
            std::vector<double *> Ap(nmat), Bp(nmat);
            for (int v = 0; v < nmat; ++v) {
                Ap[v] = P.a.data() + (size_t)v * n * n;
                Bp[v] = P.b.data() + (size_t)v * n * nrhs;
            }
            mkl_dgepack_compact(MKL_COL_MAJOR, n, n, Ap.data(), n, ap0, n, fmt, nmat);
            mkl_dgepack_compact(MKL_COL_MAJOR, n, nrhs, Bp.data(), n, bp0, n, fmt, nmat);
        }

        /* Working buffers. */
        double *ap = aligned(szA), *bp = aligned(szB), *tau = aligned(szT);
        /* Factored-state snapshots for isolating ormqr / trsm. */
        double *apF = aligned(szA), *tauF = aligned(szT); /* after geqrf */
        double *bpQ = aligned(szB);                       /* after geqrf+ormqr(Q^T) */

        /* MKL geqrf workspace (its query wants ~n*V). */
        double wq = 0;
        MKL_INT info[1];
        mkl_dgeqrf_compact(MKL_COL_MAJOR, n, n, ap0, n, tau, &wq, -1, info, fmt, nmat);
        std::vector<double> mklwork((size_t)std::max<MKL_INT>((MKL_INT)wq, 1));
        const MKL_INT lwork = (MKL_INT)wq;

        auto reset_ap = [&] { std::memcpy(ap, ap0, szA * sizeof(double)); };
        auto reset_bp = [&] { std::memcpy(bp, bp0, szB * sizeof(double)); };

        /* Build the factored snapshots once (untimed) with the GNU kernels. */
        reset_ap();
        std::memcpy(apF, ap0, szA * sizeof(double));
        dgeqrf_compact('C', n, n, apF, n, tauF, V, nmat);
        std::memcpy(bpQ, bp0, szB * sizeof(double));
        dormqr_compact('T', n, nrhs, n, apF, n, tauF, bpQ, n, V, nmat);

        /* =========================== geqrf ============================== */
        geqrf[si].t_ispc = best_time(
            reps, reset_ap, [&] { cqr_ispc_dgeqrf_compact(n, n, ap, n, tau, nmat); });
        geqrf[si].t_gnu = best_time(
            reps, reset_ap, [&] { dgeqrf_compact('C', n, n, ap, n, tau, V, nmat); });
        geqrf[si].t_mkl = best_time(reps, reset_ap, [&] {
            mkl_dgeqrf_compact(MKL_COL_MAJOR, n, n, ap, n, tau, mklwork.data(), lwork,
                               info, fmt, nmat);
        });

        /* =========================== ormqr ============================== */
        /* apply Q^T to a fresh RHS; reflectors come from the fixed snapshot apF. */
        ormqr[si].t_ispc = best_time(reps, reset_bp, [&] {
            cqr_ispc_dormqr_compact(1, n, nrhs, n, apF, n, tauF, bp, n, nmat);
        });
        ormqr[si].t_gnu = best_time(reps, reset_bp, [&] {
            dormqr_compact('T', n, nrhs, n, apF, n, tauF, bp, n, V, nmat);
        });
        /* MKL has no compact ormqr; leave t_mkl = 0 (shown as "--"). */

        /* =========================== trsm =============================== */
        /* solve R X = (Q^T B); RHS reset from the bpQ snapshot each pass. */
        auto reset_bpQ = [&] { std::memcpy(bp, bpQ, szB * sizeof(double)); };
        trsm[si].t_ispc = best_time(reps, reset_bpQ, [&] {
            cqr_ispc_dtrsm_compact(n, nrhs, 1.0, apF, n, bp, n, nmat);
        });
        trsm[si].t_mkl = best_time(reps, reset_bpQ, [&] {
            mkl_dtrsm_compact(MKL_COL_MAJOR, MKL_LEFT, MKL_UPPER, MKL_NOTRANS,
                              MKL_NONUNIT, n, nrhs, 1.0, apF, n, bp, n, fmt, nmat);
        });
        /* repo has no GNU trsm; leave t_gnu = 0. */

        /* ===================== full solve pipeline ====================== */
        auto reset_both = [&] {
            std::memcpy(ap, ap0, szA * sizeof(double));
            std::memcpy(bp, bp0, szB * sizeof(double));
        };
        double err_ispc = 0, err_gnu = 0, err_mkl = 0, err_lap = 0;

        solve[si].t_ispc = best_time(reps, reset_both, [&] {
            cqr_ispc_dgeqrf_compact(n, n, ap, n, tau, nmat);
            cqr_ispc_dormqr_compact(1, n, nrhs, n, ap, n, tau, bp, n, nmat);
            cqr_ispc_dtrsm_compact(n, nrhs, 1.0, ap, n, bp, n, nmat);
        });
        {
            std::vector<double> X((size_t)nmat * n * nrhs);
            std::vector<double *> Xp(nmat);
            for (int v = 0; v < nmat; ++v)
                Xp[v] = X.data() + (size_t)v * n * nrhs;
            mkl_dgeunpack_compact(MKL_COL_MAJOR, n, nrhs, Xp.data(), n, bp, n, fmt, nmat);
            err_ispc = maxdev_from_one(X.data(), X.size());
        }

        solve[si].t_gnu = best_time(reps, reset_both, [&] {
            dgeqrf_compact('C', n, n, ap, n, tau, V, nmat);
            dormqr_compact('T', n, nrhs, n, ap, n, tau, bp, n, V, nmat);
            mkl_dtrsm_compact(MKL_COL_MAJOR, MKL_LEFT, MKL_UPPER, MKL_NOTRANS,
                              MKL_NONUNIT, n, nrhs, 1.0, ap, n, bp, n, fmt, nmat);
        });
        {
            std::vector<double> X((size_t)nmat * n * nrhs);
            std::vector<double *> Xp(nmat);
            for (int v = 0; v < nmat; ++v)
                Xp[v] = X.data() + (size_t)v * n * nrhs;
            mkl_dgeunpack_compact(MKL_COL_MAJOR, n, nrhs, Xp.data(), n, bp, n, fmt, nmat);
            err_gnu = maxdev_from_one(X.data(), X.size());
        }

        solve[si].t_mkl = best_time(reps, reset_both, [&] {
            mkl_dgeqrf_compact(MKL_COL_MAJOR, n, n, ap, n, tau, mklwork.data(), lwork,
                               info, fmt, nmat);
            dormqr_compact('T', n, nrhs, n, ap, n, tau, bp, n, V, nmat);
            mkl_dtrsm_compact(MKL_COL_MAJOR, MKL_LEFT, MKL_UPPER, MKL_NOTRANS,
                              MKL_NONUNIT, n, nrhs, 1.0, ap, n, bp, n, fmt, nmat);
        });
        {
            std::vector<double> X((size_t)nmat * n * nrhs);
            std::vector<double *> Xp(nmat);
            for (int v = 0; v < nmat; ++v)
                Xp[v] = X.data() + (size_t)v * n * nrhs;
            mkl_dgeunpack_compact(MKL_COL_MAJOR, n, nrhs, Xp.data(), n, bp, n, fmt, nmat);
            err_mkl = maxdev_from_one(X.data(), X.size());
        }

        /* per-matrix LAPACK on dense data (working copies restored each pass). */
        std::vector<double> wa, wb, tlap(n);
        solve[si].t_lap = best_time(
            reps,
            [&] {
                wa = P.a;
                wb = P.b;
            },
            [&] {
                for (int v = 0; v < nmat; ++v) {
                    double *A = wa.data() + (size_t)v * n * n;
                    double *Bv = wb.data() + (size_t)v * n * nrhs;
                    LAPACKE_dgeqrf(LAPACK_COL_MAJOR, n, n, A, n, tlap.data());
                    LAPACKE_dormqr(LAPACK_COL_MAJOR, 'L', 'T', n, nrhs, n, A, n,
                                   tlap.data(), Bv, n);
                    cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans,
                                CblasNonUnit, n, nrhs, 1.0, A, n, Bv, n);
                }
            });
        err_lap = maxdev_from_one(wb.data(), wb.size());

        const double gate = 100.0 * n * EPS;
        if (std::max(std::max(err_ispc, err_gnu), std::max(err_mkl, err_lap)) > gate)
            std::printf("  WARN n=%d: err ispc %.1e gnu %.1e mkl %.1e lap %.1e (gate "
                        "%.1e)\n",
                        n, err_ispc, err_gnu, err_mkl, err_lap, gate);

        log_ig += std::log(solve[si].t_gnu / solve[si].t_ispc);
        log_im += std::log(solve[si].t_mkl / solve[si].t_ispc);
        log_il += std::log(solve[si].t_lap / solve[si].t_ispc);

        std::free(ap0);
        std::free(bp0);
        std::free(ap);
        std::free(bp);
        std::free(tau);
        std::free(apF);
        std::free(tauF);
        std::free(bpQ);
    }

    /* ------------------------------ report ------------------------------ */
    /* GFLOP/s: size-normalized, so absolute SIMD efficiency reads across the
     * whole range (matrices/s spans 6 decades from n=10 to n=150). Speedups are
     * time ratios, identical to what matrices/s would give. Approximate flop
     * counts per matrix (leading order): geqrf (4/3)n^3, ormqr 2 r n(n+1) for the
     * two dorm2r passes over k=n reflectors, trsm r n^2 for back-substitution. */
    auto fl_geqrf = [](int n) { return (4.0 / 3.0) * n * n * (double)n; };
    auto fl_ormqr = [&](int n) { return 2.0 * nrhs * n * (n + 1.0); };
    auto fl_trsm = [&](int n) { return 1.0 * nrhs * n * (double)n; };
    auto fl_solve = [&](int n) { return fl_geqrf(n) + fl_ormqr(n) + fl_trsm(n); };
    auto gf = [&](double t, double fpm) { return (t > 0) ? nmat * fpm / t / 1e9 : 0.0; };
    /* geomean of a per-size ratio across the sweep. */
    auto geomean = [&](auto pick) {
        double s = 0;
        for (int si = 0; si < nsizes; ++si)
            s += std::log(pick(si));
        return std::exp(s / nsizes);
    };

    print_kernel_header("[geqrf] QR factorization  A = Q R");
    for (int si = 0; si < nsizes; ++si) {
        const Row &r = geqrf[si];
        const double f = fl_geqrf(sizes[si]);
        std::printf("%4d | %10.2f     %10.2f     %10.2f    | %8.2fx    %8.2fx\n",
                    sizes[si], gf(r.t_ispc, f), gf(r.t_gnu, f), gf(r.t_mkl, f),
                    r.t_gnu / r.t_ispc, r.t_mkl / r.t_ispc);
    }
    std::printf("     geomean ISPC speedup: %.2fx vs GNU, %.2fx vs MKL\n",
                geomean([&](int si) { return geqrf[si].t_gnu / geqrf[si].t_ispc; }),
                geomean([&](int si) { return geqrf[si].t_mkl / geqrf[si].t_ispc; }));

    print_kernel_header("[ormqr] apply Q^T  B := Q^T B   (MKL has no compact ormqr)");
    for (int si = 0; si < nsizes; ++si) {
        const Row &r = ormqr[si];
        const double f = fl_ormqr(sizes[si]);
        std::printf("%4d | %10.2f     %10.2f     %10s    | %8.2fx    %11s\n", sizes[si],
                    gf(r.t_ispc, f), gf(r.t_gnu, f), "--", r.t_gnu / r.t_ispc, "--");
    }
    std::printf("     geomean ISPC speedup: %.2fx vs GNU\n",
                geomean([&](int si) { return ormqr[si].t_gnu / ormqr[si].t_ispc; }));

    print_kernel_header("[trsm] triangular solve  R X = B   (repo has no GNU trsm)");
    for (int si = 0; si < nsizes; ++si) {
        const Row &r = trsm[si];
        const double f = fl_trsm(sizes[si]);
        std::printf("%4d | %10.2f     %10s     %10.2f    | %11s    %8.2fx\n", sizes[si],
                    gf(r.t_ispc, f), "--", gf(r.t_mkl, f), "--", r.t_mkl / r.t_ispc);
    }
    std::printf("     geomean ISPC speedup: %.2fx vs MKL\n",
                geomean([&](int si) { return trsm[si].t_mkl / trsm[si].t_ispc; }));

    std::printf("\n[solve] full pipeline  geqrf -> ormqr(Q^T) -> trsm   (GFLOP/s)\n");
    std::printf("   n |   ISPC    GNU+MKL  MKLgeqrf  perMatrix   |  ISPC speedup "
                "vs GNU / MKL / perMat\n");
    std::printf("-----+------------------------------------------+-----------------"
                "--------------------\n");
    for (int si = 0; si < nsizes; ++si) {
        const Row &r = solve[si];
        const double f = fl_solve(sizes[si]);
        std::printf("%4d | %7.2f  %7.2f  %7.2f  %8.2f   |  %6.2fx / %5.2fx / %6.2fx\n",
                    sizes[si], gf(r.t_ispc, f), gf(r.t_gnu, f), gf(r.t_mkl, f),
                    gf(r.t_lap, f), r.t_gnu / r.t_ispc, r.t_mkl / r.t_ispc,
                    r.t_lap / r.t_ispc);
    }
    std::printf("-----+------------------------------------------+-----------------"
                "--------------------\n");
    std::printf("geomean full-solve speedup  ISPC vs GNU %.2fx   vs MKL %.2fx   vs "
                "per-matrix %.2fx\n",
                std::exp(log_ig / nsizes), std::exp(log_im / nsizes),
                std::exp(log_il / nsizes));
    return 0;
}
