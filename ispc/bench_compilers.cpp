/* bench_compilers.cpp
 *
 * Compiler shootout: GCC vs clang vs ISPC on the compact QR pipeline, every
 * number reported as a speedup against the corresponding MKL compact routine.
 *
 *   * "native"  -- the templated GNU `vector_size` kernels
 *                  (cqr::detail::geqrf_compact_general / ormqr_compact_general /
 *                  trsm_compact) called DIRECTLY from this TU, so whichever
 *                  compiler builds this file generates their code. Build it with
 *                  g++ for the GCC column and with clang++ for the clang column.
 *   * "ISPC"    -- cqr_ispc.o, always ISPC 1.22 regardless of the host compiler.
 *   * "MKL"     -- mkl_dgeqrf_compact / mkl_dtrsm_compact (the baseline).
 *
 * So the GCC and clang columns come from two builds of this same source; ISPC and
 * MKL appear in both as a consistency check. MKL has no compact ormqr, so ormqr
 * is reported native-vs-ISPC only. Single-threaded (MKL pinned sequential).
 *
 * Usage: bench_compilers [nmat] [reps] [nrhs]   (defaults: 1024, 3, 4)
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#include <mkl.h>
#include <mkl_compact.h>

#include "cqr_ispc.h"            /* ISPC kernels (extern "C")            */
#include "cqr_geqrf_compact.hpp" /* geqrf_compact_general<T,V>           */
#include "cqr_compact.hpp"       /* ormqr_compact_general<T,V>           */
#include "cqr_trsm_compact.hpp"  /* trsm_compact<T,V>                    */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#if defined(__clang__)
static const char *CC = "clang";
#elif defined(__GNUC__)
static const char *CC = "GCC";
#else
static const char *CC = "cc";
#endif

namespace {

using clk = std::chrono::steady_clock;

void die(const char *what)
{
    std::printf("FATAL: %s\n", what);
    std::exit(1);
}

double *aligned(size_t n)
{
    size_t bytes = ((n * sizeof(double) + 63) / 64) * 64;
    double *p = static_cast<double *>(std::aligned_alloc(64, bytes));
    if (!p) die("aligned_alloc");
    std::memset(p, 0, bytes);
    return p;
}

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

/* Dense pool with known solution X == 1 (RHS = row sums). */
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
                A[i + (size_t)i * n] += 2.0 * n;
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

/* native (this-compiler) kernels, V = 8, column-major. */
void native_geqrf(int n, double *ap, double *tau, int nm)
{
    cqr::detail::geqrf_compact_general<double, 8>(false, n, n, ap, n, tau, nm);
}
void native_ormqrT(int n, int nrhs, double *ap, double *tau, double *bp, int nm)
{
    cqr::detail::ormqr_compact_general<double, 8>(true, false, 'T', n, nrhs, n, ap, n,
                                                  tau, bp, n, nm);
}
void native_trsm(int n, int nrhs, double *ap, double *bp, int nm)
{
    cqr::detail::trsm_compact<double, 8>(n, nrhs, 1.0, ap, n, bp, n, nm);
}

const double EPS = std::numeric_limits<double>::epsilon();

struct K {
    double nat = 0, ispc = 0, mkl = 0; /* seconds, best-of-reps */
};

} /* namespace */

int main(int argc, char **argv)
{
    const int nmat = (argc > 1) ? std::atoi(argv[1]) : 1024;
    const int reps = (argc > 2) ? std::atoi(argv[2]) : 3;
    const int nrhs = (argc > 3) ? std::atoi(argv[3]) : 4;
    if (nmat <= 0 || reps <= 0 || nrhs <= 0)
        die("usage: bench_compilers [nmat>0] [reps>0] [nrhs>0]");

    mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL);
    mkl_set_num_threads(1);
    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = (fmt == MKL_COMPACT_SSE ? 16 : fmt == MKL_COMPACT_AVX ? 32 : 64) / 8;
    LAPACKE_set_nancheck(0);
    if (V != 8) {
        std::printf("needs V=8 (avx512skx-x8); host MKL format is V=%d.\n", V);
        return 77;
    }

    const int sizes[] = {10, 20, 30, 40, 50, 60, 80, 100, 120, 150};
    const int nsizes = (int)(sizeof(sizes) / sizeof(sizes[0]));

    std::printf("Compiler shootout: native C++ built with %s  vs  ISPC  vs  MKL\n", CC);
    std::printf("nmat=%d reps=%d nrhs=%d V=8 (AVX512) sequential; GFLOP/s and speedup "
                "vs MKL\n",
                nmat, reps, nrhs);

    K geqrf[16], trsm[16], solve[16];
    /* ormqr: MKL has none -> only native vs ISPC. */
    double ormqr_nat[16], ormqr_ispc[16];

    for (int si = 0; si < nsizes; ++si) {
        const int n = sizes[si];
        Pool P(n, nrhs, nmat);
        const size_t szA = (size_t)mkl_dget_size_compact(n, n, fmt, nmat);
        const size_t szT = (size_t)mkl_dget_size_compact(n, 1, fmt, nmat);
        const size_t szB = (size_t)mkl_dget_size_compact(n, nrhs, fmt, nmat);

        double *ap0 = aligned(szA), *bp0 = aligned(szB);
        {
            std::vector<double *> Ap(nmat), Bp(nmat);
            for (int v = 0; v < nmat; ++v) {
                Ap[v] = P.a.data() + (size_t)v * n * n;
                Bp[v] = P.b.data() + (size_t)v * n * nrhs;
            }
            mkl_dgepack_compact(MKL_COL_MAJOR, n, n, Ap.data(), n, ap0, n, fmt, nmat);
            mkl_dgepack_compact(MKL_COL_MAJOR, n, nrhs, Bp.data(), n, bp0, n, fmt, nmat);
        }
        double *ap = aligned(szA), *bp = aligned(szB), *tau = aligned(szT);
        double *apF = aligned(szA), *tauF = aligned(szT), *bpQ = aligned(szB);

        double wq = 0;
        MKL_INT info[1];
        mkl_dgeqrf_compact(MKL_COL_MAJOR, n, n, ap0, n, tau, &wq, -1, info, fmt, nmat);
        std::vector<double> mklwork((size_t)std::max<MKL_INT>((MKL_INT)wq, 1));
        const MKL_INT lwork = (MKL_INT)wq;

        auto reset_ap = [&] { std::memcpy(ap, ap0, szA * sizeof(double)); };
        auto reset_bp = [&] { std::memcpy(bp, bp0, szB * sizeof(double)); };

        /* Common factored snapshots (from the native kernels) for ormqr/trsm. */
        std::memcpy(apF, ap0, szA * sizeof(double));
        native_geqrf(n, apF, tauF, nmat);
        std::memcpy(bpQ, bp0, szB * sizeof(double));
        native_ormqrT(n, nrhs, apF, tauF, bpQ, nmat);
        auto reset_bpQ = [&] { std::memcpy(bp, bpQ, szB * sizeof(double)); };

        /* ---- geqrf ---- */
        geqrf[si].nat =
            best_time(reps, reset_ap, [&] { native_geqrf(n, ap, tau, nmat); });
        geqrf[si].ispc = best_time(
            reps, reset_ap, [&] { cqr_ispc_dgeqrf_compact(n, n, ap, n, tau, nmat); });
        geqrf[si].mkl = best_time(reps, reset_ap, [&] {
            mkl_dgeqrf_compact(MKL_COL_MAJOR, n, n, ap, n, tau, mklwork.data(), lwork,
                               info, fmt, nmat);
        });

        /* ---- ormqr (no MKL compact routine) ---- */
        ormqr_nat[si] = best_time(reps, reset_bp,
                                  [&] { native_ormqrT(n, nrhs, apF, tauF, bp, nmat); });
        ormqr_ispc[si] = best_time(reps, reset_bp, [&] {
            cqr_ispc_dormqr_compact(1, n, nrhs, n, apF, n, tauF, bp, n, nmat);
        });

        /* ---- trsm ---- */
        trsm[si].nat =
            best_time(reps, reset_bpQ, [&] { native_trsm(n, nrhs, apF, bp, nmat); });
        trsm[si].ispc = best_time(reps, reset_bpQ, [&] {
            cqr_ispc_dtrsm_compact(n, nrhs, 1.0, apF, n, bp, n, nmat);
        });
        trsm[si].mkl = best_time(reps, reset_bpQ, [&] {
            mkl_dtrsm_compact(MKL_COL_MAJOR, MKL_LEFT, MKL_UPPER, MKL_NOTRANS,
                              MKL_NONUNIT, n, nrhs, 1.0, apF, n, bp, n, fmt, nmat);
        });

        /* ---- full solve ---- */
        auto reset_both = [&] {
            std::memcpy(ap, ap0, szA * sizeof(double));
            std::memcpy(bp, bp0, szB * sizeof(double));
        };
        double err_nat = 0, err_ispc = 0, err_mkl = 0;

        solve[si].nat = best_time(reps, reset_both, [&] {
            native_geqrf(n, ap, tau, nmat);
            native_ormqrT(n, nrhs, ap, tau, bp, nmat);
            native_trsm(n, nrhs, ap, bp, nmat);
        });
        {
            std::vector<double> X((size_t)nmat * n * nrhs);
            std::vector<double *> Xp(nmat);
            for (int v = 0; v < nmat; ++v)
                Xp[v] = X.data() + (size_t)v * n * nrhs;
            mkl_dgeunpack_compact(MKL_COL_MAJOR, n, nrhs, Xp.data(), n, bp, n, fmt, nmat);
            err_nat = maxdev_from_one(X.data(), X.size());
        }
        solve[si].ispc = best_time(reps, reset_both, [&] {
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
        /* MKL pipeline baseline: mkl geqrf + native ormqr (MKL has none) + mkl trsm. */
        solve[si].mkl = best_time(reps, reset_both, [&] {
            mkl_dgeqrf_compact(MKL_COL_MAJOR, n, n, ap, n, tau, mklwork.data(), lwork,
                               info, fmt, nmat);
            native_ormqrT(n, nrhs, ap, tau, bp, nmat);
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
        const double gate = 100.0 * n * EPS;
        if (std::max(err_nat, std::max(err_ispc, err_mkl)) > gate)
            std::printf("  WARN n=%d err nat %.1e ispc %.1e mkl %.1e (gate %.1e)\n", n,
                        err_nat, err_ispc, err_mkl, gate);

        std::free(ap0);
        std::free(bp0);
        std::free(ap);
        std::free(bp);
        std::free(tau);
        std::free(apF);
        std::free(tauF);
        std::free(bpQ);
    }

    /* --------------------------- report --------------------------- */
    auto fl_geqrf = [](int n) { return (4.0 / 3.0) * n * n * (double)n; };
    auto fl_trsm = [&](int n) { return 1.0 * nrhs * n * (double)n; };
    auto fl_ormqr = [&](int n) { return 2.0 * nrhs * n * (n + 1.0); };
    auto fl_solve = [&](int n) { return fl_geqrf(n) + fl_ormqr(n) + fl_trsm(n); };
    auto gf = [&](double t, double fpm) { return (t > 0) ? nmat * fpm / t / 1e9 : 0.0; };
    auto geomean = [&](auto pick) {
        double s = 0;
        for (int si = 0; si < nsizes; ++si)
            s += std::log(pick(si));
        return std::exp(s / nsizes);
    };

    std::printf("\n[geqrf]  GFLOP/s (%s / ISPC / MKL)   |  speedup vs MKL: %s, ISPC\n",
                CC, CC);
    std::printf("-----+-----------------------------------+------------------------\n");
    for (int si = 0; si < nsizes; ++si) {
        const K &k = geqrf[si];
        const double f = fl_geqrf(sizes[si]);
        std::printf("%4d | %8.2f  %8.2f  %8.2f       |   %6.2fx    %6.2fx\n", sizes[si],
                    gf(k.nat, f), gf(k.ispc, f), gf(k.mkl, f), k.mkl / k.nat,
                    k.mkl / k.ispc);
    }
    std::printf("     geomean speedup vs MKL: %s %.2fx, ISPC %.2fx\n", CC,
                geomean([&](int si) { return geqrf[si].mkl / geqrf[si].nat; }),
                geomean([&](int si) { return geqrf[si].mkl / geqrf[si].ispc; }));

    std::printf("\n[trsm]   GFLOP/s (%s / ISPC / MKL)   |  speedup vs MKL: %s, ISPC\n",
                CC, CC);
    std::printf("-----+-----------------------------------+------------------------\n");
    for (int si = 0; si < nsizes; ++si) {
        const K &k = trsm[si];
        const double f = fl_trsm(sizes[si]);
        std::printf("%4d | %8.2f  %8.2f  %8.2f       |   %6.2fx    %6.2fx\n", sizes[si],
                    gf(k.nat, f), gf(k.ispc, f), gf(k.mkl, f), k.mkl / k.nat,
                    k.mkl / k.ispc);
    }
    std::printf("     geomean speedup vs MKL: %s %.2fx, ISPC %.2fx\n", CC,
                geomean([&](int si) { return trsm[si].mkl / trsm[si].nat; }),
                geomean([&](int si) { return trsm[si].mkl / trsm[si].ispc; }));

    std::printf(
        "\n[ormqr]  no MKL compact routine -- %s vs ISPC GFLOP/s (ISPC/%s ratio)\n", CC,
        CC);
    std::printf("-----+-------------------------+----------\n");
    for (int si = 0; si < nsizes; ++si) {
        const double f = fl_ormqr(sizes[si]);
        std::printf("%4d | %8.2f  %8.2f       |  %6.2fx\n", sizes[si],
                    gf(ormqr_nat[si], f), gf(ormqr_ispc[si], f),
                    ormqr_nat[si] / ormqr_ispc[si]);
    }

    std::printf("\n[solve]  full pipeline speedup vs MKL pipeline: %s-all, ISPC-all\n",
                CC);
    std::printf("-----+---------------------------------+----------------------\n");
    for (int si = 0; si < nsizes; ++si) {
        const K &k = solve[si];
        const double f = fl_solve(sizes[si]);
        std::printf("%4d | %8.2f  %8.2f  %8.2f     |  %6.2fx    %6.2fx\n", sizes[si],
                    gf(k.nat, f), gf(k.ispc, f), gf(k.mkl, f), k.mkl / k.nat,
                    k.mkl / k.ispc);
    }
    std::printf("     geomean full-solve speedup vs MKL: %s %.2fx, ISPC %.2fx\n", CC,
                geomean([&](int si) { return solve[si].mkl / solve[si].nat; }),
                geomean([&](int si) { return solve[si].mkl / solve[si].ispc; }));
    return 0;
}
