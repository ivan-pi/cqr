/* bench_cqr_ispc.cpp -- one benchmark, three toolchains.
 *
 * The batched square solve A_v X_v = B_v, every kernel reported as a speedup over
 * the matching MKL compact routine:
 *
 *   native  = the templated GNU vector_size kernels (geqrf_compact_general /
 *             ormqr_compact_general / trsm_compact) called directly, so the
 *             DRIVER compiler generates their code -- build with g++ for the GCC
 *             column, clang++ for the clang column (`make shootout` runs both).
 *   ISPC    = cqr_ispc.o (always ISPC, whatever the host compiler).
 *   MKL     = mkl_dgeqrf_compact / mkl_dtrsm_compact (baseline; no compact ormqr).
 *
 * Also times the full pipeline against per-matrix LAPACK. Single-threaded (MKL
 * pinned sequential). See README.md for numbers and the (large) measurement
 * caveats -- these are indicative, not benchmark-grade, results.
 *
 * Usage: bench_cqr_ispc [nmat] [reps] [nrhs]   (defaults 1024, 3, 4)
 * Assisted-by: Claude:claude-opus-4.8
 */
#include "bench_common.hpp"
#include "cqr_ispc.h"
#include "cqr_geqrf_compact.hpp" /* geqrf_compact_general<T,V> */
#include "cqr_compact.hpp"       /* ormqr_compact_general<T,V> */
#include "cqr_trsm_compact.hpp"  /* trsm_compact<T,V>          */

using namespace bench;
#if defined(__clang__)
static const char *CC = "clang";
#elif defined(__GNUC__)
static const char *CC = "GCC";
#else
static const char *CC = "cc";
#endif

/* native (this-compiler) kernels, V = 8, column-major. */
static void nat_geqrf(int n, double *a, double *t, int nm)
{
    cqr::detail::geqrf_compact_general<double, 8>(false, n, n, a, n, t, nm);
}
static void nat_ormqrT(int n, int r, double *a, double *t, double *b, int nm)
{
    cqr::detail::ormqr_compact_general<double, 8>(true, false, 'T', n, r, n, a, n, t, b,
                                                  n, nm);
}
static void nat_trsm(int n, int r, double *a, double *b, int nm)
{
    cqr::detail::trsm_compact<double, 8>(n, r, 1.0, a, n, b, n, nm);
}

struct Row {
    double nat = 0, ispc = 0, mkl = 0, lap = 0;
};

int main(int argc, char **argv)
{
    const int nmat = argc > 1 ? std::atoi(argv[1]) : 1024;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 3;
    const int nrhs = argc > 3 ? std::atoi(argv[3]) : 4;
    if (nmat <= 0 || reps <= 0 || nrhs <= 0)
        die("usage: bench_cqr_ispc [nmat>0] [reps>0] [nrhs>0]");

    mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL); /* sequential, no threading */
    mkl_set_num_threads(1);
    LAPACKE_set_nancheck(0);
    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = (fmt == MKL_COMPACT_SSE ? 16 : fmt == MKL_COMPACT_AVX ? 32 : 64) / 8;
    if (V != 8) {
        std::printf("needs V=8 (avx512skx-x8); host MKL format is V=%d.\n", V);
        return 77;
    }

    const int sizes[] = {10, 20, 30, 40, 50, 60, 80, 100, 120, 150};
    const int ns = (int)(sizeof(sizes) / sizeof(*sizes));
    Row geqrf[16], ormqr[16], trsm[16], solve[16];

    std::printf("native=%s  vs  ISPC  vs  MKL   nmat=%d reps=%d nrhs=%d V=8 (AVX512) "
                "sequential\n(GFLOP/s; speedups vs MKL. Indicative only -- see README "
                "caveats.)\n",
                CC, nmat, reps, nrhs);

    for (int si = 0; si < ns; ++si) {
        const int n = sizes[si];
        Pool P(n, nrhs, nmat);
        const std::size_t szA = mkl_dget_size_compact(n, n, fmt, nmat);
        const std::size_t szT = mkl_dget_size_compact(n, 1, fmt, nmat);
        const std::size_t szB = mkl_dget_size_compact(n, nrhs, fmt, nmat);
        double *ap0 = aligned(szA), *bp0 = aligned(szB); /* pristine packed A, B */
        P.pack(fmt, ap0, bp0);
        double *ap = aligned(szA), *bp = aligned(szB), *tau = aligned(szT);
        double *apF = aligned(szA), *tauF = aligned(szT), *bpQ = aligned(szB);

        MKL_INT info[1], lwork;
        double wq;
        mkl_dgeqrf_compact(MKL_COL_MAJOR, n, n, ap0, n, tau, &wq, -1, info, fmt, nmat);
        lwork = (MKL_INT)wq;
        std::vector<double> mklwork(std::max<MKL_INT>(lwork, 1));

        auto RA = [&] { std::memcpy(ap, ap0, szA * 8); }; /* restore A */
        auto RB = [&] { std::memcpy(bp, bp0, szB * 8); }; /* restore B */
        /* factored snapshots (native) for isolating ormqr / trsm */
        std::memcpy(apF, ap0, szA * 8);
        nat_geqrf(n, apF, tauF, nmat);
        std::memcpy(bpQ, bp0, szB * 8);
        nat_ormqrT(n, nrhs, apF, tauF, bpQ, nmat);
        auto RQ = [&] { std::memcpy(bp, bpQ, szB * 8); }; /* restore Q^T B */
        auto mklGEQRF = [&] {
            mkl_dgeqrf_compact(MKL_COL_MAJOR, n, n, ap, n, tau, mklwork.data(), lwork,
                               info, fmt, nmat);
        };
        auto mklTRSM = [&](double *a, double *b) {
            mkl_dtrsm_compact(MKL_COL_MAJOR, MKL_LEFT, MKL_UPPER, MKL_NOTRANS,
                              MKL_NONUNIT, n, nrhs, 1.0, a, n, b, n, fmt, nmat);
        };

        geqrf[si].nat = best_time(reps, RA, [&] { nat_geqrf(n, ap, tau, nmat); });
        geqrf[si].ispc =
            best_time(reps, RA, [&] { cqr_ispc_dgeqrf_compact(n, n, ap, n, tau, nmat); });
        geqrf[si].mkl = best_time(reps, RA, mklGEQRF);

        ormqr[si].nat =
            best_time(reps, RB, [&] { nat_ormqrT(n, nrhs, apF, tauF, bp, nmat); });
        ormqr[si].ispc = best_time(reps, RB, [&] {
            cqr_ispc_dormqr_compact(1, n, nrhs, n, apF, n, tauF, bp, n, nmat);
        });

        trsm[si].nat = best_time(reps, RQ, [&] { nat_trsm(n, nrhs, apF, bp, nmat); });
        trsm[si].ispc = best_time(
            reps, RQ, [&] { cqr_ispc_dtrsm_compact(n, nrhs, 1.0, apF, n, bp, n, nmat); });
        trsm[si].mkl = best_time(reps, RQ, [&] { mklTRSM(apF, bp); });

        auto RAB = [&] {
            RA();
            RB();
        };
        double eN, eI, eM, eL;
        solve[si].nat = best_time(reps, RAB, [&] {
            nat_geqrf(n, ap, tau, nmat);
            nat_ormqrT(n, nrhs, ap, tau, bp, nmat);
            nat_trsm(n, nrhs, ap, bp, nmat);
        });
        eN = unpack_err(n, nrhs, fmt, nmat, bp);
        solve[si].ispc = best_time(reps, RAB, [&] {
            cqr_ispc_dgeqrf_compact(n, n, ap, n, tau, nmat);
            cqr_ispc_dormqr_compact(1, n, nrhs, n, ap, n, tau, bp, n, nmat);
            cqr_ispc_dtrsm_compact(n, nrhs, 1.0, ap, n, bp, n, nmat);
        });
        eI = unpack_err(n, nrhs, fmt, nmat, bp);
        solve[si].mkl =
            best_time(reps, RAB, [&] { /* MKL geqrf+trsm, native ormqr (MKL has none) */
                                       mklGEQRF();
                                       nat_ormqrT(n, nrhs, ap, tau, bp, nmat);
                                       mklTRSM(ap, bp);
            });
        eM = unpack_err(n, nrhs, fmt, nmat, bp);
        /* per-matrix dense LAPACK (working copies restored each pass) */
        std::vector<double> wa, wb, tl(n);
        solve[si].lap = best_time(
            reps,
            [&] {
                wa = P.a;
                wb = P.b;
            },
            [&] {
                for (int v = 0; v < nmat; ++v) {
                    double *A = wa.data() + (std::size_t)v * n * n,
                           *B = wb.data() + (std::size_t)v * n * nrhs;
                    LAPACKE_dgeqrf(LAPACK_COL_MAJOR, n, n, A, n, tl.data());
                    LAPACKE_dormqr(LAPACK_COL_MAJOR, 'L', 'T', n, nrhs, n, A, n,
                                   tl.data(), B, n);
                    cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans,
                                CblasNonUnit, n, nrhs, 1.0, A, n, B, n);
                }
            });
        eL = 0;
        for (int v = 0; v < nmat; ++v)
            eL = std::max(eL, [&] {
                double e = 0;
                for (int i = 0; i < n * nrhs; ++i)
                    e = std::max(e, std::fabs(wb[(std::size_t)v * n * nrhs + i] - 1.0));
                return e;
            }());
        if (std::max({eN, eI, eM, eL}) > 100.0 * n * EPS)
            std::printf("  WARN n=%d err nat %.1e ispc %.1e mkl %.1e lap %.1e\n", n, eN,
                        eI, eM, eL);

        for (double *p : {ap0, bp0, ap, bp, tau, apF, tauF, bpQ})
            std::free(p);
    }

    /* ---- report: per-kernel (native/ISPC/MKL, speedup vs MKL) + full solve ---- */
    auto g = [&](double t, double f) { return gflops(nmat, t, f); };
    auto tbl = [&](const char *name, Row *R, double (*fl)(int, int), bool hasMKL) {
        std::printf("\n[%s]  GFLOP/s: native ISPC%s   | speedup vs MKL: native, ISPC\n",
                    name, hasMKL ? " MKL" : "");
        for (int si = 0; si < ns; ++si) {
            double f = fl(sizes[si], nrhs);
            if (hasMKL)
                std::printf("%4d | %7.2f %7.2f %7.2f  | %6.2fx %6.2fx\n", sizes[si],
                            g(R[si].nat, f), g(R[si].ispc, f), g(R[si].mkl, f),
                            R[si].mkl / R[si].nat, R[si].mkl / R[si].ispc);
            else
                std::printf("%4d | %7.2f %7.2f    --   | (no MKL compact %s)\n",
                            sizes[si], g(R[si].nat, f), g(R[si].ispc, f), name);
        }
        if (hasMKL)
            std::printf("     geomean vs MKL: native %.2fx, ISPC %.2fx\n",
                        geomean(ns, [&](int i) { return R[i].mkl / R[i].nat; }),
                        geomean(ns, [&](int i) { return R[i].mkl / R[i].ispc; }));
    };
    tbl("geqrf", geqrf, [](int n, int) { return fl_geqrf(n); }, true);
    tbl("ormqr", ormqr, fl_ormqr, false);
    tbl("trsm", trsm, fl_trsm, true);

    std::printf("\n[solve] full pipeline GFLOP/s: native ISPC MKL perMatrix | speedup "
                "vs MKL: native, ISPC\n");
    for (int si = 0; si < ns; ++si) {
        double f =
            fl_geqrf(sizes[si]) + fl_ormqr(sizes[si], nrhs) + fl_trsm(sizes[si], nrhs);
        std::printf("%4d | %6.2f %6.2f %6.2f %8.2f | %6.2fx %6.2fx\n", sizes[si],
                    g(solve[si].nat, f), g(solve[si].ispc, f), g(solve[si].mkl, f),
                    g(solve[si].lap, f), solve[si].mkl / solve[si].nat,
                    solve[si].mkl / solve[si].ispc);
    }
    std::printf("     geomean vs MKL: native %.2fx, ISPC %.2fx  (native=%s)\n",
                geomean(ns, [&](int i) { return solve[i].mkl / solve[i].nat; }),
                geomean(ns, [&](int i) { return solve[i].mkl / solve[i].ispc; }), CC);
    return 0;
}
