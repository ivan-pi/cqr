/* bench_solve.cpp -- the full batched QR solve pipeline, three ways, same data.
 *
 * Solves A_v X_v = B_v for a batch of order-n systems in MKL Compact format
 * (geqrf -> ormqr -> trsm), timing three implementations of the whole pipeline on
 * identical packed input and the same CPU target:
 *
 *   MKL      = mkl_dgeqrf_compact -> cqr_mkl_dormqr_compact -> mkl_dtrsm_compact
 *   CQR-GNU  = the templated GNU vector_size kernels (geqrf/ormqr/trsm_compact_
 *              general<double,V>), compiled by THIS translation unit's compiler
 *   ISPC     = cqr_ispc_dgeqrf/dormqr/dtrsm_compact
 *
 * MKL ships no compact ormqr, so the MKL pipeline borrows this project's
 * cqr_mkl_dormqr_compact for that one step -- it is the same reflector apply the
 * CQR-GNU path calls directly, so the MKL-vs-CQR gap is purely geqrf + trsm.
 * Build with g++ -O3 -march=native for the CQR-GNU column (the ISPC and MKL
 * columns are compiler-independent). Single-threaded (MKL pinned sequential); the
 * pool has known solution X == 1, so each pass is checked by max|x - 1|.
 * Indicative only -- see README measurement caveats.
 *
 * Usage: bench_solve [nmat] [reps] [nrhs]   (defaults 1024, 5, 4)
 * Assisted-by: Claude:claude-opus-4.8
 */
#include "bench_common.hpp"
#include "cqr_ispc.h"
#include "cqr_mkl_ext.h"         /* cqr_mkl_dormqr_compact (../src) */
#include "cqr_geqrf_compact.hpp" /* geqrf_compact_general<T,V>      */
#include "cqr_compact.hpp"       /* ormqr_compact_general<T,V>      */
#include "cqr_trsm_compact.hpp"  /* trsm_compact_general<T,V>       */

using namespace bench;
#if defined(__clang__)
static const char *CC = "clang";
#elif defined(__GNUC__)
static const char *CC = "GCC";
#else
static const char *CC = "cc";
#endif

int main(int argc, char **argv)
{
    const int nmat = argc > 1 ? std::atoi(argv[1]) : 1024;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 5;
    const int nrhs = argc > 3 ? std::atoi(argv[3]) : 4;
    if (nmat <= 0 || reps <= 0 || nrhs <= 0)
        die("usage: bench_solve [nmat>0] [reps>0] [nrhs>0]");

    mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL); /* sequential, no threading */
    mkl_set_num_threads(1);
    LAPACKE_set_nancheck(0);
    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = (fmt == MKL_COMPACT_SSE ? 16 : fmt == MKL_COMPACT_AVX ? 32 : 64) / 8;
    if (cqr_ispc_gang_width() != V || V != 8) { /* CQR-GNU is templated at V=8 */
        std::printf("Need MKL compact V=8 and a matching gang (build the ISPC target "
                    "avx512skx-x8); got V=%d, gang=%d.\n",
                    V, cqr_ispc_gang_width());
        return 77;
    }

    const int sizes[] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120};
    const int ns = (int)(sizeof(sizes) / sizeof(*sizes));

    std::printf("Full batched QR solve: MKL-compact vs CQR (GNU vectors, %s) vs ISPC\n"
                "nmat=%d reps=%d nrhs=%d V=%d, sequential, -O3 -march=native\n"
                "pipeline geqrf -> ormqr -> trsm; MKL & CQR-GNU share the ormqr step "
                "(cqr_mkl_dormqr_compact -- MKL has no compact ormqr).\n"
                "(GFLOP/s; speedup vs MKL. Indicative only -- see README caveats.)\n\n",
                CC, nmat, reps, nrhs, V);
    std::printf("   n |    MKL   CQR-GNU    ISPC | CQR/MKL  ISPC/MKL | max|x-1|\n"
                "-----+-------------------------+-------------------+---------\n");

    double gm[16], gg[16], gi[16];
    for (int si = 0; si < ns; ++si) {
        const int n = sizes[si];
        Pool P(n, nrhs, nmat);
        const std::size_t szA = mkl_dget_size_compact(n, n, fmt, nmat);
        const std::size_t szT = mkl_dget_size_compact(n, 1, fmt, nmat);
        const std::size_t szB = mkl_dget_size_compact(n, nrhs, fmt, nmat);
        double *ap0 = aligned(szA), *bp0 = aligned(szB); /* pristine packed A, B */
        P.pack(fmt, ap0, bp0);
        double *ap = aligned(szA), *bp = aligned(szB), *tau = aligned(szT);

        /* MKL geqrf workspace query; the compact ormqr/trsm need none. */
        MKL_INT info[1], lwork;
        double wq, dummy;
        mkl_dgeqrf_compact(MKL_COL_MAJOR, n, n, ap0, n, tau, &wq, -1, info, fmt, nmat);
        lwork = (MKL_INT)wq;
        std::vector<double> work((std::size_t)std::max<MKL_INT>(lwork, 1));

        auto RAB = [&] { /* restore A and B before each timed pass */
                         std::memcpy(ap, ap0, szA * 8);
                         std::memcpy(bp, bp0, szB * 8);
        };

        const double tm = best_time(reps, RAB, [&] {
            mkl_dgeqrf_compact(MKL_COL_MAJOR, n, n, ap, n, tau, work.data(), lwork, info,
                               fmt, nmat);
            cqr_mkl_dormqr_compact(MKL_COL_MAJOR, 'L', 'T', n, nrhs, n, ap, n, tau, bp, n,
                                   &dummy, 1, info, fmt, nmat);
            mkl_dtrsm_compact(MKL_COL_MAJOR, MKL_LEFT, MKL_UPPER, MKL_NOTRANS,
                              MKL_NONUNIT, n, nrhs, 1.0, ap, n, bp, n, fmt, nmat);
        });
        const double em = unpack_err(n, nrhs, fmt, nmat, bp);

        const double tg = best_time(reps, RAB, [&] {
            cqr::detail::geqrf_compact_general<double, 8>(false, n, n, ap, n, tau, nmat);
            cqr::detail::ormqr_compact_general<double, 8>(true, false, 'T', n, nrhs, n,
                                                          ap, n, tau, bp, n, nmat);
            cqr::detail::trsm_compact_general<double, 8>(
                true, true, false, false, false, n, nrhs, 1.0, ap, n, bp, n, nmat);
        });
        const double eg = unpack_err(n, nrhs, fmt, nmat, bp);

        const double ti = best_time(reps, RAB, [&] {
            cqr_ispc_dgeqrf_compact(n, n, ap, n, tau, nmat);
            cqr_ispc_dormqr_compact(1, n, nrhs, n, ap, n, tau, bp, n, nmat);
            cqr_ispc_dtrsm_compact(n, nrhs, 1.0, ap, n, bp, n, nmat);
        });
        const double ei = unpack_err(n, nrhs, fmt, nmat, bp);

        const double f = fl_geqrf(n) + fl_ormqr(n, nrhs) + fl_trsm(n, nrhs);
        gm[si] = gflops(nmat, tm, f);
        gg[si] = gflops(nmat, tg, f);
        gi[si] = gflops(nmat, ti, f);
        const double maxe = std::max({em, eg, ei});
        std::printf("%4d | %7.2f %7.2f %7.2f | %7.2fx %7.2fx | %.1e\n", n, gm[si], gg[si],
                    gi[si], gg[si] / gm[si], gi[si] / gm[si], maxe);
        if (maxe > 100.0 * n * EPS)
            std::printf("  WARN n=%d max|x-1| %.1e (mkl %.1e gnu %.1e ispc %.1e)\n", n,
                        maxe, em, eg, ei);
        for (double *p : {ap0, bp0, ap, bp, tau})
            std::free(p);
    }
    std::printf("\ngeomean GFLOP/s: MKL %.2f  CQR-GNU %.2f  ISPC %.2f   |  vs MKL: "
                "CQR-GNU %.2fx  ISPC %.2fx  (CQR-GNU=%s)\n",
                geomean(ns, [&](int i) { return gm[i]; }),
                geomean(ns, [&](int i) { return gg[i]; }),
                geomean(ns, [&](int i) { return gi[i]; }),
                geomean(ns, [&](int i) { return gg[i] / gm[i]; }),
                geomean(ns, [&](int i) { return gi[i] / gm[i]; }), CC);
    return 0;
}
