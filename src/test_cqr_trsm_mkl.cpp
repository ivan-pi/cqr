/* test_cqr_trsm_mkl.cpp
 *
 * Validation of cqr_mkl_?trsm_compact against real Intel MKL, through the genuine
 * MKL Compact pipeline (mkl_dgepack_compact / mkl_dgeunpack_compact).
 *
 * Suite 1 (design doc section 7.2) -- cross-check vs mkl_?trsm_compact over the
 *   full feature matrix (layout x side x uplo x transa x diag): both solve the
 *   same packed batch and the compact results are compared elementwise (relative,
 *   working precision).
 * Suite 2 (design doc section 7.3) -- end-to-end AX = B with no MKL compute
 *   kernel: the open pipeline cqr_mkl_dgeqrf_compact -> cqr_mkl_dormqr_compact('L','T')
 *   -> cqr_mkl_dtrsm_compact must recover a known X (gates forward error and
 *   residual). See cqr_mkl_dtrsm_compact_design.md.
 *
 * Build: needs Intel MKL; wired up by CMakeLists.txt.
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#include <mkl.h>
#include <mkl_compact.h>

#include "cqr_mkl_ext.h"
#include "cqr_mkl_alloc.h"       /* mkl_alloc_bytes (calls mkl_malloc; links MKL) */
#include "test_compact_util.hpp" /* rng/frand, max_abs_diff, norm1, batch_ptrs, gen_tri */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>

using namespace cqr::test;

namespace {

const double eps = std::numeric_limits<double>::epsilon();

int vlen(MKL_COMPACT_PACK fmt)
{
    return cqr::detail::vlen_for_format<double>(fmt);
}

double maxabs(const double *a, size_t n)
{
    double d = 0;
    for (size_t i = 0; i < n; ++i)
        d = std::max(d, std::abs(a[i]));
    return d;
}

/* ---------------- Suite 1: cross-check vs mkl_?trsm_compact ------------ */

int suite1(MKL_LAYOUT layout, MKL_SIDE side, MKL_UPLO uplo, MKL_TRANSPOSE transa,
           MKL_DIAG diag, int nm, int m, int n)
{
    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = vlen(fmt);
    const bool rowmajor = (layout == MKL_ROW_MAJOR);
    const bool left = (side == MKL_LEFT);
    const int s = left ? m : n; /* A is s x s */
    const double alpha = 0.5 + frand<double>();

    const size_t sA = (size_t)s * s, sB = (size_t)m * n;
    std::vector<double> A(nm * sA), B(nm * sB);
    for (int v = 0; v < nm; ++v) {
        gen_tri(A.data() + v * sA, s, uplo == MKL_UPPER, rowmajor);
        double *Bv = B.data() + v * sB;
        for (size_t e = 0; e < sB; ++e)
            Bv[e] = frand<double>();
    }

    /* dense leading dims for pack (a is stored in `layout`) */
    const MKL_INT ldA = s;                /* s x s */
    const MKL_INT ldB = rowmajor ? n : m; /* m x n */
    const MKL_INT ldap = s;               /* compact leading dims */
    const MKL_INT ldbp = rowmajor ? n : m;

    auto Ap = batch_ptrs<const double>(A.data(), nm, sA);
    auto Bp = batch_ptrs<const double>(B.data(), nm, sB);

    MKL_INT sz_a = mkl_dget_size_compact(s, s, fmt, nm);
    MKL_INT sz_b = mkl_dget_size_compact(m, n, fmt, nm);
    auto ap = cqr::detail::mkl_alloc_bytes<double>(sz_a);
    auto bp_cqr = cqr::detail::mkl_alloc_bytes<double>(sz_b);
    auto bp_mkl = cqr::detail::mkl_alloc_bytes<double>(sz_b);
    mkl_dgepack_compact(layout, s, s, Ap.data(), ldA, ap.get(), ldap, fmt, nm);
    mkl_dgepack_compact(layout, m, n, Bp.data(), ldB, bp_cqr.get(), ldbp, fmt, nm);
    mkl_dgepack_compact(layout, m, n, Bp.data(), ldB, bp_mkl.get(), ldbp, fmt, nm);

    /* routine under test and the MKL reference on identical inputs */
    cqr_mkl_dtrsm_compact(layout, side, uplo, transa, diag, m, n, alpha, ap.get(), ldap,
                          bp_cqr.get(), ldbp, fmt, nm);
    mkl_dtrsm_compact(layout, side, uplo, transa, diag, m, n, alpha, ap.get(), ldap,
                      bp_mkl.get(), ldbp, fmt, nm);

    /* compare the two compact result buffers elementwise */
    const size_t nB = (size_t)sz_b / sizeof(double);
    double da = max_abs_diff(bp_cqr.get(), bp_mkl.get(), nB);
    double rel = da / std::max(maxabs(bp_mkl.get(), nB), 1e-300);

    const double rtol = 50.0 * s * eps;
    bool ok = (rel <= rtol);
    std::printf(
        "  [suite1] %s side=%c uplo=%c tr=%c diag=%c V=%-2d nm=%-2d m=%-3d n=%-3d "
        "| rel %.2e (rtol %.1e) %s\n",
        rowmajor ? "row" : "col", left ? 'L' : 'R', uplo == MKL_UPPER ? 'U' : 'L',
        transa == MKL_NOTRANS ? 'N' : 'T', diag == MKL_UNIT ? 'U' : 'N', V, nm, m, n, rel,
        rtol, ok ? "OK" : "FAIL");
    return !ok;
}

/* ---------------- Suite 2: end-to-end AX = B, no MKL kernel ------------ */

int suite2(int nm, int n, int nrhs)
{
    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = vlen(fmt), m = n, k = n;

    std::vector<double> X((size_t)n * nrhs);
    for (int j = 0; j < nrhs; ++j)
        for (int i = 0; i < n; ++i)
            X[i + (size_t)j * n] = double(j + 1);

    const size_t sA = (size_t)n * n, sB = (size_t)n * nrhs;
    std::vector<double> A(nm * sA), B(nm * sB);
    for (int v = 0; v < nm; ++v) {
        double *Av = A.data() + v * sA, *Bv = B.data() + v * sB;
        for (size_t e = 0; e < sA; ++e)
            Av[e] = frand<double>();
        for (int i = 0; i < n; ++i)
            Av[i + (size_t)i * n] += 2.0; /* tame conditioning */
        for (int j = 0; j < nrhs; ++j)    /* B = A X */
            for (int i = 0; i < n; ++i) {
                double s = 0;
                for (int l = 0; l < n; ++l)
                    s += Av[i + (size_t)l * n] * X[l + (size_t)j * n];
                Bv[i + (size_t)j * n] = s;
            }
    }
    auto Ap = batch_ptrs<const double>(A.data(), nm, sA);
    auto Bp = batch_ptrs<const double>(B.data(), nm, sB);

    MKL_INT sz_a = mkl_dget_size_compact(m, n, fmt, nm);
    MKL_INT sz_t = mkl_dget_size_compact(k, 1, fmt, nm);
    MKL_INT sz_c = mkl_dget_size_compact(m, nrhs, fmt, nm);
    auto ap_buf = cqr::detail::mkl_alloc_bytes<double>(sz_a);
    auto taup_buf = cqr::detail::mkl_alloc_bytes<double>(sz_t);
    auto cp_buf = cqr::detail::mkl_alloc_bytes<double>(sz_c);
    double *ap = ap_buf.get(), *taup = taup_buf.get(), *cp = cp_buf.get();
    mkl_dgepack_compact(MKL_COL_MAJOR, m, n, Ap.data(), m, ap, m, fmt, nm);
    mkl_dgepack_compact(MKL_COL_MAJOR, m, nrhs, Bp.data(), m, cp, m, fmt, nm);

    MKL_INT info = 99;

    /* 1. our compact QR (own workspace from its own lwork query) */
    double wq_geqrf;
    cqr_mkl_dgeqrf_compact(MKL_COL_MAJOR, m, n, ap, m, taup, &wq_geqrf, -1, &info, fmt,
                           nm);
    std::vector<double> work_geqrf((size_t)std::max<MKL_INT>((MKL_INT)wq_geqrf, 1));
    cqr_mkl_dgeqrf_compact(MKL_COL_MAJOR, m, n, ap, m, taup, work_geqrf.data(),
                           (MKL_INT)work_geqrf.size(), &info, fmt, nm);

    /* 2. our compact apply Q^T (own workspace) */
    double wq_ormqr;
    cqr_mkl_dormqr_compact(MKL_COL_MAJOR, 'L', 'T', m, nrhs, k, ap, m, taup, cp, m,
                           &wq_ormqr, -1, &info, fmt, nm);
    std::vector<double> work_ormqr((size_t)std::max<MKL_INT>((MKL_INT)wq_ormqr, 1));
    cqr_mkl_dormqr_compact(MKL_COL_MAJOR, 'L', 'T', m, nrhs, k, ap, m, taup, cp, m,
                           work_ormqr.data(), (MKL_INT)work_ormqr.size(), &info, fmt, nm);

    /* 3. our compact triangular solve R X = Q^T B (no workspace, no info) */
    cqr_mkl_dtrsm_compact(MKL_COL_MAJOR, MKL_LEFT, MKL_UPPER, MKL_NOTRANS, MKL_NONUNIT, n,
                          nrhs, 1.0, ap, m, cp, m, fmt, nm);

    std::vector<double> Xhat(nm * sB);
    auto Op = batch_ptrs<double>(Xhat.data(), nm, sB);
    mkl_dgeunpack_compact(MKL_COL_MAJOR, n, nrhs, Op.data(), n, cp, m, fmt, nm);

    double worst_fwd = 0, worst_res = 0;
    std::vector<double> AX(sB);
    const double nX = std::max(norm1(X.data(), n, nrhs), 1e-300); /* X is loop-invariant */
    for (int v = 0; v < nm; ++v) {
        const double *Av = A.data() + v * sA, *Bv = B.data() + v * sB;
        const double *Xv = Xhat.data() + v * sB;
        worst_fwd = std::max(worst_fwd, max_abs_diff(Xv, X.data(), sB) / nX);
        for (int j = 0; j < nrhs; ++j)
            for (int i = 0; i < n; ++i) {
                double s = 0;
                for (int l = 0; l < n; ++l)
                    s += Av[i + (size_t)l * n] * Xv[l + (size_t)j * n];
                AX[i + (size_t)j * n] = s;
            }
        worst_res = std::max(worst_res, max_abs_diff(AX.data(), Bv, sB) /
                                            std::max(norm1(Bv, n, nrhs), 1e-300));
    }
    const double rtol = 100.0 * n * eps;
    bool ok = (info == 0) && (worst_fwd <= rtol) && (worst_res <= rtol);
    std::printf(
        "  [suite2] V=%-2d nm=%-2d n=%-3d nrhs=%d | fwd %.2e res %.2e (rtol %.1e) "
        "%s\n",
        V, nm, n, nrhs, worst_fwd, worst_res, rtol, ok ? "OK" : "FAIL");
    return !ok;
}

} /* anonymous namespace */

int main()
{
    std::printf("MKL compact format = %d, V(double) = %d\n",
                (int)mkl_get_format_compact(), vlen(mkl_get_format_compact()));

    int fails = 0;

    /* Suite 1: cross-check vs mkl_?trsm_compact over the full feature matrix. */
    for (MKL_LAYOUT lay : {MKL_COL_MAJOR, MKL_ROW_MAJOR})
        for (MKL_SIDE side : {MKL_LEFT, MKL_RIGHT})
            for (MKL_UPLO uplo : {MKL_UPPER, MKL_LOWER})
                for (MKL_TRANSPOSE tr : {MKL_NOTRANS, MKL_TRANS})
                    for (MKL_DIAG diag : {MKL_NONUNIT, MKL_UNIT})
                        fails += suite1(lay, side, uplo, tr, diag, 8, 12, 5);
    /* a couple of larger / padded batches */
    fails +=
        suite1(MKL_COL_MAJOR, MKL_LEFT, MKL_UPPER, MKL_NOTRANS, MKL_NONUNIT, 16, 32, 4);
    fails +=
        suite1(MKL_COL_MAJOR, MKL_LEFT, MKL_UPPER, MKL_NOTRANS, MKL_NONUNIT, 11, 20, 6);
    /* few-RHS no-transpose left (n < 4): the column-axpy kernel route */
    for (int nrhs : {1, 2, 3})
        for (MKL_UPLO uplo : {MKL_UPPER, MKL_LOWER})
            for (MKL_DIAG diag : {MKL_NONUNIT, MKL_UNIT})
                fails +=
                    suite1(MKL_COL_MAJOR, MKL_LEFT, uplo, MKL_NOTRANS, diag, 8, 24, nrhs);

    /* Suite 2: end-to-end solver with no MKL compute kernel. */
    fails += suite2(8, 32, 5);
    fails += suite2(8, 64, 4);
    fails += suite2(7, 32, 6); /* padded partial last group */

    if (fails) {
        std::printf("\n%d CHECK(S) FAILED\n", fails);
        return 1;
    }
    std::printf("\nall checks passed\n");
    return 0;
}
