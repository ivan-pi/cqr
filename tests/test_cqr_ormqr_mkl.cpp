/* test_cqr_ormqr_mkl.cpp
 *
 * Validation of cqr_mkl_dormqr_compact against real Intel MKL, implementing
 * the two test suites of cqr_mkl_dormqr_compact_design.md section 7 through
 * the genuine MKL Compact pipeline (mkl_dgepack_compact / mkl_dgeqrf_compact /
 * mkl_dtrsm_compact). This is the design document's "hard correctness gate
 * against standard dense LAPACK equivalents" (section 7.4).
 *
 * Suite 1 (section 7.1) -- Isolated Q application (Q^T B):
 *   A generated Householder representation (H, tau) and a target batch B are
 *   packed into MKL Compact format. cqr_mkl_dormqr_compact computes Q^T B in
 *   place. The checker materializes the dense reference Q^T B per matrix via
 *   dense LAPACK (LAPACKE_dormqr) and gates the application residual at
 *   rtol = 20 * n * eps (relative to the matrix L1 norm).
 *
 * Suite 2 (section 7.2) -- End-to-end AX = B solver:
 *   A known X (X(:,j) = j+1) defines B = A X. The compact pipeline runs
 *   mkl_dgeqrf_compact -> cqr_mkl_dormqr_compact('T') -> mkl_dtrsm_compact and
 *   the checker gates the forward error (Xhat - X) and the system residual
 *   (A Xhat - B) at rtol = 100 * n * eps (relative to the matrix L1 norm).
 *
 * Build: needs Intel MKL (headers + libmkl_rt); wired up by CMakeLists.txt.
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#include <mkl.h>
#include <mkl_compact.h>

#include "cqr_mkl_ext.h"
#include "cqr_mkl_alloc.h"       /* mkl_alloc_bytes (calls mkl_malloc; links MKL) */
#include "test_compact_util.hpp" /* frand, norm1, batch_ptrs, matmul, known_solution */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>

using namespace cqr::test;

namespace {

const double eps = std::numeric_limits<double>::epsilon();
using cqr::detail::vlen_for_format;

/* ---------------- Suite 1: isolated op(Q) C (section 7.1) -------------- */

/* L1 (max column sum) norm of an m x n matrix in the given layout. */
double norm1_layout(const double *M, int m, int n, bool rowmajor, int ld)
{
    double mx = 0;
    for (int j = 0; j < n; ++j) {
        double s = 0;
        for (int i = 0; i < m; ++i)
            s += std::abs(M[rowmajor ? (size_t)i * ld + j : i + (size_t)j * ld]);
        mx = std::max(mx, s);
    }
    return mx;
}

/* Generalized Suite 1: validate cqr_mkl_dormqr_compact's op(Q) application
 * for any (layout, side, trans) against dense LAPACKE_dormqr. A is the s x k
 * reflector batch with s = m (side='L') or n (side='R'); C is m x n. */
int suite1(MKL_LAYOUT layout, char side, char trans, int nm, int m, int n, int k)
{
    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = vlen_for_format<double>(fmt);
    const bool rowmajor = (layout == MKL_ROW_MAJOR);
    const bool left = (side == 'L' || side == 'l');
    const int s = left ? m : n; /* A is s x k, Q is s x s */
    const int lap = rowmajor ? LAPACK_ROW_MAJOR : LAPACK_COL_MAJOR;
    const int ldH = rowmajor ? k : s; /* dense leading dims */
    const int ldC = rowmajor ? n : m;

    const size_t sH = (size_t)s * k, sT = (size_t)k, sB = (size_t)m * n;
    std::vector<double> H(nm * sH), tau(nm * sT), B(nm * sB), Bref(nm * sB),
        Bout(nm * sB);

    for (int v = 0; v < nm; ++v) {
        double *Hv = H.data() + v * sH, *tv = tau.data() + v * sT;
        double *Bv = B.data() + v * sB, *Rv = Bref.data() + v * sB;
        for (size_t i = 0; i < sH; ++i)
            Hv[i] = frand<double>();
        /* boost the (i,i) diagonal (same offset in both layouts) */
        for (int i = 0; i < std::min(s, k); ++i)
            Hv[(size_t)i * ldH + i] += 2.0;
        /* turn H into a real Householder representation via dense QR */
        LAPACKE_dgeqrf(lap, s, k, Hv, ldH, tv);
        for (size_t i = 0; i < sB; ++i)
            Bv[i] = frand<double>();
        std::copy(Bv, Bv + sB, Rv);
        /* dense reference: op(Q) C */
        LAPACKE_dormqr(lap, side, trans, m, n, k, Hv, ldH, tv, Rv, ldC);
    }

    /* pack into MKL Compact format (tau is a vector: layout-agnostic) */
    auto Hp = batch_ptrs<const double>(H.data(), nm, sH);
    auto Tp = batch_ptrs<const double>(tau.data(), nm, sT);
    auto Bp = batch_ptrs<const double>(B.data(), nm, sB);

    MKL_INT sz_a = mkl_dget_size_compact(s, k, fmt, nm);
    MKL_INT sz_t = mkl_dget_size_compact(k, 1, fmt, nm);
    MKL_INT sz_c = mkl_dget_size_compact(m, n, fmt, nm);
    auto ap_buf = cqr::detail::mkl_alloc_bytes<double>(sz_a);
    auto taup_buf = cqr::detail::mkl_alloc_bytes<double>(sz_t);
    auto cp_buf = cqr::detail::mkl_alloc_bytes<double>(sz_c);
    double *ap = ap_buf.get(), *taup = taup_buf.get(), *cp = cp_buf.get();

    const MKL_INT ldap = rowmajor ? k : s; /* compact leading dims */
    const MKL_INT ldcp = rowmajor ? n : m;
    mkl_dgepack_compact(layout, s, k, Hp.data(), ldH, ap, ldap, fmt, nm);
    mkl_dgepack_compact(MKL_COL_MAJOR, k, 1, Tp.data(), k, taup, k, fmt, nm);
    mkl_dgepack_compact(layout, m, n, Bp.data(), ldC, cp, ldcp, fmt, nm);

    /* routine under test (workspace query, then compute). info is a single
     * scalar (MKL Compact convention), not a per-matrix array. */
    MKL_INT info = 99;
    double wq;
    cqr_mkl_dormqr_compact(layout, side, trans, m, n, k, ap, ldap, taup, cp, ldcp, &wq,
                           -1, &info, fmt, nm);
    cqr_mkl_dormqr_compact(layout, side, trans, m, n, k, ap, ldap, taup, cp, ldcp, &wq,
                           (MKL_INT)wq, &info, fmt, nm);

    /* unpack and compare against the dense reference */
    auto Op = batch_ptrs<double>(Bout.data(), nm, sB);
    mkl_dgeunpack_compact(layout, m, n, Op.data(), ldC, cp, ldcp, fmt, nm);

    int fails = 0;
    if (info != 0) {
        ++fails;
        std::printf("    info = %d (expected 0)\n", (int)info);
    }
    double worst = 0;
    for (int v = 0; v < nm; ++v) {
        const double *Bo = Bout.data() + v * sB, *Rv = Bref.data() + v * sB;
        double resid = max_abs_diff(Bo, Rv, sB);
        double rel = resid / std::max(norm1_layout(Rv, m, n, rowmajor, ldC), 1e-300);
        worst = std::max(worst, rel);
    }
    const double rtol = 20.0 * s * eps;
    bool ok = (worst <= rtol);
    fails += !ok;
    std::printf("  [suite1] %s side=%c trans=%c V=%-2d nm=%-2d m=%-3d n=%-3d k=%-3d | "
                "rel resid %.2e (rtol %.2e) %s\n",
                rowmajor ? "row" : "col", side, trans, V, nm, m, n, k, worst, rtol,
                ok ? "OK" : "FAIL");

    return fails;
}

/* ---------------- Suite 2: end-to-end AX = B (section 7.2) ------------- */

int suite2(int nm, int n, int nrhs)
{
    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = vlen_for_format<double>(fmt);
    const int m = n, k = n;
    const std::vector<double> X = known_solution<double>(n, nrhs);

    /* each batch in one contiguous column-major buffer, matrix v at v*stride */
    const size_t sA = (size_t)n * n, sB = (size_t)n * nrhs;
    std::vector<double> A(nm * sA), B(nm * sB);
    for (int v = 0; v < nm; ++v) {
        double *Av = A.data() + v * sA;
        gen_boosted(Av, n, n); /* diagonal boost tames cond */
        matmul(n, nrhs, n, Av, n, X.data(), n, B.data() + v * sB, n); /* B = A X */
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

    MKL_INT info = 99; /* compact info is a single scalar (MKL convention) */

    /* 1. compact QR: ap <- (H, R), taup <- tau   (workspace query first) */
    double wq;
    mkl_dgeqrf_compact(MKL_COL_MAJOR, m, n, ap, m, taup, &wq, -1, &info, fmt, nm);
    MKL_INT lwork = (MKL_INT)wq;
    std::vector<double> work((size_t)std::max<MKL_INT>(lwork, 1));
    mkl_dgeqrf_compact(MKL_COL_MAJOR, m, n, ap, m, taup, work.data(), lwork, &info, fmt,
                       nm);

    /* 2. routine under test: cp <- Q^T B */
    cqr_mkl_dormqr_compact(MKL_COL_MAJOR, 'L', 'T', m, nrhs, k, ap, m, taup, cp, m, &wq,
                           -1, &info, fmt, nm);
    cqr_mkl_dormqr_compact(MKL_COL_MAJOR, 'L', 'T', m, nrhs, k, ap, m, taup, cp, m, &wq,
                           (MKL_INT)wq, &info, fmt, nm);

    /* 3. compact upper-triangular solve: cp <- R^{-1} (Q^T B) = Xhat */
    mkl_dtrsm_compact(MKL_COL_MAJOR, MKL_LEFT, MKL_UPPER, MKL_NOTRANS, MKL_NONUNIT, n,
                      nrhs, 1.0, ap, m, cp, m, fmt, nm);

    std::vector<double> Xhat(nm * sB);
    auto Op = batch_ptrs<double>(Xhat.data(), nm, sB);
    mkl_dgeunpack_compact(MKL_COL_MAJOR, n, nrhs, Op.data(), n, cp, m, fmt, nm);

    int fails = 0;
    if (info != 0) {
        ++fails;
        std::printf("    info = %d (expected 0)\n", (int)info);
    }
    double worst_fwd = 0, worst_res = 0;
    std::vector<double> AX(sB);
    for (int v = 0; v < nm; ++v) {
        const double *Av = A.data() + v * sA, *Bv = B.data() + v * sB;
        const double *Xv = Xhat.data() + v * sB;
        double fwd =
            max_abs_diff(Xv, X.data(), sB) / std::max(norm1(X.data(), n, nrhs), 1e-300);
        worst_fwd = std::max(worst_fwd, fwd);
        matmul(n, nrhs, n, Av, n, Xv, n, AX.data(), n); /* AX = A Xhat */
        double res =
            max_abs_diff(AX.data(), Bv, sB) / std::max(norm1(Bv, n, nrhs), 1e-300);
        worst_res = std::max(worst_res, res);
    }
    const double rtol = 100.0 * n * eps;
    bool ok = (worst_fwd <= rtol && worst_res <= rtol);
    fails += !ok;
    std::printf("  [suite2] V=%-2d nm=%-2d n=%-3d nrhs=%d | fwd err %.2e res %.2e (rtol "
                "%.2e) %s\n",
                V, nm, n, nrhs, worst_fwd, worst_res, rtol, ok ? "OK" : "FAIL");

    return fails;
}

} /* anonymous namespace */

int main()
{
    std::printf("MKL compact format = %d, V(double) = %d\n",
                (int)mkl_get_format_compact(),
                vlen_for_format<double>(mkl_get_format_compact()));

    int fails = 0;
    /* Suite 1: isolated op(Q) C over the full feature matrix
     * (layout x side x trans), across batch sizes / shapes. */
    for (MKL_LAYOUT lay : {MKL_COL_MAJOR, MKL_ROW_MAJOR})
        for (char side : {'L', 'R'})
            for (char tr : {'N', 'T'}) {
                /* side='L': A,Q are m x m (k<=m); side='R': n x n (k<=n) */
                fails += suite1(lay, side, tr, 8, 43, 5, side == 'L' ? 43 : 5);
                fails += suite1(lay, side, tr, 16, 32, 8, side == 'L' ? 32 : 8);
                /* k < dim, padded partial last group */
                fails += suite1(lay, side, tr, 11, 32, 6, side == 'L' ? 20 : 4);
            }

    /* Suite 2: end-to-end solver, shapes from section 7.3 (32..512) */
    fails += suite2(8, 32, 5);
    fails += suite2(8, 64, 4);
    fails += suite2(16, 128, 3);
    fails += suite2(7, 32, 6); /* padded partial last group */

    if (fails) {
        std::printf("\n%d CHECK(S) FAILED\n", fails);
        return 1;
    }
    std::printf("\nall checks passed\n");
    return 0;
}
