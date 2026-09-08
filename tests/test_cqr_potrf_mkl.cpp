/* test_cqr_potrf_mkl.cpp
 *
 * Validation of cqr_mkl_?potrf_compact against real Intel MKL and dense LAPACK,
 * through the genuine MKL Compact pipeline (mkl_dgepack_compact /
 * mkl_dgeunpack_compact). This is the design document's correctness gate against
 * standard dense LAPACK (cqr_mkl_dpotrf_compact_design.md section 7).
 *
 * Suite 1 (section 7.1) -- Factorization invariants vs dense LAPACK, for each
 *   (uplo, layout): a random SPD batch is factored by cqr_mkl_dpotrf_compact,
 *   unpacked, and per matrix checked against the LAPACK Cholesky contract:
 *     - reconstruction residual || L L^T - A ||_1 / ||A||_1 <= 20 n eps
 *       (U^T U for upper),
 *     - the strictly-opposite triangle is bit-for-bit unchanged from the input,
 *     - since the SPD factor is unique, elementwise vs LAPACKE_dpotrf,
 *       || L_cqr - L_lapack ||_1 / ||L_lapack||_1 <= 20 n eps.
 *
 * Suite 2 (section 7.2) -- Cross-check vs mkl_dpotrf_compact: the same packed
 *   batch factored by both, compact buffers compared elementwise at a small
 *   fixed tolerance (1e-9). Run for both layouts and both uplo.
 *
 * Suite 3 (section 7.3) -- End-to-end SPD solve AX = B: B = A X for known X;
 *   cqr_mkl<T>::potrf('L') -> mkl<T>::trsm('L','L','N') ->
 *   mkl<T>::trsm('L','L','T') must recover X. Gates forward error and the
 *   system residual at 100 n eps.
 *
 * Build: needs Intel MKL (headers + libmkl_rt); wired up by CMakeLists.txt.
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#include "test_mkl_util.hpp" /* cqr_mkl<T>, mkl<T>, lapack<T> + the MKL-free helpers */

#include <cstdio>
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>

using namespace cqr::test;

namespace {

/* Logical element (i,j) of a dense n x n matrix stored in the given layout. */
template <class T> T &elem(T *a, int i, int j, int n, bool rowmajor)
{
    return rowmajor ? a[(size_t)i * n + j] : a[(size_t)j * n + i];
}

/* ---------------- Suite 1: invariants vs dense LAPACK ------------------ */

template <class T>
int suite1(MKL_LAYOUT layout, MKL_UPLO uplo, int nm, int n, double cond)
{
    const double eps = std::numeric_limits<T>::epsilon();
    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = mkl<T>::vlen(fmt);
    const bool row = (layout == MKL_ROW_MAJOR);
    const bool up = (uplo == MKL_UPPER);
    const char ul = up ? 'U' : 'L';
    const size_t sA = (size_t)n * n;

    std::vector<T> A(nm * sA);
    for (int v = 0; v < nm; ++v)
        gen_spd(A.data() + v * sA, n, cond);

    /* pack the full symmetric A, factor with the routine under test, unpack */
    auto Ap = batch_ptrs<const T>(A.data(), nm, sA);
    MKL_INT sz_a = mkl<T>::get_size(n, n, fmt, nm);
    auto ap_buf = cqr::detail::mkl_alloc_bytes<T>(sz_a);
    T *ap = ap_buf.get();
    mkl<T>::gepack(layout, n, n, Ap.data(), n, ap, n, fmt, nm);

    MKL_INT info = 99;
    cqr_mkl<T>::potrf(layout, uplo, n, ap, n, &info, fmt, nm);

    std::vector<T> H(nm * sA);
    auto Hp = batch_ptrs<T>(H.data(), nm, sA);
    mkl<T>::geunpack(layout, n, n, Hp.data(), n, ap, n, fmt, nm);

    int fails = 0;
    if (info != 0) {
        ++fails;
        std::printf("    info = %d (expected 0)\n", (int)info);
    }

    double worst_res = 0, worst_el = 0, worst_untouched = 0;
    std::vector<T> Rec(sA), Lref(sA);
    for (int v = 0; v < nm; ++v) {
        const T *Av = A.data() + v * sA;
        T *Hv = H.data() + v * sA;

        /* reconstruction residual and untouched-triangle check */
        std::vector<T> R(sA, 0.0);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                double s = 0;
                int lmax = std::min(i, j);
                if (!up) /* A = L L^T */
                    for (int l = 0; l <= lmax; ++l)
                        s += elem(Hv, i, l, n, row) * elem(Hv, j, l, n, row);
                else /* A = U^T U */
                    for (int l = 0; l <= lmax; ++l)
                        s += elem(Hv, l, i, n, row) * elem(Hv, l, j, n, row);
                R[i + (size_t)j * n] = (T)s - Av[i + (size_t)j * n];
            }
        worst_res = std::max(worst_res,
                             norm1(R.data(), n, n) / std::max(norm1(Av, n, n), 1e-300));

        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                const bool named = up ? (i <= j) : (i >= j);
                if (!named)
                    worst_untouched = std::max<double>(
                        worst_untouched,
                        std::abs(elem(Hv, i, j, n, row) - Av[i + (size_t)j * n]));
            }

        /* elementwise vs LAPACKE_dpotrf (unique SPD factor -> a sharp signal) */
        std::copy(Av, Av + sA, Lref.begin());
        lapack<T>::potrf(LAPACK_COL_MAJOR, ul, n, Lref.data(), n);
        double el = 0, lref_norm = 0;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                const bool named = up ? (i <= j) : (i >= j);
                if (named)
                    el = std::max<double>(
                        el, std::abs(elem(Hv, i, j, n, row) - Lref[i + (size_t)j * n]));
            }
        lref_norm = norm1(Lref.data(), n, n); /* triangular factor L1 norm */
        worst_el = std::max(worst_el, el / std::max(lref_norm, 1e-300));
    }

    const double rtol = 20.0 * n * eps;
    bool ok = (worst_res <= rtol) && (worst_el <= rtol) && (worst_untouched == 0.0);
    fails += !ok;
    std::printf("  [suite1] %s uplo=%c V=%-2d nm=%-2d n=%-3d cond=%.0f | res %.2e "
                "el %.2e (%.1e) untouched %.0e %s\n",
                row ? "row" : "col", ul, V, nm, n, cond, worst_res, worst_el, rtol,
                worst_untouched, ok ? "OK" : "FAIL");
    return fails;
}

/* ---------------- Suite 2: cross-check vs mkl_dpotrf_compact ----------- */

template <class T> int suite2(MKL_LAYOUT layout, MKL_UPLO uplo, int nm, int n)
{
    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = mkl<T>::vlen(fmt);
    const bool row = (layout == MKL_ROW_MAJOR);
    const char ul = (uplo == MKL_UPPER) ? 'U' : 'L';
    const size_t sA = (size_t)n * n;

    std::vector<T> A(nm * sA);
    for (int v = 0; v < nm; ++v)
        gen_spd(A.data() + v * sA, n, 0.0);
    auto Ap = batch_ptrs<const T>(A.data(), nm, sA);

    MKL_INT sz_a = mkl<T>::get_size(n, n, fmt, nm);
    auto ap1 = cqr::detail::mkl_alloc_bytes<T>(sz_a);
    auto ap2 = cqr::detail::mkl_alloc_bytes<T>(sz_a);
    mkl<T>::gepack(layout, n, n, Ap.data(), n, ap1.get(), n, fmt, nm);
    mkl<T>::gepack(layout, n, n, Ap.data(), n, ap2.get(), n, fmt, nm);

    MKL_INT info = 0;
    cqr_mkl<T>::potrf(layout, uplo, n, ap1.get(), n, &info, fmt, nm);
    mkl<T>::potrf(layout, uplo, n, ap2.get(), n, &info, fmt, nm);

    /* compare the two compact buffers elementwise (same input, same convention).
     * The two implementations do not share an arithmetic order, so exact
     * agreement is not required; the tolerance confirms the same factor. */
    double da = max_abs_diff(ap1.get(), ap2.get(), (size_t)sz_a / sizeof(double));
    const double tol = cross_tol<T>();
    bool ok = (da <= tol);
    std::printf("  [suite2] %s uplo=%c V=%-2d nm=%-2d n=%-3d | max|ap-mkl| %.2e "
                "(tol %.0e) %s\n",
                row ? "row" : "col", ul, V, nm, n, da, tol, ok ? "OK" : "FAIL");
    return !ok;
}

/* ---------------- Suite 3: end-to-end AX = B (col-major lower) --------- */

template <class T> int suite3(int nm, int n, int nrhs)
{
    const double eps = std::numeric_limits<T>::epsilon();
    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = mkl<T>::vlen(fmt);
    const std::vector<T> X = known_solution<T>(n, nrhs);

    const size_t sA = (size_t)n * n, sB = (size_t)n * nrhs;
    std::vector<T> A(nm * sA), B(nm * sB);
    for (int v = 0; v < nm; ++v) {
        T *Av = A.data() + v * sA;
        gen_spd(Av, n, 0.0);
        matmul(n, nrhs, n, Av, n, X.data(), n, B.data() + v * sB, n); /* B = A X */
    }
    auto Ap = batch_ptrs<const T>(A.data(), nm, sA);
    auto Bp = batch_ptrs<const T>(B.data(), nm, sB);

    MKL_INT sz_a = mkl<T>::get_size(n, n, fmt, nm);
    MKL_INT sz_b = mkl<T>::get_size(n, nrhs, fmt, nm);
    auto ap_buf = cqr::detail::mkl_alloc_bytes<T>(sz_a);
    auto bp_buf = cqr::detail::mkl_alloc_bytes<T>(sz_b);
    T *ap = ap_buf.get(), *bp = bp_buf.get();
    mkl<T>::gepack(MKL_COL_MAJOR, n, n, Ap.data(), n, ap, n, fmt, nm);
    mkl<T>::gepack(MKL_COL_MAJOR, n, nrhs, Bp.data(), n, bp, n, fmt, nm);

    MKL_INT info = 99;

    /* 1. our compact Cholesky: A -> L (lower). No workspace. */
    cqr_mkl<T>::potrf(MKL_COL_MAJOR, MKL_LOWER, n, ap, n, &info, fmt, nm);

    /* 2. forward then back substitution with MKL's compact trsm:
     *    B := L^{-1} B, then B := L^{-T} B = A^{-1} B = X. */
    mkl<T>::trsm(MKL_COL_MAJOR, MKL_LEFT, MKL_LOWER, MKL_NOTRANS, MKL_NONUNIT, n, nrhs,
                 T(1), ap, n, bp, n, fmt, nm);
    mkl<T>::trsm(MKL_COL_MAJOR, MKL_LEFT, MKL_LOWER, MKL_TRANS, MKL_NONUNIT, n, nrhs,
                 T(1), ap, n, bp, n, fmt, nm);

    std::vector<T> Xhat(nm * sB);
    auto Op = batch_ptrs<T>(Xhat.data(), nm, sB);
    mkl<T>::geunpack(MKL_COL_MAJOR, n, nrhs, Op.data(), n, bp, n, fmt, nm);

    int fails = 0;
    if (info != 0) {
        ++fails;
        std::printf("    info = %d (expected 0)\n", (int)info);
    }
    double worst_fwd = 0, worst_res = 0;
    std::vector<T> AX(sB);
    for (int v = 0; v < nm; ++v) {
        const T *Av = A.data() + v * sA, *Bv = B.data() + v * sB;
        const T *Xv = Xhat.data() + v * sB;
        worst_fwd = std::max(worst_fwd, max_abs_diff(Xv, X.data(), sB) /
                                            std::max(norm1(X.data(), n, nrhs), 1e-300));
        matmul(n, nrhs, n, Av, n, Xv, n, AX.data(), n);
        worst_res = std::max(worst_res, max_abs_diff(AX.data(), Bv, sB) /
                                            std::max(norm1(Bv, n, nrhs), 1e-300));
    }
    const double rtol = 100.0 * n * eps;
    bool ok = (worst_fwd <= rtol && worst_res <= rtol);
    fails += !ok;
    std::printf(
        "  [suite3] V=%-2d nm=%-2d n=%-3d nrhs=%d | fwd %.2e res %.2e (rtol %.2e) %s\n",
        V, nm, n, nrhs, worst_fwd, worst_res, rtol, ok ? "OK" : "FAIL");
    return fails;
}

} /* anonymous namespace */

template <class T> int run_suites()
{
    std::printf("\n== %s: MKL compact format = %d, V = %d ==\n", compact<T>::name,
                (int)mkl_get_format_compact(), mkl<T>::vlen(mkl_get_format_compact()));

    int fails = 0;

    /* Suite 1: invariants vs dense LAPACK, over (uplo, layout), sizes and cond */
    const MKL_LAYOUT lays[] = {MKL_COL_MAJOR, MKL_ROW_MAJOR};
    const MKL_UPLO ups[] = {MKL_LOWER, MKL_UPPER};
    for (MKL_LAYOUT L : lays)
        for (MKL_UPLO U : ups) {
            fails += suite1<T>(L, U, 8, 30, 0.0);
            fails += suite1<T>(L, U, 16, 60, 0.0);
            fails += suite1<T>(L, U, 11, 43, 0.0); /* padded partial group */
            fails += suite1<T>(L, U, 8, 40, 2.0);  /* dynamic range (cond knob) */
        }
    fails += suite1<T>(MKL_COL_MAJOR, MKL_LOWER, 8, 128, 0.0);
    fails += suite1<T>(MKL_COL_MAJOR, MKL_LOWER, 4, 3, 0.0); /* smallest, padded */

    /* Suite 2: cross-check vs mkl_dpotrf_compact, both layouts and uplo */
    for (MKL_LAYOUT L : lays)
        for (MKL_UPLO U : ups) {
            fails += suite2<T>(L, U, 8, 30);
            fails += suite2<T>(L, U, 11, 40); /* padded partial group */
        }

    /* Suite 3: end-to-end SPD solver */
    fails += suite3<T>(8, 30, 5);
    fails += suite3<T>(16, 60, 4);
    fails += suite3<T>(7, 32, 6); /* padded partial group */

    return fails;
}

int main()
{
    const int fails = run_suites<double>() + run_suites<float>();
    if (fails) {
        std::printf("\n%d CHECK(S) FAILED\n", fails);
        return 1;
    }
    std::printf("\nall checks passed\n");
    return 0;
}
