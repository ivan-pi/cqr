/* test_cqr_sytrfnp_mkl.cpp
 *
 * Validation of cqr_mkl_?sytrfnp_compact, cqr_mkl_?sytrsnp_compact and
 * cqr_mkl_?sysvnp_compact against real Intel MKL, through the genuine MKL
 * Compact pipeline (mkl_?gepack_compact / mkl_?geunpack_compact). This is the
 * design document's correctness gate (docs/cqr_mkl_dsytrfnp_compact_design.md
 * section 7). Standard LAPACK has no unpivoted dense LDL^T (?sytrf is
 * Bunch-Kaufman-pivoted, so its factors differ elementwise), so the dense
 * reference here is the algebraic contract itself plus MKL's own unpivoted
 * compact LU:
 *
 * Suite 1 (section 7.1) -- Factorization invariants, for each (uplo, layout):
 *   a random symmetric *indefinite* batch (known-good unpivoted LDL^T by
 *   construction) is factored by cqr_mkl<T>::sytrfnp, unpacked, and per matrix
 *   checked:
 *     - reconstruction residual || L D L^T - A ||_1 / ||A||_1 <= 20 n eps
 *       (U^T D U for upper),
 *     - the strictly-opposite triangle is bit-for-bit unchanged from the input.
 *
 * Suite 2 (section 7.2) -- Cross-check vs mkl_?getrfnp_compact: unpivoted LU of
 *   a symmetric matrix satisfies A = L (D L^T), i.e. it shares the unit-lower L
 *   and its U carries D on the diagonal. The same batch is factored by both
 *   routines (col-major lower vs full LU) and the strict lower triangles and
 *   diagonals are compared at the cross-check tolerance -- two independent
 *   implementations of the same pivots.
 *
 * Suite 3 (section 7.3) -- End-to-end indefinite solve A X = B: B = A X for
 *   known X; cqr_mkl<T>::sytrfnp -> cqr_mkl<T>::sytrsnp must recover X, over
 *   both uplo and both layouts, gating the system residual at 100 n eps and the
 *   forward error at 500 n eps; and cqr_mkl<T>::sysvnp on the same packed input
 *   must reproduce the two-step factor and X bit-for-bit.
 *
 * Build: needs Intel MKL (headers + libmkl_rt); wired up by CMakeLists.txt.
 *
 * Assisted-by: Claude
 */

#include "test_mkl_util.hpp" /* cqr_mkl<T>, mkl<T> + the MKL-free helpers */

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

/* ---------------- Suite 1: factorization invariants ------------------- */

template <class T> int suite1(MKL_LAYOUT layout, MKL_UPLO uplo, int nm, int n)
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
        gen_sym_ldlt(A.data() + v * sA, n);

    /* pack the full symmetric A, factor with the routine under test, unpack */
    auto Ap = batch_ptrs<const T>(A.data(), nm, sA);
    MKL_INT sz_a = mkl<T>::get_size(n, n, fmt, nm);
    auto ap_buf = cqr::detail::mkl_alloc_bytes<T>(sz_a);
    T *ap = ap_buf.get();
    mkl<T>::gepack(layout, n, n, Ap.data(), n, ap, n, fmt, nm);

    MKL_INT info = 99;
    cqr_mkl<T>::sytrfnp(layout, uplo, n, ap, n, &info, fmt, nm);

    std::vector<T> H(nm * sA);
    auto Hp = batch_ptrs<T>(H.data(), nm, sA);
    mkl<T>::geunpack(layout, n, n, Hp.data(), n, ap, n, fmt, nm);

    int fails = 0;
    if (info != 0) {
        ++fails;
        std::printf("    info = %ld (expected 0)\n", (long)info);
    }

    double worst_res = 0, worst_untouched = 0;
    std::vector<T> R(sA);
    for (int v = 0; v < nm; ++v) {
        const T *Av = A.data() + v * sA;
        T *Hv = H.data() + v * sA;

        /* reconstruction residual: unit factor off the diagonal, D on it */
        auto at = [&](int i, int j) { return elem(Hv, i, j, n, row); };
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                R[i + (size_t)j * n] =
                    (T)(ldlt_reconstruct(at, i, j, up) - (double)Av[i + (size_t)j * n]);
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
    }

    const double rtol = 20.0 * n * eps;
    bool ok = (worst_res <= rtol) && (worst_untouched == 0.0);
    fails += !ok;
    std::printf("  [suite1] %s uplo=%c V=%-2d nm=%-2d n=%-3d | res %.2e (%.1e) "
                "untouched %.0e %s\n",
                row ? "row" : "col", ul, V, nm, n, worst_res, rtol, worst_untouched,
                ok ? "OK" : "FAIL");
    return fails;
}

/* -------- Suite 2: cross-check vs mkl_?getrfnp_compact (A = L * DL^T) -- */

template <class T> int suite2(int nm, int n)
{
    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = mkl<T>::vlen(fmt);
    const size_t sA = (size_t)n * n;

    std::vector<T> A(nm * sA);
    for (int v = 0; v < nm; ++v)
        gen_sym_ldlt(A.data() + v * sA, n);
    auto Ap = batch_ptrs<const T>(A.data(), nm, sA);

    MKL_INT sz_a = mkl<T>::get_size(n, n, fmt, nm);
    auto ap1 = cqr::detail::mkl_alloc_bytes<T>(sz_a);
    auto ap2 = cqr::detail::mkl_alloc_bytes<T>(sz_a);
    mkl<T>::gepack(MKL_COL_MAJOR, n, n, Ap.data(), n, ap1.get(), n, fmt, nm);
    mkl<T>::gepack(MKL_COL_MAJOR, n, n, Ap.data(), n, ap2.get(), n, fmt, nm);

    MKL_INT info = 0;
    cqr_mkl<T>::sytrfnp(MKL_COL_MAJOR, MKL_LOWER, n, ap1.get(), n, &info, fmt, nm);
    mkl<T>::getrfnp(MKL_COL_MAJOR, n, n, ap2.get(), n, &info, fmt, nm);

    /* unpack both; the unpivoted LU of symmetric A is A = L * (D L^T), so its
     * unit-lower L must match ours and its U diagonal must be our D. The two
     * implementations do not share an arithmetic order, so a small fixed
     * tolerance confirms the same pivots, not bit equality. */
    std::vector<T> F(nm * sA), G(nm * sA);
    auto Fp = batch_ptrs<T>(F.data(), nm, sA);
    auto Gp = batch_ptrs<T>(G.data(), nm, sA);
    mkl<T>::geunpack(MKL_COL_MAJOR, n, n, Fp.data(), n, ap1.get(), n, fmt, nm);
    mkl<T>::geunpack(MKL_COL_MAJOR, n, n, Gp.data(), n, ap2.get(), n, fmt, nm);

    double dl = 0, dd = 0;
    for (int v = 0; v < nm; ++v) {
        const T *Fv = F.data() + v * sA, *Gv = G.data() + v * sA;
        for (int j = 0; j < n; ++j) {
            dd = std::max<double>(
                dd, std::abs(Fv[j + (size_t)j * n] - Gv[j + (size_t)j * n]));
            for (int i = j + 1; i < n; ++i)
                dl = std::max<double>(
                    dl, std::abs(Fv[i + (size_t)j * n] - Gv[i + (size_t)j * n]));
        }
    }
    const double tol = cross_tol<T>();
    bool ok = (dl <= tol) && (dd <= tol);
    std::printf("  [suite2] vs getrfnp V=%-2d nm=%-2d n=%-3d | max|L-L_lu| %.2e "
                "max|D-diag(U)| %.2e (tol %.0e) %s\n",
                V, nm, n, dl, dd, tol, ok ? "OK" : "FAIL");
    return !ok;
}

/* ---------------- Suite 3: end-to-end indefinite solve ----------------- */

template <class T> int suite3(MKL_LAYOUT layout, MKL_UPLO uplo, int nm, int n, int nrhs)
{
    const double eps = std::numeric_limits<T>::epsilon();
    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = mkl<T>::vlen(fmt);
    const bool row = (layout == MKL_ROW_MAJOR);
    const char ul = (uplo == MKL_UPPER) ? 'U' : 'L';

    std::vector<T> X((size_t)n * nrhs);
    for (int j = 0; j < nrhs; ++j)
        for (int i = 0; i < n; ++i)
            X[i + (size_t)j * n] = T(j + 1) + T(0.25) * T(i % 4);

    const size_t sA = (size_t)n * n, sB = (size_t)n * nrhs;
    std::vector<T> A(nm * sA), B(nm * sB);
    for (int v = 0; v < nm; ++v) {
        T *Av = A.data() + v * sA;
        gen_sym_ldlt(Av, n);
        matmul(n, nrhs, n, Av, n, X.data(), n, B.data() + v * sB, n); /* B = A X */
    }
    auto Ap = batch_ptrs<const T>(A.data(), nm, sA);

    /* mkl_?ge(un)pack_compact read/write the dense side in `layout` too, so the
     * row-major runs pack from (and unpack to) a row-major staging copy of the
     * non-square B with ld = nrhs. The symmetric square A needs no staging: its
     * row-major image is itself. */
    std::vector<T> Bsrc;
    const T *bsrc = B.data();
    if (row) {
        Bsrc.resize(nm * sB);
        for (int v = 0; v < nm; ++v)
            for (int j = 0; j < nrhs; ++j)
                for (int i = 0; i < n; ++i)
                    Bsrc[v * sB + (size_t)i * nrhs + j] = B[v * sB + i + (size_t)j * n];
        bsrc = Bsrc.data();
    }
    auto Bp = batch_ptrs<const T>(bsrc, nm, sB);

    MKL_INT sz_a = mkl<T>::get_size(n, n, fmt, nm);
    MKL_INT sz_b = mkl<T>::get_size(n, nrhs, fmt, nm);
    auto ap_buf = cqr::detail::mkl_alloc_bytes<T>(sz_a);
    auto bp_buf = cqr::detail::mkl_alloc_bytes<T>(sz_b);
    auto ap2_buf = cqr::detail::mkl_alloc_bytes<T>(sz_a); /* the fused call's copies */
    auto bp2_buf = cqr::detail::mkl_alloc_bytes<T>(sz_b);
    T *ap = ap_buf.get(), *bp = bp_buf.get(), *ap2 = ap2_buf.get(), *bp2 = bp2_buf.get();
    const MKL_INT ldb = row ? nrhs : n;
    mkl<T>::gepack(layout, n, n, Ap.data(), n, ap, n, fmt, nm);
    mkl<T>::gepack(layout, n, nrhs, Bp.data(), ldb, bp, ldb, fmt, nm);
    mkl<T>::gepack(layout, n, n, Ap.data(), n, ap2, n, fmt, nm);
    mkl<T>::gepack(layout, n, nrhs, Bp.data(), ldb, bp2, ldb, fmt, nm);

    MKL_INT info_f = 99, info_s = 99, info_v = 99;
    cqr_mkl<T>::sytrfnp(layout, uplo, n, ap, n, &info_f, fmt, nm);
    cqr_mkl<T>::sytrsnp(layout, uplo, n, nrhs, ap, n, bp, ldb, &info_s, fmt, nm);
    cqr_mkl<T>::sysvnp(layout, uplo, n, nrhs, ap2, n, bp2, ldb, &info_v, fmt, nm);

    /* fused vs two-step: the same kernels in the same order -> the same bits,
     * over the whole compact buffers (padded lanes included) */
    const size_t na = (size_t)sz_a / sizeof(T), nb = (size_t)sz_b / sizeof(T);
    const bool fused_same = std::equal(ap, ap + na, ap2) && std::equal(bp, bp + nb, bp2);

    std::vector<T> Xout(nm * sB), Xhat(nm * sB);
    auto Op = batch_ptrs<T>(Xout.data(), nm, sB);
    mkl<T>::geunpack(layout, n, nrhs, Op.data(), ldb, bp, ldb, fmt, nm);
    if (row) { /* stage back to column-major for the checks */
        for (int v = 0; v < nm; ++v)
            for (int j = 0; j < nrhs; ++j)
                for (int i = 0; i < n; ++i)
                    Xhat[v * sB + i + (size_t)j * n] =
                        Xout[v * sB + (size_t)i * nrhs + j];
    }
    else {
        Xhat = Xout;
    }

    int fails = 0;
    if (info_f != 0 || info_s != 0 || info_v != 0) {
        ++fails;
        std::printf("    info = %ld/%ld/%ld (expected 0/0/0)\n", (long)info_f,
                    (long)info_s, (long)info_v);
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
    /* the residual is what the backward-stable sweeps control; the forward
     * error additionally carries cond(A), so its gate gets headroom */
    const double rtol_res = 100.0 * n * eps;
    const double rtol_fwd = 500.0 * n * eps;
    bool ok = (worst_fwd <= rtol_fwd && worst_res <= rtol_res && fused_same);
    fails += !ok;
    std::printf("  [suite3] %s uplo=%c V=%-2d nm=%-2d n=%-3d nrhs=%d | fwd %.2e "
                "(%.1e) res %.2e (%.1e) sysv==trf+trs:%s %s\n",
                row ? "row" : "col", ul, V, nm, n, nrhs, worst_fwd, rtol_fwd, worst_res,
                rtol_res, fused_same ? "yes" : "NO", ok ? "OK" : "FAIL");
    return fails;
}

} /* anonymous namespace */

template <class T> int run_suites()
{
    std::printf("\n== %s: MKL compact format = %d, V = %d ==\n", compact<T>::name,
                (int)mkl_get_format_compact(), mkl<T>::vlen(mkl_get_format_compact()));

    int fails = 0;

    /* Suite 1: factorization invariants over (uplo, layout), sizes incl. padding */
    const MKL_LAYOUT lays[] = {MKL_COL_MAJOR, MKL_ROW_MAJOR};
    const MKL_UPLO ups[] = {MKL_LOWER, MKL_UPPER};
    for (MKL_LAYOUT L : lays)
        for (MKL_UPLO U : ups) {
            fails += suite1<T>(L, U, 8, 30);
            fails += suite1<T>(L, U, 16, 60);
            fails += suite1<T>(L, U, 11, 43); /* padded partial group */
        }
    fails += suite1<T>(MKL_COL_MAJOR, MKL_LOWER, 8, 128);
    fails += suite1<T>(MKL_COL_MAJOR, MKL_LOWER, 4, 3); /* smallest, padded */

    /* Suite 2: (L, D) cross-check vs MKL's unpivoted compact LU */
    fails += suite2<T>(8, 30);
    fails += suite2<T>(11, 40); /* padded partial group */

    /* Suite 3: end-to-end indefinite solver, both uplo and layouts, two-step
     * and fused */
    for (MKL_LAYOUT L : lays)
        for (MKL_UPLO U : ups)
            fails += suite3<T>(L, U, 8, 30, 5);
    fails += suite3<T>(MKL_COL_MAJOR, MKL_LOWER, 16, 60, 4);
    fails += suite3<T>(MKL_COL_MAJOR, MKL_LOWER, 7, 32, 6); /* padded partial group */
    fails += suite3<T>(MKL_COL_MAJOR, MKL_UPPER, 6, 25, 1); /* single RHS */

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
