/* cqr_sytrfnp_compact.hpp
 *
 * Compact (interleaved-batch) LDL^T factorization of symmetric matrices,
 * without pivoting ("np"), templated on scalar type T and interleave width V:
 *
 *     A = L D L^T  (uplo lower),   A = U^T D U  (uplo upper),
 *
 * with L (U) unit lower (upper) triangular and D diagonal, run V matrices at a
 * time. On exit D sits on the diagonal and the strict off-diagonal of the named
 * triangle holds L (U); the unit diagonal is implied, not stored.
 *
 * sytrfnp_compact()         -- tuned column-major, lower path.
 * sytrfnp_compact_general() -- column- or row-major, lower or upper.
 *
 * The square-root-free sibling of cqr_potrf_compact.hpp: symmetric *indefinite*
 * matrices factor too (negative pivots are fine -- there is no sqrt), which is
 * the point of LDL^T over Cholesky. Paired with the trsm kernels through
 * cqr_sytrsnp_compact.hpp it factors and solves batched symmetric systems.
 *
 * Algorithm: the unblocked right-looking sweep, V matrices at a time -- potf2
 * with the sqrt pivot replaced by a reciprocal. Compact format stores element
 * (i,j) of all V matrices contiguously, so the scalar code lifts with double ->
 * V-wide vector, one lane per matrix; like Cholesky the math has no
 * data-dependent branch, so no lane mask is needed. Per pivot column j (lower):
 *
 *     d = A(j,j);  invd = 1/d                   (pivot stays in place as D(j,j))
 *     A(i,j) *= invd              for i > j              (scale -> L(i,j))
 *     A(i,jj)-= A(i,j)*(A(jj,j)*d) for jj > j, i >= jj   (rank-1 trailing update,
 *                                                         L(i,j) * D(j,j) * L(jj,j))
 *
 * Only the lower trapezoid is touched, so the strictly-upper triangle passes
 * through untouched. The O(n^3) trailing update is register-blocked JB = 4
 * columns at a time so each pivot entry A(i,j) is reused across four columns,
 * as in potrf/geqrf; the extra multiply by d costs one op per block column.
 *
 * Pivots are the *updated* (Schur-complement) diagonal entries, not the original
 * ones: a zero original A(j,j) is harmless unless the pivot itself is zero. A
 * zero pivot -- a singular leading principal minor -- gets Inf/NaN in its lane's
 * factor; there is no info = j early-exit and no pivoting (both are per-lane
 * branching that does not vectorize; design section 6.2). This mirrors MKL's own
 * compact API, whose LU is likewise unpivoted (mkl_?getrfnp_compact).
 *
 * Layouts (design section 6.4): the four (layout, uplo) cases pair by transpose
 * duality (A symmetric). Column-major lower and row-major upper are contiguous
 * (a factor column is one pack apart per row) -> tuned sytrfnp_compact_group;
 * column-major upper and row-major lower are strided ->
 * sytrfnp_compact_group_strided over the same math. Note the upper convention is
 * A = U^T D U, the transpose dual of the lower factorization (matching potrf's
 * A = U^T U), NOT LAPACK ?sytrf's A = U D U^T (design section 6.3).
 *
 * Compact storage (matches mkl_?gepack_compact); group g = idx/V, slot v = idx%V:
 *     A_v(i,j) = ap[ g*ldap*n*V + (j*ldap + i)*V + v ]   (column-major)
 * Row-major swaps the in-matrix roles (i*ldap + j); the group stride ldap*n*V is
 * the same either way (A is n x n).
 *
 * Assisted-by: Claude
 */

#ifndef CQR_SYTRFNP_COMPACT_HPP
#define CQR_SYTRFNP_COMPACT_HPP

#include "cqr_compact_common.hpp" /* pack<T,V>, BatchView, make_view */

#include <cstddef>
#include <cassert>
#include <type_traits>

namespace cqr {
namespace detail {

/* ------------------------------------------------------------------
 * One group of V interleaved matrices, column-major lower (tuned path).
 *
 * a_ points at element (0,0) of the group; column j is contiguous (row step =
 * one V-wide pack), so the pivot reciprocal, the column scaling, and the rank-1
 * trailing update all walk contiguous packs. Only the lower trapezoid (i >= j)
 * is read or written. Row-major upper folds onto this same kernel by transpose
 * duality (see the file header and sytrfnp_compact_general below).
 * ------------------------------------------------------------------ */

template <typename T, int V, typename Int = int>
void sytrfnp_compact_group(Int n, T *a_, Int ldap)
{
    using VT = typename pack<T, V>::type;
    static_assert(std::is_floating_point<T>::value,
                  "sytrfnp_compact is defined for real float/double");
    assert(ldap >= n);

    VT *A = reinterpret_cast<VT *>(a_);

    for (Int j = 0; j < n; ++j) {
        VT *aj = A + j * ldap; /* pivot column j */

        /* pivot: d = A(j,j) stays in place as D(j,j); scale the sub-diagonal
         * column by 1/d so it holds L(:,j). */
        const VT d = aj[j];
        const VT invd = T(1) / d;
        for (Int i = j + 1; i < n; ++i)
            aj[i] = aj[i] * invd;

        /* symmetric rank-1 trailing update A(i,jj) -= L(i,j)*D(j,j)*L(jj,j) =
         * A(i,j)*(A(jj,j)*d), only the lower trapezoid (jj > j, i >= jj), 4
         * columns at a time so each pivot entry aj[i] is loaded once and reused
         * across the block; the multiply by d folds into the per-column w. */
        Int jj = j + 1;
        for (; jj + 4 <= n; jj += 4) {
            VT *c0 = A + (jj + 0) * ldap;
            VT *c1 = A + (jj + 1) * ldap;
            VT *c2 = A + (jj + 2) * ldap;
            VT *c3 = A + (jj + 3) * ldap;
            const VT w0 = aj[jj + 0] * d, w1 = aj[jj + 1] * d, w2 = aj[jj + 2] * d,
                     w3 = aj[jj + 3] * d;

            /* near-diagonal triangular corner: column jj+c touches only rows
             * i >= jj+c, so within rows jj..jj+3 the block fills in triangularly. */
            c0[jj + 0] -= aj[jj + 0] * w0;

            c0[jj + 1] -= aj[jj + 1] * w0;
            c1[jj + 1] -= aj[jj + 1] * w1;

            c0[jj + 2] -= aj[jj + 2] * w0;
            c1[jj + 2] -= aj[jj + 2] * w1;
            c2[jj + 2] -= aj[jj + 2] * w2;

            c0[jj + 3] -= aj[jj + 3] * w0;
            c1[jj + 3] -= aj[jj + 3] * w1;
            c2[jj + 3] -= aj[jj + 3] * w2;
            c3[jj + 3] -= aj[jj + 3] * w3;

            /* rectangular tail below the corner: all 4 columns, aj[i] reused 4x */
            for (Int i = jj + 4; i < n; ++i) {
                const VT av = aj[i];
                c0[i] -= av * w0;
                c1[i] -= av * w1;
                c2[i] -= av * w2;
                c3[i] -= av * w3;
            }
        }

        /* remainder columns (fewer than 4 left) */
        for (; jj < n; ++jj) {
            VT *cj = A + jj * ldap;
            const VT w = aj[jj] * d;
            for (Int i = jj; i < n; ++i)
                cj[i] -= aj[i] * w;
        }
    }
}

/* ------------------------------------------------------------------
 * One group, fully general: column- or row-major, lower or upper, via
 * BatchView strides. Same math; only the addressing differs. The view A(i,j)
 * presents the symmetric matrix so that factoring its lower triangle (i >= j)
 * reads/writes the named triangle's storage and lands the factor there.
 * ------------------------------------------------------------------ */

template <typename T, int V, typename Int = int>
void sytrfnp_compact_group_strided(Int n, BatchView<typename pack<T, V>::type, Int> A)
{
    using VT = typename pack<T, V>::type;
    static_assert(std::is_floating_point<T>::value,
                  "sytrfnp_compact is defined for real float/double");
    assert(A.special && A.panel);

    for (Int j = 0; j < n; ++j) {
        /* pivot: d = A(j,j) stays in place; scale the sub-diagonal column */
        const VT d = A(j, j);
        const VT invd = T(1) / d;
        for (Int i = j + 1; i < n; ++i)
            A(i, j) = A(i, j) * invd;

        /* symmetric rank-1 trailing update, lower trapezoid, 4 columns at a time */
        Int jj = j + 1;
        for (; jj + 4 <= n; jj += 4) {
            const VT w0 = A(jj + 0, j) * d, w1 = A(jj + 1, j) * d, w2 = A(jj + 2, j) * d,
                     w3 = A(jj + 3, j) * d;

            /* near-diagonal triangular corner (rows jj..jj+3) */
            A(jj + 0, jj + 0) -= A(jj + 0, j) * w0;

            A(jj + 1, jj + 0) -= A(jj + 1, j) * w0;
            A(jj + 1, jj + 1) -= A(jj + 1, j) * w1;

            A(jj + 2, jj + 0) -= A(jj + 2, j) * w0;
            A(jj + 2, jj + 1) -= A(jj + 2, j) * w1;
            A(jj + 2, jj + 2) -= A(jj + 2, j) * w2;

            A(jj + 3, jj + 0) -= A(jj + 3, j) * w0;
            A(jj + 3, jj + 1) -= A(jj + 3, j) * w1;
            A(jj + 3, jj + 2) -= A(jj + 3, j) * w2;
            A(jj + 3, jj + 3) -= A(jj + 3, j) * w3;

            /* rectangular tail below the corner: all 4 columns */
            for (Int i = jj + 4; i < n; ++i) {
                const VT av = A(i, j);
                A(i, jj + 0) -= av * w0;
                A(i, jj + 1) -= av * w1;
                A(i, jj + 2) -= av * w2;
                A(i, jj + 3) -= av * w3;
            }
        }

        /* remainder columns (fewer than 4 left) */
        for (; jj < n; ++jj) {
            const VT w = A(jj, j) * d;
            for (Int i = jj; i < n; ++i)
                A(i, jj) -= A(i, j) * w;
        }
    }
}

/* ------------------------------------------------------------------
 * All groups, fully general: any layout / uplo (the entry point both C adapters
 * call). A padded partial last group is processed too, which is harmless -- the
 * identity's LDL^T factor is the identity (L = I, D = I, no off-diagonal fill;
 * design section 6.5).
 *
 * Contiguous (row stride 1) when the factor column is contiguous: column-major
 * lower, or its transpose dual row-major upper -> the tuned kernel. The other
 * two combinations (column-major upper, row-major lower) have row stride ldap
 * -> the strided kernel.
 * ------------------------------------------------------------------ */

template <typename T, int V, typename Int = int>
void sytrfnp_compact_general(bool rowmajor, bool upper, Int n, T *ap, Int ldap, Int nm)
{
    assert(nm >= 1 && n >= 0);

    const Int ngroups = (nm + V - 1) / V;
    const std::size_t str_a = static_cast<std::size_t>(ldap) * n * V;

    /* A factor column is contiguous for column-major lower and its transpose
     * dual row-major upper; the tuned kernel serves both. */
    const bool contiguous = (!rowmajor && !upper) || (rowmajor && upper);

    for (Int g = 0; g < ngroups; ++g) {
        T *a = ap + g * str_a;
        if (contiguous)
            sytrfnp_compact_group<T, V, Int>(n, a, ldap);
        else
            /* strided: sweep the same factorization with row stride ldap,
             * column stride 1 -- column-major upper and row-major lower. */
            sytrfnp_compact_group_strided<T, V, Int>(n, make_view<T, V, Int>(a, ldap, 1));
    }
}

/* ------------------------------------------------------------------
 * All groups, column-major lower (convenience driver for direct C++ users of
 * this header; the C adapters route through sytrfnp_compact_general above).
 * ------------------------------------------------------------------ */

template <typename T, int V, typename Int = int>
void sytrfnp_compact(Int n, T *ap, Int ldap, Int nm)
{
    assert(ldap >= n && nm >= 1);

    const Int ngroups = (nm + V - 1) / V;
    const std::size_t str_a = static_cast<std::size_t>(ldap) * n * V;

    for (Int g = 0; g < ngroups; ++g)
        sytrfnp_compact_group<T, V, Int>(n, ap + g * str_a, ldap);
}

} /* namespace detail */
} /* namespace cqr */

#endif /* CQR_SYTRFNP_COMPACT_HPP */
