/* cqr_sytrfnp_compact.hpp
 *
 * Compact (interleaved-batch) LDL^T factorization of symmetric matrices,
 * without pivoting ("np"), templated on scalar type T and interleave width V:
 *
 *     A = L D L^T  (uplo lower),   A = U^T D U  (uplo upper),
 *
 * with L (U) unit lower (upper) triangular and D diagonal, run V matrices at a
 * time. On exit D sits on the diagonal and the strict off-diagonal of the named
 * triangle holds L (U); the unit diagonal is implied, not stored. MKL ships no
 * compact sytrf of any kind; the "np" suffix follows its unpivoted compact LU,
 * mkl_?getrfnp_compact.
 *
 * The square-root-free sibling of cqr_potrf_compact.hpp: symmetric *indefinite*
 * matrices factor too (negative pivots are fine -- there is no sqrt), which is
 * the point of LDL^T over Cholesky. cqr_sytrsnp_compact.hpp solves from the
 * factor and cqr_sysvnp_compact.hpp fuses the two per group.
 *
 * Algorithm: the unblocked right-looking sweep, potf2 with the sqrt pivot
 * replaced by a reciprocal, lifted double -> V-wide vector, one lane per matrix.
 * Like Cholesky the math has no data-dependent branch, so no lane mask is
 * needed. Per pivot column j (lower):
 *
 *     d = A(j,j);  invd = 1/d                      (pivot stays as D(j,j))
 *     A(i,j) *= invd                for i > j            (scale -> L(i,j))
 *     A(i,jj)-= A(i,j)*(A(jj,j)*d)  for jj > j, i >= jj  (rank-1 trailing update:
 *                                                         L(i,j) D(j,j) L(jj,j))
 *
 * Only the lower trapezoid of the *view* is touched: the kernel gets A for uplo
 * lower and A^T for upper (A symmetric, so the lower factorization of A^T lands
 * U = L(A^T)^T in the upper storage with the same D), and the strictly-opposite
 * triangle passes through untouched. Note the upper convention is therefore the
 * transpose dual A = U^T D U (matching potrf's A = U^T U), NOT LAPACK ?sytrf's
 * A = U D U^T (design section 6.3). Column-major lower and row-major upper are
 * the contiguous cases (unit row stride); the other two are strided.
 *
 * Pivots are the *updated* (Schur-complement) diagonal entries, not the input
 * ones: a zero on the input diagonal is harmless unless a pivot itself is zero.
 * A zero pivot -- a singular leading principal minor -- gets Inf/NaN in its
 * lane's factor; there is no info = j early exit and no pivoting (both are
 * per-lane branching that does not vectorize; design section 6.2).
 *
 * Compact storage (matches mkl_?gepack_compact); group g = idx/V, slot v = idx%V:
 *     A_v(i,j) = ap[ g*ldap*n*V + (j*ldap + i)*V + v ]   (column-major)
 * Row-major swaps the in-matrix roles (i*ldap + j); the group stride is the
 * same either way (A is n x n).
 *
 * Assisted-by: Claude
 */

#ifndef CQR_SYTRFNP_COMPACT_HPP
#define CQR_SYTRFNP_COMPACT_HPP

#include "cqr_compact_common.hpp"

#include <cstddef>
#include <cassert>
#include <type_traits>

namespace cqr::detail {

/* Symmetric rank-1 trailing update of the JB columns jj .. jj+JB-1 by pivot
 * column j: A(i,c) -= L(i,j) * D(j,j) * L(c,j) for i >= c, with the pivot
 * column already scaled to L and the pivot d = D(j,j) (still in place at
 * A(j,j); re-read here rather than passed, so no pack crosses a call boundary
 * by reference -- see the alignment note in cqr_compact_common.hpp) folded into
 * the per-column weight w[c] = L(c,j) * d, one extra multiply per block column.
 * The near-diagonal corner fills in triangularly (column c touches rows
 * i >= c); below it all JB columns take the same A(i,j), loaded once. JB is
 * compile-time so w[] stays in registers. */
template <int JB, typename T, int V, typename Int>
inline void sytrfnp_update_block(Int j, Int n, const BatchView<T, V, Int> &A, Int jj)
{
    using VT = typename pack<T, V>::type;
    const VT d = A(j, j);
    VT w[JB];
    for (int c = 0; c < JB; ++c)
        w[c] = A(jj + c, j) * d;
    for (int c = 0; c < JB; ++c)
        for (int r = c; r < JB; ++r)
            A(jj + r, jj + c) -= A(jj + r, j) * w[c];
    for (Int i = jj + JB; i < n; ++i) {
        const VT av = A(i, j);
        for (int c = 0; c < JB; ++c)
            A(i, jj + c) -= av * w[c];
    }
}

/* One group of V interleaved n x n matrices: factor the lower triangle of the
 * view (see the file header for how the view maps uplo/layout onto it). */
template <typename T, int V, typename Int = int>
void sytrfnp_compact_group(Int n, BatchView<T, V, Int> A)
{
    using VT = typename pack<T, V>::type;
    static_assert(std::is_floating_point_v<T>,
                  "sytrfnp_compact is defined for real float/double");
    assert(A.si && A.sj);

    for (Int j = 0; j < n; ++j) {
        /* pivot: d = A(j,j) stays in place as D(j,j); scale the sub-diagonal
         * column by 1/d so it holds L(:,j) */
        const VT d = A(j, j);
        const VT invd = T(1) / d;
        for (Int i = j + 1; i < n; ++i)
            A(i, j) = A(i, j) * invd;

        /* rank-1 trailing update, lower trapezoid, four columns at a time */
        Int jj = j + 1;
        for (; jj + 4 <= n; jj += 4)
            sytrfnp_update_block<4, T, V>(j, n, A, jj);
        for (; jj < n; ++jj)
            sytrfnp_update_block<1, T, V>(j, n, A, jj);
    }
}

/* The view whose lower triangle the kernel factors so that the factor lands in
 * the named triangle of the group at `a`: A itself for uplo lower, A^T for
 * upper (U^T D U = A is L D L^T of A^T = A). Shared with the fused sysvnp. */
template <typename T, int V, typename Int = int>
inline BatchView<T, V, Int> sytrfnp_view(bool rowmajor, bool upper, T *a, Int ldap)
{
    auto A = make_view<T, V, Int>(a, rowmajor, ldap);
    return upper ? A.transposed() : A;
}

/* ~flops of the unblocked LDL^T of one group: n^3/3 per matrix, times V. */
template <typename Int> inline double sytrfnp_flops(Int n, int V)
{
    return (double)n * n * n / 3.0 * V;
}

/* All groups, any layout / uplo. A padded partial last group is processed too,
 * harmlessly: the identity's LDL^T factor is L = I, D = I (no fill). */
template <typename T, int V, typename Int = int>
void sytrfnp_compact(bool rowmajor, bool upper, Int n, T *ap, Int ldap, Int nm)
{
    assert(nm >= 1 && n >= 0);

    const std::size_t str_a = group_stride(rowmajor, ldap, n, n, V);

    for_each_group<V>(
        nm,
        [&](Int g) {
            sytrfnp_compact_group<T, V, Int>(
                n, sytrfnp_view<T, V, Int>(rowmajor, upper, ap + g * str_a, ldap));
        },
        sytrfnp_flops(n, V));
}

} /* namespace cqr::detail */

#endif /* CQR_SYTRFNP_COMPACT_HPP */
