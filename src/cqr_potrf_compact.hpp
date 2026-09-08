/* cqr_potrf_compact.hpp
 *
 * Compact (interleaved-batch) Cholesky factorization of symmetric
 * positive-definite matrices, templated on scalar type T and interleave width V:
 *
 *     A = L L^T  (uplo lower),   A = U^T U  (uplo upper),
 *
 * with a positive diagonal, run V matrices at a time. A portable, vectorized
 * mkl_?potrf_compact.
 *
 * Algorithm: the unblocked right-looking LAPACK potf2, lifted double -> V-wide
 * vector, one lane per matrix. Cholesky has no data-dependent branch, so unlike
 * geqrf's larfg no lane mask is needed; the pivot is an unconditional sqrt.
 * Per pivot column j (lower):
 *
 *     d = sqrt(A(j,j));  A(j,j) = d;  invd = 1/d
 *     A(i,j) *= invd              for i > j             (scale pivot column)
 *     A(i,jj)-= A(i,j)*A(jj,j)    for jj > j, i >= jj   (rank-1 trailing update)
 *
 * Only the lower trapezoid of the *view* is touched: the kernel gets A for uplo
 * lower and A^T for upper (A symmetric, so U = L(A^T)^T lands in the upper
 * storage), and the strictly-opposite triangle passes through untouched as
 * ?potrf requires. Column-major lower and row-major upper are the contiguous
 * cases (unit row stride); the other two are strided.
 *
 * Positive-definiteness is assumed, not enforced (design section 6.2): a
 * non-SPD lane gets NaN/Inf in its factor; there is no info = j early exit.
 *
 * Compact storage (matches mkl_?gepack_compact); group g = idx/V, slot v = idx%V:
 *     A_v(i,j) = ap[ g*ldap*n*V + (j*ldap + i)*V + v ]   (column-major)
 * Row-major swaps the in-matrix roles (i*ldap + j); the group stride is the
 * same either way (A is n x n).
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#ifndef CQR_POTRF_COMPACT_HPP
#define CQR_POTRF_COMPACT_HPP

#include "cqr_compact_common.hpp"

#include <cstddef>
#include <cassert>
#include <type_traits>

namespace cqr {
namespace detail {

/* Rank-1 trailing update of the JB columns jj .. jj+JB-1 by pivot column j:
 * A(i,c) -= A(i,j) * A(c,j) for i >= c. The near-diagonal corner fills in
 * triangularly (column c touches rows i >= c); below it all JB columns take the
 * same A(i,j), loaded once. JB is compile-time so w[] stays in registers. */
template <int JB, typename T, int V, typename Int>
inline void potrf_update_block(Int j, Int n, const BatchView<T, V, Int> &A, Int jj)
{
    using VT = typename pack<T, V>::type;
    VT w[JB];
    for (int c = 0; c < JB; ++c)
        w[c] = A(jj + c, j);
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
void potrf_compact_group(Int n, BatchView<T, V, Int> A)
{
    using VT = typename pack<T, V>::type;
    static_assert(std::is_floating_point<T>::value,
                  "potrf_compact is defined for real float/double");
    assert(A.si && A.sj);

    for (Int j = 0; j < n; ++j) {
        /* pivot: d = sqrt(A(j,j)); scale the sub-diagonal column by 1/d */
        VT d;
        vsqrt<T, V>(d, A(j, j));
        A(j, j) = d;
        const VT invd = T(1) / d;
        for (Int i = j + 1; i < n; ++i)
            A(i, j) = A(i, j) * invd;

        /* rank-1 trailing update, lower trapezoid, four columns at a time */
        Int jj = j + 1;
        for (; jj + 4 <= n; jj += 4)
            potrf_update_block<4, T, V>(j, n, A, jj);
        for (; jj < n; ++jj)
            potrf_update_block<1, T, V>(j, n, A, jj);
    }
}

/* All groups, any layout / uplo. A padded partial last group is processed too,
 * harmlessly: the identity's Cholesky factor is the identity. */
template <typename T, int V, typename Int = int>
void potrf_compact(bool rowmajor, bool upper, Int n, T *ap, Int ldap, Int nm)
{
    assert(nm >= 1 && n >= 0);

    const std::size_t str_a = group_stride(rowmajor, ldap, n, n, V);

    const Int ngroups = (nm + V - 1) / V;
    const double flops = (double)n * n * n / 3.0 * V * ngroups; /* ~potf2, all lanes */
    CQR_OMP_PARALLEL_GROUPS(ngroups, flops)
    for (Int g = 0; g < ngroups; ++g) {
        /* The kernel factors the lower triangle of the view it is given: A
         * itself for uplo lower, A^T for upper (U^T U = A is L L^T of A^T = A). */
        auto A = make_view<T, V, Int>(ap + g * str_a, rowmajor, ldap);
        if (upper) A = A.transposed();
        potrf_compact_group<T, V, Int>(n, A);
    }
}

} /* namespace detail */
} /* namespace cqr */

#endif /* CQR_POTRF_COMPACT_HPP */
