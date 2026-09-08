/* cqr_trsm_compact.hpp
 *
 * Compact (interleaved-batch) triangular solve with multiple right-hand sides,
 * templated on scalar type T and interleave width V:
 *
 *     op(A) X = alpha B   (side='L')   or   X op(A) = alpha B   (side='R'),
 *     op(A) = A ('N') or A^T ('T'),
 *
 * A is upper/lower, unit/non-unit triangular; B is overwritten by X. The compact
 * analogue of BLAS ?trsm and a portable alternative to mkl_?trsm_compact -- the
 * solve that closes the batched QR:
 *
 *     cqr_mkl_dgeqrf_compact(A -> H, tau);          // A = Q R
 *     cqr_mkl_dormqr_compact('L','T', H, tau, B);   // B := Q^T B
 *     cqr_mkl_dtrsm_compact ('L','U','N','N', R, B) // B := R^{-1} Q^T B = X
 *
 * Algorithm: the scalar BLAS ?trsm substitution run V matrices at a time. Compact
 * format stores element (i,j) of all V contiguously, so it lifts verbatim with
 * double -> V-wide vector, one lane per matrix (no data-dependent branch). The
 * tuned side='L', column-major path is templated on the RHS block width and on
 * uplo/trans/diag; the other side/layout combinations go through one strided
 * kernel over BatchViews (docs/cqr_mkl_dtrsm_compact_design.md has the details).
 *
 * Compact storage (matches mkl_?gepack_compact); group g = idx/V, slot v = idx%V,
 * A the order-s (s = m left / n right) triangular batch, B the m x n batch:
 *     A_v(i,j) = ap[g*ldap*s*V + (j*ldap+i)*V + v]   (column-major)
 *     B_v(i,j) = bp[g*ldbp*n*V + (j*ldbp+i)*V + v]
 * Row-major swaps the in-matrix roles to i*ld + j.
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#ifndef CQR_TRSM_COMPACT_HPP
#define CQR_TRSM_COMPACT_HPP

#include "cqr_compact_common.hpp"

#include <cstddef>
#include <cassert>
#include <type_traits>

namespace cqr {
namespace detail {

/* Single-column column-oriented (gaxpy) solve, side='L', column-major, op(A)=A:
 * reads A down columns (contiguous), for the 1-column tail where a strided
 * single-column row-dot would prefetch poorly. bj is the column, scaled first. */
template <bool UPPER, bool UNIT, typename T, int V, typename Int>
inline void trsm_axpy_col(Int m, const typename pack<T, V>::type *A, Int ldap,
                          typename pack<T, V>::type *bj,
                          const typename pack<T, V>::type &va)
{
    using VT = typename pack<T, V>::type;
    for (Int i = 0; i < m; ++i)
        bj[i] = bj[i] * va;
    for (Int t = 0; t < m; ++t) {
        const Int kk = UPPER ? m - 1 - t : t;
        const VT *ak = A + kk * ldap;
        const VT xk = UNIT ? bj[kk] : bj[kk] / ak[kk];
        if (!UNIT) bj[kk] = xk;
        const Int lo = UPPER ? 0 : kk + 1;
        const Int hi = UPPER ? kk : m;
        for (Int i = lo; i < hi; ++i)
            bj[i] -= ak[i] * xk;
    }
}

/* Row-dot solve of a fixed JB-column block, side='L', column-major. JB and the
 * uplo/trans/diag config are compile-time, so the JB accumulators land in
 * registers and the c-loops unroll. B points at the first of the JB adjacent RHS
 * columns (column c at B + c*ldbp); alpha is folded into the first load. */
template <int JB, bool UPPER, bool TRAN, bool UNIT, typename T, int V, typename Int>
inline void trsm_dot_block(Int m, const typename pack<T, V>::type *A, Int ldap,
                           typename pack<T, V>::type *B, Int ldbp,
                           const typename pack<T, V>::type &va)
{
    using VT = typename pack<T, V>::type;
    /* back-substitute (sweep rows high -> low) when op(A) is upper-triangular:
     * A upper & no-trans, or A lower & trans (its transpose is upper). */
    constexpr bool back = (UPPER != TRAN);
    for (Int t = 0; t < m; ++t) {
        const Int i = back ? m - 1 - t : t;
        VT w[JB];
        for (int c = 0; c < JB; ++c)
            w[c] = B[c * ldbp + i] * va;
        const Int lo = back ? i + 1 : 0;
        const Int hi = back ? m : i;
        for (Int l = lo; l < hi; ++l) {
            /* TRAN reads A(l,i) = A[i*ldap+l] (down column i, contiguous);
             * !TRAN reads A(i,l) = A[l*ldap+i] (across row i, stride ldap). */
            const VT av = TRAN ? A[i * ldap + l] : A[l * ldap + i];
            for (int c = 0; c < JB; ++c)
                w[c] -= av * B[c * ldbp + l];
        }
        if (!UNIT) {
            const VT d = A[i * ldap + i];
            for (int c = 0; c < JB; ++c)
                w[c] = w[c] / d;
        }
        for (int c = 0; c < JB; ++c)
            B[c * ldbp + i] = w[c];
    }
}

/* One group, side='L', column-major, fully specialized on uplo/trans/diag:
 * the RHS columns are swept in 4/2/1 blocks so the 1-3 leftover columns still
 * reuse each A load (2- and 1-wide tails), instead of a one-column-at-a-time
 * remainder. The driver handles alpha = 0 (B := 0); alpha is nonzero here. */
template <bool UPPER, bool TRAN, bool UNIT, typename T, int V, typename Int>
void trsm_left_dot_tb(Int m, Int n, T alpha, const T *a_, Int ldap, T *b_, Int ldbp)
{
    using VT = typename pack<T, V>::type;
    static_assert(std::is_floating_point<T>::value,
                  "trsm_compact is defined for real float/double");
    assert(ldap >= m && ldbp >= m);

    const VT *A = reinterpret_cast<const VT *>(a_);
    VT *B = reinterpret_cast<VT *>(b_);

    VT va;
    broadcast<T, V>(va, alpha);
    Int j = 0;
    for (; j + 4 <= n; j += 4)
        trsm_dot_block<4, UPPER, TRAN, UNIT, T, V, Int>(m, A, ldap, B + j * ldbp, ldbp,
                                                        va);
    if (n - j >= 2) {
        trsm_dot_block<2, UPPER, TRAN, UNIT, T, V, Int>(m, A, ldap, B + j * ldbp, ldbp,
                                                        va);
        j += 2;
    }
    if (n - j >= 1) {
        /* The single leftover column: for op(A)=A the row-dot would stream A
         * strided with no reuse, so use the contiguous column-axpy instead;
         * op(A)=A^T reads A down a column already, so the dot 1-block is fine. */
        if constexpr (!TRAN)
            trsm_axpy_col<UPPER, UNIT, T, V, Int>(m, A, ldap, B + j * ldbp, va);
        else
            trsm_dot_block<1, UPPER, TRAN, UNIT, T, V, Int>(m, A, ldap, B + j * ldbp,
                                                            ldbp, va);
    }
}

/* Runtime (uplo, trans, diag) -> the compile-time-specialized driver. */
template <bool UPPER, bool TRAN, typename T, int V, typename Int>
inline void trsm_left_dot_u(bool unit, Int m, Int n, T alpha, const T *a, Int ldap, T *b,
                            Int ldbp)
{
    if (unit)
        trsm_left_dot_tb<UPPER, TRAN, true, T, V, Int>(m, n, alpha, a, ldap, b, ldbp);
    else
        trsm_left_dot_tb<UPPER, TRAN, false, T, V, Int>(m, n, alpha, a, ldap, b, ldbp);
}
template <bool UPPER, typename T, int V, typename Int>
inline void trsm_left_dot_t(bool tran, bool unit, Int m, Int n, T alpha, const T *a,
                            Int ldap, T *b, Int ldbp)
{
    if (tran)
        trsm_left_dot_u<UPPER, true, T, V, Int>(unit, m, n, alpha, a, ldap, b, ldbp);
    else
        trsm_left_dot_u<UPPER, false, T, V, Int>(unit, m, n, alpha, a, ldap, b, ldbp);
}
template <typename T, int V, typename Int>
inline void trsm_left_dot(bool upper, bool tran, bool unit, Int m, Int n, T alpha,
                          const T *a, Int ldap, T *b, Int ldbp)
{
    if (upper)
        trsm_left_dot_t<true, T, V, Int>(tran, unit, m, n, alpha, a, ldap, b, ldbp);
    else
        trsm_left_dot_t<false, T, V, Int>(tran, unit, m, n, alpha, a, ldap, b, ldbp);
}

/* One group, fully general: any side / layout via BatchView strides. Same
 * substitution as the tuned path; only the addressing changes (it lives in the
 * two BatchViews). side='L' sweeps a row of X at a time, side='R' a column. */
template <typename T, int V, typename Int = int>
void trsm_compact_group_strided(bool left, bool upper, bool tran, bool unit, Int m, Int n,
                                T alpha, ConstBatchView<T, V, Int> A,
                                BatchView<T, V, Int> B)
{
    using VT = typename pack<T, V>::type;
    static_assert(std::is_floating_point<T>::value,
                  "trsm_compact is defined for real float/double");

    assert(A.si && A.sj && B.si && B.sj);

    VT va;
    broadcast<T, V>(va, alpha);

    if (left) {
        /* solve op(A) X = alpha B column by column; A is m x m */
        const bool back = (upper != tran);
        for (Int j = 0; j < n; ++j)
            for (Int t = 0; t < m; ++t) {
                const Int i = back ? m - 1 - t : t;
                VT w = B(i, j) * va;
                const Int lo = back ? i + 1 : 0;
                const Int hi = back ? m : i;
                for (Int l = lo; l < hi; ++l)
                    w -= (tran ? A(l, i) : A(i, l)) * B(l, j);
                B(i, j) = unit ? w : w / A(i, i);
            }
    }
    else {
        /* solve X op(A) = alpha B, one column of X at a time; A is n x n */
        const bool fwd = (upper != tran);
        for (Int t = 0; t < n; ++t) {
            const Int j = fwd ? t : n - 1 - t;
            for (Int i = 0; i < m; ++i)
                B(i, j) = B(i, j) * va; /* scale this column of X */
            const Int lo = fwd ? 0 : j + 1;
            const Int hi = fwd ? j : n;
            for (Int l = lo; l < hi; ++l) {
                const VT c = tran ? A(j, l) : A(l, j);
                for (Int i = 0; i < m; ++i)
                    B(i, j) -= c * B(i, l);
            }
            if (!unit) {
                const VT d = A(j, j);
                for (Int i = 0; i < m; ++i)
                    B(i, j) = B(i, j) / d;
            }
        }
    }
}

/* All groups (nm matrices). A padded partial last group is processed too, which
 * is harmless: padded slots are identity triangular factors (unit diagonal), so
 * the diagonal divide never hits zero and their X = alpha B is never read back.
 * side='L' column-major routes to the tuned trsm_left_dot; the other three
 * side/layout combinations use the strided kernel. */
template <typename T, int V, typename Int = int>
void trsm_compact(bool left, bool upper, bool rowmajor, bool tran, bool unit, Int m,
                  Int n, T alpha, const T *ap, Int ldap, T *bp, Int ldbp, Int nm)
{
    assert(nm >= 1 && m >= 0 && n >= 0);

    /* A is the order-s triangular factor: s = m (left) or n (right). */
    const Int s = left ? m : n;

    /* A is s x s, B is m x n. */
    const std::size_t str_a = group_stride(rowmajor, ldap, s, s, V);
    const std::size_t str_b = group_stride(rowmajor, ldbp, m, n, V);

    const Int ngroups = (nm + V - 1) / V;
    /* ~substitution, all lanes: s^2 times the other extent; alpha = 0 is a store */
    const double flops = (double)s * s * (left ? n : m) * V * ngroups;

    /* alpha == 0 is the BLAS ?trsm fast path: B := 0 with A untouched. Handle it
     * once here -- both group kernels then assume alpha != 0 -- zeroing each
     * group's m x n block through the same strided B view the solve uses. */
    if (alpha == T(0)) {
        using VT = typename pack<T, V>::type;
        CQR_OMP_PARALLEL_GROUPS(ngroups, (double)m * n * V * ngroups)
        for (Int g = 0; g < ngroups; ++g) {
            auto B = make_view<T, V, Int>(bp + (std::size_t)g * str_b, rowmajor, ldbp);
            for (Int j = 0; j < n; ++j)
                for (Int i = 0; i < m; ++i)
                    B(i, j) = VT{};
        }
        return;
    }

    CQR_OMP_PARALLEL_GROUPS(ngroups, flops)
    for (Int g = 0; g < ngroups; ++g) {
        const T *a = ap + (std::size_t)g * str_a;
        T *b = bp + (std::size_t)g * str_b;
        if (left && !rowmajor)
            trsm_left_dot<T, V, Int>(upper, tran, unit, m, n, alpha, a, ldap, b, ldbp);
        else
            trsm_compact_group_strided<T, V, Int>(
                left, upper, tran, unit, m, n, alpha,
                make_const_view<T, V, Int>(a, rowmajor, ldap),
                make_view<T, V, Int>(b, rowmajor, ldbp));
    }
}

} /* namespace detail */
} /* namespace cqr */

#endif /* CQR_TRSM_COMPACT_HPP */
