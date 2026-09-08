/* cqr_ormqr_compact.hpp
 *
 * Compact (interleaved-batch) application of Householder reflectors,
 * templated on scalar type T and interleave width V:
 *
 *     C := op(Q) * C  (side='L')  or  C := C * op(Q)  (side='R'),
 *     op(Q) = Q ('N') or Q^T ('T').
 *
 * The missing mkl_?ormqr_compact, in portable form. Also home of larf(), the
 * one-reflector update that ?geqrf_compact reuses for its trailing columns.
 *
 * Applying the implicit Q:
 *   ?geqrf never forms Q. It returns Q as a product of k Householder reflectors
 *   plus the scalars tau(0..k-1):
 *
 *       Q = H(0) H(1) ... H(k-1),   H(kk) = I - tau(kk) * v(kk) * v(kk)^T,
 *
 *   with v(kk) unit-lower: an implicit 1 in position kk, the sub-diagonal
 *   entries stored in column kk of A, zeros above. Applying one reflector to a
 *   slice c of C (a column for side='L', a row for side='R') is a rank-1 update
 *   in two passes over i = kk+1 .. len-1 (this is LAPACK dorm2r / dlarf):
 *       w  = v^T c = c[kk] + sum_i A(i,kk) c[i]
 *       c := c - tau w v   =>   c[kk] -= tau w;  c[i] -= A(i,kk) tau w.
 *
 *   Each H is symmetric, so Q^T = H(k-1)..H(0), and the reflectors are applied
 *   in the order the product dictates (the LAPACK DORMQR table):
 *
 *       side   op(Q)      expanded                 sweep over kk
 *       'L'    Q  * C     H(0)..H(k-1) * C         descending (Backward)
 *       'L'    Q^T* C     H(k-1)..H(0) * C         ascending  (Forward)
 *       'R'    C * Q      C * H(0)..H(k-1)         ascending  (Forward)
 *       'R'    C * Q^T    C * H(k-1)..H(0)         descending (Backward)
 *
 *   For side='R' the dot/axpy run over the rows of C rather than its columns;
 *   the kernel sees that purely as C^T's BatchView, so the arithmetic is shared.
 *
 * Compact storage (matches mkl_?gepack_compact); group g = idx/V, slot v = idx%V:
 *     A_v(i,j)  = ap [ g*ldap*k*V   + (j*ldap + i)*V + v ]   (column-major)
 *     tau_v(kk) = taup[ g*k*V       +  kk*V          + v ]
 *     C_v(i,j)  = cp [ g*ldcp*n*V   + (j*ldcp + i)*V + v ]
 * A is the (ldap, k) reflector batch, exactly as LAPACK ?ormqr declares it, so
 * its per-matrix column extent -- hence the group stride -- is k.
 *
 * Assisted-by: Claude:claude-fable-5 Claude:claude-opus-4.8
 */

#ifndef CQR_ORMQR_COMPACT_HPP
#define CQR_ORMQR_COMPACT_HPP

#include "cqr_compact_common.hpp"

#include <cstddef>
#include <cassert>
#include <type_traits>

namespace cqr {
namespace detail {

/* Reflector sweep order: Forward applies kk = 0..k-1, Backward k-1..0. */
enum class Direction { Forward, Backward };

/* Apply H(kk) to the JB adjacent slices p0 .. p0+JB-1 of C (slices run along
 * C's second index; the reflector along its first, i = kk+1 .. len-1). JB is a
 * compile-time constant so w[] lives in registers and the c-loops unroll; every
 * reflector load A(i,kk) is then reused JB times, which is the point of the
 * blocking. */
template <int JB, typename T, int V, typename Int>
inline void larf_block(Int kk, Int len, const ConstBatchView<T, V, Int> &A,
                       const typename pack<T, V>::type &t, const BatchView<T, V, Int> &C,
                       Int p0)
{
    using VT = typename pack<T, V>::type;
    VT w[JB];
    for (int c = 0; c < JB; ++c)
        w[c] = C(kk, p0 + c);
    for (Int i = kk + 1; i < len; ++i) {
        const VT av = A(i, kk);
        for (int c = 0; c < JB; ++c)
            w[c] += av * C(i, p0 + c);
    }
    for (int c = 0; c < JB; ++c) {
        w[c] *= t; /* fold tau into w */
        C(kk, p0 + c) -= w[c];
    }
    for (Int i = kk + 1; i < len; ++i) {
        const VT av = A(i, kk);
        for (int c = 0; c < JB; ++c)
            C(i, p0 + c) -= av * w[c];
    }
}

/* Apply H(kk) (column kk of A, scalar t) to slices [p0, p1) of C, four at a
 * time with a one-at-a-time tail. */
template <typename T, int V, typename Int>
inline void larf(Int kk, Int len, const ConstBatchView<T, V, Int> &A,
                 const typename pack<T, V>::type &t, const BatchView<T, V, Int> &C,
                 Int p0, Int p1)
{
    Int p = p0;
    for (; p + 4 <= p1; p += 4)
        larf_block<4, T, V>(kk, len, A, t, C, p);
    for (; p < p1; ++p)
        larf_block<1, T, V>(kk, len, A, t, C, p);
}

/* One group: sweep the k reflectors over all `npanel` slices of C. A is the
 * (len x k) reflector view; C is viewed with the reflector axis first. */
template <typename T, int V, typename Int = int>
void ormqr_compact_group(Direction dir, Int len, Int npanel, Int k,
                         ConstBatchView<T, V, Int> A, const T *tau_,
                         BatchView<T, V, Int> C)
{
    using VT = typename pack<T, V>::type;
    static_assert(std::is_floating_point<T>::value,
                  "ormqr_compact is defined for real float/double");
    assert(k <= len && A.si && A.sj && C.si && C.sj);

    const VT *tau = reinterpret_cast<const VT *>(tau_);
    for (Int s = 0; s < k; ++s) {
        const Int kk = (dir == Direction::Forward) ? s : k - 1 - s;
        larf<T, V>(kk, len, A, tau[kk], C, Int(0), npanel);
    }
}

/* All groups: C (m x n) := op(Q) C or C op(Q), either layout. A is the
 * (len x k) reflector batch, len = m (side='L') or n (side='R'), declared
 * (ldap, k) as in LAPACK ?ormqr. trans is 'N', or 'T'/'C' for the transpose.
 * A padded partial last group is processed too, harmlessly: its identity
 * factors have tau = 0. */
template <typename T, int V, typename Int = int>
void ormqr_compact(bool left, bool rowmajor, char trans, Int m, Int n, Int k, const T *ap,
                   Int ldap, const T *taup, T *cp, Int ldcp, Int nm)
{
    assert(nm >= 1 && m >= 0 && n >= 0 && k >= 0);

    const bool tran = (trans == 'T' || trans == 't' || trans == 'C' || trans == 'c');
    const Direction dir = (left == tran) ? Direction::Forward : Direction::Backward;
    const Int len = left ? m : n;
    const Int npanel = left ? n : m;

    const std::size_t str_a = group_stride(rowmajor, ldap, len, k, V);
    const std::size_t str_t = (std::size_t)k * V;
    const std::size_t str_c = group_stride(rowmajor, ldcp, m, n, V);

    for_each_group<V>(nm, 4.0 * k * len * npanel * V /* ~orm2r */, [&](Int g) {
        /* A is swept down its columns. For side='R' the reflectors act on the
         * rows of C, so the kernel is handed C^T. */
        auto C = make_view<T, V, Int>(cp + g * str_c, rowmajor, ldcp);
        if (!left) C = C.transposed();
        ormqr_compact_group<T, V, Int>(
            dir, len, npanel, k,
            make_const_view<T, V, Int>(ap + g * str_a, rowmajor, ldap), taup + g * str_t,
            C);
    });
}

} /* namespace detail */
} /* namespace cqr */

#endif /* CQR_ORMQR_COMPACT_HPP */
