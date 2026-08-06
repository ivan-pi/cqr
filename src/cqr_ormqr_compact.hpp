/* cqr_ormqr_compact.hpp
 *
 * Compact (interleaved-batch) application of Householder reflectors,
 * templated on scalar type T and interleave width V:
 *
 *     C := op(Q) * C  (side='L')  or  C := C * op(Q)  (side='R'),
 *     op(Q) = Q ('N') or Q^T ('T').
 *
 * ormqr_compact()         -- tuned side='L', column-major path.
 * ormqr_compact_general() -- side in {L,R}, column- or row-major.
 *
 * The missing mkl_?ormqr_compact, in portable form.
 *
 * Design:
 *   - The pack<T,V> element and the BatchView addressing machinery live in the
 *     shared cqr_compact_common.hpp (V is the compact-format interleave width;
 *     see there for the width/hardware mapping).
 *   - The algorithm is unblocked dorm2r: pivot-free and branch-free, so
 *     the scalar code lifts verbatim with double -> V-wide vector.
 *   - RHS columns are register-blocked (JB=4) so each reflector load
 *     ak[i] is reused across JB columns, halving the dominant A traffic.
 *
 * Compact storage convention (matches MKL Compact / mkl_?gepack_compact):
 *   group g = matrix index / V, slot v = matrix index % V:
 *     A_v(i,j)  = ap [ g*ldap*k*V     + (j*ldap + i)*V + v ]
 *     tau_v(kk) = taup[ g*k*V           +  kk*V          + v ]
 *     B_v(i,j)  = bp [ g*ldbp*nrhs*V   + (j*ldbp + i)*V + v ]
 * (A is the (ldap, k) reflector batch, exactly as LAPACK ?ormqr declares it,
 * so its per-matrix column extent -- and hence the group stride -- is k.)
 *
 * Applying the implicit Q (the heart of this file):
 *   ?geqrf_compact never forms Q. It returns Q as a product of k elementary
 *   Householder reflectors together with the scalars tau(0..k-1):
 *
 *       Q = H(0) H(1) ... H(k-1),   H(kk) = I - tau(kk) * v(kk) * v(kk)^T.
 *
 *   Each reflector vector v(kk) is unit-lower -- an implicit 1 in position kk,
 *   sub-diagonal entries stored in column kk of A, zeros above:
 *       v(kk)[kk] = 1            (implicit, never read),
 *       v(kk)[i]  = A(i,kk),     i = kk+1 .. spec_len-1,   <-- the ak[i] below
 *       v(kk)[i]  = 0,           i < kk.
 *
 *   So we never touch a dense Q. Applying one reflector to a single panel slice
 *   c (a column of B for side='L', a row for side='R') is a rank-1 update, two
 *   passes over the tail i = kk+1 .. spec_len-1 (this is dorm2r):
 *       w  = v(kk)^T c = c[kk] + sum_i ak[i]*c[i]     (dot, exploiting v[kk]=1)
 *       c := c - tau(kk) * w * v(kk)                  (axpy back)
 *          => c[kk] -= tau*w;   c[i] -= tau*w*ak[i].
 *
 *   Order / transpose. Each H is symmetric (H^T = H), so Q^T = H(k-1)..H(0),
 *   and the reflectors must be applied in the order the matrix product dictates
 *   (this is the LAPACK DORMQR/DORM2R table):
 *
 *       side   op(Q)      expanded                 sweep over kk
 *       'L'    Q  * C     H(0)..H(k-1) * C         descending (Backward)
 *       'L'    Q^T* C     H(k-1)..H(0) * C         ascending  (Forward)
 *       'R'    C * Q      C * H(0)..H(k-1)         ascending  (Forward)
 *       'R'    C * Q^T    C * H(k-1)..H(0)         descending (Backward)
 *
 *   i.e. fwd = (side=='L') ? trans : !trans, as the entry points compute. For
 *   side='R' the dot/axpy run over the rows of C rather than its columns; the
 *   strided kernel expresses that solely by swapping which stride is the
 *   "special" (reflector) direction and which is the "panel" direction, so the
 *   arithmetic is shared with side='L'.
 *
 * Assisted-by: Claude:claude-fable-5 Claude:claude-opus-4.8
 */

#ifndef CQR_ORMQR_COMPACT_HPP
#define CQR_ORMQR_COMPACT_HPP

#include "cqr_compact_common.hpp" /* pack<T,V>, BatchView, make_view, make_const_view */

#include <cstddef>
#include <cassert>
#include <type_traits>

namespace cqr {
namespace detail {

/* ------------------------------------------------------------------ */
/* Reflector sweep direction (internal control flag).                  */
/*                                                                     */
/* Forward applies the reflectors in ascending order kk = 0..k-1 (the  */
/* Q^T-on-the-left case); Backward applies them descending (Q). This   */
/* replaces the former char/bool split between the two group kernels   */
/* with a single explicit type shared by both. The outer templated     */
/* entry points still take the LAPACK 'T'/'N' char and translate here. */
/* ------------------------------------------------------------------ */

enum class Direction { Forward, Backward };

/* ------------------------------------------------------------------ */
/* One group of V interleaved matrices                                 */
/* ------------------------------------------------------------------ */

template <typename T, int V, typename Int = int>
void ormqr_compact_group(Direction dir, Int m, Int nrhs, Int k, const T *a_, Int ldap,
                         const T *tau_, T *b_, Int ldbp)
{
    using VT = typename pack<T, V>::type;
    static_assert(std::is_floating_point<T>::value,
                  "ormqr_compact is defined for real float/double");

    assert(k <= m && ldap >= m && ldbp >= m);

    const VT *A = reinterpret_cast<const VT *>(a_);
    const VT *tau = reinterpret_cast<const VT *>(tau_);
    VT *B = reinterpret_cast<VT *>(b_);

    const bool fwd = (dir == Direction::Forward);

    for (Int s = 0; s < k; ++s) {
        const Int kk = fwd ? s : k - 1 - s; /* Q^T: ascending, Q: descending */
        const VT *ak = A + static_cast<std::size_t>(kk) * ldap;
        const VT t = tau[kk];

        Int j = 0;

        /* main loop: 4 RHS columns at a time; ak[i] loaded once, used 4x */
        for (; j + 4 <= nrhs; j += 4) {
            VT *b0 = B + static_cast<std::size_t>(j + 0) * ldbp;
            VT *b1 = B + static_cast<std::size_t>(j + 1) * ldbp;
            VT *b2 = B + static_cast<std::size_t>(j + 2) * ldbp;
            VT *b3 = B + static_cast<std::size_t>(j + 3) * ldbp;

            VT w0 = b0[kk], w1 = b1[kk], w2 = b2[kk], w3 = b3[kk];
            for (Int i = kk + 1; i < m; ++i) {
                const VT av = ak[i];
                w0 += av * b0[i];
                w1 += av * b1[i];
                w2 += av * b2[i];
                w3 += av * b3[i];
            }
            b0[kk] -= t * w0;
            b1[kk] -= t * w1;
            b2[kk] -= t * w2;
            b3[kk] -= t * w3;

            w0 *= t;
            w1 *= t;
            w2 *= t;
            w3 *= t; /* fold tau into w */
            for (Int i = kk + 1; i < m; ++i) {
                const VT av = ak[i];
                b0[i] -= av * w0;
                b1[i] -= av * w1;
                b2[i] -= av * w2;
                b3[i] -= av * w3;
            }
        }

        /* remainder columns */
        for (; j < nrhs; ++j) {
            VT *bj = B + static_cast<std::size_t>(j) * ldbp;
            VT w = bj[kk];
            for (Int i = kk + 1; i < m; ++i)
                w += ak[i] * bj[i];
            bj[kk] -= t * w;
            w *= t;
            for (Int i = kk + 1; i < m; ++i)
                bj[i] -= ak[i] * w;
        }
    }
}

/* ------------------------------------------------------------------ */
/* One group, fully general: any side / layout via BatchView strides.  */
/*                                                                     */
/* The unblocked dorm2r math is identical to the left/col-major kernel */
/* above; only the addressing changes, and that now lives entirely in  */
/* the two BatchViews. For reflector kk the essential Householder      */
/* vector is column kk of A (implicit 1 at the diagonal):              */
/*   A(i,kk),  i in (kk, spec_len).                                    */
/* C is swept along its special axis (rows for side='L', columns for   */
/* side='R') and register-blocked 4 panel slices at a time.            */
/* ------------------------------------------------------------------ */

template <typename T, int V, typename Int = int>
void ormqr_compact_group_strided(Direction dir, Int spec_len, Int panel_cnt, Int k,
                                 BatchView<const typename pack<T, V>::type, Int> A,
                                 const T *tau_,
                                 BatchView<typename pack<T, V>::type, Int> C)
{
    using VT = typename pack<T, V>::type;
    static_assert(std::is_floating_point<T>::value,
                  "ormqr_compact is defined for real float/double");

    /* k reflectors live along the special axis; strides must be non-degenerate
     * so distinct (i,p) map to distinct elements. */
    assert(k <= spec_len);
    assert(A.special && A.panel && C.special && C.panel);

    const VT *tau = reinterpret_cast<const VT *>(tau_);
    const bool fwd = (dir == Direction::Forward);

    for (Int s = 0; s < k; ++s) {
        const Int kk = fwd ? s : k - 1 - s; /* Q^T: ascending, Q: descending */
        const VT t = tau[kk];

        Int p = 0;

        /* main loop: 4 panel slices at a time; A(i,kk) loaded once, used 4x.
         * Both index*stride products are evaluated in 64-bit inside operator(),
         * and the panel offset p*stride is loop-invariant across i, so the
         * codegen matches the hand-strided version. */
        for (; p + 4 <= panel_cnt; p += 4) {
            VT w0 = C(kk, p + 0), w1 = C(kk, p + 1), w2 = C(kk, p + 2), w3 = C(kk, p + 3);
            for (Int i = kk + 1; i < spec_len; ++i) {
                const VT av = A(i, kk);
                w0 += av * C(i, p + 0);
                w1 += av * C(i, p + 1);
                w2 += av * C(i, p + 2);
                w3 += av * C(i, p + 3);
            }
            C(kk, p + 0) -= t * w0;
            C(kk, p + 1) -= t * w1;
            C(kk, p + 2) -= t * w2;
            C(kk, p + 3) -= t * w3;

            w0 *= t;
            w1 *= t;
            w2 *= t;
            w3 *= t; /* fold tau into w */
            for (Int i = kk + 1; i < spec_len; ++i) {
                const VT av = A(i, kk);
                C(i, p + 0) -= av * w0;
                C(i, p + 1) -= av * w1;
                C(i, p + 2) -= av * w2;
                C(i, p + 3) -= av * w3;
            }
        }

        /* remainder panel slices */
        for (; p < panel_cnt; ++p) {
            VT w = C(kk, p);
            for (Int i = kk + 1; i < spec_len; ++i)
                w += A(i, kk) * C(i, p);
            C(kk, p) -= t * w;
            w *= t;
            for (Int i = kk + 1; i < spec_len; ++i)
                C(i, p) -= A(i, kk) * w;
        }
    }
}

/* ------------------------------------------------------------------ */
/* All groups: nm matrices total (a padded partial last group is      */
/* processed too, which is harmless)                                  */
/* ------------------------------------------------------------------ */

template <typename T, int V, typename Int = int>
void ormqr_compact(char trans, Int m, Int nrhs, Int k, const T *ap, Int ldap,
                   const T *taup, T *bp, Int ldbp, Int nm)
{
    assert(trans == 'T' || trans == 't' || trans == 'N' || trans == 'n');
    assert(ldap >= m && ldbp >= m && k <= m && nm >= 1);

    const Direction dir =
        (trans == 'T' || trans == 't') ? Direction::Forward : Direction::Backward;
    const Int ngroups = (nm + V - 1) / V;
    /* A is (ldap, k): the per-matrix column extent is k, so the group stride
     * is ldap*k*V (matches LAPACK ?ormqr's A(LDA,K) declaration). */
    const std::size_t str_a = static_cast<std::size_t>(ldap) * k * V;
    const std::size_t str_t = static_cast<std::size_t>(k) * V;
    const std::size_t str_b = static_cast<std::size_t>(ldbp) * nrhs * V;

    for (Int g = 0; g < ngroups; ++g)
        ormqr_compact_group<T, V, Int>(dir, m, nrhs, k, ap + g * str_a, ldap,
                                       taup + g * str_t, bp + g * str_b, ldbp);
}

/* ------------------------------------------------------------------ */
/* All groups, fully general: side = 'L'/'R', column- or row-major.    */
/*                                                                     */
/* C is the m x n batch; A is the (spec_len x k) reflector batch with  */
/* spec_len = m (side='L') or n (side='R'), declared (ldap, k) exactly  */
/* as LAPACK ?ormqr, so its per-matrix column extent is k. The left /  */
/* column-major case routes to the tuned contiguous kernel above; the  */
/* other three combinations use the strided kernel.                    */
/* ------------------------------------------------------------------ */

template <typename T, int V, typename Int = int>
void ormqr_compact_general(bool left, bool rowmajor, char trans, Int m, Int n, Int k,
                           const T *ap, Int ldap, const T *taup, T *cp, Int ldcp, Int nm)
{
    assert(trans == 'T' || trans == 't' || trans == 'N' || trans == 'n' || trans == 'C' ||
           trans == 'c');
    assert(nm >= 1);

    const bool tran = (trans == 'T' || trans == 't' || trans == 'C' || trans == 'c');
    /* dorm2r ordering: side='L' applies ascending for Q^T, side='R' flips. */
    const bool fwd = left ? tran : !tran;
    const Direction dir = fwd ? Direction::Forward : Direction::Backward;
    const Int spec_len = left ? m : n;
    const Int panel_cnt = left ? n : m;

    /* Q has order spec_len, so there cannot be more reflectors than that. */
    assert(m >= 0 && n >= 0 && k >= 0 && k <= spec_len);

    /* element strides (in VT units). A is swept down its rows (the reflector
     * axis) with kk along its columns; for C the special axis is rows when
     * side='L' and columns when side='R'. Column-major: a row step is 1 and a
     * column step is ld; row-major flips that. */
    const std::size_t c_row = rowmajor ? (std::size_t)ldcp : 1;
    const std::size_t c_col = rowmajor ? 1 : (std::size_t)ldcp;
    const std::size_t a_special = rowmajor ? (std::size_t)ldap : 1;
    const std::size_t a_panel = rowmajor ? 1 : (std::size_t)ldap;
    const std::size_t c_special = left ? c_row : c_col;
    const std::size_t c_panel = left ? c_col : c_row;

    /* group strides (in scalar T units): elements packed per matrix is
     * ldap*(complementary extent) -- the column count k for col-major (A is
     * (ldap, k)), the row count spec_len for row-major. */
    const std::size_t str_a = (rowmajor ? (size_t)ldap * spec_len : (size_t)ldap * k) * V;
    const std::size_t str_t = (size_t)k * V;
    const std::size_t str_c = (rowmajor ? (size_t)ldcp * m : (size_t)ldcp * n) * V;

    const Int ngroups = (nm + V - 1) / V;
    for (Int g = 0; g < ngroups; ++g) {
        const T *a = ap + g * str_a;
        const T *tg = taup + g * str_t;
        T *c = cp + g * str_c;
        if (left && !rowmajor)
            ormqr_compact_group<T, V, Int>(dir, m, n, k, a, ldap, tg, c, ldcp);
        else
            ormqr_compact_group_strided<T, V, Int>(
                dir, spec_len, panel_cnt, k,
                make_const_view<T, V, Int>(a, a_special, a_panel), tg,
                make_view<T, V, Int>(c, c_special, c_panel));
    }
}

} /* namespace detail */
} /* namespace cqr */

#endif /* CQR_ORMQR_COMPACT_HPP */
