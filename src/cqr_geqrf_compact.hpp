/* cqr_geqrf_compact.hpp
 *
 * Compact (interleaved-batch) QR factorization, templated on scalar type T and
 * interleave width V:
 *
 *     A = Q R,   Q = H(0) H(1) ... H(k-1),   k = min(m, n),
 *     H(kk) = I - tau(kk) * v(kk) * v(kk)^T.
 *
 * geqrf_compact()         -- tuned column-major path.
 * geqrf_compact_general() -- column- or row-major.
 *
 * A portable, vectorized mkl_?geqrf_compact. It is the factorization companion
 * to cqr_compact.hpp's ormqr_compact (which applies the reflectors produced
 * here), and reuses the same pack<T,V> / BatchView machinery.
 *
 * Algorithm: the unblocked LAPACK geqr2 (dlarfg to build each reflector, dlarf
 * to apply it to the trailing columns), run V matrices at a time. Compact
 * format stores element (i,j) of all V matrices contiguously, so the scalar
 * algorithm lifts with double -> V-wide vector, one lane per matrix. The single
 * data-dependent branch of dlarfg (xnorm == 0 -> tau = 0) is made branch-free
 * with a lane mask + select; see larfg_pack below.
 *
 * Compact storage convention (matches MKL Compact / mkl_?gepack_compact):
 *   group g = matrix index / V, slot v = matrix index % V:
 *     A_v(i,j)  = ap [ g*ldap*n*V + (j*ldap + i)*V + v ]   (column-major)
 *     tau_v(kk) = taup[ g*k*V      +  kk*V          + v ]
 *
 * On exit each matrix holds, per LAPACK ?geqrf:
 *   - R (min(m,n) x n upper trapezoid) on and above the diagonal,
 *   - the Householder vectors v(kk) below the diagonal (implicit 1 on it),
 *   - taup(0..k-1) the reflector scalars.
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#ifndef CQR_GEQRF_COMPACT_HPP
#define CQR_GEQRF_COMPACT_HPP

#include "cqr_compact.hpp" /* pack<T,V>, BatchView, make_view, make_const_view, vsqrt */

#include <cstddef>
#include <cstdint>
#include <cassert>
#include <cmath>
#include <type_traits>

namespace cqr {
namespace detail {

/* ------------------------------------------------------------------
 * V-wide helpers for the branch-free larfg.
 * ------------------------------------------------------------------ */

/* same-width signed integer for a floating type -- the lane type of the
 * masks the GNU vector relational operators yield, and of the bit-blend. */
// clang-format off
template <typename T> struct int_bits;
template <> struct int_bits<float>  { using type = std::int32_t; };
template <> struct int_bits<double> { using type = std::int64_t; };
// clang-format on
template <typename T> using int_bits_t = typename int_bits<T>::type;

template <typename T, int V> using mask_t = typename pack<int_bits_t<T>, V>::type;

/* These V-wide helpers take and return their vectors by reference. Passing a
 * GNU vector by value would, without -march, commit the base-ISA vector
 * argument/return ABI, which GCC and Clang (rightly) flag via -Wpsabi; a
 * reference is just a pointer, so there is no such boundary -- and once inlined
 * the codegen is identical -- keeping the build warning-clean with no compiler
 * flag. Results are written through an out-parameter (named first). */

/* r := mask ? a : b, lane-wise. Mask lanes are all-ones (true) or zero (false),
 * as produced by the GNU vector relational operators. The blend runs on the
 * may-alias integer view, so it survives on any T-aligned buffer without a
 * strict-aliasing violation. */
template <typename T, int V>
inline void vselect(typename pack<T, V>::type &r, const mask_t<T, V> &mask,
                    const typename pack<T, V>::type &a,
                    const typename pack<T, V>::type &b) noexcept
{
    using MT = mask_t<T, V>;
    const MT ai = reinterpret_cast<const MT &>(a);
    const MT bi = reinterpret_cast<const MT &>(b);
    const MT rr = (ai & mask) | (bi & ~mask);
    r = reinterpret_cast<const typename pack<T, V>::type &>(rr);
}

/* ------------------------------------------------------------------
 * Branch-free larfg for one pack (V matrices at once).
 *
 * Given the diagonal x0 = A(kk,kk) and tail = sum_{i>kk} A(i,kk)^2
 * (the squared norm of the sub-diagonal part of column kk), write per
 * lane the LAPACK dlarfg quantities, with the xnorm==0 branch folded
 * into a mask so divergent lanes cost nothing:
 *   rdiag -> A(kk,kk) on exit  (beta if there is a reflector, else x0),
 *   tau   -> the reflector scalar (0 if the column is already zeroed),
 *   inv   -> 1/(x0 - beta) to scale the reflector body (0 when tau=0).
 * The mask keys on tail>0 (below-diagonal norm), exactly as dlarfg, so
 * an already-triangular column yields tau=0 with x0 unchanged and no
 * lane ever forms 0/0 or 1/0.
 * ------------------------------------------------------------------ */

template <typename T, int V>
inline void larfg_pack(const typename pack<T, V>::type &x0,
                       const typename pack<T, V>::type &tail,
                       typename pack<T, V>::type &rdiag, typename pack<T, V>::type &tau,
                       typename pack<T, V>::type &inv) noexcept
{
    using VT = typename pack<T, V>::type;
    const VT zero = VT{};
    VT norm;
    vsqrt<T, V>(norm, x0 * x0 + tail);
    VT beta;
    vselect<T, V>(beta, x0 >= 0, -norm, norm); /* beta = -copysign(norm, x0) */
    const mask_t<T, V> has = (tail > 0);       /* is there anything to zero? */
    vselect<T, V>(tau, has, (beta - x0) / beta, zero);
    vselect<T, V>(inv, has, T(1) / (x0 - beta), zero);
    vselect<T, V>(rdiag, has, beta, x0);
}

/* ------------------------------------------------------------------
 * One group of V interleaved matrices, column-major (tuned path).
 *
 * a_ points at element (0,0) of the group; column j is contiguous
 * (row step = one V-wide pack), so the larfg reduction, the reflector
 * scaling, and the trailing-column update all walk contiguous packs.
 * ------------------------------------------------------------------ */

template <typename T, int V, typename Int = int>
void geqrf_compact_group(Int m, Int n, T *a_, Int ldap, T *tau_)
{
    using VT = typename pack<T, V>::type;
    static_assert(std::is_floating_point<T>::value,
                  "geqrf_compact is defined for real float/double");
    assert(ldap >= m);

    VT *A = reinterpret_cast<VT *>(a_);
    VT *tau = reinterpret_cast<VT *>(tau_);
    const Int k = (m < n) ? m : n;

    for (Int kk = 0; kk < k; ++kk) {
        VT *akk = A + static_cast<std::size_t>(kk) * ldap; /* column kk */

        /* build reflector H(kk) from column kk, rows kk..m-1 */
        const VT x0 = akk[kk];
        VT tail = VT{};
        for (Int i = kk + 1; i < m; ++i) {
            const VT a = akk[i];
            tail += a * a;
        }
        VT rdiag, t, inv;
        larfg_pack<T, V>(x0, tail, rdiag, t, inv);
        tau[kk] = t;
        for (Int i = kk + 1; i < m; ++i)
            akk[i] = akk[i] * inv; /* reflector body */
        akk[kk] = rdiag;           /* R diagonal */

        /* apply H(kk) = I - t v v^T (v(kk)=1 implicit) to trailing columns;
         * 4 columns at a time so each reflector load akk[i] is reused 4x */
        Int j = kk + 1;
        for (; j + 4 <= n; j += 4) {
            VT *b0 = A + static_cast<std::size_t>(j + 0) * ldap;
            VT *b1 = A + static_cast<std::size_t>(j + 1) * ldap;
            VT *b2 = A + static_cast<std::size_t>(j + 2) * ldap;
            VT *b3 = A + static_cast<std::size_t>(j + 3) * ldap;

            VT w0 = b0[kk], w1 = b1[kk], w2 = b2[kk], w3 = b3[kk];
            for (Int i = kk + 1; i < m; ++i) {
                const VT av = akk[i];
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
                const VT av = akk[i];
                b0[i] -= av * w0;
                b1[i] -= av * w1;
                b2[i] -= av * w2;
                b3[i] -= av * w3;
            }
        }

        /* remainder columns */
        for (; j < n; ++j) {
            VT *bj = A + static_cast<std::size_t>(j) * ldap;
            VT w = bj[kk];
            for (Int i = kk + 1; i < m; ++i)
                w += akk[i] * bj[i];
            bj[kk] -= t * w;
            w *= t;
            for (Int i = kk + 1; i < m; ++i)
                bj[i] -= akk[i] * w;
        }
    }
}

/* ------------------------------------------------------------------
 * One group, fully general: column- or row-major via BatchView strides.
 * Same geqr2 math; only the addressing differs.
 * ------------------------------------------------------------------ */

template <typename T, int V, typename Int = int>
void geqrf_compact_group_strided(Int m, Int n,
                                 BatchView<typename pack<T, V>::type, Int> A, T *tau_)
{
    using VT = typename pack<T, V>::type;
    static_assert(std::is_floating_point<T>::value,
                  "geqrf_compact is defined for real float/double");
    assert(A.special && A.panel);

    VT *tau = reinterpret_cast<VT *>(tau_);
    const Int k = (m < n) ? m : n;

    for (Int kk = 0; kk < k; ++kk) {
        /* build reflector H(kk) from column kk, rows kk..m-1 */
        const VT x0 = A(kk, kk);
        VT tail = VT{};
        for (Int i = kk + 1; i < m; ++i) {
            const VT a = A(i, kk);
            tail += a * a;
        }
        VT rdiag, t, inv;
        larfg_pack<T, V>(x0, tail, rdiag, t, inv);
        tau[kk] = t;
        for (Int i = kk + 1; i < m; ++i)
            A(i, kk) = A(i, kk) * inv;
        A(kk, kk) = rdiag;

        /* apply H(kk) to trailing columns, 4 at a time */
        Int j = kk + 1;
        for (; j + 4 <= n; j += 4) {
            VT w0 = A(kk, j + 0), w1 = A(kk, j + 1), w2 = A(kk, j + 2), w3 = A(kk, j + 3);
            for (Int i = kk + 1; i < m; ++i) {
                const VT av = A(i, kk);
                w0 += av * A(i, j + 0);
                w1 += av * A(i, j + 1);
                w2 += av * A(i, j + 2);
                w3 += av * A(i, j + 3);
            }
            A(kk, j + 0) -= t * w0;
            A(kk, j + 1) -= t * w1;
            A(kk, j + 2) -= t * w2;
            A(kk, j + 3) -= t * w3;

            w0 *= t;
            w1 *= t;
            w2 *= t;
            w3 *= t;
            for (Int i = kk + 1; i < m; ++i) {
                const VT av = A(i, kk);
                A(i, j + 0) -= av * w0;
                A(i, j + 1) -= av * w1;
                A(i, j + 2) -= av * w2;
                A(i, j + 3) -= av * w3;
            }
        }

        /* remainder columns */
        for (; j < n; ++j) {
            VT w = A(kk, j);
            for (Int i = kk + 1; i < m; ++i)
                w += A(i, kk) * A(i, j);
            A(kk, j) -= t * w;
            w *= t;
            for (Int i = kk + 1; i < m; ++i)
                A(i, j) -= A(i, kk) * w;
        }
    }
}

/* ------------------------------------------------------------------
 * All groups: nm matrices total (a padded partial last group is
 * processed too, which is harmless -- identity factors to tau = 0).
 * ------------------------------------------------------------------ */

// TODO: review: geqrf_compact is not called in-tree (both C adapters route
// through geqrf_compact_general below); it is kept as the col-major convenience
// driver for direct C++ users of this header. Drop it if that surface is not
// wanted (design doc 8.1).
template <typename T, int V, typename Int = int>
void geqrf_compact(Int m, Int n, T *ap, Int ldap, T *taup, Int nm)
{
    assert(ldap >= m && nm >= 1);

    const Int k = (m < n) ? m : n;
    const Int ngroups = (nm + V - 1) / V;
    const std::size_t str_a = static_cast<std::size_t>(ldap) * n * V;
    const std::size_t str_t = static_cast<std::size_t>(k) * V;

    for (Int g = 0; g < ngroups; ++g)
        geqrf_compact_group<T, V, Int>(m, n, ap + g * str_a, ldap, taup + g * str_t);
}

/* ------------------------------------------------------------------
 * All groups, fully general: column- or row-major.
 *
 * Column-major routes to the tuned contiguous kernel; row-major uses
 * the strided kernel (a row step is 1, a column step is ldap, so the
 * reflector axis -- down a column -- has stride ldap).
 * ------------------------------------------------------------------ */

template <typename T, int V, typename Int = int>
void geqrf_compact_general(bool rowmajor, Int m, Int n, T *ap, Int ldap, T *taup, Int nm)
{
    assert(nm >= 1 && m >= 0 && n >= 0);

    const Int k = (m < n) ? m : n;
    const Int ngroups = (nm + V - 1) / V;
    const std::size_t str_t = static_cast<std::size_t>(k) * V;

    /* element strides (in VT units) and per-matrix group stride (in T units) */
    const std::size_t a_special = rowmajor ? (std::size_t)ldap : 1; /* down a col */
    const std::size_t a_panel = rowmajor ? 1 : (std::size_t)ldap;   /* across cols */
    const std::size_t str_a =
        (rowmajor ? (std::size_t)ldap * m : (std::size_t)ldap * n) * V;

    for (Int g = 0; g < ngroups; ++g) {
        T *a = ap + g * str_a;
        T *tg = taup + g * str_t;
        if (!rowmajor)
            geqrf_compact_group<T, V, Int>(m, n, a, ldap, tg);
        else
            geqrf_compact_group_strided<T, V, Int>(
                m, n, make_view<T, V, Int>(a, a_special, a_panel), tg);
    }
}

} /* namespace detail */
} /* namespace cqr */

#endif /* CQR_GEQRF_COMPACT_HPP */
