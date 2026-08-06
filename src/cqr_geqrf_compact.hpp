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
#include <vector>

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

/* ==================================================================
 * Blocked (Level-3) compact QR: the WY-representation panel/trailing
 * split, so the trailing matrix is streamed once per block of NB
 * columns (Level-3) instead of once per column (Level-2 geqr2). This
 * keeps throughput up once one interleaved group no longer fits in L2
 * -- above that size the unblocked kernel goes memory-bound (see
 * cqr_geqrf_compact_cache_blocking.md).
 *
 * Three pieces, each running V matrices at a time:
 *   larft_forward_compact       -- build the jb x jb block factor T,
 *   larfb_forward_left_compact  -- apply (I - V T^T V^T) to the trailing block,
 *   geqrf_blocked_compact_group -- panel-factor (reuse geqr2) + larft + larfb.
 *
 * Math identity with the unblocked kernel: geqr2 applies the reflectors in
 * ascending order, i.e. H(j+jb-1)...H(j) = (H(j)...H(j+jb-1))^T
 * = (I - V T V^T)^T = I - V T^T V^T to the trailing columns. So larfb uses
 * T^T (the LAPACK "Left, Transpose, Forward, Columnwise" case) and the blocked
 * factor reproduces the unblocked one to rounding. tau = 0 columns (padding,
 * already-triangular) give a zero T column and a no-op update, exactly as the
 * unblocked larfg_pack mask does.
 * ================================================================== */

/* Build T (jb x jb, upper-triangular, column-major dense, ld = jb) for a panel of
 * jb Householder columns at A (mm x jb, unit-lower-trapezoidal: implicit 1 on the
 * diagonal, reflector body A(i,c) below, 0 above), scalars tau[0..jb-1]. LAPACK
 * dlarft, forward/columnwise. */
template <typename T, int V, typename Int = int>
void larft_forward_compact(Int mm, Int jb, const typename pack<T, V>::type *A, Int ldap,
                           const typename pack<T, V>::type *tau,
                           typename pack<T, V>::type *Tf)
{
    using VT = typename pack<T, V>::type;
    for (Int c = 0; c < jb; ++c) {
        const VT tc = tau[c];
        /* stash z(p) = A(c,p) + sum_{i>c} A(i,p) A(i,c) in column c of T, p < c */
        for (Int p = 0; p < c; ++p) {
            VT z = A[(std::size_t)p * ldap + c];
            for (Int i = c + 1; i < mm; ++i)
                z += A[(std::size_t)p * ldap + i] * A[(std::size_t)c * ldap + i];
            Tf[(std::size_t)c * jb + p] = z;
        }
        /* T(0:c,c) = -tc * (Tupper(0:c,0:c) * z), in place: the write to z(p) only
         * follows reads of z(q>=p), so no value is clobbered before use. */
        for (Int p = 0; p < c; ++p) {
            VT s = VT{};
            for (Int q = p; q < c; ++q)
                s += Tf[(std::size_t)q * jb + p] * Tf[(std::size_t)c * jb + q];
            Tf[(std::size_t)c * jb + p] = -tc * s;
        }
        Tf[(std::size_t)c * jb + c] = tc;
    }
}

/* ------------------------------------------------------------------
 * Two 2-D (4x4) register-tiled GEMM microkernels for larfb's rectangular
 * bulk (the V2 block below the jb x jb triangular corner). Both reduce down
 * the contiguous row axis and hold a 4x4 output tile in registers, so each
 * loaded operand feeds 16 FMAs and the trailing matrix is streamed once per
 * 4 columns rather than once per reflector. Sizes need not be multiples of 4
 * (2x and 1x edge paths). These are the compact-format analogue of a packed
 * GEMM microkernel; A-panel packing buys little here because the reduction
 * is already contiguous (see cqr_geqrf_compact_cache_blocking.md).
 * ------------------------------------------------------------------ */

/* W(0:jb, 0:nb) += A(i0:mm, 0:jb)^T * C(i0:mm, 0:nb), reducing rows [i0, mm).
 * A/C column-major (col stride ldap/ldc), W column-major (col stride ldw). */
template <typename T, int V, typename Int = int>
void larfb_tn_acc(Int jb, Int nb, Int i0, Int mm, const typename pack<T, V>::type *A,
                  Int ldap, const typename pack<T, V>::type *C, Int ldc,
                  typename pack<T, V>::type *W, Int ldw)
{
    using VT = typename pack<T, V>::type;
    Int c = 0;
    for (; c + 4 <= jb; c += 4) {
        const VT *A0 = A + (std::size_t)(c + 0) * ldap,
                 *A1 = A + (std::size_t)(c + 1) * ldap,
                 *A2 = A + (std::size_t)(c + 2) * ldap,
                 *A3 = A + (std::size_t)(c + 3) * ldap;
        Int l = 0;
        for (; l + 4 <= nb; l += 4) {
            const VT *C0 = C + (std::size_t)(l + 0) * ldc,
                     *C1 = C + (std::size_t)(l + 1) * ldc,
                     *C2 = C + (std::size_t)(l + 2) * ldc,
                     *C3 = C + (std::size_t)(l + 3) * ldc;
            VT w00{}, w01{}, w02{}, w03{}, w10{}, w11{}, w12{}, w13{};
            VT w20{}, w21{}, w22{}, w23{}, w30{}, w31{}, w32{}, w33{};
            for (Int i = i0; i < mm; ++i) {
                const VT a0 = A0[i], a1 = A1[i], a2 = A2[i], a3 = A3[i];
                const VT b0 = C0[i], b1 = C1[i], b2 = C2[i], b3 = C3[i];
                w00 += a0 * b0;
                w10 += a1 * b0;
                w20 += a2 * b0;
                w30 += a3 * b0;
                w01 += a0 * b1;
                w11 += a1 * b1;
                w21 += a2 * b1;
                w31 += a3 * b1;
                w02 += a0 * b2;
                w12 += a1 * b2;
                w22 += a2 * b2;
                w32 += a3 * b2;
                w03 += a0 * b3;
                w13 += a1 * b3;
                w23 += a2 * b3;
                w33 += a3 * b3;
            }
            VT *W0 = W + (std::size_t)(l + 0) * ldw + c,
               *W1 = W + (std::size_t)(l + 1) * ldw + c;
            VT *W2p = W + (std::size_t)(l + 2) * ldw + c,
               *W3 = W + (std::size_t)(l + 3) * ldw + c;
            W0[0] += w00;
            W0[1] += w10;
            W0[2] += w20;
            W0[3] += w30;
            W1[0] += w01;
            W1[1] += w11;
            W1[2] += w21;
            W1[3] += w31;
            W2p[0] += w02;
            W2p[1] += w12;
            W2p[2] += w22;
            W2p[3] += w32;
            W3[0] += w03;
            W3[1] += w13;
            W3[2] += w23;
            W3[3] += w33;
        }
        for (; l < nb; ++l) { /* 4 c x 1 l */
            const VT *Cl = C + (std::size_t)l * ldc;
            VT w0{}, w1{}, w2{}, w3{};
            for (Int i = i0; i < mm; ++i) {
                const VT b = Cl[i];
                w0 += A0[i] * b;
                w1 += A1[i] * b;
                w2 += A2[i] * b;
                w3 += A3[i] * b;
            }
            VT *Wl = W + (std::size_t)l * ldw + c;
            Wl[0] += w0;
            Wl[1] += w1;
            Wl[2] += w2;
            Wl[3] += w3;
        }
    }
    for (; c < jb; ++c) { /* 1 c x 4 l */
        const VT *Ac = A + (std::size_t)c * ldap;
        Int l = 0;
        for (; l + 4 <= nb; l += 4) {
            const VT *C0 = C + (std::size_t)(l + 0) * ldc,
                     *C1 = C + (std::size_t)(l + 1) * ldc,
                     *C2 = C + (std::size_t)(l + 2) * ldc,
                     *C3 = C + (std::size_t)(l + 3) * ldc;
            VT w0{}, w1{}, w2{}, w3{};
            for (Int i = i0; i < mm; ++i) {
                const VT a = Ac[i];
                w0 += a * C0[i];
                w1 += a * C1[i];
                w2 += a * C2[i];
                w3 += a * C3[i];
            }
            W[(std::size_t)(l + 0) * ldw + c] += w0;
            W[(std::size_t)(l + 1) * ldw + c] += w1;
            W[(std::size_t)(l + 2) * ldw + c] += w2;
            W[(std::size_t)(l + 3) * ldw + c] += w3;
        }
        for (; l < nb; ++l) {
            const VT *Cl = C + (std::size_t)l * ldc;
            VT w{};
            for (Int i = i0; i < mm; ++i)
                w += Ac[i] * Cl[i];
            W[(std::size_t)l * ldw + c] += w;
        }
    }
}

/* C(i0:mm, 0:nb) -= A(i0:mm, 0:jb) * W2(0:jb, 0:nb), rows [i0, mm), reduce c. */
template <typename T, int V, typename Int = int>
void larfb_nn_sub(Int jb, Int nb, Int i0, Int mm, const typename pack<T, V>::type *A,
                  Int ldap, const typename pack<T, V>::type *W2, Int ldw,
                  typename pack<T, V>::type *C, Int ldc)
{
    using VT = typename pack<T, V>::type;
    Int i = i0;
    for (; i + 4 <= mm; i += 4) {
        Int l = 0;
        for (; l + 4 <= nb; l += 4) {
            VT c00{}, c01{}, c02{}, c03{}, c10{}, c11{}, c12{}, c13{};
            VT c20{}, c21{}, c22{}, c23{}, c30{}, c31{}, c32{}, c33{};
            for (Int c = 0; c < jb; ++c) {
                const VT *Ac = A + (std::size_t)c * ldap + i;
                const VT a0 = Ac[0], a1 = Ac[1], a2 = Ac[2], a3 = Ac[3];
                const VT b0 = W2[(std::size_t)(l + 0) * ldw + c],
                         b1 = W2[(std::size_t)(l + 1) * ldw + c],
                         b2 = W2[(std::size_t)(l + 2) * ldw + c],
                         b3 = W2[(std::size_t)(l + 3) * ldw + c];
                c00 += a0 * b0;
                c10 += a1 * b0;
                c20 += a2 * b0;
                c30 += a3 * b0;
                c01 += a0 * b1;
                c11 += a1 * b1;
                c21 += a2 * b1;
                c31 += a3 * b1;
                c02 += a0 * b2;
                c12 += a1 * b2;
                c22 += a2 * b2;
                c32 += a3 * b2;
                c03 += a0 * b3;
                c13 += a1 * b3;
                c23 += a2 * b3;
                c33 += a3 * b3;
            }
            VT *C0 = C + (std::size_t)(l + 0) * ldc + i,
               *C1 = C + (std::size_t)(l + 1) * ldc + i;
            VT *C2 = C + (std::size_t)(l + 2) * ldc + i,
               *C3 = C + (std::size_t)(l + 3) * ldc + i;
            C0[0] -= c00;
            C0[1] -= c10;
            C0[2] -= c20;
            C0[3] -= c30;
            C1[0] -= c01;
            C1[1] -= c11;
            C1[2] -= c21;
            C1[3] -= c31;
            C2[0] -= c02;
            C2[1] -= c12;
            C2[2] -= c22;
            C2[3] -= c32;
            C3[0] -= c03;
            C3[1] -= c13;
            C3[2] -= c23;
            C3[3] -= c33;
        }
        for (; l < nb; ++l) { /* 4 i x 1 l */
            VT s0{}, s1{}, s2{}, s3{};
            for (Int c = 0; c < jb; ++c) {
                const VT *Ac = A + (std::size_t)c * ldap + i;
                const VT b = W2[(std::size_t)l * ldw + c];
                s0 += Ac[0] * b;
                s1 += Ac[1] * b;
                s2 += Ac[2] * b;
                s3 += Ac[3] * b;
            }
            VT *Cl = C + (std::size_t)l * ldc + i;
            Cl[0] -= s0;
            Cl[1] -= s1;
            Cl[2] -= s2;
            Cl[3] -= s3;
        }
    }
    for (; i < mm; ++i) { /* 1 i x 4 l */
        Int l = 0;
        for (; l + 4 <= nb; l += 4) {
            VT s0{}, s1{}, s2{}, s3{};
            for (Int c = 0; c < jb; ++c) {
                const VT a = A[(std::size_t)c * ldap + i];
                s0 += a * W2[(std::size_t)(l + 0) * ldw + c];
                s1 += a * W2[(std::size_t)(l + 1) * ldw + c];
                s2 += a * W2[(std::size_t)(l + 2) * ldw + c];
                s3 += a * W2[(std::size_t)(l + 3) * ldw + c];
            }
            C[(std::size_t)(l + 0) * ldc + i] -= s0;
            C[(std::size_t)(l + 1) * ldc + i] -= s1;
            C[(std::size_t)(l + 2) * ldc + i] -= s2;
            C[(std::size_t)(l + 3) * ldc + i] -= s3;
        }
        for (; l < nb; ++l) {
            VT s{};
            for (Int c = 0; c < jb; ++c)
                s += A[(std::size_t)c * ldap + i] * W2[(std::size_t)l * ldw + c];
            C[(std::size_t)l * ldc + i] -= s;
        }
    }
}

/* Apply C := (I - V T^T V^T) C to the trailing block C (mm x nt) from the left, V
 * the panel at A (mm x jb, unit-lower-trapezoidal), T at Tf (jb x jb upper-tri).
 * Trailing-column-tiled by NC so the workspace stays cache-resident and V is
 * reused across tiles; scratch W must hold 2*jb*NC packs (W and W2). The jb x jb
 * unit-lower corner (V1) is handled with small direct loops; the (mm-jb) x jb
 * bulk (V2) goes through the 2-D register-tiled kernels above. */
template <typename T, int V, typename Int = int>
void larfb_forward_left_compact(Int mm, Int nt, Int jb,
                                const typename pack<T, V>::type *A, Int ldap,
                                const typename pack<T, V>::type *Tf,
                                typename pack<T, V>::type *C, Int ldc,
                                typename pack<T, V>::type *W, Int NC)
{
    using VT = typename pack<T, V>::type;
    VT *W2 = W + (std::size_t)jb * NC;
    for (Int l0 = 0; l0 < nt; l0 += NC) {
        const Int nb = (nt - l0 < NC) ? (nt - l0) : NC;
        VT *Cb = C + (std::size_t)l0 * ldc; /* C(:, l0 : l0+nb) */

        /* W = V^T Cb :  W(c,l) = Cb(c,l) + sum_{i>c} A(i,c) Cb(i,l). The jb x jb
         * unit-lower corner (rows < jt) is direct; the bulk rows [jt,mm) go through
         * the 2-D register-tiled kernel. */
        const Int jt = (jb < mm) ? jb : mm;
        for (Int c = 0; c < jb; ++c)
            for (Int l = 0; l < nb; ++l) {
                VT w = Cb[(std::size_t)l * ldc + c];
                for (Int i = c + 1; i < jt; ++i)
                    w += A[(std::size_t)c * ldap + i] * Cb[(std::size_t)l * ldc + i];
                W[(std::size_t)l * jb + c] = w;
            }
        larfb_tn_acc<T, V, Int>(jb, nb, jt, mm, A, ldap, Cb, ldc, W, jb);

        /* W2 = T^T W :  W2(c,l) = sum_{p<=c} T(p,c) W(p,l)  (small, jb x nb) */
        for (Int ll = 0; ll < nb; ++ll)
            for (Int c = 0; c < jb; ++c) {
                VT s = VT{};
                for (Int p = 0; p <= c; ++p)
                    s += Tf[(std::size_t)c * jb + p] * W[(std::size_t)ll * jb + p];
                W2[(std::size_t)ll * jb + c] = s;
            }

        /* Cb -= V W2 :  bulk rows [jt,mm) 2-D tiled; the jb x jb unit-lower corner
         * (rows < jt, C(i,l) -= W2(i,l) + sum_{c<i} A(i,c) W2(c,l)) direct. */
        larfb_nn_sub<T, V, Int>(jb, nb, jt, mm, A, ldap, W2, jb, Cb, ldc);
        for (Int l = 0; l < nb; ++l)
            for (Int i = 0; i < jt; ++i) {
                VT acc = W2[(std::size_t)l * jb + i];
                for (Int c = 0; c < i; ++c)
                    acc += A[(std::size_t)c * ldap + i] * W2[(std::size_t)l * jb + c];
                Cb[(std::size_t)l * ldc + i] -= acc;
            }
    }
}

/* One group of V interleaved matrices, column-major, blocked. NB = panel width,
 * NC = trailing-column tile; scratch holds NB*NB + 2*NB*NC packs (T + larfb
 * workspace). Each panel is factored by the unblocked geqr2 (cache-resident: only
 * jb wide), then its block reflector is applied to the trailing columns in one
 * Level-3 sweep. */
template <typename T, int V, typename Int = int>
void geqrf_blocked_compact_group(Int m, Int n, T *a_, Int ldap, T *tau_, Int NB, Int NC,
                                 typename pack<T, V>::type *scratch)
{
    using VT = typename pack<T, V>::type;
    VT *A = reinterpret_cast<VT *>(a_);
    VT *Tf = scratch;
    VT *W = scratch + (std::size_t)NB * NB;
    const Int k = (m < n) ? m : n;

    for (Int j = 0; j < k; j += NB) {
        const Int jb = (k - j < NB) ? (k - j) : NB;
        const Int mm = m - j;

        /* panel factorization: unblocked geqr2 on A[j:m, j:j+jb] */
        geqrf_compact_group<T, V, Int>(mm, jb, a_ + (std::size_t)(j * ldap + j) * V, ldap,
                                       tau_ + (std::size_t)j * V);

        /* apply the block reflector to the trailing columns j+jb .. n-1 */
        const Int nt = n - (j + jb);
        if (nt > 0) {
            const VT *Vp = A + (std::size_t)j * ldap + j;    /* &A(j, j) */
            const VT *tp = reinterpret_cast<VT *>(tau_) + j; /* &tau(j) */
            larft_forward_compact<T, V, Int>(mm, jb, Vp, ldap, tp, Tf);
            VT *Cp = A + (std::size_t)(j + jb) * ldap + j; /* &A(j, j+jb) */
            larfb_forward_left_compact<T, V, Int>(mm, nt, jb, Vp, ldap, Tf, Cp, ldap, W,
                                                  NC);
        }
    }
}

/* All groups, column-major, blocked. Allocates the per-thread scratch once and
 * drives geqrf_blocked_compact_group over every pack (a padded partial last group
 * factors identities to tau = 0, a harmless no-op). NB/NC are the tuning knobs;
 * NB is clamped to k = min(m,n). */
template <typename T, int V, typename Int = int>
void geqrf_blocked_compact(Int m, Int n, T *ap, Int ldap, T *taup, Int nm, Int NB, Int NC)
{
    using VT = typename pack<T, V>::type;
    assert(ldap >= m && nm >= 1);

    const Int k = (m < n) ? m : n;
    if (NB < 1) NB = 1;
    if (NC < 1) NC = 1;
    if (k > 0 && NB > k) NB = k;

    const Int ngroups = (nm + V - 1) / V;
    const std::size_t str_a = (std::size_t)ldap * n * V;
    const std::size_t str_t = (std::size_t)k * V;
    /* scratch: NB*NB (T) + 2*NB*NC (larfb W/W2) packs, held as a plain-T buffer
     * (std::vector<VT> would warn -Wignored-attributes) and aliased as packs. */
    const std::size_t npacks = (std::size_t)NB * NB + 2 * (std::size_t)NB * NC;
    std::vector<T> scratch(npacks * V);
    VT *sc = reinterpret_cast<VT *>(scratch.data());

    for (Int g = 0; g < ngroups; ++g)
        geqrf_blocked_compact_group<T, V, Int>(m, n, ap + g * str_a, ldap,
                                               taup + g * str_t, NB, NC, sc);
}

} /* namespace detail */
} /* namespace cqr */

#endif /* CQR_GEQRF_COMPACT_HPP */
