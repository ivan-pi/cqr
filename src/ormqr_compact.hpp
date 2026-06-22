/* ormqr_compact.hpp
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
 *   - V is the compact-format interleave width (number of matrices whose
 *     element (i,j) is stored contiguously). It does NOT need to match the
 *     hardware vector width:
 *       x86:   V*sizeof(T) = 16/32/64 bytes maps exactly to XMM/YMM/ZMM.
 *       NEON:  128-bit registers; V=4 or V=8 doubles lower to short
 *              unrolled bursts of 2/4 independent fmla v*.2d chains,
 *              which wide cores (Apple M-series) execute very well.
 *       SVE:   compile fixed-width with -msve-vector-bits=512 on A64FX
 *              to map V=8 doubles onto one SVE register.
 *   - The algorithm is unblocked dorm2r: pivot-free and branch-free, so
 *     the scalar code lifts verbatim with double -> V-wide vector.
 *   - RHS columns are register-blocked (JB=4) so each reflector load
 *     ak[i] is reused across JB columns, halving the dominant A traffic.
 *
 * Compact storage convention (matches MKL Compact / mkl_?gepack_compact):
 *   group g = matrix index / V, slot v = matrix index % V:
 *     A_v(i,j)  = ap [ g*ldap*ncols_a*V + (j*ldap + i)*V + v ]
 *     tau_v(kk) = taup[ g*k*V           +  kk*V          + v ]
 *     B_v(i,j)  = bp [ g*ldbp*nrhs*V   + (j*ldbp + i)*V + v ]
 */

#ifndef ORMQR_COMPACT_HPP
#define ORMQR_COMPACT_HPP

#include <cstddef>
#include <cassert>

namespace ormqr {

/* ------------------------------------------------------------------ */
/* pack<T,V>::type : the V-wide SIMD element                          */
/* ------------------------------------------------------------------ */

/* Define the V-wide element as a GNU vector type when the compiler provides
 * the vector_size and may_alias attributes, detected directly via
 * __has_attribute (itself guarded for preprocessors that predate it). A
 * compiler that supplies these attributes -- GCC, Clang, Intel icpx/icpc --
 * uses the vector type; any other stops at the #error below. */
#if defined(__has_attribute)
#  if __has_attribute(vector_size) && __has_attribute(__may_alias__)
#    define CQR_HAS_GNU_VECTORS 1
#  endif
#endif

#if defined(CQR_HAS_GNU_VECTORS)

template <typename T, int V>
struct pack {
    /* aligned(alignof(T)) relaxes the alignment requirement so the type
     * is valid on any T-aligned buffer (unaligned vector loads are free
     * on all modern hardware); may_alias exempts it from strict-aliasing
     * violations when viewing a plain T array. */
    using type __attribute__((vector_size(V * sizeof(T)),
                              aligned(alignof(T)), may_alias)) = T;
};

#else
#error "ormqr_compact requires the GNU vector extensions " \
       "(__attribute__((vector_size)) with may_alias); compile with a " \
       "compiler that supports them (GCC, Clang, Intel icpx/icpc) in GNU " \
       "mode (-std=gnu++17)."
#endif

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
void ormqr_compact_group(Direction dir, Int m, Int nrhs, Int k,
                         const T *a_, Int ldap,
                         const T *tau_,
                         T *b_, Int ldbp)
{
    using VT = typename pack<T, V>::type;

    const VT *A   = reinterpret_cast<const VT *>(a_);
    const VT *tau = reinterpret_cast<const VT *>(tau_);
    VT       *B   = reinterpret_cast<VT *>(b_);

    const bool fwd = (dir == Direction::Forward);

    for (Int s = 0; s < k; ++s) {
        const Int kk = fwd ? s : k - 1 - s;     /* Q^T: ascending, Q: descending */
        const VT *ak = A + static_cast<std::size_t>(kk) * ldap;
        const VT  t  = tau[kk];

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
                w0 += av * b0[i]; w1 += av * b1[i];
                w2 += av * b2[i]; w3 += av * b3[i];
            }
            b0[kk] -= t * w0; b1[kk] -= t * w1;
            b2[kk] -= t * w2; b3[kk] -= t * w3;

            w0 *= t; w1 *= t; w2 *= t; w3 *= t;  /* fold tau into w */
            for (Int i = kk + 1; i < m; ++i) {
                const VT av = ak[i];
                b0[i] -= av * w0; b1[i] -= av * w1;
                b2[i] -= av * w2; b3[i] -= av * w3;
            }
        }

        /* remainder columns */
        for (; j < nrhs; ++j) {
            VT *bj = B + static_cast<std::size_t>(j) * ldbp;
            VT  w  = bj[kk];
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
/* One group, fully general: any side / layout via explicit strides.   */
/*                                                                     */
/* The unblocked dorm2r math is identical to the left/col-major kernel */
/* above; only the addressing changes. For reflector kk the essential  */
/* Householder vector is column kk of A (implicit 1 at the diagonal):  */
/*   ak[i] = A(i,kk),  i in (kk, spec_len),                            */
/* indexed by the "special" stride. C is swept along two strides:      */
/* the special index (rows for side='L', columns for side='R') and the */
/* panel index (the other dimension), register-blocked 4 at a time.    */
/*                                                                     */
/* All strides are in units of the V-wide pack element VT.             */
/* ------------------------------------------------------------------ */

template <typename T, int V, typename Int = int>
void ormqr_compact_group_strided(Direction dir, Int spec_len, Int panel_cnt, Int k,
                                 const T *a_, std::size_t a_spec, std::size_t a_kk,
                                 const T *tau_,
                                 T *c_, std::size_t c_spec, std::size_t c_panel)
{
    using VT = typename pack<T, V>::type;

    const VT *A   = reinterpret_cast<const VT *>(a_);
    const VT *tau = reinterpret_cast<const VT *>(tau_);
    VT       *C   = reinterpret_cast<VT *>(c_);

    const bool fwd = (dir == Direction::Forward);

    for (Int s = 0; s < k; ++s) {
        const Int kk = fwd ? s : k - 1 - s;     /* Q^T: ascending, Q: descending */
        const VT *ak = A + static_cast<std::size_t>(kk) * a_kk;
        const VT  t  = tau[kk];
        const std::size_t dk = static_cast<std::size_t>(kk) * c_spec;

        Int p = 0;

        /* main loop: 4 panel slices at a time; ak[i] loaded once, used 4x */
        for (; p + 4 <= panel_cnt; p += 4) {
            VT *c0 = C + static_cast<std::size_t>(p + 0) * c_panel;
            VT *c1 = C + static_cast<std::size_t>(p + 1) * c_panel;
            VT *c2 = C + static_cast<std::size_t>(p + 2) * c_panel;
            VT *c3 = C + static_cast<std::size_t>(p + 3) * c_panel;

            VT w0 = c0[dk], w1 = c1[dk], w2 = c2[dk], w3 = c3[dk];
            for (Int i = kk + 1; i < spec_len; ++i) {
                const VT av = ak[static_cast<std::size_t>(i) * a_spec];
                const std::size_t di = static_cast<std::size_t>(i) * c_spec;
                w0 += av * c0[di]; w1 += av * c1[di];
                w2 += av * c2[di]; w3 += av * c3[di];
            }
            c0[dk] -= t * w0; c1[dk] -= t * w1;
            c2[dk] -= t * w2; c3[dk] -= t * w3;

            w0 *= t; w1 *= t; w2 *= t; w3 *= t;  /* fold tau into w */
            for (Int i = kk + 1; i < spec_len; ++i) {
                const VT av = ak[static_cast<std::size_t>(i) * a_spec];
                const std::size_t di = static_cast<std::size_t>(i) * c_spec;
                c0[di] -= av * w0; c1[di] -= av * w1;
                c2[di] -= av * w2; c3[di] -= av * w3;
            }
        }

        /* remainder panel slices */
        for (; p < panel_cnt; ++p) {
            VT *cp = C + static_cast<std::size_t>(p) * c_panel;
            VT  w  = cp[dk];
            for (Int i = kk + 1; i < spec_len; ++i)
                w += ak[static_cast<std::size_t>(i) * a_spec] * cp[static_cast<std::size_t>(i) * c_spec];
            cp[dk] -= t * w;
            w *= t;
            for (Int i = kk + 1; i < spec_len; ++i)
                cp[static_cast<std::size_t>(i) * c_spec] -= ak[static_cast<std::size_t>(i) * a_spec] * w;
        }
    }
}

/* ------------------------------------------------------------------ */
/* All groups: nm matrices total (a padded partial last group is      */
/* processed too, which is harmless)                                  */
/* ------------------------------------------------------------------ */

template <typename T, int V, typename Int = int>
void ormqr_compact(char trans, Int m, Int nrhs, Int k,
                   const T *ap, Int ldap, Int ncols_a,
                   const T *taup,
                   T *bp, Int ldbp,
                   Int nm)
{
    assert(trans == 'T' || trans == 't' || trans == 'N' || trans == 'n');
    assert(ldap >= m && ldbp >= m && k <= m && nm >= 1);

    const Direction dir = (trans == 'T' || trans == 't')
                              ? Direction::Forward : Direction::Backward;
    const Int ngroups   = (nm + V - 1) / V;
    const std::size_t str_a = static_cast<std::size_t>(ldap) * ncols_a * V;
    const std::size_t str_t = static_cast<std::size_t>(k) * V;
    const std::size_t str_b = static_cast<std::size_t>(ldbp) * nrhs * V;

    for (Int g = 0; g < ngroups; ++g)
        ormqr_compact_group<T, V, Int>(dir, m, nrhs, k,
                                       ap + static_cast<std::size_t>(g) * str_a, ldap,
                                       taup + static_cast<std::size_t>(g) * str_t,
                                       bp + static_cast<std::size_t>(g) * str_b, ldbp);
}

/* ------------------------------------------------------------------ */
/* All groups, fully general: side = 'L'/'R', column- or row-major.    */
/*                                                                     */
/* C is the m x n batch; A is the (spec_len x k) reflector batch with  */
/* spec_len = m (side='L') or n (side='R'); ncols_a is the packed      */
/* column count of A (column-major group stride only). The left /      */
/* column-major case routes to the tuned contiguous kernel above; the  */
/* other three combinations use the strided kernel.                    */
/* ------------------------------------------------------------------ */

template <typename T, int V, typename Int = int>
void ormqr_compact_general(bool left, bool rowmajor, char trans,
                           Int m, Int n, Int k,
                           const T *ap, Int ldap, Int ncols_a,
                           const T *taup,
                           T *cp, Int ldcp,
                           Int nm)
{
    assert(trans == 'T' || trans == 't' || trans == 'N' || trans == 'n' ||
           trans == 'C' || trans == 'c');
    assert(nm >= 1);

    const bool tran = (trans == 'T' || trans == 't' ||
                       trans == 'C' || trans == 'c');
    /* dorm2r ordering: side='L' applies ascending for Q^T, side='R' flips. */
    const bool fwd       = left ? tran : !tran;
    const Direction dir  = fwd ? Direction::Forward : Direction::Backward;
    const Int  spec_len  = left ? m : n;
    const Int  panel_cnt = left ? n : m;

    /* element strides (in VT units) for the reflector column of A and for
     * the special / panel sweep of C. */
    std::size_t a_spec, a_kk, c_spec, c_panel;
    if (!rowmajor) { a_spec = 1;     a_kk = (size_t)ldap; }
    else           { a_spec = (size_t)ldap; a_kk = 1;     }
    if (left) {
        if (!rowmajor) { c_spec = 1;            c_panel = (size_t)ldcp; }
        else           { c_spec = (size_t)ldcp; c_panel = 1;            }
    } else {
        if (!rowmajor) { c_spec = (size_t)ldcp; c_panel = 1;            }
        else           { c_spec = 1;            c_panel = (size_t)ldcp; }
    }

    /* group strides (in scalar T units): elements packed per matrix is
     * ldap*(complementary extent), which is the column count for col-major
     * and the row count for row-major. */
    const std::size_t str_a = (rowmajor ? (size_t)ldap * spec_len
                                        : (size_t)ldap * ncols_a) * V;
    const std::size_t str_t = (size_t)k * V;
    const std::size_t str_c = (rowmajor ? (size_t)ldcp * m
                                        : (size_t)ldcp * n) * V;

    const Int ngroups = (nm + V - 1) / V;
    for (Int g = 0; g < ngroups; ++g) {
        const T *a  = ap   + (size_t)g * str_a;
        const T *tg = taup + (size_t)g * str_t;
        T       *c  = cp   + (size_t)g * str_c;
        if (left && !rowmajor)
            ormqr_compact_group<T, V, Int>(dir, m, n, k, a, ldap, tg, c, ldcp);
        else
            ormqr_compact_group_strided<T, V, Int>(dir, spec_len, panel_cnt, k,
                                                   a, a_spec, a_kk, tg,
                                                   c, c_spec, c_panel);
    }
}

} /* namespace ormqr */

#endif /* ORMQR_COMPACT_HPP */
