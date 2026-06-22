/* ormqr_compact.hpp
 *
 * Compact (interleaved-batch) application of Householder reflectors,
 * templated on scalar type T and interleave width V:
 *
 *     B := op(Q) * B,   op(Q) = Q ('N') or Q^T ('T'),  side = 'L'
 *
 * The missing mkl_?ormqr_compact / a portable armpl_?ormqr_interleave_batch.
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

#pragma once

#include <cstddef>
#include <cassert>

namespace ormqr {

/* ------------------------------------------------------------------ */
/* pack<T,V>::type : the V-wide SIMD element                          */
/* ------------------------------------------------------------------ */

#if defined(__GNUC__) || defined(__clang__)

template <typename T, int V>
struct pack {
    /* aligned(alignof(T)) relaxes the alignment requirement so the type
     * is valid on any T-aligned buffer (unaligned vector loads are free
     * on all modern hardware); may_alias exempts it from strict-aliasing
     * violations when viewing a plain T array. */
    using type __attribute__((vector_size(V * sizeof(T)),
                              aligned(alignof(T)), may_alias)) = T;
};

#else /* portable fallback (e.g. MSVC): element-wise operator overloads */

template <typename T, int V>
struct pack {
    struct type {
        T v[V];
        friend type operator+(type a, type b) {
            for (int i = 0; i < V; ++i) a.v[i] += b.v[i]; return a;
        }
        friend type operator-(type a, type b) {
            for (int i = 0; i < V; ++i) a.v[i] -= b.v[i]; return a;
        }
        friend type operator*(type a, type b) {
            for (int i = 0; i < V; ++i) a.v[i] *= b.v[i]; return a;
        }
        type& operator+=(type b) { return *this = *this + b; }
        type& operator-=(type b) { return *this = *this - b; }
        type& operator*=(type b) { return *this = *this * b; }
    };
};

#endif

/* ------------------------------------------------------------------ */
/* One group of V interleaved matrices                                 */
/* ------------------------------------------------------------------ */

template <typename T, int V>
void ormqr_compact_group(char trans, int m, int nrhs, int k,
                         const T *a_, int ldap,
                         const T *tau_,
                         T *b_, int ldbp)
{
    using VT = typename pack<T, V>::type;

    const VT *A   = reinterpret_cast<const VT *>(a_);
    const VT *tau = reinterpret_cast<const VT *>(tau_);
    VT       *B   = reinterpret_cast<VT *>(b_);

    const bool fwd = (trans == 'T' || trans == 't');

    for (int s = 0; s < k; ++s) {
        const int kk = fwd ? s : k - 1 - s;     /* Q^T: ascending, Q: descending */
        const VT *ak = A + static_cast<size_t>(kk) * ldap;
        const VT  t  = tau[kk];

        int j = 0;

        /* main loop: 4 RHS columns at a time; ak[i] loaded once, used 4x */
        for (; j + 4 <= nrhs; j += 4) {
            VT *b0 = B + static_cast<size_t>(j + 0) * ldbp;
            VT *b1 = B + static_cast<size_t>(j + 1) * ldbp;
            VT *b2 = B + static_cast<size_t>(j + 2) * ldbp;
            VT *b3 = B + static_cast<size_t>(j + 3) * ldbp;

            VT w0 = b0[kk], w1 = b1[kk], w2 = b2[kk], w3 = b3[kk];
            for (int i = kk + 1; i < m; ++i) {
                const VT av = ak[i];
                w0 += av * b0[i]; w1 += av * b1[i];
                w2 += av * b2[i]; w3 += av * b3[i];
            }
            b0[kk] -= t * w0; b1[kk] -= t * w1;
            b2[kk] -= t * w2; b3[kk] -= t * w3;

            w0 *= t; w1 *= t; w2 *= t; w3 *= t;  /* fold tau into w */
            for (int i = kk + 1; i < m; ++i) {
                const VT av = ak[i];
                b0[i] -= av * w0; b1[i] -= av * w1;
                b2[i] -= av * w2; b3[i] -= av * w3;
            }
        }

        /* remainder columns */
        for (; j < nrhs; ++j) {
            VT *bj = B + static_cast<size_t>(j) * ldbp;
            VT  w  = bj[kk];
            for (int i = kk + 1; i < m; ++i)
                w += ak[i] * bj[i];
            bj[kk] -= t * w;
            w *= t;
            for (int i = kk + 1; i < m; ++i)
                bj[i] -= ak[i] * w;
        }
    }
}

/* ------------------------------------------------------------------ */
/* All groups: nm matrices total (a padded partial last group is      */
/* processed too, which is harmless)                                  */
/* ------------------------------------------------------------------ */

template <typename T, int V>
void ormqr_compact(char trans, int m, int nrhs, int k,
                   const T *ap, int ldap, int ncols_a,
                   const T *taup,
                   T *bp, int ldbp,
                   int nm)
{
    assert(trans == 'T' || trans == 't' || trans == 'N' || trans == 'n');
    assert(ldap >= m && ldbp >= m && k <= m && nm >= 1);

    const int ngroups   = (nm + V - 1) / V;
    const size_t str_a  = static_cast<size_t>(ldap) * ncols_a * V;
    const size_t str_t  = static_cast<size_t>(k) * V;
    const size_t str_b  = static_cast<size_t>(ldbp) * nrhs * V;

    for (int g = 0; g < ngroups; ++g)
        ormqr_compact_group<T, V>(trans, m, nrhs, k,
                                  ap + g * str_a, ldap,
                                  taup + g * str_t,
                                  bp + g * str_b, ldbp);
}

} /* namespace ormqr */
