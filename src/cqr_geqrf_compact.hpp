/* cqr_geqrf_compact.hpp
 *
 * Compact (interleaved-batch) QR factorization, templated on scalar type T and
 * interleave width V:
 *
 *     A = Q R,   Q = H(0) H(1) ... H(k-1),   k = min(m, n),
 *     H(kk) = I - tau(kk) * v(kk) * v(kk)^T.
 *
 * A portable, vectorized mkl_?geqrf_compact. The algorithm is the unblocked
 * LAPACK geqr2 -- dlarfg to build each reflector, dlarf (the larf() shared with
 * ormqr) to apply it to the trailing columns -- run V matrices at a time: the
 * compact format stores element (i,j) of all V matrices contiguously, so the
 * scalar algorithm lifts with double -> V-wide vector, one lane per matrix. The
 * single data-dependent branch of dlarfg (xnorm == 0 -> tau = 0) becomes a lane
 * mask + select in larfg_pack.
 *
 * Compact storage (matches mkl_?gepack_compact); group g = idx/V, slot v = idx%V:
 *     A_v(i,j)  = ap [ g*ldap*n*V + (j*ldap + i)*V + v ]   (column-major)
 *     tau_v(kk) = taup[ g*k*V      +  kk*V          + v ]
 * On exit each matrix holds, per LAPACK ?geqrf, R (min(m,n) x n upper trapezoid)
 * on and above the diagonal, the Householder vectors v(kk) below it (implicit 1
 * on the diagonal), and taup(0..k-1) the reflector scalars.
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#ifndef CQR_GEQRF_COMPACT_HPP
#define CQR_GEQRF_COMPACT_HPP

#include "cqr_compact_common.hpp"
#include "cqr_ormqr_compact.hpp" /* larf */

#include <cstddef>
#include <cstdint>
#include <cassert>
#include <type_traits>

namespace cqr {
namespace detail {

/* Same-width signed integer for a floating type: the lane type of the masks the
 * GNU vector relational operators yield, and of the bit-blend below. */
// clang-format off
template <typename T> struct int_bits;
template <> struct int_bits<float>  { using type = std::int32_t; };
template <> struct int_bits<double> { using type = std::int64_t; };
// clang-format on
template <typename T, int V>
using mask_t = typename pack<typename int_bits<T>::type, V>::type;

/* r := mask ? a : b, lane-wise; mask lanes are all-ones or zero, as the vector
 * comparisons produce. The blend runs on the may-alias integer view. */
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

/* Branch-free dlarfg for one pack. Given the diagonal x0 = A(kk,kk) and
 * tail = sum_{i>kk} A(i,kk)^2, write per lane
 *   rdiag -> A(kk,kk) on exit (beta if there is a reflector, else x0),
 *   tau   -> the reflector scalar (0 if the column is already zeroed),
 *   inv   -> 1/(x0 - beta) to scale the reflector body (0 when tau = 0).
 * The mask keys on tail > 0 exactly as dlarfg, so an already-triangular column
 * (and a padded identity lane) yields tau = 0 with x0 unchanged, and no lane
 * ever forms 0/0 or 1/0. */
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
    const mask_t<T, V> has = (tail > 0);       /* anything to zero? */
    vselect<T, V>(tau, has, (beta - x0) / beta, zero);
    vselect<T, V>(inv, has, T(1) / (x0 - beta), zero);
    vselect<T, V>(rdiag, has, beta, x0);
}

/* One group of V interleaved m x n matrices, through the layout-agnostic view. */
template <typename T, int V, typename Int = int>
void geqrf_compact_group(Int m, Int n, BatchView<T, V, Int> A, T *tau_)
{
    using VT = typename pack<T, V>::type;
    static_assert(std::is_floating_point<T>::value,
                  "geqrf_compact is defined for real float/double");
    assert(A.si && A.sj);

    VT *tau = reinterpret_cast<VT *>(tau_);
    const auto Ac = A.as_const();
    const Int k = (m < n) ? m : n;

    for (Int kk = 0; kk < k; ++kk) {
        /* build H(kk) from column kk, rows kk..m-1 */
        VT tail = VT{};
        for (Int i = kk + 1; i < m; ++i)
            tail += A(i, kk) * A(i, kk);
        VT rdiag, t, inv;
        larfg_pack<T, V>(A(kk, kk), tail, rdiag, t, inv);
        tau[kk] = t;
        for (Int i = kk + 1; i < m; ++i)
            A(i, kk) = A(i, kk) * inv; /* reflector body */
        A(kk, kk) = rdiag;             /* R diagonal */

        /* apply H(kk) to the trailing columns kk+1 .. n-1 */
        larf<T, V>(kk, m, Ac, t, A, kk + 1, n);
    }
}

/* All groups, either layout. A padded partial last group is processed too,
 * harmlessly: an identity lane factors to R = I, tau = 0. */
template <typename T, int V, typename Int = int>
void geqrf_compact(bool rowmajor, Int m, Int n, T *ap, Int ldap, T *taup, Int nm)
{
    assert(nm >= 1 && m >= 0 && n >= 0);

    const Int k = (m < n) ? m : n;
    const std::size_t str_a = group_stride(rowmajor, ldap, m, n, V);
    const std::size_t str_t = (std::size_t)k * V;

    const Int ngroups = (nm + V - 1) / V;
    const double flops = 2.0 * m * n * k * V * ngroups; /* ~geqr2, all lanes */
    CQR_OMP_PARALLEL_GROUPS(ngroups, flops)
    for (Int g = 0; g < ngroups; ++g)
        geqrf_compact_group<T, V, Int>(
            m, n, make_view<T, V, Int>(ap + g * str_a, rowmajor, ldap), taup + g * str_t);
}

} /* namespace detail */
} /* namespace cqr */

#endif /* CQR_GEQRF_COMPACT_HPP */
