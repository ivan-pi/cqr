/* cqr_trsm_compact.hpp
 *
 * A templated compact (interleaved-batch) triangular solve in the GNU
 * `vector_size` style of ../src, giving GCC and clang a trsm of their own to pit
 * against mkl_dtrsm_compact in the shootout (the repo otherwise uses MKL's). Same
 * QR-solve case as the ISPC cqr_ispc_dtrsm_compact and MKL's
 * mkl_dtrsm_compact(LEFT, UPPER, NOTRANS, NONUNIT): back-substitution on the
 * upper-triangular R, x_i = (alpha b_i - sum_{l>i} A(i,l) x_l) / A(i,i), one group
 * at a time, 4 RHS columns blocked.
 * Assisted-by: Claude:claude-opus-4.8
 */

#ifndef CQR_TRSM_COMPACT_HPP
#define CQR_TRSM_COMPACT_HPP

#include "cqr_compact.hpp" /* pack<T,V> */

#include <cstddef>
#include <cassert>
#include <type_traits>

namespace cqr {
namespace detail {

/* One group: B := alpha * R^-1 B, R upper-triangular (n x n), non-unit diag. */
template <typename T, int V, typename Int = int>
void trsm_compact_group(Int n, Int nrhs, T alpha, const T *a_, Int ldap, T *b_, Int ldbp)
{
    using VT = typename pack<T, V>::type;
    static_assert(std::is_floating_point<T>::value,
                  "trsm_compact is defined for real float/double");
    assert(ldap >= n && ldbp >= n);

    const VT *A = reinterpret_cast<const VT *>(a_);
    VT *B = reinterpret_cast<VT *>(b_);

    Int j = 0;
    /* 4 RHS columns at a time: each A(i,l) load is reused across 4 columns and
     * the reciprocal diagonal 1/A(i,i) is formed once per row. */
    for (; j + 4 <= nrhs; j += 4) {
        VT *b0 = B + static_cast<std::size_t>(j + 0) * ldbp;
        VT *b1 = B + static_cast<std::size_t>(j + 1) * ldbp;
        VT *b2 = B + static_cast<std::size_t>(j + 2) * ldbp;
        VT *b3 = B + static_cast<std::size_t>(j + 3) * ldbp;
        for (Int i = n - 1; i >= 0; --i) {
            VT s0 = alpha * b0[i], s1 = alpha * b1[i];
            VT s2 = alpha * b2[i], s3 = alpha * b3[i];
            for (Int l = i + 1; l < n; ++l) {
                const VT r = A[static_cast<std::size_t>(l) * ldap + i]; /* A(i,l) */
                s0 -= r * b0[l];
                s1 -= r * b1[l];
                s2 -= r * b2[l];
                s3 -= r * b3[l];
            }
            const VT dinv =
                T(1) / A[static_cast<std::size_t>(i) * ldap + i]; /* 1/A(i,i) */
            b0[i] = s0 * dinv;
            b1[i] = s1 * dinv;
            b2[i] = s2 * dinv;
            b3[i] = s3 * dinv;
        }
    }
    /* remainder columns */
    for (; j < nrhs; ++j) {
        VT *bj = B + static_cast<std::size_t>(j) * ldbp;
        for (Int i = n - 1; i >= 0; --i) {
            VT s = alpha * bj[i];
            for (Int l = i + 1; l < n; ++l)
                s -= A[static_cast<std::size_t>(l) * ldap + i] * bj[l];
            bj[i] = s / A[static_cast<std::size_t>(i) * ldap + i];
        }
    }
}

/* All groups (a padded partial last group is processed too, harmlessly). */
template <typename T, int V, typename Int = int>
void trsm_compact(Int n, Int nrhs, T alpha, const T *ap, Int ldap, T *bp, Int ldbp,
                  Int nm)
{
    assert(nm >= 1);
    const Int ngroups = (nm + V - 1) / V;
    const std::size_t str_a = static_cast<std::size_t>(ldap) * n * V;
    const std::size_t str_b = static_cast<std::size_t>(ldbp) * nrhs * V;
    for (Int g = 0; g < ngroups; ++g)
        trsm_compact_group<T, V, Int>(n, nrhs, alpha, ap + g * str_a, ldap,
                                      bp + g * str_b, ldbp);
}

} /* namespace detail */
} /* namespace cqr */

#endif /* CQR_TRSM_COMPACT_HPP */
