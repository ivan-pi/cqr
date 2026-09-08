/* cqr_sytrsnp_compact.hpp
 *
 * Compact (interleaved-batch) solve A X = B from the unpivoted LDL^T
 * factorization of cqr_sytrfnp_compact.hpp, templated on scalar type T and
 * interleave width V. With the factor held in ap (D on the diagonal, unit L or
 * U strictly off it) and B (n x nrhs) in bp, each matrix's solve is three
 * in-place substitution sweeps:
 *
 *     uplo lower, A = L D L^T:   L z = B;   D w = z;   L^T X = w
 *     uplo upper, A = U^T D U:   U^T z = B; D w = z;   U   X = w
 *
 * The two triangular sweeps are the compact trsm group kernels with a unit
 * diagonal (the stored diagonal -- D -- is not read by them): column-major
 * takes trsm's tuned side='L' row-dot path, row-major the strided kernel. The
 * diagonal solve between them, B(i,:) *= 1/D(i), is the one operation trsm
 * cannot express, an O(n*nrhs) row scaling next to the O(n^2*nrhs) sweeps. No
 * workspace; B is overwritten with X. The LAPACK ?sytrs analogue, minus ipiv.
 *
 * A zero D(i) -- a singular matrix, or a lane already poisoned by a zero pivot
 * in the factorization -- turns into Inf/NaN in that lane's solution, matching
 * the factorization's no-check contract (design section 6.2). Padded slots of
 * a partial last group hold identity factors (D = I), so the whole group runs
 * unmasked, as in the factorization and trsm.
 *
 * Compact storage is as in cqr_sytrfnp_compact.hpp / cqr_trsm_compact.hpp: the
 * factor batch ap is n x n with leading dimension ldap, the RHS batch bp is
 * n x nrhs with leading dimension ldbp (>= n column-major, >= nrhs row-major).
 *
 * Assisted-by: Claude
 */

#ifndef CQR_SYTRSNP_COMPACT_HPP
#define CQR_SYTRSNP_COMPACT_HPP

#include "cqr_compact_common.hpp"
#include "cqr_trsm_compact.hpp"

#include <cstddef>
#include <cassert>
#include <type_traits>

namespace cqr::detail {

/* One group: the three sweeps on the factor at `a` and the RHS block at `b`.
 * The stored factor is transposed in the first sweep exactly when it is upper
 * (U^T z = B), and in the second exactly when it is lower (L^T X = w). */
template <typename T, int V, typename Int = int>
void sytrsnp_compact_group(bool rowmajor, bool upper, Int n, Int nrhs, const T *a,
                           Int ldap, T *b, Int ldbp)
{
    using VT = typename pack<T, V>::type;
    static_assert(std::is_floating_point_v<T>,
                  "sytrsnp_compact is defined for real float/double");

    /* unit-triangular sweep with op(F) = F (tran false) or F^T (tran true) */
    const auto sweep = [&](bool tran) {
        trsm_compact_group<T, V, Int>(/*left=*/true, upper, rowmajor, tran, /*unit=*/true,
                                      n, nrhs, T(1), a, ldap, b, ldbp);
    };

    sweep(upper); /* L z = B  or  U^T z = B */

    /* diagonal solve w = D^{-1} z: one reciprocal per factor row, reused
     * across the nrhs columns */
    const auto A = make_const_view<T, V, Int>(a, rowmajor, ldap);
    const auto B = make_view<T, V, Int>(b, rowmajor, ldbp);
    for (Int i = 0; i < n; ++i) {
        const VT invd = T(1) / A(i, i);
        for (Int j = 0; j < nrhs; ++j)
            B(i, j) = B(i, j) * invd;
    }

    sweep(!upper); /* L^T X = w  or  U X = w */
}

/* ~flops of the three sweeps of one group: 2 n^2 nrhs per matrix, times V. */
template <typename Int> inline double sytrsnp_flops(Int n, Int nrhs, int V)
{
    return 2.0 * n * n * nrhs * V;
}

/* All groups, any layout / uplo. */
template <typename T, int V, typename Int = int>
void sytrsnp_compact(bool rowmajor, bool upper, Int n, Int nrhs, const T *ap, Int ldap,
                     T *bp, Int ldbp, Int nm)
{
    assert(nm >= 1 && n >= 0 && nrhs >= 0);

    const std::size_t str_a = group_stride(rowmajor, ldap, n, n, V);
    const std::size_t str_b = group_stride(rowmajor, ldbp, n, nrhs, V);

    for_each_group<V>(
        nm,
        [&](Int g) {
            sytrsnp_compact_group<T, V, Int>(rowmajor, upper, n, nrhs, ap + g * str_a,
                                             ldap, bp + g * str_b, ldbp);
        },
        sytrsnp_flops(n, nrhs, V));
}

} /* namespace cqr::detail */

#endif /* CQR_SYTRSNP_COMPACT_HPP */
