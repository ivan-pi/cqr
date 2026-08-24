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
 * The two triangular sweeps are the existing compact trsm kernels with a unit
 * diagonal (the stored diagonal -- D -- is not read by them); the diagonal solve
 * between them, B(i,:) *= 1/D(i), is the one operation trsm cannot express, an
 * O(n*nrhs) row scaling that is negligible next to the O(n^2*nrhs) sweeps. No
 * workspace is needed; B is overwritten with X.
 *
 * A zero D(i) -- a singular matrix, or a lane already poisoned by a zero pivot
 * in the factorization -- turns into Inf/NaN in that lane's solution, matching
 * the factorization's no-check contract (design section 6.2). Padded slots of a
 * partial last group hold identity factors (D = I), so the whole group runs
 * unmasked, as in the factorization and trsm.
 *
 * Compact storage is as in cqr_sytrfnp_compact.hpp / cqr_trsm_compact.hpp; the
 * factor batch ap is n x n with leading dimension ldap, the RHS batch bp is
 * n x nrhs with leading dimension ldbp.
 *
 * Assisted-by: Claude
 */

#ifndef CQR_SYTRSNP_COMPACT_HPP
#define CQR_SYTRSNP_COMPACT_HPP

#include "cqr_compact_common.hpp" /* pack<T,V>, BatchView, make_view */
#include "cqr_trsm_compact.hpp"   /* trsm_compact_general */

#include <cstddef>
#include <cassert>
#include <type_traits>

namespace cqr {
namespace detail {

/* ------------------------------------------------------------------
 * All groups, any layout / uplo (the entry point both C adapters call).
 *
 * The triangular sweeps delegate to trsm_compact_general (side left, unit
 * diagonal), which routes column-major to its tuned kernel; the diagonal solve
 * walks the strided views directly -- one reciprocal per factor row, reused
 * across the nrhs columns.
 * ------------------------------------------------------------------ */

template <typename T, int V, typename Int = int>
void sytrsnp_compact_general(bool rowmajor, bool upper, Int n, Int nrhs, const T *ap,
                             Int ldap, T *bp, Int ldbp, Int nm)
{
    using VT = typename pack<T, V>::type;
    static_assert(std::is_floating_point<T>::value,
                  "sytrsnp_compact is defined for real float/double");
    assert(nm >= 1 && n >= 0 && nrhs >= 0);

    /* 1. first unit-triangular sweep: L z = B (lower) or U^T z = B (upper) --
     * the stored factor is transposed exactly when it is upper. */
    trsm_compact_general<T, V, Int>(/*left=*/true, upper, rowmajor, /*tran=*/upper,
                                    /*unit=*/true, n, nrhs, T(1), ap, ldap, bp, ldbp, nm);

    /* 2. diagonal solve w = D^{-1} z: scale row i of B by 1/D(i). Element
     * strides as in trsm_compact_general: column-major steps rows by 1 and
     * columns by ld, row-major the reverse; the per-group strides are
     * ldap*n (A is n x n either way) and ldbp*(rows|cols) of B. */
    const Int a_row = rowmajor ? ldap : Int(1);
    const Int a_col = rowmajor ? Int(1) : ldap;
    const Int b_row = rowmajor ? ldbp : Int(1);
    const Int b_col = rowmajor ? Int(1) : ldbp;
    const std::size_t str_a = static_cast<std::size_t>(ldap) * n * V;
    const std::size_t str_b = (rowmajor ? static_cast<std::size_t>(ldbp) * n
                                        : static_cast<std::size_t>(ldbp) * nrhs) *
                              V;

    const Int ngroups = (nm + V - 1) / V;
    for (Int g = 0; g < ngroups; ++g) {
        auto A = make_const_view<T, V, Int>(ap + g * str_a, a_row, a_col);
        auto B = make_view<T, V, Int>(bp + g * str_b, b_row, b_col);
        for (Int i = 0; i < n; ++i) {
            const VT invd = T(1) / A(i, i);
            for (Int j = 0; j < nrhs; ++j)
                B(i, j) = B(i, j) * invd;
        }
    }

    /* 3. second unit-triangular sweep: L^T X = w (lower) or U X = w (upper). */
    trsm_compact_general<T, V, Int>(/*left=*/true, upper, rowmajor, /*tran=*/!upper,
                                    /*unit=*/true, n, nrhs, T(1), ap, ldap, bp, ldbp, nm);
}

} /* namespace detail */
} /* namespace cqr */

#endif /* CQR_SYTRSNP_COMPACT_HPP */
