/* cqr_sysvnp_compact.hpp
 *
 * Compact (interleaved-batch) symmetric solve A X = B in one pass, without
 * pivoting: the unpivoted LDL^T factorization of cqr_sytrfnp_compact.hpp fused
 * with the three-sweep solve of cqr_sytrsnp_compact.hpp, per group. The LAPACK
 * ?sysv analogue (factor, then solve), minus ipiv and workspace.
 *
 * Fusing matters for throughput, not arithmetic: the result is bit-identical
 * to sytrfnp followed by sytrsnp (the same group kernels run in the same order
 * on the same data), but each group's factor is solved with while it is still
 * cache-resident, instead of the batch being streamed once for the
 * factorization and once more for the solve -- the penalty the solve benchmark
 * measured for whole-pool pipelines (PLANS.md) -- and the whole solve threads
 * as one group loop. On exit ap holds the factor (D on the diagonal, unit L or
 * U strictly off it, the opposite triangle untouched) and bp holds X.
 *
 * Assisted-by: Claude
 */

#ifndef CQR_SYSVNP_COMPACT_HPP
#define CQR_SYSVNP_COMPACT_HPP

#include "cqr_compact_common.hpp"
#include "cqr_sytrfnp_compact.hpp"
#include "cqr_sytrsnp_compact.hpp"

#include <cstddef>
#include <cassert>

namespace cqr::detail {

/* All groups, any layout / uplo: per group, factor then solve. */
template <typename T, int V, typename Int = int>
void sysvnp_compact(bool rowmajor, bool upper, Int n, Int nrhs, T *ap, Int ldap, T *bp,
                    Int ldbp, Int nm)
{
    assert(nm >= 1 && n >= 0 && nrhs >= 0);

    const std::size_t str_a = group_stride(rowmajor, ldap, n, n, V);
    const std::size_t str_b = group_stride(rowmajor, ldbp, n, nrhs, V);

    for_each_group<V>(
        nm,
        [&](Int g) {
            T *a = ap + g * str_a;
            sytrfnp_compact_group<T, V, Int>(
                n, make_lower_view<T, V, Int>(a, rowmajor, upper, ldap));
            sytrsnp_compact_group<T, V, Int>(rowmajor, upper, n, nrhs, a, ldap,
                                             bp + g * str_b, ldbp);
        },
        sytrfnp_flops(n, V) + sytrsnp_flops(n, nrhs, V));
}

} /* namespace cqr::detail */

#endif /* CQR_SYSVNP_COMPACT_HPP */
