/* cqr_ormqr_compact_dispatch.cpp
 *
 * extern "C" wrappers around the templated implementation; dispatch on
 * the runtime interleave width V to a compile-time instantiation.
 *
 * Assisted-by: Claude:claude-fable-5 Claude:claude-opus-4.8
 */

#include "cqr_compact.h"
#include "cqr_ormqr_compact.hpp"

namespace {

/* Validate the arguments LAPACK-style and dispatch on the interleave width V.
 * Returns 0 on success, or -j if the j-th argument (1-based, in signature
 * order) had an illegal value. Pointer arguments are not inspected, matching
 * LAPACK; never aborts the host process. */
template <typename T>
int dispatch(char trans, int m, int nrhs, int k, const T *ap, int ldap, const T *taup,
             T *bp, int ldbp, int V, int nm)
{
    const bool trans_ok = (trans == 'T' || trans == 't' || trans == 'N' || trans == 'n');
    const int ldmin = (m < 1 ? 1 : m); /* max(1, m) */

    if (!trans_ok) return -1;
    if (m < 0) return -2;
    if (nrhs < 0) return -3;
    if (k < 0 || k > m) return -4;
    if (ldap < ldmin) return -6;
    if (ldbp < ldmin) return -9;
    if (V != 2 && V != 4 && V != 8 && V != 16) return -10;
    if (nm < 0) return -11;

    /* Nothing to compute for an empty problem (also keeps the kernel's
     * nm >= 1 / k >= 1 invariants satisfied below). */
    if (m == 0 || nrhs == 0 || k == 0 || nm == 0) return 0;

    switch (V) {
    case 2:
        cqr::detail::ormqr_compact<T, 2>(trans, m, nrhs, k, ap, ldap, taup, bp, ldbp, nm);
        break;
    case 4:
        cqr::detail::ormqr_compact<T, 4>(trans, m, nrhs, k, ap, ldap, taup, bp, ldbp, nm);
        break;
    case 8:
        cqr::detail::ormqr_compact<T, 8>(trans, m, nrhs, k, ap, ldap, taup, bp, ldbp, nm);
        break;
    case 16:
        cqr::detail::ormqr_compact<T, 16>(trans, m, nrhs, k, ap, ldap, taup, bp, ldbp,
                                          nm);
        break;
    }
    return 0;
}

} /* anonymous namespace */

extern "C" int dormqr_compact(char trans, int m, int nrhs, int k, const double *ap,
                              int ldap, const double *taup, double *bp, int ldbp, int V,
                              int nm)
{
    return dispatch<double>(trans, m, nrhs, k, ap, ldap, taup, bp, ldbp, V, nm);
}

extern "C" int sormqr_compact(char trans, int m, int nrhs, int k, const float *ap,
                              int ldap, const float *taup, float *bp, int ldbp, int V,
                              int nm)
{
    return dispatch<float>(trans, m, nrhs, k, ap, ldap, taup, bp, ldbp, V, nm);
}
