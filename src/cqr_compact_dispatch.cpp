/* cqr_compact_dispatch.cpp
 *
 * extern "C" wrappers around the templated implementation; dispatch on
 * the runtime interleave width V to a compile-time instantiation.
 */

#include "cqr_compact.hpp"
#include "cqr_compact.h"

namespace {

/* Returns 0 on success or -11 if the interleave width V (the 11th argument)
 * is unsupported -- LAPACK sign convention, never aborts the host process. */
template <typename T>
int dispatch(char trans, int m, int nrhs, int k,
             const T *ap, int ldap, int ncols_a,
             const T *taup, T *bp, int ldbp,
             int V, int nm)
{
    switch (V) {
    case 2:  cqr::detail::ormqr_compact<T, 2>(trans, m, nrhs, k, ap, ldap, ncols_a, taup, bp, ldbp, nm); return 0;
    case 4:  cqr::detail::ormqr_compact<T, 4>(trans, m, nrhs, k, ap, ldap, ncols_a, taup, bp, ldbp, nm); return 0;
    case 8:  cqr::detail::ormqr_compact<T, 8>(trans, m, nrhs, k, ap, ldap, ncols_a, taup, bp, ldbp, nm); return 0;
    case 16: cqr::detail::ormqr_compact<T, 16>(trans, m, nrhs, k, ap, ldap, ncols_a, taup, bp, ldbp, nm); return 0;
    default: return -11;   /* V is the 11th argument */
    }
}

} /* anonymous namespace */

extern "C" int dormqr_compact(char trans, int m, int nrhs, int k,
                              const double *ap, int ldap, int ncols_a,
                              const double *taup,
                              double *bp, int ldbp,
                              int V, int nm)
{
    return dispatch<double>(trans, m, nrhs, k, ap, ldap, ncols_a, taup, bp, ldbp, V, nm);
}

extern "C" int sormqr_compact(char trans, int m, int nrhs, int k,
                              const float *ap, int ldap, int ncols_a,
                              const float *taup,
                              float *bp, int ldbp,
                              int V, int nm)
{
    return dispatch<float>(trans, m, nrhs, k, ap, ldap, ncols_a, taup, bp, ldbp, V, nm);
}
