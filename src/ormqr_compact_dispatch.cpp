/* ormqr_compact.cpp
 *
 * extern "C" wrappers around the templated implementation; dispatch on
 * the runtime interleave width V to a compile-time instantiation.
 */

#include "ormqr_compact.hpp"
#include "ormqr_compact.h"

#include <cstdio>
#include <cstdlib>

namespace {

template <typename T>
void dispatch(char trans, int m, int nrhs, int k,
              const T *ap, int ldap, int ncols_a,
              const T *taup, T *bp, int ldbp,
              int V, int nm)
{
    switch (V) {
    case 2:  ormqr::ormqr_compact<T, 2>(trans, m, nrhs, k, ap, ldap, ncols_a, taup, bp, ldbp, nm); break;
    case 4:  ormqr::ormqr_compact<T, 4>(trans, m, nrhs, k, ap, ldap, ncols_a, taup, bp, ldbp, nm); break;
    case 8:  ormqr::ormqr_compact<T, 8>(trans, m, nrhs, k, ap, ldap, ncols_a, taup, bp, ldbp, nm); break;
    case 16: ormqr::ormqr_compact<T, 16>(trans, m, nrhs, k, ap, ldap, ncols_a, taup, bp, ldbp, nm); break;
    default:
        std::fprintf(stderr, "ormqr_compact: unsupported interleave width V=%d "
                             "(supported: 2, 4, 8, 16)\n", V);
        std::abort();
    }
}

} /* anonymous namespace */

extern "C" void dormqr_compact(char trans, int m, int nrhs, int k,
                               const double *ap, int ldap, int ncols_a,
                               const double *taup,
                               double *bp, int ldbp,
                               int V, int nm)
{
    dispatch<double>(trans, m, nrhs, k, ap, ldap, ncols_a, taup, bp, ldbp, V, nm);
}

extern "C" void sormqr_compact(char trans, int m, int nrhs, int k,
                               const float *ap, int ldap, int ncols_a,
                               const float *taup,
                               float *bp, int ldbp,
                               int V, int nm)
{
    dispatch<float>(trans, m, nrhs, k, ap, ldap, ncols_a, taup, bp, ldbp, V, nm);
}
