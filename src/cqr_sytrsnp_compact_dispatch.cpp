/* cqr_sytrsnp_compact_dispatch.cpp
 *
 * extern "C" wrappers around the templated unpivoted-LDL^T solve; validate the
 * arguments LAPACK-style and dispatch on the runtime interleave width V to a
 * compile-time instantiation (cqr::detail::sytrsnp_compact_general).
 *
 * Assisted-by: Claude
 */

#include "cqr_compact.h"
#include "cqr_sytrsnp_compact.hpp"

#include <cassert>

namespace {

/* Returns 0 on success, or -j if the j-th argument (1-based, in signature
 * order) had an illegal value. The ap and bp pointers (arguments 5 and 7) are
 * not inspected, matching LAPACK; never aborts the host process. */
template <typename T>
int dispatch(char layout, char uplo, int n, int nrhs, const T *ap, int ldap, T *bp,
             int ldbp, int V, int nm)
{
    const bool col = (layout == 'C' || layout == 'c');
    const bool row = (layout == 'R' || layout == 'r');
    const bool lo = (uplo == 'L' || uplo == 'l');
    const bool up = (uplo == 'U' || uplo == 'u');
    const int ldamin = (n < 1 ? 1 : n); /* max(1, n); A square, layout-independent */
    const int ldbmin = row ? (nrhs < 1 ? 1 : nrhs) : (n < 1 ? 1 : n); /* max(1, n|nrhs) */

    if (!col && !row) return -1;
    if (!lo && !up) return -2;
    if (n < 0) return -3;
    if (nrhs < 0) return -4;
    if (ldap < ldamin) return -6;
    if (ldbp < ldbmin) return -8;
    if (V != 2 && V != 4 && V != 8 && V != 16) return -9;
    if (nm < 0) return -10;

    /* Nothing to solve for an empty problem (also keeps the kernel's nm >= 1
     * invariant satisfied below). */
    if (n == 0 || nrhs == 0 || nm == 0) return 0;

    /* Non-empty problem: the buffers are about to be dereferenced. LAPACK does
     * not inspect pointers, and neither does a release build, but a debug assert
     * catches an accidental null before it becomes a wild write. */
    assert(ap != nullptr && bp != nullptr);

    const bool rowmajor = row;
    const bool upper = up;

    switch (V) {
    case 2:
        cqr::detail::sytrsnp_compact_general<T, 2>(rowmajor, upper, n, nrhs, ap, ldap, bp,
                                                   ldbp, nm);
        break;
    case 4:
        cqr::detail::sytrsnp_compact_general<T, 4>(rowmajor, upper, n, nrhs, ap, ldap, bp,
                                                   ldbp, nm);
        break;
    case 8:
        cqr::detail::sytrsnp_compact_general<T, 8>(rowmajor, upper, n, nrhs, ap, ldap, bp,
                                                   ldbp, nm);
        break;
    case 16:
        cqr::detail::sytrsnp_compact_general<T, 16>(rowmajor, upper, n, nrhs, ap, ldap,
                                                    bp, ldbp, nm);
        break;
    }
    return 0;
}

} /* anonymous namespace */

extern "C" int dsytrsnp_compact(char layout, char uplo, int n, int nrhs, const double *ap,
                                int ldap, double *bp, int ldbp, int V, int nm)
{
    return dispatch<double>(layout, uplo, n, nrhs, ap, ldap, bp, ldbp, V, nm);
}

extern "C" int ssytrsnp_compact(char layout, char uplo, int n, int nrhs, const float *ap,
                                int ldap, float *bp, int ldbp, int V, int nm)
{
    return dispatch<float>(layout, uplo, n, nrhs, ap, ldap, bp, ldbp, V, nm);
}
