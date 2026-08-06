/* cqr_potrf_compact_dispatch.cpp
 *
 * extern "C" wrappers around the templated potrf implementation; validate the
 * arguments LAPACK-style and dispatch on the runtime interleave width V to a
 * compile-time instantiation.
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#include "cqr_compact.h"
#include "cqr_potrf_compact.hpp"

#include <cassert>

namespace {

/* Returns 0 on success, or -j if the j-th argument (1-based, in signature
 * order) had an illegal value. The ap pointer (argument 4) is not inspected,
 * matching LAPACK; never aborts the host process. */
template <typename T>
int dispatch(char layout, char uplo, int n, T *ap, int ldap, int V, int nm)
{
    const bool col = (layout == 'C' || layout == 'c');
    const bool row = (layout == 'R' || layout == 'r');
    const bool lo = (uplo == 'L' || uplo == 'l');
    const bool up = (uplo == 'U' || uplo == 'u');
    const int ldmin = (n < 1 ? 1 : n); /* max(1, n); square, layout-independent */

    if (!col && !row) return -1;
    if (!lo && !up) return -2;
    if (n < 0) return -3;
    if (ldap < ldmin) return -5;
    if (V != 2 && V != 4 && V != 8 && V != 16) return -6;
    if (nm < 0) return -7;

    /* Nothing to compute for an empty problem (also keeps the kernel's
     * nm >= 1 invariant satisfied below). */
    if (n == 0 || nm == 0) return 0;

    /* Non-empty problem: the buffer is about to be dereferenced. LAPACK does not
     * inspect pointers, and neither does a release build, but a debug assert
     * catches an accidental null before it becomes a wild write. */
    assert(ap != nullptr);

    const bool rowmajor = row;
    const bool upper = up;

    /* V is the compact interleave width, not necessarily one hardware register:
     * pack<T,V> is a GNU vector the compiler maps to registers or short unrolled
     * bursts, so every width is valid for both types (e.g. V=16 doubles is a
     * legal 1024-bit vector lowered to two AVX-512 ZMM ops). MKL's format -> V
     * mapping only ever selects 2/4/8 for double and 4/8/16 for float. */
    switch (V) {
    case 2:
        cqr::detail::potrf_compact_general<T, 2>(rowmajor, upper, n, ap, ldap, nm);
        break;
    case 4:
        cqr::detail::potrf_compact_general<T, 4>(rowmajor, upper, n, ap, ldap, nm);
        break;
    case 8:
        cqr::detail::potrf_compact_general<T, 8>(rowmajor, upper, n, ap, ldap, nm);
        break;
    case 16:
        cqr::detail::potrf_compact_general<T, 16>(rowmajor, upper, n, ap, ldap, nm);
        break;
    }
    return 0;
}

} /* anonymous namespace */

extern "C" int dpotrf_compact(char layout, char uplo, int n, double *ap, int ldap, int V,
                              int nm)
{
    return dispatch<double>(layout, uplo, n, ap, ldap, V, nm);
}

extern "C" int spotrf_compact(char layout, char uplo, int n, float *ap, int ldap, int V,
                              int nm)
{
    return dispatch<float>(layout, uplo, n, ap, ldap, V, nm);
}
