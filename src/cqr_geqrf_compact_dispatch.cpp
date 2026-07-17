/* cqr_geqrf_compact_dispatch.cpp
 *
 * extern "C" wrappers around the templated geqrf implementation; validate the
 * arguments LAPACK-style and dispatch on the runtime interleave width V to a
 * compile-time instantiation.
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#include "cqr_geqrf_compact.h"
#include "cqr_geqrf_compact.hpp"

#include <cassert>

namespace {

/* Returns 0 on success, or -j if the j-th argument (1-based, in signature
 * order) had an illegal value. Pointer arguments are not inspected, matching
 * LAPACK; never aborts the host process. */
template <typename T>
int dispatch(char layout, int m, int n,
             T *ap, int ldap, T *taup, int V, int nm)
{
    const bool col = (layout == 'C' || layout == 'c');
    const bool row = (layout == 'R' || layout == 'r');
    const int  ldmin = row ? (n < 1 ? 1 : n) : (m < 1 ? 1 : m);  /* max(1, .) */

    if (!col && !row)                           return -1;
    if (m < 0)                                  return -2;
    if (n < 0)                                  return -3;
    if (ldap < ldmin)                           return -5;
    if (V != 2 && V != 4 && V != 8 && V != 16)  return -7;
    if (nm < 0)                                 return -8;

    /* Nothing to compute for an empty problem (also keeps the kernel's
     * nm >= 1 invariant satisfied below). */
    if (m == 0 || n == 0 || nm == 0) return 0;

    /* Non-empty problem: the buffers are about to be dereferenced. LAPACK does
     * not inspect pointers, and neither does a release build, but a debug assert
     * catches an accidental null before it becomes a wild write. */
    assert(ap != nullptr && taup != nullptr);

    /* V is the compact interleave width, not necessarily one hardware register:
     * pack<T,V> is a GNU vector the compiler maps to registers or short unrolled
     * bursts, so every width is valid for both types (e.g. V=16 doubles is a
     * legal 1024-bit vector lowered to two AVX-512 ZMM ops). MKL's format -> V
     * mapping only ever selects 2/4/8 for double and 4/8/16 for float. */
    switch (V) {
    case 2:  cqr::detail::geqrf_compact_general<T, 2>(row, m, n, ap, ldap, taup, nm); break;
    case 4:  cqr::detail::geqrf_compact_general<T, 4>(row, m, n, ap, ldap, taup, nm); break;
    case 8:  cqr::detail::geqrf_compact_general<T, 8>(row, m, n, ap, ldap, taup, nm); break;
    case 16: cqr::detail::geqrf_compact_general<T, 16>(row, m, n, ap, ldap, taup, nm); break;
    }
    return 0;
}

} /* anonymous namespace */

extern "C" int dgeqrf_compact(char layout, int m, int n,
                              double *ap, int ldap, double *taup,
                              int V, int nm)
{
    return dispatch<double>(layout, m, n, ap, ldap, taup, V, nm);
}

extern "C" int sgeqrf_compact(char layout, int m, int n,
                              float *ap, int ldap, float *taup,
                              int V, int nm)
{
    return dispatch<float>(layout, m, n, ap, ldap, taup, V, nm);
}
