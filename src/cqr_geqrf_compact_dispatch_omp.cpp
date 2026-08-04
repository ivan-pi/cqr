/* cqr_geqrf_compact_dispatch_omp.cpp
 *
 * extern "C" wrappers around the OpenMP SIMD geqrf kernel
 * (cqr_geqrf_compact_omp.hpp). Identical surface, validation, and V-dispatch to
 * cqr_geqrf_compact_dispatch.cpp -- it defines the same dgeqrf_compact /
 * sgeqrf_compact symbols, only the backend kernel differs -- so it is a literal
 * drop-in replacement for the C entry points: link a test against this instead
 * of the vector-types dispatch and the very same buffers must come out equal.
 * The two are therefore never linked into one program (colliding C symbols).
 *
 * Build with -fopenmp-simd for the SIMD lowering; without it the kernel still
 * produces a correct factorization, just scalar (see the header).
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#include "cqr_compact.h"
#include "cqr_geqrf_compact_omp.hpp"

#include <cassert>

namespace {

/* Returns 0 on success, or -j if the j-th argument (1-based, in signature
 * order) had an illegal value -- the same LAPACK-style validation as the
 * vector-types dispatch. Pointer arguments are not inspected; never aborts. */
template <typename T>
int dispatch(char layout, int m, int n, T *ap, int ldap, T *taup, int V, int nm)
{
    const bool col = (layout == 'C' || layout == 'c');
    const bool row = (layout == 'R' || layout == 'r');
    const int ldmin = row ? (n < 1 ? 1 : n) : (m < 1 ? 1 : m); /* max(1, .) */

    if (!col && !row) return -1;
    if (m < 0) return -2;
    if (n < 0) return -3;
    if (ldap < ldmin) return -5;
    if (V != 2 && V != 4 && V != 8 && V != 16) return -7;
    if (nm < 0) return -8;

    if (m == 0 || n == 0 || nm == 0) return 0; /* empty problem: no-op */

    assert(ap != nullptr && taup != nullptr);

    switch (V) {
    case 2:
        cqr::detail::omp_simd::geqrf_compact_general_omp<T, 2>(row, m, n, ap, ldap, taup,
                                                               nm);
        break;
    case 4:
        cqr::detail::omp_simd::geqrf_compact_general_omp<T, 4>(row, m, n, ap, ldap, taup,
                                                               nm);
        break;
    case 8:
        cqr::detail::omp_simd::geqrf_compact_general_omp<T, 8>(row, m, n, ap, ldap, taup,
                                                               nm);
        break;
    case 16:
        cqr::detail::omp_simd::geqrf_compact_general_omp<T, 16>(row, m, n, ap, ldap, taup,
                                                                nm);
        break;
    }
    return 0;
}

} /* anonymous namespace */

extern "C" int dgeqrf_compact(char layout, int m, int n, double *ap, int ldap,
                              double *taup, int V, int nm)
{
    return dispatch<double>(layout, m, n, ap, ldap, taup, V, nm);
}

extern "C" int sgeqrf_compact(char layout, int m, int n, float *ap, int ldap, float *taup,
                              int V, int nm)
{
    return dispatch<float>(layout, m, n, ap, ldap, taup, V, nm);
}
