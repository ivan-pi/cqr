/* cqr_syrk_compact_dispatch.cpp
 *
 * extern "C" wrappers around the templated compact symmetric rank-k update;
 * validate the arguments LAPACK/BLAS-style and dispatch on the runtime interleave
 * width V to a compile-time instantiation (cqr::detail::syrk_compact_general).
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#include "cqr_compact.h"
#include "cqr_syrk_compact.hpp"

#include <cassert>

namespace {

/* Validate the arguments LAPACK/BLAS-style and dispatch on the interleave width
 * V. Returns 0 on success, or -j if the j-th argument (1-based, in signature
 * order) had an illegal value. Scalar (alpha, beta) and pointer (ap, cp)
 * arguments are not inspected, matching LAPACK/BLAS; never aborts the host
 * process. */
template <typename T>
int dispatch(char layout, char uplo, char trans, int n, int k, T alpha, const T *ap,
             int ldap, T beta, T *cp, int ldcp, int V, int nm)
{
    const bool col = (layout == 'C' || layout == 'c');
    const bool row = (layout == 'R' || layout == 'r');
    const bool up = (uplo == 'U' || uplo == 'u');
    const bool lo = (uplo == 'L' || uplo == 'l');
    const bool trans_ok = (trans == 'N' || trans == 'n' || trans == 'T' || trans == 't' ||
                           trans == 'C' || trans == 'c');
    const bool tran = (trans == 'T' || trans == 't' || trans == 'C' || trans == 'c');

    /* A is n x k (trans='N') or k x n (trans='T'); the leading dim bounds its
     * stored-axis extent -- rows for column-major, columns for row-major. */
    const int arows = tran ? k : n;
    const int acols = tran ? n : k;
    const int ldamin = row ? (acols < 1 ? 1 : acols) : (arows < 1 ? 1 : arows);
    const int ldcmin = (n < 1 ? 1 : n); /* C is n x n, layout-independent */

    if (!col && !row) return -1;
    if (!up && !lo) return -2;
    if (!trans_ok) return -3;
    if (n < 0) return -4;
    if (k < 0) return -5;
    /* -6 alpha, -9 beta: scalars; every value (including 0) is legal. */
    if (ldap < ldamin) return -8;
    if (ldcp < ldcmin) return -11;
    if (V != 2 && V != 4 && V != 8 && V != 16) return -12;
    if (nm < 0) return -13;

    /* Nothing to produce for an empty problem (also keeps the kernel's nm >= 1
     * invariant satisfied below). k == 0 is NOT empty: it means C := beta*C (an
     * empty contraction yields the zero update), so it flows to the kernel like
     * any other value, as does alpha == 0. */
    if (n == 0 || nm == 0) return 0;

    /* Non-empty problem: the buffers are about to be dereferenced. LAPACK does
     * not inspect pointers, and neither does a release build, but a debug assert
     * catches an accidental null before it becomes a wild write. */
    assert(ap != nullptr && cp != nullptr);

    const bool rowmajor = row;
    const bool upper = up;

    /* V is the compact interleave width, not necessarily one hardware register:
     * pack<T,V> is a GNU vector the compiler maps to registers or short unrolled
     * bursts, so every width is valid for both types. MKL's format -> V mapping
     * only ever selects 2/4/8 for double and 4/8/16 for float. */
    switch (V) {
    case 2:
        cqr::detail::syrk_compact_general<T, 2>(upper, tran, rowmajor, n, k, alpha, ap,
                                                ldap, beta, cp, ldcp, nm);
        break;
    case 4:
        cqr::detail::syrk_compact_general<T, 4>(upper, tran, rowmajor, n, k, alpha, ap,
                                                ldap, beta, cp, ldcp, nm);
        break;
    case 8:
        cqr::detail::syrk_compact_general<T, 8>(upper, tran, rowmajor, n, k, alpha, ap,
                                                ldap, beta, cp, ldcp, nm);
        break;
    case 16:
        cqr::detail::syrk_compact_general<T, 16>(upper, tran, rowmajor, n, k, alpha, ap,
                                                 ldap, beta, cp, ldcp, nm);
        break;
    }
    return 0;
}

} /* anonymous namespace */

extern "C" int dsyrk_compact(char layout, char uplo, char trans, int n, int k,
                             double alpha, const double *ap, int ldap, double beta,
                             double *cp, int ldcp, int V, int nm)
{
    return dispatch<double>(layout, uplo, trans, n, k, alpha, ap, ldap, beta, cp, ldcp, V,
                            nm);
}

extern "C" int ssyrk_compact(char layout, char uplo, char trans, int n, int k,
                             float alpha, const float *ap, int ldap, float beta,
                             float *cp, int ldcp, int V, int nm)
{
    return dispatch<float>(layout, uplo, trans, n, k, alpha, ap, ldap, beta, cp, ldcp, V,
                           nm);
}
