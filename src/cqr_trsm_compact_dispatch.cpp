/* cqr_trsm_compact_dispatch.cpp
 *
 * extern "C" wrappers around the templated compact triangular solve; validate
 * the arguments LAPACK-style and dispatch on the runtime interleave width V to
 * a compile-time instantiation (cqr::detail::trsm_compact_general).
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#include "cqr_compact.h"
#include "cqr_trsm_compact.hpp"

#include <cassert>

namespace {

/* Validate the arguments LAPACK-style and dispatch on the interleave width V.
 * Returns 0 on success, or -j if the j-th argument (1-based, in signature
 * order) had an illegal value. Pointer and scalar (alpha) arguments are not
 * inspected, matching LAPACK/BLAS; never aborts the host process. */
template <typename T>
int dispatch(char layout, char side, char uplo, char transa, char diag, int m, int n,
             T alpha, const T *ap, int ldap, T *bp, int ldbp, int V, int nm)
{
    const bool col = (layout == 'C' || layout == 'c');
    const bool row = (layout == 'R' || layout == 'r');
    const bool left = (side == 'L' || side == 'l');
    const bool right = (side == 'R' || side == 'r');
    const bool up = (uplo == 'U' || uplo == 'u');
    const bool lo = (uplo == 'L' || uplo == 'l');
    const bool trans_ok = (transa == 'N' || transa == 'n' || transa == 'T' ||
                           transa == 't' || transa == 'C' || transa == 'c');
    const bool diag_ok = (diag == 'U' || diag == 'u' || diag == 'N' || diag == 'n');

    /* A is the order-s triangular factor: s = m (left) or n (right). */
    const int s = left ? m : n;
    const int ldamin = (s < 1 ? 1 : s);                         /* max(1, s)          */
    const int ldbmin = row ? (n < 1 ? 1 : n) : (m < 1 ? 1 : m); /* max(1, m|n) */

    if (!col && !row) return -1;
    if (!left && !right) return -2;
    if (!up && !lo) return -3;
    if (!trans_ok) return -4;
    if (!diag_ok) return -5;
    if (m < 0) return -6;
    if (n < 0) return -7;
    /* -8 alpha: a scalar; every value (including 0) is legal, so never checked. */
    if (ldap < ldamin) return -10;
    if (ldbp < ldbmin) return -12;
    if (V != 2 && V != 4 && V != 8 && V != 16) return -13;
    if (nm < 0) return -14;

    /* Nothing to solve for an empty problem (also keeps the kernel's nm >= 1
     * invariant satisfied below). alpha == 0 is NOT empty: it means B := 0, so
     * it flows to the kernel like any other value. */
    if (m == 0 || n == 0 || nm == 0) return 0;

    /* Non-empty problem: the buffers are about to be dereferenced. LAPACK does
     * not inspect pointers, and neither does a release build, but a debug assert
     * catches an accidental null before it becomes a wild write. */
    assert(ap != nullptr && bp != nullptr);

    const bool rowmajor = row;
    const bool upper = up;
    const bool tran = (transa == 'T' || transa == 't' || transa == 'C' || transa == 'c');
    const bool unit = (diag == 'U' || diag == 'u');

    switch (V) {
    case 2:
        cqr::detail::trsm_compact_general<T, 2>(left, upper, rowmajor, tran, unit, m, n,
                                                alpha, ap, ldap, bp, ldbp, nm);
        break;
    case 4:
        cqr::detail::trsm_compact_general<T, 4>(left, upper, rowmajor, tran, unit, m, n,
                                                alpha, ap, ldap, bp, ldbp, nm);
        break;
    case 8:
        cqr::detail::trsm_compact_general<T, 8>(left, upper, rowmajor, tran, unit, m, n,
                                                alpha, ap, ldap, bp, ldbp, nm);
        break;
    case 16:
        cqr::detail::trsm_compact_general<T, 16>(left, upper, rowmajor, tran, unit, m, n,
                                                 alpha, ap, ldap, bp, ldbp, nm);
        break;
    }
    return 0;
}

} /* anonymous namespace */

extern "C" int dtrsm_compact(char layout, char side, char uplo, char transa, char diag,
                             int m, int n, double alpha, const double *ap, int ldap,
                             double *bp, int ldbp, int V, int nm)
{
    return dispatch<double>(layout, side, uplo, transa, diag, m, n, alpha, ap, ldap, bp,
                            ldbp, V, nm);
}

extern "C" int strsm_compact(char layout, char side, char uplo, char transa, char diag,
                             int m, int n, float alpha, const float *ap, int ldap,
                             float *bp, int ldbp, int V, int nm)
{
    return dispatch<float>(layout, side, uplo, transa, diag, m, n, alpha, ap, ldap, bp,
                           ldbp, V, nm);
}
