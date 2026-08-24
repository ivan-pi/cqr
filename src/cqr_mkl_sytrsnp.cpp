/* cqr_mkl_sytrsnp.cpp
 *
 * Implementation of cqr_mkl_?sytrsnp_compact (design document section 8.1): the
 * solve companion of cqr_mkl_?sytrfnp_compact. A thin C-linkage adapter that
 * unwraps MKL_COMPACT_PACK to the interleave width V and MKL_LAYOUT / MKL_UPLO
 * to flags, then forwards to cqr::detail::sytrsnp_compact_general<T,V>
 * (instantiated on MKL_INT so ILP64 dimensions are not narrowed), which runs
 * the three in-place sweeps
 *
 *     L z = B;  D w = z;  L^T X = w      (MKL_LOWER, A = L D L^T)
 *     U^T z = B;  D w = z;  U X = w      (MKL_UPPER, A = U^T D U)
 *
 * over the compact batch. No workspace. Following the MKL Compact convention
 * the routine validates no arguments; a zero D(i) (singular lane) yields
 * Inf/NaN in that lane's solution, not an error. info is a single scalar
 * status, 0 on success, -1 for an unrecognized format.
 *
 * Assisted-by: Claude
 */

#include "cqr_mkl_ext.h"
#include "cqr_sytrsnp_compact.hpp"

namespace {

template <typename T>
void run(MKL_LAYOUT layout, MKL_UPLO uplo, MKL_INT n, MKL_INT nrhs, const T *ap,
         MKL_INT ldap, T *bp, MKL_INT ldbp, MKL_INT *info, MKL_COMPACT_PACK format,
         MKL_INT nm)
{
    /* Empty problem: nothing to do (also keeps the kernel's nm >= 1 invariant). */
    if (n == 0 || nrhs == 0 || nm == 0) {
        if (info) *info = 0;
        return;
    }

    const bool rowmajor = (layout == MKL_ROW_MAJOR);
    const bool upper = (uplo == MKL_UPPER);

    MKL_INT status = 0;
    switch (format) {
    case MKL_COMPACT_SSE:
        cqr::detail::sytrsnp_compact_general<T, 16 / sizeof(T), MKL_INT>(
            rowmajor, upper, n, nrhs, ap, ldap, bp, ldbp, nm);
        break;
    case MKL_COMPACT_AVX:
        cqr::detail::sytrsnp_compact_general<T, 32 / sizeof(T), MKL_INT>(
            rowmajor, upper, n, nrhs, ap, ldap, bp, ldbp, nm);
        break;
    case MKL_COMPACT_AVX512:
        cqr::detail::sytrsnp_compact_general<T, 64 / sizeof(T), MKL_INT>(
            rowmajor, upper, n, nrhs, ap, ldap, bp, ldbp, nm);
        break;
    default: status = -1; /* unrecognised pack format: cannot select a kernel */
    }
    if (info) *info = status;
}

} /* anonymous namespace */

extern "C" void cqr_mkl_dsytrsnp_compact(MKL_LAYOUT layout, MKL_UPLO uplo, MKL_INT n,
                                         MKL_INT nrhs, const double *ap, MKL_INT ldap,
                                         double *bp, MKL_INT ldbp, MKL_INT *info,
                                         MKL_COMPACT_PACK format, MKL_INT nm)
{
    run<double>(layout, uplo, n, nrhs, ap, ldap, bp, ldbp, info, format, nm);
}

extern "C" void cqr_mkl_ssytrsnp_compact(MKL_LAYOUT layout, MKL_UPLO uplo, MKL_INT n,
                                         MKL_INT nrhs, const float *ap, MKL_INT ldap,
                                         float *bp, MKL_INT ldbp, MKL_INT *info,
                                         MKL_COMPACT_PACK format, MKL_INT nm)
{
    run<float>(layout, uplo, n, nrhs, ap, ldap, bp, ldbp, info, format, nm);
}
