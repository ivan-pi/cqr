/* cqr_mkl_potrf.cpp
 *
 * Implementation of cqr_mkl_?potrf_compact (design document section 8.1):
 * a thin C-linkage adapter that
 *   1. unwraps MKL_COMPACT_PACK + the scalar type to the interleave width V,
 *   2. unwraps MKL_LAYOUT / MKL_UPLO to the kernel's rowmajor/upper flags,
 *   3. forwards to cqr::detail::potrf_compact_general<T,V>, instantiated on
 *      MKL_INT so ILP64 dimensions are not narrowed.
 *
 * Unlike ?geqrf, ?potrf needs no workspace, so -- like mkl_?potrf_compact --
 * there are no work/lwork arguments and no workspace query.
 *
 * Following the MKL Compact convention, the routine validates no arguments and
 * runs no positive-definiteness test: a non-SPD lane poisons itself with NaN/Inf
 * rather than reporting info = j (design section 6.2). info is a single scalar
 * status, 0 on success; the one value it can set is dispatch-level, info = -1 for
 * an unrecognized format.
 *
 * On exit the named triangle of each matrix holds its Cholesky factor (L for
 * MKL_LOWER, U for MKL_UPPER); the opposite triangle is left untouched. The
 * compact addressing is documented in cqr_potrf_compact.hpp.
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#include "cqr_mkl_ext.h"
#include "cqr_potrf_compact.hpp"

namespace {

/* Thin adapter shared by both precisions: format -> V, layout/uplo -> flags, and
 * a forward to the templated kernel. No argument validation and no SPD check
 * (MKL Compact convention); info is a single scalar status. */
template <typename T>
void run(MKL_LAYOUT layout, MKL_UPLO uplo, MKL_INT n, T *ap, MKL_INT ldap, MKL_INT *info,
         MKL_COMPACT_PACK format, MKL_INT nm)
{
    /* Empty problem: nothing to do (also keeps the kernel's nm >= 1 invariant). */
    if (n == 0 || nm == 0) {
        if (info) *info = 0;
        return;
    }

    const bool rowmajor = (layout == MKL_ROW_MAJOR);
    const bool upper = (uplo == MKL_UPPER);

    /* Dispatch on the pack format: it names the SIMD register width, so the
     * interleave width is that many bytes / sizeof(T) (SSE 16 B, AVX 32 B,
     * AVX-512 64 B -> FP64 2/4/8, FP32 4/8/16), a compile-time constant per case.
     * Instantiate on MKL_INT so 64-bit (ILP64) dimensions are not narrowed. */
    MKL_INT status = 0;
    switch (format) {
    case MKL_COMPACT_SSE:
        cqr::detail::potrf_compact_general<T, 16 / sizeof(T), MKL_INT>(rowmajor, upper, n,
                                                                       ap, ldap, nm);
        break;
    case MKL_COMPACT_AVX:
        cqr::detail::potrf_compact_general<T, 32 / sizeof(T), MKL_INT>(rowmajor, upper, n,
                                                                       ap, ldap, nm);
        break;
    case MKL_COMPACT_AVX512:
        cqr::detail::potrf_compact_general<T, 64 / sizeof(T), MKL_INT>(rowmajor, upper, n,
                                                                       ap, ldap, nm);
        break;
    default: status = -1; /* unrecognised pack format: cannot select a kernel */
    }
    if (info) *info = status;
}

} /* anonymous namespace */

extern "C" void cqr_mkl_dpotrf_compact(MKL_LAYOUT layout, MKL_UPLO uplo, MKL_INT n,
                                       double *ap, MKL_INT ldap, MKL_INT *info,
                                       MKL_COMPACT_PACK format, MKL_INT nm)
{
    run<double>(layout, uplo, n, ap, ldap, info, format, nm);
}

extern "C" void cqr_mkl_spotrf_compact(MKL_LAYOUT layout, MKL_UPLO uplo, MKL_INT n,
                                       float *ap, MKL_INT ldap, MKL_INT *info,
                                       MKL_COMPACT_PACK format, MKL_INT nm)
{
    run<float>(layout, uplo, n, ap, ldap, info, format, nm);
}
