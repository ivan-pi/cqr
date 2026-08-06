/* cqr_mkl_trsm.cpp
 *
 * cqr_mkl_?trsm_compact (design doc section 8.1): a thin C-linkage adapter that
 * unwraps the MKL enums to bool flags and MKL_COMPACT_PACK to the interleave
 * width V, then forwards to cqr::detail::trsm_compact_general<T, MKL_INT> (on
 * MKL_INT so ILP64 dimensions are not narrowed).
 *
 * Mirrors mkl_?trsm_compact exactly: like the BLAS ?trsm it batches, it takes no
 * workspace and reports no info, and (Compact convention) does not validate its
 * arguments -- use the portable dtrsm_compact / strsm_compact (cqr_compact.h) for
 * LAPACK/BLAS-style info = -j checking. Semantics and compact storage layout are
 * documented in cqr_trsm_compact.hpp.
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#include "cqr_mkl_ext.h"
#include "cqr_trsm_compact.hpp"

namespace {

template <typename T>
void run(MKL_LAYOUT layout, MKL_SIDE side, MKL_UPLO uplo, MKL_TRANSPOSE transa,
         MKL_DIAG diag, MKL_INT m, MKL_INT n, T alpha, const T *ap, MKL_INT ldap, T *bp,
         MKL_INT ldbp, MKL_COMPACT_PACK format, MKL_INT nm)
{
    /* alpha == 0 is NOT empty -- it means B := 0 -- so it flows to the kernel. */
    if (m == 0 || n == 0 || nm == 0) return;

    const bool rowmajor = (layout == MKL_ROW_MAJOR);
    const bool left = (side == MKL_LEFT);
    const bool upper = (uplo == MKL_UPPER);
    const bool tran = (transa != MKL_NOTRANS); /* MKL_CONJTRANS folds to T (real) */
    const bool unit = (diag == MKL_UNIT);

    /* Dispatch on the pack format; the interleave width V = (SIMD bytes)/sizeof(T)
     * is a compile-time constant in each case (SSE = 16 B, AVX = 32 B,
     * AVX512 = 64 B). An unrecognized format selects no kernel; with no info to
     * report that is a silent no-op (BLAS trsm convention). */
    switch (format) {
    case MKL_COMPACT_SSE:
        cqr::detail::trsm_compact_general<T, 16 / sizeof(T), MKL_INT>(
            left, upper, rowmajor, tran, unit, m, n, alpha, ap, ldap, bp, ldbp, nm);
        break;
    case MKL_COMPACT_AVX:
        cqr::detail::trsm_compact_general<T, 32 / sizeof(T), MKL_INT>(
            left, upper, rowmajor, tran, unit, m, n, alpha, ap, ldap, bp, ldbp, nm);
        break;
    case MKL_COMPACT_AVX512:
        cqr::detail::trsm_compact_general<T, 64 / sizeof(T), MKL_INT>(
            left, upper, rowmajor, tran, unit, m, n, alpha, ap, ldap, bp, ldbp, nm);
        break;
    default: break; /* unrecognized pack format: no kernel, no-op */
    }
}

} /* anonymous namespace */

extern "C" void cqr_mkl_dtrsm_compact(MKL_LAYOUT layout, MKL_SIDE side, MKL_UPLO uplo,
                                      MKL_TRANSPOSE transa, MKL_DIAG diag, MKL_INT m,
                                      MKL_INT n, double alpha, const double *ap,
                                      MKL_INT ldap, double *bp, MKL_INT ldbp,
                                      MKL_COMPACT_PACK format, MKL_INT nm)
{
    run<double>(layout, side, uplo, transa, diag, m, n, alpha, ap, ldap, bp, ldbp, format,
                nm);
}

extern "C" void cqr_mkl_strsm_compact(MKL_LAYOUT layout, MKL_SIDE side, MKL_UPLO uplo,
                                      MKL_TRANSPOSE transa, MKL_DIAG diag, MKL_INT m,
                                      MKL_INT n, float alpha, const float *ap,
                                      MKL_INT ldap, float *bp, MKL_INT ldbp,
                                      MKL_COMPACT_PACK format, MKL_INT nm)
{
    run<float>(layout, side, uplo, transa, diag, m, n, alpha, ap, ldap, bp, ldbp, format,
               nm);
}
