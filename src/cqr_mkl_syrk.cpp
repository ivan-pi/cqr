/* cqr_mkl_syrk.cpp
 *
 * cqr_mkl_?syrk_compact (design doc section 8.1): a thin C-linkage adapter that
 * unwraps the MKL enums to bool flags and MKL_COMPACT_PACK to the interleave
 * width V, then forwards to cqr::detail::syrk_compact_general<T, MKL_INT> (on
 * MKL_INT so ILP64 dimensions are not narrowed).
 *
 * The Compact-format counterpart of BLAS ?syrk, completing the compact BLAS-3
 * set alongside mkl_?gemm_compact and mkl_?trsm_compact. Like the BLAS ?syrk it
 * batches -- and like the compact BLAS-3 routines it mirrors -- it takes no
 * workspace and reports no info, and (Compact convention) does not validate its
 * arguments: use the portable dsyrk_compact / ssyrk_compact (cqr_compact.h) for
 * LAPACK/BLAS-style info = -j checking. Semantics and compact storage layout are
 * documented in cqr_syrk_compact.hpp.
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#include "cqr_mkl_ext.h"
#include "cqr_syrk_compact.hpp"

namespace {

template <typename T>
void run(MKL_LAYOUT layout, MKL_UPLO uplo, MKL_TRANSPOSE trans, MKL_INT n, MKL_INT k,
         T alpha, const T *ap, MKL_INT ldap, T beta, T *cp, MKL_INT ldcp,
         MKL_COMPACT_PACK format, MKL_INT nm)
{
    /* Empty problem: no output to produce. n == 0 leaves nothing to update;
     * nm == 0 means no matrices (also keeps the kernel's nm >= 1 invariant).
     * k == 0 and alpha == 0 are NOT short-circuited: both still scale C by beta,
     * which the kernel performs (an empty contraction yields the zero update). */
    if (n == 0 || nm == 0) return;

    const bool rowmajor = (layout == MKL_ROW_MAJOR);
    const bool upper = (uplo == MKL_UPPER);
    /* real ?syrk: A^T only -- MKL_CONJTRANS folds to MKL_TRANS (conjugation is
     * ?herk). NOTRANS gives A*A^T, anything else A^T*A. */
    const bool tran = (trans != MKL_NOTRANS);

    /* Dispatch on the pack format; the interleave width V = (SIMD bytes)/sizeof(T)
     * is a compile-time constant in each case (SSE = 16 B, AVX = 32 B,
     * AVX512 = 64 B). An unrecognized format selects no kernel; with no info to
     * report that is a silent no-op (BLAS syrk convention). */
    switch (format) {
    case MKL_COMPACT_SSE:
        cqr::detail::syrk_compact_general<T, 16 / sizeof(T), MKL_INT>(
            upper, tran, rowmajor, n, k, alpha, ap, ldap, beta, cp, ldcp, nm);
        break;
    case MKL_COMPACT_AVX:
        cqr::detail::syrk_compact_general<T, 32 / sizeof(T), MKL_INT>(
            upper, tran, rowmajor, n, k, alpha, ap, ldap, beta, cp, ldcp, nm);
        break;
    case MKL_COMPACT_AVX512:
        cqr::detail::syrk_compact_general<T, 64 / sizeof(T), MKL_INT>(
            upper, tran, rowmajor, n, k, alpha, ap, ldap, beta, cp, ldcp, nm);
        break;
    default: break; /* unrecognized pack format: no kernel, no-op */
    }
}

} /* anonymous namespace */

extern "C" void cqr_mkl_dsyrk_compact(MKL_LAYOUT layout, MKL_UPLO uplo,
                                      MKL_TRANSPOSE trans, MKL_INT n, MKL_INT k,
                                      double alpha, const double *ap, MKL_INT ldap,
                                      double beta, double *cp, MKL_INT ldcp,
                                      MKL_COMPACT_PACK format, MKL_INT nm)
{
    run<double>(layout, uplo, trans, n, k, alpha, ap, ldap, beta, cp, ldcp, format, nm);
}

extern "C" void cqr_mkl_ssyrk_compact(MKL_LAYOUT layout, MKL_UPLO uplo,
                                      MKL_TRANSPOSE trans, MKL_INT n, MKL_INT k,
                                      float alpha, const float *ap, MKL_INT ldap,
                                      float beta, float *cp, MKL_INT ldcp,
                                      MKL_COMPACT_PACK format, MKL_INT nm)
{
    run<float>(layout, uplo, trans, n, k, alpha, ap, ldap, beta, cp, ldcp, format, nm);
}
