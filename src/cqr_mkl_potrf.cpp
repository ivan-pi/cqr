/* cqr_mkl_potrf.cpp
 *
 * Implementation of cqr_mkl_?potrf_compact (design document section 8.1):
 * a thin C-linkage adapter that
 *   1. unwraps MKL_COMPACT_PACK + the scalar type to the interleave width V,
 *   2. unwraps MKL_LAYOUT / MKL_UPLO to the kernel's rowmajor/upper flags,
 *   3. forwards to the templated kernel cqr::detail::potrf_compact_general<T,V>,
 *      instantiated on MKL_INT so ILP64 dimensions are not narrowed.
 *
 * Unlike ?geqrf, ?potrf needs no workspace, so -- like mkl_?potrf_compact --
 * there are no work/lwork arguments and no workspace query.
 *
 * Following the MKL Compact convention, the routine does NOT validate its
 * arguments: compact routines skip error checking for vectorization and make
 * the caller responsible for passing consistent parameters (see "Numerical
 * Limitations for Compact BLAS and Compact LAPACK Routines"). It also performs
 * no positive-definiteness test -- a non-SPD lane poisons itself with NaN/Inf
 * rather than reporting info = j (design section 6.2). MKL likewise leaves the
 * compact `info` reserved; we write it as a single scalar status (0 on success).
 * The one value it can set is dispatch-level: an unrecognized `format` selects
 * no kernel and sets info = -1.
 *
 * On exit the named triangle of each matrix in ap holds its Cholesky factor
 * (L for MKL_LOWER, U for MKL_UPPER); the strictly-opposite triangle is left
 * untouched -- exactly the mkl_?potrf_compact / LAPACK ?potrf convention.
 *
 * Compact-format addressing (matches mkl_?gepack_compact); group g = idx/V,
 * slot v = idx%V. Column-major (the addressing the tuned kernel sweeps natively):
 *   A_v(i,j) = ap[ g*ldap*n*V + (j*ldap + i)*V + v ]
 * Row-major swaps the in-matrix index roles (i*ldap + j); the group stride
 * ldap*n*V is the same either way (A is n x n).
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#include "cqr_mkl_ext.h"
#include "cqr_potrf_compact.hpp"

namespace {

using cqr::detail::vlen_for_format;

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

    MKL_INT status = 0;
    /* Instantiate on MKL_INT so 64-bit (ILP64) dimensions are not narrowed. */
    switch (vlen_for_format<T>(format)) {
    case 2:
        cqr::detail::potrf_compact_general<T, 2, MKL_INT>(rowmajor, upper, n, ap, ldap,
                                                          nm);
        break;
    case 4:
        cqr::detail::potrf_compact_general<T, 4, MKL_INT>(rowmajor, upper, n, ap, ldap,
                                                          nm);
        break;
    case 8:
        cqr::detail::potrf_compact_general<T, 8, MKL_INT>(rowmajor, upper, n, ap, ldap,
                                                          nm);
        break;
    case 16:
        cqr::detail::potrf_compact_general<T, 16, MKL_INT>(rowmajor, upper, n, ap, ldap,
                                                           nm);
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
