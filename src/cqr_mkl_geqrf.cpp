/* cqr_mkl_geqrf.cpp
 *
 * Implementation of cqr_mkl_?geqrf_compact (design document section 8.1):
 * a thin C-linkage adapter that
 *   1. handles the lwork = -1 workspace query (the kernel needs no scratch),
 *   2. unwraps MKL_COMPACT_PACK + the scalar type to the interleave width V,
 *   3. forwards to the templated kernel cqr::detail::geqrf_compact_general<T,V>,
 *      instantiated on MKL_INT so ILP64 dimensions are not narrowed.
 *
 * Following the MKL Compact convention, the routine does NOT validate its
 * arguments: compact routines skip error checking for vectorization and make
 * the caller responsible for passing consistent parameters (see "Numerical
 * Limitations for Compact BLAS and Compact LAPACK Routines"). MKL likewise
 * leaves the compact `info` reserved; we write it as a single scalar status
 * (0 on success), not a per-matrix array.
 *
 * On exit each matrix in ap holds R (on/above the diagonal) and the Householder
 * vectors (below); taup holds the k = min(m,n) reflector scalars per matrix --
 * exactly the mkl_?geqrf_compact / LAPACK ?geqrf storage convention.
 *
 * Compact-format addressing (matches mkl_?gepack_compact); group g = idx/V,
 * slot v = idx%V. Column-major (the addressing the kernel sweeps natively):
 *   A_v(i,j)  = ap [ g*ldap*n*V + (j*ldap + i)*V + v ]
 *   tau_v(kk) = taup[ g*k*V      +  kk*V          + v ]
 * Row-major swaps the in-matrix index roles (i*ldap + j) and the group stride's
 * complementary extent (ldap*m).
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#include "cqr_mkl_ext.h"
#include "cqr_geqrf_compact.hpp"

namespace {

/* Thin adapter shared by both precisions: workspace query, format -> V, and a
 * forward to the templated kernel. No argument validation (MKL Compact
 * convention); info is a single scalar status. */
template <typename T>
void run(MKL_LAYOUT layout, MKL_INT m, MKL_INT n, T *ap, MKL_INT ldap, T *taup, T *work,
         MKL_INT lwork, MKL_INT *info, MKL_COMPACT_PACK format, MKL_INT nm)
{
    /* Workspace query: the unblocked kernel needs no scratch, so the optimal
     * (and minimum) lwork is 1. */
    if (lwork == -1) {
        if (work) work[0] = T(1);
        if (info) *info = 0;
        return;
    }

    /* Empty problem: nothing to do (also keeps the kernel's nm >= 1 invariant). */
    if (m == 0 || n == 0 || nm == 0) {
        if (info) *info = 0;
        return;
    }

    const bool rowmajor = (layout == MKL_ROW_MAJOR);

    /* Dispatch on the pack format: it names the SIMD register width, so the
     * interleave width is that many bytes / sizeof(T) (SSE 16 B, AVX 32 B,
     * AVX-512 64 B -> FP64 2/4/8, FP32 4/8/16), a compile-time constant per case.
     * Instantiate on MKL_INT so 64-bit (ILP64) dimensions are not narrowed. */
    MKL_INT status = 0;
    switch (format) {
    case MKL_COMPACT_SSE:
        cqr::detail::geqrf_compact_general<T, 16 / sizeof(T), MKL_INT>(rowmajor, m, n, ap,
                                                                       ldap, taup, nm);
        break;
    case MKL_COMPACT_AVX:
        cqr::detail::geqrf_compact_general<T, 32 / sizeof(T), MKL_INT>(rowmajor, m, n, ap,
                                                                       ldap, taup, nm);
        break;
    case MKL_COMPACT_AVX512:
        cqr::detail::geqrf_compact_general<T, 64 / sizeof(T), MKL_INT>(rowmajor, m, n, ap,
                                                                       ldap, taup, nm);
        break;
    default: status = -1; /* unrecognised pack format: cannot select a kernel */
    }
    if (info) *info = status;
}

} /* anonymous namespace */

extern "C" void cqr_mkl_dgeqrf_compact(MKL_LAYOUT layout, MKL_INT m, MKL_INT n,
                                       double *ap, MKL_INT ldap, double *taup,
                                       double *work, MKL_INT lwork, MKL_INT *info,
                                       MKL_COMPACT_PACK format, MKL_INT nm)
{
    run<double>(layout, m, n, ap, ldap, taup, work, lwork, info, format, nm);
}

extern "C" void cqr_mkl_sgeqrf_compact(MKL_LAYOUT layout, MKL_INT m, MKL_INT n, float *ap,
                                       MKL_INT ldap, float *taup, float *work,
                                       MKL_INT lwork, MKL_INT *info,
                                       MKL_COMPACT_PACK format, MKL_INT nm)
{
    run<float>(layout, m, n, ap, ldap, taup, work, lwork, info, format, nm);
}
