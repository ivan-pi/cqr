/* cqr_mkl_ext.cpp
 *
 * Implementation of cqr_mkl_?ormqr_compact (design document section 8.1):
 * a thin C-linkage adapter that
 *   1. handles the lwork = -1 workspace query (the kernel needs no scratch),
 *   2. unwraps MKL_COMPACT_PACK + the scalar type to the interleave width V,
 *   3. forwards to the templated kernel cqr::detail::ormqr_compact_general<T,V>,
 *      instantiated on MKL_INT so ILP64 dimensions are not narrowed.
 *
 * Following the MKL Compact convention, the routine does NOT validate its
 * arguments: compact routines skip error checking for vectorization and make
 * the caller responsible for passing consistent parameters (see "Numerical
 * Limitations for Compact BLAS and Compact LAPACK Routines"). MKL likewise
 * leaves the compact `info` reserved; we write it as a single scalar status
 * (0 on success), not a per-matrix array.
 *
 * Supports side in {L,R} and layout in {MKL_COL_MAJOR, MKL_ROW_MAJOR} for
 * trans in {N,T} (C is folded to T for real types). A is the (s x k) reflector
 * batch, s = m (side='L') or n (side='R'); C is the m x n batch.
 *
 * Compact-format addressing (matches mkl_?gepack_compact); group g = idx/V,
 * slot v = idx%V. Column-major (the addressing the kernel sweeps natively):
 *   A_v(i,j)  = ap [ g*ldap*k*V    + (j*ldap + i)*V + v ]
 *   tau_v(kk) = taup[ g*k*V        +  kk*V          + v ]
 *   C_v(i,j)  = cp [ g*ldcp*n*V    + (j*ldcp + i)*V + v ]
 * Row-major swaps the in-matrix index roles (i*ld + j) and the group stride's
 * complementary extent (ldap*s for A, ldcp*m for C).
 */

#include "cqr_mkl_ext.h"
#include "cqr_compact.hpp"

namespace {

using cqr::detail::vlen_for_format;

/* Thin adapter shared by both precisions: workspace query, format -> V, and a
 * forward to the templated kernel. No argument validation (MKL Compact
 * convention); info is a single scalar status. */
template <typename T>
void run(MKL_LAYOUT layout, char side, char trans,
         MKL_INT m, MKL_INT n, MKL_INT k,
         const T *ap, MKL_INT ldap, const T *taup,
         T *cp, MKL_INT ldcp, T *work, MKL_INT lwork, MKL_INT *info,
         MKL_COMPACT_PACK format, MKL_INT nm)
{
    /* Workspace query: the branch-free kernel needs no scratch, so the
     * optimal (and minimum) lwork is 1. */
    if (lwork == -1) {
        if (work) work[0] = T(1);
        if (info)  *info  = 0;
        return;
    }

    /* Empty problem: nothing to do (also keeps the kernel's nm >= 1 invariant). */
    if (m == 0 || n == 0 || k == 0 || nm == 0) {
        if (info) *info = 0;
        return;
    }

    const bool rowmajor = (layout == MKL_ROW_MAJOR);
    const bool left     = (side == 'L' || side == 'l');
    const bool tran     = (trans == 'T' || trans == 't' ||
                           trans == 'C' || trans == 'c');
    const char tr       = tran ? 'T' : 'N';

    MKL_INT status = 0;
    /* Instantiate on MKL_INT so 64-bit (ILP64) dimensions are not narrowed. */
    switch (vlen_for_format<T>(format)) {
    case 2:  cqr::detail::ormqr_compact_general<T, 2, MKL_INT>(left, rowmajor, tr, m, n, k, ap, ldap, k, taup, cp, ldcp, nm); break;
    case 4:  cqr::detail::ormqr_compact_general<T, 4, MKL_INT>(left, rowmajor, tr, m, n, k, ap, ldap, k, taup, cp, ldcp, nm); break;
    case 8:  cqr::detail::ormqr_compact_general<T, 8, MKL_INT>(left, rowmajor, tr, m, n, k, ap, ldap, k, taup, cp, ldcp, nm); break;
    case 16: cqr::detail::ormqr_compact_general<T, 16, MKL_INT>(left, rowmajor, tr, m, n, k, ap, ldap, k, taup, cp, ldcp, nm); break;
    default: status = -1;   /* unrecognised pack format: cannot select a kernel */
    }
    if (info) *info = status;
}

} /* anonymous namespace */

extern "C" void cqr_mkl_dormqr_compact(MKL_LAYOUT layout, char side, char trans,
                                       MKL_INT m, MKL_INT n, MKL_INT k,
                                       const double *ap, MKL_INT ldap,
                                       const double *taup,
                                       double *cp, MKL_INT ldcp,
                                       double *work, MKL_INT lwork, MKL_INT *info,
                                       MKL_COMPACT_PACK format, MKL_INT nm)
{
    run<double>(layout, side, trans, m, n, k, ap, ldap, taup, cp, ldcp,
                work, lwork, info, format, nm);
}

extern "C" void cqr_mkl_sormqr_compact(MKL_LAYOUT layout, char side, char trans,
                                       MKL_INT m, MKL_INT n, MKL_INT k,
                                       const float *ap, MKL_INT ldap,
                                       const float *taup,
                                       float *cp, MKL_INT ldcp,
                                       float *work, MKL_INT lwork, MKL_INT *info,
                                       MKL_COMPACT_PACK format, MKL_INT nm)
{
    run<float>(layout, side, trans, m, n, k, ap, ldap, taup, cp, ldcp,
               work, lwork, info, format, nm);
}
