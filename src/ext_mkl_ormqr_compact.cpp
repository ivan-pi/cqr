/* ext_mkl_ormqr_compact.cpp
 *
 * Implementation of ext_mkl_?ormqr_compact (design document section 8.1):
 * a C-linkage dispatcher that
 *   1. validates the arguments and reports illegal values through info[],
 *   2. handles the lwork = -1 workspace query,
 *   3. unwraps MKL_COMPACT_PACK + the scalar type to the interleave width V,
 *   4. dispatches to the templated kernel ormqr::ormqr_compact_general<T,V>.
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

#include "ext_mkl_ormqr_compact.h"
#include "ormqr_compact.hpp"
#include "compact_format.hpp"

#include <algorithm>

namespace {

using ormqr::detail::vlen_for_format;

/* Shared validation + dispatch for both precisions. Returns the 1-based
 * index of the first illegal argument (LAPACK convention), 0 if all valid. */
template <typename T>
int validate_and_dispatch(MKL_LAYOUT layout, char side, char trans,
                          MKL_INT m, MKL_INT n, MKL_INT k,
                          const T *ap, MKL_INT ldap,
                          const T *taup,
                          T *cp, MKL_INT ldcp,
                          T *work, MKL_INT lwork,
                          MKL_COMPACT_PACK format, MKL_INT nm)
{
    const bool rowmajor = (layout == MKL_ROW_MAJOR);
    const bool left  = (side == 'L' || side == 'l');
    const bool right = (side == 'R' || side == 'r');
    const bool tran = (trans == 'T' || trans == 't' ||
                       trans == 'C' || trans == 'c');
    const bool notran = (trans == 'N' || trans == 'n');
    const int  V = vlen_for_format<T>(format);

    /* A is (spec_len x k) with spec_len = m (side='L') or n (side='R');
     * C is m x n. A leading dimension counts rows (col-major) or columns
     * (row-major); likewise for C. */
    const MKL_INT spec_len = left ? m : n;

    /* Argument checks, in signature order (section 5: info[i] = -j). */
    if (layout != MKL_COL_MAJOR && layout != MKL_ROW_MAJOR) return 1;
    if (!left && !right)                      return 2;
    if (!tran && !notran)                     return 3;
    if (m < 0)                                return 4;
    if (n < 0)                                return 5;
    if (k < 0 || (left && k > m) || (right && k > n)) return 6;
    if (ldap < std::max<MKL_INT>(1, rowmajor ? k : spec_len)) return 8;
    if (ldcp < std::max<MKL_INT>(1, rowmajor ? n : m))        return 11;
    if (lwork < 1 && lwork != -1)             return 13;
    if (V == 0)                               return 15; /* unknown format */
    if (nm < 0)                               return 16;

    /* Workspace query (lwork = -1): this branch-free kernel needs none, so
     * the minimum (and optimal) workspace is 1. */
    if (lwork == -1) {
        if (work) work[0] = T(1);
        return 0;
    }

    /* Nothing to compute for an empty problem. */
    if (m == 0 || n == 0 || k == 0 || nm == 0)
        return 0;

    const char tr = tran ? 'T' : 'N';
    /* Instantiate the kernel on MKL_INT so the public (possibly 64-bit ILP64)
     * dimensions are carried through without narrowing to int. */
    switch (V) {
    case 2:  ormqr::ormqr_compact_general<T, 2, MKL_INT>(left, rowmajor, tr, m, n, k, ap, ldap, k, taup, cp, ldcp, nm); break;
    case 4:  ormqr::ormqr_compact_general<T, 4, MKL_INT>(left, rowmajor, tr, m, n, k, ap, ldap, k, taup, cp, ldcp, nm); break;
    case 8:  ormqr::ormqr_compact_general<T, 8, MKL_INT>(left, rowmajor, tr, m, n, k, ap, ldap, k, taup, cp, ldcp, nm); break;
    case 16: ormqr::ormqr_compact_general<T, 16, MKL_INT>(left, rowmajor, tr, m, n, k, ap, ldap, k, taup, cp, ldcp, nm); break;
    default: return 15; /* V derived from format unsupported by the kernel */
    }
    return 0;
}

template <typename T>
void run(MKL_LAYOUT layout, char side, char trans,
         MKL_INT m, MKL_INT n, MKL_INT k,
         const T *ap, MKL_INT ldap, const T *taup,
         T *cp, MKL_INT ldcp, T *work, MKL_INT lwork, MKL_INT *info,
         MKL_COMPACT_PACK format, MKL_INT nm)
{
    const int bad = validate_and_dispatch<T>(layout, side, trans, m, n, k,
                                              ap, ldap, taup, cp, ldcp,
                                              work, lwork, format, nm);
    /* info is an array of size nm; every matrix in the batch shares the
     * same (identical) dimensions, so the status is uniform (section 5). */
    if (info) {
        const MKL_INT count = (nm > 0) ? nm : 1;
        for (MKL_INT i = 0; i < count; ++i) info[i] = (bad == 0) ? 0 : -bad;
    }
}

} /* anonymous namespace */

extern "C" void ext_mkl_dormqr_compact(MKL_LAYOUT layout, char side, char trans,
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

extern "C" void ext_mkl_sormqr_compact(MKL_LAYOUT layout, char side, char trans,
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
