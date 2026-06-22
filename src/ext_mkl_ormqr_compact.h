#ifndef EXT_MKL_ORMQR_COMPACT_H
#define EXT_MKL_ORMQR_COMPACT_H

/* ext_mkl_?ormqr_compact
 *
 * The missing mkl_?ormqr_compact: apply Q or Q^T from a Compact-format QR
 * factorization (mkl_?geqrf_compact) to a Compact-format batch of general
 * matrices C, filling the gap between mkl_?geqrf_compact and the
 * application of the resulting orthogonal transformations.
 *
 * The API mirrors MKL's native compact ecosystem exactly (MKL_LAYOUT +
 * MKL_COMPACT_PACK), per the design document ext_mkl_dormqr_compact_design.md.
 * It is a thin C-linkage dispatcher (section 8.1): it unwraps the
 * MKL_COMPACT_PACK opaque format to recover the interleave width V, then
 * dispatches to the templated kernel in ormqr_compact.hpp.
 *
 * The numerical engine is the unblocked, branch-free dorm2r/dorm2l applied
 * across the V interleaved matrices of each compact pack; see
 * ormqr_compact.hpp.
 *
 * Implemented scope (Phase 1 baseline of the design document):
 *   layout : MKL_COL_MAJOR    (the Compact-format convention assumed by
 *                              mkl_?geqrf_compact and the AX=B solver)
 *   side   : 'L' / 'l'        (op(Q) * C, the solve case)
 *   trans  : 'N','n' (Q)  or  'T','t','C','c' (Q^T; C==T for real data)
 *
 * Parameter values outside this scope are reported as an illegal argument
 * value through info[] (LAPACK convention, info[i] = -j), never silently
 * miscomputed.
 */

#include "mkl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

void ext_mkl_dormqr_compact(MKL_LAYOUT layout, char side, char trans,
                            MKL_INT m, MKL_INT n, MKL_INT k,
                            const double *ap, MKL_INT ldap,
                            const double *taup,
                            double *cp, MKL_INT ldcp,
                            double *work, MKL_INT lwork, MKL_INT *info,
                            MKL_COMPACT_PACK format, MKL_INT nm);

void ext_mkl_sormqr_compact(MKL_LAYOUT layout, char side, char trans,
                            MKL_INT m, MKL_INT n, MKL_INT k,
                            const float *ap, MKL_INT ldap,
                            const float *taup,
                            float *cp, MKL_INT ldcp,
                            float *work, MKL_INT lwork, MKL_INT *info,
                            MKL_COMPACT_PACK format, MKL_INT nm);

#ifdef __cplusplus
}
#endif

#endif /* EXT_MKL_ORMQR_COMPACT_H */
