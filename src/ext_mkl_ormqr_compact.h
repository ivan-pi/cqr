#ifndef EXT_MKL_ORMQR_COMPACT_H
#define EXT_MKL_ORMQR_COMPACT_H

/* ext_mkl_?ormqr_compact -- apply Q (or Q^T) of a Compact-format QR
 *
 * This is the missing mkl_?ormqr_compact. It multiplies a Compact-format
 * batch of general matrices C by the orthogonal factor Q (or Q^T) produced by
 * mkl_?geqrf_compact, filling the gap between the compact QR factorization and
 * the application of its reflectors. The API mirrors MKL's native compact
 * ecosystem (MKL_LAYOUT + MKL_COMPACT_PACK); see the full parameter reference
 * in ext_mkl_dormqr_compact_design.md.
 *
 * Typical use -- the batched AX = B solver:
 *     mkl_dgeqrf_compact (..., A -> H, tau);          // A = Q R
 *     ext_mkl_dormqr_compact('L','T', ..., H, tau, B); // B := Q^T B
 *     mkl_dtrsm_compact  (..., U, R, B);              // B := R^{-1} Q^T B = X
 *
 * The matrices are passed in the same Compact buffers used elsewhere in the
 * MKL compact API: pack with mkl_?gepack_compact, query/obtain the opaque
 * `format` from mkl_get_format_compact(), and pass the total batch size `nm`.
 * C is overwritten with op(Q)*C. With lwork = -1 the call is a workspace query
 * returning the optimal lwork in work[0]. Per-matrix status is reported in the
 * length-`nm` array info[]: info[i] = 0 on success, or -j if the j-th argument
 * had an illegal value (LAPACK convention).
 *
 * Supported arguments: layout = MKL_COL_MAJOR or MKL_ROW_MAJOR,
 * side = 'L'/'l' (op(Q) C) or 'R'/'r' (C op(Q)), trans = 'N'/'n' (Q) or
 * 'T'/'t'/'C'/'c' (Q^T), for FP64 and FP32. Values outside this set are
 * reported through info[] rather than miscomputed.
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
