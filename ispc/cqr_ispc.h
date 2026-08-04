#ifndef CQR_ISPC_H
#define CQR_ISPC_H

/* cqr_ispc.h -- extern "C" surface of the ISPC compact-QR prototype
 * (cqr_ispc.ispc). Drop-in siblings of the portable C API in
 * ../src/cqr_compact.h over the same MKL Compact buffers, with two deliberate
 * differences: the interleave width is fixed at compile time by the ISPC gang
 * (build --target=avx512skx-x8 -> V = programCount = 8, the AVX-512 FP64 format),
 * and it is column-major / double only. No argument checking; an empty problem is
 * a no-op; a partial final group must be identity-padded by the caller.
 * Assisted-by: Claude:claude-opus-4.8 */

#ifdef __cplusplus
extern "C" {
#endif

/* QR: ap (m x n) -> (R, Householder vectors); taup -> k=min(m,n) tau per matrix. */
void cqr_ispc_dgeqrf_compact(int m, int n, double *ap, int ldap, double *taup, int nm);

/* Apply Q (trans=0) or Q^T (trans=1) from the left: bp := op(Q) bp. */
void cqr_ispc_dormqr_compact(int trans, int m, int nrhs, int k, double *ap, int ldap,
                             double *taup, double *bp, int ldbp, int nm);

/* Solve R X = alpha B (left, upper, no-trans, non-unit); bp -> X. */
void cqr_ispc_dtrsm_compact(int n, int nrhs, double alpha, double *ap, int ldap,
                            double *bp, int ldbp, int nm);

#ifdef __cplusplus
}
#endif

#endif /* CQR_ISPC_H */
