#ifndef CQR_ISPC_H
#define CQR_ISPC_H

/* cqr_ispc.h -- extern "C" surface of the ISPC compact-QR prototype
 * (cqr_ispc.ispc). Drop-in siblings of the portable C API in
 * ../src/cqr_compact.h over the same MKL Compact buffers, double only (column-
 * major, except cqr_ispc_dpotrf_compact, which takes a layout flag). The
 * interleave width V is fixed at compile time by the ISPC gang width
 * (set by the --target's -xN suffix): the data must be packed at V == the gang
 * width -- query it with cqr_ispc_gang_width(). For MKL Compact interop that is
 * MKL's format V (8 on AVX-512, from --target=avx512skx-x8). No argument
 * checking; an empty problem is a no-op; a partial final group must be
 * identity-padded by the caller.
 * Assisted-by: Claude:claude-opus-4.8 */

#ifdef __cplusplus
extern "C" {
#endif

/* The gang width these kernels were compiled for; the compact data must be packed
 * at this interleave width V. */
int cqr_ispc_gang_width(void);

/* QR: ap (m x n) -> (R, Householder vectors); taup -> k=min(m,n) tau per matrix. */
void cqr_ispc_dgeqrf_compact(int m, int n, double *ap, int ldap, double *taup, int nm);

/* Apply Q (trans=0) or Q^T (trans=1) from the left: bp := op(Q) bp. */
void cqr_ispc_dormqr_compact(int trans, int m, int nrhs, int k, double *ap, int ldap,
                             double *taup, double *bp, int ldbp, int nm);

/* Solve R X = alpha B (left, upper, no-trans, non-unit); bp -> X. */
void cqr_ispc_dtrsm_compact(int n, int nrhs, double alpha, double *ap, int ldap,
                            double *bp, int ldbp, int nm);

/* Cholesky A = L L^T (upper=0) or U^T U (upper=1); ap (n x n) -> its factor in the
 * named triangle. rowmajor/upper select the layout/uplo, as mkl_dpotrf_compact. */
void cqr_ispc_dpotrf_compact(int rowmajor, int upper, int n, double *ap, int ldap,
                             int nm);

#ifdef __cplusplus
}
#endif

#endif /* CQR_ISPC_H */
