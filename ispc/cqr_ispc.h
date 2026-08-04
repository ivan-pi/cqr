#ifndef CQR_ISPC_H
#define CQR_ISPC_H

/* cqr_ispc.h -- extern "C" surface of the ISPC compact-QR prototype.
 *
 * These are the symbols ISPC exports from cqr_ispc.ispc (verified against the
 * compiler-generated header). They are drop-in siblings of the portable C API
 * in ../src/cqr_compact.h -- same MKL Compact (interleaved) storage, same
 * unblocked math -- with two differences the prototype makes on purpose:
 *
 *   1. The interleave width V is fixed at compile time by the ISPC gang width
 *      (build --target=avx512skx-x8  ->  programCount == V == 8, the AVX-512
 *      FP64 compact format). The GNU-vector API instead takes V at runtime and
 *      dispatches; ISPC bakes one width per target. Callers must pass matrices
 *      packed at V = 8 (mkl_get_format_compact() == MKL_COMPACT_AVX512).
 *   2. Column-major only, double precision only -- the tuned solver path.
 *
 * To use them as literal drop-ins for the cqr_compact.h names, wrap or #define:
 *   #define dgeqrf_compact(layout,m,n,ap,ld,tau,V,nm) \
 *           cqr_ispc_dgeqrf_compact((m),(n),(ap),(ld),(tau),(nm))
 * (asserting layout=='C' and V==8). See test_cqr_ispc.cpp / bench_cqr_ispc.cpp.
 *
 * No argument checking (MKL Compact convention); an empty problem is a no-op.
 * A partial final group must be padded to a full pack of identities by the
 * caller, exactly as for the MKL compact routines, so the kernels run unmasked.
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#ifdef __cplusplus
extern "C" {
#endif

/* QR factorization A = Q R (unblocked geqr2). On exit ap holds R on/above the
 * diagonal and the Householder vectors below; taup holds k = min(m,n) reflector
 * scalars per matrix. Column-major, V = 8. */
void cqr_ispc_dgeqrf_compact(int m, int n, double *ap, int ldap, double *taup, int nm);

/* Apply Q (trans=0) or Q^T (trans=1) from the left: B := op(Q) B. ap/taup are
 * the reflectors from cqr_ispc_dgeqrf_compact, (ldap, k) per matrix; bp is the
 * compact (m x nrhs) block, overwritten. Column-major, V = 8. */
void cqr_ispc_dormqr_compact(int trans, int m, int nrhs, int k, double *ap, int ldap,
                             double *taup, double *bp, int ldbp, int nm);

/* Triangular solve B := alpha * op(A)^-1 B for the QR-solve case: side left,
 * A upper-triangular (n x n), no transpose, non-unit diagonal (back-substitution).
 * The compact analogue of mkl_dtrsm_compact(LEFT, UPPER, NOTRANS, NONUNIT).
 * Column-major, V = 8. */
void cqr_ispc_dtrsm_compact(int n, int nrhs, double alpha, double *ap, int ldap,
                            double *bp, int ldbp, int nm);

#ifdef __cplusplus
}
#endif

#endif /* CQR_ISPC_H */
