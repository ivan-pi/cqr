#ifndef CQR_COMPACT_H
#define CQR_COMPACT_H

/* Portable C API (FFI-stable) for the templated C++ kernels in
 * cqr_geqrf_compact.hpp and cqr_compact.hpp. Batched QR of matrices stored in
 * MKL Compact (interleaved) format, with an explicit interleave width V and no
 * MKL dependency -- the portable form of mkl_?geqrf_compact / a mkl_?ormqr_compact:
 *
 *   dgeqrf_compact / sgeqrf_compact  -- QR factorization  A = Q R
 *   dormqr_compact / sormqr_compact  -- apply Q or Q^T from the left, B := op(Q) B
 *
 * Compact layout (matches mkl_?gepack_compact); group g = idx/V, slot v = idx%V:
 *   A_v(i,j)  = ap [ g*ldap*ncol*V + (j*ldap + i)*V + v ]   (column-major)
 *   tau_v(kk) = taup[ g*k*V         +  kk*V          + v ]
 *   B_v(i,j)  = bp [ g*ldbp*nrhs*V  + (j*ldbp + i)*V + v ]
 * where ncol is n for ?geqrf (the full matrix) and k for ?ormqr (A is the
 * (ldap, k) reflector batch, exactly as LAPACK ?ormqr's A(LDA,K)). Row-major
 * ?geqrf swaps the in-matrix index roles (i -> i*ldap + j).
 *
 * V is the interleave width: 2, 4, 8, or 16 elements (MKL: SSE d=2/s=4,
 * AVX d=4/s=8, AVX512 d=8/s=16; any of these also work on NEON/SVE as unrolled
 * bursts). Pointer arguments are not inspected in release builds (LAPACK
 * convention). An empty problem is a valid no-op returning 0. The routines never
 * abort the calling process.
 *
 * Assisted-by: Claude:claude-fable-5 Claude:claude-opus-4.8
 */

#ifdef __cplusplus
extern "C" {
#endif

/* QR factorization: on exit ap holds R (on/above the diagonal) and the
 * Householder vectors (below), taup holds the k = min(m,n) reflector scalars --
 * the LAPACK ?geqrf storage convention, one matrix per compact lane.
 *   layout   'C'/'c' column-major (tuned) or 'R'/'r' row-major
 *   m, n     rows, columns of A
 *   ap       compact A (m x n); overwritten with (R, Householder vectors)
 *   ldap     compact leading dimension (>= m col-major, >= n row-major)
 *   taup     compact tau output (k = min(m,n) per matrix, ld = k)
 *   V, nm    interleave width; total number of matrices (padded last group)
 * Returns 0, or -j (LAPACK sign convention) for an illegal j-th argument:
 *   -1 layout   -2 m (<0)   -3 n (<0)   -5 ldap   -7 V (not 2/4/8/16)   -8 nm (<0)
 */
int dgeqrf_compact(char layout, int m, int n, double *ap, int ldap, double *taup, int V,
                   int nm);

int sgeqrf_compact(char layout, int m, int n, float *ap, int ldap, float *taup, int V,
                   int nm);

/* Apply Q (or Q^T) of a compact QR to a compact RHS block from the left,
 * B := op(Q) B -- the missing mkl_?ormqr_compact (side='L') between
 * mkl_?geqrf_compact and mkl_?trsm_compact.
 *   trans    'T' (Q^T B, the solve case) or 'N' (Q B)
 *   m, nrhs  rows of B (and A); columns of B
 *   k        number of reflectors (min(m,n) of the factorization)
 *   ap       compact reflectors from ?geqrf_compact, (ldap, k) per matrix
 *   ldap     compact leading dimension of A (>= m)
 *   taup     compact tau (k per matrix, ld = k)
 *   bp       compact B (m x nrhs), overwritten with op(Q) B
 *   ldbp     compact leading dimension of B (>= m)
 *   V, nm    interleave width; total number of matrices (padded last group)
 * Returns 0, or -j for an illegal j-th argument:
 *   -1 trans   -2 m (<0)   -3 nrhs (<0)   -4 k (<0 or >m)   -6 ldap
 *   -9 ldbp    -10 V (not 2/4/8/16)   -11 nm (<0)
 */
int dormqr_compact(char trans, int m, int nrhs, int k, const double *ap, int ldap,
                   const double *taup, double *bp, int ldbp, int V, int nm);

int sormqr_compact(char trans, int m, int nrhs, int k, const float *ap, int ldap,
                   const float *taup, float *bp, int ldbp, int V, int nm);

#ifdef __cplusplus
}
#endif

#endif /* CQR_COMPACT_H */
