#ifndef CQR_COMPACT_H
#define CQR_COMPACT_H

/* C API (FFI-stable) for the templated C++ implementation in
 * cqr_compact.hpp. Apply Q or Q^T from a compact-format QR
 * factorization (mkl_?geqrf_compact) to a compact-format RHS block,
 * from the left:  B := op(Q) * B.
 *
 * This is the missing mkl_?ormqr_compact (side='L') between
 * mkl_?geqrf_compact and mkl_?trsm_compact.
 *
 * Compact layout (matches mkl_?gepack_compact); group g = idx/V, slot v = idx%V:
 *   A_v(i,j)  = ap [ g*ldap*k*V     + (j*ldap + i)*V + v ]
 *   tau_v(kk) = taup[ g*k*V           +  kk*V          + v ]
 *   B_v(i,j)  = bp [ g*ldbp*nrhs*V   + (j*ldbp + i)*V + v ]
 *
 * A is the (ldap, k) reflector batch, declared exactly as LAPACK ?ormqr's
 * A(LDA,K): only k columns are referenced, so the per-matrix column extent --
 * and hence the group stride -- is k. (For a wide or partial-reflector factor
 * whose compact buffer was packed with more than k columns, pack just the k
 * reflector columns, the compact analogue of presenting A as (LDA, K).)
 *
 * Parameters:
 *   trans    'T' (Q^T B, the solve case) or 'N' (Q B)
 *   m        rows of B (and A)
 *   nrhs     columns of B
 *   k        number of reflectors (min(m,n) of the factorization)
 *   ap       compact A from ?geqrf_compact (reflectors below diagonal)
 *   ldap     compact leading dimension of A (>= m), in V-wide elements
 *   taup     compact tau (k per matrix, ld = k)
 *   bp       compact B, overwritten with op(Q)*B
 *   ldbp     compact leading dimension of B (>= m), in V-wide elements
 *   V        interleave width: 2, 4, 8, or 16 elements
 *            (MKL: SSE d=2/s=4, AVX d=4/s=8, AVX512 d=8/s=16;
 *             any of these also work on NEON/SVE as unrolled bursts)
 *   nm       total number of matrices (partial last group is padded)
 *
 * Returns 0 on success, or -j (LAPACK sign convention) if the j-th argument,
 * counted in signature order, had an illegal value:
 *   -1 trans (not 'T'/'t'/'N'/'n')   -2 m (<0)        -3 nrhs (<0)
 *   -4 k (<0 or >m)                  -6 ldap (<max(1,m))
 *   -9 ldbp (<max(1,m))              -10 V (not 2/4/8/16)
 *   -11 nm (<0)
 * Pointer arguments are not inspected (LAPACK convention). An empty problem
 * (m, nrhs, k, or nm == 0) is a valid no-op returning 0. The routine never
 * aborts the calling process.
 *
 * Assisted-by: Claude:claude-fable-5 Claude:claude-opus-4.8
 */

#ifdef __cplusplus
extern "C" {
#endif

int dormqr_compact(char trans, int m, int nrhs, int k, const double *ap, int ldap,
                   const double *taup, double *bp, int ldbp, int V, int nm);

int sormqr_compact(char trans, int m, int nrhs, int k, const float *ap, int ldap,
                   const float *taup, float *bp, int ldbp, int V, int nm);

#ifdef __cplusplus
}
#endif

#endif /* CQR_COMPACT_H */
