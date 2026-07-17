#ifndef CQR_GEQRF_COMPACT_H
#define CQR_GEQRF_COMPACT_H

/* C API (FFI-stable) for the templated C++ implementation in
 * cqr_geqrf_compact.hpp. Compute the QR factorization of a batch of general
 * m x n matrices stored in Compact (interleaved-batch) format, in place:
 *
 *     A = Q R,   Q = H(0) H(1) ... H(k-1),   k = min(m, n),
 *     H(kk) = I - tau(kk) * v(kk) * v(kk)^T.
 *
 * This is the portable form of mkl_?geqrf_compact: on exit each matrix holds R
 * on and above the diagonal, the Householder vectors below it, and taup holds
 * the k reflector scalars -- exactly the LAPACK ?geqrf storage convention, one
 * matrix per compact lane. It pairs with dormqr_compact (cqr_compact.h), which
 * applies the reflectors this routine produces.
 *
 * Compact layout (matches mkl_?gepack_compact); group g = idx/V, slot v = idx%V:
 *   A_v(i,j)  = ap [ g*ldap*n*V + (j*ldap + i)*V + v ]   (column-major)
 *   tau_v(kk) = taup[ g*k*V      +  kk*V          + v ]
 * Row-major swaps the in-matrix index roles (i -> i*ldap + j) and the group
 * stride's complementary extent (ldap*m instead of ldap*n).
 *
 * Parameters:
 *   layout   'C'/'c' column-major (tuned) or 'R'/'r' row-major
 *   m        rows of A
 *   n        columns of A
 *   ap       compact A (m x n); overwritten with (R, Householder vectors)
 *   ldap     compact leading dimension of A (>= m col-major, >= n row-major),
 *            in V-wide elements
 *   taup     compact tau output (k = min(m,n) per matrix, ld = k)
 *   V        interleave width: 2, 4, 8, or 16 elements
 *            (MKL: SSE d=2/s=4, AVX d=4/s=8, AVX512 d=8/s=16;
 *             any of these also work on NEON/SVE as unrolled bursts)
 *   nm       total number of matrices (partial last group is padded)
 *
 * Returns 0 on success, or -j (LAPACK sign convention) if the j-th argument,
 * counted in signature order, had an illegal value:
 *   -1 layout (not 'C'/'c'/'R'/'r')   -2 m (<0)       -3 n (<0)
 *   -5 ldap (< max(1,m) col-major / < max(1,n) row-major)
 *   -7 V (not 2/4/8/16)               -8 nm (<0)
 * Pointer arguments are not inspected in release builds (LAPACK convention); a
 * debug-only assert guards against a null ap/taup on a non-empty problem. An
 * empty problem (m, n, or nm == 0) is a valid no-op returning 0. The routine
 * never aborts the calling process.
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#ifdef __cplusplus
extern "C" {
#endif

int dgeqrf_compact(char layout, int m, int n,
                   double *ap, int ldap, double *taup,
                   int V, int nm);

int sgeqrf_compact(char layout, int m, int n,
                   float *ap, int ldap, float *taup,
                   int V, int nm);

#ifdef __cplusplus
}
#endif

#endif /* CQR_GEQRF_COMPACT_H */
