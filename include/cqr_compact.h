#ifndef CQR_COMPACT_H
#define CQR_COMPACT_H

/* Portable, FFI-stable C API for this project's batched QR, Cholesky, and
 * triangular-solve kernels: dense factorizations (and the triangular solve that
 * closes a linear solve) of many small matrices stored in the compact
 * (interleaved) format, with an explicit interleave width V.
 *
 *   dgeqrf_compact / sgeqrf_compact  -- QR factorization  A = Q R
 *   dormqr_compact / sormqr_compact  -- apply Q or Q^T from the left, B := op(Q) B
 *   dpotrf_compact / spotrf_compact  -- Cholesky factorization  A = L L^T or A = U^T U
 *   dtrsm_compact  / strsm_compact   -- triangular solve  op(A) X = alpha B, etc.
 *
 * Compact layout; group g = idx/V, slot v = idx%V:
 *   A_v(i,j)  = ap [ g*ldap*ncol*V + (j*ldap + i)*V + v ]   (column-major)
 *   tau_v(kk) = taup[ g*k*V         +  kk*V          + v ]
 *   B_v(i,j)  = bp [ g*ldbp*nrhs*V  + (j*ldbp + i)*V + v ]
 * where ncol is n for ?geqrf (the full matrix) and k for ?ormqr (A is the
 * (ldap, k) reflector batch, exactly as LAPACK ?ormqr's A(LDA,K)). Row-major
 * ?geqrf swaps the in-matrix index roles (i -> i*ldap + j).
 *
 * V is the interleave width: 2, 4, 8, or 16 elements (SSE d=2/s=4, AVX d=4/s=8,
 * AVX512 d=8/s=16; any of these also work on NEON/SVE as unrolled bursts).
 * Pointer arguments are not inspected in release builds (LAPACK convention). An
 * empty problem is a valid no-op returning 0. The routines never abort the
 * calling process.
 *
 * Threading: built with OpenMP (the default, -DCQR_WITH_OPENMP=ON), each routine
 * runs its loop over groups of V matrices as a static-schedule parallel loop on
 * a team of min(ngroups, omp_get_max_threads()) threads, active only when the
 * call has at least two groups and enough work (about 2e5 flops, the measured
 * fork/join break-even; -DCQR_OMP_MIN_FLOPS overrides). The thread count is the
 * one OpenMP reports at the current nesting level, so calling these routines
 * from your own parallel loop leaves the inner loop serial (no competing pools)
 * unless you enable nested parallelism, e.g. OMP_NUM_THREADS=8,2
 * OMP_MAX_ACTIVE_LEVELS=2. Results do not depend on the thread count.
 *
 * Alignment: the compact buffers may start at any address aligned to the scalar
 * type (the SIMD element carries relaxed alignment, so loads and stores never
 * fault); results are identical regardless. For full speed, align each buffer's
 * base to the pack width in bytes -- 64 covers every format (V*sizeof(T) <= 64),
 * e.g. posix_memalign or std::aligned_alloc. Because every element sits at a
 * pack-multiple offset, a pack-aligned base keeps every vector access on one
 * cache line; a non-pack-aligned base splits each access across two lines,
 * costing up to ~40% on small, cache-resident sizes.
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
 * B := op(Q) B -- the reflector-application (side='L') step between a compact
 * QR factorization (?geqrf_compact) and a triangular solve.
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

/* Cholesky factorization of a batch of symmetric positive-definite n x n
 * matrices A: A = L L^T (uplo 'L') or A = U^T U (uplo 'U'), one matrix per
 * compact lane. On exit the named triangle of ap holds its Cholesky factor; the
 * strictly-opposite triangle is neither referenced nor modified.
 *   layout   'C'/'c' column-major (tuned when lower) or 'R'/'r' row-major
 *   uplo     'L'/'l' factor/store the lower triangle L, or 'U'/'u' the upper U
 *   n        order of each A
 *   ap       compact A (n x n); the named triangle is overwritten with L or U
 *   ldap     compact leading dimension (>= n)
 *   V, nm    interleave width; total number of matrices (padded last group)
 * Positive-definiteness is assumed, not checked: a non-SPD lane yields NaN/Inf
 * in its factor rather than an error (see cqr_mkl_dpotrf_compact_design.md 6.2).
 * Returns 0, or -j (LAPACK sign convention) for an illegal j-th argument:
 *   -1 layout   -2 uplo   -3 n (<0)   -5 ldap (< max(1,n))
 *   -6 V (not 2/4/8/16)   -7 nm (<0)
 */
int dpotrf_compact(char layout, char uplo, int n, double *ap, int ldap, int V, int nm);

int spotrf_compact(char layout, char uplo, int n, float *ap, int ldap, int V, int nm);

/* Triangular solve with multiple right-hand sides -- the portable form of
 * mkl_?trsm_compact, the step that closes the batched QR solve. Solves in place
 *   op(A) X = alpha B   (side='L')   or   X op(A) = alpha B   (side='R'),
 * with A the order-s (s = m for side='L', n for side='R') unit/non-unit,
 * upper/lower triangular factor and op(A) = A ('N') or A^T ('T'/'C'). B (m x n)
 * is overwritten by X.
 *   layout   'C'/'c' column-major (tuned) or 'R'/'r' row-major
 *   side     'L' (op(A) X = alpha B) or 'R' (X op(A) = alpha B)
 *   uplo     'U' A upper triangular or 'L' A lower triangular
 *   transa   'N' (A) or 'T'/'C' (A^T; 'C' == 'T' for the real types)
 *   diag     'U' A has a unit diagonal (not read) or 'N' non-unit
 *   m, n     rows, columns of B (A is s x s, s = m for 'L', n for 'R')
 *   alpha    scalar multiplying B; alpha = 0 sets B := 0 (A not referenced)
 *   ap       compact triangular A (s x s per matrix)
 *   ldap     compact leading dimension of A (>= max(1, s))
 *   bp       compact B (m x n), overwritten with X
 *   ldbp     compact leading dimension of B (>= m col-major, >= n row-major)
 *   V, nm    interleave width; total number of matrices (padded last group)
 * Returns 0, or -j for an illegal j-th argument:
 *   -1 layout  -2 side   -3 uplo   -4 transa   -5 diag   -6 m (<0)   -7 n (<0)
 *   -10 ldap   -12 ldbp  -13 V (not 2/4/8/16)  -14 nm (<0)
 * (alpha, ap and bp are never inspected, matching LAPACK/BLAS.) */
int dtrsm_compact(char layout, char side, char uplo, char transa, char diag, int m, int n,
                  double alpha, const double *ap, int ldap, double *bp, int ldbp, int V,
                  int nm);

int strsm_compact(char layout, char side, char uplo, char transa, char diag, int m, int n,
                  float alpha, const float *ap, int ldap, float *bp, int ldbp, int V,
                  int nm);

#ifdef __cplusplus
}
#endif

#endif /* CQR_COMPACT_H */
