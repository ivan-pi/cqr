#ifndef CQR_MKL_EXT_H
#define CQR_MKL_EXT_H

/* cqr_mkl_ext.h -- batched QR, Cholesky, LDL^T, and triangular solve for
 * matrices in Intel MKL's Compact format.
 *
 *   cqr_mkl_?geqrf_compact   -- QR factorization of a Compact-format batch
 *   cqr_mkl_?ormqr_compact   -- apply Q (or Q^T) of a Compact-format QR
 *   cqr_mkl_?potrf_compact   -- Cholesky factorization of an SPD Compact-format batch
 *   cqr_mkl_?sytrfnp_compact -- unpivoted LDL^T factorization of a symmetric batch
 *   cqr_mkl_?sytrsnp_compact -- solve A X = B from that LDL^T factor
 *   cqr_mkl_?sysvnp_compact  -- LDL^T factor + solve, fused per group
 *   cqr_mkl_?trsm_compact    -- triangular solve op(A) X = alpha B (and variants)
 *
 * All take MKL's MKL_LAYOUT + MKL_COMPACT_PACK interface, so they drop into the
 * MKL compact ecosystem (pack with mkl_?gepack_compact, take `format` from
 * mkl_get_format_compact(), pass the total batch size `nm`), but are backed by
 * this project's portable SIMD kernels. cqr_mkl_?geqrf_compact,
 * cqr_mkl_?potrf_compact and cqr_mkl_?trsm_compact are signature- and
 * storage-compatible alternatives to the MKL routines of the same name, so they
 * mix freely with them; cqr_mkl_?ormqr_compact is the apply-Q step MKL omits
 * (LAPACK ?ormqr plus the layout/format/nm arguments), and the three symmetric
 * "np" routines are the unpivoted LDL^T MKL has no compact form of at all (the
 * suffix follows MKL's own unpivoted mkl_?getrfnp_compact). Together they factor
 * and solve batched systems entirely in the compact format:
 *
 *     cqr_mkl_dgeqrf_compact (..., A -> H, tau);       // A = Q R
 *     cqr_mkl_dormqr_compact('L','T', ..., H, tau, B); // B := Q^T B
 *     cqr_mkl_dtrsm_compact  (..., U, R, B);           // B := R^{-1} Q^T B = X
 *
 *     cqr_mkl_dsysvnp_compact(..., A -> (L, D), B);   // A = L D L^T; B := A^{-1} B
 *
 * Conventions shared by all routines:
 *   - No argument checking (compact routines skip it for vectorization); the
 *     caller passes consistent parameters. Use the portable cqr_compact.h
 *     entry points for LAPACK-style info = -j validation.
 *   - `info` is a single scalar status (MKL leaves the compact info reserved):
 *     0 on success, -1 for an unrecognized `format`.
 *   - Workspace: ?geqrf and ?ormqr take work/lwork like their LAPACK namesakes;
 *     with lwork = -1 the call is a query returning the optimal lwork in
 *     work[0] -- 1, these kernels need no scratch. Size each routine's work
 *     from ITS OWN query and give each its own buffer: MKL's mkl_?geqrf_compact
 *     needs ~n*V, and because compact routines skip checking, an undersized or
 *     shared work array is undefined behavior (silent heap corruption on some
 *     MKL builds).
 *   - The reflector batch A of ?ormqr is dimensioned (ldap, k) as in LAPACK, so
 *     the compact buffer must be packed with exactly k columns (group stride
 *     ldap*k*V). Square and tall factors from ?geqrf_compact satisfy this as
 *     they are; a wide factor (more columns than reflectors) must be repacked
 *     with its k reflector columns only.
 *
 *   - Threading: built with OpenMP, the loop over groups of V matrices runs in
 *     parallel (static schedule, one thread per group at most) whenever the
 *     call has two or more groups and enough work to pay for the fork; from
 *     inside a caller's own parallel loop it stays serial unless nested
 *     parallelism is enabled (OMP_NUM_THREADS=8,2 with OMP_MAX_ACTIVE_LEVELS=2).
 *     See the note in cqr_compact.h.
 *
 * Per-routine parameter references:
 * docs/cqr_mkl_d{geqrf,ormqr,potrf,sytrfnp,trsm}_compact_design.md.
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#include "mkl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* QR factorization: A -> (R, Householder vectors) in ap, tau in taup. Drop-in
 * for mkl_?geqrf_compact (identical signature). */
void cqr_mkl_dgeqrf_compact(MKL_LAYOUT layout, MKL_INT m, MKL_INT n, double *ap,
                            MKL_INT ldap, double *taup, double *work, MKL_INT lwork,
                            MKL_INT *info, MKL_COMPACT_PACK format, MKL_INT nm);

void cqr_mkl_sgeqrf_compact(MKL_LAYOUT layout, MKL_INT m, MKL_INT n, float *ap,
                            MKL_INT ldap, float *taup, float *work, MKL_INT lwork,
                            MKL_INT *info, MKL_COMPACT_PACK format, MKL_INT nm);

/* C := op(Q) C (side 'L') or C op(Q) (side 'R'), op(Q) = Q ('N') or Q^T
 * ('T'/'C'), from the (H, tau) of a compact QR: LAPACK ?ormqr plus layout,
 * format and nm. */
void cqr_mkl_dormqr_compact(MKL_LAYOUT layout, char side, char trans, MKL_INT m,
                            MKL_INT n, MKL_INT k, const double *ap, MKL_INT ldap,
                            const double *taup, double *cp, MKL_INT ldcp, double *work,
                            MKL_INT lwork, MKL_INT *info, MKL_COMPACT_PACK format,
                            MKL_INT nm);

void cqr_mkl_sormqr_compact(MKL_LAYOUT layout, char side, char trans, MKL_INT m,
                            MKL_INT n, MKL_INT k, const float *ap, MKL_INT ldap,
                            const float *taup, float *cp, MKL_INT ldcp, float *work,
                            MKL_INT lwork, MKL_INT *info, MKL_COMPACT_PACK format,
                            MKL_INT nm);

/* Cholesky factorization of a batch of symmetric positive-definite n x n
 * matrices: A = L L^T (MKL_LOWER) or A = U^T U (MKL_UPPER). Drop-in for
 * mkl_?potrf_compact (identical signature). On exit the named triangle holds
 * the factor; the other is untouched. No workspace and no SPD test: a non-SPD
 * lane poisons itself with NaN/Inf rather than reporting info = j. */
void cqr_mkl_dpotrf_compact(MKL_LAYOUT layout, MKL_UPLO uplo, MKL_INT n, double *ap,
                            MKL_INT ldap, MKL_INT *info, MKL_COMPACT_PACK format,
                            MKL_INT nm);

void cqr_mkl_spotrf_compact(MKL_LAYOUT layout, MKL_UPLO uplo, MKL_INT n, float *ap,
                            MKL_INT ldap, MKL_INT *info, MKL_COMPACT_PACK format,
                            MKL_INT nm);

/* LDL^T factorization without pivoting of a batch of symmetric n x n matrices:
 * A = L D L^T (MKL_LOWER) or A = U^T D U (MKL_UPPER; the transpose dual, as
 * ?potrf's U^T U, not LAPACK ?sytrf's U D U^T). mkl_?potrf_compact's signature
 * (MKL has no compact sytrf). On exit the diagonal holds D and the strict
 * off-diagonal of the named triangle the unit L/U; the other triangle is
 * untouched. No workspace, no ipiv, no singularity test: indefinite matrices
 * factor (no sqrt), but a zero pivot -- a singular leading principal minor --
 * poisons its lane with Inf/NaN rather than reporting info = j. */
void cqr_mkl_dsytrfnp_compact(MKL_LAYOUT layout, MKL_UPLO uplo, MKL_INT n, double *ap,
                              MKL_INT ldap, MKL_INT *info, MKL_COMPACT_PACK format,
                              MKL_INT nm);

void cqr_mkl_ssytrfnp_compact(MKL_LAYOUT layout, MKL_UPLO uplo, MKL_INT n, float *ap,
                              MKL_INT ldap, MKL_INT *info, MKL_COMPACT_PACK format,
                              MKL_INT nm);

/* Solve A X = B from the factor of cqr_mkl_?sytrfnp_compact (same uplo), in
 * place: L z = B; D w = z; L^T X = w (MKL_LOWER), or U^T z = B; D w = z;
 * U X = w (MKL_UPPER). LAPACK ?sytrs minus ipiv, plus layout/format/nm. B is
 * n x nrhs with ldbp >= n (column-major) or >= nrhs (row-major); the factor is
 * not modified. A zero D(i) yields Inf/NaN in that lane's X, not an error. */
void cqr_mkl_dsytrsnp_compact(MKL_LAYOUT layout, MKL_UPLO uplo, MKL_INT n, MKL_INT nrhs,
                              const double *ap, MKL_INT ldap, double *bp, MKL_INT ldbp,
                              MKL_INT *info, MKL_COMPACT_PACK format, MKL_INT nm);

void cqr_mkl_ssytrsnp_compact(MKL_LAYOUT layout, MKL_UPLO uplo, MKL_INT n, MKL_INT nrhs,
                              const float *ap, MKL_INT ldap, float *bp, MKL_INT ldbp,
                              MKL_INT *info, MKL_COMPACT_PACK format, MKL_INT nm);

/* Symmetric solve A X = B in one pass: cqr_mkl_?sytrfnp_compact followed by
 * cqr_mkl_?sytrsnp_compact, fused per group of V matrices so each factor is
 * solved with while cache-resident and the whole solve is one threaded group
 * loop. LAPACK ?sysv minus ipiv and workspace. On exit ap holds the (D, L|U)
 * factor exactly as ?sytrfnp leaves it and bp holds X; the result is
 * bit-identical to the two separate calls. */
void cqr_mkl_dsysvnp_compact(MKL_LAYOUT layout, MKL_UPLO uplo, MKL_INT n, MKL_INT nrhs,
                             double *ap, MKL_INT ldap, double *bp, MKL_INT ldbp,
                             MKL_INT *info, MKL_COMPACT_PACK format, MKL_INT nm);

void cqr_mkl_ssysvnp_compact(MKL_LAYOUT layout, MKL_UPLO uplo, MKL_INT n, MKL_INT nrhs,
                             float *ap, MKL_INT ldap, float *bp, MKL_INT ldbp,
                             MKL_INT *info, MKL_COMPACT_PACK format, MKL_INT nm);

/* Triangular solve with multiple right-hand sides, in place:
 *     op(A) X = alpha B  (MKL_LEFT)   or   X op(A) = alpha B  (MKL_RIGHT),
 * A the order-s (s = m left, n right) unit/non-unit upper/lower triangular
 * factor, op(A) = A (MKL_NOTRANS) or A^T (MKL_TRANS / MKL_CONJTRANS, the same
 * for real types), B (m x n) overwritten by X. Drop-in for mkl_?trsm_compact
 * (identical signature): like the BLAS ?trsm it batches, no workspace, no info. */
void cqr_mkl_dtrsm_compact(MKL_LAYOUT layout, MKL_SIDE side, MKL_UPLO uplo,
                           MKL_TRANSPOSE transa, MKL_DIAG diag, MKL_INT m, MKL_INT n,
                           double alpha, const double *ap, MKL_INT ldap, double *bp,
                           MKL_INT ldbp, MKL_COMPACT_PACK format, MKL_INT nm);

void cqr_mkl_strsm_compact(MKL_LAYOUT layout, MKL_SIDE side, MKL_UPLO uplo,
                           MKL_TRANSPOSE transa, MKL_DIAG diag, MKL_INT m, MKL_INT n,
                           float alpha, const float *ap, MKL_INT ldap, float *bp,
                           MKL_INT ldbp, MKL_COMPACT_PACK format, MKL_INT nm);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

/* ------------------------------------------------------------------ *
 * C++-only internal helper (for this project's implementation, tests   *
 * and examples; not part of the FFI-stable C surface above). It uses   *
 * MKL only for its *types* (MKL_COMPACT_PACK) -- like the C API, it     *
 * needs the MKL headers but adds no MKL link dependency. The RAII       *
 * buffer helpers that *call* mkl_malloc / mkl_free live in the separate *
 * cqr_mkl_alloc.h, so including this header does not pull in the MKL    *
 * link line.                                                            *
 * ------------------------------------------------------------------ */

#include <cstddef>

namespace cqr::detail {

/* Interleave width V for a given MKL Compact pack format and scalar type T.
 * MKL packs V = (SIMD register bytes) / sizeof(T):
 *   SSE = 16 B, AVX = 32 B, AVX512 = 64 B.
 * Returns 0 for an unrecognised format. */
template <typename T> inline int vlen_for_format(MKL_COMPACT_PACK format)
{
    int bytes;
    switch (format) {
    case MKL_COMPACT_SSE: bytes = 16; break;
    case MKL_COMPACT_AVX: bytes = 32; break;
    case MKL_COMPACT_AVX512: bytes = 64; break;
    default: return 0;
    }
    return bytes / static_cast<int>(sizeof(T));
}

/* Inverse of vlen_for_format: the pack format whose interleave width for scalar
 * type T is v, i.e. v * sizeof(T) register bytes (16/32/64 -> SSE/AVX/AVX512).
 * Any other v returns MKL_COMPACT_SSE; callers validate v against the host's
 * native width before use. */
template <typename T> inline MKL_COMPACT_PACK format_for_vlen(int v)
{
    switch (v * static_cast<int>(sizeof(T))) {
    case 16: return MKL_COMPACT_SSE;
    case 32: return MKL_COMPACT_AVX;
    case 64: return MKL_COMPACT_AVX512;
    default: return MKL_COMPACT_SSE; /* unrecognised; caller validates */
    }
}

/* Human-readable name of the SIMD ISA behind an MKL Compact pack format
 * ("SSE"/"AVX"/"AVX512", or "unknown"). */
inline const char *compact_format_name(MKL_COMPACT_PACK format)
{
    switch (format) {
    case MKL_COMPACT_SSE: return "SSE";
    case MKL_COMPACT_AVX: return "AVX";
    case MKL_COMPACT_AVX512: return "AVX512";
    default: return "unknown";
    }
}

} /* namespace cqr::detail */

#endif /* __cplusplus */

#endif /* CQR_MKL_EXT_H */
