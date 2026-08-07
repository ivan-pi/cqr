#ifndef CQR_MKL_EXT_H
#define CQR_MKL_EXT_H

/* cqr_mkl_ext.h -- batched QR, Cholesky, and triangular solve for matrices in
 * Intel MKL's Compact format.
 *
 *   cqr_mkl_?geqrf_compact -- QR factorization of a Compact-format batch
 *   cqr_mkl_?ormqr_compact -- apply Q (or Q^T) of a Compact-format QR
 *   cqr_mkl_?potrf_compact -- Cholesky factorization of an SPD Compact-format batch
 *   cqr_mkl_?trsm_compact  -- triangular solve op(A) X = alpha B (and variants)
 *
 * All use MKL's MKL_LAYOUT + MKL_COMPACT_PACK interface, so they drop into the
 * MKL compact ecosystem, but are backed by this project's own portable SIMD
 * kernels rather than MKL's. Together they factor and solve batched systems
 * entirely in the compact format, with no MKL compute kernel.
 *
 * cqr_mkl_?ormqr_compact is the missing mkl_?ormqr_compact: it multiplies a
 * Compact-format batch of general matrices C by the orthogonal factor Q (or
 * Q^T), filling the gap between MKL's compact QR factorization and the
 * application of its reflectors. cqr_mkl_?geqrf_compact and cqr_mkl_?potrf_compact
 * are portable, open alternatives to mkl_?geqrf_compact / mkl_?potrf_compact
 * producing the QR and Cholesky factors -- signature- and storage-compatible, so
 * they mix freely with MKL's native compact routines. cqr_mkl_?trsm_compact is a
 * portable, open alternative to mkl_?trsm_compact (identical signature), the step
 * that closes a batched solve, so it runs end to end with no MKL compute kernel.
 * The API mirrors MKL's native compact ecosystem (MKL_LAYOUT + MKL_COMPACT_PACK);
 * see the full parameter reference in the per-routine design docs
 * (cqr_mkl_d{geqrf,ormqr,potrf,trsm}_compact_design.md).
 *
 * Typical use -- the batched AX = B solver (now MKL-compute-free):
 *     cqr_mkl_dgeqrf_compact (..., A -> H, tau);       // A = Q R
 *     cqr_mkl_dormqr_compact('L','T', ..., H, tau, B); // B := Q^T B
 *     cqr_mkl_dtrsm_compact  (..., U, R, B);           // B := R^{-1} Q^T B = X
 *
 * The matrices are passed in the same Compact buffers used elsewhere in the
 * MKL compact API: pack with mkl_?gepack_compact, query/obtain the opaque
 * `format` from mkl_get_format_compact(), and pass the total batch size `nm`.
 * C is overwritten with op(Q)*C. With lwork = -1 the call is a workspace query
 * returning the optimal lwork in work[0] (this kernel needs none, so 1).
 *
 * Following the MKL Compact convention, this routine does NOT validate its
 * arguments -- compact routines skip error checking for vectorization, so the
 * caller is responsible for passing consistent parameters. `info` is a single
 * scalar (MKL leaves the compact info reserved), set to 0 on success.
 *
 * Workspace: give each routine a work array sized from ITS OWN lwork = -1 query.
 * Different routines need different amounts -- cqr's kernels need none (their
 * query returns 1); MKL's mkl_?geqrf_compact needs ~n*V -- so never size one
 * routine's work from another's query, and never share a single work buffer
 * across, say, geqrf and ormqr. Because compact routines skip argument checking,
 * an undersized work array is undefined behavior: harmless on some MKL builds,
 * silent heap corruption on others.
 *
 * Supported arguments: layout = MKL_COL_MAJOR or MKL_ROW_MAJOR,
 * side = 'L'/'l' (op(Q) C) or 'R'/'r' (C op(Q)), trans = 'N'/'n' (Q) or
 * 'T'/'t'/'C'/'c' (Q^T), for FP64 and FP32.
 *
 * Signature: exactly LAPACK ?ormqr plus the three arguments MKL's compact
 * routines add (layout, format, nm) -- no extras. As in ?ormqr, A is the
 * order-s reflector batch (s = m for side='L', n for side='R') dimensioned
 * (ldap, k): only its first k columns are read, and the compact buffer must be
 * *packed with exactly k columns*, so each matrix occupies ldap*k elements and
 * the group stride is ldap*k*V. For square and tall factors k equals the
 * factored column count, so the buffer returned by mkl_?geqrf_compact is
 * consumed directly. A wide factor (more columns than reflectors, k < cols)
 * must pack only its k reflector columns; alternatively the portable
 * dormqr_compact (cqr_compact.h) takes an explicit packed-column count for the
 * non-conforming layout.
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#include "mkl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* QR factorization: A -> (R, Householder vectors) in ap, tau in taup.
 * Drop-in for mkl_?geqrf_compact (identical signature). No argument checking;
 * info is a single scalar status (0 on success). With lwork = -1 the call is a
 * workspace query returning the optimal lwork in work[0] (this kernel needs
 * none, so 1). See cqr_mkl_dgeqrf_compact_design.md. */
void cqr_mkl_dgeqrf_compact(MKL_LAYOUT layout, MKL_INT m, MKL_INT n, double *ap,
                            MKL_INT ldap, double *taup, double *work, MKL_INT lwork,
                            MKL_INT *info, MKL_COMPACT_PACK format, MKL_INT nm);

void cqr_mkl_sgeqrf_compact(MKL_LAYOUT layout, MKL_INT m, MKL_INT n, float *ap,
                            MKL_INT ldap, float *taup, float *work, MKL_INT lwork,
                            MKL_INT *info, MKL_COMPACT_PACK format, MKL_INT nm);

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
 * matrices in Compact format: A = L L^T (MKL_LOWER) or A = U^T U (MKL_UPPER).
 * Drop-in for mkl_?potrf_compact (identical signature). On exit the named
 * triangle holds the factor; the other is untouched. No workspace (no
 * work/lwork), no argument checking, and no SPD test -- a non-SPD lane poisons
 * itself with NaN/Inf, not info = j. info is a scalar status (0 ok, -1 for an
 * unrecognized format). See cqr_mkl_dpotrf_compact_design.md. */
void cqr_mkl_dpotrf_compact(MKL_LAYOUT layout, MKL_UPLO uplo, MKL_INT n, double *ap,
                            MKL_INT ldap, MKL_INT *info, MKL_COMPACT_PACK format,
                            MKL_INT nm);

void cqr_mkl_spotrf_compact(MKL_LAYOUT layout, MKL_UPLO uplo, MKL_INT n, float *ap,
                            MKL_INT ldap, MKL_INT *info, MKL_COMPACT_PACK format,
                            MKL_INT nm);

/* Triangular solve with multiple right-hand sides. For every matrix in the
 * batch, solves in place
 *
 *     op(A) X = alpha B    (side = MKL_LEFT)   or
 *     X op(A) = alpha B    (side = MKL_RIGHT),
 *
 * where:
 *   - A is the order-s (s = m for LEFT, n for RIGHT) unit/non-unit, upper/lower
 *     triangular factor,
 *   - op(A) = A (MKL_NOTRANS) or A^T (MKL_TRANS / MKL_CONJTRANS, folded to A^T
 *     for the real types),
 *   - B (m x n) is overwritten by the solution X.
 *
 * Drop-in for mkl_?trsm_compact (identical signature): like the BLAS ?trsm it
 * batches, it takes no workspace and reports no info. It does no argument
 * checking (Compact convention) -- use dtrsm_compact / strsm_compact
 * (cqr_compact.h) for LAPACK/BLAS-style validation. See
 * cqr_mkl_dtrsm_compact_design.md. */
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

namespace cqr {
namespace detail {

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

} /* namespace detail */
} /* namespace cqr */

#endif /* __cplusplus */

#endif /* CQR_MKL_EXT_H */
