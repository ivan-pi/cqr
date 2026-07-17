#ifndef CQR_MKL_EXT_H
#define CQR_MKL_EXT_H

/* cqr_mkl_ext.h -- batched QR for matrices in Intel MKL's Compact format.
 *
 *   cqr_mkl_?geqrf_compact -- QR factorization of a Compact-format batch
 *   cqr_mkl_?ormqr_compact -- apply Q (or Q^T) of a Compact-format QR
 *
 * Both use MKL's MKL_LAYOUT + MKL_COMPACT_PACK interface, so they drop into the
 * MKL compact ecosystem, but are backed by this project's own portable SIMD
 * kernels rather than MKL's.
 *
 * cqr_mkl_?ormqr_compact is the missing mkl_?ormqr_compact: it multiplies a
 * Compact-format batch of general matrices C by the orthogonal factor Q (or
 * Q^T), filling the gap between MKL's compact QR factorization and the
 * application of its reflectors. cqr_mkl_?geqrf_compact is a portable, open
 * alternative to mkl_?geqrf_compact producing those reflectors -- signature- and
 * storage-compatible, so the two can be mixed freely with MKL's native compact
 * routines. Together (?geqrf -> ?ormqr -> mkl_?trsm_compact) they form an
 * all-open Compact-format QR pipeline. The API mirrors MKL's native compact
 * ecosystem (MKL_LAYOUT + MKL_COMPACT_PACK); see the full parameter reference
 * in cqr_mkl_dormqr_compact_design.md and cqr_mkl_dgeqrf_compact_design.md.
 *
 * Typical use -- the batched AX = B solver:
 *     cqr_mkl_dgeqrf_compact (..., A -> H, tau);       // A = Q R
 *     cqr_mkl_dormqr_compact('L','T', ..., H, tau, B); // B := Q^T B
 *     mkl_dtrsm_compact  (..., U, R, B);              // B := R^{-1} Q^T B = X
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
void cqr_mkl_dgeqrf_compact(MKL_LAYOUT layout, MKL_INT m, MKL_INT n,
                            double *ap, MKL_INT ldap, double *taup,
                            double *work, MKL_INT lwork, MKL_INT *info,
                            MKL_COMPACT_PACK format, MKL_INT nm);

void cqr_mkl_sgeqrf_compact(MKL_LAYOUT layout, MKL_INT m, MKL_INT n,
                            float *ap, MKL_INT ldap, float *taup,
                            float *work, MKL_INT lwork, MKL_INT *info,
                            MKL_COMPACT_PACK format, MKL_INT nm);

void cqr_mkl_dormqr_compact(MKL_LAYOUT layout, char side, char trans,
                            MKL_INT m, MKL_INT n, MKL_INT k,
                            const double *ap, MKL_INT ldap,
                            const double *taup,
                            double *cp, MKL_INT ldcp,
                            double *work, MKL_INT lwork, MKL_INT *info,
                            MKL_COMPACT_PACK format, MKL_INT nm);

void cqr_mkl_sormqr_compact(MKL_LAYOUT layout, char side, char trans,
                            MKL_INT m, MKL_INT n, MKL_INT k,
                            const float *ap, MKL_INT ldap,
                            const float *taup,
                            float *cp, MKL_INT ldcp,
                            float *work, MKL_INT lwork, MKL_INT *info,
                            MKL_COMPACT_PACK format, MKL_INT nm);

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
template <typename T>
inline int vlen_for_format(MKL_COMPACT_PACK format)
{
    int bytes;
    switch (format) {
    case MKL_COMPACT_SSE:    bytes = 16; break;
    case MKL_COMPACT_AVX:    bytes = 32; break;
    case MKL_COMPACT_AVX512: bytes = 64; break;
    default:                 return 0;
    }
    return bytes / static_cast<int>(sizeof(T));
}

} /* namespace detail */
} /* namespace cqr */

#endif /* __cplusplus */

#endif /* CQR_MKL_EXT_H */
