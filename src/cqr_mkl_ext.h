#ifndef CQR_MKL_EXT_H
#define CQR_MKL_EXT_H

/* cqr_mkl_ext.h -- the MKL Compact routines this project adds that are
 * missing from Intel MKL's own compact API.
 *
 * cqr_mkl_?ormqr_compact -- apply Q (or Q^T) of a Compact-format QR
 *
 * This is the missing mkl_?ormqr_compact. It multiplies a Compact-format
 * batch of general matrices C by the orthogonal factor Q (or Q^T) produced by
 * mkl_?geqrf_compact, filling the gap between the compact QR factorization and
 * the application of its reflectors. The API mirrors MKL's native compact
 * ecosystem (MKL_LAYOUT + MKL_COMPACT_PACK); see the full parameter reference
 * in cqr_mkl_dormqr_compact_design.md.
 *
 * Typical use -- the batched AX = B solver:
 *     mkl_dgeqrf_compact (..., A -> H, tau);          // A = Q R
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
 * Packed column count of A (`ncols_a`). A is the reflector batch of order
 * s = m (side='L') or n (side='R'); op(Q) reads only its first k columns, but
 * the compact buffer is addressed by its *packed* column count, which is the
 * column count of the original factored matrix. For square/tall factors that
 * equals k (the common case, and what the AX=B pipeline uses); for a wide
 * factor (more original columns than reflectors) it exceeds k. ncols_a must be
 * the value passed to mkl_?gepack_compact / mkl_?get_size_compact when A was
 * packed, and must satisfy ncols_a >= k.
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#include "mkl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

void cqr_mkl_dormqr_compact(MKL_LAYOUT layout, char side, char trans,
                            MKL_INT m, MKL_INT n, MKL_INT k,
                            const double *ap, MKL_INT ldap, MKL_INT ncols_a,
                            const double *taup,
                            double *cp, MKL_INT ldcp,
                            double *work, MKL_INT lwork, MKL_INT *info,
                            MKL_COMPACT_PACK format, MKL_INT nm);

void cqr_mkl_sormqr_compact(MKL_LAYOUT layout, char side, char trans,
                            MKL_INT m, MKL_INT n, MKL_INT k,
                            const float *ap, MKL_INT ldap, MKL_INT ncols_a,
                            const float *taup,
                            float *cp, MKL_INT ldcp,
                            float *work, MKL_INT lwork, MKL_INT *info,
                            MKL_COMPACT_PACK format, MKL_INT nm);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

/* ------------------------------------------------------------------ *
 * C++-only internal helpers (for this project's implementation and    *
 * tests/examples; not part of the FFI-stable C surface above). These  *
 * are MKL-dependent on purpose -- the MKL-free headers (cqr_compact.h, *
 * cqr_compact.hpp) carry nothing from this file.                      *
 * ------------------------------------------------------------------ */

#include "mkl_service.h"   /* mkl_malloc / mkl_free */

#include <cstddef>
#include <memory>
#include <new>

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

/* Stateless deleter calling mkl_free -- usable as a zero-size unique_ptr
 * deleter, so the owning handle is no larger than a bare pointer. */
struct mkl_deleter {
    void operator()(void *p) const noexcept { mkl_free(p); }
};

/* Owning handle for an mkl_malloc'd buffer of T, freed automatically. */
template <typename T>
using mkl_buffer = std::unique_ptr<T[], mkl_deleter>;

/* Allocate `bytes` of `align`-aligned storage via mkl_malloc, wrapped for
 * RAII. Sized in bytes (not elements) to match mkl_?get_size_compact(),
 * whose return value already accounts for the compact format's interleave
 * padding. Throws std::bad_alloc on allocation failure. */
template <typename T>
mkl_buffer<T> mkl_alloc_bytes(std::size_t bytes, int align = 64)
{
    void *p = mkl_malloc(bytes, align);
    if (!p) throw std::bad_alloc();
    return mkl_buffer<T>(static_cast<T *>(p));
}

} /* namespace detail */
} /* namespace cqr */

#endif /* __cplusplus */

#endif /* CQR_MKL_EXT_H */
