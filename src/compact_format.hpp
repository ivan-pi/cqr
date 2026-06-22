#ifndef COMPACT_FORMAT_HPP
#define COMPACT_FORMAT_HPP

/* Internal helpers shared between the ext_mkl_?ormqr_compact implementation
 * and the project's own test / benchmark facilities. This header is NOT part
 * of the public API -- everything lives in namespace ormqr::detail to make
 * that explicit, and users should never include it.
 */

#include "mkl_types.h"

#include <cstddef>

namespace ormqr {
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
} /* namespace ormqr */

#endif /* COMPACT_FORMAT_HPP */
