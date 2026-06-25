#ifndef CQR_MKL_ALLOC_H
#define CQR_MKL_ALLOC_H

/* cqr_mkl_alloc.h -- optional RAII buffer helpers for MKL Compact-format work.
 *
 * mkl_alloc_bytes() and the mkl_buffer handle wrap mkl_malloc / mkl_free to
 * allocate the aligned, interleave-padded buffers that the MKL Compact API
 * consumes (size them with mkl_?get_size_compact). The buffer owns its
 * storage and frees it automatically.
 *
 * These helpers call MKL runtime functions, so a translation unit that
 * includes this header must link MKL (e.g. CMake target MKL::Compact).
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#include "mkl_service.h"   /* mkl_malloc / mkl_free */

#include <cstddef>
#include <memory>
#include <new>

namespace cqr {
namespace detail {

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

#endif /* CQR_MKL_ALLOC_H */
