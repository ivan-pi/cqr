/* cqr_compact_common.hpp
 *
 * Shared machinery for the compact (interleaved-batch) kernels:
 *
 *   pack<T,V>             -- the V-wide SIMD element (a GNU vector type).
 *   vsqrt / broadcast     -- lane-wise helpers (geqrf/potrf pivots, trsm's alpha).
 *   BatchView             -- a strided 2-D view of one group of V interleaved
 *                            matrices; every kernel addresses its operands
 *                            through it, so one kernel serves every layout.
 *   make_view / make_const_view -- reinterpret a packed T buffer as such a view.
 *   for_vlen              -- runtime interleave width -> compile-time V.
 *
 * V is the compact-format interleave width (the number of matrices whose
 * element (i,j) is stored contiguously). It does NOT need to match the hardware
 * vector width:
 *   x86:   V*sizeof(T) = 16/32/64 bytes maps exactly to XMM/YMM/ZMM.
 *   NEON:  128-bit registers; V=4 or V=8 doubles lower to short unrolled bursts
 *          of 2/4 independent fmla v*.2d chains, which wide cores execute well.
 *   SVE:   compile fixed-width with -msve-vector-bits=512 on A64FX to map V=8
 *          doubles onto one SVE register.
 *
 * Assisted-by: Claude:claude-fable-5 Claude:claude-opus-4.8
 */

#ifndef CQR_COMPACT_COMMON_HPP
#define CQR_COMPACT_COMMON_HPP

#include <cstddef>
#include <cmath>
#include <type_traits>

namespace cqr {
namespace detail {

/* ------------------------------------------------------------------ */
/* pack<T,V>::type : the V-wide SIMD element                          */
/* ------------------------------------------------------------------ */

/* The V-wide element is a GNU vector type, available wherever the compiler
 * provides the vector_size and may_alias attributes (GCC, Clang, Intel icpx),
 * under strict -std=c++17 as well as GNU mode. */
#if defined(__has_attribute)
#if __has_attribute(vector_size) && __has_attribute(__may_alias__)
#define CQR_HAS_GNU_VECTORS 1
#endif
#endif

#if defined(CQR_HAS_GNU_VECTORS)

template <typename T, int V> struct pack {
    /* GNU vector_size needs a power-of-two byte width; the supported interleave
     * widths are 2/4/8/16, matching the C API. Checked here, the single
     * chokepoint, so a bad width fails with this message. */
    static_assert(V == 2 || V == 4 || V == 8 || V == 16,
                  "interleave width V must be 2, 4, 8, or 16");
    /* aligned(alignof(T)) makes the type valid on any T-aligned buffer
     * (unaligned vector loads are free on all modern hardware); may_alias
     * exempts it from strict-aliasing violations when viewing a plain T array. */
    using type
        __attribute__((vector_size(V * sizeof(T)), aligned(alignof(T)), may_alias)) = T;
};

#else
#error "cqr compact kernels require the GNU vector extensions " \
       "(__attribute__((vector_size)) with may_alias); compile with GCC, Clang, " \
       "or Intel icpx."
#endif

/* ------------------------------------------------------------------ */
/* Lane-wise vector helpers.                                           */
/*                                                                     */
/* Vectors are passed by reference: passing a GNU vector by value would */
/* commit the base-ISA vector argument ABI without -march, which GCC   */
/* and Clang flag via -Wpsabi. Once inlined the codegen is identical.  */
/* Results are written through an out-parameter (named first).        */
/* ------------------------------------------------------------------ */

/* r := sqrt(x), lane-wise. The short loop lowers to one vsqrt* on GCC/Clang. */
template <typename T, int V>
inline void vsqrt(typename pack<T, V>::type &r,
                  const typename pack<T, V>::type &x) noexcept
{
    for (int v = 0; v < V; ++v)
        r[v] = std::sqrt(x[v]);
}

/* v := x broadcast to all V lanes. GNU vector types broadcast a scalar in
 * arithmetic but not in assignment; `x - VT{}` subtracts an all-zero vector,
 * leaving x in every lane (and, unlike `VT{} + x`, preserving a zero's sign). */
template <typename T, int V>
inline void broadcast(typename pack<T, V>::type &v, T x) noexcept
{
    v = x - typename pack<T, V>::type{};
}

/* ------------------------------------------------------------------ */
/* BatchView: a strided 2-D view of one group of V interleaved matrices*/
/*                                                                     */
/* A(i,j) = data[i*si + j*sj], addressing element (i,j) of all V       */
/* matrices of the group at once (the element type is the V-wide pack; */
/* strides are in packs). A column-major group has si = 1, sj = ldap;  */
/* row-major swaps them. Because the kernels only ever go through this  */
/* view, one kernel per routine covers every layout -- and ormqr's     */
/* side='R' is just the view of C^T. Offsets are formed in Int (the    */
/* batch and matrix dimensions fit int), which keeps the strided sweep  */
/* vectorizable; the per-group base offset, which can exceed Int, is   */
/* applied to the pointer before the view is built.                    */
/*                                                                     */
/* The view is templated on (T, V), not on the pack type, on purpose:  */
/* the pack's relaxed aligned(alignof(T)) lives on its typedef, and     */
/* clang strips typedef alignment from class template *arguments* --   */
/* a BatchView<pack<T,V>::type> would see a naturally aligned vector   */
/* and emit aligned loads/stores that fault on 16-byte-aligned buffers  */
/* (GCC keeps the typedef alignment either way). Naming the typedef    */
/* inside the class keeps every access unaligned on both compilers.    */
/* ------------------------------------------------------------------ */

template <typename T, int V, bool Const> struct view_elem {
    using type = typename pack<T, V>::type;
};
template <typename T, int V> struct view_elem<T, V, true> {
    using type = const typename pack<T, V>::type;
};

template <typename T, int V, typename Int = int, bool Const = false> struct BatchView {
    using VT = typename view_elem<T, V, Const>::type;

    VT *data;
    Int si; /* stride along the first index  */
    Int sj; /* stride along the second index */

    inline VT &operator()(Int i, Int j) const noexcept { return data[i * si + j * sj]; }

    BatchView<T, V, Int, true> as_const() const noexcept { return {data, si, sj}; }
};
template <typename T, int V, typename Int = int>
using ConstBatchView = BatchView<T, V, Int, true>;

/* Reinterpret a packed T buffer as a group view of V-wide pack elements (a const
 * view for read-only operands such as the reflector batch A). */
template <typename T, int V, typename Int = int>
BatchView<T, V, Int> make_view(T *p, Int si, Int sj) noexcept
{
    return {reinterpret_cast<typename pack<T, V>::type *>(p), si, sj};
}
template <typename T, int V, typename Int = int>
ConstBatchView<T, V, Int> make_const_view(const T *p, Int si, Int sj) noexcept
{
    return {reinterpret_cast<const typename pack<T, V>::type *>(p), si, sj};
}

/* Runtime interleave width -> compile-time instantiation: calls f with a
 * std::integral_constant<int, V> for V in {2, 4, 8, 16} and returns true, or
 * returns false (f not called) for any other width. V is the compact interleave
 * width, not necessarily one hardware register: every width is valid for both
 * scalar types (V=16 doubles is a legal 1024-bit vector lowered to two AVX-512
 * ops); MKL's formats only ever select 2/4/8 for double and 4/8/16 for float. */
template <typename F> bool for_vlen(int V, F &&f)
{
    switch (V) {
    case 2: f(std::integral_constant<int, 2>{}); return true;
    case 4: f(std::integral_constant<int, 4>{}); return true;
    case 8: f(std::integral_constant<int, 8>{}); return true;
    case 16: f(std::integral_constant<int, 16>{}); return true;
    default: return false;
    }
}

} /* namespace detail */
} /* namespace cqr */

#endif /* CQR_COMPACT_COMMON_HPP */
