/* cqr_compact_common.hpp
 *
 * Shared machinery for the compact (interleaved-batch) kernels: the V-wide
 * packed vector type and the small lane-wise / addressing helpers built on it.
 * Included by every routine header (geqrf, ormqr, potrf, trsm) so the pack type
 * and views are defined once:
 *
 *   pack<T,V>                    -- the V-wide SIMD element (GNU vector type).
 *   vsqrt / broadcast            -- lane-wise helpers (geqrf/potrf norms and
 *                                   pivots, trsm's alpha scaling).
 *   BatchView                    -- a strided 2-D view of one group of V
 *                                   interleaved matrices.
 *   make_view / make_const_view  -- reinterpret a packed T buffer as that view.
 *
 * V is the compact-format interleave width (the number of matrices whose
 * element (i,j) is stored contiguously). It does NOT need to match the hardware
 * vector width:
 *   x86:   V*sizeof(T) = 16/32/64 bytes maps exactly to XMM/YMM/ZMM.
 *   NEON:  128-bit registers; V=4 or V=8 doubles lower to short unrolled bursts
 *          of 2/4 independent fmla v*.2d chains, which wide cores (Apple
 *          M-series) execute very well.
 *   SVE:   compile fixed-width with -msve-vector-bits=512 on A64FX to map V=8
 *          doubles onto one SVE register.
 *
 * Assisted-by: Claude:claude-fable-5 Claude:claude-opus-4.8
 */

#ifndef CQR_COMPACT_COMMON_HPP
#define CQR_COMPACT_COMMON_HPP

#include <cstddef>
#include <cmath>

namespace cqr {
namespace detail {

/* ------------------------------------------------------------------ */
/* pack<T,V>::type : the V-wide SIMD element                          */
/* ------------------------------------------------------------------ */

/* Define the V-wide element as a GNU vector type when the compiler provides
 * the vector_size and may_alias attributes, detected directly via
 * __has_attribute (itself guarded for preprocessors that predate it). A
 * compiler that supplies these attributes -- GCC, Clang, Intel icpx/icpc --
 * uses the vector type; any other stops at the #error below. */
#if defined(__has_attribute)
#if __has_attribute(vector_size) && __has_attribute(__may_alias__)
#define CQR_HAS_GNU_VECTORS 1
#endif
#endif

#if defined(CQR_HAS_GNU_VECTORS)

template <typename T, int V> struct pack {
    /* GNU vector_size requires a power-of-two byte width; the supported
     * interleave widths are 2/4/8/16, matching the C API. Check it here -- the
     * single chokepoint -- so a bad width fails with this message instead of a
     * cryptic error inside the attribute instantiation. */
    static_assert(V == 2 || V == 4 || V == 8 || V == 16,
                  "interleave width V must be 2, 4, 8, or 16");
    /* aligned(alignof(T)) relaxes the alignment requirement so the type
     * is valid on any T-aligned buffer (unaligned vector loads are free
     * on all modern hardware); may_alias exempts it from strict-aliasing
     * violations when viewing a plain T array. */
    using type
        __attribute__((vector_size(V * sizeof(T)), aligned(alignof(T)), may_alias)) = T;
};

#else
#error "cqr compact kernels require the GNU vector extensions " \
       "(__attribute__((vector_size)) with may_alias); compile with a " \
       "compiler that supports them (GCC, Clang, Intel icpx/icpc). These " \
       "attributes are available under strict -std=c++17, not only GNU mode."
#endif

/* ------------------------------------------------------------------ */
/* Lane-wise vector helpers shared by the compact kernels.             */
/*                                                                     */
/* These V-wide helpers take and return their vectors by reference.    */
/* Passing a GNU vector by value would, without -march, commit the     */
/* base-ISA vector argument/return ABI, which GCC and Clang (rightly)  */
/* flag via -Wpsabi; a reference is just a pointer, so there is no such */
/* boundary -- and once inlined the codegen is identical -- keeping the */
/* build warning-clean with no compiler flag. Results are written      */
/* through an out-parameter (named first).                             */
/* ------------------------------------------------------------------ */

/* r := sqrt(x), lane-wise. The short loop lowers to one vsqrt* on GCC/Clang; it
 * runs once per column, negligible next to the O(n^2)/O(n^3) vector arithmetic.
 * Used by geqrf's larfg (column norm) and potrf's pivot. */
template <typename T, int V>
inline void vsqrt(typename pack<T, V>::type &r,
                  const typename pack<T, V>::type &x) noexcept
{
    for (int v = 0; v < V; ++v)
        r[v] = std::sqrt(x[v]);
}

/* v := x broadcast to all V lanes. GNU vector types broadcast a scalar in
 * arithmetic but not in assignment (`v = x;` is a compile error); `x - VT{}`
 * subtracts an all-zero vector, leaving x in every lane (and, unlike `VT{} + x`,
 * it preserves the sign of a zero x). Used by trsm to scale B by alpha. */
template <typename T, int V>
inline void broadcast(typename pack<T, V>::type &v, T x) noexcept
{
    v = x - typename pack<T, V>::type{};
}

/* ------------------------------------------------------------------ */
/* BatchView: a strided 2-D view of one group of V interleaved matrices*/
/*                                                                     */
/* The element type is the V-wide pack (use a const pack for read-only */
/* operands such as the reflector batch A). Indices are in elements;   */
/* strides are in units of the V-wide pack, so one BatchView addresses */
/* element (i,p) of every matrix in the group at once. The two axes    */
/* are named for their role in the reflector sweep, not for row/col:   */
/*   special -- the axis the Householder vector runs along             */
/*              (rows of A and, for side='L', of C; columns for 'R'),  */
/*   panel   -- the orthogonal axis, register-blocked 4 at a time.     */
/* ------------------------------------------------------------------ */

template <typename VT, typename Int = int> struct BatchView {
    VT *const data = nullptr;
    const Int special = 0; /* stride along the swept (reflector) axis */
    const Int panel = 0;   /* stride along the orthogonal panel axis  */

    /* The batch and matrix dimensions fit in Int (compact targets many small
     * matrices), so the element offset is formed in Int with no widening of the
     * induction variables -- which keeps the strided sweep vectorizable. The
     * per-group base offset, which can exceed Int, is applied to the pointer by
     * the caller before the view is built. */
    inline VT &operator()(Int i, Int p) const noexcept
    {
        return data[i * special + p * panel];
    }
};

/* Reinterpret a packed T buffer as a group view of V-wide pack elements. The
 * strides are per-matrix leading dimensions (in pack units), so they take the
 * kernel's Int like the dimensions do. */
template <typename T, int V, typename Int = int>
BatchView<const typename pack<T, V>::type, Int> make_const_view(const T *p, Int special,
                                                                Int panel) noexcept
{
    using VT = typename pack<T, V>::type;
    return {reinterpret_cast<const VT *>(p), special, panel};
}
template <typename T, int V, typename Int = int>
BatchView<typename pack<T, V>::type, Int> make_view(T *p, Int special, Int panel) noexcept
{
    using VT = typename pack<T, V>::type;
    return {reinterpret_cast<VT *>(p), special, panel};
}

} /* namespace detail */
} /* namespace cqr */

#endif /* CQR_COMPACT_COMMON_HPP */
