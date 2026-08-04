/* cqr_geqrf_compact_omp.hpp
 *
 * Compact (interleaved-batch) QR factorization -- OpenMP SIMD variant.
 *
 * Same math and same storage convention as cqr_geqrf_compact.hpp (the unblocked
 * LAPACK geqr2: dlarfg to build each reflector, dlarf to apply it), but a
 * different vectorization mechanism: instead of GNU vector types, the batch is
 * driven by an OpenMP SIMD *outer-loop* vectorization:
 *
 *     #pragma omp simd simdlen(V)
 *     for (int v = 0; v < V; ++v) { ... one scalar geqr2 for matrix (g*V+v) ... }
 *
 * The inner code is the ordinary scalar geqr2 written from the perspective of a
 * single matrix element A(i,j): the vector-types kernel gets that single-element
 * view for free (arithmetic operators lift over the whole pack), whereas here we
 * recover it with a helper macro CQR_A(i,j) that closes over the SIMD lane index
 * v and the layout strides. The V lanes carry V independent matrices, so the
 * loop body has no cross-lane dependence -- exactly the property omp simd
 * asserts -- and every per-lane scalar (x0, tail, tau, w, ...) is an automatic
 * SIMD-private. Because the whole batch shares one (m, n), the control flow
 * (loop bounds, the pivot index kk) is uniform across lanes, which is what lets
 * the compiler vectorize the outer loop cleanly.
 *
 * Compact layout puts element (i,j) of the V matrices in a group contiguously,
 * so for a fixed (i,j) the V lanes v = 0..V-1 are unit-stride in memory: the
 * omp-simd load/store of CQR_A(i,j) across the lane loop is one contiguous
 * vector access, the same traffic pattern the vector-types kernel issues by
 * hand. V here (2/4/8/16) is the interleave width, not necessarily one hardware
 * register; the compiler maps simdlen(V) to registers or short unrolled bursts.
 *
 * Correctness does not depend on -fopenmp-simd: without it the pragma is ignored
 * and each group runs as a plain scalar loop over its V matrices (still a valid
 * factorization). The flag is what turns the outer loop into SIMD code; only
 * throughput rides on it. See cqr_mkl_dgeqrf_compact_design.md for the algorithm
 * and the branch-free larfg rationale.
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#ifndef CQR_GEQRF_COMPACT_OMP_HPP
#define CQR_GEQRF_COMPACT_OMP_HPP

#include <cstddef>
#include <cassert>
#include <cmath>
#include <type_traits>

namespace cqr {
namespace detail {
namespace omp_simd {

/* ------------------------------------------------------------------
 * One group of V interleaved matrices, column- or row-major.
 *
 * a_ points at element (0,0) of the group. The layout is expressed by two
 * element strides (in units of the V-wide slot): rs steps one row, cs steps one
 * column. Column-major has rs=1, cs=ldap (a column is contiguous); row-major
 * swaps them. The lane index v is the innermost address term, coefficient 1, so
 * consecutive lanes are consecutive doubles.
 * ------------------------------------------------------------------ */

template <typename T, int V, typename Int = int>
void geqrf_compact_group_omp(bool rowmajor, Int m, Int n, T *a_, Int ldap, T *tau_)
{
    static_assert(std::is_floating_point<T>::value,
                  "geqrf_compact is defined for real float/double");
    assert(ldap >= (rowmajor ? n : m));

    const Int k = (m < n) ? m : n;
    const std::size_t rs = rowmajor ? static_cast<std::size_t>(ldap) : 1; /* row step  */
    const std::size_t cs = rowmajor ? 1 : static_cast<std::size_t>(ldap); /* col step  */

/* Single-element view of matrix (g*V + v), closing over the ambient lane v. This
 * is the macro the design note calls for: it hands the scalar geqr2 below the
 * same A(i,j) it would write against a plain 2-D array, with the compact
 * interleave (the *V + v) folded into the address. Undefined at the end of the
 * function so the name does not leak. */
#define CQR_A(i, j)                                                                      \
    a_[((static_cast<std::size_t>(i)) * rs + (static_cast<std::size_t>(j)) * cs) * V +   \
       static_cast<std::size_t>(v)]

#pragma omp simd simdlen(V)
    for (int v = 0; v < V; ++v) {
        for (Int kk = 0; kk < k; ++kk) {
            /* build reflector H(kk) from column kk, rows kk..m-1 (branch-free
             * larfg: the xnorm==0 branch folds into the `has` select, keyed on
             * the below-diagonal norm exactly as dlarfg). */
            const T x0 = CQR_A(kk, kk);
            T tail = T(0);
            for (Int i = kk + 1; i < m; ++i) {
                const T a = CQR_A(i, kk);
                tail += a * a;
            }
            const T norm = std::sqrt(x0 * x0 + tail);
            const T beta = (x0 >= T(0)) ? -norm : norm; /* -copysign(norm, x0) */
            const bool has = (tail > T(0));             /* anything to zero?    */
            const T t = has ? (beta - x0) / beta : T(0);
            const T inv = has ? T(1) / (x0 - beta) : T(0);
            tau_[static_cast<std::size_t>(kk) * V + static_cast<std::size_t>(v)] = t;
            for (Int i = kk + 1; i < m; ++i)
                CQR_A(i, kk) = CQR_A(i, kk) * inv; /* reflector body */
            CQR_A(kk, kk) = has ? beta : x0;       /* R diagonal     */

            /* apply H(kk) = I - t v v^T (v(kk)=1 implicit) to trailing columns */
            for (Int j = kk + 1; j < n; ++j) {
                T w = CQR_A(kk, j);
                for (Int i = kk + 1; i < m; ++i)
                    w += CQR_A(i, kk) * CQR_A(i, j);
                CQR_A(kk, j) -= t * w;
                w *= t; /* fold tau into w */
                for (Int i = kk + 1; i < m; ++i)
                    CQR_A(i, j) -= CQR_A(i, kk) * w;
            }
        }
    }

#undef CQR_A
}

/* ------------------------------------------------------------------
 * All groups, column- or row-major. Group strides match the vector-types
 * geqrf_compact_general so the two kernels are interchangeable behind the same
 * compact buffer. A padded partial final group is processed too (identity
 * factors to tau = 0, a no-op), so the lane loop always runs full width.
 * ------------------------------------------------------------------ */

template <typename T, int V, typename Int = int>
void geqrf_compact_general_omp(bool rowmajor, Int m, Int n, T *ap, Int ldap, T *taup,
                               Int nm)
{
    assert(nm >= 1 && m >= 0 && n >= 0);

    const Int k = (m < n) ? m : n;
    const Int ngroups = (nm + V - 1) / V;
    const std::size_t str_t = static_cast<std::size_t>(k) * V;
    const std::size_t str_a = (rowmajor ? static_cast<std::size_t>(ldap) * m
                                        : static_cast<std::size_t>(ldap) * n) *
                              V;

    for (Int g = 0; g < ngroups; ++g)
        geqrf_compact_group_omp<T, V, Int>(rowmajor, m, n, ap + g * str_a, ldap,
                                           taup + g * str_t);
}

/* ==================================================================
 * Variant: inner-loop SIMD.
 *
 * The kernel above puts `#pragma omp simd` on the *outer* lane loop, which is
 * the natural expression of batch (outer-loop) vectorization -- but neither GCC
 * nor Clang vectorizes it: the loop body is a nest of consecutive inner loops,
 * which GCC's vectorizer rejects outright ("two or more consecutive inner
 * loops") and Clang's declines to interchange, so it runs scalar (see
 * cqr_geqrf_omp_simd_results.md). The idiom the compilers *do* vectorize is to
 * pull the lane loop *inside*, innermost, around each element operation: each
 * `#pragma omp simd simdlen(V)` loop is then a straight-line contiguous stride-1
 * sweep over the V lanes, exactly the shape auto-vectorization handles. The
 * per-element view survives -- CQR_AV(i,j) still names element (i,j) of the
 * current lane -- at the cost of small per-column stack temporaries (tail, tau,
 * w, ...) that hold one value per lane between the split simd loops.
 *
 * This is the same batch SIMD as the vector-types kernel, just spelled with the
 * compiler doing the lane packing; it is the fair "what can -fopenmp-simd
 * actually deliver" data point in the benchmark.
 * ================================================================== */

template <typename T, int V, typename Int = int>
void geqrf_compact_group_omp_inner(bool rowmajor, Int m, Int n, T *a_, Int ldap, T *tau_)
{
    static_assert(std::is_floating_point<T>::value,
                  "geqrf_compact is defined for real float/double");
    assert(ldap >= (rowmajor ? n : m));

    const Int k = (m < n) ? m : n;
    const std::size_t rs = rowmajor ? static_cast<std::size_t>(ldap) : 1;
    const std::size_t cs = rowmajor ? 1 : static_cast<std::size_t>(ldap);

/* Element (i,j) of lane v -- explicit lane index, since here v is the innermost
 * loop variable rather than the ambient one. */
#define CQR_AV(i, j)                                                                     \
    a_[((static_cast<std::size_t>(i)) * rs + (static_cast<std::size_t>(j)) * cs) * V +   \
       static_cast<std::size_t>(v)]

    T tau[V], inv[V], w[V]; /* one value per lane, live between split simd loops */

    for (Int kk = 0; kk < k; ++kk) {
        /* larfg over the V lanes: tail[v] = below-diagonal sum of squares. */
        T tail[V], x0[V];
#pragma omp simd simdlen(V)
        for (int v = 0; v < V; ++v) {
            x0[v] = CQR_AV(kk, kk);
            tail[v] = T(0);
        }
        for (Int i = kk + 1; i < m; ++i)
#pragma omp simd simdlen(V)
            for (int v = 0; v < V; ++v) {
                const T a = CQR_AV(i, kk);
                tail[v] += a * a;
            }
#pragma omp simd simdlen(V)
        for (int v = 0; v < V; ++v) {
            const T norm = std::sqrt(x0[v] * x0[v] + tail[v]);
            const T beta = (x0[v] >= T(0)) ? -norm : norm;
            const bool has = (tail[v] > T(0));
            tau[v] = has ? (beta - x0[v]) / beta : T(0);
            inv[v] = has ? T(1) / (x0[v] - beta) : T(0);
            tau_[static_cast<std::size_t>(kk) * V + static_cast<std::size_t>(v)] = tau[v];
            CQR_AV(kk, kk) = has ? beta : x0[v]; /* R diagonal */
        }
        for (Int i = kk + 1; i < m; ++i)
#pragma omp simd simdlen(V)
            for (int v = 0; v < V; ++v)
                CQR_AV(i, kk) = CQR_AV(i, kk) * inv[v]; /* reflector body */

        /* apply H(kk) to trailing columns, one column at a time. */
        for (Int j = kk + 1; j < n; ++j) {
#pragma omp simd simdlen(V)
            for (int v = 0; v < V; ++v)
                w[v] = CQR_AV(kk, j);
            for (Int i = kk + 1; i < m; ++i)
#pragma omp simd simdlen(V)
                for (int v = 0; v < V; ++v)
                    w[v] += CQR_AV(i, kk) * CQR_AV(i, j);
#pragma omp simd simdlen(V)
            for (int v = 0; v < V; ++v) {
                CQR_AV(kk, j) -= tau[v] * w[v];
                w[v] *= tau[v]; /* fold tau into w */
            }
            for (Int i = kk + 1; i < m; ++i)
#pragma omp simd simdlen(V)
                for (int v = 0; v < V; ++v)
                    CQR_AV(i, j) -= CQR_AV(i, kk) * w[v];
        }
    }

#undef CQR_AV
}

template <typename T, int V, typename Int = int>
void geqrf_compact_general_omp_inner(bool rowmajor, Int m, Int n, T *ap, Int ldap,
                                     T *taup, Int nm)
{
    assert(nm >= 1 && m >= 0 && n >= 0);

    const Int k = (m < n) ? m : n;
    const Int ngroups = (nm + V - 1) / V;
    const std::size_t str_t = static_cast<std::size_t>(k) * V;
    const std::size_t str_a = (rowmajor ? static_cast<std::size_t>(ldap) * m
                                        : static_cast<std::size_t>(ldap) * n) *
                              V;

    for (Int g = 0; g < ngroups; ++g)
        geqrf_compact_group_omp_inner<T, V, Int>(rowmajor, m, n, ap + g * str_a, ldap,
                                                 taup + g * str_t);
}

} /* namespace omp_simd */
} /* namespace detail */
} /* namespace cqr */

#endif /* CQR_GEQRF_COMPACT_OMP_HPP */
