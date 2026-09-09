/* cqr_compact_sycl.hpp
 *
 * SYCL device-side building blocks for compact (interleaved-batch) QR.
 *
 * These are composable PRIMITIVES, not a batched API. Each primitive is the
 * per-lane work for ONE matrix -- slot v of compact group g -- written in the
 * SPMD "me" style. You own the launch: a single parallel_for over the batch,
 * V work-items per group with reqd_sub_group_size(V) so the V lanes of a
 * sub-group are the V interleaved matrices and their memory accesses coalesce.
 * Inside that kernel you call the primitives, and -- the point of this design --
 * you FUSE your own steps around them with no intermediate global-memory passes
 * or host<->device copies: fill / pack the matrix straight into the compact
 * slot, factor it, apply Q, solve, unpack the result, all in one kernel.
 *
 *   q.parallel_for(sycl::nd_range<1>{ngroups*V, V},
 *       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(V)]] {
 *           const int g = it.get_group(0), v = it.get_local_id(0);
 *           auto A = cqr::gpu::col_major<double, V>(ap, g, v, lda, n);
 *           for (...) A(i, j) = fill(...);                 // your pack, fused
 *           cqr::gpu::geqrf_slot<double, V>(g, v, ap, lda, tau, n, n);
 *           cqr::gpu::ormqr_slot<double, V>(g, v, ap, lda, tau, bp, ldb,
 *                                           n, nrhs, n, true);  // trans: apply Q^T
 *           cqr::gpu::trsm_upper_slot<double, V>(g, v, ap, lda, bp, ldb, n, nrhs);
 *           for (...) xout[...] = B(i, j);                 // your unpack, fused
 *       });
 *
 * Compact layout (matches mkl_?gepack_compact / the CPU cqr_compact.hpp);
 * group g = idx/V, slot v = idx%V, column-major:
 *   element (i,j) of matrix idx = p[g*ld*ncols*V + (j*ld + i)*V + v]
 * where ncols is the matrix's column count used for the group stride (n for a
 * full m x n matrix, k for a reflector panel (ld,k), nrhs for an RHS block).
 * Row-major swaps the in-matrix index roles.
 *
 * The math is the unblocked LAPACK geqr2 / dorm2r with JB=4 register blocking,
 * identical to the CPU kernels; here the "lane" is the work-item, so the
 * branch-free larfg select is an ordinary scalar branch. No sub-group
 * collectives are used (each lane owns an independent matrix); the sub-group is
 * purely the SIMD packing that makes V lanes one hardware vector.
 *
 * Header-only inline templates: include this and call the primitives from your
 * own kernel; they compile into it (no separate device library, no
 * SYCL_EXTERNAL). Real types (float/double); column-major is the tuned path,
 * geqrf also takes row-major.
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#ifndef CQR_COMPACT_SYCL_HPP
#define CQR_COMPACT_SYCL_HPP

#include <sycl/sycl.hpp>

#include <cstddef>
#include <type_traits>

namespace cqr {
namespace gpu {

/* ------------------------------------------------------------------ *
 * Views: address one matrix (slot v of group g) inside a compact buffer.
 * Cheap (a pointer + two strides); build one per operand in your kernel.
 * ET may be `const T` for read-only operands (reflectors, R).
 * ------------------------------------------------------------------ */

template <typename ET, int V>
struct matrix_view {
    ET *p;                     /* &element(0,0) of this slot */
    std::size_t istep, jstep;  /* element strides, V folded in */
    ET &operator()(int i, int j) const
    {
        return p[static_cast<std::size_t>(i) * istep +
                 static_cast<std::size_t>(j) * jstep];
    }
};

/* Column-major view. group_ncols is the per-matrix column count that sets the
 * group stride (n for an m x n matrix, k for an (ld,k) reflector panel, nrhs
 * for an RHS block). */
template <typename ET, int V>
inline matrix_view<ET, V> col_major(ET *base, int g, int v, int ld, int group_ncols)
{
    ET *p = base + static_cast<std::size_t>(g) * ld * group_ncols * V + v;
    return {p, static_cast<std::size_t>(V), static_cast<std::size_t>(ld) * V};
}

/* Row-major view. group_nrows sets the group stride (m for an m x n matrix). */
template <typename ET, int V>
inline matrix_view<ET, V> row_major(ET *base, int g, int v, int ld, int group_nrows)
{
    ET *p = base + static_cast<std::size_t>(g) * ld * group_nrows * V + v;
    return {p, static_cast<std::size_t>(ld) * V, static_cast<std::size_t>(V)};
}

/* tau(kk) of slot (g,v): taup[g*k*V + kk*V + v]. */
template <typename ET, int V>
struct tau_view {
    ET *p;
    ET &operator()(int kk) const { return p[static_cast<std::size_t>(kk) * V]; }
};
template <typename ET, int V>
inline tau_view<ET, V> tau_at(ET *base, int g, int v, int k)
{
    return {base + static_cast<std::size_t>(g) * k * V + v};
}

/* ------------------------------------------------------------------ *
 * larfg on column kk of A (rows kk..m-1), branch-free select as a scalar
 * branch. On exit A(kk,kk) is the R diagonal, A(kk+1..,kk) the reflector body
 * (implicit 1 on the diagonal), and tau_out the reflector scalar.
 * ------------------------------------------------------------------ */
template <typename T, int V>
inline void larfg_column(matrix_view<T, V> A, int kk, int m, T &tau_out)
{
    const T x0 = A(kk, kk);
    T tail = T(0);
    for (int i = kk + 1; i < m; ++i) {
        const T a = A(i, kk);
        tail += a * a;
    }
    const T norm = sycl::sqrt(x0 * x0 + tail);
    const T beta = (x0 >= T(0)) ? -norm : norm; /* -copysign(norm, x0) */
    if (tail > T(0)) {
        const T inv = T(1) / (x0 - beta);
        for (int i = kk + 1; i < m; ++i)
            A(i, kk) = A(i, kk) * inv;
        A(kk, kk) = beta;
        tau_out = (beta - x0) / beta;
    } else { /* column already zeroed: no reflector */
        A(kk, kk) = x0;
        tau_out = T(0);
    }
}

/* ------------------------------------------------------------------ *
 * geqrf_slot: QR-factorize matrix (g,v) in place (unblocked geqr2, JB=4
 * trailing-column update). On exit ap holds R (on/above diagonal) and the
 * Householder vectors (below); taup holds the k = min(m,n) reflector scalars.
 * ------------------------------------------------------------------ */
template <typename T, int V>
inline void geqrf_slot(int g, int v, T *ap, int ldap, T *taup, int m, int n,
                       bool rowmajor = false)
{
    matrix_view<T, V> A = rowmajor ? row_major<T, V>(ap, g, v, ldap, m)
                                   : col_major<T, V>(ap, g, v, ldap, n);
    const int k = (m < n) ? m : n;
    tau_view<T, V> tau = tau_at<T, V>(taup, g, v, k);

    for (int kk = 0; kk < k; ++kk) {
        T t;
        larfg_column<T, V>(A, kk, m, t);
        tau(kk) = t;

        int j = kk + 1;
        for (; j + 4 <= n; j += 4) { /* apply H(kk) to 4 trailing columns */
            T w0 = A(kk, j), w1 = A(kk, j + 1), w2 = A(kk, j + 2), w3 = A(kk, j + 3);
            for (int i = kk + 1; i < m; ++i) {
                const T av = A(i, kk);
                w0 += av * A(i, j);
                w1 += av * A(i, j + 1);
                w2 += av * A(i, j + 2);
                w3 += av * A(i, j + 3);
            }
            A(kk, j) -= t * w0;
            A(kk, j + 1) -= t * w1;
            A(kk, j + 2) -= t * w2;
            A(kk, j + 3) -= t * w3;
            w0 *= t;
            w1 *= t;
            w2 *= t;
            w3 *= t;
            for (int i = kk + 1; i < m; ++i) {
                const T av = A(i, kk);
                A(i, j) -= av * w0;
                A(i, j + 1) -= av * w1;
                A(i, j + 2) -= av * w2;
                A(i, j + 3) -= av * w3;
            }
        }
        for (; j < n; ++j) {
            T w = A(kk, j);
            for (int i = kk + 1; i < m; ++i)
                w += A(i, kk) * A(i, j);
            A(kk, j) -= t * w;
            w *= t;
            for (int i = kk + 1; i < m; ++i)
                A(i, j) -= A(i, kk) * w;
        }
    }
}

/* ------------------------------------------------------------------ *
 * ormqr_slot: apply op(Q) to matrix (g,v)'s RHS block, in place, side='L',
 * column-major (the C API contract). A is the (ldap,k) reflector panel from
 * geqrf; B is (ldbp,nrhs). trans=true applies Q^T (ascending sweep), false Q.
 * JB=4 RHS-column register blocking.
 * ------------------------------------------------------------------ */
template <typename T, int V>
inline void ormqr_slot(int g, int v, const T *ap, int ldap, const T *taup, T *bp,
                       int ldbp, int m, int nrhs, int k, bool trans)
{
    matrix_view<const T, V> A = col_major<const T, V>(ap, g, v, ldap, k);
    matrix_view<T, V> B = col_major<T, V>(bp, g, v, ldbp, nrhs);
    tau_view<const T, V> tau = tau_at<const T, V>(taup, g, v, k);
    const bool fwd = trans;

    for (int s = 0; s < k; ++s) {
        const int kk = fwd ? s : k - 1 - s;
        const T t = tau(kk);
        int j = 0;
        for (; j + 4 <= nrhs; j += 4) {
            T w0 = B(kk, j), w1 = B(kk, j + 1), w2 = B(kk, j + 2), w3 = B(kk, j + 3);
            for (int i = kk + 1; i < m; ++i) {
                const T av = A(i, kk);
                w0 += av * B(i, j);
                w1 += av * B(i, j + 1);
                w2 += av * B(i, j + 2);
                w3 += av * B(i, j + 3);
            }
            B(kk, j) -= t * w0;
            B(kk, j + 1) -= t * w1;
            B(kk, j + 2) -= t * w2;
            B(kk, j + 3) -= t * w3;
            w0 *= t;
            w1 *= t;
            w2 *= t;
            w3 *= t;
            for (int i = kk + 1; i < m; ++i) {
                const T av = A(i, kk);
                B(i, j) -= av * w0;
                B(i, j + 1) -= av * w1;
                B(i, j + 2) -= av * w2;
                B(i, j + 3) -= av * w3;
            }
        }
        for (; j < nrhs; ++j) {
            T w = B(kk, j);
            for (int i = kk + 1; i < m; ++i)
                w += A(i, kk) * B(i, j);
            B(kk, j) -= t * w;
            w *= t;
            for (int i = kk + 1; i < m; ++i)
                B(i, j) -= A(i, kk) * w;
        }
    }
}

/* ------------------------------------------------------------------ *
 * trsm_upper_slot: solve R X = B in place (upper-triangular back
 * substitution), R being the n x n upper triangle of the factored matrix
 * (g,v) in ap (ldap,n). Completes an AX=B solve after geqrf + apply Q^T.
 * ------------------------------------------------------------------ */
template <typename T, int V>
inline void trsm_upper_slot(int g, int v, const T *ap, int ldap, T *bp, int ldbp, int n,
                            int nrhs)
{
    matrix_view<const T, V> R = col_major<const T, V>(ap, g, v, ldap, n);
    matrix_view<T, V> B = col_major<T, V>(bp, g, v, ldbp, nrhs);
    /* Row loop outermost so the diagonal is read once per row instead of once
     * per (row, rhs). Back-substitution permits the swap: every row below i is
     * already solved for all columns before row i is reached. The division
     * itself is left alone -- hoisting a reciprocal instead would change the
     * result in the last ulp, and the sub-group variant below is documented to
     * agree with this one bit for bit. */
    for (int i = n - 1; i >= 0; --i) {
        const T d = R(i, i);
        for (int j = 0; j < nrhs; ++j) {
            T s = B(i, j);
            for (int l = i + 1; l < n; ++l)
                s -= R(i, l) * B(l, j);
            B(i, j) = s / d;
        }
    }
}

/* ================================================================== *
 * Variant 2: EXPLICIT sub-group (SIMD) primitives.
 *
 * Same Strategy A mapping -- one matrix per lane, the sub-group pinned 1-to-1 to
 * a hardware SIMD unit -- but memory is moved with explicit sub-group block
 * operations (sycl::ext::oneapi::experimental::group_load / group_store) instead
 * of per-lane indexing. The compact layout makes this exact: element (i,j) of
 * the V matrices is a contiguous V-block, so one group_load fills the whole
 * SIMD register (lane v <- matrix v), the pattern the Intel "Sub-groups and SIMD
 * Vectorization" guide recommends for coalesced, predictable SIMD codegen
 * (block reads/writes) rather than relying on the compiler to prove the per-lane
 * accesses coalesce.
 *
 * Call these from a kernel launched with reqd_sub_group_size(V); pass the
 * sub-group. Every group_load/group_store is a sub-group COLLECTIVE, so it must
 * run in convergent control flow -- fine here because the compact uniform-shape
 * rule makes every loop trip count identical across lanes, and larfg is done
 * branch-free (lane-wise selects, no divergent branch), exactly as the CPU
 * kernel's larfg_pack. No cross-lane shuffles/reductions are used: each lane's
 * matrix is independent, so the sub-group is purely the SIMD packing.
 * ================================================================== */

namespace se = sycl::ext::oneapi::experimental;

/* Block-access view of one V-matrix compact block for the whole sub-group.
 * `base` is the lane-0 address of element (0,0) (no +v -- group_load/store
 * distribute the V contiguous elements across the lanes). */
template <typename ET, int V>
struct sg_view {
    using VT = typename std::remove_const<ET>::type; /* value type of a block load */
    sycl::sub_group sg;
    ET *base;
    std::size_t istep, jstep;
    VT load(int i, int j) const
    {
        VT v;
        se::group_load(sg, base + static_cast<std::size_t>(i) * istep +
                               static_cast<std::size_t>(j) * jstep, v);
        return v;
    }
    /* store() is instantiated only for non-const ET (never called on const views). */
    void store(int i, int j, const VT &val) const
    {
        se::group_store(sg, val, base + static_cast<std::size_t>(i) * istep +
                                     static_cast<std::size_t>(j) * jstep);
    }
};

/* geqrf, explicit sub-group. Branch-free larfg (selects) so the block loads
 * stay convergent. Same result as geqrf_slot to working precision. */
template <typename T, int V>
inline void geqrf_slot_sg(sycl::sub_group sg, int g, T *ap, int ldap, T *taup, int m,
                          int n, bool rowmajor = false)
{
    const std::size_t gbase =
        static_cast<std::size_t>(g) * ldap * (rowmajor ? m : n) * V;
    sg_view<T, V> A{sg, ap + gbase,
                    (rowmajor ? static_cast<std::size_t>(ldap) : 1) * V,
                    (rowmajor ? 1 : static_cast<std::size_t>(ldap)) * V};
    const int k = (m < n) ? m : n;
    T *Tb = taup + static_cast<std::size_t>(g) * k * V;

    for (int kk = 0; kk < k; ++kk) {
        const T x0 = A.load(kk, kk);
        T tail = T(0);
        for (int i = kk + 1; i < m; ++i) {
            const T a = A.load(i, kk);
            tail += a * a;
        }
        const T norm = sycl::sqrt(x0 * x0 + tail);
        const T beta = (x0 >= T(0)) ? -norm : norm;
        const bool has = tail > T(0);
        const T t = has ? (beta - x0) / beta : T(0);
        const T inv = has ? T(1) / (x0 - beta) : T(0);
        const T rdiag = has ? beta : x0;
        se::group_store(sg, t, Tb + static_cast<std::size_t>(kk) * V);
        for (int i = kk + 1; i < m; ++i)
            A.store(i, kk, A.load(i, kk) * inv);
        A.store(kk, kk, rdiag);

        int j = kk + 1;
        for (; j + 4 <= n; j += 4) {
            T w0 = A.load(kk, j), w1 = A.load(kk, j + 1), w2 = A.load(kk, j + 2),
              w3 = A.load(kk, j + 3);
            for (int i = kk + 1; i < m; ++i) {
                const T av = A.load(i, kk);
                w0 += av * A.load(i, j);
                w1 += av * A.load(i, j + 1);
                w2 += av * A.load(i, j + 2);
                w3 += av * A.load(i, j + 3);
            }
            A.store(kk, j, A.load(kk, j) - t * w0);
            A.store(kk, j + 1, A.load(kk, j + 1) - t * w1);
            A.store(kk, j + 2, A.load(kk, j + 2) - t * w2);
            A.store(kk, j + 3, A.load(kk, j + 3) - t * w3);
            const T tw0 = t * w0, tw1 = t * w1, tw2 = t * w2, tw3 = t * w3;
            for (int i = kk + 1; i < m; ++i) {
                const T av = A.load(i, kk);
                A.store(i, j, A.load(i, j) - av * tw0);
                A.store(i, j + 1, A.load(i, j + 1) - av * tw1);
                A.store(i, j + 2, A.load(i, j + 2) - av * tw2);
                A.store(i, j + 3, A.load(i, j + 3) - av * tw3);
            }
        }
        for (; j < n; ++j) {
            T w = A.load(kk, j);
            for (int i = kk + 1; i < m; ++i)
                w += A.load(i, kk) * A.load(i, j);
            A.store(kk, j, A.load(kk, j) - t * w);
            const T tw = t * w;
            for (int i = kk + 1; i < m; ++i)
                A.store(i, j, A.load(i, j) - A.load(i, kk) * tw);
        }
    }
}

/* ormqr, explicit sub-group (side='L', column-major, JB=4). */
template <typename T, int V>
inline void ormqr_slot_sg(sycl::sub_group sg, int g, const T *ap, int ldap,
                          const T *taup, T *bp, int ldbp, int m, int nrhs, int k,
                          bool trans)
{
    sg_view<const T, V> A{sg, ap + static_cast<std::size_t>(g) * ldap * k * V,
                          static_cast<std::size_t>(V), static_cast<std::size_t>(ldap) * V};
    sg_view<T, V> B{sg, bp + static_cast<std::size_t>(g) * ldbp * nrhs * V,
                    static_cast<std::size_t>(V), static_cast<std::size_t>(ldbp) * V};
    const T *Tb = taup + static_cast<std::size_t>(g) * k * V;
    const bool fwd = trans;

    for (int s = 0; s < k; ++s) {
        const int kk = fwd ? s : k - 1 - s;
        T t;
        se::group_load(sg, Tb + static_cast<std::size_t>(kk) * V, t);
        int j = 0;
        for (; j + 4 <= nrhs; j += 4) {
            T w0 = B.load(kk, j), w1 = B.load(kk, j + 1), w2 = B.load(kk, j + 2),
              w3 = B.load(kk, j + 3);
            for (int i = kk + 1; i < m; ++i) {
                const T av = A.load(i, kk);
                w0 += av * B.load(i, j);
                w1 += av * B.load(i, j + 1);
                w2 += av * B.load(i, j + 2);
                w3 += av * B.load(i, j + 3);
            }
            B.store(kk, j, B.load(kk, j) - t * w0);
            B.store(kk, j + 1, B.load(kk, j + 1) - t * w1);
            B.store(kk, j + 2, B.load(kk, j + 2) - t * w2);
            B.store(kk, j + 3, B.load(kk, j + 3) - t * w3);
            const T tw0 = t * w0, tw1 = t * w1, tw2 = t * w2, tw3 = t * w3;
            for (int i = kk + 1; i < m; ++i) {
                const T av = A.load(i, kk);
                B.store(i, j, B.load(i, j) - av * tw0);
                B.store(i, j + 1, B.load(i, j + 1) - av * tw1);
                B.store(i, j + 2, B.load(i, j + 2) - av * tw2);
                B.store(i, j + 3, B.load(i, j + 3) - av * tw3);
            }
        }
        for (; j < nrhs; ++j) {
            T w = B.load(kk, j);
            for (int i = kk + 1; i < m; ++i)
                w += A.load(i, kk) * B.load(i, j);
            B.store(kk, j, B.load(kk, j) - t * w);
            const T tw = t * w;
            for (int i = kk + 1; i < m; ++i)
                B.store(i, j, B.load(i, j) - A.load(i, kk) * tw);
        }
    }
}

/* trsm (upper), explicit sub-group. */
template <typename T, int V>
inline void trsm_upper_slot_sg(sycl::sub_group sg, int g, const T *ap, int ldap, T *bp,
                               int ldbp, int n, int nrhs)
{
    sg_view<const T, V> R{sg, ap + static_cast<std::size_t>(g) * ldap * n * V,
                          static_cast<std::size_t>(V), static_cast<std::size_t>(ldap) * V};
    sg_view<T, V> B{sg, bp + static_cast<std::size_t>(g) * ldbp * nrhs * V,
                    static_cast<std::size_t>(V), static_cast<std::size_t>(ldbp) * V};
    /* Row loop outermost: the diagonal is column-invariant, so this turns
     * n*nrhs sub-group block reads of R(i,i) into n. Same division as the
     * per-lane variant above, so the two stay bit-identical. */
    for (int i = n - 1; i >= 0; --i) {
        const T d = R.load(i, i);
        for (int j = 0; j < nrhs; ++j) {
            T s = B.load(i, j);
            for (int l = i + 1; l < n; ++l)
                s -= R.load(i, l) * B.load(l, j);
            B.store(i, j, s / d);
        }
    }
}

} /* namespace gpu */
} /* namespace cqr */

#endif /* CQR_COMPACT_SYCL_HPP */
