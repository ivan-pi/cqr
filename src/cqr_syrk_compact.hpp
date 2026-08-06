/* cqr_syrk_compact.hpp
 *
 * Compact (interleaved-batch) symmetric rank-k update, templated on scalar type
 * T and interleave width V:
 *
 *     C := alpha * A * A^T + beta * C   (trans='N', A is n x k)
 *     C := alpha * A^T * A + beta * C   (trans='T', A is k x n)
 *
 * C is the symmetric n x n result; only the uplo triangle (lower or upper) is
 * read and written. This is the missing mkl_?syrk_compact, in portable form --
 * the companion to mkl_?gemm_compact / mkl_?trsm_compact, exploiting the
 * symmetry MKL's gemm_compact cannot (half the flops, a triangle of writes). Its
 * headline use is the Gram matrix of a Cholesky QR: A^T A -> potrf -> trsm.
 *
 * syrk_compact_group()         -- tuned trans='T' (A^T A), column-major path.
 * syrk_compact_group_strided() -- any uplo / trans / layout, via BatchView.
 * syrk_compact_general()       -- all groups; the sole batch entry point.
 *
 * Algorithm: the dot-product form. Each output C(i,j) is a length-k inner
 * product of two "vectors" of A along the contraction axis, accumulated in
 * registers and combined with beta*C on store. Compact format stores element
 * (i,j) of all V matrices contiguously, so the scalar code lifts verbatim with
 * T -> V-wide vector, one lane per matrix (no data-dependent branch). Preferring
 * the dot form over the rank-1 (outer-product) form writes C exactly once and
 * blocks cleanly against the triangle: for a fixed row i, every column j in that
 * row's triangle is a *full* length-k dot (no diagonal-corner peeling). Output
 * columns are register-blocked (JB=4) so each A vector load is reused across four
 * columns -- the same reuse the ormqr/trsm kernels apply across their columns.
 * beta is folded in on store; when beta==0 the element is overwritten rather than
 * read, so an uninitialised / NaN C is allowed, matching reference BLAS ?syrk.
 *
 * Compact storage convention (matches MKL Compact / mkl_?gepack_compact); group
 * g = idx/V, slot v = idx%V, C the symmetric n x n batch. For the tuned trans='T'
 * column-major path A is the (ldap, n) batch, so
 *     A_v(p,i) = ap[ g*ldap*n*V + (i*ldap + p)*V + v ]   (column i is contiguous)
 *     C_v(i,j) = cp[ g*ldcp*n*V + (j*ldcp + i)*V + v ]
 * The strided path expresses every other uplo/trans/layout solely by which
 * physical stride is the n-index axis, the contraction axis, and the C
 * row/column -- exactly as side='L'/'R' is a stride choice in the ormqr kernel.
 *
 * Scope: real types (s/d). Complex would be ?herk (conjugated), a separate
 * routine, not a trans='C' mode here -- mirroring the real-only ormqr/trsm scope.
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#ifndef CQR_SYRK_COMPACT_HPP
#define CQR_SYRK_COMPACT_HPP

#include "cqr_compact_common.hpp" /* pack<T,V>, BatchView, make_view, make_const_view */

#include <cstddef>
#include <cassert>
#include <type_traits>

namespace cqr {
namespace detail {

/* ------------------------------------------------------------------ */
/* One group of V interleaved matrices: tuned trans='T', column-major. */
/*                                                                     */
/* C := alpha A^T A + beta C, A the (ldap, n) batch, C the (ldcp, n)    */
/* batch. Column i of A starts at a_ + i*ldap and its k entries are      */
/* contiguous, so the length-k dot that forms C(i,j) streams both        */
/* operands down contiguous packs -- the Cholesky-QR Gram matrix in the  */
/* layout LAPACK produces its factors in. JB=4 reuses each A(:,i) load    */
/* across four output columns j.                                         */
/* ------------------------------------------------------------------ */

template <typename T, int V, typename Int = int>
void syrk_compact_group(bool upper, Int n, Int k, T alpha, T beta, const T *a_, Int ldap,
                        T *c_, Int ldcp)
{
    using VT = typename pack<T, V>::type;
    static_assert(std::is_floating_point<T>::value,
                  "syrk_compact is defined for real float/double");
    assert(ldap >= k && ldcp >= n && k >= 0);

    const VT *A = reinterpret_cast<const VT *>(a_);
    VT *C = reinterpret_cast<VT *>(c_);
    /* beta==0 overwrites C (must not read it: it may be uninitialised/NaN). */
    const bool overwrite = (beta == T(0));

    for (Int i = 0; i < n; ++i) {
        const VT *Ai = A + static_cast<std::size_t>(i) * ldap; /* column i of A */
        const Int jlo = upper ? i : 0;                         /* row i's triangle */
        const Int jhi = upper ? n : i + 1;

        Int j = jlo;

        /* main loop: 4 output columns at a time; A(:,i) loaded once, used 4x */
        for (; j + 4 <= jhi; j += 4) {
            const VT *Aj0 = A + static_cast<std::size_t>(j + 0) * ldap;
            const VT *Aj1 = A + static_cast<std::size_t>(j + 1) * ldap;
            const VT *Aj2 = A + static_cast<std::size_t>(j + 2) * ldap;
            const VT *Aj3 = A + static_cast<std::size_t>(j + 3) * ldap;
            VT w0{}, w1{}, w2{}, w3{};
            for (Int p = 0; p < k; ++p) {
                const VT aip = Ai[p];
                w0 += aip * Aj0[p];
                w1 += aip * Aj1[p];
                w2 += aip * Aj2[p];
                w3 += aip * Aj3[p];
            }
            VT &c0 = C[i + static_cast<std::size_t>(j + 0) * ldcp];
            VT &c1 = C[i + static_cast<std::size_t>(j + 1) * ldcp];
            VT &c2 = C[i + static_cast<std::size_t>(j + 2) * ldcp];
            VT &c3 = C[i + static_cast<std::size_t>(j + 3) * ldcp];
            if (overwrite) {
                c0 = alpha * w0;
                c1 = alpha * w1;
                c2 = alpha * w2;
                c3 = alpha * w3;
            }
            else {
                c0 = alpha * w0 + beta * c0;
                c1 = alpha * w1 + beta * c1;
                c2 = alpha * w2 + beta * c2;
                c3 = alpha * w3 + beta * c3;
            }
        }

        /* remainder columns */
        for (; j < jhi; ++j) {
            const VT *Aj = A + static_cast<std::size_t>(j) * ldap;
            VT w{};
            for (Int p = 0; p < k; ++p)
                w += Ai[p] * Aj[p];
            VT &cij = C[i + static_cast<std::size_t>(j) * ldcp];
            cij = overwrite ? (alpha * w) : (alpha * w + beta * cij);
        }
    }
}

/* ------------------------------------------------------------------ */
/* One group, fully general: any uplo / trans / layout via BatchView.  */
/*                                                                     */
/* A is viewed as A(idx, p): idx is the C-index (n) axis, p the         */
/* contraction (k) axis -- so trans='N' vs 'T' is just which physical   */
/* stride plays each role, exactly as side='L'/'R' is in the ormqr      */
/* kernel. C is viewed as C(i, j) with its own row/column strides. The  */
/* dot-product math is identical to the tuned kernel above; only the    */
/* addressing, now carried by the two BatchViews, changes.             */
/* ------------------------------------------------------------------ */

template <typename T, int V, typename Int = int>
void syrk_compact_group_strided(bool upper, Int n, Int k, T alpha, T beta,
                                BatchView<const typename pack<T, V>::type, Int> A,
                                BatchView<typename pack<T, V>::type, Int> C)
{
    using VT = typename pack<T, V>::type;
    static_assert(std::is_floating_point<T>::value,
                  "syrk_compact is defined for real float/double");
    /* strides must be non-degenerate so distinct indices map to distinct slots */
    assert(A.special && A.panel && C.special && C.panel);

    const bool overwrite = (beta == T(0));

    for (Int i = 0; i < n; ++i) {
        const Int jlo = upper ? i : 0;
        const Int jhi = upper ? n : i + 1;

        Int j = jlo;

        /* main loop: 4 output columns at a time; A(i,:) loaded once, used 4x */
        for (; j + 4 <= jhi; j += 4) {
            VT w0{}, w1{}, w2{}, w3{};
            for (Int p = 0; p < k; ++p) {
                const VT aip = A(i, p);
                w0 += aip * A(j + 0, p);
                w1 += aip * A(j + 1, p);
                w2 += aip * A(j + 2, p);
                w3 += aip * A(j + 3, p);
            }
            VT &c0 = C(i, j + 0);
            VT &c1 = C(i, j + 1);
            VT &c2 = C(i, j + 2);
            VT &c3 = C(i, j + 3);
            if (overwrite) {
                c0 = alpha * w0;
                c1 = alpha * w1;
                c2 = alpha * w2;
                c3 = alpha * w3;
            }
            else {
                c0 = alpha * w0 + beta * c0;
                c1 = alpha * w1 + beta * c1;
                c2 = alpha * w2 + beta * c2;
                c3 = alpha * w3 + beta * c3;
            }
        }

        /* remainder columns */
        for (; j < jhi; ++j) {
            VT w{};
            for (Int p = 0; p < k; ++p)
                w += A(i, p) * A(j, p);
            VT &cij = C(i, j);
            cij = overwrite ? (alpha * w) : (alpha * w + beta * cij);
        }
    }
}

/* ------------------------------------------------------------------ */
/* All groups, fully general: uplo, trans in {N,T}, col- or row-major.  */
/* This is the sole batch entry point -- both C adapters call it for     */
/* every case, and it dispatches the tuned path itself (below).          */
/*                                                                     */
/* trans='N': A is n x k; trans='T': A is k x n. The trans='T',          */
/* column-major case routes to the tuned contiguous kernel; the other    */
/* three combinations use the strided kernel (same dot-product math,     */
/* correctness-first addressing). A padded partial last group is         */
/* processed too -- its padding slots carry whatever mkl_?gepack_compact  */
/* wrote, and their (garbage) C triangle is simply never read back.       */
/* ------------------------------------------------------------------ */

template <typename T, int V, typename Int = int>
void syrk_compact_general(bool upper, bool trans, bool rowmajor, Int n, Int k, T alpha,
                          const T *ap, Int ldap, T beta, T *cp, Int ldcp, Int nm)
{
    assert(n >= 0 && k >= 0 && nm >= 1);

    /* A element strides (in VT units): one along the C-index (n) axis, one along
     * the contraction (k) axis. trans='T' runs the n index down A's columns
     * (A is k x n), trans='N' down its rows (A is n x k); column-major makes the
     * first physical axis unit-stride, row-major the second. */
    const Int a_nidx = trans ? (rowmajor ? 1 : ldap) : (rowmajor ? ldap : 1);
    const Int a_kidx = trans ? (rowmajor ? ldap : 1) : (rowmajor ? 1 : ldap);

    /* C is symmetric n x n: row index i, column index j. The uplo triangle is
     * defined on the math indices (i,j) regardless of layout; only the strides
     * differ. */
    const Int c_row = rowmajor ? ldcp : 1;
    const Int c_col = rowmajor ? 1 : ldcp;

    /* group strides (scalar T units). A's packed per-matrix extent is ldap times
     * the count of its non-leading axis; C's is ldcp*n. These span the whole
     * batch, so widen to size_t before the product to avoid overflow. */
    const Int a_lines = rowmajor ? (trans ? k : n) : (trans ? n : k);
    const std::size_t str_a = static_cast<std::size_t>(ldap) * a_lines * V;
    const std::size_t str_c = static_cast<std::size_t>(ldcp) * n * V;

    const bool tuned = (trans && !rowmajor); /* A^T A, column-major: contiguous */

    /* Hoist the tuned-vs-strided choice out of the group loop: it is invariant
     * across groups, so each branch gets its own loop rather than a per-group
     * test. Costs one duplicated loop header; leaves the compiler no chance to
     * keep the dispatch in the hot path. */
    const Int ngroups = (nm + V - 1) / V;
    if (tuned) {
        for (Int g = 0; g < ngroups; ++g)
            syrk_compact_group<T, V, Int>(upper, n, k, alpha, beta,
                                          ap + (std::size_t)g * str_a, ldap,
                                          cp + (std::size_t)g * str_c, ldcp);
    }
    else {
        for (Int g = 0; g < ngroups; ++g)
            syrk_compact_group_strided<T, V, Int>(
                upper, n, k, alpha, beta,
                make_const_view<T, V, Int>(ap + (std::size_t)g * str_a, a_nidx, a_kidx),
                make_view<T, V, Int>(cp + (std::size_t)g * str_c, c_row, c_col));
    }
}

} /* namespace detail */
} /* namespace cqr */

#endif /* CQR_SYRK_COMPACT_HPP */
