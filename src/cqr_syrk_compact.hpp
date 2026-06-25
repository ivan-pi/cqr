/* cqr_syrk_compact.hpp
 *
 * Compact (interleaved-batch) symmetric rank-k update, templated on scalar
 * type T and interleave width V:
 *
 *     C := alpha * A * A^T + beta * C   (trans='N', A is n x k)
 *     C := alpha * A^T * A + beta * C   (trans='T', A is k x n)
 *
 * C is the symmetric n x n result; only the uplo triangle (lower or upper) is
 * read and written. This is the missing mkl_?syrk_compact, in portable form --
 * the companion to mkl_?gemm_compact / mkl_?trsm_compact, exploiting the
 * symmetry MKL's gemm_compact cannot (half the flops, a triangle of writes).
 *
 * syrk_compact_group()         -- tuned trans='N', column-major path.
 * syrk_compact_group_strided() -- any uplo / trans / layout, via BatchView.
 *
 * Design:
 *   - V is the compact-format interleave width: element (i,j) of V consecutive
 *     matrices is stored contiguously, so the scalar code lifts verbatim with
 *     T -> V-wide vector. See cqr_compact.hpp for the pack<T,V> definition and
 *     the BatchView strided-view helpers, reused here unchanged.
 *   - The kernel is the dot-product form: each output C(i,j) is a length-k
 *     inner product of two "rows" of A (the contraction axis), accumulated in
 *     registers and combined with beta*C on store. Choosing dot-product over
 *     the rank-1 (outer-product) form means C is written exactly once and the
 *     Householder-free arithmetic stays branch-free; it also blocks cleanly
 *     against the triangle, because for a fixed row i every column j in that
 *     row's triangle is a *full* length-k dot (no diagonal-corner peeling).
 *   - Output columns are register-blocked (JB=4) so each A(i,p) load is reused
 *     across 4 columns, halving the dominant A traffic -- the same reuse trick
 *     the ormqr kernel applies across its RHS columns.
 *   - beta is folded in on store. When beta==0 the element is overwritten
 *     rather than read, so an uninitialised / NaN C is allowed, matching the
 *     reference BLAS ?syrk contract.
 *
 * Compact storage convention (matches MKL Compact / mkl_?gepack_compact):
 *   group g = matrix index / V, slot v = matrix index % V. For the tuned
 *   trans='N', column-major path A is the (ldap, k) batch and C the (ldcp, n)
 *   batch, so
 *     A_v(i,p) = ap[ g*ldap*k*V   + (p*ldap + i)*V + v ]
 *     C_v(i,j) = cp[ g*ldcp*n*V   + (j*ldcp + i)*V + v ]
 *   The strided path expresses every other uplo/trans/layout solely by which
 *   stride is the n-index axis, the contraction axis, and the C row/column.
 *
 * Scope: real types (s/d). Complex would be ?herk (conjugated), a separate
 * routine, not a trans='C' mode here -- mirroring the real-only ormqr scope.
 */

#ifndef CQR_SYRK_COMPACT_HPP
#define CQR_SYRK_COMPACT_HPP

#include <cstddef>
#include <cassert>
#include <type_traits>

/* pack<T,V>, BatchView, make_view / make_const_view, and the GNU-vector guard
 * all live in the ormqr kernel header; reuse them verbatim. */
#include "cqr_compact.hpp"

namespace cqr {
namespace detail {

/* ------------------------------------------------------------------ */
/* One group of V interleaved matrices: tuned trans='N', column-major. */
/*                                                                     */
/* A(i,p) = A[i + p*ldap] and C(i,j) = C[i + j*ldcp] as V-wide packs.   */
/* The inner contraction over p strides by ldap (A's column step); for  */
/* the small orders this format targets the columns stay resident, and  */
/* writing each C(i,j) once keeps C traffic minimal. JB=4 reuses the     */
/* A(i,p) load across four output columns.                              */
/* ------------------------------------------------------------------ */

template <typename T, int V, typename Int = int>
void syrk_compact_group(bool lower, Int n, Int k, T alpha, T beta,
                        const T *a_, Int ldap, T *c_, Int ldcp)
{
    using VT = typename pack<T, V>::type;
    static_assert(std::is_floating_point<T>::value,
                  "syrk_compact is defined for real float/double");
    assert(ldap >= n && ldcp >= n && k >= 0);

    const VT *A = reinterpret_cast<const VT *>(a_);
    VT       *C = reinterpret_cast<VT *>(c_);
    /* beta==0 overwrites C (must not read it: it may be uninitialised/NaN). */
    const bool overwrite = (beta == T(0));

    for (Int i = 0; i < n; ++i) {
        const VT *Ai  = A + i;                 /* A(i,p) = Ai[p*ldap] */
        const Int jlo = lower ? 0     : i;     /* row i's triangle columns */
        const Int jhi = lower ? i + 1 : n;

        Int j = jlo;

        /* main loop: 4 output columns at a time; A(i,p) loaded once, used 4x */
        for (; j + 4 <= jhi; j += 4) {
            const VT *Aj0 = A + (j + 0), *Aj1 = A + (j + 1),
                     *Aj2 = A + (j + 2), *Aj3 = A + (j + 3);
            VT w0{}, w1{}, w2{}, w3{};
            for (Int p = 0; p < k; ++p) {
                const std::size_t pp  = static_cast<std::size_t>(p) * ldap;
                const VT          aip = Ai[pp];
                w0 += aip * Aj0[pp]; w1 += aip * Aj1[pp];
                w2 += aip * Aj2[pp]; w3 += aip * Aj3[pp];
            }
            VT &c0 = C[i + static_cast<std::size_t>(j + 0) * ldcp];
            VT &c1 = C[i + static_cast<std::size_t>(j + 1) * ldcp];
            VT &c2 = C[i + static_cast<std::size_t>(j + 2) * ldcp];
            VT &c3 = C[i + static_cast<std::size_t>(j + 3) * ldcp];
            if (overwrite) {
                c0 = alpha * w0; c1 = alpha * w1;
                c2 = alpha * w2; c3 = alpha * w3;
            } else {
                c0 = alpha * w0 + beta * c0; c1 = alpha * w1 + beta * c1;
                c2 = alpha * w2 + beta * c2; c3 = alpha * w3 + beta * c3;
            }
        }

        /* remainder columns */
        for (; j < jhi; ++j) {
            const VT *Aj = A + j;
            VT w{};
            for (Int p = 0; p < k; ++p) {
                const std::size_t pp = static_cast<std::size_t>(p) * ldap;
                w += Ai[pp] * Aj[pp];
            }
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
void syrk_compact_group_strided(bool lower, Int n, Int k, T alpha, T beta,
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
        const Int jlo = lower ? 0     : i;
        const Int jhi = lower ? i + 1 : n;

        Int j = jlo;

        /* main loop: 4 output columns at a time; A(i,p) loaded once, used 4x */
        for (; j + 4 <= jhi; j += 4) {
            VT w0{}, w1{}, w2{}, w3{};
            for (Int p = 0; p < k; ++p) {
                const VT aip = A(i, p);
                w0 += aip * A(j + 0, p); w1 += aip * A(j + 1, p);
                w2 += aip * A(j + 2, p); w3 += aip * A(j + 3, p);
            }
            VT &c0 = C(i, j + 0); VT &c1 = C(i, j + 1);
            VT &c2 = C(i, j + 2); VT &c3 = C(i, j + 3);
            if (overwrite) {
                c0 = alpha * w0; c1 = alpha * w1;
                c2 = alpha * w2; c3 = alpha * w3;
            } else {
                c0 = alpha * w0 + beta * c0; c1 = alpha * w1 + beta * c1;
                c2 = alpha * w2 + beta * c2; c3 = alpha * w3 + beta * c3;
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
/* All groups: nm matrices total. A padded partial last group is        */
/* processed too -- its padding slots carry whatever mkl_?gepack_compact */
/* wrote, and their (garbage) C triangle is simply never read back.      */
/* ------------------------------------------------------------------ */

template <typename T, int V, typename Int = int>
void syrk_compact(bool lower, Int n, Int k, T alpha,
                  const T *ap, Int ldap, T beta,
                  T *cp, Int ldcp, Int nm)
{
    assert(ldap >= n && ldcp >= n && k >= 0 && nm >= 1);

    const Int ngroups = (nm + V - 1) / V;
    /* trans='N', column-major: A is (ldap, k), C is (ldcp, n). */
    const std::size_t str_a = static_cast<std::size_t>(ldap) * k * V;
    const std::size_t str_c = static_cast<std::size_t>(ldcp) * n * V;

    for (Int g = 0; g < ngroups; ++g)
        syrk_compact_group<T, V, Int>(lower, n, k, alpha, beta,
                                      ap + g * str_a, ldap,
                                      cp + g * str_c, ldcp);
}

/* ------------------------------------------------------------------ */
/* All groups, fully general: uplo, trans in {N,T}, col- or row-major.  */
/*                                                                     */
/* trans='N': A is n x k; trans='T': A is k x n. The trans='N',         */
/* column-major case routes to the tuned contiguous kernel; the other   */
/* three combinations use the strided kernel (same dot-product math,    */
/* correctness-first addressing).                                       */
/* ------------------------------------------------------------------ */

template <typename T, int V, typename Int = int>
void syrk_compact_general(bool lower, bool trans, bool rowmajor,
                          Int n, Int k, T alpha,
                          const T *ap, Int ldap, T beta,
                          T *cp, Int ldcp, Int nm)
{
    assert(n >= 0 && k >= 0 && nm >= 1);

    /* A element strides (in VT units): one along the C-index (n) axis, one
     * along the contraction (k) axis. Column-major makes A's first declared
     * axis unit-stride, row-major its second; trans swaps which is which. */
    const std::size_t a_nidx = trans ? (rowmajor ? 1 : (std::size_t)ldap)
                                     : (rowmajor ? (std::size_t)ldap : 1);
    const std::size_t a_kidx = trans ? (rowmajor ? (std::size_t)ldap : 1)
                                     : (rowmajor ? 1 : (std::size_t)ldap);

    /* C is symmetric n x n: row index i, column index j. The uplo triangle is
     * defined on the math indices (i,j) regardless of layout; only the strides
     * differ. */
    const std::size_t c_row = rowmajor ? (std::size_t)ldcp : 1;
    const std::size_t c_col = rowmajor ? 1 : (std::size_t)ldcp;

    /* group strides (scalar T units). A's packed per-matrix extent is ldap
     * times the count of its non-leading axis; C's is ldcp*n. */
    const std::size_t a_lines = rowmajor ? (trans ? (std::size_t)k : (std::size_t)n)
                                         : (trans ? (std::size_t)n : (std::size_t)k);
    const std::size_t str_a = (std::size_t)ldap * a_lines * V;
    const std::size_t str_c = (std::size_t)ldcp * n * V;

    const bool tuned = (!trans && !rowmajor);

    const Int ngroups = (nm + V - 1) / V;
    for (Int g = 0; g < ngroups; ++g) {
        const T *a = ap + g * str_a;
        T       *c = cp + g * str_c;
        if (tuned)
            syrk_compact_group<T, V, Int>(lower, n, k, alpha, beta,
                                          a, ldap, c, ldcp);
        else
            syrk_compact_group_strided<T, V, Int>(
                lower, n, k, alpha, beta,
                make_const_view<T, V, Int>(a, a_nidx, a_kidx),
                make_view<T, V, Int>(c, c_row, c_col));
    }
}

} /* namespace detail */
} /* namespace cqr */

#endif /* CQR_SYRK_COMPACT_HPP */
