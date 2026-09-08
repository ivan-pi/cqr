/* cqr_gels_compact.hpp
 *
 * Compact (interleaved-batch) least-squares / minimum-norm solve, templated on
 * scalar type T and interleave width V -- the compact form of LAPACK ?gels:
 *
 *     op(A) X = B,   op(A) = A ('N') or A^T ('T'),   A is m x n,
 *
 * assuming op(A) has full rank. When op(A) has more rows than columns the
 * system is overdetermined and X is the least-squares solution
 * (min ||B - op(A) X||_F); when it has more columns than rows it is
 * underdetermined and X is the minimum-norm solution (min ||X||_F subject to
 * op(A) X = B). The solve runs per group of V matrices in one call -- factor,
 * apply Q, back-substitute -- on the group's cache-resident buffers, which is
 * what a geqrf -> ormqr -> trsm chain of whole-batch calls cannot do.
 *
 * Four cases collapse to one kernel. Let F be the *tall* orientation of A:
 *     F = A    (m >= n, LAPACK's QR path)   or   F = A^T  (m < n, LAPACK's LQ path),
 * p x q with p = max(m, n), q = min(m, n). In the compact buffer F is A's
 * BatchView, transposed when m < n (free: the strides swap), and geqrf's kernel
 * over that view IS the LQ factorization in ?gelqf's storage convention (L on
 * and below A's diagonal, reflector rows to its right). With F = Q [R; 0]:
 *
 *   overdetermined  F X = B          B := Q^T B, then R X = B(0:q); rows q..p-1
 *   (m >= n & 'N',                   of B hold the residual (their squared
 *    m <  n & 'T')                   column norms are the residual sums of
 *                                    squares, as for ?gels)
 *   underdetermined F^T X = B        R^T Y = B(0:q), B(q:p) := 0, then B := Q B
 *   (m >= n & 'T',                   (X = Q [Y; 0] is the minimum-norm solution)
 *    m <  n & 'N')
 *
 * The overdetermined reduction is fused: geqrf_compact_group applies each
 * reflector to B as it is built (the QR of [F | B] truncated to q reflectors),
 * so Q^T B costs no second sweep over the reflectors. The underdetermined case
 * needs the factorization complete before Q is applied, so it runs the three
 * steps in sequence.
 *
 * Compact storage (matches mkl_?gepack_compact); group g = idx/V, slot v = idx%V:
 *     A_v(i,j)  = ap  [ g*ldap*n*V    + (j*ldap + i)*V + v ]   (column-major)
 *     B_v(i,j)  = bp  [ g*ldbp*nrhs*V + (j*ldbp + i)*V + v ]
 *     tau_v(kk) = work[ g*q*V         +  kk*V          + v ]
 * B is max(m, n) x nrhs per matrix, as ?gels declares it (ldb >= max(m, n)):
 * on entry its first (rows of op(A)) rows hold the right-hand sides, on exit
 * its first (columns of op(A)) rows the solution. On exit ap holds the QR (m >= n)
 * or LQ (m < n) factorization of A, and work the reflector scalars tau, in
 * compact format: together they are the (H, tau) that ?ormqr_compact accepts.
 *
 * Assisted-by: Claude:claude-fable-5
 */

#ifndef CQR_GELS_COMPACT_HPP
#define CQR_GELS_COMPACT_HPP

#include "cqr_compact_common.hpp"
#include "cqr_geqrf_compact.hpp"
#include "cqr_ormqr_compact.hpp"
#include "cqr_trsm_compact.hpp"

#include <cstddef>
#include <cassert>
#include <type_traits>

namespace cqr::detail {

/* Workspace: q = min(m, n) reflector scalars per matrix, one slot per group so
 * the groups can run in parallel -- the size of a compact tau buffer for the
 * batch (mkl_?get_size_compact(min(m, n), 1, format, nm) in scalars). At least
 * 1, like LAPACK's lwork. */
template <typename Int> Int gels_lwork(Int m, Int n, Int nm, int V) noexcept
{
    const Int q = (m < n) ? m : n;
    const Int ngroups = (nm + V - 1) / V;
    const Int need = q * V * ngroups;
    return need < 1 ? Int(1) : need;
}

/* The triangular step on the q x q upper triangle R of F: R X = B(0:q) (tran =
 * false) or R^T Y = B(0:q) (true), the first q rows of B in place. The tuned
 * column-major trsm path applies when both views have unit row stride (F is A
 * itself, column-major, and so is B); otherwise the strided kernel. */
template <typename T, int V, typename Int>
inline void gels_trsm(bool tran, Int q, Int nrhs, const ConstBatchView<T, V, Int> &F,
                      const BatchView<T, V, Int> &B)
{
    if (F.si == 1 && B.si == 1)
        trsm_left_dot<T, V, Int>(true, tran, false, q, nrhs, T(1),
                                 reinterpret_cast<const T *>(F.data), F.sj,
                                 reinterpret_cast<T *>(B.data), B.sj);
    else
        trsm_compact_group_strided<T, V, Int>(true, true, tran, false, q, nrhs, T(1), F,
                                              B);
}

/* One group: F (p x q, p >= q) is the tall view of A, B its p x nrhs
 * right-hand-side view, tau_ the group's q-scalar scratch. overdet selects
 * F X = B (least squares) over F^T X = B (minimum norm). */
template <typename T, int V, typename Int = int>
void gels_compact_group(bool overdet, Int p, Int q, Int nrhs, BatchView<T, V, Int> F,
                        T *tau_, BatchView<T, V, Int> B)
{
    using VT = typename pack<T, V>::type;
    static_assert(std::is_floating_point_v<T>,
                  "gels_compact is defined for real float/double");
    assert(p >= q && F.si && F.sj && B.si && B.sj);

    const auto Fc = F.as_const();
    if (overdet) {
        /* F = Q [R; 0], with B := Q^T B fused into the factorization; then
         * R X = (Q^T B)(0:q). */
        geqrf_compact_group<T, V, Int>(p, q, F, tau_, B, nrhs);
        gels_trsm<T, V, Int>(false, q, nrhs, Fc, B);
    }
    else {
        /* F^T = [R^T 0] Q^T: solve R^T Y = B(0:q), then X = Q [Y; 0]. */
        geqrf_compact_group<T, V, Int>(p, q, F, tau_);
        gels_trsm<T, V, Int>(true, q, nrhs, Fc, B);
        for (Int j = 0; j < nrhs; ++j)
            for (Int i = q; i < p; ++i)
                B(i, j) = VT{};
        ormqr_compact_group<T, V, Int>(Direction::Backward, p, nrhs, q, Fc, tau_, B);
    }
}

/* All groups, either layout. trans is 'N', or 'T'/'C' for the transpose. work
 * holds gels_lwork(m, n, nm, V) scalars. A padded partial last group is
 * processed too, harmlessly: an identity lane (of A, and of B) factors to
 * R = I, tau = 0, and solves to X = B. A rank-deficient op(A) is not detected:
 * a zero diagonal of R divides through to Inf/NaN in that lane, as ?trsm. */
template <typename T, int V, typename Int = int>
void gels_compact(bool rowmajor, char trans, Int m, Int n, Int nrhs, T *ap, Int ldap,
                  T *bp, Int ldbp, T *work, Int nm)
{
    assert(nm >= 1 && m >= 0 && n >= 0 && nrhs >= 0);

    const bool tran = (trans == 'T' || trans == 't' || trans == 'C' || trans == 'c');
    const bool tall = (m >= n);
    const Int p = tall ? m : n, q = tall ? n : m;
    const bool overdet = (tall != tran); /* op(A) has p rows and q columns */

    const std::size_t str_a = group_stride(rowmajor, ldap, m, n, V);
    const std::size_t str_b = group_stride(rowmajor, ldbp, p, nrhs, V);
    const std::size_t str_t = (std::size_t)q * V;

    if (q == 0) {
        /* an empty op(A): the (minimum-norm) solution is X = 0, as ?gels sets
         * B(0:max(m,n), :) := 0 when min(m, n) = 0 */
        using VT = typename pack<T, V>::type;
        for_each_group<V>(
            nm,
            [&](Int g) {
                auto B = make_view<T, V, Int>(bp + g * str_b, rowmajor, ldbp);
                for (Int j = 0; j < nrhs; ++j)
                    for (Int i = 0; i < p; ++i)
                        B(i, j) = VT{};
            },
            (double)p * nrhs * V /* stores per group */);
        return;
    }

    for_each_group<V>(
        nm,
        [&](Int g) {
            const auto A = make_view<T, V, Int>(ap + g * str_a, rowmajor, ldap);
            gels_compact_group<T, V, Int>(
                overdet, p, q, nrhs, tall ? A : A.transposed(), work + g * str_t,
                make_view<T, V, Int>(bp + g * str_b, rowmajor, ldbp));
        },
        /* ~geqr2 + apply-Q + substitution flops per group */
        (2.0 * p * q * q + 4.0 * p * q * nrhs + 1.0 * q * q * nrhs) * V);
}

} /* namespace cqr::detail */

#endif /* CQR_GELS_COMPACT_HPP */
