// test_compact_util.hpp
//
// Shared helpers for the compact-format test suites: a seeded RNG, error
// metrics, input generation, scalar LAPACK-style reference kernels, precision
// overloads of the portable C API, and Compact pack/unpack. Header-only and
// MKL-free, so the BLAS-free portable tests use it too.
//
// Assisted-by: Claude:claude-opus-4.8

#ifndef TEST_COMPACT_UTIL_HPP
#define TEST_COMPACT_UTIL_HPP

#include "cqr_compact.h"
#include "cqr_matrix_view.hpp"

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>
#include <algorithm>

namespace cqr::test {

// The dense strided view every suite addresses its host-side matrices through
// (src/cqr_matrix_view.hpp); the kernels' BatchView is the compact analogue.
using cqr::detail::ConstMatrixView;
using cqr::detail::mat_view;
using cqr::detail::MatrixView;

// One RNG per test binary (each test is a separate executable, so there is no
// cross-test coupling); the seed only has to be fixed, not unique.
inline std::mt19937_64 &rng()
{
    static std::mt19937_64 g(2026);
    return g;
}

template <class T> T frand()
{
    static std::uniform_real_distribution<T> dist(T(-1), T(1));
    return dist(rng());
}

// max |a - b| over n elements.
template <class T> double max_abs_diff(const T *a, const T *b, size_t n)
{
    double d = 0;
    for (size_t i = 0; i < n; ++i)
        d = std::max(d, (double)std::abs(a[i] - b[i]));
    return d;
}

// L1 (max column sum) norm of a column-major m x n matrix, leading dim m.
template <class T> double norm1(const T *M, int m, int n)
{
    double mx = 0;
    for (int j = 0; j < n; ++j) {
        double s = 0;
        for (int i = 0; i < m; ++i)
            s += std::abs(M[i + (size_t)j * m]);
        mx = std::max(mx, s);
    }
    return mx;
}

// C (m x n) := A (m x k) * B (k x n), all column-major with the given leading
// dimensions -- the plain triple loop, for forming right-hand sides and residuals.
template <class T>
void matmul(int m, int n, int k, const T *A, int lda, const T *B, int ldb, T *C, int ldc)
{
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            T s = 0;
            for (int l = 0; l < k; ++l)
                s += A[i + (size_t)l * lda] * B[l + (size_t)j * ldb];
            C[i + (size_t)j * ldc] = s;
        }
}

// The known solution the solve checks recover: X(:,j) = j+1 (ones, twos, ...),
// column-major n x nrhs.
template <class T> std::vector<T> known_solution(int n, int nrhs)
{
    std::vector<T> X((size_t)n * nrhs);
    for (int j = 0; j < nrhs; ++j)
        for (int i = 0; i < n; ++i)
            X[i + (size_t)j * n] = T(j + 1);
    return X;
}

// ----------------------- reference kernels (scalar) -----------------
// Column-major, in place, templated on T: the unblocked LAPACK algorithms the
// vectorized kernels execute V lanes at a time, so a correct kernel matches
// them to working precision.

// dlarfg: reflector from (alpha, x[0..m-2]); alpha := beta on exit.
template <class T> void ref_larfg(int m, T *alpha, T *x, T *tau)
{
    T xnorm = 0;
    for (int i = 0; i < m - 1; ++i)
        xnorm = std::hypot(xnorm, x[i]);
    if (xnorm == T(0)) {
        *tau = 0;
        return;
    }
    T beta = -std::copysign(std::hypot(*alpha, xnorm), *alpha);
    *tau = (beta - *alpha) / beta;
    T scal = T(1) / (*alpha - beta);
    for (int i = 0; i < m - 1; ++i)
        x[i] *= scal;
    *alpha = beta;
}

// dgeqr2: unblocked Householder QR, (H, tau) in the LAPACK convention.
template <class T> void ref_geqr2(int m, int n, T *A, int lda, T *tau)
{
    int k = std::min(m, n);
    for (int kk = 0; kk < k; ++kk) {
        ref_larfg(m - kk, &A[kk + kk * lda], &A[(kk + 1) + kk * lda], &tau[kk]);
        for (int j = kk + 1; j < n; ++j) {
            T w = A[kk + j * lda];
            for (int i = kk + 1; i < m; ++i)
                w += A[i + kk * lda] * A[i + j * lda];
            A[kk + j * lda] -= tau[kk] * w;
            for (int i = kk + 1; i < m; ++i)
                A[i + j * lda] -= tau[kk] * A[i + kk * lda] * w;
        }
    }
}

// dorm2r, side='L': B := Q^T B (trans 'T') or Q B ('N') from (H, tau).
template <class T>
void ref_orm2r(char trans, int m, int nrhs, int k, const T *A, int lda, const T *tau,
               T *B, int ldb)
{
    bool fwd = (trans == 'T');
    for (int s = 0; s < k; ++s) {
        int kk = fwd ? s : k - 1 - s;
        for (int j = 0; j < nrhs; ++j) {
            T w = B[kk + j * ldb];
            for (int i = kk + 1; i < m; ++i)
                w += A[i + kk * lda] * B[i + j * ldb];
            B[kk + j * ldb] -= tau[kk] * w;
            for (int i = kk + 1; i < m; ++i)
                B[i + j * ldb] -= tau[kk] * A[i + kk * lda] * w;
        }
    }
}

// Back substitution R X = B with R the upper triangle of an n x n array.
template <class T>
void ref_trsm_upper(int n, int nrhs, const T *R, int lda, T *B, int ldb)
{
    for (int j = 0; j < nrhs; ++j)
        for (int i = n - 1; i >= 0; --i) {
            T s = B[i + j * ldb];
            for (int l = i + 1; l < n; ++l)
                s -= R[i + l * lda] * B[l + j * ldb];
            B[i + j * ldb] = s / R[i + i * lda];
        }
}

// ----------------------- portable C API, by scalar type ------------
// compact<T>::geqrf / ormqr / potrf / sytrfnp / sytrsnp / sysvnp / trsm forward
// to the d/s entry points of cqr_compact.h, so the templated suites call one name for both precisions;
// compact<T>::name labels their output.

template <class T> struct compact;

// clang-format off
// NOLINTBEGIN(bugprone-macro-parentheses): T is a type name, p a token to paste
#define CQR_TEST_COMPACT_DISPATCH(T, p, label)                                             \
template <> struct compact<T> {                                                            \
    static constexpr const char *name = label;                                             \
    static int geqrf(char lay, int m, int n, T *a, int ld, T *tau, int V, int nm)          \
    { return p##geqrf_compact(lay, m, n, a, ld, tau, V, nm); }                             \
    static int ormqr(char tr, int m, int nrhs, int k, const T *a, int lda, const T *tau,   \
                     T *b, int ldb, int V, int nm)                                         \
    { return p##ormqr_compact(tr, m, nrhs, k, a, lda, tau, b, ldb, V, nm); }               \
    static int potrf(char lay, char up, int n, T *a, int ld, int V, int nm)                \
    { return p##potrf_compact(lay, up, n, a, ld, V, nm); }                                 \
    static int sytrfnp(char lay, char up, int n, T *a, int ld, int V, int nm)              \
    { return p##sytrfnp_compact(lay, up, n, a, ld, V, nm); }                               \
    static int sytrsnp(char lay, char up, int n, int nrhs, const T *a, int lda, T *b,      \
                       int ldb, int V, int nm)                                             \
    { return p##sytrsnp_compact(lay, up, n, nrhs, a, lda, b, ldb, V, nm); }                \
    static int sysvnp(char lay, char up, int n, int nrhs, T *a, int lda, T *b, int ldb,    \
                      int V, int nm)                                                       \
    { return p##sysvnp_compact(lay, up, n, nrhs, a, lda, b, ldb, V, nm); }                 \
    static int trsm(char lay, char si, char up, char tr, char di, int m, int n, T alpha,   \
                    const T *a, int lda, T *b, int ldb, int V, int nm)                     \
    { return p##trsm_compact(lay, si, up, tr, di, m, n, alpha, a, lda, b, ldb, V, nm); }   \
};
// NOLINTEND(bugprone-macro-parentheses)
// clang-format on

CQR_TEST_COMPACT_DISPATCH(double, d, "double")
CQR_TEST_COMPACT_DISPATCH(float, s, "float")
#undef CQR_TEST_COMPACT_DISPATCH

// ----------------------- input generation ----------------------------

// nm pointers into base, `stride` apart -- one per matrix, for the MKL pack API.
template <class T> std::vector<T *> batch_ptrs(T *base, int nm, size_t stride)
{
    std::vector<T *> p(nm);
    for (int v = 0; v < nm; ++v)
        p[v] = base + (size_t)v * stride;
    return p;
}

// Random m x n matrix (column-major) with the leading diagonal boosted by
// `boost`, which tames the conditioning of the square/tall QR and solve tests.
template <class T> void gen_boosted(T *A, int m, int n, T boost = T(2))
{
    for (size_t e = 0; e < (size_t)m * n; ++e)
        A[e] = frand<T>();
    for (int i = 0; i < std::min(m, n); ++i)
        A[i + (size_t)i * m] += boost;
}

// Symmetric positive-definite n x n matrix (column-major): A = M^T M + n*I, a
// tame condition number. cond > 0 squeezes the spectrum by a symmetric
// congruence D A D, D = diag(10^{-cond*i/(n-1)}) -- dynamic range, still SPD.
//
// The result is symmetric to the bit, not just to working precision. M^T M is
// symmetric in exact arithmetic, but A(i,j) and A(j,i) come from two separately
// evaluated dot products, so a value-unsafe FP model (icpx defaults to
// -fp-model=fast) can round them one ULP apart. The row-major "opposite triangle
// untouched" checks are sensitive to this: they read a row-major factor back
// through its transpose, comparing A(j,i) with A(i,j), so a sub-ULP asymmetry
// there reads as the kernel having written the wrong triangle. The trailing
// mirror pins A(j,i) == A(i,j) exactly, on every compiler and FP model.
template <class T> void gen_spd(T *A, int n, double cond = 0.0)
{
    std::vector<T> M((size_t)n * n);
    for (auto &x : M)
        x = frand<T>();
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            T s = 0;
            for (int l = 0; l < n; ++l)
                s += M[l + (size_t)i * n] * M[l + (size_t)j * n];
            A[i + (size_t)j * n] = s + (i == j ? T(n) : T(0));
        }
    if (cond > 0.0)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                double si = std::pow(10.0, -cond * (n > 1 ? (double)i / (n - 1) : 0.0));
                double sj = std::pow(10.0, -cond * (n > 1 ? (double)j / (n - 1) : 0.0));
                A[i + (size_t)j * n] *= (T)(si * sj);
            }
    // Mirror the lower triangle onto the upper so A(j,i) == A(i,j) bit-for-bit
    // (the congruence above scales A(i,j) and A(j,i) by the same si*sj, so it
    // preserves whatever symmetry M^T M produced; enforce it exactly here).
    for (int j = 0; j < n; ++j)
        for (int i = j + 1; i < n; ++i)
            A[j + (size_t)i * n] = A[i + (size_t)j * n];
}

// Symmetric *indefinite* n x n matrix (column-major) with a known-good
// unpivoted LDL^T: A = L D L^T built from a random unit-lower L (entries in
// (-0.5, 0.5), so element growth in re-factorization stays mild) and a diagonal
// D with |d| in [0.5, 2.5) and mixed signs (d_1 is forced negative for n >= 2,
// so the matrix is genuinely indefinite -- Cholesky would fail on it). Every
// leading principal minor is prod(d_1..d_k) != 0, so the unpivoted
// factorization exists and is exactly this (L, D). As in gen_spd, the upper
// triangle is mirrored from the lower so A is symmetric to the bit.
template <class T> void gen_sym_ldlt(T *A, int n)
{
    std::vector<T> Ls((size_t)n * n, T(0)), d(n);
    const auto L = mat_view(Ls.data(), n, n);
    const auto Av = mat_view(A, n, n);
    for (int j = 0; j < n; ++j) {
        L(j, j) = T(1);
        for (int i = j + 1; i < n; ++i)
            L(i, j) = T(0.5) * frand<T>();
        T mag = T(0.5) + std::abs(frand<T>()) * T(2);
        d[j] = (j == 1 || frand<T>() < 0) ? -mag : mag; // mixed signs, d_1 < 0
    }
    for (int j = 0; j < n; ++j)
        for (int i = j; i < n; ++i) {
            T s = 0;
            for (int l = 0; l <= j; ++l)
                s += L(i, l) * d[l] * L(j, l);
            Av(i, j) = s;
        }
    for (int j = 0; j < n; ++j)
        for (int i = j + 1; i < n; ++i)
            Av(j, i) = Av(i, j);
}

// Element (i,j) of an n x n unpivoted LDL^T factor as stored by ?sytrfnp:
// A(i,j) = sum_{l <= min(i,j)} F(i,l) D(l) F(j,l) with F the unit factor read
// from the strict lower triangle (uplo 'L') or, for the transpose dual
// A = U^T D U, from the strict upper one (F(i,l) = U(l,i)). `at(i,j)` reads the
// stored factor. Shared by the reconstruction checks of both ?sytrfnp suites.
template <class At> double ldlt_reconstruct(const At &at, int i, int j, bool upper)
{
    double s = 0;
    for (int l = 0; l <= std::min(i, j); ++l) {
        const double fi = (l == i) ? 1.0 : (upper ? (double)at(l, i) : (double)at(i, l));
        const double fj = (l == j) ? 1.0 : (upper ? (double)at(l, j) : (double)at(j, l));
        s += fi * (double)at(l, l) * fj;
    }
    return s;
}

// Fill one order-s triangular matrix (leading dim s) in the given layout:
// random in the referenced triangle, the diagonal boosted away from zero for
// conditioning, the other (never-referenced) triangle zeroed. Shared by the
// ?trsm suites; A is square, so lda = s for both layouts.
template <class T> void gen_tri(T *A, int s, bool upper, bool rowmajor = false)
{
    auto at = [&](int i, int j) -> T & {
        return A[rowmajor ? (size_t)i * s + j : i + (size_t)j * s];
    };
    for (int i = 0; i < s; ++i)
        for (int j = 0; j < s; ++j) {
            bool ref = upper ? (i <= j) : (i >= j);
            at(i, j) = ref ? frand<T>() : T(0);
        }
    for (int d = 0; d < s; ++d)
        at(d, d) = (at(d, d) >= 0 ? T(1) : T(-1)) * (T(2) + std::abs(frand<T>()));
}

// Apply a triangular operator to a general matrix -- the "forward" direction of
// a ?trsm, for checking a solve's defining residual ||op(A) X - alpha B||.
// R (m x n, column-major, ld m) := op(A) X (side 'L') or X op(A) (side 'R'),
// with A the order-s (s = m for 'L', n for 'R') triangular factor: uplo 'U'/'L',
// op(A) = A ('N') or A^T ('T'/'C'), unit ('U') or non-unit ('N') diagonal.
template <class T>
void tri_apply(char side, char uplo, char transa, char diag, int m, int n, const T *A,
               int lda, const T *X, int ldx, T *R)
{
    const bool left = (side == 'L' || side == 'l');
    const bool upper = (uplo == 'U' || uplo == 'u');
    const bool tran = (transa == 'T' || transa == 't' || transa == 'C' || transa == 'c');
    const bool unit = (diag == 'U' || diag == 'u');
    const int s = left ? m : n;
    // op(A)(i,k): unit or A(i,i) on the diagonal; off-diagonal is the referenced
    // entry of A (op = A) or of its transpose (op = A^T), else zero.
    auto Mop = [&](int i, int k) -> T {
        if (i == k) return unit ? T(1) : A[i + (size_t)i * lda];
        const bool ref = tran ? (upper ? i > k : i < k) : (upper ? k > i : k < i);
        if (!ref) return T(0);
        return tran ? A[k + (size_t)i * lda] : A[i + (size_t)k * lda];
    };
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            T acc = 0;
            if (left)
                for (int k = 0; k < s; ++k)
                    acc += Mop(i, k) * X[k + (size_t)j * ldx];
            else
                for (int k = 0; k < s; ++k)
                    acc += X[i + (size_t)k * ldx] * Mop(k, j);
            R[i + (size_t)j * m] = acc;
        }
}

// ----------------------- dense batches and compact packing -----------

// A batch of `count` column-major rows x cols matrices in one contiguous buffer;
// matrix idx starts at idx*rows*cols with leading dimension rows.
template <class T> class MatrixBatch {
  public:
    MatrixBatch(int count, int rows, int cols)
        : count_(count), rows_(rows), cols_(cols), a_((size_t)count * rows * cols)
    {
    }
    // clang-format off
    int count() const { return count_; }
    int rows()  const { return rows_; }
    int cols()  const { return cols_; }
    T       *operator[](int idx)       { return a_.data() + (size_t)idx * rows_ * cols_; }
    const T *operator[](int idx) const { return a_.data() + (size_t)idx * rows_ * cols_; }
    MatrixView<T>       view(int idx)       { return mat_view((*this)[idx], rows_, cols_); }
    ConstMatrixView<T>  view(int idx) const { return mat_view((*this)[idx], rows_, cols_); }
    T       &operator()(int idx, int i, int j)       { return (*this)[idx][i + (size_t)j * rows_]; }
    const T &operator()(int idx, int i, int j) const { return (*this)[idx][i + (size_t)j * rows_]; }
    // clang-format on

  private:
    int count_, rows_, cols_;
    std::vector<T> a_;
};

// Compact pack/unpack (matches mkl_?gepack_compact). group g = idx/V, slot
// v = idx%V; element (i,j) of matrix idx lives at, per layout,
//   col-major: p[g*ldp*cols*V + (j*ldp + i)*V + v]
//   row-major: p[g*ldp*rows*V + (i*ldp + j)*V + v]
// Padded slots (idx >= nm) carry the identity.
template <class T>
void pack_compact(const MatrixBatch<T> &Mk, T *p, int ldp, int V, bool rowmajor = false)
{
    const int m = Mk.rows(), n = Mk.cols(), nm = Mk.count();
    const int ng = (nm + V - 1) / V;
    const size_t gstride = (size_t)ldp * (rowmajor ? m : n) * V;
    for (int g = 0; g < ng; ++g)
        for (int v = 0; v < V; ++v) {
            int idx = g * V + v;
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < m; ++i) {
                    size_t off = rowmajor ? ((size_t)i * ldp + j) : ((size_t)j * ldp + i);
                    p[g * gstride + off * V + v] =
                        (idx < nm) ? Mk(idx, i, j) : (i == j ? T(1) : T(0));
                }
        }
}

template <class T>
void unpack_compact(MatrixBatch<T> &Mk, const T *p, int ldp, int V, bool rowmajor = false)
{
    const int m = Mk.rows(), n = Mk.cols(), nm = Mk.count();
    const int ng = (nm + V - 1) / V;
    const size_t gstride = (size_t)ldp * (rowmajor ? m : n) * V;
    for (int g = 0; g < ng; ++g)
        for (int v = 0; v < V; ++v) {
            int idx = g * V + v;
            if (idx >= nm) continue;
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < m; ++i) {
                    size_t off = rowmajor ? ((size_t)i * ldp + j) : ((size_t)j * ldp + i);
                    Mk(idx, i, j) = p[g * gstride + off * V + v];
                }
        }
}

// A tau batch (k scalars per matrix) is packed as k x 1 matrices; padded slots
// get tau = 0 (the identity's reflectors).
template <class T> void pack_tau(const MatrixBatch<T> &tau, T *tp, int V)
{
    const int k = tau.rows(), nm = tau.count(), ng = (nm + V - 1) / V;
    for (int g = 0; g < ng; ++g)
        for (int v = 0; v < V; ++v) {
            int idx = g * V + v;
            for (int kk = 0; kk < k; ++kk)
                tp[(size_t)g * k * V + (size_t)kk * V + v] =
                    (idx < nm) ? tau[idx][kk] : T(0);
        }
}

template <class T> void unpack_tau(MatrixBatch<T> &tau, const T *tp, int V)
{
    const int k = tau.rows(), nm = tau.count(), ng = (nm + V - 1) / V;
    for (int g = 0; g < ng; ++g)
        for (int v = 0; v < V; ++v) {
            int idx = g * V + v;
            if (idx >= nm) continue;
            for (int kk = 0; kk < k; ++kk)
                tau[idx][kk] = tp[(size_t)g * k * V + (size_t)kk * V + v];
        }
}

} // namespace cqr::test

#endif // TEST_COMPACT_UTIL_HPP
