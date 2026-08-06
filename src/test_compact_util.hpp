// test_compact_util.hpp
//
// Shared helpers for the compact-format test suites: a seeded RNG, error
// metrics, SPD input generation, a triangular-operator apply, and Compact
// pack/unpack. Header-only and MKL-free, so the BLAS-free portable tests use it
// too.
//
// Assisted-by: Claude:claude-opus-4.8

#ifndef TEST_COMPACT_UTIL_HPP
#define TEST_COMPACT_UTIL_HPP

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>
#include <algorithm>

namespace cqr {
namespace test {

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

// max |a| over n elements (BLAS i?amax magnitude, without the index).
template <class T> double maxabs(const T *a, size_t n)
{
    double d = 0;
    for (size_t i = 0; i < n; ++i)
        d = std::max(d, (double)std::abs(a[i]));
    return d;
}

// L1 (max column sum) norm of a column-major m x n matrix. Templated so the
// portable FP32/FP64 suites can reuse it; existing double callers deduce
// T = double and are unaffected.
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

// nm pointers into base, `stride` apart -- one per matrix, for the MKL pack API.
template <class T> std::vector<T *> batch_ptrs(T *base, int nm, size_t stride)
{
    std::vector<T *> p(nm);
    for (int v = 0; v < nm; ++v)
        p[v] = base + (size_t)v * stride;
    return p;
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

} // namespace test
} // namespace cqr

#endif // TEST_COMPACT_UTIL_HPP
