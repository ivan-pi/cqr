// test_cqr_trsm_compact.cpp
//
// Self-contained validation of the templated compact triangular solve
// (dtrsm_compact / strsm_compact), with no BLAS dependency. The reference is a
// scalar BLAS-?trsm implemented directly -- the same math the vectorized kernel
// executes V lanes at a time, so a correct kernel matches it to working
// precision.
//
// Two parts:
//   1. C API argument validation -- exercises the LAPACK/BLAS-style info = -j
//      contract of the portable entry points.
//   2. Numerical correctness over side / uplo / transa / diag, precisions and
//      interleave widths (column-major, the tuned path): forward error vs a
//      scalar reference solve, plus the solve's own residual ||op(A) X - alpha B||
//      formed with an independent triangular multiply (so a bug shared by the
//      reference and the kernel cannot pass unseen).
//
// Assisted-by: Claude:claude-opus-4.8

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <limits>
#include <algorithm>

#include "test_compact_util.hpp" // compact<T>, frand, gen_tri, tri_apply, MatrixBatch, pack/unpack

using namespace cqr::test;

// ----------------------- reference kernel (scalar) ------------------
// Dense BLAS ?trsm: solves op(A) X = alpha B (side='L') or
// X op(A) = alpha B (side='R') in place, A the order-s triangular factor.
// B is pre-scaled by alpha (so alpha == 0 gives B := 0), then a unit-alpha
// substitution runs. Only the referenced triangle of A is touched; the
// diagonal is skipped entirely when diag='U'.

template <class T, class Av>
static void ref_trsm(char side, char uplo, char transa, char diag, T alpha, Av A,
                     MatrixView<T> B)
{
    const bool left = (side == 'L' || side == 'l');
    const bool upper = (uplo == 'U' || uplo == 'u');
    const bool tran = (transa == 'T' || transa == 't' || transa == 'C' || transa == 'c');
    const bool unit = (diag == 'U' || diag == 'u');
    const int m = B.rows, n = B.cols;
    auto Ae = [&](int i, int j) -> T { return A(i, j); };
    auto Be = [&](int i, int j) -> T & { return B(i, j); };

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            Be(i, j) *= alpha; // B := alpha B (alpha == 0 -> B := 0)

    if (left) {
        // solve op(A) X = B column by column; A is m x m
        const bool back = (upper != tran);
        for (int j = 0; j < n; ++j)
            for (int t = 0; t < m; ++t) {
                int i = back ? m - 1 - t : t;
                T s = Be(i, j);
                if (back)
                    for (int l = i + 1; l < m; ++l)
                        s -= (tran ? Ae(l, i) : Ae(i, l)) * Be(l, j);
                else
                    for (int l = 0; l < i; ++l)
                        s -= (tran ? Ae(l, i) : Ae(i, l)) * Be(l, j);
                Be(i, j) = unit ? s : s / Ae(i, i);
            }
    }
    else {
        // solve X op(A) = B column of X at a time; A is n x n
        const bool fwd = (upper != tran);
        for (int t = 0; t < n; ++t) {
            int j = fwd ? t : n - 1 - t;
            if (fwd)
                for (int l = 0; l < j; ++l) {
                    T a = tran ? Ae(j, l) : Ae(l, j);
                    for (int i = 0; i < m; ++i)
                        Be(i, j) -= a * Be(i, l);
                }
            else
                for (int l = j + 1; l < n; ++l) {
                    T a = tran ? Ae(j, l) : Ae(l, j);
                    for (int i = 0; i < m; ++i)
                        Be(i, j) -= a * Be(i, l);
                }
            if (!unit) {
                T d = Ae(j, j);
                for (int i = 0; i < m; ++i)
                    Be(i, j) /= d;
            }
        }
    }
}

// Padded pack slots carry the identity -- for a triangular A that is a unit
// diagonal, so the kernel's divisions never hit a zero pivot in the padding.

// --------------------------- one numerical case ---------------------
// Column-major; A is the order-s triangular factor, B is m x n.

template <class T, int V>
static int run_case(char side, char uplo, char transa, char diag, int nm, int m, int n)
{
    const bool left = (side == 'L');
    const int s = left ? m : n;
    const T eps = std::numeric_limits<T>::epsilon();
    const T alpha = T(0.5) + frand<T>(); // a non-trivial, non-zero scalar

    // triangular A (s x s), column-major: random in the referenced triangle,
    // diagonal boosted for conditioning, the other triangle zeroed.
    MatrixBatch<T> A(nm, s, s);
    const bool up = (uplo == 'U');
    for (int idx = 0; idx < nm; ++idx)
        gen_tri(A.view(idx), up);

    // random B (m x n) and its reference solution
    MatrixBatch<T> B(nm, m, n), Xref(nm, m, n);
    for (int idx = 0; idx < nm; ++idx) {
        for (size_t e = 0; e < (size_t)m * n; ++e)
            B[idx][e] = frand<T>();
        std::copy(B[idx], B[idx] + (size_t)m * n, Xref[idx]);
        ref_trsm(side, uplo, transa, diag, alpha, A.view(idx), Xref.view(idx));
    }

    // pack, solve with the routine under test, unpack
    int ng = (nm + V - 1) / V;
    std::vector<T> ap((size_t)ng * s * s * V), bp((size_t)ng * m * n * V);
    pack_compact(A, ap.data(), s, V);
    pack_compact(B, bp.data(), m, V);

    int info = compact<T>::trsm('C', side, uplo, transa, diag, m, n, alpha, ap.data(), s,
                                bp.data(), m, V, nm);

    MatrixBatch<T> Bout(nm, m, n);
    unpack_compact(Bout, bp.data(), m, V);

    // Two gates: (1) forward error vs the scalar reference solve, and (2) the
    // solve's own defining residual ||op(A) X - alpha B||, formed with an
    // independent triangular multiply -- so a bug shared by ref_trsm and the
    // kernel cannot slip through (the ?trsm analogue of the reconstruction /
    // round-trip identities the geqrf/potrf/ormqr self-tests check).
    double worst_fwd = 0, worst_res = 0;
    std::vector<T> Rs((size_t)m * n), aBs((size_t)m * n);
    const auto R = mat_view(Rs.data(), m, n), aB = mat_view(aBs.data(), m, n);
    for (int idx = 0; idx < nm; ++idx) {
        worst_fwd =
            std::max(worst_fwd, max_abs_diff(Bout[idx], Xref[idx], (size_t)m * n) /
                                    std::max(norm1(Xref.view(idx)), 1e-300));
        tri_apply(side, uplo, transa, diag, A.view(idx), Bout.view(idx), R);
        for (size_t e = 0; e < (size_t)m * n; ++e)
            aBs[e] = alpha * B[idx][e];
        worst_res =
            std::max(worst_res, max_abs_diff(Rs.data(), aBs.data(), (size_t)m * n) /
                                    std::max(norm1(aB), 1e-300));
    }
    const double worst = std::max(worst_fwd, worst_res);

    // Backward-stable triangular solve on a diagonal-boosted A: both the forward
    // error and the residual stay near working precision.
    const double rtol = 1e3 * s * (double)eps;
    const bool ok = (worst <= rtol);
    std::printf("  T=%-6s V=%-2d side=%c uplo=%c tr=%c diag=%c nm=%-2d m=%-3d n=%-3d | "
                "fwd %.1e res %.1e (rtol %.1e) info=%d %s\n",
                compact<T>::name, V, side, uplo, transa, diag, nm, m, n, worst_fwd,
                worst_res, rtol, info, (ok && info == 0) ? "OK" : "FAIL");

    return (info != 0) + !ok;
}

// --------------------- C API argument validation --------------------

static int test_validation()
{
    const int m = 8, n = 6, V = 4, nm = 4;
    // buffers sized for the largest valid case (A up to m x m, B up to m x n)
    std::vector<double> A((size_t)m * m * V, 0), B((size_t)m * n * V, 0);
    auto call = [&A, &B](char lay, char si, char up, char tr, char di, int m_, int n_,
                         int lda_, int ldb_, int V_, int nm_) {
        return dtrsm_compact(lay, si, up, tr, di, m_, n_, 1.0, A.data(), lda_, B.data(),
                             ldb_, V_, nm_);
    };
    // clang-format off
    struct { const char *what; int got, want; } t[] = {
        {"valid L col",  call('C','L','U','N','N', m, n,  m,   m,   V, nm),   0},
        {"valid R col",  call('C','R','L','T','U', m, n,  n,   m,   V, nm),   0},
        {"valid row",    call('R','L','U','N','N', m, n,  m,   n,   V, nm),   0},
        {"transa=C",     call('C','L','U','C','N', m, n,  m,   m,   V, nm),   0},
        {"bad layout",   call('X','L','U','N','N', m, n,  m,   m,   V, nm),  -1},
        {"bad side",     call('C','X','U','N','N', m, n,  m,   m,   V, nm),  -2},
        {"bad uplo",     call('C','L','X','N','N', m, n,  m,   m,   V, nm),  -3},
        {"bad transa",   call('C','L','U','X','N', m, n,  m,   m,   V, nm),  -4},
        {"bad diag",     call('C','L','U','N','X', m, n,  m,   m,   V, nm),  -5},
        {"m<0",          call('C','L','U','N','N', -1, n, m,   m,   V, nm),  -6},
        {"n<0",          call('C','L','U','N','N', m, -1, m,   m,   V, nm),  -7},
        {"ldap<m (L)",   call('C','L','U','N','N', m, n,  m-1, m,   V, nm), -10},
        {"ldap<n (R)",   call('C','R','U','N','N', m, n,  n-1, m,   V, nm), -10},
        {"ldbp<m (col)", call('C','L','U','N','N', m, n,  m,   m-1, V, nm), -12},
        {"ldbp<n (row)", call('R','L','U','N','N', m, n,  m,   n-1, V, nm), -12},
        {"bad V",        call('C','L','U','N','N', m, n,  m,   m,   3, nm), -13},
        {"nm<0",         call('C','L','U','N','N', m, n,  m,   m,   V, -1), -14},
        {"empty m=0",    call('C','L','U','N','N', 0, n,  1,   1,   V, nm),   0},
        {"empty n=0",    call('C','L','U','N','N', m, 0,  m,   m,   V, nm),   0},
        {"empty nm=0",   call('C','L','U','N','N', m, n,  m,   m,   V, 0),    0},
    };
    // clang-format on
    int bad = 0;
    for (auto &c : t)
        bad += (c.got != c.want);
    std::printf("C API validation: %zu checks | %s\n", sizeof(t) / sizeof(t[0]),
                bad ? "FAIL" : "OK");
    for (auto &c : t)
        if (c.got != c.want)
            std::printf("  %-13s got=%d want=%d\n", c.what, c.got, c.want);
    return bad ? 1 : 0;
}

// ------------------------------- main --------------------------------

int main()
{
    std::printf("compact trsm portable test\n");

    int fails = 0;
    fails += test_validation();

    std::printf("numerical cases (column-major, vs scalar reference):\n");
    // full side x uplo x transa x diag matrix at a representative shape/width
    for (char side : {'L', 'R'})
        for (char uplo : {'U', 'L'})
            for (char tr : {'N', 'T'})
                for (char di : {'N', 'U'})
                    fails += run_case<double, 4>(side, uplo, tr, di, 8, 12, 5);

    // precisions, widths, and a padded partial final group
    fails += run_case<double, 2>('L', 'U', 'N', 'N', 6, 16, 4);
    fails += run_case<double, 8>('L', 'L', 'T', 'N', 16, 24, 3);
    fails += run_case<double, 8>('L', 'U', 'N', 'N', 11, 20, 4); // padded last group
    fails += run_case<float, 8>('L', 'U', 'N', 'N', 16, 16, 4);
    fails += run_case<float, 16>('R', 'L', 'N', 'U', 32, 10, 7);
    fails += run_case<double, 4>('L', 'U', 'N', 'N', 40, 16, 4); // 10 groups: OpenMP path

    // few-RHS no-transpose left (n = 1,2,3): the column-axpy kernel route
    // (n >= 4 above routes to the row-dot kernel), across uplo / diag / width
    for (int nrhs : {1, 2, 3})
        for (char uplo : {'U', 'L'}) {
            fails += run_case<double, 4>('L', uplo, 'N', 'N', 8, 16, nrhs);
            fails += run_case<double, 8>('L', uplo, 'N', 'U', 11, 20, nrhs); // padded
        }
    fails += run_case<float, 8>('L', 'U', 'N', 'N', 16, 16, 1);
    fails += run_case<float, 16>('L', 'L', 'N', 'N', 32, 12, 2);

    if (fails) {
        std::printf("\n%d CHECK(S) FAILED\n", fails);
        return 1;
    }
    std::printf("\nall checks passed\n");
    return 0;
}
