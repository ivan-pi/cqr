// test_cqr_syrk_compact.cpp
//
// Self-contained validation of the templated compact symmetric rank-k update
// (dsyrk_compact / ssyrk_compact), with no BLAS dependency. The reference is a
// scalar ?syrk implemented directly -- the same terms the vectorized kernel
// accumulates V lanes at a time, but summed in a deliberately different order
// (rank-1 outer products, p outermost) than the kernel's per-element dot (p
// innermost), so a shared arithmetic bug cannot pass unseen while a correct
// kernel still agrees to working precision.
//
// Two parts:
//   1. C API argument validation -- exercises the LAPACK/BLAS-style info = -j
//      contract of the portable entry points.
//   2. Numerical correctness over the full uplo x trans x layout matrix,
//      precisions, and interleave widths (including padded partial groups):
//      the whole n x n C is compared, so the check simultaneously gates the
//      active-triangle result and that the opposite triangle is left untouched.
//      All four (trans, layout) combinations are covered so the tuned
//      (trans='T', column-major) kernel and the strided kernel are both exercised.
//
// Assisted-by: Claude:claude-opus-4.8

#include <cstdio>
#include <cmath>
#include <vector>
#include <limits>
#include <algorithm>

#include "cqr_compact.h"         // dsyrk_compact / ssyrk_compact (C entry points)
#include "test_compact_util.hpp" // rng/frand, max_abs_diff, norm1, MatrixBatch, pack/unpack

using namespace cqr::test;

// ----------------------- reference kernel (scalar) ------------------
// Dense column-major BLAS ?syrk: C := alpha op(A) op(A)^T + beta C, writing only
// the uplo triangle of the symmetric n x n C. op(A) = A (trans='N', A is n x k)
// or A^T (trans='T'/'C', A is k x n). The rank-k product is accumulated as a sum
// of rank-1 outer products (contraction index p outermost) -- a different
// summation order from the kernel's inner-product form, so the two round
// differently and a bug common to both cannot hide. beta = 0 overwrites (the
// prior C is not read), matching the kernel and reference BLAS ?syrk.

template <class T>
static void ref_syrk(char uplo, char trans, int n, int k, T alpha, const T *A, int lda,
                     T beta, T *C, int ldc)
{
    const bool upper = (uplo == 'U' || uplo == 'u');
    const bool tran = (trans == 'T' || trans == 't' || trans == 'C' || trans == 'c');
    // op(A)(i,p): A(i,p) for 'N', A(p,i) for 'T' (both from column-major A).
    auto Aop = [&](int i, int p) -> T {
        return tran ? A[p + (size_t)i * lda] : A[i + (size_t)p * lda];
    };
    auto Ce = [&](int i, int j) -> T & { return C[i + (size_t)j * ldc]; };

    // C := beta C (or 0) on the active triangle first, then accumulate.
    for (int j = 0; j < n; ++j) {
        const int ilo = upper ? 0 : j, ihi = upper ? j + 1 : n;
        for (int i = ilo; i < ihi; ++i)
            Ce(i, j) = (beta == T(0)) ? T(0) : beta * Ce(i, j);
    }
    for (int p = 0; p < k; ++p)
        for (int j = 0; j < n; ++j) {
            const int ilo = upper ? 0 : j, ihi = upper ? j + 1 : n;
            for (int i = ilo; i < ihi; ++i)
                Ce(i, j) += alpha * Aop(i, p) * Aop(j, p);
        }
}

// precision-overloaded shim: pick d/s by the pointer type
static int syrk_c(char lay, char up, char tr, int n, int k, double al, const double *a,
                  int lda, double be, double *c, int ldc, int V, int nm)
{
    return dsyrk_compact(lay, up, tr, n, k, al, a, lda, be, c, ldc, V, nm);
}
static int syrk_c(char lay, char up, char tr, int n, int k, float al, const float *a,
                  int lda, float be, float *c, int ldc, int V, int nm)
{
    return ssyrk_compact(lay, up, tr, n, k, al, a, lda, be, c, ldc, V, nm);
}

// --------------------------- one numerical case ---------------------
// A is n x k (trans='N') or k x n (trans='T'); C is the symmetric n x n result.
// The MatrixBatch backing is column-major; pack_compact serializes it in the
// requested compact layout, so a single reference (column-major) covers both.

template <class T, int V>
static int run_case(bool rowmajor, char uplo, char trans, int nm, int n, int k)
{
    const bool tran = (trans == 'T');
    const T eps = std::numeric_limits<T>::epsilon();
    const T alpha = T(0.5) + frand<T>(); // non-trivial, non-zero coefficients
    const T beta = T(0.25) + frand<T>();

    const int Arows = tran ? k : n, Acols = tran ? n : k;
    const int ldAd = Arows;                    // dense (col-major) leading dim
    const int ldAp = rowmajor ? Acols : Arows; // compact leading dim
    const int ldC = n;

    // random A and a random (non-symmetric) C; syrk touches only the uplo
    // triangle, so checking the whole matrix confirms the opposite triangle is
    // left intact.
    MatrixBatch<T> A(nm, Arows, Acols), C(nm, n, n), Cref(nm, n, n);
    for (int idx = 0; idx < nm; ++idx) {
        for (size_t e = 0; e < (size_t)Arows * Acols; ++e)
            A[idx][e] = frand<T>();
        for (size_t e = 0; e < (size_t)n * n; ++e)
            C[idx][e] = frand<T>();
        std::copy(C[idx], C[idx] + (size_t)n * n, Cref[idx]);
        ref_syrk<T>(uplo, trans, n, k, alpha, A[idx], ldAd, beta, Cref[idx], ldC);
    }

    // pack, run the routine under test, unpack
    const int ng = (nm + V - 1) / V;
    std::vector<T> ap((size_t)ng * Arows * Acols * V), cp((size_t)ng * n * n * V);
    pack_compact(A, ap.data(), ldAp, V, rowmajor);
    pack_compact(C, cp.data(), ldC, V, rowmajor);

    const int info = syrk_c(rowmajor ? 'R' : 'C', uplo, trans, n, k, alpha, ap.data(),
                            ldAp, beta, cp.data(), ldC, V, nm);

    MatrixBatch<T> Cout(nm, n, n);
    unpack_compact(Cout, cp.data(), ldC, V, rowmajor);

    // whole-matrix relative error: active triangle correct + opposite untouched.
    double worst = 0;
    for (int idx = 0; idx < nm; ++idx)
        worst = std::max(worst, max_abs_diff(Cout[idx], Cref[idx], (size_t)n * n) /
                                    std::max(norm1(Cref[idx], n, n), 1e-300));

    // The reference sums k terms in a different order than the kernel, so the two
    // agree only to working precision (scaled by the accumulation length k).
    const double rtol = 32.0 * (k + 1) * (double)eps;
    const bool ok = (worst <= rtol) && (info == 0);
    std::printf("  T=%-6s V=%-2d %s uplo=%c trans=%c nm=%-2d n=%-3d k=%-3d | "
                "rel %.2e (rtol %.2e) info=%d %s\n",
                sizeof(T) == 8 ? "double" : "float", V, rowmajor ? "row" : "col", uplo,
                trans, nm, n, k, worst, rtol, info, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

// --------------------- C API argument validation --------------------

static int test_validation()
{
    const int n = 6, k = 4, V = 4, nm = 4;
    // buffers sized for the largest valid case (A up to 6x6, C is 6x6), zeroed so
    // the valid calls that actually run the kernel produce a well-defined result.
    std::vector<double> A((size_t)n * n * V, 0), C((size_t)n * n * V, 0);
    auto call = [&A, &C](char lay, char up, char tr, int n_, int k_, int ldap_, int ldcp_,
                         int V_, int nm_) {
        return dsyrk_compact(lay, up, tr, n_, k_, 1.0, A.data(), ldap_, 0.0, C.data(),
                             ldcp_, V_, nm_);
    };
    // clang-format off
    struct { const char *what; int got, want; } t[] = {
        {"valid N col",  call('C','U','N', n, k, n, n, V, nm),   0},
        {"valid T col",  call('C','L','T', n, k, k, n, V, nm),   0},
        {"valid N row",  call('R','U','N', n, k, k, n, V, nm),   0},
        {"valid T row",  call('R','L','T', n, k, n, n, V, nm),   0},
        {"trans=C",      call('C','U','C', n, k, k, n, V, nm),   0},
        {"k=0 (C:=bC)",  call('C','U','N', n, 0, n, n, V, nm),   0},
        {"bad layout",   call('X','U','N', n, k, n, n, V, nm),  -1},
        {"bad uplo",     call('C','X','N', n, k, n, n, V, nm),  -2},
        {"bad trans",    call('C','U','X', n, k, n, n, V, nm),  -3},
        {"n<0",          call('C','U','N', -1, k, n, n, V, nm), -4},
        {"k<0",          call('C','U','N', n, -1, n, n, V, nm), -5},
        {"ldap<n (Ncol)",call('C','U','N', n, k, n-1, n, V, nm), -8},
        {"ldap<k (Tcol)",call('C','U','T', n, k, k-1, n, V, nm), -8},
        {"ldap<k (Nrow)",call('R','U','N', n, k, k-1, n, V, nm), -8},
        {"ldcp<n",       call('C','U','N', n, k, n, n-1, V, nm),-11},
        {"bad V",        call('C','U','N', n, k, n, n, 3, nm), -12},
        {"nm<0",         call('C','U','N', n, k, n, n, V, -1), -13},
        {"empty n=0",    call('C','U','N', 0, k, 1, 1, V, nm),   0},
        {"empty nm=0",   call('C','U','N', n, k, n, n, V, 0),    0},
    };
    // clang-format on
    int bad = 0;
    for (auto &c : t)
        bad += (c.got != c.want);
    std::printf("C API validation: %zu checks | %s\n", sizeof(t) / sizeof(t[0]),
                bad ? "FAIL" : "OK");
    for (auto &c : t)
        if (c.got != c.want)
            std::printf("  %-15s got=%d want=%d\n", c.what, c.got, c.want);
    return bad ? 1 : 0;
}

// ------------------------------- main --------------------------------

int main()
{
    std::printf("compact syrk portable test\n");

    int fails = 0;
    fails += test_validation();

    std::printf("numerical cases (vs scalar reference, rank-1 accumulation order):\n");
    // full uplo x trans x layout matrix at a representative shape/width: the
    // (T,col) case runs the tuned kernel, the other three the strided kernel.
    for (bool rowmajor : {false, true})
        for (char uplo : {'U', 'L'})
            for (char trans : {'N', 'T'})
                fails += run_case<double, 4>(rowmajor, uplo, trans, 8, 9, 5);

    // precisions, widths, tall/wide factors, and padded partial final groups
    fails += run_case<double, 2>(false, 'L', 'T', 6, 12, 4);
    fails += run_case<double, 8>(false, 'U', 'T', 16, 10, 20); // wide (k > n)
    fails += run_case<double, 8>(false, 'L', 'N', 11, 12, 3);  // padded last group
    fails += run_case<double, 8>(true, 'U', 'N', 11, 7, 9);    // row-major, padded
    fails += run_case<float, 8>(false, 'U', 'T', 16, 16, 4);
    fails += run_case<float, 16>(false, 'L', 'T', 32, 10, 7);
    fails += run_case<float, 16>(true, 'L', 'N', 32, 8, 6); // row-major strided

    // corner cases: single matrix, n = 1, k = 1 (rank-1), n = k = JB width
    for (char trans : {'N', 'T'})
        for (char uplo : {'U', 'L'}) {
            fails += run_case<double, 4>(false, uplo, trans, 1, 5, 3); // nm = 1
            fails += run_case<double, 4>(false, uplo, trans, 5, 1, 4); // n = 1
            fails += run_case<double, 4>(false, uplo, trans, 4, 6, 1); // k = 1
            fails += run_case<double, 4>(false, uplo, trans, 8, 4, 4); // n = k = 4
        }

    if (fails) {
        std::printf("\n%d CHECK(S) FAILED\n", fails);
        return 1;
    }
    std::printf("\nall checks passed\n");
    return 0;
}
