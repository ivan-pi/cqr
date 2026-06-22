/* solve_qr_dense.cpp
 *
 * Worked example: solving a square system AX = B by QR, spelled out as the
 * explicit LAPACK sequence and cross-checked against the one-call
 * LAPACKE_dgels driver. This is the dense, single-matrix analogue of the
 * Compact-format batch pipeline this repository extends (design doc 7.2):
 *
 *     dgeqrf   A -> (H,R,tau)   A = Q R          (mkl_dgeqrf_compact)
 *     dormqr   B <- Q^T B       R X = Q^T B      (ext_mkl_dormqr_compact)
 *     dtrsm    R X = (Q^T B)    X = R^{-1}(Q^T B) (mkl_dtrsm_compact)
 *
 * LAPACKE_dgels does all three internally (reducing to this QR solve when
 * m == n, trans = 'N'); we run both and confirm they agree with each other
 * and with the known exact solution.
 *
 * Build: needs LAPACKE + a BLAS (here Intel MKL, the project's BLAS); wired
 * up by CMakeLists.txt as the `solve_qr_dense` target.
 */

#include <mkl.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>

namespace {

const double eps = std::numeric_limits<double>::epsilon();

double frand() { return 2.0 * std::rand() / (double)RAND_MAX - 1.0; }

/* Report and abort on the spot if cond is false. */
void check(bool cond, const char *what)
{
    if (!cond) { std::printf("FAILED: %s\n", what); std::exit(1); }
}

/* Minimal column-major dense matrix: owns its storage and hands raw pointers
 * (data(), ld()) to BLAS/LAPACK. Leading dimension == row count. */
class Matrix {
public:
    Matrix(int rows, int cols)
        : rows_(rows), cols_(cols), a_((size_t)rows * cols) {}

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int ld()   const { return rows_; }
    double       *data()       { return a_.data(); }
    const double *data() const { return a_.data(); }
    double &operator()(int i, int j)       { return a_[i + (size_t)j * rows_]; }
    double  operator()(int i, int j) const { return a_[i + (size_t)j * rows_]; }

private:
    int rows_, cols_;
    std::vector<double> a_;
};

/* 1-norm of a matrix, ||A||_1 (max column sum). */
double norm1(const Matrix &A)
{
    return LAPACKE_dlange(LAPACK_COL_MAJOR, '1', A.rows(), A.cols(), A.data(), A.ld());
}

/* Relative 1-norm difference ||A - B||_1 / ||B||_1 (A, B same shape). */
double rel_diff(const Matrix &A, const Matrix &B)
{
    Matrix D = A;                                 /* D <- A */
    cblas_daxpy((size_t)D.rows() * D.cols(), -1.0, B.data(), 1, D.data(), 1);  /* D <- A - B */
    return norm1(D) / std::max(norm1(B), 1e-300);
}

void solve(int n, int nrhs)
{
    /* Known exact solution X(:,j) = j+1, so B = A X and the solve must
     * recover X (mirrors the compact suite-2 construction). */
    Matrix A(n, n), X(n, nrhs), B(n, nrhs);
    for (int j = 0; j < nrhs; ++j)
        for (int i = 0; i < n; ++i) X(i, j) = double(j + 1);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) A(i, j) = frand();
    for (int i = 0; i < n; ++i) A(i, i) += 2.0;   /* tame conditioning */
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, nrhs, n,
                1.0, A.data(), A.ld(), X.data(), X.ld(), 0.0, B.data(), B.ld());  /* B = A X */

    /* --- Path 1: explicit QR solve  dgeqrf -> dormqr -> dtrsm --- */
    Matrix Aqr = A, Xqr = B;
    std::vector<double> tau(n);
    lapack_int info = LAPACKE_dgeqrf(LAPACK_COL_MAJOR, n, n, Aqr.data(), Aqr.ld(), tau.data());
    if (!info)  /* Xqr <- Q^T B  (the step ext_mkl_dormqr_compact fills) */
        info = LAPACKE_dormqr(LAPACK_COL_MAJOR, 'L', 'T', n, nrhs, n,
                              Aqr.data(), Aqr.ld(), tau.data(), Xqr.data(), Xqr.ld());
    check(info == 0, "manual QR (dgeqrf/dormqr)");
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                n, nrhs, 1.0, Aqr.data(), Aqr.ld(), Xqr.data(), Xqr.ld());  /* Xqr <- R^{-1} Xqr */

    /* --- Path 2: naive forward driver  LAPACKE_dgels --- */
    Matrix Adg = A, Xdg = B;
    info = LAPACKE_dgels(LAPACK_COL_MAJOR, 'N', n, n, nrhs,
                         Adg.data(), Adg.ld(), Xdg.data(), Xdg.ld());
    check(info == 0, "LAPACKE_dgels");

    /* --- Compare both paths to the exact X and to each other --- */
    double fwd_qr = rel_diff(Xqr, X);
    double fwd_dg = rel_diff(Xdg, X);
    double agree  = rel_diff(Xqr, Xdg);

    Matrix R = B;                                 /* R <- A Xqr - B */
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, nrhs, n,
                1.0, A.data(), A.ld(), Xqr.data(), Xqr.ld(), -1.0, R.data(), R.ld());
    double res_qr = norm1(R) / std::max(norm1(B), 1e-300);

    const double rtol = 100.0 * n * eps;
    bool ok = (fwd_qr <= rtol && fwd_dg <= rtol && agree <= rtol && res_qr <= rtol);
    std::printf("  n=%-4d nrhs=%d | manual fwd %.2e  dgels fwd %.2e  "
                "paths agree %.2e  manual res %.2e  (rtol %.2e) %s\n",
                n, nrhs, fwd_qr, fwd_dg, agree, res_qr, rtol, ok ? "OK" : "FAIL");
    check(ok, "QR solve accuracy within rtol");
}

} /* anonymous namespace */

int main()
{
    std::srand(42);
    std::printf("Dense QR solve example: dgeqrf -> dormqr -> dtrsm  vs  LAPACKE_dgels\n");

    solve(32,  5);
    solve(64,  4);
    solve(128, 3);
    solve(256, 1);

    std::printf("\nall checks passed\n");
    return 0;
}
