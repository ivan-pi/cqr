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
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>

namespace {

const double eps = std::numeric_limits<double>::epsilon();

double frand() { return 2.0 * std::rand() / (double)RAND_MAX - 1.0; }

/* Relative 1-norm difference ||A - B||_1 / ||B||_1, A and B column-major m x n. */
double rel_diff(const double *A, const double *B, int m, int n)
{
    std::vector<double> D(A, A + (size_t)m * n);
    cblas_daxpy((size_t)m * n, -1.0, B, 1, D.data(), 1);          /* D <- A - B */
    double num = LAPACKE_dlange(LAPACK_COL_MAJOR, '1', m, n, D.data(), m);
    double den = LAPACKE_dlange(LAPACK_COL_MAJOR, '1', m, n, B, m);
    return num / std::max(den, 1e-300);
}

int solve(int n, int nrhs)
{
    /* Known exact solution X(:,j) = j+1, so B = A X and the solve must
     * recover X (mirrors the compact suite-2 construction). */
    const size_t sA = (size_t)n * n, sB = (size_t)n * nrhs;
    std::vector<double> A(sA), X(sB), B(sB);
    for (int j = 0; j < nrhs; ++j)
        for (int i = 0; i < n; ++i) X[i + (size_t)j * n] = double(j + 1);
    for (size_t i = 0; i < sA; ++i) A[i] = frand();
    for (int i = 0; i < n; ++i) A[i + (size_t)i * n] += 2.0;  /* tame conditioning */
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, nrhs, n,
                1.0, A.data(), n, X.data(), n, 0.0, B.data(), n);   /* B = A X */

    /* --- Path 1: explicit QR solve  dgeqrf -> dormqr -> dtrsm --- */
    std::vector<double> Aqr(A), Xqr(B), tau(n);
    lapack_int info = LAPACKE_dgeqrf(LAPACK_COL_MAJOR, n, n, Aqr.data(), n, tau.data());
    if (!info)  /* Xqr <- Q^T B  (the step ext_mkl_dormqr_compact fills) */
        info = LAPACKE_dormqr(LAPACK_COL_MAJOR, 'L', 'T', n, nrhs, n,
                              Aqr.data(), n, tau.data(), Xqr.data(), n);
    if (info) { std::printf("  manual QR info=%d\n", (int)info); return 1; }
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                n, nrhs, 1.0, Aqr.data(), n, Xqr.data(), n);  /* Xqr <- R^{-1} Xqr */

    /* --- Path 2: naive forward driver  LAPACKE_dgels --- */
    std::vector<double> Adg(A), Xdg(B);
    info = LAPACKE_dgels(LAPACK_COL_MAJOR, 'N', n, n, nrhs, Adg.data(), n, Xdg.data(), n);
    if (info) { std::printf("  dgels info=%d\n", (int)info); return 1; }

    /* --- Compare both paths to the exact X and to each other --- */
    double fwd_qr = rel_diff(Xqr.data(), X.data(), n, nrhs);
    double fwd_dg = rel_diff(Xdg.data(), X.data(), n, nrhs);
    double agree  = rel_diff(Xqr.data(), Xdg.data(), n, nrhs);

    std::vector<double> R(B);                                  /* R <- A Xqr - B */
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, nrhs, n,
                1.0, A.data(), n, Xqr.data(), n, -1.0, R.data(), n);
    double res_qr = LAPACKE_dlange(LAPACK_COL_MAJOR, '1', n, nrhs, R.data(), n) /
                    std::max(LAPACKE_dlange(LAPACK_COL_MAJOR, '1', n, nrhs, B.data(), n), 1e-300);

    const double rtol = 100.0 * n * eps;
    bool ok = (fwd_qr <= rtol && fwd_dg <= rtol && agree <= rtol && res_qr <= rtol);
    std::printf("  n=%-4d nrhs=%d | manual fwd %.2e  dgels fwd %.2e  "
                "paths agree %.2e  manual res %.2e  (rtol %.2e) %s\n",
                n, nrhs, fwd_qr, fwd_dg, agree, res_qr, rtol, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

} /* anonymous namespace */

int main()
{
    std::srand(42);
    std::printf("Dense QR solve example: dgeqrf -> dormqr -> dtrsm  vs  LAPACKE_dgels\n");

    int fails = 0;
    fails += solve(32,  5);
    fails += solve(64,  4);
    fails += solve(128, 3);
    fails += solve(256, 1);

    if (fails) { std::printf("\n%d CHECK(S) FAILED\n", fails); return 1; }
    std::printf("\nall checks passed\n");
    return 0;
}
