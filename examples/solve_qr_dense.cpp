/* solve_qr_dense.cpp
 *
 * Worked example: solving a square linear system AX = B with a QR
 * factorization, spelled out as the explicit LAPACK three-step sequence and
 * cross-checked against the one-call LAPACKE_dgels driver.
 *
 * This is the *dense, single-matrix* analogue of the Compact-format batch
 * pipeline this repository extends (design doc section 7.2). It exists to
 * document, in ordinary LAPACK, the exact sequence the compact solver mirrors:
 *
 *     dense LAPACK            compact pipeline (one matrix per SIMD lane)
 *     -----------            -------------------------------------------
 *     dgeqrf   A -> (H,R,tau)   mkl_dgeqrf_compact
 *     dormqr   B <- Q^T B       ext_mkl_dormqr_compact      <-- the gap filled
 *     dtrsm    R X = (Q^T B)    mkl_dtrsm_compact
 *
 * For a square (m == n), full-rank A, solving AX = B by QR proceeds as:
 *
 *   1. A = Q R                        (dgeqrf: R in the upper triangle of A,
 *                                      Q as Householder reflectors H + tau)
 *   2. multiply both sides by Q^T :   R X = Q^T B
 *                                      (dormqr applies Q^T to B in place)
 *   3. back-substitute :              X = R^{-1} (Q^T B)
 *                                      (dtrsm, R upper-triangular)
 *
 * The "naive" forward path LAPACKE_dgels does all three internally (it reduces
 * to exactly this QR solve when m == n, trans = 'N'). We run both and report
 * that they agree with each other and with the known exact solution.
 *
 * Build: needs LAPACKE + a BLAS (here Intel MKL, the project's BLAS); wired up
 * by CMakeLists.txt as the `solve_qr_dense` target.
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

/* L1 (max column sum) norm of a column-major m x n matrix */
double norm1(const double *M, int m, int n)
{
    double mx = 0;
    for (int j = 0; j < n; ++j) {
        double s = 0;
        for (int i = 0; i < m; ++i) s += std::abs(M[i + (size_t)j * m]);
        mx = std::max(mx, s);
    }
    return mx;
}

double maxdiff(const double *a, const double *b, size_t n)
{
    double d = 0;
    for (size_t i = 0; i < n; ++i) d = std::max(d, std::abs(a[i] - b[i]));
    return d;
}

/* AX into out, all column-major n x nrhs / n x n */
void gemm_AX(const double *A, const double *X, double *out, int n, int nrhs)
{
    for (int j = 0; j < nrhs; ++j)
        for (int i = 0; i < n; ++i) {
            double s = 0;
            for (int l = 0; l < n; ++l) s += A[i + (size_t)l * n] * X[l + (size_t)j * n];
            out[i + (size_t)j * n] = s;
        }
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
    gemm_AX(A.data(), X.data(), B.data(), n, nrhs);

    /* ================================================================== *
     * Path 1: explicit QR solve  dgeqrf -> dormqr -> dtrsm               *
     * ================================================================== */
    std::vector<double> Aqr(A), Bqr(B), tau(n);

    /* 1. A = Q R. On return the upper triangle of Aqr holds R; the lower
     *    triangle + tau encode the Householder reflectors defining Q.      */
    lapack_int info = LAPACKE_dgeqrf(LAPACK_COL_MAJOR, n, n, Aqr.data(), n, tau.data());
    if (info != 0) { std::printf("  dgeqrf info=%d\n", (int)info); return 1; }

    /* 2. Bqr <- Q^T B.  side='L', trans='T', k=n reflectors.
     *    (This is the step the compact API was missing: ext_mkl_dormqr_compact.) */
    info = LAPACKE_dormqr(LAPACK_COL_MAJOR, 'L', 'T', n, nrhs, n,
                          Aqr.data(), n, tau.data(), Bqr.data(), n);
    if (info != 0) { std::printf("  dormqr info=%d\n", (int)info); return 1; }

    /* 3. R X = (Q^T B), R upper-triangular non-unit. cblas_dtrsm solves in
     *    place: Bqr <- R^{-1} Bqr = Xhat. */
    cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                n, nrhs, 1.0, Aqr.data(), n, Bqr.data(), n);
    /* Bqr now holds Xhat from the manual path. */

    /* ================================================================== *
     * Path 2: naive forward driver  LAPACKE_dgels (QR solve when m == n) *
     * ================================================================== */
    std::vector<double> Adg(A), Bdg(B);
    info = LAPACKE_dgels(LAPACK_COL_MAJOR, 'N', n, n, nrhs, Adg.data(), n, Bdg.data(), n);
    if (info != 0) { std::printf("  dgels info=%d\n", (int)info); return 1; }
    /* Bdg now holds Xhat from dgels (first n rows; m == n here). */

    /* ================================================================== *
     * Compare: each path vs the exact X, and the two paths vs each other *
     * ================================================================== */
    const double xnorm = std::max(norm1(X.data(), n, nrhs), 1e-300);
    const double bnorm = std::max(norm1(B.data(), n, nrhs), 1e-300);

    double fwd_qr   = maxdiff(Bqr.data(), X.data(), sB) / xnorm;
    double fwd_dg   = maxdiff(Bdg.data(), X.data(), sB) / xnorm;
    double agree    = maxdiff(Bqr.data(), Bdg.data(), sB) / xnorm;

    /* system residual A Xhat - B for the manual path */
    std::vector<double> AX(sB);
    gemm_AX(A.data(), Bqr.data(), AX.data(), n, nrhs);
    double res_qr = maxdiff(AX.data(), B.data(), sB) / bnorm;

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
