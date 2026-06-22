/* xcheck_armpl_ormqr.cpp
 *
 * Cross-check driver: Claude's compact ormqr (dormqr_compact) versus the
 * Arm Performance Libraries interleave-batch routines.
 *
 * Key fact this driver relies on: with the stride choices used in the
 * RBF-FD solver (istrd = ninter, jstrd = ninter*m, single nbatch group),
 * the ArmPL interleave-batch layout is byte-identical to the compact
 * layout assumed by dormqr_compact:
 *
 *     element (i,j) of matrix v  ->  p[(j*m + i)*ninter + v]
 *
 * i.e. V = ninter, ldap = m, nm = ninter (one group). Therefore
 * dormqr_compact can consume the factorization produced by
 * armpl_dgeqrfrr_interleave_batch (reflectors in A_p, tau_p) directly,
 * and the two ormqr implementations can be compared on the SAME
 * reflectors -- isolating exactly the routine under test.
 *
 * Pipeline mirrored from batch_rbf_fd_d_template (batch_rbf_armpl.cpp):
 *
 *   1. assemble interleaved A_p, B_p
 *   2. armpl_dgeqrfrr_interleave_batch        (rank-revealing QR, AP = QR)
 *   3. ormqr 'L','T'   <-- compared: ArmPL vs dormqr_compact
 *   4. armpl_dtrsm_interleave_batch           (R y = Q^T b)
 *   5. deinterleave + jpvt back-permutation   (x = P y)
 *
 * Checks per ninter:
 *   [layout]  manual interleave round-trips through armpl_dge_deinterleave
 *   [ormqr T] ArmPL vs compact Q^T B, elementwise on interleaved data
 *   [ormqr N] both implementations recover the original B from Q^T B
 *   [solve]   both pipelines recover X(:,j) = j+1 (B = A*X by row sums);
 *             ArmPL-X vs compact-X compared directly as well
 *
 * Test design: X(:,j) = j+1 (ones, twos, threes, ...), B = A*X, so the
 * b columns are scaled row sums of A.
 *
 * Build (GNU + ArmPL):
 *   g++ -O3 -mcpu=native -std=c++17 xcheck_armpl_ormqr.cpp \
 *       ormqr_compact_dispatch.cpp \
 *       -I${ARMPL_DIR}/include -L${ARMPL_DIR}/lib -larmpl_lp64 -lm
 *
 * Build (Arm Compiler for Linux):
 *   armclang++ -O3 -mcpu=native -std=c++17 -armpl \
 *       xcheck_armpl_ormqr.cpp ormqr_compact_dispatch.cpp
 *
 * Build (macOS, ArmPL for macOS + Apple Clang):
 *   clang++ -O3 -mcpu=apple-m1 -std=c++17 xcheck_armpl_ormqr.cpp \
 *       ormqr_compact_dispatch.cpp \
 *       -I${ARMPL_DIR}/include -L${ARMPL_DIR}/lib -larmpl -lm
 *
 * Logic-only test without ArmPL (stub headers, any platform):
 *   g++ -O3 -std=c++17 -Iarmpl_stub xcheck_armpl_ormqr.cpp \
 *       ormqr_compact_dispatch.cpp -lm
 *
 * Usage:  ./xcheck [np] [nq]      (defaults: np = 43, nq = 5)
 */

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

#include <armpl.h>
#include <armpl_interleave_batch.h>

#include "ormqr_compact.h"

/* ------------------------------------------------------------------ */

static void armpl_check_procedure(armpl_status_t result, const char *file, int line)
{
    if (result != ARMPL_STATUS_SUCCESS) {
        std::fprintf(stderr, "[%s:%d] ArmPL failed with status: %d\n",
                     file, line, static_cast<int>(result));
        std::exit(EXIT_FAILURE);
    }
}
#define ARMPL_CHECK(stmt) armpl_check_procedure((stmt), __FILE__, __LINE__)

static double frand() { return 2.0 * rand() / (double)RAND_MAX - 1.0; }

/* manual interleave with the solver's stride convention:
 * p[(j*rows + i)*ninter + v] = M_v(i,j) */
static void interleave(int ninter, int rows, int cols,
                       const std::vector<std::vector<double>> &M,
                       std::vector<double> &p)
{
    for (int v = 0; v < ninter; ++v)
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i)
                p[((size_t)j * rows + i) * ninter + v] = M[v][i + (size_t)j * rows];
}

static double max_abs_diff(const double *a, const double *b, size_t n)
{
    double d = 0;
    for (size_t i = 0; i < n; ++i) d = std::max(d, std::abs(a[i] - b[i]));
    return d;
}

static double max_abs(const double *a, size_t n)
{
    double d = 0;
    for (size_t i = 0; i < n; ++i) d = std::max(d, std::abs(a[i]));
    return d;
}

/* ------------------------------------------------------------------ */

static int run_one(int ninter, int np, int nq)
{
    const int m = np, n = np, nrhs = nq;
    const int nbatch = 1;
    int fails = 0;

    /* strides exactly as in batch_rbf_fd_d_template */
    const armpl_int_t istrd_A = ninter, jstrd_A = (armpl_int_t)ninter * m;
    const armpl_int_t bstrd_A = jstrd_A * n;
    const armpl_int_t istrd_jpvt = ninter, bstrd_jpvt = (armpl_int_t)ninter * m;
    const armpl_int_t istrd_tau = ninter,  bstrd_tau  = (armpl_int_t)ninter * m;
    const armpl_int_t istrd_B = ninter, jstrd_B = (armpl_int_t)ninter * m;
    const armpl_int_t bstrd_B = jstrd_B * nrhs;

    /* ---- generate systems: B = A*X with X(:,j) = j+1 ---- */
    std::vector<double> X((size_t)m * nrhs);
    for (int j = 0; j < nrhs; ++j)
        for (int i = 0; i < m; ++i)
            X[i + (size_t)j * m] = (double)(j + 1);

    std::vector<std::vector<double>> A(ninter), B(ninter);
    for (int v = 0; v < ninter; ++v) {
        A[v].resize((size_t)m * n);
        B[v].resize((size_t)m * nrhs);
        for (auto &x : A[v]) x = frand();
        for (int i = 0; i < m; ++i) A[v][i + (size_t)i * m] += 2.0; /* tame cond */
        for (int j = 0; j < nrhs; ++j)
            for (int i = 0; i < m; ++i) {
                double s = 0;
                for (int l = 0; l < m; ++l)
                    s += A[v][i + (size_t)l * m] * X[l + (size_t)j * m];
                B[v][i + (size_t)j * m] = s;
            }
    }

    /* ---- interleave ---- */
    std::vector<double> A_p((size_t)ninter * m * n);
    std::vector<double> B_orig((size_t)ninter * m * nrhs);
    interleave(ninter, m, n, A, A_p);
    interleave(ninter, m, nrhs, B, B_orig);

    std::vector<double> tau((size_t)ninter * m);
    std::vector<armpl_int_t> jpvt((size_t)ninter * m);
    std::vector<armpl_int_t> rank(ninter);
    std::vector<armpl_int_t> nk((size_t)ninter * nbatch, n);

    /* ---- [layout] our interleave convention == ArmPL's ---- */
    {
        std::vector<double> tmp((size_t)m * nrhs);
        ARMPL_CHECK(armpl_dge_deinterleave(ninter, 0, m, nrhs,
                                           tmp.data(), 1, m,
                                           B_orig.data(), istrd_B, jstrd_B));
        double e = max_abs_diff(tmp.data(), B[0].data(), (size_t)m * nrhs);
        bool ok = (e == 0.0);
        fails += !ok;
        std::printf("  [layout ] manual interleave vs armpl_dge_deinterleave: %.2e %s\n",
                    e, ok ? "OK" : "FAIL");
    }

    /* ---- factorize once (shared by both ormqr paths) ---- */
    ARMPL_CHECK(armpl_dgeqrfrr_interleave_batch(ninter, nbatch, m, n,
        A_p.data(), bstrd_A, istrd_A, jstrd_A,
        jpvt.data(), bstrd_jpvt, istrd_jpvt,
        tau.data(), bstrd_tau, istrd_tau, rank.data()));

    for (int v = 0; v < ninter; ++v) {
        if (rank[v] != n) {
            std::printf("  [rank   ] matrix %d: rank %d < n=%d FAIL\n",
                        v, (int)rank[v], n);
            ++fails;
        }
    }

    /* ---- [ormqr T] ArmPL vs compact on identical reflectors ---- */
    std::vector<double> B_armpl = B_orig, B_ours = B_orig;

    ARMPL_CHECK(armpl_dormqr_interleave_batch(ninter, nbatch, 'L', 'T', m, nrhs,
        nk.data(),
        A_p.data(), bstrd_A, istrd_A, jstrd_A,
        tau.data(), bstrd_tau, istrd_tau,
        B_armpl.data(), bstrd_B, istrd_B, jstrd_B));

    /* same operation, our kernel: V = ninter, one group, ldap = m */
    dormqr_compact('T', m, nrhs, n,
                   A_p.data(), m, n, tau.data(),
                   B_ours.data(), m, ninter, ninter);

    const double bscale = max_abs(B_armpl.data(), B_armpl.size());
    {
        double e = max_abs_diff(B_armpl.data(), B_ours.data(), B_armpl.size());
        bool ok = (e <= 1e-12 * bscale);
        fails += !ok;
        std::printf("  [ormqr T] ArmPL vs compact, max|dQtB|/|QtB|: %.2e %s\n",
                    e / bscale, ok ? "OK" : "FAIL");
    }

    /* ---- [ormqr N] both recover original B from Q^T B ---- */
    {
        std::vector<double> r1 = B_armpl, r2 = B_ours;
        ARMPL_CHECK(armpl_dormqr_interleave_batch(ninter, nbatch, 'L', 'N', m, nrhs,
            nk.data(),
            A_p.data(), bstrd_A, istrd_A, jstrd_A,
            tau.data(), bstrd_tau, istrd_tau,
            r1.data(), bstrd_B, istrd_B, jstrd_B));
        dormqr_compact('N', m, nrhs, n,
                       A_p.data(), m, n, tau.data(),
                       r2.data(), m, ninter, ninter);
        double e1 = max_abs_diff(r1.data(), B_orig.data(), r1.size());
        double e2 = max_abs_diff(r2.data(), B_orig.data(), r2.size());
        bool ok1 = (e1 <= 1e-11 * bscale), ok2 = (e2 <= 1e-11 * bscale);
        fails += !ok1 + !ok2;
        std::printf("  [ormqr N] QQt roundtrip, ArmPL: %.2e %s | compact: %.2e %s\n",
                    e1 / bscale, ok1 ? "OK" : "FAIL",
                    e2 / bscale, ok2 ? "OK" : "FAIL");
    }

    /* ---- finish both pipelines with the SAME ArmPL trsm ---- */
    const double alpha = 1.0;
    ARMPL_CHECK(armpl_dtrsm_interleave_batch(ninter, nbatch, 'L', 'U', 'N', 'N',
        n, nrhs, alpha,
        A_p.data(), bstrd_A, istrd_A, jstrd_A,
        B_armpl.data(), bstrd_B, istrd_B, jstrd_B));
    ARMPL_CHECK(armpl_dtrsm_interleave_batch(ninter, nbatch, 'L', 'U', 'N', 'N',
        n, nrhs, alpha,
        A_p.data(), bstrd_A, istrd_A, jstrd_A,
        B_ours.data(), bstrd_B, istrd_B, jstrd_B));

    /* ---- deinterleave + jpvt back-permutation (as in the solver) ---- */
    auto extract_X = [&](const std::vector<double> &Bp, int v,
                         std::vector<double> &Xout) {
        std::vector<double> col_B((size_t)m * nrhs);
        ARMPL_CHECK(armpl_dge_deinterleave(ninter, v, m, nrhs,
                                           col_B.data(), 1, m,
                                           Bp.data(), istrd_B, jstrd_B));
        for (int rhs = 0; rhs < nrhs; ++rhs)
            for (int j = 0; j < m; ++j) {
                int k = (int)jpvt[(size_t)j * ninter + v];
                Xout[k + (size_t)rhs * m] = col_B[j + (size_t)rhs * m];
            }
    };

    double eXa = 0, eXo = 0, eXd = 0;
    std::vector<double> Xa((size_t)m * nrhs), Xo((size_t)m * nrhs);
    for (int v = 0; v < ninter; ++v) {
        extract_X(B_armpl, v, Xa);
        extract_X(B_ours, v, Xo);
        eXa = std::max(eXa, max_abs_diff(Xa.data(), X.data(), Xa.size()));
        eXo = std::max(eXo, max_abs_diff(Xo.data(), X.data(), Xo.size()));
        eXd = std::max(eXd, max_abs_diff(Xa.data(), Xo.data(), Xa.size()));
    }
    {
        const double xscale = (double)nrhs; /* max entry of exact X */
        bool oka = (eXa <= 1e-9 * xscale * m);
        bool oko = (eXo <= 1e-9 * xscale * m);
        bool okd = (eXd <= 1e-11 * xscale * m);
        fails += !oka + !oko + !okd;
        std::printf("  [solve  ] X err, ArmPL path: %.2e %s | compact path: %.2e %s"
                    " | path diff: %.2e %s\n",
                    eXa, oka ? "OK" : "FAIL", eXo, oko ? "OK" : "FAIL",
                    eXd, okd ? "OK" : "FAIL");
    }

    return fails;
}

/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    const int np = (argc > 1) ? std::atoi(argv[1]) : 43; /* nz=28 + npoly=15 */
    const int nq = (argc > 2) ? std::atoi(argv[2]) : 5;

    srand(42);
    int fails = 0;

    /* dormqr_compact dispatches V in {2,4,8,16}; ninter = 6 would need
     * either padding the batch to 8 or the non-power-of-two fallback
     * pack type (GNU vector_size requires power-of-two widths). */
    const int ninters[] = {2, 4, 8, 16};

    for (int ni : ninters) {
        std::printf("ninter = %-2d  np = %d  nq = %d\n", ni, np, nq);
        fails += run_one(ni, np, nq);
    }

    if (fails) { std::printf("\n%d CHECK(S) FAILED\n", fails); return 1; }
    std::printf("\nall checks passed\n");
    return 0;
}
