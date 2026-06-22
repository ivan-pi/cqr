/* armpl_stub/armpl.h
 *
 * *** STUB — FOR DRIVER LOGIC TESTING ONLY. NOT ARM PL. ***
 *
 * Minimal functional fakes of the ArmPL interleave-batch routines used by
 * xcheck_armpl_ormqr.cpp, so the cross-check driver can be compiled and
 * its logic exercised on any platform without libarmpl. The fakes are
 * naive scalar implementations:
 *
 *   - armpl_dgeqrfrr_interleave_batch: column-pivoted Householder QR
 *     (greedy max-norm pivoting, LAPACK reflector convention, 0-based
 *     jpvt), so the back-permutation path is exercised nontrivially.
 *   - armpl_dormqr_interleave_batch:   scalar dorm2r per matrix
 *   - armpl_dtrsm_interleave_batch:    scalar upper trsm per matrix
 *   - armpl_dge_deinterleave:          strided gather of one matrix
 *
 * Addressing convention (matches ArmPL docs and the RBF-FD solver):
 *   element (i,j) of matrix v, batch b:
 *     p[ b*bstrd + v + i*istrd + j*jstrd ]
 *
 * When building against real ArmPL, do NOT put this directory on the
 * include path.
 */

#ifndef ARMPL_STUB_H
#define ARMPL_STUB_H

#include <cmath>
#include <cstddef>
#include <vector>

typedef int armpl_int_t;

typedef enum {
    ARMPL_STATUS_SUCCESS = 0,
    ARMPL_STATUS_INPUT_PARAMETER_ERROR = 1,
    ARMPL_STATUS_EXECUTION_FAILURE = 2
} armpl_status_t;

/* ------------------------------------------------------------------ */
/* internal helpers (deliberately simple)                              */
/* ------------------------------------------------------------------ */

namespace armpl_stub_detail {

inline size_t at(armpl_int_t v, int i, int j,
                 armpl_int_t istrd, armpl_int_t jstrd)
{
    return (size_t)v + (size_t)i * istrd + (size_t)j * jstrd;
}

/* gather matrix v into column-major buffer */
inline void gather(armpl_int_t ninter, armpl_int_t v, int m, int n,
                   const double *p, armpl_int_t istrd, armpl_int_t jstrd,
                   double *M, int ldm)
{
    (void)ninter;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            M[i + (size_t)j * ldm] = p[at(v, i, j, istrd, jstrd)];
}

/* scatter column-major buffer back into matrix v */
inline void scatter(armpl_int_t ninter, armpl_int_t v, int m, int n,
                    const double *M, int ldm,
                    double *p, armpl_int_t istrd, armpl_int_t jstrd)
{
    (void)ninter;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            p[at(v, i, j, istrd, jstrd)] = M[i + (size_t)j * ldm];
}

/* column-pivoted Householder QR (dgeqp3-lite, greedy max column norm);
 * jpvt[j] = ORIGINAL (0-based) index of the column placed at position j */
inline void geqp3_lite(int m, int n, double *A, int lda,
                       armpl_int_t *jpvt, double *tau, armpl_int_t *rank)
{
    const int k = (m < n) ? m : n;
    for (int j = 0; j < n; ++j) jpvt[j] = j;

    for (int kk = 0; kk < k; ++kk) {
        /* pivot: column with largest trailing norm */
        int piv = kk;
        double best = -1.0;
        for (int j = kk; j < n; ++j) {
            double s = 0;
            for (int i = kk; i < m; ++i) s += A[i + (size_t)j * lda] * A[i + (size_t)j * lda];
            if (s > best) { best = s; piv = j; }
        }
        if (piv != kk) {
            for (int i = 0; i < m; ++i) {
                double t = A[i + (size_t)kk * lda];
                A[i + (size_t)kk * lda] = A[i + (size_t)piv * lda];
                A[i + (size_t)piv * lda] = t;
            }
            armpl_int_t t = jpvt[kk]; jpvt[kk] = jpvt[piv]; jpvt[piv] = t;
        }

        /* dlarfg */
        double alpha = A[kk + (size_t)kk * lda];
        double xnorm = 0;
        for (int i = kk + 1; i < m; ++i) xnorm = std::hypot(xnorm, A[i + (size_t)kk * lda]);
        if (xnorm == 0.0) { tau[kk] = 0.0; continue; }
        double beta = -std::copysign(std::hypot(alpha, xnorm), alpha);
        tau[kk] = (beta - alpha) / beta;
        double scal = 1.0 / (alpha - beta);
        for (int i = kk + 1; i < m; ++i) A[i + (size_t)kk * lda] *= scal;
        A[kk + (size_t)kk * lda] = beta;

        /* apply H(kk) to trailing columns */
        for (int j = kk + 1; j < n; ++j) {
            double w = A[kk + (size_t)j * lda];
            for (int i = kk + 1; i < m; ++i)
                w += A[i + (size_t)kk * lda] * A[i + (size_t)j * lda];
            A[kk + (size_t)j * lda] -= tau[kk] * w;
            for (int i = kk + 1; i < m; ++i)
                A[i + (size_t)j * lda] -= tau[kk] * A[i + (size_t)kk * lda] * w;
        }
    }
    *rank = k; /* full rank assumed for the stub */
}

/* scalar dorm2r, side='L' */
inline void orm2r(char trans, int m, int nrhs, int k,
                  const double *A, int lda, const double *tau,
                  double *B, int ldb)
{
    const bool fwd = (trans == 'T' || trans == 't');
    for (int s = 0; s < k; ++s) {
        int kk = fwd ? s : k - 1 - s;
        for (int j = 0; j < nrhs; ++j) {
            double w = B[kk + (size_t)j * ldb];
            for (int i = kk + 1; i < m; ++i)
                w += A[i + (size_t)kk * lda] * B[i + (size_t)j * ldb];
            B[kk + (size_t)j * ldb] -= tau[kk] * w;
            for (int i = kk + 1; i < m; ++i)
                B[i + (size_t)j * ldb] -= tau[kk] * A[i + (size_t)kk * lda] * w;
        }
    }
}

} /* namespace armpl_stub_detail */

/* ------------------------------------------------------------------ */
/* stubbed ArmPL entry points (signatures shaped after the RBF solver) */
/* ------------------------------------------------------------------ */

static inline armpl_status_t armpl_dgeqrfrr_interleave_batch(
    armpl_int_t ninter, armpl_int_t nbatch, armpl_int_t m, armpl_int_t n,
    double *a_p, armpl_int_t bstrd_a, armpl_int_t istrd_a, armpl_int_t jstrd_a,
    armpl_int_t *jpvt_p, armpl_int_t bstrd_jpvt, armpl_int_t istrd_jpvt,
    double *tau_p, armpl_int_t bstrd_tau, armpl_int_t istrd_tau,
    armpl_int_t *rank_p)
{
    using namespace armpl_stub_detail;
    const int k = (m < n) ? m : n;
    std::vector<double> M((size_t)m * n), tau(k);
    std::vector<armpl_int_t> pv(n);
    for (armpl_int_t b = 0; b < nbatch; ++b)
        for (armpl_int_t v = 0; v < ninter; ++v) {
            gather(ninter, v, m, n, a_p + (size_t)b * bstrd_a, istrd_a, jstrd_a, M.data(), m);
            geqp3_lite(m, n, M.data(), m, pv.data(), tau.data(), &rank_p[b * ninter + v]);
            scatter(ninter, v, m, n, M.data(), m, a_p + (size_t)b * bstrd_a, istrd_a, jstrd_a);
            for (int kk = 0; kk < k; ++kk)
                tau_p[(size_t)b * bstrd_tau + v + (size_t)kk * istrd_tau] = tau[kk];
            for (int j = 0; j < n; ++j)
                jpvt_p[(size_t)b * bstrd_jpvt + v + (size_t)j * istrd_jpvt] = pv[j];
        }
    return ARMPL_STATUS_SUCCESS;
}

static inline armpl_status_t armpl_dormqr_interleave_batch(
    armpl_int_t ninter, armpl_int_t nbatch, char side, char trans,
    armpl_int_t m, armpl_int_t nrhs, const armpl_int_t *nk_p,
    const double *a_p, armpl_int_t bstrd_a, armpl_int_t istrd_a, armpl_int_t jstrd_a,
    const double *tau_p, armpl_int_t bstrd_tau, armpl_int_t istrd_tau,
    double *b_p, armpl_int_t bstrd_b, armpl_int_t istrd_b, armpl_int_t jstrd_b)
{
    using namespace armpl_stub_detail;
    if (side != 'L' && side != 'l') return ARMPL_STATUS_INPUT_PARAMETER_ERROR;
    std::vector<double> M((size_t)m * m), B((size_t)m * nrhs), tau(m);
    for (armpl_int_t b = 0; b < nbatch; ++b)
        for (armpl_int_t v = 0; v < ninter; ++v) {
            const int k = (int)nk_p[b * ninter + v];
            gather(ninter, v, m, k, a_p + (size_t)b * bstrd_a, istrd_a, jstrd_a, M.data(), m);
            gather(ninter, v, m, nrhs, b_p + (size_t)b * bstrd_b, istrd_b, jstrd_b, B.data(), m);
            for (int kk = 0; kk < k; ++kk)
                tau[kk] = tau_p[(size_t)b * bstrd_tau + v + (size_t)kk * istrd_tau];
            orm2r(trans, m, nrhs, k, M.data(), m, tau.data(), B.data(), m);
            scatter(ninter, v, m, nrhs, B.data(), m, b_p + (size_t)b * bstrd_b, istrd_b, jstrd_b);
        }
    return ARMPL_STATUS_SUCCESS;
}

static inline armpl_status_t armpl_dtrsm_interleave_batch(
    armpl_int_t ninter, armpl_int_t nbatch, char side, char uplo,
    char transa, char diag, armpl_int_t n, armpl_int_t nrhs, double alpha,
    const double *a_p, armpl_int_t bstrd_a, armpl_int_t istrd_a, armpl_int_t jstrd_a,
    double *b_p, armpl_int_t bstrd_b, armpl_int_t istrd_b, armpl_int_t jstrd_b)
{
    using namespace armpl_stub_detail;
    if (side != 'L' || uplo != 'U' || transa != 'N' || diag != 'N')
        return ARMPL_STATUS_INPUT_PARAMETER_ERROR; /* only the solver's case */
    std::vector<double> R((size_t)n * n), B((size_t)n * nrhs);
    for (armpl_int_t b = 0; b < nbatch; ++b)
        for (armpl_int_t v = 0; v < ninter; ++v) {
            gather(ninter, v, n, n, a_p + (size_t)b * bstrd_a, istrd_a, jstrd_a, R.data(), n);
            gather(ninter, v, n, nrhs, b_p + (size_t)b * bstrd_b, istrd_b, jstrd_b, B.data(), n);
            for (int j = 0; j < nrhs; ++j)
                for (int i = n - 1; i >= 0; --i) {
                    double s = alpha * B[i + (size_t)j * n];
                    for (int l = i + 1; l < n; ++l)
                        s -= R[i + (size_t)l * n] * B[l + (size_t)j * n];
                    B[i + (size_t)j * n] = s / R[i + (size_t)i * n];
                }
            scatter(ninter, v, n, nrhs, B.data(), n, b_p + (size_t)b * bstrd_b, istrd_b, jstrd_b);
        }
    return ARMPL_STATUS_SUCCESS;
}

static inline armpl_status_t armpl_dge_deinterleave(
    armpl_int_t ninter, armpl_int_t v, armpl_int_t m, armpl_int_t n,
    double *B, armpl_int_t incb, armpl_int_t ldb,
    const double *a_p, armpl_int_t istrd, armpl_int_t jstrd)
{
    using namespace armpl_stub_detail;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            B[(size_t)i * incb + (size_t)j * ldb] = a_p[at(v, i, j, istrd, jstrd)];
    (void)ninter;
    return ARMPL_STATUS_SUCCESS;
}

#endif /* ARMPL_STUB_H */
