/* test_cqr_ispc.cpp -- correctness for the ISPC compact-QR prototype.
 *
 * Per-kernel UNIT tests, each against an independent oracle so a failure isolates
 * to ONE routine, plus the end-to-end AX=B integration test:
 *
 *   geqrf  A = Q R reconstruction residual + orthogonality of Q (formed by LAPACK
 *          dorgqr from the ISPC reflectors) -- square and tall, padded group.
 *   ormqr  B := op(Q) B for BOTH Q (trans=0) and Q^T (trans=1) vs LAPACKE_dormqr
 *          on a LAPACK factorization (independent of the ISPC geqrf), plus a
 *          Q(Q^T B)=B round-trip -- square and tall.
 *   trsm   R X = alpha B vs mkl_dtrsm_compact and vs a known X, for alpha != 1 and
 *          nrhs = 1 and 6 (exercises the remainder and the 4-blocked paths).
 *   potrf  A = L L^T / U^T U vs mkl_dpotrf_compact and LAPACKE_dpotrf, both uplo
 *          (lower = contiguous path, upper = strided), reconstruction + untouched
 *          triangle, plus non-SPD lane isolation and an SPD solve (+ mkl trsm).
 *   solve  full ISPC pipeline vs known X, and the factor vs the GNU kernel (a few
 *          ULP; often bit-identical, but that is input/compiler-dependent).
 *
 * MKL/LAPACK supply only pack/unpack and the oracles; the ISPC kernels use no MKL.
 * Assisted-by: Claude:claude-opus-4.8
 */
#include <mkl.h>
#include <mkl_compact.h>

#include "cqr_ispc.h"
#include "cqr_geqrf_compact.hpp" /* geqrf_compact_general<T,V> (GNU reference) */
#include "cqr_compact.hpp"       /* ormqr_compact_general<T,V> (GNU reference) */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace {
using Mat = std::vector<double>; /* dense column-major */
using Batch = std::vector<Mat>;
const double EPS = std::numeric_limits<double>::epsilon();

std::mt19937_64 rng(12345);
double frand()
{
    static std::uniform_real_distribution<double> d(-1, 1);
    return d(rng);
}
int failures = 0;
void expect(bool c, const char *w)
{
    if (!c) {
        std::printf("  FAIL: %s\n", w);
        ++failures;
    }
}
double *acalloc(std::size_t n) /* 64B-aligned, zeroed (deterministic padding) */
{
    std::size_t b = ((n * 8 + 63) / 64) * 64;
    double *p = static_cast<double *>(std::aligned_alloc(64, b));
    std::memset(p, 0, b);
    return p;
}
Batch randbatch(int rows, int cols, int nm)
{
    Batch M(nm, Mat((std::size_t)rows * cols));
    for (auto &m : M)
        for (double &x : m)
            x = frand();
    return M;
}
/* pack nm dense (rows x cols, ld=rows) matrices into a fresh compact buffer. */
double *packc(MKL_COMPACT_PACK fmt, int rows, int cols, const Batch &M, int nm)
{
    double *c = acalloc(mkl_dget_size_compact(rows, cols, fmt, nm));
    std::vector<double *> p(nm);
    for (int v = 0; v < nm; ++v)
        p[v] = const_cast<double *>(M[v].data());
    mkl_dgepack_compact(MKL_COL_MAJOR, rows, cols, p.data(), rows, c, rows, fmt, nm);
    return c;
}
Batch unpackc(MKL_COMPACT_PACK fmt, int rows, int cols, double *c, int nm)
{
    Batch M(nm, Mat((std::size_t)rows * cols));
    std::vector<double *> p(nm);
    for (int v = 0; v < nm; ++v)
        p[v] = M[v].data();
    mkl_dgeunpack_compact(MKL_COL_MAJOR, rows, cols, p.data(), rows, c, rows, fmt, nm);
    return M;
}
double froben(int rows, int cols, const double *a)
{
    return LAPACKE_dlange(LAPACK_COL_MAJOR, 'F', rows, cols, a, rows);
}
double maxabs_diff(const Batch &X, const Batch &Y, int elems, int nm)
{
    double d = 0;
    for (int v = 0; v < nm; ++v)
        for (int i = 0; i < elems; ++i)
            d = std::max(d, std::fabs(X[v][i] - Y[v][i]));
    return d;
}

/* -------- unit: geqrf -- reconstruction + orthogonality (Q from ISPC factors) -- */
void unit_geqrf(MKL_COMPACT_PACK fmt, int nm, int m, int n)
{
    const int k = std::min(m, n);
    Batch A = randbatch(m, n, nm);
    double *apc = packc(fmt, m, n, A, nm);
    double *tauc = acalloc(mkl_dget_size_compact(k, 1, fmt, nm));
    cqr_ispc_dgeqrf_compact(m, n, apc, m, tauc, nm);
    Batch F = unpackc(fmt, m, n, apc, nm), TAU = unpackc(fmt, k, 1, tauc, nm);

    double res = 0, orth = 0;
    for (int v = 0; v < nm; ++v) {
        Mat R((std::size_t)k * n, 0.0); /* R (k x n), upper trapezoid, from F */
        for (int j = 0; j < n; ++j)
            for (int i = 0; i <= std::min(j, k - 1); ++i)
                R[i + (std::size_t)j * k] = F[v][i + (std::size_t)j * m];
        Mat Q = F[v]; /* dorgqr overwrites with Q (m x k) */
        LAPACKE_dorgqr(LAPACK_COL_MAJOR, m, k, k, Q.data(), m, TAU[v].data());
        Mat QtQ((std::size_t)k * k, 0.0); /* ||I - Q^T Q||_F */
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, k, k, m, 1.0, Q.data(), m,
                    Q.data(), m, 0.0, QtQ.data(), k);
        for (int i = 0; i < k; ++i)
            QtQ[i + (std::size_t)i * k] -= 1.0;
        orth = std::max(orth, froben(k, k, QtQ.data()));
        Mat QR((std::size_t)m * n, 0.0); /* ||A - Q R||_F / ||A||_F */
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, Q.data(), m,
                    R.data(), k, 0.0, QR.data(), m);
        for (std::size_t i = 0; i < (std::size_t)m * n; ++i)
            QR[i] = A[v][i] - QR[i];
        res = std::max(res, froben(m, n, QR.data()) /
                                std::max(froben(m, n, A[v].data()), 1e-300));
    }
    const double gate = 50.0 * n * EPS;
    expect(res <= gate, "geqrf: A = Q R residual");
    expect(orth <= gate, "geqrf: Q orthonormal");
    std::printf(
        "  [geqrf] m=%-3d n=%-3d nm=%-3d | residual %.2e  orth %.2e (gate %.1e)\n", m, n,
        nm, res, orth, gate);
    std::free(apc);
    std::free(tauc);
}

/* -------- unit: ormqr -- vs LAPACKE_dormqr on a LAPACK factorization ---------- */
void unit_ormqr(MKL_COMPACT_PACK fmt, int nm, int m, int n, int nrhs)
{
    const int k = std::min(m, n);
    Batch A = randbatch(m, n, nm), TAU(nm, Mat(k));
    for (int v = 0; v < nm; ++v) /* reflectors from LAPACK, not the ISPC geqrf */
        LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, A[v].data(), m, TAU[v].data());
    double *apc = packc(fmt, m, n, A, nm), *tauc = packc(fmt, k, 1, TAU, nm);

    for (int trans = 0; trans <= 1; ++trans) {
        Batch B = randbatch(m, nrhs, nm), Bref = B;
        double *bpc = packc(fmt, m, nrhs, B, nm);
        cqr_ispc_dormqr_compact(trans, m, nrhs, k, apc, m, tauc, bpc, m, nm);
        Batch Bispc = unpackc(fmt, m, nrhs, bpc, nm);
        double nrm = 0;
        for (int v = 0; v < nm; ++v) {
            LAPACKE_dormqr(LAPACK_COL_MAJOR, 'L', trans ? 'T' : 'N', m, nrhs, k,
                           A[v].data(), m, TAU[v].data(), Bref[v].data(), m);
            nrm = std::max(nrm, froben(m, nrhs, Bref[v].data()));
        }
        const double d = maxabs_diff(Bispc, Bref, m * nrhs, nm);
        const double gate = 100.0 * m * EPS * std::max(nrm, 1.0);
        expect(d <= gate, trans ? "ormqr: Q^T B vs LAPACK" : "ormqr: Q B vs LAPACK");
        std::printf("  [ormqr] m=%-3d nrhs=%d %s | max|diff| %.2e (gate %.1e)\n", m, nrhs,
                    trans ? "Q^T" : "Q  ", d, gate);
        std::free(bpc);
    }
    /* round-trip: Q (Q^T B) == B */
    Batch B = randbatch(m, nrhs, nm), B0 = B;
    double *bpc = packc(fmt, m, nrhs, B, nm);
    cqr_ispc_dormqr_compact(1, m, nrhs, k, apc, m, tauc, bpc, m, nm);
    cqr_ispc_dormqr_compact(0, m, nrhs, k, apc, m, tauc, bpc, m, nm);
    const double d = maxabs_diff(unpackc(fmt, m, nrhs, bpc, nm), B0, m * nrhs, nm);
    expect(d <= 100.0 * m * EPS, "ormqr: Q (Q^T B) = B round-trip");
    std::free(bpc);
    std::free(apc);
    std::free(tauc);
}

/* -------- unit: trsm -- vs mkl_dtrsm_compact and a known X -------------------- */
void unit_trsm(MKL_COMPACT_PACK fmt, int nm, int n, int nrhs, double alpha)
{
    Batch R(nm, Mat((std::size_t)n * n, 0.0)); /* upper-tri, diagonally boosted */
    for (int v = 0; v < nm; ++v) {
        for (int j = 0; j < n; ++j)
            for (int i = 0; i <= j; ++i)
                R[v][i + (std::size_t)j * n] = frand();
        for (int i = 0; i < n; ++i)
            R[v][i + (std::size_t)i * n] += 2.0 * n;
    }
    /* B = (1/alpha) R Xex so that solving R X = alpha B recovers Xex. */
    Batch Xex = randbatch(n, nrhs, nm), B(nm, Mat((std::size_t)n * nrhs));
    double maxX = 0;
    for (int v = 0; v < nm; ++v) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, nrhs, n, 1.0 / alpha,
                    R[v].data(), n, Xex[v].data(), n, 0.0, B[v].data(), n);
        for (double x : Xex[v])
            maxX = std::max(maxX, std::fabs(x));
    }
    double *apc = packc(fmt, n, n, R, nm);
    double *bpc = packc(fmt, n, nrhs, B, nm), *bmklc = packc(fmt, n, nrhs, B, nm);
    cqr_ispc_dtrsm_compact(n, nrhs, alpha, apc, n, bpc, n, nm);
    mkl_dtrsm_compact(MKL_COL_MAJOR, MKL_LEFT, MKL_UPPER, MKL_NOTRANS, MKL_NONUNIT, n,
                      nrhs, alpha, apc, n, bmklc, n, fmt, nm);
    Batch Xi = unpackc(fmt, n, nrhs, bpc, nm), Xm = unpackc(fmt, n, nrhs, bmklc, nm);
    const double dknown = maxabs_diff(Xi, Xex, n * nrhs, nm);
    const double dmkl = maxabs_diff(Xi, Xm, n * nrhs, nm);
    const double gate = 100.0 * n * EPS * std::max(1.0, maxX);
    expect(dknown <= gate, "trsm: known solution");
    expect(dmkl <= gate, "trsm: vs mkl_dtrsm_compact");
    std::printf(
        "  [trsm ] n=%-3d nrhs=%d alpha=%.1f | known %.2e  mkl %.2e (gate %.1e)\n", n,
        nrhs, alpha, dknown, dmkl, gate);
    std::free(apc);
    std::free(bpc);
    std::free(bmklc);
}

/* -------- integration: full solve vs known X, and bit-for-bit vs GNU ---------- */
void integration_solve(MKL_COMPACT_PACK fmt, int nm, int n, int nrhs, bool equiv)
{
    Mat X((std::size_t)n * nrhs); /* known X(i,j) = (j+1) + 0.5 i */
    for (int j = 0; j < nrhs; ++j)
        for (int i = 0; i < n; ++i)
            X[i + (std::size_t)j * n] = (j + 1) + 0.5 * i;
    Batch A(nm, Mat((std::size_t)n * n)), B(nm, Mat((std::size_t)n * nrhs));
    for (int v = 0; v < nm; ++v) {
        for (auto &x : A[v])
            x = frand();
        for (int i = 0; i < n; ++i)
            A[v][i + (std::size_t)i * n] += 2.0 * n;
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, nrhs, n, 1.0,
                    A[v].data(), n, X.data(), n, 0.0, B[v].data(), n);
    }
    double *ap0 = packc(fmt, n, n, A, nm), *bp0 = packc(fmt, n, nrhs, B, nm);
    const std::size_t szA = mkl_dget_size_compact(n, n, fmt, nm),
                      szT = mkl_dget_size_compact(n, 1, fmt, nm),
                      szB = mkl_dget_size_compact(n, nrhs, fmt, nm);
    double *api = acalloc(szA), *tpi = acalloc(szT), *bpi = acalloc(szB);
    std::memcpy(api, ap0, szA * 8);
    std::memcpy(bpi, bp0, szB * 8);
    cqr_ispc_dgeqrf_compact(n, n, api, n, tpi, nm);
    cqr_ispc_dormqr_compact(1, n, nrhs, n, api, n, tpi, bpi, n, nm);
    cqr_ispc_dtrsm_compact(n, nrhs, 1.0, api, n, bpi, n, nm);
    Batch Xi = unpackc(fmt, n, nrhs, bpi, nm);
    double fwd = 0;
    for (int v = 0; v < nm; ++v)
        for (int e = 0; e < n * nrhs; ++e)
            fwd = std::max(fwd, std::fabs(Xi[v][e] - X[e]));
    const double rtol = 100.0 * n * EPS * (1.0 + 0.5 * n);
    expect(fwd <= rtol, "solve: ISPC recovers known X");

    double ef = 0;
    if (equiv) { /* factor must match the GNU kernel to a few ULP (often identical,
                  * but exact bit-identity is input- and compiler-dependent) */
        double *apr = acalloc(szA), *tpr = acalloc(szT);
        std::memcpy(apr, ap0, szA * 8);
        cqr::detail::geqrf_compact_general<double, 8>(false, n, n, apr, n, tpr, nm);
        Batch Fi = unpackc(fmt, n, n, api, nm), Fr = unpackc(fmt, n, n, apr, nm);
        ef = maxabs_diff(Fi, Fr, n * n, nm);
        expect(ef <= 1e-11 * (1.0 + 2.0 * n), "solve: ISPC factor matches GNU geqrf");
        std::free(apr);
        std::free(tpr);
    }
    std::printf("  [solve] n=%-3d nm=%-3d nrhs=%d | fwd %.2e (rtol %.1e)%s", n, nm, nrhs,
                fwd, rtol, equiv ? "" : "\n");
    if (equiv) std::printf("  | vs GNU factor %.1e\n", ef);
    for (double *p : {ap0, bp0, api, tpi, bpi})
        std::free(p);
}

/* SPD batch A = M^T M + n I (symmetric positive-definite), dense column-major. */
Batch spdbatch(int n, int nm)
{
    Batch A(nm, Mat((std::size_t)n * n));
    for (int v = 0; v < nm; ++v) {
        Mat M = randbatch(n, n, 1)[0];
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                double s = 0;
                for (int l = 0; l < n; ++l)
                    s += M[l + (std::size_t)i * n] * M[l + (std::size_t)j * n];
                A[v][i + (std::size_t)j * n] = s + (i == j ? (double)n : 0.0);
            }
    }
    return A;
}

/* -------- unit: potrf -- Cholesky vs mkl_dpotrf_compact and LAPACKE_dpotrf ----- */
void unit_potrf(MKL_COMPACT_PACK fmt, int nm, int n, bool upper)
{
    const char ul = upper ? 'U' : 'L';
    Batch A = spdbatch(n, nm);
    double *apc = packc(fmt, n, n, A, nm); /* factored by the ISPC kernel */
    double *amc = packc(fmt, n, n, A, nm); /* factored by MKL (the oracle) */
    cqr_ispc_dpotrf_compact(0 /*col-major*/, upper ? 1 : 0, n, apc, n, nm);
    MKL_INT info = 0;
    mkl_dpotrf_compact(MKL_COL_MAJOR, upper ? MKL_UPPER : MKL_LOWER, n, amc, n, &info,
                       fmt, nm);
    Batch Fi = unpackc(fmt, n, n, apc, nm), Fm = unpackc(fmt, n, n, amc, nm);

    double res = 0, untouched = 0, elap = 0;
    for (int v = 0; v < nm; ++v) {
        /* reconstruction: the named triangle's factor times its transpose == A */
        Mat Rec((std::size_t)n * n, 0.0);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                double s = 0;
                const int lmax = std::min(i, j);
                for (int l = 0; l <= lmax; ++l)
                    s += upper ? Fi[v][l + (std::size_t)i * n] *
                                     Fi[v][l + (std::size_t)j * n]
                               : Fi[v][i + (std::size_t)l * n] *
                                     Fi[v][j + (std::size_t)l * n];
                Rec[i + (std::size_t)j * n] = s - A[v][i + (std::size_t)j * n];
            }
        res = std::max(res, froben(n, n, Rec.data()) /
                                std::max(froben(n, n, A[v].data()), 1e-300));
        /* the opposite triangle must pass through bit-for-bit; elementwise vs the
         * unique LAPACK SPD factor is a sharp per-element signal. */
        Mat L = A[v];
        LAPACKE_dpotrf(LAPACK_COL_MAJOR, ul, n, L.data(), n);
        double el = 0;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                const bool named = upper ? (i <= j) : (i >= j);
                const std::size_t e = i + (std::size_t)j * n;
                if (named)
                    el = std::max(el, std::fabs(Fi[v][e] - L[e]));
                else
                    untouched = std::max(untouched, std::fabs(Fi[v][e] - A[v][e]));
            }
        elap = std::max(elap, el / std::max(froben(n, n, L.data()), 1e-300));
    }
    const double dmkl = maxabs_diff(Fi, Fm, n * n, nm);
    const double gate = 50.0 * n * EPS;
    expect(res <= gate, "potrf: reconstruction A = R^T R");
    expect(untouched == 0.0, "potrf: opposite triangle untouched");
    expect(elap <= gate, "potrf: vs LAPACKE_dpotrf");
    expect(dmkl <= 1e-9, "potrf: vs mkl_dpotrf_compact");
    std::printf("  [potrf] n=%-3d nm=%-3d %s | recon %.2e  lapack %.2e  mkl %.2e  "
                "untouched %.0e (gate %.1e)\n",
                n, nm, upper ? "U" : "L", res, elap, dmkl, untouched, gate);
    std::free(apc);
    std::free(amc);
}

/* -------- unit: potrf -- a non-SPD lane poisons only itself, not its siblings -- */
void unit_potrf_nonspd(MKL_COMPACT_PACK fmt, int nm, int n)
{
    const int bad = nm / 2;
    Batch A = spdbatch(n, nm);
    A[bad][0] = -1.0; /* A(0,0) < 0: the very first pivot sqrt is NaN for this lane */
    double *apc = packc(fmt, n, n, A, nm);
    cqr_ispc_dpotrf_compact(0, 0, n, apc, n, nm); /* lower */
    Batch F = unpackc(fmt, n, n, apc, nm);

    double sib = 0;
    bool bad_nan = !std::isfinite(F[bad][0]);
    for (int v = 0; v < nm; ++v) {
        if (v == bad) continue;
        Mat L = A[v];
        LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', n, L.data(), n);
        double d = 0;
        for (int j = 0; j < n; ++j)
            for (int i = j; i < n; ++i)
                d = std::max(d, std::fabs(F[v][i + (std::size_t)j * n] -
                                          L[i + (std::size_t)j * n]));
        sib = std::max(sib, d / std::max(froben(n, n, L.data()), 1e-300));
    }
    expect(bad_nan, "potrf: non-SPD lane yields NaN/Inf");
    expect(sib <= 50.0 * n * EPS, "potrf: sibling lanes unaffected by non-SPD lane");
    std::printf("  [potrf] non-SPD lane %d of %d | sibling %.2e (bad lane non-finite: "
                "%s)\n",
                bad, nm, sib, bad_nan ? "yes" : "no");
    std::free(apc);
}

/* -------- integration: SPD solve A X = B via ISPC potrf + two MKL trsm --------- */
void integration_spd_solve(MKL_COMPACT_PACK fmt, int nm, int n, int nrhs)
{
    Mat X((std::size_t)n * nrhs); /* known X(i,j) = (j+1) + 0.5 i */
    for (int j = 0; j < nrhs; ++j)
        for (int i = 0; i < n; ++i)
            X[i + (std::size_t)j * n] = (j + 1) + 0.5 * i;
    Batch A = spdbatch(n, nm), B(nm, Mat((std::size_t)n * nrhs));
    for (int v = 0; v < nm; ++v)
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, nrhs, n, 1.0,
                    A[v].data(), n, X.data(), n, 0.0, B[v].data(), n);
    double *apc = packc(fmt, n, n, A, nm), *bpc = packc(fmt, n, nrhs, B, nm);
    cqr_ispc_dpotrf_compact(0, 0, n, apc, n, nm); /* A = L L^T (lower) */
    /* L Y = B (no-trans), then L^T X = Y (trans) -- MKL trsm (ISPC's is upper-only) */
    mkl_dtrsm_compact(MKL_COL_MAJOR, MKL_LEFT, MKL_LOWER, MKL_NOTRANS, MKL_NONUNIT, n,
                      nrhs, 1.0, apc, n, bpc, n, fmt, nm);
    mkl_dtrsm_compact(MKL_COL_MAJOR, MKL_LEFT, MKL_LOWER, MKL_TRANS, MKL_NONUNIT, n, nrhs,
                      1.0, apc, n, bpc, n, fmt, nm);
    Batch Xi = unpackc(fmt, n, nrhs, bpc, nm);
    double fwd = 0;
    for (int v = 0; v < nm; ++v)
        for (int e = 0; e < n * nrhs; ++e)
            fwd = std::max(fwd, std::fabs(Xi[v][e] - X[e]));
    const double rtol = 100.0 * n * EPS * (1.0 + 0.5 * n);
    expect(fwd <= rtol, "potrf+trsm: SPD solve recovers known X");
    std::printf("  [spd  ] n=%-3d nm=%-3d nrhs=%d | fwd %.2e (rtol %.1e)\n", n, nm, nrhs,
                fwd, rtol);
    std::free(apc);
    std::free(bpc);
}
} /* namespace */

int main()
{
    mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL);
    mkl_set_num_threads(1);
    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = (fmt == MKL_COMPACT_SSE ? 16 : fmt == MKL_COMPACT_AVX ? 32 : 64) / 8;
    std::printf("ISPC compact-QR correctness (compact format=%d, V=%d, ISPC gang=%d)\n",
                (int)fmt, V, cqr_ispc_gang_width());
    if (cqr_ispc_gang_width() != V) { /* kernels need gang width == MKL's V */
        std::printf("ISPC gang width %d != MKL compact V %d -- rebuild the ISPC target "
                    "with a width-%d gang (e.g. avx512skx-x%d).\n",
                    cqr_ispc_gang_width(), V, V, V);
        return 77;
    }

    /* geqrf: square, tall, and a padded partial group (nm % 8 != 0). */
    unit_geqrf(fmt, 8, 16, 16);
    unit_geqrf(fmt, 16, 40, 40);
    unit_geqrf(fmt, 8, 24, 12); /* tall m>n */
    unit_geqrf(fmt, 7, 20, 20); /* padded */

    /* ormqr: Q and Q^T (inside), square and tall, blocked and remainder nrhs. */
    unit_ormqr(fmt, 8, 20, 20, 5);
    unit_ormqr(fmt, 16, 40, 40, 1);
    unit_ormqr(fmt, 8, 24, 12, 3); /* tall: k = 12 < m = 24 */
    unit_ormqr(fmt, 5, 16, 16, 4); /* padded */

    /* trsm: alpha != 1, nrhs = 1 (remainder only) and 6 (4-block + remainder). */
    unit_trsm(fmt, 8, 20, 6, 1.0);
    unit_trsm(fmt, 8, 20, 6, 2.5);
    unit_trsm(fmt, 16, 40, 1, -0.5);
    unit_trsm(fmt, 7, 32, 3, 1.0); /* padded */

    /* integration: end-to-end solve (equiv-checked when nm % 8 == 0). */
    integration_solve(fmt, 8, 30, 4, true);
    integration_solve(fmt, 16, 100, 8, true);
    integration_solve(fmt, 13, 64, 2, false); /* padded */

    /* potrf: lower (contiguous) and upper (strided) paths, square and padded. */
    unit_potrf(fmt, 8, 24, false);
    unit_potrf(fmt, 8, 24, true);
    unit_potrf(fmt, 16, 50, false);
    unit_potrf(fmt, 16, 50, true);
    unit_potrf(fmt, 7, 20, false); /* padded */
    unit_potrf(fmt, 7, 20, true);  /* padded, strided */
    unit_potrf_nonspd(fmt, 8, 24);
    integration_spd_solve(fmt, 8, 30, 4);
    integration_spd_solve(fmt, 16, 100, 6);

    if (failures) {
        std::printf("\n%d CHECK(S) FAILED\n", failures);
        return 1;
    }
    std::printf("\nall checks passed\n");
    return 0;
}
