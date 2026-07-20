/* test_cqr_geqrf_mkl.cpp
 *
 * Validation of cqr_mkl_?geqrf_compact against real Intel MKL and dense LAPACK,
 * through the genuine MKL Compact pipeline (mkl_dgepack_compact /
 * mkl_dgeunpack_compact). This is the design document's "hard correctness gate
 * against standard dense LAPACK equivalents" (cqr_mkl_dgeqrf_compact_design.md
 * section 7).
 *
 * Suite 1 (section 7.1) -- Factorization invariants vs dense LAPACK:
 *   A random batch is factored by cqr_mkl_dgeqrf_compact, unpacked, and per
 *   matrix checked against the LAPACK-style QR contract used by the GPU
 *   competition checker: Q is materialized from (H, tau) via LAPACKE_dorgqr,
 *   R = triu(H), and the routine gates
 *     - factorization residual || R - Q^T A ||_1 / ||A||_1  <= 20 * n * eps,
 *     - orthogonality         || Q^T Q - I ||_1            <= 100 * n * eps.
 *   (H, tau) are also compared elementwise to LAPACKE_dgeqrf as a diagnostic.
 *
 * Suite 2 (section 7.2) -- Cross-check vs mkl_dgeqrf_compact:
 *   The same packed batch is factored by both cqr_mkl_dgeqrf_compact and the
 *   native mkl_dgeqrf_compact; the two compact buffers are compared elementwise.
 *   Run for both column- and row-major (exercises the strided kernel).
 *
 * Suite 3 (section 7.3) -- End-to-end AX = B:
 *   B = A X for known X; cqr_mkl_dgeqrf_compact -> cqr_mkl_dormqr_compact('L','T')
 *   -> mkl_dtrsm_compact must recover X. Gates forward error and residual.
 *
 * Build: needs Intel MKL (headers + libmkl_rt); wired up by CMakeLists.txt.
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#include <mkl.h>
#include <mkl_compact.h>

#include "cqr_mkl_ext.h"
#include "cqr_mkl_alloc.h" /* mkl_alloc_bytes (calls mkl_malloc; links MKL) */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <random>
#include <vector>
#include <algorithm>

namespace {

const double eps = std::numeric_limits<double>::epsilon();

int vlen(MKL_COMPACT_PACK fmt)
{
    return cqr::detail::vlen_for_format<double>(fmt);
}

std::mt19937_64 rng(7);

double frand()
{
    static std::uniform_real_distribution<double> dist(-1.0, 1.0);
    return dist(rng);
}

/* L1 (max column sum) norm of a column-major m x n matrix */
double norm1(const double *M, int m, int n)
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

double maxdiff(const double *a, const double *b, size_t n)
{
    double d = 0;
    for (size_t i = 0; i < n; ++i)
        d = std::max(d, std::abs(a[i] - b[i]));
    return d;
}

template <class T> std::vector<T *> batch_ptrs(T *base, int nm, size_t stride)
{
    std::vector<T *> p(nm);
    for (int v = 0; v < nm; ++v)
        p[v] = base + (size_t)v * stride;
    return p;
}

/* Input structures (a subset of the GPU competition's stress set). The QR
 * residual and orthogonality are backward-stable quantities, so they must hold
 * to working precision for every structure -- rank deficiency and near-collinear
 * columns included -- exactly as they do for dense LAPACK. */
enum Structure { DENSE, RANK_DEFICIENT, NEAR_COLLINEAR };

/* build a random m x n matrix (column-major). For DENSE, a diagonal boost tames
 * the conditioning and cond applies the competition's column scaling
 * (columns *= logspace(0,-cond,n)). The structured variants make the last column
 * a (near-)copy of the first, so the trailing reflector sees a (near-)zero
 * sub-diagonal norm -- the branch the masked larfg must get right. */
void gen_matrix(double *A, int m, int n, double cond, Structure s = DENSE)
{
    for (int i = 0; i < m * n; ++i)
        A[i] = frand();
    if (s == DENSE) {
        for (int i = 0; i < std::min(m, n); ++i)
            A[i + (size_t)i * m] += 2.0;
        if (cond > 0.0)
            for (int j = 0; j < n; ++j) {
                double sc = std::pow(10.0, -cond * (n > 1 ? (double)j / (n - 1) : 0.0));
                for (int i = 0; i < m; ++i)
                    A[i + (size_t)j * m] *= sc;
            }
    }
    else if (n >= 2) {
        const double noise = (s == NEAR_COLLINEAR) ? 1e-9 : 0.0; /* exact dup if 0 */
        for (int i = 0; i < m; ++i)
            A[i + (size_t)(n - 1) * m] = A[i] * (1.0 + noise * frand());
    }
}

/* ---------------- Suite 1: invariants vs dense LAPACK (col-major) ------ */

int suite1(int nm, int m, int n, double cond, Structure structure = DENSE)
{
    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = vlen(fmt);
    const int k = std::min(m, n);
    const size_t sA = (size_t)m * n, sT = (size_t)k;

    std::vector<double> A(nm * sA);
    for (int v = 0; v < nm; ++v)
        gen_matrix(A.data() + v * sA, m, n, cond, structure);

    /* pack A, factor with the routine under test, unpack (H, tau) */
    auto Ap = batch_ptrs<const double>(A.data(), nm, sA);
    MKL_INT sz_a = mkl_dget_size_compact(m, n, fmt, nm);
    MKL_INT sz_t = mkl_dget_size_compact(k, 1, fmt, nm);
    auto ap_buf = cqr::detail::mkl_alloc_bytes<double>(sz_a);
    auto taup_buf = cqr::detail::mkl_alloc_bytes<double>(sz_t);
    double *ap = ap_buf.get(), *taup = taup_buf.get();
    mkl_dgepack_compact(MKL_COL_MAJOR, m, n, Ap.data(), m, ap, m, fmt, nm);

    MKL_INT info = 99;
    double wq;
    cqr_mkl_dgeqrf_compact(MKL_COL_MAJOR, m, n, ap, m, taup, &wq, -1, &info, fmt, nm);
    cqr_mkl_dgeqrf_compact(MKL_COL_MAJOR, m, n, ap, m, taup, &wq, (MKL_INT)wq, &info, fmt,
                           nm);

    std::vector<double> H(nm * sA), tau(nm * sT);
    auto Hp = batch_ptrs<double>(H.data(), nm, sA);
    auto Tp = batch_ptrs<double>(tau.data(), nm, sT);
    mkl_dgeunpack_compact(MKL_COL_MAJOR, m, n, Hp.data(), m, ap, m, fmt, nm);
    mkl_dgeunpack_compact(MKL_COL_MAJOR, k, 1, Tp.data(), k, taup, k, fmt, nm);

    int fails = 0;
    if (info != 0) {
        ++fails;
        std::printf("    info = %d (expected 0)\n", (int)info);
    }

    double worst_res = 0, worst_orth = 0, worst_el = 0;
    std::vector<double> Q((size_t)m * k), QtA((size_t)k * n), QtQ((size_t)k * k);
    std::vector<double> Href(sA), tauref(sT);
    for (int v = 0; v < nm; ++v) {
        const double *Av = A.data() + v * sA;
        const double *Hv = H.data() + v * sA, *tv = tau.data() + v * sT;

        /* Q = householder_product(H, tau): first k columns of Q (m x k). The
         * reflectors occupy the first k columns of the m x n H, i.e. the first
         * m*k contiguous (column-major) elements. */
        std::copy(Hv, Hv + (size_t)m * k, Q.begin());
        LAPACKE_dorgqr(LAPACK_COL_MAJOR, m, k, k, Q.data(), m, tv);

        /* residual R - Q^T A, R = triu(H) (k x n) */
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, k, n, m, 1.0, Q.data(), m,
                    Av, m, 0.0, QtA.data(), k);
        double resid = 0;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < k; ++i) {
                double R = (i <= j) ? Hv[i + (size_t)j * m] : 0.0; /* triu */
                resid = std::max(resid, std::abs(R - QtA[i + (size_t)j * k]));
            }
        worst_res = std::max(worst_res, resid / std::max(norm1(Av, m, n), 1e-300));

        /* orthogonality Q^T Q - I (k x k) */
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, k, k, m, 1.0, Q.data(), m,
                    Q.data(), m, 0.0, QtQ.data(), k);
        for (int i = 0; i < k; ++i)
            QtQ[i + (size_t)i * k] -= 1.0;
        worst_orth = std::max(worst_orth, norm1(QtQ.data(), k, k));

        /* diagnostic: elementwise vs LAPACKE_dgeqrf */
        std::copy(Av, Av + sA, Href.begin());
        LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, Href.data(), m, tauref.data());
        double el =
            std::max(maxdiff(Hv, Href.data(), sA), maxdiff(tv, tauref.data(), sT));
        worst_el = std::max(worst_el, el / std::max(norm1(Av, m, n), 1e-300));
    }

    // TODO: review: worst_el is printed as a diagnostic only (design doc 7.1). A
    // loose gate (say 100*n*eps) restricted to the DENSE cases would also catch
    // sign-convention regressions the residual gates cannot; rank-deficient cases
    // must stay ungated (reflectors are not unique there, el ~ 1e-2 is expected).
    const double rtol_res = 20.0 * n * eps, rtol_orth = 100.0 * n * eps;
    bool ok = (worst_res <= rtol_res) && (worst_orth <= rtol_orth);
    fails += !ok;
    const char *sname = structure == DENSE            ? "dense"
                        : structure == RANK_DEFICIENT ? "rankdef"
                                                      : "collin";
    std::printf("  [suite1] V=%-2d nm=%-2d m=%-3d n=%-3d cond=%.0f %-8s| res %.2e (%.1e) "
                "orth %.2e (%.1e) el %.1e %s\n",
                V, nm, m, n, cond, sname, worst_res, rtol_res, worst_orth, rtol_orth,
                worst_el, ok ? "OK" : "FAIL");
    return fails;
}

/* ---------------- Suite 2: cross-check vs mkl_dgeqrf_compact ----------- */

int suite2(MKL_LAYOUT layout, int nm, int m, int n)
{
    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = vlen(fmt);
    const int k = std::min(m, n);
    const bool row = (layout == MKL_ROW_MAJOR);
    const size_t sA = (size_t)m * n;
    const MKL_INT ldA = row ? n : m, ldc = row ? n : m;

    std::vector<double> A(nm * sA);
    for (int v = 0; v < nm; ++v)
        gen_matrix(A.data() + v * sA, m, n, 0.0);
    /* A was generated column-major; for a row-major run reinterpret the same
     * numbers as a row-major m x n (a genuinely different matrix, still fine). */
    auto Ap = batch_ptrs<const double>(A.data(), nm, sA);

    MKL_INT sz_a = mkl_dget_size_compact(m, n, fmt, nm);
    MKL_INT sz_t = mkl_dget_size_compact(k, 1, fmt, nm);
    auto ap1 = cqr::detail::mkl_alloc_bytes<double>(sz_a);
    auto tp1 = cqr::detail::mkl_alloc_bytes<double>(sz_t);
    auto ap2 = cqr::detail::mkl_alloc_bytes<double>(sz_a);
    auto tp2 = cqr::detail::mkl_alloc_bytes<double>(sz_t);
    mkl_dgepack_compact(layout, m, n, Ap.data(), ldA, ap1.get(), ldc, fmt, nm);
    mkl_dgepack_compact(layout, m, n, Ap.data(), ldA, ap2.get(), ldc, fmt, nm);

    MKL_INT info = 0;
    /* cqr's unblocked kernel needs no scratch (its query returns 1). */
    double wq_cqr;
    cqr_mkl_dgeqrf_compact(layout, m, n, ap1.get(), ldc, tp1.get(), &wq_cqr, -1, &info,
                           fmt, nm);
    cqr_mkl_dgeqrf_compact(layout, m, n, ap1.get(), ldc, tp1.get(), &wq_cqr,
                           (MKL_INT)wq_cqr, &info, fmt, nm);

    /* MKL's compact geqrf DOES need workspace (~n*V doubles), so query its own
     * lwork -- reusing cqr's lwork=1 would under-size work. Compact routines skip
     * argument checking, so an under-sized work array is undefined behavior: some
     * MKL builds tolerate it, others overrun the heap ("malloc unaligned tcache"). */
    double wq_mkl;
    mkl_dgeqrf_compact(layout, m, n, ap2.get(), ldc, tp2.get(), &wq_mkl, -1, &info, fmt,
                       nm);
    std::vector<double> work((size_t)std::max<MKL_INT>((MKL_INT)wq_mkl, 1));
    mkl_dgeqrf_compact(layout, m, n, ap2.get(), ldc, tp2.get(), work.data(),
                       (MKL_INT)work.size(), &info, fmt, nm);

    /* compare the two compact buffers elementwise (same input, same convention) */
    double da = maxdiff(ap1.get(), ap2.get(), (size_t)sz_a / sizeof(double));
    double dt = maxdiff(tp1.get(), tp2.get(), (size_t)sz_t / sizeof(double));
    // TODO: review: fixed cross-check tolerance (design doc 7.2). Observed
    // agreement with MKL is ~1e-14 on these inputs; a tolerance scaled with n*eps
    // would be a tighter regression signal than the flat 1e-9 if wanted.
    const double tol = 1e-9;
    bool ok = (da <= tol && dt <= tol);
    std::printf("  [suite2] %s V=%-2d nm=%-2d m=%-3d n=%-3d | max|ap-mkl| %.2e "
                "max|tau-mkl| %.2e (tol %.0e) %s\n",
                row ? "row" : "col", V, nm, m, n, da, dt, tol, ok ? "OK" : "FAIL");
    return !ok;
}

/* ---------------- Suite 3: end-to-end AX = B (col-major) --------------- */

int suite3(int nm, int n, int nrhs)
{
    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = vlen(fmt), m = n, k = n;

    std::vector<double> X((size_t)n * nrhs);
    for (int j = 0; j < nrhs; ++j)
        for (int i = 0; i < n; ++i)
            X[i + (size_t)j * n] = double(j + 1);

    const size_t sA = (size_t)n * n, sB = (size_t)n * nrhs;
    std::vector<double> A(nm * sA), B(nm * sB);
    for (int v = 0; v < nm; ++v) {
        double *Av = A.data() + v * sA, *Bv = B.data() + v * sB;
        gen_matrix(Av, n, n, 0.0);
        for (int j = 0; j < nrhs; ++j)
            for (int i = 0; i < n; ++i) {
                double s = 0;
                for (int l = 0; l < n; ++l)
                    s += Av[i + (size_t)l * n] * X[l + (size_t)j * n];
                Bv[i + (size_t)j * n] = s;
            }
    }
    auto Ap = batch_ptrs<const double>(A.data(), nm, sA);
    auto Bp = batch_ptrs<const double>(B.data(), nm, sB);

    MKL_INT sz_a = mkl_dget_size_compact(m, n, fmt, nm);
    MKL_INT sz_t = mkl_dget_size_compact(k, 1, fmt, nm);
    MKL_INT sz_c = mkl_dget_size_compact(m, nrhs, fmt, nm);
    auto ap_buf = cqr::detail::mkl_alloc_bytes<double>(sz_a);
    auto taup_buf = cqr::detail::mkl_alloc_bytes<double>(sz_t);
    auto cp_buf = cqr::detail::mkl_alloc_bytes<double>(sz_c);
    double *ap = ap_buf.get(), *taup = taup_buf.get(), *cp = cp_buf.get();
    mkl_dgepack_compact(MKL_COL_MAJOR, m, n, Ap.data(), m, ap, m, fmt, nm);
    mkl_dgepack_compact(MKL_COL_MAJOR, m, nrhs, Bp.data(), m, cp, m, fmt, nm);

    MKL_INT info = 99;

    /* 1. our compact QR. Each routine owns a work array sized from its OWN lwork
     *    query -- never carry one routine's lwork over to another (see the
     *    workspace note in cqr_mkl_ext.h): geqrf and ormqr can need different
     *    amounts, and an undersized work array is undefined behavior. */
    double wq_geqrf;
    cqr_mkl_dgeqrf_compact(MKL_COL_MAJOR, m, n, ap, m, taup, &wq_geqrf, -1, &info, fmt,
                           nm);
    std::vector<double> work_geqrf((size_t)std::max<MKL_INT>((MKL_INT)wq_geqrf, 1));
    cqr_mkl_dgeqrf_compact(MKL_COL_MAJOR, m, n, ap, m, taup, work_geqrf.data(),
                           (MKL_INT)work_geqrf.size(), &info, fmt, nm);

    /* 2. our compact apply Q^T, with its own separately queried workspace. */
    double wq_ormqr;
    cqr_mkl_dormqr_compact(MKL_COL_MAJOR, 'L', 'T', m, nrhs, k, ap, m, taup, cp, m,
                           &wq_ormqr, -1, &info, fmt, nm);
    std::vector<double> work_ormqr((size_t)std::max<MKL_INT>((MKL_INT)wq_ormqr, 1));
    cqr_mkl_dormqr_compact(MKL_COL_MAJOR, 'L', 'T', m, nrhs, k, ap, m, taup, cp, m,
                           work_ormqr.data(), (MKL_INT)work_ormqr.size(), &info, fmt, nm);

    /* 3. MKL compact triangular solve R X = Q^T B */
    mkl_dtrsm_compact(MKL_COL_MAJOR, MKL_LEFT, MKL_UPPER, MKL_NOTRANS, MKL_NONUNIT, n,
                      nrhs, 1.0, ap, m, cp, m, fmt, nm);

    std::vector<double> Xhat(nm * sB);
    auto Op = batch_ptrs<double>(Xhat.data(), nm, sB);
    mkl_dgeunpack_compact(MKL_COL_MAJOR, n, nrhs, Op.data(), n, cp, m, fmt, nm);

    int fails = 0;
    if (info != 0) {
        ++fails;
        std::printf("    info = %d (expected 0)\n", (int)info);
    }
    double worst_fwd = 0, worst_res = 0;
    std::vector<double> AX(sB);
    for (int v = 0; v < nm; ++v) {
        const double *Av = A.data() + v * sA, *Bv = B.data() + v * sB;
        const double *Xv = Xhat.data() + v * sB;
        worst_fwd = std::max(worst_fwd, maxdiff(Xv, X.data(), sB) /
                                            std::max(norm1(X.data(), n, nrhs), 1e-300));
        for (int j = 0; j < nrhs; ++j)
            for (int i = 0; i < n; ++i) {
                double s = 0;
                for (int l = 0; l < n; ++l)
                    s += Av[i + (size_t)l * n] * Xv[l + (size_t)j * n];
                AX[i + (size_t)j * n] = s;
            }
        worst_res = std::max(worst_res, maxdiff(AX.data(), Bv, sB) /
                                            std::max(norm1(Bv, n, nrhs), 1e-300));
    }
    const double rtol = 100.0 * n * eps;
    bool ok = (worst_fwd <= rtol && worst_res <= rtol);
    fails += !ok;
    std::printf(
        "  [suite3] V=%-2d nm=%-2d n=%-3d nrhs=%d | fwd %.2e res %.2e (rtol %.2e) %s\n",
        V, nm, n, nrhs, worst_fwd, worst_res, rtol, ok ? "OK" : "FAIL");
    return fails;
}

} /* anonymous namespace */

int main()
{
    std::printf("MKL compact format = %d, V(double) = %d\n",
                (int)mkl_get_format_compact(), vlen(mkl_get_format_compact()));

    int fails = 0;

    /* Suite 1: invariants vs dense LAPACK, square + rectangular, a range of cond */
    fails += suite1(8, 30, 30, 0.0);
    fails += suite1(16, 60, 60, 0.0);
    fails += suite1(11, 43, 43, 0.0); /* padded partial group */
    fails += suite1(8, 64, 20, 0.0);  /* tall */
    fails += suite1(8, 20, 64, 0.0);  /* wide */
    fails += suite1(8, 30, 30, 4.0);  /* column-scaled (dynamic range) */
    fails += suite1(8, 128, 128, 0.0);
    /* conditioning-robustness stress: backward-stable gates must still hold */
    fails += suite1(8, 40, 40, 0.0, RANK_DEFICIENT);
    fails += suite1(8, 40, 40, 0.0, NEAR_COLLINEAR);
    fails += suite1(11, 60, 24, 0.0, RANK_DEFICIENT); /* wide-ish, padded group */

    /* Suite 2: cross-check vs mkl_dgeqrf_compact, both layouts */
    fails += suite2(MKL_COL_MAJOR, 8, 30, 30);
    fails += suite2(MKL_COL_MAJOR, 8, 64, 20);
    fails += suite2(MKL_ROW_MAJOR, 8, 30, 30);
    fails += suite2(MKL_ROW_MAJOR, 11, 40, 24);

    /* Suite 3: end-to-end solver */
    fails += suite3(8, 30, 5);
    fails += suite3(8, 60, 4);
    fails += suite3(16, 128, 3);
    fails += suite3(7, 32, 6); /* padded partial group */

    if (fails) {
        std::printf("\n%d CHECK(S) FAILED\n", fails);
        return 1;
    }
    std::printf("\nall checks passed\n");
    return 0;
}
