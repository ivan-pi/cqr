/* test_cqr_syrk_mkl.cpp
 *
 * Validation of cqr_mkl_?syrk_compact against real Intel MKL, through the genuine
 * MKL Compact pipeline (mkl_?gepack_compact / mkl_?geunpack_compact), in three
 * independent ways over the full feature matrix (precision x layout x uplo x
 * trans) and a range of alpha/beta and batch shapes (including padded partial
 * last groups). See cqr_mkl_dsyrk_compact_design.md.
 *
 * Suite A (design doc 7.1) -- vs dense per-matrix cblas_?syrk:
 *   The compact result is checked, element for element over the *whole* n x n C,
 *   against a per-matrix reference from the same seed. Because ?syrk writes only
 *   the uplo triangle, this simultaneously gates the active-triangle result and
 *   that the opposite triangle is left untouched.
 * Suite B (design doc 7.2) -- vs mkl_?gemm_compact (same compact pipeline):
 *   The full product alpha*A*op(A) + beta*C is formed with mkl_?gemm_compact into
 *   a second packed C; since gemm writes the whole matrix and syrk only one
 *   triangle, the comparison is restricted to the active uplo triangle.
 * Suite C (design doc 7.3) -- end-to-end Cholesky QR, no MKL compute kernel:
 *   G = A^T A (cqr_mkl_?syrk_compact) -> R = chol(G) (cqr_mkl_?potrf_compact) ->
 *   Q = A R^{-1} (cqr_mkl_?trsm_compact) must satisfy Q R = A and Q^T Q = I.
 *
 * Build: needs Intel MKL; wired up by CMakeLists.txt.
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#include <mkl.h>
#include <mkl_compact.h>

#include "cqr_mkl_ext.h"
#include "cqr_mkl_alloc.h"       /* mkl_alloc_bytes (calls mkl_malloc; links MKL) */
#include "test_compact_util.hpp" /* rng/frand, max_abs_diff, norm1, batch_ptrs */

#include <cstdio>
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>
#include <type_traits>

using namespace cqr::test;

namespace {

/* max_abs_diff, norm1, maxabs, batch_ptrs and frand come from
 * test_compact_util.hpp (shared across the compact test suites). */

/* ------------------------------------------------------------------ */
/* Precision-dispatched MKL / CBLAS wrappers: overload on the scalar    */
/* pointer type, or specialize the size query (which carries no pointer).*/
/* ------------------------------------------------------------------ */

template <typename T>
MKL_INT compact_size(MKL_INT m, MKL_INT n, MKL_COMPACT_PACK f, MKL_INT nm);
template <>
MKL_INT compact_size<double>(MKL_INT m, MKL_INT n, MKL_COMPACT_PACK f, MKL_INT nm)
{
    return mkl_dget_size_compact(m, n, f, nm);
}
template <>
MKL_INT compact_size<float>(MKL_INT m, MKL_INT n, MKL_COMPACT_PACK f, MKL_INT nm)
{
    return mkl_sget_size_compact(m, n, f, nm);
}

void pack(MKL_LAYOUT l, MKL_INT m, MKL_INT n, const double *const *a, MKL_INT lda,
          double *ap, MKL_INT ldap, MKL_COMPACT_PACK f, MKL_INT nm)
{
    mkl_dgepack_compact(l, m, n, a, lda, ap, ldap, f, nm);
}
void pack(MKL_LAYOUT l, MKL_INT m, MKL_INT n, const float *const *a, MKL_INT lda,
          float *ap, MKL_INT ldap, MKL_COMPACT_PACK f, MKL_INT nm)
{
    mkl_sgepack_compact(l, m, n, a, lda, ap, ldap, f, nm);
}

void unpack(MKL_LAYOUT l, MKL_INT m, MKL_INT n, double **a, MKL_INT lda, const double *ap,
            MKL_INT ldap, MKL_COMPACT_PACK f, MKL_INT nm)
{
    mkl_dgeunpack_compact(l, m, n, a, lda, ap, ldap, f, nm);
}
void unpack(MKL_LAYOUT l, MKL_INT m, MKL_INT n, float **a, MKL_INT lda, const float *ap,
            MKL_INT ldap, MKL_COMPACT_PACK f, MKL_INT nm)
{
    mkl_sgeunpack_compact(l, m, n, a, lda, ap, ldap, f, nm);
}

void gemm_compact(MKL_LAYOUT l, MKL_TRANSPOSE ta, MKL_TRANSPOSE tb, MKL_INT m, MKL_INT n,
                  MKL_INT k, double alpha, const double *ap, MKL_INT ldap,
                  const double *bp, MKL_INT ldbp, double beta, double *cp, MKL_INT ldcp,
                  MKL_COMPACT_PACK f, MKL_INT nm)
{
    mkl_dgemm_compact(l, ta, tb, m, n, k, alpha, ap, ldap, bp, ldbp, beta, cp, ldcp, f,
                      nm);
}
void gemm_compact(MKL_LAYOUT l, MKL_TRANSPOSE ta, MKL_TRANSPOSE tb, MKL_INT m, MKL_INT n,
                  MKL_INT k, float alpha, const float *ap, MKL_INT ldap, const float *bp,
                  MKL_INT ldbp, float beta, float *cp, MKL_INT ldcp, MKL_COMPACT_PACK f,
                  MKL_INT nm)
{
    mkl_sgemm_compact(l, ta, tb, m, n, k, alpha, ap, ldap, bp, ldbp, beta, cp, ldcp, f,
                      nm);
}

/* dense per-matrix reference */
void syrk(CBLAS_LAYOUT l, CBLAS_UPLO u, CBLAS_TRANSPOSE t, MKL_INT n, MKL_INT k,
          double alpha, const double *a, MKL_INT lda, double beta, double *c, MKL_INT ldc)
{
    cblas_dsyrk(l, u, t, n, k, alpha, a, lda, beta, c, ldc);
}
void syrk(CBLAS_LAYOUT l, CBLAS_UPLO u, CBLAS_TRANSPOSE t, MKL_INT n, MKL_INT k,
          float alpha, const float *a, MKL_INT lda, float beta, float *c, MKL_INT ldc)
{
    cblas_ssyrk(l, u, t, n, k, alpha, a, lda, beta, c, ldc);
}

/* routine under test */
void syrk_compact(MKL_LAYOUT l, MKL_UPLO u, MKL_TRANSPOSE t, MKL_INT n, MKL_INT k,
                  double alpha, const double *ap, MKL_INT ldap, double beta, double *cp,
                  MKL_INT ldcp, MKL_COMPACT_PACK f, MKL_INT nm)
{
    cqr_mkl_dsyrk_compact(l, u, t, n, k, alpha, ap, ldap, beta, cp, ldcp, f, nm);
}
void syrk_compact(MKL_LAYOUT l, MKL_UPLO u, MKL_TRANSPOSE t, MKL_INT n, MKL_INT k,
                  float alpha, const float *ap, MKL_INT ldap, float beta, float *cp,
                  MKL_INT ldcp, MKL_COMPACT_PACK f, MKL_INT nm)
{
    cqr_mkl_ssyrk_compact(l, u, t, n, k, alpha, ap, ldap, beta, cp, ldcp, f, nm);
}

const char *pname(bool dbl)
{
    return dbl ? "d" : "s";
}

/* element (i,j) of a dense matrix stored in the given layout (ld = column stride
 * col-major / row stride row-major) */
template <class T> T elem(const T *p, int ld, int i, int j, bool rowmajor)
{
    return rowmajor ? p[(size_t)i * ld + j] : p[(size_t)j * ld + i];
}

/* max |X - Y| over the active uplo triangle of an n x n matrix (both stored in
 * the same layout) */
template <class T>
double tri_maxdiff(const T *X, const T *Y, int n, int ld, bool lower, bool rowmajor)
{
    double d = 0;
    for (int i = 0; i < n; ++i) {
        const int jlo = lower ? 0 : i, jhi = lower ? i + 1 : n;
        for (int j = jlo; j < jhi; ++j)
            d = std::max(d, (double)std::abs(elem(X, ld, i, j, rowmajor) -
                                             elem(Y, ld, i, j, rowmajor)));
    }
    return d;
}

/* ------------------------------------------------------------------ */
/* Suite A: cqr_mkl_?syrk_compact vs per-matrix dense cblas_?syrk      */
/* ------------------------------------------------------------------ */

template <typename T>
int suiteA(bool rowmajor, bool lower, bool trans, int nm, int n, int k, T alpha, T beta)
{
    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = cqr::detail::vlen_for_format<T>(fmt);
    const T eps = std::numeric_limits<T>::epsilon();

    const MKL_LAYOUT ml = rowmajor ? MKL_ROW_MAJOR : MKL_COL_MAJOR;
    const CBLAS_LAYOUT cl = rowmajor ? CblasRowMajor : CblasColMajor;
    const MKL_UPLO mu = lower ? MKL_LOWER : MKL_UPPER;
    const CBLAS_UPLO cu = lower ? CblasLower : CblasUpper;
    const MKL_TRANSPOSE mtr = trans ? MKL_TRANS : MKL_NOTRANS;
    const CBLAS_TRANSPOSE ctr = trans ? CblasTrans : CblasNoTrans;

    /* A is n x k (notrans) or k x n (trans); the dense == compact leading dim is
     * the stored-axis extent (rows col-major, cols row-major). */
    const int Arows = trans ? k : n, Acols = trans ? n : k;
    const int ldA = rowmajor ? Acols : Arows;
    const int ldC = n; /* C is n x n in both layouts */
    const size_t sA = (size_t)Arows * Acols, sC = (size_t)n * n;

    std::vector<T> A(nm * sA), C(nm * sC), Cref(nm * sC), Cout(nm * sC);
    std::generate(A.begin(), A.end(), frand<T>);
    std::generate(C.begin(), C.end(), frand<T>);
    Cref = C;
    for (int v = 0; v < nm; ++v)
        syrk(cl, cu, ctr, n, k, alpha, A.data() + v * sA, ldA, beta, Cref.data() + v * sC,
             ldC);

    auto Ap = batch_ptrs<const T>(A.data(), nm, sA);
    auto Cp = batch_ptrs<const T>(C.data(), nm, sC);
    auto ap = cqr::detail::mkl_alloc_bytes<T>(compact_size<T>(Arows, Acols, fmt, nm));
    auto cp = cqr::detail::mkl_alloc_bytes<T>(compact_size<T>(n, n, fmt, nm));
    pack(ml, Arows, Acols, Ap.data(), ldA, ap.get(), ldA, fmt, nm);
    pack(ml, n, n, Cp.data(), ldC, cp.get(), ldC, fmt, nm);

    syrk_compact(ml, mu, mtr, n, k, alpha, ap.get(), ldA, beta, cp.get(), ldC, fmt, nm);

    auto Op = batch_ptrs<T>(Cout.data(), nm, sC);
    unpack(ml, n, n, Op.data(), ldC, cp.get(), ldC, fmt, nm);

    double worst = 0;
    for (int v = 0; v < nm; ++v) {
        /* whole matrix: active triangle correct + opposite triangle intact */
        double rel = max_abs_diff(Cout.data() + v * sC, Cref.data() + v * sC, sC) /
                     std::max(maxabs(Cref.data() + v * sC, sC), 1e-300);
        worst = std::max(worst, rel);
    }
    const double rtol = 32.0 * (k + 1) * (double)eps;
    const bool ok = (worst <= rtol);
    std::printf("  [A:cblas] %s%s uplo=%c trans=%c V=%-2d nm=%-2d n=%-3d k=%-3d "
                "a=%+.1f b=%+.1f | rel %.2e (rtol %.1e) %s\n",
                pname(std::is_same<T, double>::value), rowmajor ? "/row" : "/col",
                lower ? 'L' : 'U', trans ? 'T' : 'N', V, nm, n, k, (double)alpha,
                (double)beta, worst, rtol, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

/* ------------------------------------------------------------------ */
/* Suite B: cqr_mkl_?syrk_compact vs mkl_?gemm_compact (triangle only) */
/* ------------------------------------------------------------------ */

template <typename T>
int suiteB(bool rowmajor, bool lower, bool trans, int nm, int n, int k, T alpha, T beta)
{
    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = cqr::detail::vlen_for_format<T>(fmt);
    const T eps = std::numeric_limits<T>::epsilon();

    const MKL_LAYOUT ml = rowmajor ? MKL_ROW_MAJOR : MKL_COL_MAJOR;
    const MKL_UPLO mu = lower ? MKL_LOWER : MKL_UPPER;
    const MKL_TRANSPOSE mtr = trans ? MKL_TRANS : MKL_NOTRANS;
    /* gemm forms the same product: A*A^T via (NoTrans, Trans); A^T*A via
     * (Trans, NoTrans). */
    const MKL_TRANSPOSE ga = trans ? MKL_TRANS : MKL_NOTRANS;
    const MKL_TRANSPOSE gb = trans ? MKL_NOTRANS : MKL_TRANS;

    const int Arows = trans ? k : n, Acols = trans ? n : k;
    const int ldA = rowmajor ? Acols : Arows;
    const int ldC = n;
    const size_t sA = (size_t)Arows * Acols, sC = (size_t)n * n;

    std::vector<T> A(nm * sA), C(nm * sC);
    std::generate(A.begin(), A.end(), frand<T>);
    std::generate(C.begin(), C.end(), frand<T>);

    auto Ap = batch_ptrs<const T>(A.data(), nm, sA);
    auto Cp = batch_ptrs<const T>(C.data(), nm, sC);
    auto ap = cqr::detail::mkl_alloc_bytes<T>(compact_size<T>(Arows, Acols, fmt, nm));
    auto cs = cqr::detail::mkl_alloc_bytes<T>(compact_size<T>(n, n, fmt, nm)); /* syrk */
    auto cg = cqr::detail::mkl_alloc_bytes<T>(compact_size<T>(n, n, fmt, nm)); /* gemm */
    pack(ml, Arows, Acols, Ap.data(), ldA, ap.get(), ldA, fmt, nm);
    pack(ml, n, n, Cp.data(), ldC, cs.get(), ldC, fmt, nm);
    pack(ml, n, n, Cp.data(), ldC, cg.get(), ldC, fmt, nm);

    syrk_compact(ml, mu, mtr, n, k, alpha, ap.get(), ldA, beta, cs.get(), ldC, fmt, nm);
    gemm_compact(ml, ga, gb, n, n, k, alpha, ap.get(), ldA, ap.get(), ldA, beta, cg.get(),
                 ldC, fmt, nm);

    std::vector<T> Sout(nm * sC), Gout(nm * sC);
    auto Sp = batch_ptrs<T>(Sout.data(), nm, sC);
    auto Gp = batch_ptrs<T>(Gout.data(), nm, sC);
    unpack(ml, n, n, Sp.data(), ldC, cs.get(), ldC, fmt, nm);
    unpack(ml, n, n, Gp.data(), ldC, cg.get(), ldC, fmt, nm);

    double worst = 0;
    for (int v = 0; v < nm; ++v) {
        const T *sp = Sout.data() + v * sC, *gp = Gout.data() + v * sC;
        /* compare only the triangle syrk wrote (gemm filled the whole matrix) */
        double diff = tri_maxdiff(sp, gp, n, ldC, lower, rowmajor);
        worst = std::max(worst, diff / std::max(maxabs(gp, sC), 1e-300));
    }
    const double rtol = 32.0 * (k + 1) * (double)eps;
    const bool ok = (worst <= rtol);
    std::printf("  [B:gemm ] %s%s uplo=%c trans=%c V=%-2d nm=%-2d n=%-3d k=%-3d "
                "a=%+.1f b=%+.1f | rel %.2e (rtol %.1e) %s\n",
                pname(std::is_same<T, double>::value), rowmajor ? "/row" : "/col",
                lower ? 'L' : 'U', trans ? 'T' : 'N', V, nm, n, k, (double)alpha,
                (double)beta, worst, rtol, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

/* ------------------------------------------------------------------ */
/* Suite C: end-to-end Cholesky QR, no MKL compute kernel             */
/*   G = A^T A (syrk, upper) -> R = chol(G) (potrf, upper) ->          */
/*   Q = A R^{-1} (trsm, right/upper) ; check Q R = A and Q^T Q = I.   */
/* ------------------------------------------------------------------ */

int suiteC(int nm, int m, int n)
{
    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = cqr::detail::vlen_for_format<double>(fmt);
    const double eps = std::numeric_limits<double>::epsilon();

    const size_t sA = (size_t)m * n, sG = (size_t)n * n;
    std::vector<double> A(nm * sA);
    /* tall random A (m >= n) is well-conditioned */
    std::generate(A.begin(), A.end(), frand<double>);

    auto Ap = batch_ptrs<const double>(A.data(), nm, sA);
    auto a_buf =
        cqr::detail::mkl_alloc_bytes<double>(mkl_dget_size_compact(m, n, fmt, nm));
    auto g_buf =
        cqr::detail::mkl_alloc_bytes<double>(mkl_dget_size_compact(n, n, fmt, nm));
    double *ap = a_buf.get(), *gp = g_buf.get();
    mkl_dgepack_compact(MKL_COL_MAJOR, m, n, Ap.data(), m, ap, m, fmt, nm);

    MKL_INT info = 99;

    /* 1. Gram matrix G = A^T A (upper triangle) via our compact syrk */
    cqr_mkl_dsyrk_compact(MKL_COL_MAJOR, MKL_UPPER, MKL_TRANS, n, m, 1.0, ap, m, 0.0, gp,
                          n, fmt, nm);

    /* 2. R = chol(G): G = R^T R, R upper (potrf reads the upper triangle syrk wrote) */
    cqr_mkl_dpotrf_compact(MKL_COL_MAJOR, MKL_UPPER, n, gp, n, &info, fmt, nm);

    /* 3. Q = A R^{-1}: solve Q R = A (right side, upper, no-trans), A -> Q in place */
    cqr_mkl_dtrsm_compact(MKL_COL_MAJOR, MKL_RIGHT, MKL_UPPER, MKL_NOTRANS, MKL_NONUNIT,
                          m, n, 1.0, gp, n, ap, m, fmt, nm);

    std::vector<double> Q(nm * sA), R(nm * sG);
    auto Qp = batch_ptrs<double>(Q.data(), nm, sA);
    auto Rp = batch_ptrs<double>(R.data(), nm, sG);
    mkl_dgeunpack_compact(MKL_COL_MAJOR, m, n, Qp.data(), m, ap, m, fmt, nm);
    mkl_dgeunpack_compact(MKL_COL_MAJOR, n, n, Rp.data(), n, gp, n, fmt, nm);

    double worst_recon = 0, worst_orth = 0;
    std::vector<double> QR(sA), QtQ(sG);
    for (int v = 0; v < nm; ++v) {
        const double *Av = A.data() + v * sA, *Qv = Q.data() + v * sA;
        const double *Rv = R.data() + v * sG;
        /* reconstruction Q R (R upper triangular) vs the original A */
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) {
                double s = 0;
                for (int l = 0; l <= j; ++l) /* R upper: R(l,j) nonzero for l <= j */
                    s += Qv[i + (size_t)l * m] * Rv[l + (size_t)j * n];
                QR[i + (size_t)j * m] = s;
            }
        worst_recon = std::max(worst_recon, max_abs_diff(QR.data(), Av, sA) /
                                                std::max(norm1(Av, m, n), 1e-300));
        /* orthogonality Q^T Q vs I */
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                double s = 0;
                for (int l = 0; l < m; ++l)
                    s += Qv[l + (size_t)i * m] * Qv[l + (size_t)j * m];
                QtQ[i + (size_t)j * n] = s - (i == j ? 1.0 : 0.0);
            }
        worst_orth = std::max(worst_orth, maxabs(QtQ.data(), sG));
    }
    /* reconstruction is backward stable (independent of conditioning); the
     * Cholesky-QR orthogonality error grows like cond(A)^2 * eps, which stays
     * small for these tall, well-conditioned random inputs. */
    const double rtol_recon = 50.0 * n * eps;
    const double rtol_orth = 1e3 * n * eps;
    const bool ok =
        (info == 0) && (worst_recon <= rtol_recon) && (worst_orth <= rtol_orth);
    std::printf("  [C:cqr  ] V=%-2d nm=%-2d m=%-3d n=%-3d | recon %.2e (rtol %.1e) "
                "orth %.2e (rtol %.1e) info=%d %s\n",
                V, nm, m, n, worst_recon, rtol_recon, worst_orth, rtol_orth, (int)info,
                ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

/* Run Suites A and B over the full feature matrix for one precision T. */
template <typename T> int run_precision()
{
    int fails = 0;
    /* (nm, n, k), spanning ordinary shapes and corner cases */
    const int shapes[][3] = {
        {8, 9, 4},   /* baseline; one full group at V=8 (double)         */
        {16, 5, 7},  /* wide factor (k > n), nm a multiple of V          */
        {11, 12, 3}, /* tall factor (k < n), padded partial last group   */
        {1, 5, 3},   /* nm = 1: smallest batch (a single partial group)  */
        {5, 1, 4},   /* n = 1: degenerate single-element triangle        */
        {4, 6, 1},   /* k = 1: rank-1 update                             */
        {17, 4, 4},  /* nm one past a full group; n = k = the JB=4 width  */
    };
    /* (alpha, beta): identity, scaled-accumulate, beta=0 overwrite, alpha=0. */
    const T coeffs[][2] = {
        {T(1), T(0)},
        {T(2), T(0.5)},
        {T(-1.5), T(1)},
        {T(0), T(0.5)},
    };

    for (bool rowmajor : {false, true})
        for (bool lower : {false, true})
            for (bool trans : {false, true})
                for (auto &[nm, n, k] : shapes)
                    for (auto &[alpha, beta] : coeffs) {
                        fails += suiteA<T>(rowmajor, lower, trans, nm, n, k, alpha, beta);
                        fails += suiteB<T>(rowmajor, lower, trans, nm, n, k, alpha, beta);
                    }
    return fails;
}

} /* anonymous namespace */

int main()
{
    std::printf("MKL compact format = %d, V(double) = %d, V(float) = %d\n",
                (int)mkl_get_format_compact(),
                cqr::detail::vlen_for_format<double>(mkl_get_format_compact()),
                cqr::detail::vlen_for_format<float>(mkl_get_format_compact()));

    int fails = 0;
    std::printf("-- double (Suites A vs cblas, B vs gemm_compact) --\n");
    fails += run_precision<double>();
    std::printf("-- single (Suites A vs cblas, B vs gemm_compact) --\n");
    fails += run_precision<float>();

    std::printf("-- Cholesky QR end-to-end (Suite C: syrk -> potrf -> trsm) --\n");
    fails += suiteC(8, 32, 5);
    fails += suiteC(8, 64, 8);
    fails += suiteC(7, 40, 6); /* padded partial last group */

    if (fails) {
        std::printf("\n%d CHECK(S) FAILED\n", fails);
        return 1;
    }
    std::printf("\nall checks passed\n");
    return 0;
}
