/* test_cqr_syrk_ext.cpp
 *
 * Validation of cqr_mkl_?syrk_compact against real Intel MKL, in two
 * independent ways, over the full feature matrix (precision x layout x uplo x
 * trans) and a range of alpha/beta and batch shapes (including a padded
 * partial last group). Both suites pack with mkl_?gepack_compact, run the
 * routine under test on the compact buffers, and unpack to compare.
 *
 * Suite A -- vs dense cblas_?syrk (per matrix):
 *   The compact result is checked, element for element over the *whole* n x n
 *   C, against a per-matrix reference computed by cblas_?syrk from the same
 *   seed C. Because ?syrk writes only the uplo triangle, this simultaneously
 *   verifies that the active triangle is correct and that the opposite
 *   triangle is left untouched (not clobbered).
 *
 * Suite B -- vs mkl_?gemm_compact (compact, same pipeline):
 *   The full product alpha*A*op(A) + beta*C is formed with mkl_?gemm_compact
 *   (transb = T for A*A^T, transa = T for A^T*A) into a second copy of the
 *   packed C. Since gemm writes the *entire* matrix while syrk writes only one
 *   triangle, the comparison is restricted to the active uplo triangle.
 *
 * Build: needs Intel MKL (headers + libmkl_rt); wired up by CMakeLists.txt.
 */

#include <mkl.h>
#include <mkl_compact.h>

#include "cqr_mkl_ext.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>
#include <type_traits>

namespace {

/* ------------------------------------------------------------------ */
/* Precision-dispatched MKL / CBLAS wrappers. Overloaded on the scalar */
/* pointer type where the argument disambiguates; templated for the    */
/* size query, which carries no pointer.                               */
/* ------------------------------------------------------------------ */

template <typename T> MKL_INT cmp_size(MKL_INT m, MKL_INT n, MKL_COMPACT_PACK f, MKL_INT nm);
template <> MKL_INT cmp_size<double>(MKL_INT m, MKL_INT n, MKL_COMPACT_PACK f, MKL_INT nm)
{ return mkl_dget_size_compact(m, n, f, nm); }
template <> MKL_INT cmp_size<float>(MKL_INT m, MKL_INT n, MKL_COMPACT_PACK f, MKL_INT nm)
{ return mkl_sget_size_compact(m, n, f, nm); }

void gepack(MKL_LAYOUT l, MKL_INT m, MKL_INT n, const double *const *a, MKL_INT lda,
            double *ap, MKL_INT ldap, MKL_COMPACT_PACK f, MKL_INT nm)
{ mkl_dgepack_compact(l, m, n, const_cast<const double **>(a), lda, ap, ldap, f, nm); }
void gepack(MKL_LAYOUT l, MKL_INT m, MKL_INT n, const float *const *a, MKL_INT lda,
            float *ap, MKL_INT ldap, MKL_COMPACT_PACK f, MKL_INT nm)
{ mkl_sgepack_compact(l, m, n, const_cast<const float **>(a), lda, ap, ldap, f, nm); }

void geunpack(MKL_LAYOUT l, MKL_INT m, MKL_INT n, double **a, MKL_INT lda,
              const double *ap, MKL_INT ldap, MKL_COMPACT_PACK f, MKL_INT nm)
{ mkl_dgeunpack_compact(l, m, n, a, lda, ap, ldap, f, nm); }
void geunpack(MKL_LAYOUT l, MKL_INT m, MKL_INT n, float **a, MKL_INT lda,
              const float *ap, MKL_INT ldap, MKL_COMPACT_PACK f, MKL_INT nm)
{ mkl_sgeunpack_compact(l, m, n, a, lda, ap, ldap, f, nm); }

void gemm_cmp(MKL_LAYOUT l, MKL_TRANSPOSE ta, MKL_TRANSPOSE tb,
              MKL_INT m, MKL_INT n, MKL_INT k, double alpha,
              const double *ap, MKL_INT ldap, const double *bp, MKL_INT ldbp,
              double beta, double *cp, MKL_INT ldcp, MKL_COMPACT_PACK f, MKL_INT nm)
{ mkl_dgemm_compact(l, ta, tb, m, n, k, alpha, ap, ldap, bp, ldbp, beta, cp, ldcp, f, nm); }
void gemm_cmp(MKL_LAYOUT l, MKL_TRANSPOSE ta, MKL_TRANSPOSE tb,
              MKL_INT m, MKL_INT n, MKL_INT k, float alpha,
              const float *ap, MKL_INT ldap, const float *bp, MKL_INT ldbp,
              float beta, float *cp, MKL_INT ldcp, MKL_COMPACT_PACK f, MKL_INT nm)
{ mkl_sgemm_compact(l, ta, tb, m, n, k, alpha, ap, ldap, bp, ldbp, beta, cp, ldcp, f, nm); }

void ref_syrk(CBLAS_LAYOUT l, CBLAS_UPLO u, CBLAS_TRANSPOSE t, MKL_INT n, MKL_INT k,
              double alpha, const double *a, MKL_INT lda, double beta, double *c, MKL_INT ldc)
{ cblas_dsyrk(l, u, t, n, k, alpha, a, lda, beta, c, ldc); }
void ref_syrk(CBLAS_LAYOUT l, CBLAS_UPLO u, CBLAS_TRANSPOSE t, MKL_INT n, MKL_INT k,
              float alpha, const float *a, MKL_INT lda, float beta, float *c, MKL_INT ldc)
{ cblas_ssyrk(l, u, t, n, k, alpha, a, lda, beta, c, ldc); }

void uut_syrk(MKL_LAYOUT l, MKL_UPLO u, MKL_TRANSPOSE t, MKL_INT n, MKL_INT k,
              double alpha, const double *ap, MKL_INT ldap, double beta,
              double *cp, MKL_INT ldcp, MKL_COMPACT_PACK f, MKL_INT nm)
{ cqr_mkl_dsyrk_compact(l, u, t, n, k, alpha, ap, ldap, beta, cp, ldcp, f, nm); }
void uut_syrk(MKL_LAYOUT l, MKL_UPLO u, MKL_TRANSPOSE t, MKL_INT n, MKL_INT k,
              float alpha, const float *ap, MKL_INT ldap, float beta,
              float *cp, MKL_INT ldcp, MKL_COMPACT_PACK f, MKL_INT nm)
{ cqr_mkl_ssyrk_compact(l, u, t, n, k, alpha, ap, ldap, beta, cp, ldcp, f, nm); }

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

double frand() { return 2.0 * std::rand() / (double)RAND_MAX - 1.0; }

/* matrix base pointers into one contiguous batch buffer (matrix v at v*stride) */
template <class T>
std::vector<T *> batch_ptrs(T *base, int nm, size_t stride)
{
    std::vector<T *> p(nm);
    for (int v = 0; v < nm; ++v) p[v] = base + (size_t)v * stride;
    return p;
}

/* element offset of C(i,j) for an n x n matrix in the given layout */
inline size_t cidx(int i, int j, int ld, bool rowmajor)
{ return rowmajor ? (size_t)i * ld + j : (size_t)j * ld + i; }

template <class T>
double maxabs(const T *a, size_t n)
{ double d = 0; for (size_t i = 0; i < n; ++i) d = std::max(d, std::abs((double)a[i])); return d; }

template <class T>
double maxdiff(const T *a, const T *b, size_t n)
{ double d = 0; for (size_t i = 0; i < n; ++i) d = std::max(d, std::abs((double)a[i] - (double)b[i])); return d; }

/* max |X - Y| over the active uplo triangle of an n x n matrix */
template <class T>
double tri_maxdiff(const T *X, const T *Y, int n, bool lower, int ld, bool rowmajor)
{
    double d = 0;
    for (int i = 0; i < n; ++i) {
        const int jlo = lower ? 0 : i, jhi = lower ? i + 1 : n;
        for (int j = jlo; j < jhi; ++j) {
            const size_t o = cidx(i, j, ld, rowmajor);
            d = std::max(d, std::abs((double)X[o] - (double)Y[o]));
        }
    }
    return d;
}

const char *pname(bool dbl) { return dbl ? "d" : "s"; }

/* ------------------------------------------------------------------ */
/* Suite A: cqr_mkl_?syrk_compact vs per-matrix dense cblas_?syrk      */
/* ------------------------------------------------------------------ */

template <typename T>
int suiteA(bool rowmajor, bool lower, bool trans,
           int nm, int n, int k, T alpha, T beta)
{
    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = cqr::detail::vlen_for_format<T>(fmt);
    const T   eps = std::numeric_limits<T>::epsilon();

    const MKL_LAYOUT     ml  = rowmajor ? MKL_ROW_MAJOR : MKL_COL_MAJOR;
    const CBLAS_LAYOUT   cl  = rowmajor ? CblasRowMajor : CblasColMajor;
    const MKL_UPLO       mu  = lower ? MKL_LOWER : MKL_UPPER;
    const CBLAS_UPLO     cu  = lower ? CblasLower : CblasUpper;
    const MKL_TRANSPOSE  mtr = trans ? MKL_TRANS : MKL_NOTRANS;
    const CBLAS_TRANSPOSE ctr = trans ? CblasTrans : CblasNoTrans;

    /* A is n x k (notrans) or k x n (trans); leading dim along the stored axis */
    const int Arows = trans ? k : n, Acols = trans ? n : k;
    const int ldA   = rowmajor ? Acols : Arows;   /* dense == compact leading dim */
    const int ldC   = n;                           /* C is n x n in both layouts   */
    const size_t sA = (size_t)Arows * Acols, sC = (size_t)n * n;

    std::vector<T> A(nm * sA), C(nm * sC), Cref(nm * sC), Cout(nm * sC);
    for (int v = 0; v < nm; ++v) {
        T *Av = A.data() + v * sA, *Cv = C.data() + v * sC, *Rv = Cref.data() + v * sC;
        for (size_t i = 0; i < sA; ++i) Av[i] = (T)frand();
        for (size_t i = 0; i < sC; ++i) Cv[i] = (T)frand();
        std::copy(Cv, Cv + sC, Rv);
        ref_syrk(cl, cu, ctr, n, k, alpha, Av, ldA, beta, Rv, ldC);
    }

    auto Ap = batch_ptrs<const T>(A.data(), nm, sA);
    auto Cp = batch_ptrs<const T>(C.data(), nm, sC);

    auto ap_buf = cqr::detail::mkl_alloc_bytes<T>(cmp_size<T>(Arows, Acols, fmt, nm));
    auto cp_buf = cqr::detail::mkl_alloc_bytes<T>(cmp_size<T>(n, n, fmt, nm));
    T *ap = ap_buf.get(), *cp = cp_buf.get();

    gepack(ml, Arows, Acols, Ap.data(), ldA, ap, ldA, fmt, nm);
    gepack(ml, n, n, Cp.data(), ldC, cp, ldC, fmt, nm);

    uut_syrk(ml, mu, mtr, n, k, alpha, ap, ldA, beta, cp, ldC, fmt, nm);

    auto Op = batch_ptrs<T>(Cout.data(), nm, sC);
    geunpack(ml, n, n, Op.data(), ldC, cp, ldC, fmt, nm);

    double worst = 0;
    for (int v = 0; v < nm; ++v) {
        const T *Co = Cout.data() + v * sC, *Rv = Cref.data() + v * sC;
        /* whole matrix: active triangle correctness + opposite triangle intact */
        double rel = maxdiff(Co, Rv, sC) / std::max(maxabs(Rv, sC), 1e-300);
        worst = std::max(worst, rel);
    }
    const double rtol = 32.0 * (k + 1) * (double)eps;
    bool ok = (worst <= rtol);
    std::printf("  [A:cblas] %s%s uplo=%c trans=%c V=%-2d nm=%-2d n=%-3d k=%-3d a=%+.1f b=%+.1f | rel %.2e (rtol %.2e) %s\n",
                pname(std::is_same<T, double>::value), rowmajor ? "/row" : "/col",
                lower ? 'L' : 'U', trans ? 'T' : 'N', V, nm, n, k,
                (double)alpha, (double)beta, worst, rtol, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

/* ------------------------------------------------------------------ */
/* Suite B: cqr_mkl_?syrk_compact vs mkl_?gemm_compact (triangle only) */
/* ------------------------------------------------------------------ */

template <typename T>
int suiteB(bool rowmajor, bool lower, bool trans,
           int nm, int n, int k, T alpha, T beta)
{
    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = cqr::detail::vlen_for_format<T>(fmt);
    const T   eps = std::numeric_limits<T>::epsilon();

    const MKL_LAYOUT    ml  = rowmajor ? MKL_ROW_MAJOR : MKL_COL_MAJOR;
    const MKL_UPLO      mu  = lower ? MKL_LOWER : MKL_UPPER;
    const MKL_TRANSPOSE mtr = trans ? MKL_TRANS : MKL_NOTRANS;
    /* gemm forms the same product: C = alpha*op(A)*op(A)^... For A*A^T use
     * (NoTrans, Trans); for A^T*A use (Trans, NoTrans). */
    const MKL_TRANSPOSE ga = trans ? MKL_TRANS   : MKL_NOTRANS;
    const MKL_TRANSPOSE gb = trans ? MKL_NOTRANS : MKL_TRANS;

    const int Arows = trans ? k : n, Acols = trans ? n : k;
    const int ldA   = rowmajor ? Acols : Arows;
    const int ldC   = n;
    const size_t sA = (size_t)Arows * Acols, sC = (size_t)n * n;

    std::vector<T> A(nm * sA), C(nm * sC), Csyrk(nm * sC), Cgemm(nm * sC);
    for (int v = 0; v < nm; ++v) {
        T *Av = A.data() + v * sA, *Cv = C.data() + v * sC;
        for (size_t i = 0; i < sA; ++i) Av[i] = (T)frand();
        for (size_t i = 0; i < sC; ++i) Cv[i] = (T)frand();
    }

    auto Ap = batch_ptrs<const T>(A.data(), nm, sA);
    auto Cp = batch_ptrs<const T>(C.data(), nm, sC);

    auto ap_buf  = cqr::detail::mkl_alloc_bytes<T>(cmp_size<T>(Arows, Acols, fmt, nm));
    auto cs_buf  = cqr::detail::mkl_alloc_bytes<T>(cmp_size<T>(n, n, fmt, nm));
    auto cg_buf  = cqr::detail::mkl_alloc_bytes<T>(cmp_size<T>(n, n, fmt, nm));
    T *ap = ap_buf.get(), *cs = cs_buf.get(), *cg = cg_buf.get();

    gepack(ml, Arows, Acols, Ap.data(), ldA, ap, ldA, fmt, nm);
    gepack(ml, n, n, Cp.data(), ldC, cs, ldC, fmt, nm);   /* C copy for syrk */
    gepack(ml, n, n, Cp.data(), ldC, cg, ldC, fmt, nm);   /* C copy for gemm */

    uut_syrk(ml, mu, mtr, n, k, alpha, ap, ldA, beta, cs, ldC, fmt, nm);
    gemm_cmp(ml, ga, gb, n, n, k, alpha, ap, ldA, ap, ldA, beta, cg, ldC, fmt, nm);

    std::vector<T> Sout(nm * sC), Gout(nm * sC);
    auto Sp = batch_ptrs<T>(Sout.data(), nm, sC);
    auto Gp = batch_ptrs<T>(Gout.data(), nm, sC);
    geunpack(ml, n, n, Sp.data(), ldC, cs, ldC, fmt, nm);
    geunpack(ml, n, n, Gp.data(), ldC, cg, ldC, fmt, nm);

    double worst = 0;
    for (int v = 0; v < nm; ++v) {
        const T *Sv = Sout.data() + v * sC, *Gv = Gout.data() + v * sC;
        /* compare only the triangle syrk wrote (gemm filled the whole matrix) */
        double rel = tri_maxdiff(Sv, Gv, n, lower, ldC, rowmajor) /
                     std::max(maxabs(Gv, sC), 1e-300);
        worst = std::max(worst, rel);
    }
    const double rtol = 32.0 * (k + 1) * (double)eps;
    bool ok = (worst <= rtol);
    std::printf("  [B:gemm ] %s%s uplo=%c trans=%c V=%-2d nm=%-2d n=%-3d k=%-3d a=%+.1f b=%+.1f | rel %.2e (rtol %.2e) %s\n",
                pname(std::is_same<T, double>::value), rowmajor ? "/row" : "/col",
                lower ? 'L' : 'U', trans ? 'T' : 'N', V, nm, n, k,
                (double)alpha, (double)beta, worst, rtol, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

/* Run both suites over the full feature matrix for one precision T. */
template <typename T>
int run_precision()
{
    int fails = 0;
    /* (nm, n, k); the nm=11 row gives a padded partial last group */
    const int shapes[][3] = { {8, 9, 4}, {16, 5, 7}, {11, 12, 3} };
    /* (alpha, beta): identity, scaled-accumulate, and the beta=0 overwrite */
    const T coeffs[][2] = { {T(1), T(0)}, {T(2), T(0.5)}, {T(-1.5), T(1)} };

    for (bool rowmajor : {false, true})
        for (bool lower : {false, true})
            for (bool trans : {false, true})
                for (auto &s : shapes)
                    for (auto &c : coeffs) {
                        fails += suiteA<T>(rowmajor, lower, trans, s[0], s[1], s[2], c[0], c[1]);
                        fails += suiteB<T>(rowmajor, lower, trans, s[0], s[1], s[2], c[0], c[1]);
                    }
    return fails;
}

} /* anonymous namespace */

int main()
{
    std::srand(42);
    std::printf("MKL compact format = %d, V(double) = %d, V(float) = %d\n",
                (int)mkl_get_format_compact(),
                cqr::detail::vlen_for_format<double>(mkl_get_format_compact()),
                cqr::detail::vlen_for_format<float>(mkl_get_format_compact()));

    int fails = 0;
    std::printf("-- double --\n"); fails += run_precision<double>();
    std::printf("-- single --\n"); fails += run_precision<float>();

    if (fails) { std::printf("\n%d CHECK(S) FAILED\n", fails); return 1; }
    std::printf("\nall checks passed\n");
    return 0;
}
