/* test_cqr_syrk_ext.cpp
 *
 * Validation of cqr_mkl_?syrk_compact against real Intel MKL, in two
 * independent ways, over the full feature matrix (precision x layout x uplo x
 * trans) and a range of alpha/beta and batch shapes (including a padded
 * partial last group). Both suites pack the inputs into MKL Compact format,
 * run the routine under test on the compact buffers, and unpack to compare.
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
#include <cmath>
#include <limits>
#include <vector>
#include <span>
#include <random>
#include <numeric>
#include <algorithm>
#include <type_traits>

namespace {

/* ------------------------------------------------------------------ */
/* Precision-dispatched MKL / CBLAS wrappers. Each names the generic    */
/* operation; the s/d variant is picked by overload on the scalar       */
/* pointer type, or by explicit specialization for the size query (which */
/* carries no pointer to overload on).                                  */
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
    mkl_dgepack_compact(l, m, n, const_cast<const double **>(a), lda, ap, ldap, f, nm);
}
void pack(MKL_LAYOUT l, MKL_INT m, MKL_INT n, const float *const *a, MKL_INT lda,
          float *ap, MKL_INT ldap, MKL_COMPACT_PACK f, MKL_INT nm)
{
    mkl_sgepack_compact(l, m, n, const_cast<const float **>(a), lda, ap, ldap, f, nm);
}

void unpack(MKL_LAYOUT l, MKL_INT m, MKL_INT n, double **a, MKL_INT lda,
            const double *ap, MKL_INT ldap, MKL_COMPACT_PACK f, MKL_INT nm)
{
    mkl_dgeunpack_compact(l, m, n, a, lda, ap, ldap, f, nm);
}
void unpack(MKL_LAYOUT l, MKL_INT m, MKL_INT n, float **a, MKL_INT lda,
            const float *ap, MKL_INT ldap, MKL_COMPACT_PACK f, MKL_INT nm)
{
    mkl_sgeunpack_compact(l, m, n, a, lda, ap, ldap, f, nm);
}

void gemm_compact(MKL_LAYOUT l, MKL_TRANSPOSE ta, MKL_TRANSPOSE tb,
                  MKL_INT m, MKL_INT n, MKL_INT k, double alpha,
                  const double *ap, MKL_INT ldap, const double *bp, MKL_INT ldbp,
                  double beta, double *cp, MKL_INT ldcp, MKL_COMPACT_PACK f, MKL_INT nm)
{
    mkl_dgemm_compact(l, ta, tb, m, n, k, alpha, ap, ldap, bp, ldbp, beta, cp, ldcp, f, nm);
}
void gemm_compact(MKL_LAYOUT l, MKL_TRANSPOSE ta, MKL_TRANSPOSE tb,
                  MKL_INT m, MKL_INT n, MKL_INT k, float alpha,
                  const float *ap, MKL_INT ldap, const float *bp, MKL_INT ldbp,
                  float beta, float *cp, MKL_INT ldcp, MKL_COMPACT_PACK f, MKL_INT nm)
{
    mkl_sgemm_compact(l, ta, tb, m, n, k, alpha, ap, ldap, bp, ldbp, beta, cp, ldcp, f, nm);
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
                  double alpha, const double *ap, MKL_INT ldap, double beta,
                  double *cp, MKL_INT ldcp, MKL_COMPACT_PACK f, MKL_INT nm)
{
    cqr_mkl_dsyrk_compact(l, u, t, n, k, alpha, ap, ldap, beta, cp, ldcp, f, nm);
}
void syrk_compact(MKL_LAYOUT l, MKL_UPLO u, MKL_TRANSPOSE t, MKL_INT n, MKL_INT k,
                  float alpha, const float *ap, MKL_INT ldap, float beta,
                  float *cp, MKL_INT ldcp, MKL_COMPACT_PACK f, MKL_INT nm)
{
    cqr_mkl_ssyrk_compact(l, u, t, n, k, alpha, ap, ldap, beta, cp, ldcp, f, nm);
}

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

std::mt19937 rng;   /* seeded in main for reproducibility */

template <class T>
void fill_random(std::vector<T> &v)
{
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::ranges::generate(v, [&] { return static_cast<T>(dist(rng)); });
}

/* matrix base pointers into one contiguous batch buffer (matrix v at v*stride) */
template <class T>
std::vector<T *> batch_ptrs(T *base, int nm, size_t stride)
{
    std::vector<T *> p(nm);
    for (int v = 0; v < nm; ++v) p[v] = base + static_cast<size_t>(v) * stride;
    return p;
}

/* Minimal layout-aware view of a single dense matrix: (i,j) indexing that
 * hides the column-major vs row-major offset arithmetic. The layout is a
 * template parameter, so each access resolves at compile time (no per-element
 * branch) -- the two layouts are simply two instantiations. */
template <class T, bool RowMajor>
struct MatrixView {
    T  *const  data;
    const int  ld;
    T &operator()(int i, int j) const
    {
        if constexpr (RowMajor) return data[static_cast<size_t>(i) * ld + j];
        else                    return data[static_cast<size_t>(j) * ld + i];
    }
};

template <class T>
std::span<const T> cspan(const T *p, size_t n) { return {p, n}; }

/* index of the max-magnitude element (BLAS i?amax), precision-dispatched.
 * CBLAS i?amax is 0-based (unlike Fortran's 1-based), so the result indexes
 * the span directly. */
CBLAS_INDEX iamax(std::span<const float> x)
{
    return cblas_isamax(static_cast<MKL_INT>(x.size()), x.data(), 1);
}
CBLAS_INDEX iamax(std::span<const double> x)
{
    return cblas_idamax(static_cast<MKL_INT>(x.size()), x.data(), 1);
}

/* Reductions stay in the operand precision T (no float -> double promotion):
 * maxabs is exactly |x[i?amax]|; maxdiff has no single BLAS call, so it folds
 * the elementwise |a-b| in T. */
template <class T>
T maxabs(std::span<const T> a)
{
    if (a.empty()) return T(0);
    const CBLAS_INDEX i = iamax(a);
    return std::abs(a[i]);
}

template <class T>
T maxdiff(std::span<const T> a, std::span<const T> b)
{
    return std::transform_reduce(
        a.begin(), a.end(), b.begin(), T(0),
        [](T x, T y) { return std::max(x, y); },
        [](T x, T y) { return std::abs(x - y); });
}

/* max |X - Y| over the active uplo triangle of an n x n matrix */
template <class T, bool RowMajor>
T tri_maxdiff(const MatrixView<const T, RowMajor> &X,
              const MatrixView<const T, RowMajor> &Y, int n, bool lower)
{
    T d = 0;
    for (int i = 0; i < n; ++i) {
        const int jlo = lower ? 0 : i;
        const int jhi = lower ? i + 1 : n;
        for (int j = jlo; j < jhi; ++j)
            d = std::max(d, std::abs(X(i, j) - Y(i, j)));
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

    const MKL_LAYOUT      ml  = rowmajor ? MKL_ROW_MAJOR : MKL_COL_MAJOR;
    const CBLAS_LAYOUT    cl  = rowmajor ? CblasRowMajor : CblasColMajor;
    const MKL_UPLO        mu  = lower ? MKL_LOWER : MKL_UPPER;
    const CBLAS_UPLO      cu  = lower ? CblasLower : CblasUpper;
    const MKL_TRANSPOSE   mtr = trans ? MKL_TRANS : MKL_NOTRANS;
    const CBLAS_TRANSPOSE ctr = trans ? CblasTrans : CblasNoTrans;

    /* A is n x k (notrans) or k x n (trans); leading dim along the stored axis */
    const int Arows = trans ? k : n, Acols = trans ? n : k;
    const int ldA   = rowmajor ? Acols : Arows;   /* dense == compact leading dim */
    const int ldC   = n;                           /* C is n x n in both layouts   */
    const size_t sA = static_cast<size_t>(Arows) * Acols, sC = static_cast<size_t>(n) * n;

    /* A is a general rectangular operand; C is filled non-symmetrically on
     * purpose. syrk only ever references the active uplo triangle, so checking
     * the whole matrix below also confirms the opposite triangle is untouched. */
    std::vector<T> A(nm * sA), C(nm * sC), Cref(nm * sC), Cout(nm * sC);
    fill_random(A);
    fill_random(C);
    Cref = C;
    for (int v = 0; v < nm; ++v)
        syrk(cl, cu, ctr, n, k, alpha, A.data() + v * sA, ldA,
             beta, Cref.data() + v * sC, ldC);

    auto ap_buf = cqr::detail::mkl_alloc_bytes<T>(compact_size<T>(Arows, Acols, fmt, nm));
    auto cp_buf = cqr::detail::mkl_alloc_bytes<T>(compact_size<T>(n, n, fmt, nm));
    T *ap = ap_buf.get(), *cp = cp_buf.get();

    {
        auto Ap = batch_ptrs<const T>(A.data(), nm, sA);
        auto Cp = batch_ptrs<const T>(C.data(), nm, sC);
        pack(ml, Arows, Acols, Ap.data(), ldA, ap, ldA, fmt, nm);
        pack(ml, n, n, Cp.data(), ldC, cp, ldC, fmt, nm);
    }

    syrk_compact(ml, mu, mtr, n, k, alpha, ap, ldA, beta, cp, ldC, fmt, nm);

    {
        auto Op = batch_ptrs<T>(Cout.data(), nm, sC);
        unpack(ml, n, n, Op.data(), ldC, cp, ldC, fmt, nm);
    }

    double worst = 0;
    for (int v = 0; v < nm; ++v) {
        /* whole matrix: active triangle correctness + opposite triangle intact */
        double rel = static_cast<double>(maxdiff(cspan(Cout.data() + v * sC, sC),
                                                 cspan(Cref.data() + v * sC, sC))) /
                     std::max(static_cast<double>(maxabs(cspan(Cref.data() + v * sC, sC))),
                              1e-300);
        worst = std::max(worst, rel);
    }
    const double rtol = 32.0 * (k + 1) * static_cast<double>(eps);
    bool ok = (worst <= rtol);
    std::printf("  [A:cblas] %s%s uplo=%c trans=%c V=%-2d nm=%-2d n=%-3d k=%-3d a=%+.1f b=%+.1f | rel %.2e (rtol %.2e) %s\n",
                pname(std::is_same<T, double>::value), rowmajor ? "/row" : "/col",
                lower ? 'L' : 'U', trans ? 'T' : 'N', V, nm, n, k,
                static_cast<double>(alpha), static_cast<double>(beta),
                worst, rtol, ok ? "OK" : "FAIL");
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
    /* gemm forms the same product: for A*A^T use (NoTrans, Trans); for A^T*A
     * use (Trans, NoTrans). */
    const MKL_TRANSPOSE ga = trans ? MKL_TRANS   : MKL_NOTRANS;
    const MKL_TRANSPOSE gb = trans ? MKL_NOTRANS : MKL_TRANS;

    const int Arows = trans ? k : n, Acols = trans ? n : k;
    const int ldA   = rowmajor ? Acols : Arows;
    const int ldC   = n;
    const size_t sA = static_cast<size_t>(Arows) * Acols, sC = static_cast<size_t>(n) * n;

    /* A general, C non-symmetric (see Suite A); the triangle-only comparison
     * below is valid because syrk and gemm share the same packed C(i,j) there. */
    std::vector<T> A(nm * sA), C(nm * sC);
    fill_random(A);
    fill_random(C);

    auto ap_buf = cqr::detail::mkl_alloc_bytes<T>(compact_size<T>(Arows, Acols, fmt, nm));
    auto cs_buf = cqr::detail::mkl_alloc_bytes<T>(compact_size<T>(n, n, fmt, nm));
    auto cg_buf = cqr::detail::mkl_alloc_bytes<T>(compact_size<T>(n, n, fmt, nm));
    T *ap = ap_buf.get(), *cs = cs_buf.get(), *cg = cg_buf.get();

    {
        auto Ap = batch_ptrs<const T>(A.data(), nm, sA);
        auto Cp = batch_ptrs<const T>(C.data(), nm, sC);
        pack(ml, Arows, Acols, Ap.data(), ldA, ap, ldA, fmt, nm);
        pack(ml, n, n, Cp.data(), ldC, cs, ldC, fmt, nm);   /* C copy for syrk */
        pack(ml, n, n, Cp.data(), ldC, cg, ldC, fmt, nm);   /* C copy for gemm */
    }

    syrk_compact(ml, mu, mtr, n, k, alpha, ap, ldA, beta, cs, ldC, fmt, nm);
    gemm_compact(ml, ga, gb, n, n, k, alpha, ap, ldA, ap, ldA, beta, cg, ldC, fmt, nm);

    std::vector<T> Sout(nm * sC), Gout(nm * sC);
    {
        auto Sp = batch_ptrs<T>(Sout.data(), nm, sC);
        auto Gp = batch_ptrs<T>(Gout.data(), nm, sC);
        unpack(ml, n, n, Sp.data(), ldC, cs, ldC, fmt, nm);
        unpack(ml, n, n, Gp.data(), ldC, cg, ldC, fmt, nm);
    }

    double worst = 0;
    for (int v = 0; v < nm; ++v) {
        const T *sp = Sout.data() + v * sC;
        const T *gp = Gout.data() + v * sC;
        /* compare only the triangle syrk wrote (gemm filled the whole matrix);
         * the layout is a compile-time view parameter, picked once per matrix */
        T diff = rowmajor
            ? tri_maxdiff(MatrixView<const T, true >{sp, ldC},
                          MatrixView<const T, true >{gp, ldC}, n, lower)
            : tri_maxdiff(MatrixView<const T, false>{sp, ldC},
                          MatrixView<const T, false>{gp, ldC}, n, lower);
        double rel = static_cast<double>(diff) /
                     std::max(static_cast<double>(maxabs(cspan(gp, sC))), 1e-300);
        worst = std::max(worst, rel);
    }
    const double rtol = 32.0 * (k + 1) * static_cast<double>(eps);
    bool ok = (worst <= rtol);
    std::printf("  [B:gemm ] %s%s uplo=%c trans=%c V=%-2d nm=%-2d n=%-3d k=%-3d a=%+.1f b=%+.1f | rel %.2e (rtol %.2e) %s\n",
                pname(std::is_same<T, double>::value), rowmajor ? "/row" : "/col",
                lower ? 'L' : 'U', trans ? 'T' : 'N', V, nm, n, k,
                static_cast<double>(alpha), static_cast<double>(beta),
                worst, rtol, ok ? "OK" : "FAIL");
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
    rng.seed(42);
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
