/* test_cqr_gels_mkl.cpp
 *
 * Validation of cqr_mkl_?gels_compact against real Intel MKL and dense LAPACK,
 * through the genuine MKL Compact pipeline (mkl_?gepack_compact /
 * mkl_?geunpack_compact) -- the "hard correctness gate against standard dense
 * LAPACK equivalents" of cqr_mkl_dgels_compact_design.md section 7.
 *
 * Suite 1 (section 7.1) -- Solution vs dense LAPACKE_?gels:
 *   For every (layout, trans) over square, tall and wide A -- so all four
 *   least-squares / minimum-norm cases in both layouts -- a random batch is
 *   solved by cqr_mkl_?gels_compact and per matrix by LAPACKE_?gels on the same
 *   input. Gated: the forward error of X vs LAPACK's (relative, 100 * max(m,n)
 *   * eps); the defining property formed independently -- in the least-squares
 *   case the residual sums of squares in rows n..m-1 of B against
 *   ||B - op(A) X||^2, in the minimum-norm case the residual op(A) X - B; the
 *   workspace query; and the factorization left in ap and work, elementwise vs
 *   LAPACKE_?geqrf (m >= n) or LAPACKE_?gelqf (m < n), which pins the ?gelqf
 *   storage convention of the wide case.
 *
 * Suite 2 (section 7.2) -- Cross-check vs the three-step compact pipeline:
 *   On the same packed square batch with B = A X for a known X,
 *   cqr_mkl_?gels_compact and mkl_?geqrf_compact -> cqr_mkl_?ormqr_compact ->
 *   mkl_?trsm_compact must agree: the factorization in ap and the tau (gels's
 *   work vs geqrf's taup) elementwise at the cross-check tolerance, and the two
 *   solutions with each other and with X at the solve gate (100 * n * eps,
 *   relative), since the two triangular solves order their arithmetic
 *   differently.
 *
 * Build: needs Intel MKL (headers + libmkl_rt); wired up by CMakeLists.txt.
 *
 * Assisted-by: Claude:claude-fable-5
 */

#include "test_mkl_util.hpp" /* cqr_mkl<T>, mkl<T>, lapack<T> + the MKL-free helpers */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>

using namespace cqr::test;

namespace {

/* Element (i,j) of an r x c matrix stored in the given layout with leading
 * dimension ld. */
template <class T> T &at(T *M, int i, int j, bool rowmajor, int ld)
{
    return M[rowmajor ? (size_t)i * ld + j : i + (size_t)j * ld];
}

/* L1 (max column sum) norm of an r x c matrix in the given layout. */
template <class T> double norm1_layout(const T *M, int r, int c, bool rowmajor, int ld)
{
    double mx = 0;
    for (int j = 0; j < c; ++j) {
        double s = 0;
        for (int i = 0; i < r; ++i)
            s += std::abs(at(M, i, j, rowmajor, ld));
        mx = std::max(mx, s);
    }
    return mx;
}

/* ---------------- Suite 1: solution vs dense LAPACKE_?gels ------------- */

template <class T>
int suite1(MKL_LAYOUT layout, char trans, int nm, int m, int n, int nrhs)
{
    const double eps = std::numeric_limits<T>::epsilon();
    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = mkl<T>::vlen(fmt);
    const bool row = (layout == MKL_ROW_MAJOR);
    const bool tran = (trans == 'T');
    const bool tall = (m >= n);
    const int p = std::max(m, n), q = std::min(m, n);
    const bool overdet = (tall != tran);
    const int rows_op = tran ? n : m,
              cols_op = tran ? m : n; /* op(A): rows_op x cols_op */
    const int lap = row ? LAPACK_ROW_MAJOR : LAPACK_COL_MAJOR;
    const int lda = row ? n : m, ldb = row ? nrhs : p; /* dense == compact lds here */
    const size_t sA = (size_t)m * n, sB = (size_t)p * nrhs, sT = (size_t)q;

    /* random A with a boosted diagonal (well-conditioned op(A)) and random B,
     * as the buffers of the requested layout */
    std::vector<T> A(nm * sA), B(nm * sB);
    for (int v = 0; v < nm; ++v) {
        T *Av = A.data() + v * sA;
        for (size_t e = 0; e < sA; ++e)
            Av[e] = frand<T>();
        for (int i = 0; i < q; ++i)
            at(Av, i, i, row, lda) += T(2);
        for (size_t e = 0; e < sB; ++e)
            B[v * sB + e] = frand<T>();
    }

    /* pack, query the workspace, solve, unpack (X and the factorization) */
    auto Ap = batch_ptrs<const T>(A.data(), nm, sA);
    auto Bp = batch_ptrs<const T>(B.data(), nm, sB);
    auto ap_buf = cqr::detail::mkl_alloc_bytes<T>(mkl<T>::get_size(m, n, fmt, nm));
    auto bp_buf = cqr::detail::mkl_alloc_bytes<T>(mkl<T>::get_size(p, nrhs, fmt, nm));
    T *ap = ap_buf.get(), *bp = bp_buf.get();
    mkl<T>::gepack(layout, m, n, Ap.data(), lda, ap, lda, fmt, nm);
    mkl<T>::gepack(layout, p, nrhs, Bp.data(), ldb, bp, ldb, fmt, nm);

    MKL_INT info = 99;
    T wq = -1;
    cqr_mkl<T>::gels(layout, trans, m, n, nrhs, ap, lda, bp, ldb, &wq, -1, &info, fmt,
                     nm);
    const MKL_INT lwork_want = std::max<MKL_INT>(1, (MKL_INT)q * V * ((nm + V - 1) / V));
    const bool ok_query = (info == 0) && ((MKL_INT)wq == lwork_want);
    std::vector<T> work((size_t)std::max<MKL_INT>((MKL_INT)wq, 1));
    cqr_mkl<T>::gels(layout, trans, m, n, nrhs, ap, lda, bp, ldb, work.data(),
                     (MKL_INT)work.size(), &info, fmt, nm);

    std::vector<T> X(nm * sB), H(nm * sA), tau(nm * sT);
    auto Xp = batch_ptrs<T>(X.data(), nm, sB);
    auto Hp = batch_ptrs<T>(H.data(), nm, sA);
    auto Tp = batch_ptrs<T>(tau.data(), nm, sT);
    mkl<T>::geunpack(layout, p, nrhs, Xp.data(), ldb, bp, ldb, fmt, nm);
    mkl<T>::geunpack(layout, m, n, Hp.data(), lda, ap, lda, fmt, nm);
    mkl<T>::geunpack(MKL_COL_MAJOR, q, 1, Tp.data(), q, work.data(), q, fmt, nm);

    int fails = 0;
    if (info != 0) {
        ++fails;
        std::printf("    info = %ld (expected 0)\n", (long)info);
    }

    double worst_fwd = 0, worst_prop = 0, worst_fac = 0;
    std::vector<T> Aref(sA), Xref(sB), tref(sT), R((size_t)rows_op * nrhs);
    for (int v = 0; v < nm; ++v) {
        const T *Av = A.data() + v * sA, *Bv = B.data() + v * sB;
        const T *Xv = X.data() + v * sB, *Hv = H.data() + v * sA,
                *tv = tau.data() + v * sT;

        /* dense reference solve */
        std::copy(Av, Av + sA, Aref.begin());
        std::copy(Bv, Bv + sB, Xref.begin());
        const lapack_int li =
            lapack<T>::gels(lap, trans, m, n, nrhs, Aref.data(), lda, Xref.data(), ldb);
        if (li != 0) ++fails;

        /* forward error over the solution rows, relative to ||X_lapack||_1 */
        double fwd = 0;
        for (int j = 0; j < nrhs; ++j)
            for (int i = 0; i < cols_op; ++i)
                fwd = std::max(fwd, (double)std::abs(at(Xv, i, j, row, ldb) -
                                                     at(Xref.data(), i, j, row, ldb)));
        worst_fwd = std::max(
            worst_fwd,
            fwd / std::max(norm1_layout(Xref.data(), cols_op, nrhs, row, ldb), 1e-300));

        /* R = op(A) X - B, formed independently */
        for (int j = 0; j < nrhs; ++j)
            for (int i = 0; i < rows_op; ++i) {
                double s = 0;
                for (int l = 0; l < cols_op; ++l)
                    s +=
                        (double)(tran ? at(Av, l, i, row, lda) : at(Av, i, l, row, lda)) *
                        at(Xv, l, j, row, ldb);
                R[i + (size_t)j * rows_op] = (T)(s - at(Bv, i, j, row, ldb));
            }
        if (overdet) {
            /* rows cols_op..rows_op-1 of X carry the residual: their squared
             * column norms are the residual sums of squares (relative to
             * ||b_j||^2: a square system has no residual rows and none) */
            for (int j = 0; j < nrhs; ++j) {
                double rss = 0, rss_x = 0, bb = 0;
                for (int i = 0; i < rows_op; ++i) {
                    rss +=
                        (double)R[i + (size_t)j * rows_op] * R[i + (size_t)j * rows_op];
                    bb += (double)at(Bv, i, j, row, ldb) * at(Bv, i, j, row, ldb);
                }
                for (int i = cols_op; i < rows_op; ++i)
                    rss_x += (double)at(Xv, i, j, row, ldb) * at(Xv, i, j, row, ldb);
                worst_prop =
                    std::max(worst_prop, std::abs(rss - rss_x) / std::max(bb, 1e-300));
            }
        }
        else {
            /* op(A) X = B */
            const double scale =
                std::max(norm1_layout(Av, m, n, row, lda), 1e-300) *
                std::max(norm1_layout(Xv, cols_op, nrhs, row, ldb), 1e-300);
            for (size_t e = 0; e < (size_t)rows_op * nrhs; ++e)
                worst_prop = std::max(worst_prop, std::abs((double)R[e]) / scale);
        }

        /* the factorization left behind: QR (tall) or LQ (wide) of A, vs LAPACK */
        std::copy(Av, Av + sA, Aref.begin());
        if (tall)
            lapack<T>::geqrf(lap, m, n, Aref.data(), lda, tref.data());
        else
            lapack<T>::gelqf(lap, m, n, Aref.data(), lda, tref.data());
        const double fac = std::max(max_abs_diff(Hv, Aref.data(), sA),
                                    max_abs_diff(tv, tref.data(), sT));
        worst_fac =
            std::max(worst_fac, fac / std::max(norm1_layout(Av, m, n, row, lda), 1e-300));
    }

    /* forward error: a cond(op(A))-dependent quantity, well conditioned here;
     * the residual checks are backward-stable; the factorization compares an
     * unblocked kernel to LAPACK's blocked one (same math, different order) */
    const double rtol_fwd = 100.0 * p * eps, rtol_prop = 100.0 * p * eps;
    const double rtol_fac = 100.0 * p * eps;
    const bool ok = (worst_fwd <= rtol_fwd) && (worst_prop <= rtol_prop) &&
                    (worst_fac <= rtol_fac) && ok_query;
    fails += !ok;
    std::printf("  [suite1] %s trans=%c V=%-2d nm=%-2d m=%-3d n=%-3d nrhs=%d %-6s | fwd "
                "%.2e (%.1e) %s %.2e fac %.2e %s%s\n",
                row ? "row" : "col", trans, V, nm, m, n, nrhs,
                overdet ? "lstsq" : "minnrm", worst_fwd, rtol_fwd,
                overdet ? "rss" : "res", worst_prop, worst_fac, ok ? "OK" : "FAIL",
                ok_query ? "" : " (workspace query)");
    return fails;
}

/* ---------------- Suite 2: cross-check vs the three-step pipeline ------ */

template <class T> int suite2(int nm, int n, int nrhs)
{
    const double eps = std::numeric_limits<T>::epsilon();
    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = mkl<T>::vlen(fmt);
    const int m = n, k = n;
    const size_t sA = (size_t)n * n, sB = (size_t)n * nrhs;
    const std::vector<T> X = known_solution<T>(n, nrhs);

    std::vector<T> A(nm * sA), B(nm * sB);
    for (int v = 0; v < nm; ++v) {
        gen_boosted(A.data() + v * sA, n, n);
        matmul(n, nrhs, n, A.data() + v * sA, n, X.data(), n, B.data() + v * sB, n);
    }
    auto Ap = batch_ptrs<const T>(A.data(), nm, sA);
    auto Bp = batch_ptrs<const T>(B.data(), nm, sB);

    const MKL_INT sz_a = mkl<T>::get_size(m, n, fmt, nm);
    const MKL_INT sz_t = mkl<T>::get_size(k, 1, fmt, nm);
    const MKL_INT sz_b = mkl<T>::get_size(m, nrhs, fmt, nm);
    auto ap1 = cqr::detail::mkl_alloc_bytes<T>(sz_a),
         ap2 = cqr::detail::mkl_alloc_bytes<T>(sz_a);
    auto bp1 = cqr::detail::mkl_alloc_bytes<T>(sz_b),
         bp2 = cqr::detail::mkl_alloc_bytes<T>(sz_b);
    auto tp2 = cqr::detail::mkl_alloc_bytes<T>(sz_t);
    mkl<T>::gepack(MKL_COL_MAJOR, m, n, Ap.data(), m, ap1.get(), m, fmt, nm);
    mkl<T>::gepack(MKL_COL_MAJOR, m, n, Ap.data(), m, ap2.get(), m, fmt, nm);
    mkl<T>::gepack(MKL_COL_MAJOR, m, nrhs, Bp.data(), m, bp1.get(), m, fmt, nm);
    mkl<T>::gepack(MKL_COL_MAJOR, m, nrhs, Bp.data(), m, bp2.get(), m, fmt, nm);

    MKL_INT info = 0;

    /* 1. the one-shot solve; its work is the tau of the factorization */
    T wq;
    cqr_mkl<T>::gels(MKL_COL_MAJOR, 'N', m, n, nrhs, ap1.get(), m, bp1.get(), m, &wq, -1,
                     &info, fmt, nm);
    std::vector<T> work((size_t)std::max<MKL_INT>((MKL_INT)wq, 1));
    cqr_mkl<T>::gels(MKL_COL_MAJOR, 'N', m, n, nrhs, ap1.get(), m, bp1.get(), m,
                     work.data(), (MKL_INT)work.size(), &info, fmt, nm);

    /* 2. the pipeline: MKL's geqrf (its own workspace), cqr's ormqr, MKL's trsm */
    T wq_mkl;
    mkl<T>::geqrf(MKL_COL_MAJOR, m, n, ap2.get(), m, tp2.get(), &wq_mkl, -1, &info, fmt,
                  nm);
    std::vector<T> work_mkl((size_t)std::max<MKL_INT>((MKL_INT)wq_mkl, 1));
    mkl<T>::geqrf(MKL_COL_MAJOR, m, n, ap2.get(), m, tp2.get(), work_mkl.data(),
                  (MKL_INT)work_mkl.size(), &info, fmt, nm);
    T dummy;
    cqr_mkl<T>::ormqr(MKL_COL_MAJOR, 'L', 'T', m, nrhs, k, ap2.get(), m, tp2.get(),
                      bp2.get(), m, &dummy, 1, &info, fmt, nm);
    mkl<T>::trsm(MKL_COL_MAJOR, MKL_LEFT, MKL_UPPER, MKL_NOTRANS, MKL_NONUNIT, n, nrhs,
                 T(1), ap2.get(), m, bp2.get(), m, fmt, nm);

    /* the factorizations elementwise; the solutions relative to ||X||_1 */
    const size_t na = (size_t)sz_a / sizeof(T), nb = (size_t)sz_b / sizeof(T);
    const size_t nt = (size_t)sz_t / sizeof(T);
    const double nx = norm1(X.data(), n, nrhs);
    const double da = max_abs_diff(ap1.get(), ap2.get(), na);
    const double dt = max_abs_diff(work.data(), tp2.get(), std::min(nt, work.size()));
    const double dx = max_abs_diff(bp1.get(), bp2.get(), nb) / nx;
    std::vector<T> X1(nm * sB);
    auto X1p = batch_ptrs<T>(X1.data(), nm, sB);
    mkl<T>::geunpack(MKL_COL_MAJOR, n, nrhs, X1p.data(), n, bp1.get(), m, fmt, nm);
    double fwd = 0;
    for (int v = 0; v < nm; ++v)
        fwd = std::max(fwd, max_abs_diff(X1.data() + v * sB, X.data(), sB) / nx);
    const double tol = cross_tol<T>(), rtol = 100.0 * n * eps;
    const bool ok =
        (da <= tol && dt <= tol && work.size() >= nt) && (dx <= rtol) && (fwd <= rtol);
    std::printf(
        "  [suite2] V=%-2d nm=%-2d n=%-3d nrhs=%d | ap-mkl %.2e tau-mkl %.2e (tol "
        "%.0e) | X-pipe %.2e fwd %.2e (rtol %.1e) %s\n",
        V, nm, n, nrhs, da, dt, tol, dx, fwd, rtol, ok ? "OK" : "FAIL");
    return !ok;
}

} /* anonymous namespace */

template <class T> int run_suites()
{
    std::printf("\n== %s: MKL compact format = %d, V = %d ==\n", compact<T>::name,
                (int)mkl_get_format_compact(), mkl<T>::vlen(mkl_get_format_compact()));

    int fails = 0;

    /* Suite 1: every (layout, trans) over square, tall and wide shapes */
    for (MKL_LAYOUT lay : {MKL_COL_MAJOR, MKL_ROW_MAJOR})
        for (char tr : {'N', 'T'}) {
            fails += suite1<T>(lay, tr, 8, 32, 32, 5);  /* square */
            fails += suite1<T>(lay, tr, 8, 64, 20, 4);  /* tall */
            fails += suite1<T>(lay, tr, 8, 20, 64, 4);  /* wide */
            fails += suite1<T>(lay, tr, 11, 43, 17, 3); /* padded partial group */
            fails += suite1<T>(lay, tr, 7, 96, 128, 1); /* single RHS, padded group */
        }

    /* Suite 2: the square solve vs the three-step compact pipeline */
    fails += suite2<T>(8, 30, 5);
    fails += suite2<T>(16, 48, 4);
    fails += suite2<T>(7, 32, 6); /* padded partial group */

    return fails;
}

int main()
{
    const int fails = run_suites<double>() + run_suites<float>();
    if (fails) {
        std::printf("\n%d CHECK(S) FAILED\n", fails);
        return 1;
    }
    std::printf("\nall checks passed\n");
    return 0;
}
