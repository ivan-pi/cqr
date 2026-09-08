// test_cqr_gels_compact.cpp
//
// Self-contained validation of the templated compact least-squares /
// minimum-norm solve (dgels_compact / sgels_compact), with no BLAS dependency.
// The reference is ref_gels: the unblocked algorithm (geqr2 of the tall
// orientation, orm2r, back substitution) in scalar form -- the same steps the
// vectorized kernel executes V lanes at a time, so a correct kernel matches it
// to working precision -- and, independently of it, the properties that define
// the two solutions.
//
// Checks per (T, V, layout, trans, shape):
//   1. X (all max(m,n) rows of B, the residual rows included) == ref_gels
//   2. the factorization left in A and the tau left in work == ref_gels
//   3. least squares: the normal equations op(A)^T (B - op(A) X) = 0, and the
//      residual sums of squares in rows n..m-1 of B equal ||B - op(A) X||^2
//      minimum norm: op(A) X = B, and X equals the minimum-norm solution formed
//      the other way, X = op(A)^T Z with (op(A) op(A)^T) Z = B
// plus LAPACK-style argument validation of the C API, the workspace query, and
// the min(m,n) = 0 case (B := 0).
//
// Assisted-by: Claude:claude-fable-5

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <limits>
#include <algorithm>

#include "test_compact_util.hpp" // compact<T>, scalar references, MatrixBatch, pack/unpack

using namespace cqr::test;

namespace {

// C (rows x cols) := op(A) X, op(A) = A (m x n) or A^T, all column-major;
// X has (columns of op(A)) rows, leading dim ldx.
template <class T>
void opA_mul(bool tran, int m, int n, int ncols, const T *A, const T *X, int ldx, T *C)
{
    const int rows = tran ? n : m, inner = tran ? m : n;
    for (int j = 0; j < ncols; ++j)
        for (int i = 0; i < rows; ++i) {
            T s = 0;
            for (int l = 0; l < inner; ++l)
                s += (tran ? A[l + (size_t)i * m] : A[i + (size_t)l * m]) *
                     X[l + (size_t)j * ldx];
            C[i + (size_t)j * rows] = s;
        }
}

// --------------------------- one test case --------------------------

template <class T, int V>
int run_case(char layout, char trans, int nm, int m, int n, int nrhs)
{
    const bool row = (layout == 'R');
    const bool tran = (trans == 'T');
    const bool tall = (m >= n);
    const int p = std::max(m, n), q = std::min(m, n);
    const bool overdet = (tall != tran);
    const int rows_op = tran ? n : m,
              cols_op = tran ? m : n; // op(A) is rows_op x cols_op
    const T eps = std::numeric_limits<T>::epsilon();

    // random A (diagonal-boosted, so op(A) is well conditioned) and random B;
    // the reference works on copies
    MatrixBatch<T> A(nm, m, n), Aref(nm, m, n), B(nm, p, nrhs), Bref(nm, p, nrhs),
        tau_ref(nm, q, 1);
    for (int idx = 0; idx < nm; ++idx) {
        gen_boosted(A[idx], m, n);
        for (size_t e = 0; e < (size_t)p * nrhs; ++e)
            B[idx][e] = frand<T>();
        std::copy(A[idx], A[idx] + (size_t)m * n, Aref[idx]);
        std::copy(B[idx], B[idx] + (size_t)p * nrhs, Bref[idx]);
        ref_gels(trans, m, n, nrhs, Aref[idx], m, Bref[idx], p, tau_ref[idx]);
    }

    // pack in the requested layout, query the workspace, solve, unpack
    const int ng = (nm + V - 1) / V;
    const int lda = row ? n : m, ldb = row ? nrhs : p;
    std::vector<T> ap((size_t)ng * m * n * V), bp((size_t)ng * p * nrhs * V);
    pack_compact(A, ap.data(), lda, V, row);
    pack_compact(B, bp.data(), ldb, V, row);

    T wq = -1;
    int info = compact<T>::gels(layout, trans, m, n, nrhs, ap.data(), lda, bp.data(), ldb,
                                &wq, -1, V, nm);
    const bool ok_query = (info == 0) && (wq == T(std::max(1, q * V * ng)));
    std::vector<T> work((size_t)std::max<T>(wq, 1));
    info = compact<T>::gels(layout, trans, m, n, nrhs, ap.data(), lda, bp.data(), ldb,
                            work.data(), (int)work.size(), V, nm);

    MatrixBatch<T> Aout(nm, m, n), Bout(nm, p, nrhs), tau_out(nm, q, 1);
    unpack_compact(Aout, ap.data(), lda, V, row);
    unpack_compact(Bout, bp.data(), ldb, V, row);
    unpack_tau(tau_out, work.data(), V);

    // check 1 & 2: X (all p rows), the factorization, and tau vs the reference
    double e_x = 0, e_h = 0, e_t = 0, nrm_b = 0;
    for (int idx = 0; idx < nm; ++idx) {
        e_x = std::max(e_x, max_abs_diff(Bout[idx], Bref[idx], (size_t)p * nrhs));
        e_h = std::max(e_h, max_abs_diff(Aout[idx], Aref[idx], (size_t)m * n));
        e_t = std::max(e_t, max_abs_diff(tau_out[idx], tau_ref[idx], (size_t)q));
        nrm_b = std::max(nrm_b, norm1(Bref[idx], p, nrhs));
    }

    // check 3: the defining property, formed without the reference
    double e_prop = 0, e_rss = 0;
    std::vector<T> R((size_t)rows_op * nrhs), S((size_t)cols_op * nrhs);
    for (int idx = 0; idx < nm; ++idx) {
        const T *Av = A[idx], *Xv = Bout[idx]; // X: the first cols_op rows of Bout
        const double scale = std::max(norm1(Av, m, n), 1e-300) *
                             std::max(norm1(B[idx], rows_op, nrhs), 1e-300);
        if (overdet) {
            // r = B - op(A) X (rows_op x nrhs); normal equations op(A)^T r = 0
            opA_mul(tran, m, n, nrhs, Av, Xv, p, R.data());
            for (int j = 0; j < nrhs; ++j)
                for (int i = 0; i < rows_op; ++i)
                    R[i + (size_t)j * rows_op] =
                        B[idx][i + (size_t)j * p] - R[i + (size_t)j * rows_op];
            opA_mul(!tran, m, n, nrhs, Av, R.data(), rows_op, S.data());
            for (size_t e = 0; e < (size_t)cols_op * nrhs; ++e)
                e_prop = std::max(e_prop, std::abs((double)S[e]) / scale);
            // rows cols_op..rows_op-1 of B hold the residual: squared column
            // norms == ||r_j||^2 (relative to ||b_j||^2: a square system has no
            // residual rows and a zero residual)
            for (int j = 0; j < nrhs; ++j) {
                double rss = 0, rss_b = 0, bb = 0;
                for (int i = 0; i < rows_op; ++i) {
                    rss +=
                        (double)R[i + (size_t)j * rows_op] * R[i + (size_t)j * rows_op];
                    bb += (double)B[idx][i + (size_t)j * p] * B[idx][i + (size_t)j * p];
                }
                for (int i = cols_op; i < rows_op; ++i)
                    rss_b += (double)Xv[i + (size_t)j * p] * Xv[i + (size_t)j * p];
                e_rss = std::max(e_rss, std::abs(rss - rss_b) / std::max(bb, 1e-300));
            }
        }
        else {
            // op(A) X = B (rows_op x nrhs)
            opA_mul(tran, m, n, nrhs, Av, Xv, p, R.data());
            for (int j = 0; j < nrhs; ++j)
                for (int i = 0; i < rows_op; ++i)
                    e_prop =
                        std::max(e_prop, std::abs((double)R[i + (size_t)j * rows_op] -
                                                  B[idx][i + (size_t)j * p]) /
                                             scale);
            // minimum norm the other way: X = op(A)^T Z, G Z = B with the
            // Gram matrix G = op(A) op(A)^T (rows_op x rows_op, SPD), solved by
            // the scalar QR references
            std::vector<T> AopT((size_t)cols_op * rows_op), G((size_t)rows_op * rows_op),
                Z((size_t)rows_op * nrhs), tg(rows_op), Xmn((size_t)cols_op * nrhs);
            for (int j = 0; j < rows_op; ++j) // AopT = op(A)^T, cols_op x rows_op
                for (int i = 0; i < cols_op; ++i)
                    AopT[i + (size_t)j * cols_op] =
                        tran ? Av[i + (size_t)j * m] : Av[j + (size_t)i * m];
            opA_mul(tran, m, n, rows_op, Av, AopT.data(), cols_op, G.data());
            for (int j = 0; j < nrhs; ++j)
                for (int i = 0; i < rows_op; ++i)
                    Z[i + (size_t)j * rows_op] = B[idx][i + (size_t)j * p];
            ref_geqr2(rows_op, rows_op, G.data(), rows_op, tg.data());
            ref_orm2r('T', rows_op, nrhs, rows_op, G.data(), rows_op, tg.data(), Z.data(),
                      rows_op);
            ref_trsm_upper(rows_op, nrhs, G.data(), rows_op, Z.data(), rows_op);
            opA_mul(!tran, m, n, nrhs, Av, Z.data(), rows_op, Xmn.data());
            for (int j = 0; j < nrhs; ++j)
                for (int i = 0; i < cols_op; ++i)
                    e_rss =
                        std::max(e_rss, std::abs((double)Xmn[i + (size_t)j * cols_op] -
                                                 Xv[i + (size_t)j * p]) /
                                            std::max(norm1(Xv, cols_op, nrhs), 1e-300));
        }
    }

    const double tol_ref = 200.0 * eps * p; // same op sequence, ~eps
    const double tol_x = 200.0 * eps * p * std::max(nrm_b, 1.0);
    const double tol_prop = 100.0 * eps * p; // backward-stable quantities
    const double tol_mn = 1e4 * eps * p;     // cond(G) = cond(op(A))^2
    const bool ok_x = e_x <= tol_x, ok_h = e_h <= tol_ref, ok_t = e_t <= tol_ref;
    const bool ok_p = e_prop <= tol_prop, ok_r = e_rss <= (overdet ? tol_prop : tol_mn);

    std::printf(
        "T=%-6s V=%-2d %s trans=%c nm=%-2d m=%-3d n=%-3d nrhs=%d %-6s | X:%.1e %s "
        "H:%.1e %s tau:%.1e %s | %s:%.1e %s %s:%.1e %s | info=%d%s\n",
        compact<T>::name, V, row ? "row" : "col", trans, nm, m, n, nrhs,
        overdet ? "lstsq" : "minnrm", e_x, ok_x ? "OK" : "FAIL", e_h,
        ok_h ? "OK" : "FAIL", e_t, ok_t ? "OK" : "FAIL", overdet ? "normal" : "resid",
        e_prop, ok_p ? "OK" : "FAIL", overdet ? "rss" : "minnorm", e_rss,
        ok_r ? "OK" : "FAIL", info, ok_query ? "" : " QUERY-FAIL");
    return (info != 0) + !ok_query + !ok_x + !ok_h + !ok_t + !ok_p + !ok_r;
}

// --------------------- C API argument validation --------------------

int test_validation()
{
    const int m = 8, n = 6, nrhs = 3, V = 4, nm = 5, ng = 2;
    const int lda = 8, ldb = 8; // column-major: ldap >= m, ldbp >= max(m,n)
    std::vector<double> ap((size_t)ng * lda * n * V, 0),
        bp((size_t)ng * ldb * nrhs * V, 0), work((size_t)n * V * ng, 0);
    const int lw = (int)work.size();
    auto call = [&](char lay, char tr, int m_, int n_, int nrhs_, int ldap_, int ldbp_,
                    int lwork_, int V_, int nm_) {
        return dgels_compact(lay, tr, m_, n_, nrhs_, ap.data(), ldap_, bp.data(), ldbp_,
                             work.data(), lwork_, V_, nm_);
    };
    // clang-format off
    struct { const char *what; int got, want; } t[] = {
        {"valid col",   call('C', 'N', m, n, nrhs, lda, ldb, lw,   V, nm),   0},
        {"valid col T", call('C', 'T', m, n, nrhs, lda, ldb, lw,   V, nm),   0},
        {"valid row",   call('R', 'N', m, n, nrhs, n,   nrhs, lw,  V, nm),   0}, // row-major lds
        {"bad layout",  call('X', 'N', m, n, nrhs, lda, ldb, lw,   V, nm),  -1},
        {"bad trans",   call('C', 'X', m, n, nrhs, lda, ldb, lw,   V, nm),  -2},
        {"m<0",         call('C', 'N', -1, n, nrhs, lda, ldb, lw,  V, nm),  -3},
        {"n<0",         call('C', 'N', m, -1, nrhs, lda, ldb, lw,  V, nm),  -4},
        {"nrhs<0",      call('C', 'N', m, n, -1, lda, ldb, lw,     V, nm),  -5},
        {"ldap<m",      call('C', 'N', m, n, nrhs, m-1, ldb, lw,   V, nm),  -7},
        {"ldbp<max",    call('C', 'N', m, n, nrhs, lda, m-1, lw,   V, nm),  -9},
        {"ldbp<max T",  call('C', 'T', n, m, nrhs, n,   m-1, lw,   V, nm),  -9}, // wide A, B has m rows
        {"lwork small", call('C', 'N', m, n, nrhs, lda, ldb, lw-1, V, nm), -11},
        {"bad V",       call('C', 'N', m, n, nrhs, lda, ldb, lw,   3, nm), -12},
        {"nm<0",        call('C', 'N', m, n, nrhs, lda, ldb, lw,   V, -1), -13},
        {"query",       call('C', 'N', m, n, nrhs, lda, ldb, -1,   V, nm),   0},
        {"empty nrhs",  call('C', 'N', m, n, 0, lda, ldb, lw,      V, nm),   0},
        {"empty nm=0",  call('C', 'N', m, n, nrhs, lda, ldb, lw,   V, 0),    0},
    };
    // clang-format on
    int bad = 0;
    for (auto &c : t)
        bad += (c.got != c.want);
    // the query (the last valid call above) reported the minimum lwork
    const bool ok_q = (work[0] == (double)lw);
    bad += !ok_q;
    std::printf("C API validation: %zu checks | %s\n", sizeof(t) / sizeof(t[0]) + 1,
                bad ? "FAIL" : "OK");
    for (auto &c : t)
        if (c.got != c.want)
            std::printf("  %-12s got=%d want=%d\n", c.what, c.got, c.want);
    if (!ok_q) std::printf("  query       work[0]=%g want=%d\n", work[0], lw);
    return bad ? 1 : 0;
}

// min(m,n) = 0 with a non-empty B: the solution of the empty system, B := 0
// over all max(m,n) rows -- LAPACK ?gels's quick return -- in both layouts.
int test_empty_op()
{
    const int V = 4, nm = 6, ng = 2, n = 5, nrhs = 3;
    int bad = 0;
    for (char lay : {'C', 'R'}) {
        const bool row = (lay == 'R');
        const int ldb = row ? nrhs : n;
        std::vector<double> bp((size_t)ng * n * nrhs * V, 1.0), work(1);
        double *ap = nullptr;        // m = 0: A is empty and never touched
        const int lda = row ? n : 1; // (but its leading dimension is still checked)
        int info = dgels_compact(lay, 'N', 0, n, nrhs, ap, lda, bp.data(), ldb,
                                 work.data(), 1, V, nm);
        double mx = 0;
        for (double x : bp)
            mx = std::max(mx, std::abs(x));
        const bool ok = (info == 0) && (mx == 0.0);
        bad += !ok;
        std::printf("empty op(A) (m=0, n=%d) %s: B := 0 | max|B| %.1e info=%d %s\n", n,
                    row ? "row" : "col", mx, info, ok ? "OK" : "FAIL");
    }
    return bad;
}

} // namespace

// ------------------------------- main --------------------------------

int main()
{
    int fails = 0;
    fails += test_validation();
    fails += test_empty_op();

    // every (layout, trans) over square, tall and wide A, so all four
    // over-/underdetermined cases run in both layouts
    for (char lay : {'C', 'R'})
        for (char tr : {'N', 'T'}) {
            fails += run_case<double, 4>(lay, tr, 8, 30, 30, 3);  // square
            fails += run_case<double, 4>(lay, tr, 8, 40, 24, 5);  // tall
            fails += run_case<double, 4>(lay, tr, 8, 24, 40, 5);  // wide
            fails += run_case<double, 8>(lay, tr, 11, 43, 17, 4); // padded partial group
            fails += run_case<double, 2>(lay, tr, 4, 12, 20, 1);  // single RHS, V=2
            fails += run_case<float, 8>(lay, tr, 11, 32, 20, 3);  // padded partial group
            fails += run_case<float, 16>(lay, tr, 32, 20, 32, 2);
        }
    fails += run_case<double, 4>('C', 'N', 40, 24, 16, 3); // 10 groups: OpenMP path
    fails += run_case<double, 4>('C', 'T', 40, 24, 16, 3);
    fails += run_case<double, 4>('C', 'N', 8, 3, 3, 2);   // smallest supported
    fails += run_case<double, 4>('C', 'N', 8, 5, 1, 2);   // a single column
    fails += run_case<double, 4>('C', 'N', 8, 1, 5, 2);   // a single row
    fails += run_case<double, 4>('C', 'N', 8, 64, 20, 6); // wider RHS blocks (4+2 tail)
    fails += run_case<double, 4>('C', 'T', 8, 64, 20, 7); // (4+2+1 tail)

    if (fails) {
        std::printf("\n%d CHECK(S) FAILED\n", fails);
        return 1;
    }
    std::printf("\nall checks passed\n");
    return 0;
}
