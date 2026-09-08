// test_cqr_sytrfnp_compact.cpp
//
// Self-contained validation of the templated compact unpivoted LDL^T
// factorization (dsytrfnp_compact / ssytrfnp_compact), its solve companion
// (dsytrsnp_compact / ssytrsnp_compact) and the fused factor-and-solve
// (dsysvnp_compact / ssysvnp_compact), with no BLAS dependency. The reference is
// the unblocked unpivoted factorization implemented in scalar form -- the same
// algorithm the vectorized kernel executes V lanes at a time, so a correct
// kernel matches it to working precision.
//
// Checks per (T, V, uplo, layout), on symmetric *indefinite* batches:
//   1. named-triangle factor  ==  scalar reference factor  (elementwise)
//   2. reconstruction  L D L^T == A  (lower) / U^T D U == A  (upper)
//   3. the strictly-opposite triangle of the compact buffer is bit-for-bit
//      unchanged from the input (the routine must not reference or write it)
//   4. end-to-end solve: factor + ?sytrsnp_compact recovers a known X, and
//      ?sysvnp_compact reproduces that factor and X bit-for-bit
// plus:
//   - a zero on the *input* diagonal with nonsingular leading minors factors
//     fine (the pivots are the updated Schur-complement entries),
//   - a zero-*pivot* lane poisons itself with Inf/NaN without contaminating
//     its siblings (design section 6.2),
//   - LAPACK-style argument validation of the three C APIs.
//
// Assisted-by: Claude

#include <cstdio>
#include <cmath>
#include <vector>
#include <limits>
#include <algorithm>
#include <functional>
#include <utility>

#include "test_compact_util.hpp" // compact<T>, gen_sym_ldlt, ldlt_reconstruct, pack/unpack

using namespace cqr::test;

// ----------------------- reference kernel (scalar) ------------------
// Unblocked right-looking unpivoted LDL^T on a dense column-major n x n matrix,
// in place. Lower: A = L D L^T; upper: A = U^T D U (the transpose dual, NOT
// LAPACK sytrf's U D U^T). D lands on the diagonal, the unit factor strictly
// off it. Only the named triangle is read or written. No singularity check
// (mirrors the routine under test): a zero pivot yields Inf/NaN.

template <class T> static void ref_sytf2np(char uplo, int n, T *A, int lda)
{
    const bool upper = (uplo == 'U' || uplo == 'u');
    if (!upper) {
        for (int j = 0; j < n; ++j) {
            T d = A[j + (size_t)j * lda];
            T invd = T(1) / d;
            for (int i = j + 1; i < n; ++i)
                A[i + (size_t)j * lda] *= invd; // scale pivot column -> L(:,j)
            for (int jj = j + 1; jj < n; ++jj)  // rank-1 trailing update, lower
                for (int i = jj; i < n; ++i)
                    A[i + (size_t)jj * lda] -=
                        A[i + (size_t)j * lda] * (A[jj + (size_t)j * lda] * d);
        }
    }
    else {
        for (int j = 0; j < n; ++j) {
            T d = A[j + (size_t)j * lda];
            T invd = T(1) / d;
            for (int c = j + 1; c < n; ++c)
                A[j + (size_t)c * lda] *= invd; // scale pivot row -> U(j,:)
            // rank-1 trailing update, upper, row at a time with w = U(j,r)*d
            // hoisted: the same operand pairing as the kernel's transposed
            // sweep, so the two stay bit-comparable.
            for (int r = j + 1; r < n; ++r) {
                const T w = A[j + (size_t)r * lda] * d;
                for (int c = r; c < n; ++c)
                    A[r + (size_t)c * lda] -= A[j + (size_t)c * lda] * w;
            }
        }
    }
}

// --------------------------- one test case --------------------------

template <class T, int V> static int run_case(int nm, int n, char uplo, char layout)
{
    const T eps = std::numeric_limits<T>::epsilon();
    const bool rowmajor = (layout == 'R' || layout == 'r');
    const bool upper = (uplo == 'U' || uplo == 'u');

    // random symmetric-indefinite batch + scalar reference factor for this uplo
    MatrixBatch<T> A(nm, n, n), Aref(nm, n, n);
    for (int idx = 0; idx < nm; ++idx) {
        gen_sym_ldlt(A[idx], n);
        std::copy(A[idx], A[idx] + (size_t)n * n, Aref[idx]);
        ref_sytf2np(uplo, n, Aref[idx], n);
    }

    // pack the full symmetric A, factor with the routine under test, unpack
    int ng = (nm + V - 1) / V;
    std::vector<T> ap((size_t)ng * n * n * V);
    pack_compact(A, ap.data(), n, V, rowmajor);
    int info = compact<T>::sytrfnp(layout, uplo, n, ap.data(), n, V, nm);
    MatrixBatch<T> Aout(nm, n, n);
    unpack_compact(Aout, ap.data(), n, V, rowmajor);

    double e_fac = 0, e_rec = 0, e_untouched = 0, a_norm = 1;
    for (int idx = 0; idx < nm; ++idx) {
        a_norm = std::max(a_norm, norm1(A[idx], n, n));
        // check 1: named-triangle factor vs scalar reference (elementwise)
        // check 3: strictly-opposite triangle unchanged from the input A
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                const bool named = upper ? (i <= j) : (i >= j);
                if (named)
                    e_fac = std::max(e_fac,
                                     (double)std::abs(Aout(idx, i, j) - Aref(idx, i, j)));
                else
                    e_untouched = std::max(
                        e_untouched, (double)std::abs(Aout(idx, i, j) - A(idx, i, j)));
            }

        // check 2: reconstruction of A from the named triangle's (D, L|U)
        auto at = [&](int i, int j) { return Aout(idx, i, j); };
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                e_rec = std::max(e_rec, std::abs(ldlt_reconstruct(at, i, j, upper) -
                                                 (double)A(idx, i, j)));
    }

    const double scale = std::max(1, n);
    const double tol_fac = 20.0 * eps * scale;          // same op sequence
    const double tol_rec = 40.0 * eps * scale * a_norm; // O(n) accumulation in recon
    bool ok_f = e_fac <= tol_fac;
    bool ok_r = e_rec <= tol_rec;
    bool ok_u = e_untouched == 0.0; // must be bit-for-bit unchanged
    bool ok_i = (info == 0);

    std::printf("T=%-6s V=%-2d uplo=%c lay=%c nm=%-2d n=%-3d | fac:%.1e %s rec:%.1e %s "
                "untouched:%s | info=%d %s\n",
                compact<T>::name, V, uplo, layout, nm, n, e_fac, ok_f ? "OK" : "FAIL",
                e_rec, ok_r ? "OK" : "FAIL", ok_u ? "OK" : "FAIL", info,
                ok_i ? "OK" : "FAIL");
    return (!ok_f) + (!ok_r) + (!ok_u) + (!ok_i);
}

// --------------------- end-to-end solve A X = B ---------------------
// Two-step (sytrfnp, then sytrsnp) against a known X, and the fused sysvnp
// against the two-step result: the fused driver runs the same group kernels in
// the same order on the same data, so its factor and X must match bit-for-bit.

template <class T, int V>
static int run_solve(int nm, int n, int nrhs, char uplo, char layout)
{
    const T eps = std::numeric_limits<T>::epsilon();
    const bool rowmajor = (layout == 'R' || layout == 'r');
    const int ldb = rowmajor ? nrhs : n;

    // known X, B = A X densely
    MatrixBatch<T> A(nm, n, n), B(nm, n, nrhs);
    const std::vector<T> X = known_solution<T>(n, nrhs);
    for (int idx = 0; idx < nm; ++idx) {
        gen_sym_ldlt(A[idx], n);
        matmul(n, nrhs, n, A[idx], n, X.data(), n, B[idx], n);
    }

    int ng = (nm + V - 1) / V;
    std::vector<T> ap((size_t)ng * n * n * V), bp((size_t)ng * n * nrhs * V);
    pack_compact(A, ap.data(), n, V, rowmajor);
    pack_compact(B, bp.data(), ldb, V, rowmajor);
    std::vector<T> ap2 = ap, bp2 = bp; // the fused call's copies

    int info_f = compact<T>::sytrfnp(layout, uplo, n, ap.data(), n, V, nm);
    int info_s =
        compact<T>::sytrsnp(layout, uplo, n, nrhs, ap.data(), n, bp.data(), ldb, V, nm);
    int info_v =
        compact<T>::sysvnp(layout, uplo, n, nrhs, ap2.data(), n, bp2.data(), ldb, V, nm);

    MatrixBatch<T> Xhat(nm, n, nrhs);
    unpack_compact(Xhat, bp.data(), ldb, V, rowmajor);

    double e_fwd = 0, e_res = 0;
    const size_t sB = (size_t)n * nrhs;
    std::vector<T> AX(sB);
    for (int idx = 0; idx < nm; ++idx) {
        e_fwd = std::max(e_fwd, max_abs_diff(Xhat[idx], X.data(), sB) /
                                    std::max(norm1(X.data(), n, nrhs), 1e-300));
        matmul(n, nrhs, n, A[idx], n, Xhat[idx], n, AX.data(), n);
        e_res = std::max(e_res, max_abs_diff(AX.data(), B[idx], sB) /
                                    std::max(norm1(B[idx], n, nrhs), 1e-300));
    }
    // fused vs two-step, on the raw compact buffers (padded lanes included)
    const bool fused_same = (ap2 == ap) && (bp2 == bp);

    // The residual gate is what the (backward-stable) sweeps control; the
    // forward error additionally carries cond(A) of the indefinite batch, so
    // its gate gets conditioning headroom.
    const double rtol_res = 100.0 * std::max(1, n) * eps;
    const double rtol_fwd = 500.0 * std::max(1, n) * eps;
    bool ok = (e_fwd <= rtol_fwd) && (e_res <= rtol_res) && fused_same && (info_f == 0) &&
              (info_s == 0) && (info_v == 0);
    std::printf("T=%-6s V=%-2d uplo=%c lay=%c nm=%-2d n=%-3d nrhs=%d solve | fwd:%.1e "
                "(%.1e) res:%.1e (%.1e) sysv==trf+trs:%s info=%d/%d/%d %s\n",
                compact<T>::name, V, uplo, layout, nm, n, nrhs, e_fwd, rtol_fwd, e_res,
                rtol_res, fused_same ? "yes" : "NO", info_f, info_s, info_v,
                ok ? "OK" : "FAIL");
    return !ok;
}

// -------- zero on the input diagonal, nonsingular leading minors -----
// The pivots are the *updated* Schur-complement diagonal entries, so a zero on
// the original diagonal is harmless as long as no leading principal minor
// vanishes. Handcrafted: A = L D L^T with L(1,0) = 1, D = diag(2,-2,1,-1,...)
// gives A(1,1) = 1*2*1 + (-2) = 0 while the minors are prod(d) != 0. The
// factorization must stay finite and recover (L, D) exactly as the scalar
// reference does; Cholesky would already have failed on d_1 < 0.

template <class T, int V> static int run_zerodiag(char uplo, char layout)
{
    const int n = 4, nm = V; // one full pack
    const bool rowmajor = (layout == 'R' || layout == 'r');
    const bool upper = (uplo == 'U' || uplo == 'u');

    std::vector<T> L((size_t)n * n, T(0)), d = {T(2), T(-2), T(1), T(-1)};
    for (int j = 0; j < n; ++j)
        L[j + (size_t)j * n] = T(1);
    L[1 + 0 * n] = T(1); // makes A(1,1) = d0 * 1 + d1 = 0
    L[2 + 0 * n] = T(0.5);
    L[3 + 1 * n] = T(-0.75);
    L[3 + 2 * n] = T(0.25);

    // (L, D) read as one stored factor: D on the diagonal, L strictly below it
    auto ld = [&](int i, int j) { return i == j ? d[i] : L[i + (size_t)j * n]; };
    MatrixBatch<T> A(nm, n, n), Aref(nm, n, n);
    for (int idx = 0; idx < nm; ++idx) {
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                A(idx, i, j) = (T)ldlt_reconstruct(ld, i, j, false);
        std::copy(A[idx], A[idx] + (size_t)n * n, Aref[idx]);
        ref_sytf2np(uplo, n, Aref[idx], n);
    }

    std::vector<T> ap((size_t)n * n * V);
    pack_compact(A, ap.data(), n, V, rowmajor);
    int info = compact<T>::sytrfnp(layout, uplo, n, ap.data(), n, V, nm);
    MatrixBatch<T> Aout(nm, n, n);
    unpack_compact(Aout, ap.data(), n, V, rowmajor);

    double e = 0;
    bool finite = true;
    for (int idx = 0; idx < nm; ++idx)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                const bool named = upper ? (i <= j) : (i >= j);
                if (!named) continue;
                if (!std::isfinite((double)Aout(idx, i, j))) finite = false;
                e = std::max(e, (double)std::abs(Aout(idx, i, j) - Aref(idx, i, j)));
            }
    const double tol = 20.0 * std::numeric_limits<T>::epsilon() * n;
    bool ok = finite && (e <= tol) && (info == 0);
    std::printf("T=%-6s V=%-2d uplo=%c lay=%c zero-diagonal A(1,1)=0 | err:%.1e "
                "finite:%s %s\n",
                compact<T>::name, V, uplo, layout, e, finite ? "yes" : "NO",
                ok ? "OK" : "FAIL");
    return !ok;
}

// ------------- zero-pivot lane isolation (design section 6.2) --------
// A single lane with a singular leading 1x1 minor (A(0,0) = 0, A(0,1) != 0)
// shares a pack with factorable siblings. The routine takes no safeguarded path
// and reports no error (info stays 0): the bad lane's 1/0 poisons *its own*
// factor with Inf/NaN -- while every sibling lane, computed with the same
// unmasked SIMD instructions, must stay finite and match the scalar reference.

template <class T, int V> static int run_zeropivot(int n, char uplo, char layout)
{
    const bool rowmajor = (layout == 'R' || layout == 'r');
    const bool upper = (uplo == 'U' || uplo == 'u');
    const int nm = V;      // one full pack, so all V lanes are exercised together
    const int badlane = 1; // this lane has a zero pivot; the rest factor fine

    MatrixBatch<T> A(nm, n, n), Aref(nm, n, n);
    for (int idx = 0; idx < nm; ++idx) {
        if (idx == badlane) {
            // symmetric, but the leading 1x1 minor is singular: the very first
            // pivot is A(0,0) = 0, so invd = 1/0 = Inf poisons the lane.
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    A(idx, i, j) = (i == j) ? T(1) : T(0);
            A(idx, 0, 0) = T(0);
            A(idx, 0, 1) = A(idx, 1, 0) = T(1);
        }
        else {
            gen_sym_ldlt(A[idx], n);
            std::copy(A[idx], A[idx] + (size_t)n * n, Aref[idx]);
            ref_sytf2np(uplo, n, Aref[idx], n);
        }
    }

    std::vector<T> ap((size_t)n * n * V);
    pack_compact(A, ap.data(), n, V, rowmajor);
    int info = compact<T>::sytrfnp(layout, uplo, n, ap.data(), n, V, nm);
    MatrixBatch<T> Aout(nm, n, n);
    unpack_compact(Aout, ap.data(), n, V, rowmajor);

    double e_sib = 0; // worst error over the factorable sibling lanes
    bool sib_finite = true;
    bool bad_poisoned = false; // the zero-pivot lane must carry Inf/NaN
    for (int idx = 0; idx < nm; ++idx)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                const bool named = upper ? (i <= j) : (i >= j);
                if (!named) continue;
                const T x = Aout(idx, i, j);
                if (idx == badlane) {
                    if (!std::isfinite((double)x)) bad_poisoned = true;
                }
                else {
                    if (!std::isfinite((double)x)) sib_finite = false;
                    e_sib = std::max(e_sib, (double)std::abs(x - Aref(idx, i, j)));
                }
            }

    const double tol = 20.0 * std::numeric_limits<T>::epsilon() * std::max(1, n);
    bool ok_sib = sib_finite && (e_sib <= tol); // siblings uncontaminated & correct
    bool ok_bad = bad_poisoned;                 // poison confined but present
    bool ok_info = (info == 0);                 // GIGO: no early-exit, no error

    std::printf("T=%-6s V=%-2d uplo=%c lay=%c n=%-3d zero-pivot lane | siblings:%.1e %s "
                "poison:%s info=%d %s\n",
                compact<T>::name, V, uplo, layout, n, e_sib, ok_sib ? "OK" : "FAIL",
                ok_bad ? "OK" : "FAIL", info, ok_info ? "OK" : "FAIL");
    return (!ok_sib) + (!ok_bad) + (!ok_info);
}

// --------------------- C API argument validation --------------------

static int test_validation()
{
    const int n = 8, nrhs = 3, V = 4, nm = 4, ld = 8;
    std::vector<double> ap((size_t)ld * n * V, 0), bp((size_t)ld * nrhs * V, 0);
    for (int v = 0; v < V; ++v) // seed a unit diagonal so factoring/solving is sane
        for (int i = 0; i < n; ++i)
            ap[((size_t)i * ld + i) * V + v] = 1.0;
    auto callf = [&](char lay, char up, int n_, int ldap_, int V_, int nm_) {
        return dsytrfnp_compact(lay, up, n_, ap.data(), ldap_, V_, nm_);
    };
    // ?sytrsnp and ?sysvnp share one signature and one validation, so one table
    // of solve cases runs against both entry points
    using solve_fn = std::function<int(char, char, int, int, int, int, int, int)>;
    const std::pair<const char *, solve_fn> solvers[] = {
        {"trs",
         [&](char lay, char up, int n_, int nrhs_, int ldap_, int ldbp_, int V_,
             int nm_) {
             return dsytrsnp_compact(lay, up, n_, nrhs_, ap.data(), ldap_, bp.data(),
                                     ldbp_, V_, nm_);
         }},
        {"sv",
         [&](char lay, char up, int n_, int nrhs_, int ldap_, int ldbp_, int V_,
             int nm_) {
             return dsysvnp_compact(lay, up, n_, nrhs_, ap.data(), ldap_, bp.data(),
                                    ldbp_, V_, nm_);
         }},
    };
    struct Case {
        const char *what;
        int got, want;
    };
    std::vector<Case> t;
    // clang-format off
    t.insert(t.end(), {
        {"trf valid col L", callf('C', 'L', n,  ld,  V, nm),   0},
        {"trf valid row U", callf('R', 'U', n,  ld,  V, nm),   0},
        {"trf bad layout",  callf('X', 'L', n,  ld,  V, nm),  -1},
        {"trf bad uplo",    callf('C', 'X', n,  ld,  V, nm),  -2},
        {"trf n<0",         callf('C', 'L', -1, ld,  V, nm),  -3},
        {"trf ldap<n",      callf('C', 'L', n,  n-1, V, nm),  -5},
        {"trf bad V",       callf('C', 'L', n,  ld,  3, nm),  -6},
        {"trf nm<0",        callf('C', 'L', n,  ld,  V, -1),  -7},
        {"trf empty n=0",   callf('C', 'L', 0,  1,   V, nm),   0},
        {"trf empty nm=0",  callf('C', 'L', n,  ld,  V, 0),    0},
    });
    for (const auto &[tag, calls] : solvers)
        t.insert(t.end(), {
            {"valid col L", calls('C', 'L', n, nrhs, ld, n,    V, nm),   0},
            {"valid row U", calls('R', 'U', n, nrhs, ld, nrhs, V, nm),   0},
            {"bad layout",  calls('X', 'L', n, nrhs, ld, n,    V, nm),  -1},
            {"bad uplo",    calls('C', 'X', n, nrhs, ld, n,    V, nm),  -2},
            {"n<0",         calls('C', 'L', -1, nrhs, ld, n,   V, nm),  -3},
            {"nrhs<0",      calls('C', 'L', n, -1,  ld, n,     V, nm),  -4},
            {"ldap<n",      calls('C', 'L', n, nrhs, n-1, n,   V, nm),  -6},
            {"ldbp<n",      calls('C', 'L', n, nrhs, ld, n-1,  V, nm),  -8},
            {"ldbp<nrhs R", calls('R', 'L', n, nrhs, ld, nrhs-1, V, nm), -8},
            {"bad V",       calls('C', 'L', n, nrhs, ld, n,    3, nm),  -9},
            {"nm<0",        calls('C', 'L', n, nrhs, ld, n,    V, -1), -10},
            {"empty nrhs",  calls('C', 'L', n, 0,   ld, n,     V, nm),   0},
        });
    // clang-format on
    int bad = 0;
    for (auto &c : t)
        bad += (c.got != c.want);
    std::printf("C API validation: %zu checks | %s\n", t.size(), bad ? "FAIL" : "OK");
    for (auto &c : t)
        if (c.got != c.want)
            std::printf("  %-12s got=%d want=%d\n", c.what, c.got, c.want);
    return bad ? 1 : 0;
}

// ------------------------------- main --------------------------------

int main()
{
    int fails = 0;
    fails += test_validation();

    // Full feature matrix: both uplo x both layouts, several (V, n, nm),
    // including padded partial final groups (nm not a multiple of V).
    const char uplos[] = {'L', 'U'};
    const char lays[] = {'C', 'R'};
    for (char u : uplos)
        for (char l : lays) {
            fails += run_case<double, 2>(4, 16, u, l);
            fails += run_case<double, 4>(8, 30, u, l);
            fails += run_case<double, 8>(16, 40, u, l);
            fails += run_case<double, 8>(11, 43, u, l); // padded partial group
            fails += run_case<double, 4>(3, 3, u, l);   // smallest, padded
            fails += run_case<float, 8>(16, 30, u, l);
            fails += run_case<float, 16>(17, 24, u, l); // padded partial group
        }

    // End-to-end solve A X = B on indefinite batches, closing the pipeline
    // (factor + sytrsnp, and the fused sysvnp), over uplo/layout/precision,
    // padded groups, and RHS counts that exercise trsm's 4/2/1 column blocks.
    for (char u : uplos)
        for (char l : lays) {
            fails += run_solve<double, 4>(8, 30, 5, u, l);
            fails += run_solve<double, 8>(11, 43, 4, u, l); // padded partial group
            fails += run_solve<double, 2>(6, 17, 1, u, l);  // single RHS
            fails += run_solve<float, 8>(16, 24, 3, u, l);
        }

    // Zero on the input diagonal (nonsingular minors): must factor cleanly.
    for (char u : uplos)
        for (char l : lays) {
            fails += run_zerodiag<double, 4>(u, l);
            fails += run_zerodiag<float, 8>(u, l);
        }

    // Zero-pivot lane isolation (design 6.2): a poisoned lane must not
    // contaminate its siblings, over both uplo, both layouts, both precisions.
    for (char u : uplos)
        for (char l : lays) {
            fails += run_zeropivot<double, 4>(20, u, l);
            fails += run_zeropivot<float, 8>(16, u, l);
        }

    // 10 groups: takes the OpenMP group loop when the team has <= 10 threads.
    fails += run_case<double, 4>(40, 20, 'L', 'C');
    fails += run_solve<double, 4>(40, 20, 4, 'L', 'C');

    if (fails) {
        std::printf("\n%d CHECK(S) FAILED\n", fails);
        return 1;
    }
    std::printf("\nall checks passed\n");
    return 0;
}
