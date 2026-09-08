// test_cqr_geqrf_compact.cpp
//
// Self-contained validation of the templated compact QR factorization
// (dgeqrf_compact / sgeqrf_compact), with no BLAS dependency. The reference is
// the unblocked Householder QR (LAPACK dgeqr2 / dlarfg) implemented in scalar
// form -- the same algorithm the vectorized kernel executes V lanes at a time,
// so a correct kernel matches it to working precision.
//
// Checks per (T, V):
//   1. compact (H, tau)  ==  scalar geqr2 (H, tau)   (elementwise, ~eps*scale)
//   2. reconstruction  Q * triu(H) == A               (valid factorization)
//   3. reflectors are usable: ormqr_compact('T') then triangular solve
//      recovers a known X from B = A X                 (in-situ with ormqr)
// plus LAPACK-style argument validation of the C API.
//
// Assisted-by: Claude:claude-opus-4.8

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <limits>
#include <algorithm>

#include "test_compact_util.hpp" // C API shims, scalar references, MatrixBatch, pack/unpack

using namespace cqr::test;

// --------------------------- one test case --------------------------

template <class T, int V> static int run_case(int nm, int m, int n)
{
    const int k = std::min(m, n);
    const T eps = std::numeric_limits<T>::epsilon();

    // random A batch, diagonal-boosted so the columns stay well conditioned
    MatrixBatch<T> A(nm, m, n), Aref(nm, m, n), tau_ref(nm, k, 1);
    for (int idx = 0; idx < nm; ++idx) {
        gen_boosted(A[idx], m, n);
        std::copy(A[idx], A[idx] + (size_t)m * n, Aref[idx]); // Aref <- A
        ref_geqr2(m, n, Aref[idx], m, tau_ref[idx]);          // reference (H, tau)
    }

    // pack A, factor with the routine under test, unpack (H, tau)
    int ng = (nm + V - 1) / V;
    std::vector<T> ap((size_t)ng * m * n * V), tp((size_t)ng * k * V);
    pack_compact(A, ap.data(), m, V);

    int info = geqrf_c('C', m, n, ap.data(), m, tp.data(), V, nm);

    MatrixBatch<T> Aout(nm, m, n), tau_out(nm, k, 1);
    unpack_compact(Aout, ap.data(), m, V);
    unpack_tau(tau_out, tp.data(), V);

    // check 1: (H, tau) match the scalar reference elementwise
    double e_h = 0, e_t = 0;
    for (int idx = 0; idx < nm; ++idx) {
        e_h = std::max(e_h, max_abs_diff(Aout[idx], Aref[idx], (size_t)m * n));
        e_t = std::max(e_t, max_abs_diff(tau_out[idx], tau_ref[idx], (size_t)k));
    }

    // check 2: reconstruction Q * triu(H) == A (valid factorization)
    double e_rec = 0;
    for (int idx = 0; idx < nm; ++idx) {
        std::vector<T> Rec((size_t)m * n, T(0)); // start from R
        for (int j = 0; j < n; ++j)
            for (int i = 0; i <= std::min(j, k - 1); ++i)
                Rec[i + (size_t)j * m] = Aout(idx, i, j);
        ref_orm2r('N', m, n, k, Aout[idx], m, tau_out[idx], Rec.data(), m);
        e_rec = std::max(e_rec, max_abs_diff(Rec.data(), A[idx], (size_t)m * n));
    }

    // check 3: solve A X = B with the produced reflectors (square only), using
    // the project's ormqr kernel + a triangular solve. X(:,j) = j+1.
    double e_solve = -1;
    if (m == n) {
        const int nrhs = 3;
        const std::vector<T> X = known_solution<T>(n, nrhs);
        MatrixBatch<T> B(nm, n, nrhs);
        std::vector<T> bp((size_t)ng * n * nrhs * V);
        for (int idx = 0; idx < nm; ++idx)
            matmul(n, nrhs, n, A[idx], n, X.data(), n, B[idx], n); /* B = A X */
        pack_compact(B, bp.data(), n, V);
        ormqr_c('T', n, nrhs, k, ap.data(), n, tp.data(), bp.data(), n, V, nm);
        MatrixBatch<T> Bo(nm, n, nrhs);
        unpack_compact(Bo, bp.data(), n, V);
        e_solve = 0;
        for (int idx = 0; idx < nm; ++idx) {
            ref_trsm_upper(n, nrhs, Aout[idx], n, Bo[idx], n);
            e_solve =
                std::max(e_solve, max_abs_diff(Bo[idx], X.data(), (size_t)n * nrhs));
        }
    }

    const double scale = std::max(1, m);
    const double tol_fac = 200.0 * eps * scale; // same op sequence, ~eps
    const double tol_rec = 200.0 * eps * scale;
    const double tol_sol = 1e5 * eps * m; // cond(A)-dependent
    bool ok_h = e_h <= tol_fac, ok_t = e_t <= tol_fac, ok_r = e_rec <= tol_rec;
    bool ok_s = (e_solve < 0) || (e_solve <= tol_sol);

    std::printf("T=%-6s V=%-2d nm=%-2d m=%-3d n=%-3d | H:%.1e %s tau:%.1e %s rec:%.1e %s",
                sizeof(T) == 8 ? "double" : "float", V, nm, m, n, e_h,
                ok_h ? "OK" : "FAIL", e_t, ok_t ? "OK" : "FAIL", e_rec,
                ok_r ? "OK" : "FAIL");
    if (e_solve >= 0) std::printf(" solve:%.1e %s", e_solve, ok_s ? "OK" : "FAIL");
    std::printf(" | info=%d\n", info);
    return (info != 0) + !ok_h + !ok_t + !ok_r + !ok_s;
}

// --------------------- C API argument validation --------------------

static int test_validation()
{
    const int m = 8, n = 6, V = 4, nm = 4, ld = 8;
    std::vector<double> ap((size_t)ld * n * V, 0), tau((size_t)std::min(m, n) * V, 0);
    auto call = [&](char lay, int m_, int n_, int ldap_, int V_, int nm_) {
        return dgeqrf_compact(lay, m_, n_, ap.data(), ldap_, tau.data(), V_, nm_);
    };
    // clang-format off
    struct { const char *what; int got, want; } t[] = {
        {"valid col",   call('C', m, n,  ld,  V, nm),   0},
        {"valid row",   call('R', m, n,  n,   V, nm),   0},   // row-major ld >= n
        {"bad layout",  call('X', m, n,  ld,  V, nm),  -1},
        {"m<0",         call('C', -1, n, ld,  V, nm),  -2},
        {"n<0",         call('C', m, -1, ld,  V, nm),  -3},
        {"ldap<m",      call('C', m, n,  m-1, V, nm),  -5},
        {"bad V",       call('C', m, n,  ld,  3, nm),  -7},
        {"nm<0",        call('C', m, n,  ld,  V, -1),  -8},
        {"empty m=0",   call('C', 0, n,  1,   V, nm),   0},
        {"empty nm=0",  call('C', m, n,  ld,  V, 0),    0},
    };
    // clang-format on
    int bad = 0;
    for (auto &c : t)
        bad += (c.got != c.want);
    std::printf("C API validation: %zu checks | %s\n", sizeof(t) / sizeof(t[0]),
                bad ? "FAIL" : "OK");
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

    // square (the emphasis)
    fails += run_case<double, 2>(4, 30, 30);
    fails += run_case<double, 4>(8, 30, 30);
    fails += run_case<double, 8>(16, 60, 60);
    fails += run_case<double, 8>(11, 43, 43); // padded partial group
    fails += run_case<double, 4>(8, 3, 3);    // smallest supported
    // rectangular
    fails += run_case<double, 4>(8, 64, 20); // tall
    fails += run_case<double, 4>(8, 20, 64); // wide
    // float
    fails += run_case<float, 8>(16, 30, 30);
    fails += run_case<float, 16>(32, 43, 17);

    if (fails) {
        std::printf("\n%d CHECK(S) FAILED\n", fails);
        return 1;
    }
    std::printf("\nall checks passed\n");
    return 0;
}
