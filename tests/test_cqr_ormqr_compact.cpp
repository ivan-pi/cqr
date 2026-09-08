/* test_cqr_ormqr_compact.cpp
 *
 * Self-contained validation of the templated compact ormqr.
 * Reference: unblocked Householder QR (dgeqr2-style, LAPACK reflector
 * convention) + scalar dorm2r + back substitution, all templated on T.
 *
 * Test design: X(:,j) = j+1 (ones, twos, threes, ...), B = A*X, so the
 * b columns are scaled row sums of A; the solve must recover X.
 *
 * Checks per (T, V):
 *   1. compact Q^T B  ==  scalar dorm2r Q^T B (elementwise, ~eps)
 *   2. back substitution recovers X
 *   3. applying 'N' after 'T' recovers the original B  (Q Q^T = I)
 *
 * Assisted-by: Claude:claude-fable-5 Claude:claude-opus-4.8
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <limits>
#include <algorithm>

#include "test_compact_util.hpp" // compact<T>, scalar references, MatrixBatch, pack/unpack

using namespace cqr::test;

/* ----------------------- reference kernels (scalar) ----------------- */
/* ref_larfg / ref_geqr2 / ref_orm2r / ref_trsm_upper come from test_compact_util.hpp. */

/* Column-pivoted Householder QR (dgeqp3-style, greedy max trailing-column
 * norm). On exit A holds the reflectors below the diagonal and R on/above it
 * for the *permuted* matrix A(:,jpvt); jpvt[j] is the ORIGINAL (0-based)
 * column index placed at position j, so A(:,jpvt) = Q R. Used to drive the
 * kernel from a rank-revealing factorization + back-permutation solve --
 * the production (RBF-FD) use case the kernel must support. */
template <class T> static void ref_geqp3(int m, int n, T *A, int lda, int *jpvt, T *tau)
{
    const int k = std::min(m, n);
    for (int j = 0; j < n; ++j)
        jpvt[j] = j;

    for (int kk = 0; kk < k; ++kk) {
        int piv = kk;
        T best = -1;
        for (int j = kk; j < n; ++j) {
            T s = 0;
            for (int i = kk; i < m; ++i)
                s += A[i + (size_t)j * lda] * A[i + (size_t)j * lda];
            if (s > best) {
                best = s;
                piv = j;
            }
        }
        if (piv != kk) {
            for (int i = 0; i < m; ++i)
                std::swap(A[i + (size_t)kk * lda], A[i + (size_t)piv * lda]);
            std::swap(jpvt[kk], jpvt[piv]);
        }
        ref_larfg(m - kk, &A[kk + (size_t)kk * lda], &A[(kk + 1) + (size_t)kk * lda],
                  &tau[kk]);
        for (int j = kk + 1; j < n; ++j) {
            T w = A[kk + (size_t)j * lda];
            for (int i = kk + 1; i < m; ++i)
                w += A[i + (size_t)kk * lda] * A[i + (size_t)j * lda];
            A[kk + (size_t)j * lda] -= tau[kk] * w;
            for (int i = kk + 1; i < m; ++i)
                A[i + (size_t)j * lda] -= tau[kk] * A[i + (size_t)kk * lda] * w;
        }
    }
}

/* --------------------------- one test case -------------------------- */

template <class T, int V> static int run_case(int nm, int m, int nrhs)
{
    const int k = m;
    const T eps = std::numeric_limits<T>::epsilon();
    const double tol_exact = 100.0 * eps;   /* same op sequence */
    const double tol_solve = 1e5 * eps * m; /* cond(A)-dependent */

    const std::vector<T> X = known_solution<T>(m, nrhs);

    MatrixBatch<T> A(nm, m, m), Afac(nm, m, m), B(nm, m, nrhs), Bref(nm, m, nrhs),
        Bout(nm, m, nrhs), tau(nm, m, 1);
    for (int kk = 0; kk < nm; ++kk) {
        T *a = A[kk];
        gen_boosted(a, m, m); /* diagonal boost tames cond for float */
        matmul(m, nrhs, m, a, m, X.data(), m, B[kk], m); /* B = A X */

        std::copy(a, a + (size_t)m * m, Afac[kk]);
        ref_geqr2(m, m, Afac[kk], m, tau[kk]);
        std::copy(B[kk], B[kk] + (size_t)m * nrhs, Bref[kk]);
        ref_orm2r('T', m, nrhs, k, Afac[kk], m, tau[kk], Bref[kk], m);
    }

    int ng = (nm + V - 1) / V;
    std::vector<T> ap((size_t)ng * m * m * V), tp((size_t)ng * k * V),
        bp((size_t)ng * m * nrhs * V);
    pack_compact(Afac, ap.data(), m, V);
    pack_tau(tau, tp.data(), V);
    pack_compact(B, bp.data(), m, V);

    /* check 1: compact Q^T B vs scalar */
    compact<T>::ormqr('T', m, nrhs, k, ap.data(), m, tp.data(), bp.data(), m, V, nm);
    unpack_compact(Bout, bp.data(), m, V);
    double e1 = 0;
    for (int kk = 0; kk < nm; ++kk)
        e1 = std::max<double>(e1, max_abs_diff(Bout[kk], Bref[kk], (size_t)m * nrhs));

    /* check 2: solve recovers X */
    double e2 = 0;
    for (int kk = 0; kk < nm; ++kk) {
        ref_trsm_upper(m, nrhs, Afac[kk], m, Bout[kk], m);
        e2 = std::max<double>(e2, max_abs_diff(Bout[kk], X.data(), (size_t)m * nrhs));
    }

    /* check 3: 'N' undoes 'T' */
    compact<T>::ormqr('N', m, nrhs, k, ap.data(), m, tp.data(), bp.data(), m, V, nm);
    unpack_compact(Bout, bp.data(), m, V);
    double e3 = 0;
    for (int kk = 0; kk < nm; ++kk)
        e3 = std::max<double>(e3, max_abs_diff(Bout[kk], B[kk], (size_t)m * nrhs));

    /* scale-aware: B entries are O(m), QQ^t roundtrip accumulates a bit */
    bool ok1 = e1 <= tol_exact * m, ok2 = e2 <= tol_solve, ok3 = e3 <= tol_exact * m * 10;
    std::printf("T=%-6s V=%-2d nm=%-2d m=%-3d nrhs=%d | QtB: %.2e %s | solve X: %.2e %s "
                "| QQt=I: %.2e %s\n",
                compact<T>::name, V, nm, m, nrhs, e1, ok1 ? "OK" : "FAIL", e2,
                ok2 ? "OK" : "FAIL", e3, ok3 ? "OK" : "FAIL");
    return !ok1 + !ok2 + !ok3;
}

/* ------------------- pivoted-QR + back-permutation solve ------------ */
/* Salvaged from the former ArmPL cross-check: feed the kernel reflectors
 * from a column-pivoted (rank-revealing) QR and recover X through a
 * jpvt back-permutation, i.e. solve A x = b with A(:,jpvt) = Q R:
 *   R y = Q^T b   (kernel applies Q^T),   x(jpvt(j)) = y(j).
 * Unpivoted tests never exercise this end-to-end permuted pipeline. */
template <class T, int V> static int run_case_pivoted(int nm, int m, int nrhs)
{
    const int k = m; /* square, full rank */
    const T eps = std::numeric_limits<T>::epsilon();
    const double tol_solve = 1e5 * eps * m; /* cond(A)-dependent */

    const std::vector<T> X = known_solution<T>(m, nrhs);

    MatrixBatch<T> Afac(nm, m, m), B(nm, m, nrhs), Bout(nm, m, nrhs), tau(nm, m, 1);
    MatrixBatch<int> jpvt(nm, m, 1);
    for (int kk = 0; kk < nm; ++kk) {
        std::vector<T> A((size_t)m * m);
        gen_boosted(A.data(), m, m); /* diagonal boost tames cond */
        matmul(m, nrhs, m, A.data(), m, X.data(), m, B[kk], m); /* B = A X */

        std::copy(A.begin(), A.end(), Afac[kk]);
        ref_geqp3(m, m, Afac[kk], m, jpvt[kk], tau[kk]);
    }

    int ng = (nm + V - 1) / V;
    std::vector<T> ap((size_t)ng * m * m * V), tp((size_t)ng * k * V),
        bp((size_t)ng * m * nrhs * V);
    pack_compact(Afac, ap.data(), m, V);
    pack_tau(tau, tp.data(), V);
    pack_compact(B, bp.data(), m, V);

    /* kernel: c := Q^T b */
    compact<T>::ormqr('T', m, nrhs, k, ap.data(), m, tp.data(), bp.data(), m, V, nm);
    unpack_compact(Bout, bp.data(), m, V);

    /* R y = c, then back-permute x(jpvt(j)) = y(j); compare against X */
    double e = 0;
    std::vector<T> x((size_t)m * nrhs);
    for (int kk = 0; kk < nm; ++kk) {
        ref_trsm_upper(m, nrhs, Afac[kk], m, Bout[kk], m);
        for (int j = 0; j < nrhs; ++j)
            for (int i = 0; i < m; ++i)
                x[jpvt[kk][i] + (size_t)j * m] = Bout[kk][i + (size_t)j * m];
        e = std::max<double>(e, max_abs_diff(x.data(), X.data(), (size_t)m * nrhs));
    }

    bool ok = e <= tol_solve;
    std::printf("T=%-6s V=%-2d nm=%-2d m=%-3d nrhs=%d | pivoted solve X: %.2e %s\n",
                compact<T>::name, V, nm, m, nrhs, e, ok ? "OK" : "FAIL");
    return !ok;
}

/* --------------------------- micro-benchmark ------------------------ */

template <class T> static void bench(int V, int nm, int m, int nrhs, int reps)
{
    const int k = m;
    int ng = (nm + V - 1) / V;
    std::vector<T> ap((size_t)ng * m * m * V), tp((size_t)ng * k * V),
        bp((size_t)ng * m * nrhs * V);
    for (auto &x : ap)
        x = frand<T>() * T(1e-3);
    for (auto &x : tp)
        x = frand<T>() * T(1e-3);
    for (auto &x : bp)
        x = frand<T>();

    timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < reps; ++r)
        compact<T>::ormqr((r & 1) ? 'N' : 'T', m, nrhs, k, ap.data(), m, tp.data(),
                          bp.data(), m, V, nm);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double sec = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);

    double fl_mat = 0;
    for (int kk = 0; kk < k; ++kk)
        fl_mat += 4.0 * (m - kk) * nrhs;
    std::printf("T=%-6s V=%-2d nm=%-2d m=%-3d nrhs=%d | %8.3f us/rep | %7.2f GFLOP/s\n",
                compact<T>::name, V, nm, m, nrhs, 1e6 * sec / reps,
                fl_mat * nm * reps / sec * 1e-9);
    volatile T sink = bp[0];
    (void)sink;
}

/* --------------------- C API argument validation -------------------- */
/* The public C entry points must reject illegal arguments LAPACK-style
 * (return -j for the j-th argument), not assert or miscompute. */
static int test_validation()
{
    const int m = 8, nrhs = 2, k = 8, V = 4, nm = 4, ld = 8;
    std::vector<double> ap((size_t)ld * k * V, 0), tau((size_t)k * V, 0),
        bp((size_t)ld * nrhs * V, 0);
    auto call = [&](char tr, int m_, int nrhs_, int k_, int ldap_, int ldbp_, int V_,
                    int nm_) {
        return dormqr_compact(tr, m_, nrhs_, k_, ap.data(), ldap_, tau.data(), bp.data(),
                              ldbp_, V_, nm_);
    };

    // clang-format off
    struct { const char *what; int got, want; } t[] = {
        {"valid",          call('T', m, nrhs, k,   ld,    ld,    V, nm),   0},
        {"bad trans",      call('X', m, nrhs, k,   ld,    ld,    V, nm),  -1},
        {"m<0",            call('T', -1, nrhs, k,  ld,    ld,    V, nm),  -2},
        {"nrhs<0",         call('T', m, -1, k,     ld,    ld,    V, nm),  -3},
        {"k>m",            call('T', m, nrhs, m+1, ld,    ld,    V, nm),  -4},
        {"ldap<m",         call('T', m, nrhs, k,   m-1,   ld,    V, nm),  -6},
        {"ldbp<m",         call('T', m, nrhs, k,   ld,    m-1,   V, nm),  -9},
        {"bad V",          call('T', m, nrhs, k,   ld,    ld,    3, nm), -10},
        {"nm<0",           call('T', m, nrhs, k,   ld,    ld,    V, -1), -11},
        {"empty m=0",      call('T', 0, nrhs, 0,   1,     1,     V, nm),   0},
        {"empty nm=0",     call('T', m, nrhs, k,   ld,    ld,    V, 0),    0},
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

/* ------------------------------- main -------------------------------- */

int main(int argc, char **)
{
    int fails = 0;

    const int m = 43, nrhs = 5; /* nz=28 + npoly=15 */

    fails += test_validation();

    fails += run_case<double, 2>(4, m, nrhs);
    fails += run_case<double, 4>(8, m, nrhs);
    fails += run_case<double, 8>(16, m, nrhs);
    fails += run_case<double, 8>(11, m, nrhs); /* padded partial group */
    fails +=
        run_case<double, 4>(40, m, nrhs); /* 10 groups: OpenMP path when threads <= 10 */
    fails += run_case<float, 4>(8, m, nrhs);
    fails += run_case<float, 8>(16, m, nrhs);
    fails += run_case<float, 16>(32, m, nrhs);

    /* column-pivoted QR + back-permutation solve (salvaged cross-check) */
    fails += run_case_pivoted<double, 4>(8, m, nrhs);
    fails += run_case_pivoted<double, 8>(11, m, nrhs); /* padded partial group */
    fails += run_case_pivoted<float, 8>(16, m, nrhs);

    if (argc > 1) { /* run benchmark only when asked (skip under qemu) */
        std::printf("\n-- micro-benchmark (single core) --\n");
        bench<double>(8, 8, 43, 8, 200000);
        bench<double>(4, 8, 43, 8, 200000);
        bench<double>(2, 8, 43, 8, 200000);
        bench<float>(16, 16, 43, 8, 100000);
        bench<float>(8, 16, 43, 8, 100000);
    }

    if (fails) {
        std::printf("\n%d CHECK(S) FAILED\n", fails);
        return 1;
    }
    std::printf("\nall checks passed\n");
    return 0;
}
