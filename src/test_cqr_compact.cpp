/* test_cqr_compact.cpp
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
 * Build (native):   g++ -O3 -march=native -std=c++17 cqr_compact_dispatch.cpp test_cqr_compact.cpp -o test_cqr
 * Build (AArch64):  aarch64-linux-gnu-g++ -O3 -march=armv8.2-a -std=c++17 -static ...
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <random>
#include <vector>
#include <limits>
#include <algorithm>

#include "cqr_compact.h"
#include "cqr_compact.hpp"

/* ----------------------- reference kernels (scalar) ----------------- */

template <class T>
static void ref_larfg(int m, T *alpha, T *x, T *tau)
{
    T xnorm = 0;
    for (int i = 0; i < m - 1; ++i) xnorm = std::hypot(xnorm, x[i]);
    if (xnorm == T(0)) { *tau = 0; return; }
    T beta = -std::copysign(std::hypot(*alpha, xnorm), *alpha);
    *tau = (beta - *alpha) / beta;
    T scal = T(1) / (*alpha - beta);
    for (int i = 0; i < m - 1; ++i) x[i] *= scal;
    *alpha = beta;
}

template <class T>
static void ref_geqr2(int m, int n, T *A, int lda, T *tau)
{
    int k = std::min(m, n);
    for (int kk = 0; kk < k; ++kk) {
        ref_larfg(m - kk, &A[kk + kk * lda], &A[(kk + 1) + kk * lda], &tau[kk]);
        for (int j = kk + 1; j < n; ++j) {
            T w = A[kk + j * lda];
            for (int i = kk + 1; i < m; ++i) w += A[i + kk * lda] * A[i + j * lda];
            A[kk + j * lda] -= tau[kk] * w;
            for (int i = kk + 1; i < m; ++i) A[i + j * lda] -= tau[kk] * A[i + kk * lda] * w;
        }
    }
}

/* Column-pivoted Householder QR (dgeqp3-style, greedy max trailing-column
 * norm). On exit A holds the reflectors below the diagonal and R on/above it
 * for the *permuted* matrix A(:,jpvt); jpvt[j] is the ORIGINAL (0-based)
 * column index placed at position j, so A(:,jpvt) = Q R. Used to drive the
 * kernel from a rank-revealing factorization + back-permutation solve --
 * the production (RBF-FD) use case the kernel must support. */
template <class T>
static void ref_geqp3(int m, int n, T *A, int lda, int *jpvt, T *tau)
{
    const int k = std::min(m, n);
    for (int j = 0; j < n; ++j) jpvt[j] = j;

    for (int kk = 0; kk < k; ++kk) {
        int piv = kk;
        T best = -1;
        for (int j = kk; j < n; ++j) {
            T s = 0;
            for (int i = kk; i < m; ++i) s += A[i + (size_t)j * lda] * A[i + (size_t)j * lda];
            if (s > best) { best = s; piv = j; }
        }
        if (piv != kk) {
            for (int i = 0; i < m; ++i) std::swap(A[i + (size_t)kk * lda], A[i + (size_t)piv * lda]);
            std::swap(jpvt[kk], jpvt[piv]);
        }
        ref_larfg(m - kk, &A[kk + (size_t)kk * lda], &A[(kk + 1) + (size_t)kk * lda], &tau[kk]);
        for (int j = kk + 1; j < n; ++j) {
            T w = A[kk + (size_t)j * lda];
            for (int i = kk + 1; i < m; ++i) w += A[i + (size_t)kk * lda] * A[i + (size_t)j * lda];
            A[kk + (size_t)j * lda] -= tau[kk] * w;
            for (int i = kk + 1; i < m; ++i) A[i + (size_t)j * lda] -= tau[kk] * A[i + (size_t)kk * lda] * w;
        }
    }
}

template <class T>
static void ref_orm2r(char trans, int m, int nrhs, int k,
                      const T *A, int lda, const T *tau, T *B, int ldb)
{
    bool fwd = (trans == 'T');
    for (int s = 0; s < k; ++s) {
        int kk = fwd ? s : k - 1 - s;
        for (int j = 0; j < nrhs; ++j) {
            T w = B[kk + j * ldb];
            for (int i = kk + 1; i < m; ++i) w += A[i + kk * lda] * B[i + j * ldb];
            B[kk + j * ldb] -= tau[kk] * w;
            for (int i = kk + 1; i < m; ++i) B[i + j * ldb] -= tau[kk] * A[i + kk * lda] * w;
        }
    }
}

template <class T>
static void ref_trsm_upper(int n, int nrhs, const T *R, int lda, T *B, int ldb)
{
    for (int j = 0; j < nrhs; ++j)
        for (int i = n - 1; i >= 0; --i) {
            T s = B[i + j * ldb];
            for (int l = i + 1; l < n; ++l) s -= R[i + l * lda] * B[l + j * ldb];
            B[i + j * ldb] = s / R[i + i * lda];
        }
}

/* ----------------------- compact pack / unpack ---------------------- */

template <class T>
static void pack_compact(int m, int n, const std::vector<std::vector<T>> &Mk,
                         int ldm, T *p, int ldp, int V, int nm)
{
    int ng = (nm + V - 1) / V;
    for (int g = 0; g < ng; ++g)
        for (int v = 0; v < V; ++v) {
            int idx = g * V + v;
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < m; ++i)
                    p[(size_t)g * ldp * n * V + ((size_t)j * ldp + i) * V + v] =
                        (idx < nm) ? Mk[idx][i + (size_t)j * ldm]
                                   : (i == j ? T(1) : T(0));
        }
}

template <class T>
static void unpack_compact(int m, int n, std::vector<std::vector<T>> &Mk,
                           int ldm, const T *p, int ldp, int V, int nm)
{
    int ng = (nm + V - 1) / V;
    for (int g = 0; g < ng; ++g)
        for (int v = 0; v < V; ++v) {
            int idx = g * V + v;
            if (idx >= nm) continue;
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < m; ++i)
                    Mk[idx][i + (size_t)j * ldm] =
                        p[(size_t)g * ldp * n * V + ((size_t)j * ldp + i) * V + v];
        }
}

/* ----------------------------- helpers ------------------------------ */

static std::mt19937_64 rng(42);

template <class T>
static T frand()
{
    /* one distribution per instantiation (T), reused across calls */
    static std::uniform_real_distribution<T> dist(T(-1), T(1));
    return dist(rng);
}

template <class T>
static T max_abs_diff(const std::vector<T> &a, const std::vector<T> &b)
{
    T d = 0;
    for (size_t i = 0; i < a.size(); ++i) d = std::max(d, std::abs(a[i] - b[i]));
    return d;
}

/* --------------------------- one test case -------------------------- */

template <class T, int V>
static int run_case(int nm, int m, int nrhs)
{
    const int k = m;
    const T eps = std::numeric_limits<T>::epsilon();
    const double tol_exact = 100.0 * eps;        /* same op sequence */
    const double tol_solve = 1e5 * eps * m;      /* cond(A)-dependent */

    std::vector<T> X((size_t)m * nrhs);
    for (int j = 0; j < nrhs; ++j)
        for (int i = 0; i < m; ++i)
            X[i + (size_t)j * m] = T(j + 1);     /* ones, twos, threes, ... */

    std::vector<std::vector<T>> A(nm), Afac(nm), B(nm), Bref(nm), Bout(nm), tau(nm);
    for (int kk = 0; kk < nm; ++kk) {
        A[kk].resize((size_t)m * m);
        B[kk].resize((size_t)m * nrhs);
        tau[kk].resize(m);
        for (auto &x : A[kk]) x = frand<T>();
        for (int i = 0; i < m; ++i) A[kk][i + (size_t)i * m] += T(2); /* tame cond for float */

        for (int j = 0; j < nrhs; ++j)            /* B = A*X */
            for (int i = 0; i < m; ++i) {
                T s = 0;
                for (int l = 0; l < m; ++l) s += A[kk][i + (size_t)l * m] * X[l + (size_t)j * m];
                B[kk][i + (size_t)j * m] = s;
            }

        Afac[kk] = A[kk];
        ref_geqr2(m, m, Afac[kk].data(), m, tau[kk].data());
        Bref[kk] = B[kk];
        ref_orm2r('T', m, nrhs, k, Afac[kk].data(), m, tau[kk].data(), Bref[kk].data(), m);
        Bout[kk].resize((size_t)m * nrhs);
    }

    int ng = (nm + V - 1) / V;
    std::vector<T> ap((size_t)ng * m * m * V), tp((size_t)ng * k * V),
                   bp((size_t)ng * m * nrhs * V);
    pack_compact(m, m, Afac, m, ap.data(), m, V, nm);
    {   /* tau as k x 1 matrices */
        for (int g = 0; g < ng; ++g)
            for (int v = 0; v < V; ++v) {
                int idx = g * V + v;
                for (int kk = 0; kk < k; ++kk)
                    tp[(size_t)g * k * V + (size_t)kk * V + v] =
                        (idx < nm) ? tau[idx][kk] : T(0);
            }
    }
    pack_compact(m, nrhs, B, m, bp.data(), m, V, nm);

    /* check 1: compact Q^T B vs scalar */
    cqr::detail::ormqr_compact<T, V>('T', m, nrhs, k, ap.data(), m, m, tp.data(),
                               bp.data(), m, nm);
    unpack_compact(m, nrhs, Bout, m, bp.data(), m, V, nm);
    double e1 = 0;
    for (int kk = 0; kk < nm; ++kk) e1 = std::max<double>(e1, max_abs_diff(Bout[kk], Bref[kk]));

    /* check 2: solve recovers X */
    double e2 = 0;
    for (int kk = 0; kk < nm; ++kk) {
        ref_trsm_upper(m, nrhs, Afac[kk].data(), m, Bout[kk].data(), m);
        e2 = std::max<double>(e2, max_abs_diff(Bout[kk], X));
    }

    /* check 3: 'N' undoes 'T' */
    cqr::detail::ormqr_compact<T, V>('N', m, nrhs, k, ap.data(), m, m, tp.data(),
                               bp.data(), m, nm);
    unpack_compact(m, nrhs, Bout, m, bp.data(), m, V, nm);
    double e3 = 0;
    for (int kk = 0; kk < nm; ++kk) e3 = std::max<double>(e3, max_abs_diff(Bout[kk], B[kk]));

    /* scale-aware: B entries are O(m), QQ^t roundtrip accumulates a bit */
    bool ok1 = e1 <= tol_exact * m, ok2 = e2 <= tol_solve, ok3 = e3 <= tol_exact * m * 10;
    std::printf("T=%-6s V=%-2d nm=%-2d m=%-3d nrhs=%d | QtB: %.2e %s | solve X: %.2e %s | QQt=I: %.2e %s\n",
                sizeof(T) == 8 ? "double" : "float", V, nm, m, nrhs,
                e1, ok1 ? "OK" : "FAIL", e2, ok2 ? "OK" : "FAIL", e3, ok3 ? "OK" : "FAIL");
    return !ok1 + !ok2 + !ok3;
}

/* ------------------- pivoted-QR + back-permutation solve ------------ */
/* Salvaged from the former ArmPL cross-check: feed the kernel reflectors
 * from a column-pivoted (rank-revealing) QR and recover X through a
 * jpvt back-permutation, i.e. solve A x = b with A(:,jpvt) = Q R:
 *   R y = Q^T b   (kernel applies Q^T),   x(jpvt(j)) = y(j).
 * Unpivoted tests never exercise this end-to-end permuted pipeline. */
template <class T, int V>
static int run_case_pivoted(int nm, int m, int nrhs)
{
    const int k = m;                              /* square, full rank */
    const T eps = std::numeric_limits<T>::epsilon();
    const double tol_solve = 1e5 * eps * m;       /* cond(A)-dependent */

    std::vector<T> X((size_t)m * nrhs);
    for (int j = 0; j < nrhs; ++j)
        for (int i = 0; i < m; ++i)
            X[i + (size_t)j * m] = T(j + 1);      /* ones, twos, threes, ... */

    std::vector<std::vector<T>> Afac(nm), B(nm), Bout(nm), tau(nm);
    std::vector<std::vector<int>> jpvt(nm);
    for (int kk = 0; kk < nm; ++kk) {
        std::vector<T> A((size_t)m * m);
        for (auto &x : A) x = frand<T>();
        for (int i = 0; i < m; ++i) A[i + (size_t)i * m] += T(2);  /* tame cond */

        B[kk].resize((size_t)m * nrhs);
        for (int j = 0; j < nrhs; ++j)            /* B = A*X */
            for (int i = 0; i < m; ++i) {
                T s = 0;
                for (int l = 0; l < m; ++l) s += A[i + (size_t)l * m] * X[l + (size_t)j * m];
                B[kk][i + (size_t)j * m] = s;
            }

        Afac[kk] = A;
        tau[kk].resize(m);
        jpvt[kk].resize(m);
        ref_geqp3(m, m, Afac[kk].data(), m, jpvt[kk].data(), tau[kk].data());
        Bout[kk].resize((size_t)m * nrhs);
    }

    int ng = (nm + V - 1) / V;
    std::vector<T> ap((size_t)ng * m * m * V), tp((size_t)ng * k * V),
                   bp((size_t)ng * m * nrhs * V);
    pack_compact(m, m, Afac, m, ap.data(), m, V, nm);
    for (int g = 0; g < ng; ++g)
        for (int v = 0; v < V; ++v) {
            int idx = g * V + v;
            for (int kk = 0; kk < k; ++kk)
                tp[(size_t)g * k * V + (size_t)kk * V + v] =
                    (idx < nm) ? tau[idx][kk] : T(0);
        }
    pack_compact(m, nrhs, B, m, bp.data(), m, V, nm);

    /* kernel: c := Q^T b */
    cqr::detail::ormqr_compact<T, V>('T', m, nrhs, k, ap.data(), m, m, tp.data(),
                               bp.data(), m, nm);
    unpack_compact(m, nrhs, Bout, m, bp.data(), m, V, nm);

    /* R y = c, then back-permute x(jpvt(j)) = y(j); compare against X */
    double e = 0;
    std::vector<T> x((size_t)m * nrhs);
    for (int kk = 0; kk < nm; ++kk) {
        ref_trsm_upper(m, nrhs, Afac[kk].data(), m, Bout[kk].data(), m);
        for (int j = 0; j < nrhs; ++j)
            for (int i = 0; i < m; ++i)
                x[jpvt[kk][i] + (size_t)j * m] = Bout[kk][i + (size_t)j * m];
        e = std::max<double>(e, max_abs_diff(x, X));
    }

    bool ok = e <= tol_solve;
    std::printf("T=%-6s V=%-2d nm=%-2d m=%-3d nrhs=%d | pivoted solve X: %.2e %s\n",
                sizeof(T) == 8 ? "double" : "float", V, nm, m, nrhs,
                e, ok ? "OK" : "FAIL");
    return !ok;
}

/* --------------------------- micro-benchmark ------------------------ */

template <class T>
static void bench(int V, int nm, int m, int nrhs, int reps)
{
    const int k = m;
    int ng = (nm + V - 1) / V;
    std::vector<T> ap((size_t)ng * m * m * V), tp((size_t)ng * k * V),
                   bp((size_t)ng * m * nrhs * V);
    for (auto &x : ap) x = frand<T>() * T(1e-3);
    for (auto &x : tp) x = frand<T>() * T(1e-3);
    for (auto &x : bp) x = frand<T>();

    timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < reps; ++r) {
        if (sizeof(T) == 8)
            dormqr_compact((r & 1) ? 'N' : 'T', m, nrhs, k,
                           (const double *)ap.data(), m, m, (const double *)tp.data(),
                           (double *)bp.data(), m, V, nm);
        else
            sormqr_compact((r & 1) ? 'N' : 'T', m, nrhs, k,
                           (const float *)ap.data(), m, m, (const float *)tp.data(),
                           (float *)bp.data(), m, V, nm);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double sec = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);

    double fl_mat = 0;
    for (int kk = 0; kk < k; ++kk) fl_mat += 4.0 * (m - kk) * nrhs;
    std::printf("T=%-6s V=%-2d nm=%-2d m=%-3d nrhs=%d | %8.3f us/rep | %7.2f GFLOP/s\n",
                sizeof(T) == 8 ? "double" : "float", V, nm, m, nrhs,
                1e6 * sec / reps, fl_mat * nm * reps / sec * 1e-9);
    volatile T sink = bp[0]; (void)sink;
}

/* --------------------- C API argument validation -------------------- */
/* The public C entry points must reject illegal arguments LAPACK-style
 * (return -j for the j-th argument), not assert or miscompute. */
static int test_validation()
{
    const int m = 8, nrhs = 2, k = 8, V = 4, nm = 4, ld = 8, nca = 8;
    std::vector<double> ap((size_t)ld * nca * V, 0), tau((size_t)k * V, 0),
                        bp((size_t)ld * nrhs * V, 0);
    auto call = [&](char tr, int m_, int nrhs_, int k_, int ldap_, int nca_,
                    int ldbp_, int V_, int nm_) {
        return dormqr_compact(tr, m_, nrhs_, k_, ap.data(), ldap_, nca_,
                              tau.data(), bp.data(), ldbp_, V_, nm_);
    };

    struct { const char *what; int got, want; } t[] = {
        {"valid",          call('T', m, nrhs, k,   ld,    nca,   ld,    V, nm),   0},
        {"bad trans",      call('X', m, nrhs, k,   ld,    nca,   ld,    V, nm),  -1},
        {"m<0",            call('T', -1, nrhs, k,  ld,    nca,   ld,    V, nm),  -2},
        {"nrhs<0",         call('T', m, -1, k,     ld,    nca,   ld,    V, nm),  -3},
        {"k>m",            call('T', m, nrhs, m+1, ld,    nca,   ld,    V, nm),  -4},
        {"ldap<m",         call('T', m, nrhs, k,   m-1,   nca,   ld,    V, nm),  -6},
        {"ncols_a<k",      call('T', m, nrhs, k,   ld,    k-1,   ld,    V, nm),  -7},
        {"ldbp<m",         call('T', m, nrhs, k,   ld,    nca,   m-1,   V, nm), -10},
        {"bad V",          call('T', m, nrhs, k,   ld,    nca,   ld,    3, nm), -11},
        {"nm<0",           call('T', m, nrhs, k,   ld,    nca,   ld,    V, -1), -12},
        {"empty m=0",      call('T', 0, nrhs, 0,   1,     0,     1,     V, nm),   0},
        {"empty nm=0",     call('T', m, nrhs, k,   ld,    nca,   ld,    V, 0),    0},
    };
    int bad = 0;
    for (auto &c : t) bad += (c.got != c.want);
    std::printf("C API validation: %zu checks | %s\n",
                sizeof(t) / sizeof(t[0]), bad ? "FAIL" : "OK");
    for (auto &c : t)
        if (c.got != c.want)
            std::printf("  %-12s got=%d want=%d\n", c.what, c.got, c.want);
    return bad ? 1 : 0;
}

/* ------------------------------- main -------------------------------- */

int main(int argc, char **)
{
    int fails = 0;

    const int m = 43, nrhs = 5;  /* nz=28 + npoly=15 */

    fails += test_validation();

    fails += run_case<double, 2>(4, m, nrhs);
    fails += run_case<double, 4>(8, m, nrhs);
    fails += run_case<double, 8>(16, m, nrhs);
    fails += run_case<double, 8>(11, m, nrhs);   /* padded partial group */
    fails += run_case<float, 4>(8, m, nrhs);
    fails += run_case<float, 8>(16, m, nrhs);
    fails += run_case<float, 16>(32, m, nrhs);

    /* column-pivoted QR + back-permutation solve (salvaged cross-check) */
    fails += run_case_pivoted<double, 4>(8, m, nrhs);
    fails += run_case_pivoted<double, 8>(11, m, nrhs);   /* padded partial group */
    fails += run_case_pivoted<float, 8>(16, m, nrhs);

    if (argc > 1) {  /* run benchmark only when asked (skip under qemu) */
        std::printf("\n-- micro-benchmark (single core) --\n");
        bench<double>(8, 8, 43, 8, 200000);
        bench<double>(4, 8, 43, 8, 200000);
        bench<double>(2, 8, 43, 8, 200000);
        bench<float>(16, 16, 43, 8, 100000);
        bench<float>(8, 16, 43, 8, 100000);
    }

    if (fails) { std::printf("\n%d CHECK(S) FAILED\n", fails); return 1; }
    std::printf("\nall checks passed\n");
    return 0;
}
