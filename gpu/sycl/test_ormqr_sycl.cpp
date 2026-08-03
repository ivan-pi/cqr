/* test_ormqr_sycl.cpp
 *
 * ormqr validation for the SYCL backend, routed through the portable C API
 * (dormqr_compact / sormqr_compact) so the calls dispatch to the SYCL kernels
 * in cqr_compact_sycl.cpp.
 *
 * This mirrors the checks of src/test_cqr_compact.cpp. That CPU test calls the
 * internal C++ template cqr::detail::ormqr_compact<T,V> directly, so it cannot
 * be relinked onto a different backend; this file runs the same three numerical
 * checks against the same scalar reference and the same tolerances, but through
 * the C entry points:
 *   1. Q^T B            == scalar dorm2r Q^T B          (elementwise, ~eps)
 *   2. back substitution recovers a known X from B = A X
 *   3. applying 'N' after 'T' recovers B                (Q Q^T = I)
 * plus LAPACK-style C API argument validation (info = -j).
 *
 * (The companion geqrf suite, src/test_cqr_geqrf_compact.cpp, already drives the
 * SYCL geqrf AND ormqr through the C API when linked against this backend; this
 * file adds the trans='N' and round-trip coverage that suite's solve omits.)
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#include <cstdio>
#include <cmath>
#include <random>
#include <vector>
#include <limits>
#include <algorithm>

#include "cqr_compact.h"

// ----------------------- reference kernels (scalar) -----------------

template <class T> static void ref_larfg(int m, T *alpha, T *x, T *tau)
{
    T xnorm = 0;
    for (int i = 0; i < m - 1; ++i)
        xnorm = std::hypot(xnorm, x[i]);
    if (xnorm == T(0)) {
        *tau = 0;
        return;
    }
    T beta = -std::copysign(std::hypot(*alpha, xnorm), *alpha);
    *tau = (beta - *alpha) / beta;
    T scal = T(1) / (*alpha - beta);
    for (int i = 0; i < m - 1; ++i)
        x[i] *= scal;
    *alpha = beta;
}

template <class T> static void ref_geqr2(int m, int n, T *A, int lda, T *tau)
{
    int k = std::min(m, n);
    for (int kk = 0; kk < k; ++kk) {
        ref_larfg(m - kk, &A[kk + kk * lda], &A[(kk + 1) + kk * lda], &tau[kk]);
        for (int j = kk + 1; j < n; ++j) {
            T w = A[kk + j * lda];
            for (int i = kk + 1; i < m; ++i)
                w += A[i + kk * lda] * A[i + j * lda];
            A[kk + j * lda] -= tau[kk] * w;
            for (int i = kk + 1; i < m; ++i)
                A[i + j * lda] -= tau[kk] * A[i + kk * lda] * w;
        }
    }
}

template <class T>
static void ref_orm2r(char trans, int m, int nrhs, int k, const T *A, int lda,
                      const T *tau, T *B, int ldb)
{
    bool fwd = (trans == 'T');
    for (int s = 0; s < k; ++s) {
        int kk = fwd ? s : k - 1 - s;
        for (int j = 0; j < nrhs; ++j) {
            T w = B[kk + j * ldb];
            for (int i = kk + 1; i < m; ++i)
                w += A[i + kk * lda] * B[i + j * ldb];
            B[kk + j * ldb] -= tau[kk] * w;
            for (int i = kk + 1; i < m; ++i)
                B[i + j * ldb] -= tau[kk] * A[i + kk * lda] * w;
        }
    }
}

template <class T>
static void ref_trsm_upper(int n, int nrhs, const T *R, int lda, T *B, int ldb)
{
    for (int j = 0; j < nrhs; ++j)
        for (int i = n - 1; i >= 0; --i) {
            T s = B[i + j * ldb];
            for (int l = i + 1; l < n; ++l)
                s -= R[i + l * lda] * B[l + j * ldb];
            B[i + j * ldb] = s / R[i + i * lda];
        }
}

// ----------------------- compact pack / unpack ----------------------
// Column-major m x n matrices; element (i,j) of matrix idx = g*V+v lives at
// p[g*ldp*n*V + (j*ldp+i)*V + v]; padded slots carry the identity.

template <class T>
static void pack_compact(int m, int n, const std::vector<std::vector<T>> &Mk, int ldm,
                         T *p, int ldp, int V, int nm)
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
static void unpack_compact(int m, int n, std::vector<std::vector<T>> &Mk, int ldm,
                           const T *p, int ldp, int V, int nm)
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

// ----------------------------- helpers ------------------------------

static std::mt19937_64 rng(42);

template <class T> static T frand()
{
    static std::uniform_real_distribution<T> dist(T(-1), T(1));
    return dist(rng);
}

template <class T> static T max_abs_diff(const std::vector<T> &a, const std::vector<T> &b)
{
    T d = 0;
    for (size_t i = 0; i < a.size(); ++i)
        d = std::max(d, std::abs(a[i] - b[i]));
    return d;
}

// precision-overloaded C API shim
static int ormqr_c(char tr, int m, int nr, int k, const double *a, int lda,
                   const double *t, double *b, int ldb, int V, int nm)
{
    return dormqr_compact(tr, m, nr, k, a, lda, t, b, ldb, V, nm);
}
static int ormqr_c(char tr, int m, int nr, int k, const float *a, int lda, const float *t,
                   float *b, int ldb, int V, int nm)
{
    return sormqr_compact(tr, m, nr, k, a, lda, t, b, ldb, V, nm);
}

// --------------------------- one test case --------------------------

template <class T, int V> static int run_case(int nm, int m, int nrhs)
{
    const int k = m;
    const T eps = std::numeric_limits<T>::epsilon();
    const double tol_exact = 100.0 * eps; // same op sequence as the scalar ref
    const double tol_solve = 1e5 * eps * m; // cond(A)-dependent

    std::vector<T> X((size_t)m * nrhs);
    for (int j = 0; j < nrhs; ++j)
        for (int i = 0; i < m; ++i)
            X[i + (size_t)j * m] = T(j + 1); // ones, twos, threes, ...

    std::vector<std::vector<T>> A(nm), Afac(nm), B(nm), Bref(nm), Bout(nm), tau(nm);
    for (int idx = 0; idx < nm; ++idx) {
        A[idx].resize((size_t)m * m);
        B[idx].resize((size_t)m * nrhs);
        tau[idx].resize(m);
        Bout[idx].resize((size_t)m * nrhs);
        for (auto &x : A[idx])
            x = frand<T>();
        for (int i = 0; i < m; ++i)
            A[idx][i + (size_t)i * m] += T(2); // tame the conditioning

        for (int j = 0; j < nrhs; ++j) // B = A * X
            for (int i = 0; i < m; ++i) {
                T s = 0;
                for (int l = 0; l < m; ++l)
                    s += A[idx][i + (size_t)l * m] * X[l + (size_t)j * m];
                B[idx][i + (size_t)j * m] = s;
            }

        Afac[idx] = A[idx];
        ref_geqr2(m, m, Afac[idx].data(), m, tau[idx].data()); // reflectors + tau
        Bref[idx] = B[idx];
        ref_orm2r('T', m, nrhs, k, Afac[idx].data(), m, tau[idx].data(), Bref[idx].data(),
                  m); // reference Q^T B
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

    // check 1: SYCL Q^T B vs scalar reference
    int info1 = ormqr_c('T', m, nrhs, k, ap.data(), m, tp.data(), bp.data(), m, V, nm);
    unpack_compact(m, nrhs, Bout, m, bp.data(), m, V, nm);
    double e1 = 0;
    for (int idx = 0; idx < nm; ++idx)
        e1 = std::max(e1, (double)max_abs_diff(Bout[idx], Bref[idx]));

    // check 2: back substitution recovers X
    double e2 = 0;
    for (int idx = 0; idx < nm; ++idx) {
        ref_trsm_upper(m, nrhs, Afac[idx].data(), m, Bout[idx].data(), m);
        e2 = std::max(e2, (double)max_abs_diff(Bout[idx], X));
    }

    // check 3: 'N' undoes 'T' (Q Q^T = I); bp still holds Q^T B
    int info3 = ormqr_c('N', m, nrhs, k, ap.data(), m, tp.data(), bp.data(), m, V, nm);
    unpack_compact(m, nrhs, Bout, m, bp.data(), m, V, nm);
    double e3 = 0;
    for (int idx = 0; idx < nm; ++idx)
        e3 = std::max(e3, (double)max_abs_diff(Bout[idx], B[idx]));

    bool ok1 = e1 <= tol_exact * m, ok2 = e2 <= tol_solve,
         ok3 = e3 <= tol_exact * m * 10;
    std::printf("T=%-6s V=%-2d nm=%-2d m=%-3d nrhs=%d | QtB:%.1e %s solve:%.1e %s "
                "QQt=I:%.1e %s | info=%d,%d\n",
                sizeof(T) == 8 ? "double" : "float", V, nm, m, nrhs, e1,
                ok1 ? "OK" : "FAIL", e2, ok2 ? "OK" : "FAIL", e3, ok3 ? "OK" : "FAIL",
                info1, info3);
    return (info1 != 0) + (info3 != 0) + !ok1 + !ok2 + !ok3;
}

// --------------------- C API argument validation --------------------

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
        {"valid",       call('T', m, nrhs, k,   ld,  ld,  V, nm),   0},
        {"bad trans",   call('X', m, nrhs, k,   ld,  ld,  V, nm),  -1},
        {"m<0",         call('T', -1, nrhs, k,  ld,  ld,  V, nm),  -2},
        {"nrhs<0",      call('T', m, -1, k,     ld,  ld,  V, nm),  -3},
        {"k>m",         call('T', m, nrhs, m+1, ld,  ld,  V, nm),  -4},
        {"ldap<m",      call('T', m, nrhs, k,   m-1, ld,  V, nm),  -6},
        {"ldbp<m",      call('T', m, nrhs, k,   ld,  m-1, V, nm),  -9},
        {"bad V",       call('T', m, nrhs, k,   ld,  ld,  3, nm), -10},
        {"nm<0",        call('T', m, nrhs, k,   ld,  ld,  V, -1), -11},
        {"empty m=0",   call('T', 0, nrhs, 0,   1,   1,   V, nm),   0},
        {"empty nm=0",  call('T', m, nrhs, k,   ld,  ld,  V, 0),    0},
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

    fails += run_case<double, 2>(4, 43, 5);
    fails += run_case<double, 4>(8, 43, 5);
    fails += run_case<double, 8>(16, 43, 5);
    fails += run_case<double, 8>(11, 43, 5); // padded partial group
    fails += run_case<float, 4>(8, 43, 5);
    fails += run_case<float, 8>(16, 43, 5);
    fails += run_case<float, 16>(32, 43, 5);

    if (fails) {
        std::printf("\n%d CHECK(S) FAILED\n", fails);
        return 1;
    }
    std::printf("\nall checks passed\n");
    return 0;
}
