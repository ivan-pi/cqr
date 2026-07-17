/* test_cqr_geqrf_compact.cpp
 *
 * Self-contained validation of the templated compact QR factorization
 * (dgeqrf_compact / sgeqrf_compact), with no BLAS dependency. The reference is
 * the unblocked Householder QR (LAPACK dgeqr2 / dlarfg) implemented in scalar
 * form -- the same algorithm the vectorized kernel executes V lanes at a time,
 * so a correct kernel matches it to working precision.
 *
 * Checks per (T, V):
 *   1. compact (H, tau)  ==  scalar geqr2 (H, tau)   (elementwise, ~eps*scale)
 *   2. reconstruction  Q * triu(H) == A               (valid factorization)
 *   3. reflectors are usable: ormqr_compact('T') then triangular solve
 *      recovers a known X from B = A X                 (in-situ with ormqr)
 * plus LAPACK-style argument validation of the C API.
 *
 * Build (native): g++ -O3 -march=native -std=c++17 cqr_compact_dispatch.cpp \
 *                 cqr_geqrf_compact_dispatch.cpp test_cqr_geqrf_compact.cpp -o t
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>
#include <limits>
#include <algorithm>

#include "cqr_geqrf_compact.h"
#include "cqr_geqrf_compact.hpp"
#include "cqr_compact.h"       /* dormqr_compact, to exercise the reflectors */

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

/* Apply Q (side='L', trans) to a dense m x nrhs B in place (scalar dorm2r),
 * for the reconstruction check (Q * R) and the solve check. */
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
/* Column-major m x n matrices: element (i,j) of matrix idx = g*V+v lives at
 * p[g*ldp*n*V + (j*ldp+i)*V + v]; padded slots carry the identity. */

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

static std::mt19937_64 rng(12345);

template <class T>
static T frand()
{
    static std::uniform_real_distribution<T> dist(T(-1), T(1));
    return dist(rng);
}

template <class T>
static double max_abs_diff(const std::vector<T> &a, const std::vector<T> &b)
{
    double d = 0;
    for (size_t i = 0; i < a.size(); ++i) d = std::max(d, (double)std::abs(a[i] - b[i]));
    return d;
}

/* precision-overloaded shims: pick d/s by the pointer type */
static int geqrf_c(char l, int m, int n, double *a, int ld, double *t, int V, int nm)
{ return dgeqrf_compact(l, m, n, a, ld, t, V, nm); }
static int geqrf_c(char l, int m, int n, float *a, int ld, float *t, int V, int nm)
{ return sgeqrf_compact(l, m, n, a, ld, t, V, nm); }
static void ormqr_c(char tr, int m, int nr, int k, const double *a, int lda, const double *t, double *b, int ldb, int V, int nm)
{ dormqr_compact(tr, m, nr, k, a, lda, t, b, ldb, V, nm); }
static void ormqr_c(char tr, int m, int nr, int k, const float *a, int lda, const float *t, float *b, int ldb, int V, int nm)
{ sormqr_compact(tr, m, nr, k, a, lda, t, b, ldb, V, nm); }

/* --------------------------- one test case -------------------------- */

template <class T, int V>
static int run_case(int nm, int m, int n)
{
    const int k = std::min(m, n);
    const T eps = std::numeric_limits<T>::epsilon();

    /* random A batch, diagonal-boosted so the columns stay well conditioned */
    std::vector<std::vector<T>> A(nm), Aref(nm), tau_ref(nm);
    for (int idx = 0; idx < nm; ++idx) {
        A[idx].resize((size_t)m * n);
        for (auto &x : A[idx]) x = frand<T>();
        for (int i = 0; i < std::min(m, n); ++i) A[idx][i + (size_t)i * m] += T(2);
        Aref[idx] = A[idx];
        tau_ref[idx].assign(k, T(0));
        ref_geqr2(m, n, Aref[idx].data(), m, tau_ref[idx].data());   /* reference */
    }

    /* pack A, factor with the routine under test, unpack (H, tau) */
    int ng = (nm + V - 1) / V;
    std::vector<T> ap((size_t)ng * m * n * V), tp((size_t)ng * k * V);
    pack_compact(m, n, A, m, ap.data(), m, V, nm);

    int info = geqrf_c('C', m, n, ap.data(), m, tp.data(), V, nm);

    std::vector<std::vector<T>> Aout(nm), tau_out(nm);
    for (int idx = 0; idx < nm; ++idx) { Aout[idx].resize((size_t)m * n); tau_out[idx].resize(k); }
    unpack_compact(m, n, Aout, m, ap.data(), m, V, nm);
    for (int g = 0; g < ng; ++g)
        for (int v = 0; v < V; ++v) {
            int idx = g * V + v;
            if (idx >= nm) continue;
            for (int kk = 0; kk < k; ++kk)
                tau_out[idx][kk] = tp[(size_t)g * k * V + (size_t)kk * V + v];
        }

    /* check 1: (H, tau) match the scalar reference elementwise */
    double e_h = 0, e_t = 0;
    for (int idx = 0; idx < nm; ++idx) {
        e_h = std::max(e_h, max_abs_diff(Aout[idx], Aref[idx]));
        e_t = std::max(e_t, max_abs_diff(tau_out[idx], tau_ref[idx]));
    }

    /* check 2: reconstruction Q * triu(H) == A (valid factorization) */
    double e_rec = 0;
    for (int idx = 0; idx < nm; ++idx) {
        std::vector<T> Rec((size_t)m * n, T(0));            /* start from R */
        for (int j = 0; j < n; ++j)
            for (int i = 0; i <= std::min(j, k - 1); ++i)
                Rec[i + (size_t)j * m] = Aout[idx][i + (size_t)j * m];
        ref_orm2r('N', m, n, k, Aout[idx].data(), m, tau_out[idx].data(), Rec.data(), m);
        e_rec = std::max(e_rec, max_abs_diff(Rec, A[idx]));
    }

    /* check 3: solve A X = B with the produced reflectors (square only), using
     * the project's ormqr kernel + a triangular solve. X(:,j) = j+1. */
    double e_solve = -1;
    if (m == n) {
        const int nrhs = 3;
        std::vector<T> X((size_t)n * nrhs);
        for (int j = 0; j < nrhs; ++j)
            for (int i = 0; i < n; ++i) X[i + (size_t)j * n] = T(j + 1);
        std::vector<std::vector<T>> B(nm);
        std::vector<T> bp((size_t)ng * n * nrhs * V);
        for (int idx = 0; idx < nm; ++idx) {
            B[idx].assign((size_t)n * nrhs, T(0));
            for (int j = 0; j < nrhs; ++j)
                for (int i = 0; i < n; ++i) {
                    T s = 0;
                    for (int l = 0; l < n; ++l) s += A[idx][i + (size_t)l * n] * X[l + (size_t)j * n];
                    B[idx][i + (size_t)j * n] = s;
                }
        }
        pack_compact(n, nrhs, B, n, bp.data(), n, V, nm);
        ormqr_c('T', n, nrhs, k, ap.data(), n, tp.data(), bp.data(), n, V, nm);
        std::vector<std::vector<T>> Bo(nm);
        for (int idx = 0; idx < nm; ++idx) Bo[idx].resize((size_t)n * nrhs);
        unpack_compact(n, nrhs, Bo, n, bp.data(), n, V, nm);
        e_solve = 0;
        for (int idx = 0; idx < nm; ++idx) {
            ref_trsm_upper(n, nrhs, Aout[idx].data(), n, Bo[idx].data(), n);
            e_solve = std::max(e_solve, max_abs_diff(Bo[idx], X));
        }
    }

    const double scale = std::max(1, m);
    const double tol_fac = 200.0 * eps * scale;   /* same op sequence, ~eps */
    const double tol_rec = 200.0 * eps * scale;
    const double tol_sol = 1e5 * eps * m;         /* cond(A)-dependent */
    bool ok_h = e_h <= tol_fac, ok_t = e_t <= tol_fac, ok_r = e_rec <= tol_rec;
    bool ok_s = (e_solve < 0) || (e_solve <= tol_sol);

    std::printf("T=%-6s V=%-2d nm=%-2d m=%-3d n=%-3d | H:%.1e %s tau:%.1e %s rec:%.1e %s",
                sizeof(T) == 8 ? "double" : "float", V, nm, m, n,
                e_h, ok_h ? "OK" : "FAIL", e_t, ok_t ? "OK" : "FAIL",
                e_rec, ok_r ? "OK" : "FAIL");
    if (e_solve >= 0) std::printf(" solve:%.1e %s", e_solve, ok_s ? "OK" : "FAIL");
    std::printf(" | info=%d\n", info);
    return (info != 0) + !ok_h + !ok_t + !ok_r + !ok_s;
}

/* --------------------- C API argument validation -------------------- */

static int test_validation()
{
    const int m = 8, n = 6, V = 4, nm = 4, ld = 8;
    std::vector<double> ap((size_t)ld * n * V, 0), tau((size_t)std::min(m, n) * V, 0);
    auto call = [&](char lay, int m_, int n_, int ldap_, int V_, int nm_) {
        return dgeqrf_compact(lay, m_, n_, ap.data(), ldap_, tau.data(), V_, nm_);
    };
    struct { const char *what; int got, want; } t[] = {
        {"valid col",   call('C', m, n,  ld,  V, nm),   0},
        {"valid row",   call('R', m, n,  n,   V, nm),   0},   /* row-major ld >= n */
        {"bad layout",  call('X', m, n,  ld,  V, nm),  -1},
        {"m<0",         call('C', -1, n, ld,  V, nm),  -2},
        {"n<0",         call('C', m, -1, ld,  V, nm),  -3},
        {"ldap<m",      call('C', m, n,  m-1, V, nm),  -5},
        {"bad V",       call('C', m, n,  ld,  3, nm),  -7},
        {"nm<0",        call('C', m, n,  ld,  V, -1),  -8},
        {"empty m=0",   call('C', 0, n,  1,   V, nm),   0},
        {"empty nm=0",  call('C', m, n,  ld,  V, 0),    0},
    };
    int bad = 0;
    for (auto &c : t) bad += (c.got != c.want);
    std::printf("C API validation: %zu checks | %s\n", sizeof(t) / sizeof(t[0]), bad ? "FAIL" : "OK");
    for (auto &c : t) if (c.got != c.want) std::printf("  %-12s got=%d want=%d\n", c.what, c.got, c.want);
    return bad ? 1 : 0;
}

/* ------------------------------- main -------------------------------- */

int main()
{
    int fails = 0;
    fails += test_validation();

    /* square (the emphasis) */
    fails += run_case<double, 2>(4,  30, 30);
    fails += run_case<double, 4>(8,  30, 30);
    fails += run_case<double, 8>(16, 60, 60);
    fails += run_case<double, 8>(11, 43, 43);   /* padded partial group */
    fails += run_case<double, 4>(8,  3,  3);    /* smallest supported */
    /* rectangular */
    fails += run_case<double, 4>(8,  64, 20);   /* tall */
    fails += run_case<double, 4>(8,  20, 64);   /* wide */
    /* float */
    fails += run_case<float, 8>(16, 30, 30);
    fails += run_case<float, 16>(32, 43, 17);

    if (fails) { std::printf("\n%d CHECK(S) FAILED\n", fails); return 1; }
    std::printf("\nall checks passed\n");
    return 0;
}
