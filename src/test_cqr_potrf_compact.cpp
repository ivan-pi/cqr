// test_cqr_potrf_compact.cpp
//
// Self-contained validation of the templated compact Cholesky factorization
// (dpotrf_compact / spotrf_compact), with no BLAS dependency. The reference is
// the unblocked Cholesky (LAPACK ?potf2) implemented in scalar form -- the same
// algorithm the vectorized kernel executes V lanes at a time, so a correct
// kernel matches it to working precision.
//
// Checks per (T, V, uplo, layout):
//   1. named-triangle factor  ==  scalar potf2 factor   (elementwise, ~eps*scale)
//   2. reconstruction  L L^T == A  (lower) / U^T U == A  (upper)
//   3. the strictly-opposite triangle of the compact buffer is bit-for-bit
//      unchanged from the input (the routine must not reference or write it)
// plus LAPACK-style argument validation of the C API.
//
// Assisted-by: Claude:claude-opus-4.8

#include <cstdio>
#include <cmath>
#include <vector>
#include <limits>
#include <algorithm>

#include "cqr_compact.h"         // dpotrf_compact / spotrf_compact
#include "test_compact_util.hpp" // frand, gen_spd, MatrixBatch, pack/unpack, max_abs_diff

using namespace cqr::test;

// ----------------------- reference kernel (scalar) ------------------
// Unblocked right-looking potf2 on a dense column-major n x n matrix, in place.
// Lower: A = L L^T, factor in the lower triangle. Upper: A = U^T U, factor in
// the upper triangle. Only the named triangle is read or written. No SPD check
// (mirrors the routine under test): a bad pivot yields NaN/Inf.

template <class T> static void ref_potf2(char uplo, int n, T *A, int lda)
{
    const bool upper = (uplo == 'U' || uplo == 'u');
    if (!upper) {
        for (int j = 0; j < n; ++j) {
            T d = std::sqrt(A[j + (size_t)j * lda]);
            A[j + (size_t)j * lda] = d;
            T invd = T(1) / d;
            for (int i = j + 1; i < n; ++i)
                A[i + (size_t)j * lda] *= invd; // scale pivot column
            for (int jj = j + 1; jj < n; ++jj)  // rank-1 trailing update, lower
                for (int i = jj; i < n; ++i)
                    A[i + (size_t)jj * lda] -=
                        A[i + (size_t)j * lda] * A[jj + (size_t)j * lda];
        }
    }
    else {
        for (int j = 0; j < n; ++j) {
            T d = std::sqrt(A[j + (size_t)j * lda]);
            A[j + (size_t)j * lda] = d;
            T invd = T(1) / d;
            for (int c = j + 1; c < n; ++c)
                A[j + (size_t)c * lda] *= invd; // scale pivot row
            for (int c = j + 1; c < n; ++c)     // rank-1 trailing update, upper
                for (int r = j + 1; r <= c; ++r)
                    A[r + (size_t)c * lda] -=
                        A[j + (size_t)r * lda] * A[j + (size_t)c * lda];
        }
    }
}

// MatrixBatch, pack_compact/unpack_compact, frand, max_abs_diff and gen_spd
// live in test_compact_util.hpp (shared across the compact test suites).

// precision-overloaded shim: pick d/s by the pointer type
static int potrf_c(char lay, char up, int n, double *a, int ld, int V, int nm)
{
    return dpotrf_compact(lay, up, n, a, ld, V, nm);
}
static int potrf_c(char lay, char up, int n, float *a, int ld, int V, int nm)
{
    return spotrf_compact(lay, up, n, a, ld, V, nm);
}

// --------------------------- one test case --------------------------

template <class T, int V> static int run_case(int nm, int n, char uplo, char layout)
{
    const T eps = std::numeric_limits<T>::epsilon();
    const bool rowmajor = (layout == 'R' || layout == 'r');
    const bool upper = (uplo == 'U' || uplo == 'u');

    // random SPD batch + scalar reference factor for this uplo
    MatrixBatch<T> A(nm, n, n), Aref(nm, n, n);
    for (int idx = 0; idx < nm; ++idx) {
        gen_spd(A[idx], n);
        std::copy(A[idx], A[idx] + (size_t)n * n, Aref[idx]);
        ref_potf2(uplo, n, Aref[idx], n);
    }

    // pack the full symmetric A, factor with the routine under test, unpack
    int ng = (nm + V - 1) / V;
    std::vector<T> ap((size_t)ng * n * n * V);
    pack_compact(A, ap.data(), n, V, rowmajor);
    int info = potrf_c(layout, uplo, n, ap.data(), n, V, nm);
    MatrixBatch<T> Aout(nm, n, n);
    unpack_compact(Aout, ap.data(), n, V, rowmajor);

    double e_fac = 0, e_rec = 0, e_untouched = 0;
    std::vector<T> Rec((size_t)n * n);
    for (int idx = 0; idx < nm; ++idx) {
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

        // check 2: reconstruction of A from the named triangle's factor
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                double s = 0;
                if (!upper) // A = L L^T : sum_l L(i,l) L(j,l), l <= min(i,j)
                    for (int l = 0; l <= std::min(i, j); ++l)
                        s += (double)Aout(idx, i, l) * (double)Aout(idx, j, l);
                else // A = U^T U : sum_l U(l,i) U(l,j), l <= min(i,j)
                    for (int l = 0; l <= std::min(i, j); ++l)
                        s += (double)Aout(idx, l, i) * (double)Aout(idx, l, j);
                Rec[i + (size_t)j * n] = (T)s;
            }
        e_rec = std::max(e_rec, max_abs_diff(Rec.data(), A[idx], (size_t)n * n));
    }

    const double scale = std::max(1, n);
    const double tol_fac = 20.0 * eps * scale;         // same op sequence, unique factor
    const double tol_rec = 40.0 * eps * scale * scale; // O(n) accumulation in recon
    bool ok_f = e_fac <= tol_fac;
    bool ok_r = e_rec <= tol_rec;
    bool ok_u = e_untouched == 0.0; // must be bit-for-bit unchanged
    bool ok_i = (info == 0);

    std::printf("T=%-6s V=%-2d uplo=%c lay=%c nm=%-2d n=%-3d | fac:%.1e %s rec:%.1e %s "
                "untouched:%s | info=%d %s\n",
                sizeof(T) == 8 ? "double" : "float", V, uplo, layout, nm, n, e_fac,
                ok_f ? "OK" : "FAIL", e_rec, ok_r ? "OK" : "FAIL", ok_u ? "OK" : "FAIL",
                info, ok_i ? "OK" : "FAIL");
    return (!ok_f) + (!ok_r) + (!ok_u) + (!ok_i);
}

// ------------- non-SPD lane isolation (design section 6.2) ----------
// A single non-SPD matrix shares a pack with SPD siblings. The routine takes no
// safeguarded path and reports no error (info stays 0): the bad lane simply
// hits a non-positive pivot, so sqrt/reciprocal poison *its own* factor with
// NaN/Inf -- while every sibling lane, computed with the same unmasked SIMD
// instructions, must stay finite and bit-exact vs the scalar reference. This is
// the "graceful garbage in, garbage out" contract and, crucially, gates that a
// poisoned lane never contaminates its neighbors.

template <class T, int V> static int run_nonspd(int n, char uplo, char layout)
{
    const bool rowmajor = (layout == 'R' || layout == 'r');
    const bool upper = (uplo == 'U' || uplo == 'u');
    const int nm = V;      // one full pack, so all V lanes are exercised together
    const int badlane = 1; // this lane is non-SPD; the rest are SPD

    MatrixBatch<T> A(nm, n, n), Aref(nm, n, n);
    for (int idx = 0; idx < nm; ++idx) {
        if (idx == badlane) {
            // symmetric but indefinite: identity with a negative leading pivot,
            // so the very first sqrt(A(0,0)) = sqrt(-1) = NaN.
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    A(idx, i, j) = (i == j) ? T(1) : T(0);
            A(idx, 0, 0) = T(-1);
        }
        else {
            gen_spd(A[idx], n);
            std::copy(A[idx], A[idx] + (size_t)n * n, Aref[idx]);
            ref_potf2(uplo, n, Aref[idx], n);
        }
    }

    int ng = (nm + V - 1) / V;
    std::vector<T> ap((size_t)ng * n * n * V);
    pack_compact(A, ap.data(), n, V, rowmajor);
    int info = potrf_c(layout, uplo, n, ap.data(), n, V, nm);
    MatrixBatch<T> Aout(nm, n, n);
    unpack_compact(Aout, ap.data(), n, V, rowmajor);

    double e_spd = 0; // worst error over the SPD sibling lanes
    bool spd_finite = true;
    bool bad_poisoned = false; // the non-SPD lane must carry NaN/Inf
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
                    if (!std::isfinite((double)x)) spd_finite = false;
                    e_spd = std::max(e_spd, (double)std::abs(x - Aref(idx, i, j)));
                }
            }

    const double tol = 20.0 * std::numeric_limits<T>::epsilon() * std::max(1, n);
    bool ok_spd = spd_finite && (e_spd <= tol); // siblings uncontaminated & correct
    bool ok_bad = bad_poisoned;                 // poison confined but present
    bool ok_info = (info == 0);                 // GIGO: no early-exit, no error

    std::printf("T=%-6s V=%-2d uplo=%c lay=%c n=%-3d non-SPD lane | siblings:%.1e %s "
                "poison:%s info=%d %s\n",
                sizeof(T) == 8 ? "double" : "float", V, uplo, layout, n, e_spd,
                ok_spd ? "OK" : "FAIL", ok_bad ? "OK" : "FAIL", info,
                ok_info ? "OK" : "FAIL");
    return (!ok_spd) + (!ok_bad) + (!ok_info);
}

// --------------------- C API argument validation --------------------

static int test_validation()
{
    const int n = 8, V = 4, nm = 4, ld = 8;
    std::vector<double> ap((size_t)ld * n * V, 0);
    for (int g = 0; g < 1; ++g) // seed a valid identity-ish diagonal so factoring is sane
        for (int v = 0; v < V; ++v)
            for (int i = 0; i < n; ++i)
                ap[((size_t)i * ld + i) * V + v] = 1.0;
    auto call = [&](char lay, char up, int n_, int ldap_, int V_, int nm_) {
        return dpotrf_compact(lay, up, n_, ap.data(), ldap_, V_, nm_);
    };
    // clang-format off
    struct { const char *what; int got, want; } t[] = {
        {"valid col L",  call('C', 'L', n,  ld,  V, nm),   0},
        {"valid row U",  call('R', 'U', n,  ld,  V, nm),   0},
        {"bad layout",   call('X', 'L', n,  ld,  V, nm),  -1},
        {"bad uplo",     call('C', 'X', n,  ld,  V, nm),  -2},
        {"n<0",          call('C', 'L', -1, ld,  V, nm),  -3},
        {"ldap<n",       call('C', 'L', n,  n-1, V, nm),  -5},
        {"bad V",        call('C', 'L', n,  ld,  3, nm),  -6},
        {"nm<0",         call('C', 'L', n,  ld,  V, -1),  -7},
        {"empty n=0",    call('C', 'L', 0,  1,   V, nm),   0},
        {"empty nm=0",   call('C', 'L', n,  ld,  V, 0),    0},
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

    // Non-SPD lane isolation (design 6.2): a poisoned lane must not contaminate
    // its SPD siblings, over both uplo, both layouts, and both precisions.
    for (char u : uplos)
        for (char l : lays) {
            fails += run_nonspd<double, 4>(20, u, l);
            fails += run_nonspd<float, 8>(16, u, l);
        }

    if (fails) {
        std::printf("\n%d CHECK(S) FAILED\n", fails);
        return 1;
    }
    std::printf("\nall checks passed\n");
    return 0;
}
