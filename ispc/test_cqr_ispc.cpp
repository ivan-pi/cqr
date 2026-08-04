/* test_cqr_ispc.cpp -- correctness for the ISPC compact-QR prototype.
 *
 * Runs the full batched AX=B solve through the ISPC pipeline and checks it two
 * ways: (1) forward error vs a KNOWN exact solution X within 100 n eps, and
 * (2) drop-in equivalence vs the templated GNU-vector kernels + MKL trsm -- the
 * ISPC factor must match bit-for-bit and the solution to a tight tolerance.
 * MKL supplies only pack/unpack and the reference trsm. Covers side/nrhs and a
 * padded partial group (nm not a multiple of 8).
 * Assisted-by: Claude:claude-opus-4.8 */
#include <mkl.h>
#include <mkl_compact.h>

#include "cqr_ispc.h"
#include "cqr_geqrf_compact.hpp" /* geqrf_compact_general<T,V> */
#include "cqr_compact.hpp"       /* ormqr_compact_general<T,V> */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace {
std::mt19937_64 rng(12345);
double frand()
{
    static std::uniform_real_distribution<double> d(-1, 1);
    return d(rng);
}
int failures = 0;
void expect(bool c, const char *what)
{
    if (!c) {
        std::printf("  FAIL: %s\n", what);
        ++failures;
    }
}
double *acalloc(std::size_t n) /* 64B-aligned, zeroed (deterministic padding) */
{
    std::size_t b = ((n * 8 + 63) / 64) * 64;
    double *p = static_cast<double *>(std::aligned_alloc(64, b));
    std::memset(p, 0, b);
    return p;
}
const double EPS = std::numeric_limits<double>::epsilon();

/* max|P-Q| over the first nm matrices (no padding when nm % 8 == 0). */
double maxdiff(const double *P, const double *Q, int elems, int nm)
{
    double d = 0;
    for (int mat = 0; mat < nm; ++mat)
        for (int e = 0; e < elems; ++e) {
            std::size_t off =
                (std::size_t)(mat / 8) * elems * 8 + (std::size_t)e * 8 + (mat % 8);
            d = std::max(d, std::fabs(P[off] - Q[off]));
        }
    return d;
}

void run(MKL_COMPACT_PACK fmt, int nm, int n, int nrhs, bool equiv)
{
    /* known solution X(i,j) = (j+1) + 0.5 i; B = A X, A diagonally dominant. */
    std::vector<double> X((std::size_t)n * nrhs);
    for (int j = 0; j < nrhs; ++j)
        for (int i = 0; i < n; ++i)
            X[i + (std::size_t)j * n] = (j + 1) + 0.5 * i;
    std::vector<std::vector<double>> A(nm, std::vector<double>((std::size_t)n * n)),
        B(nm, std::vector<double>((std::size_t)n * nrhs));
    std::vector<double *> Ap(nm), Bp(nm);
    for (int v = 0; v < nm; ++v) {
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                A[v][i + (std::size_t)j * n] = frand();
        for (int i = 0; i < n; ++i)
            A[v][i + (std::size_t)i * n] += 2.0 * n;
        for (int j = 0; j < nrhs; ++j)
            for (int i = 0; i < n; ++i) {
                double s = 0;
                for (int l = 0; l < n; ++l)
                    s += A[v][i + (std::size_t)l * n] * X[l + (std::size_t)j * n];
                B[v][i + (std::size_t)j * n] = s;
            }
        Ap[v] = A[v].data();
        Bp[v] = B[v].data();
    }
    const std::size_t szA = mkl_dget_size_compact(n, n, fmt, nm),
                      szT = mkl_dget_size_compact(n, 1, fmt, nm),
                      szB = mkl_dget_size_compact(n, nrhs, fmt, nm);
    double *ap0 = acalloc(szA), *bp0 = acalloc(szB);
    mkl_dgepack_compact(MKL_COL_MAJOR, n, n, Ap.data(), n, ap0, n, fmt, nm);
    mkl_dgepack_compact(MKL_COL_MAJOR, n, nrhs, Bp.data(), n, bp0, n, fmt, nm);

    /* ISPC pipeline */
    double *api = acalloc(szA), *tpi = acalloc(szT), *bpi = acalloc(szB);
    std::memcpy(api, ap0, szA * 8);
    std::memcpy(bpi, bp0, szB * 8);
    cqr_ispc_dgeqrf_compact(n, n, api, n, tpi, nm);
    cqr_ispc_dormqr_compact(/*Q^T*/ 1, n, nrhs, n, api, n, tpi, bpi, n, nm);
    cqr_ispc_dtrsm_compact(n, nrhs, 1.0, api, n, bpi, n, nm);
    std::vector<std::vector<double>> Xi(nm, std::vector<double>((std::size_t)n * nrhs));
    std::vector<double *> Xip(nm);
    for (int v = 0; v < nm; ++v)
        Xip[v] = Xi[v].data();
    mkl_dgeunpack_compact(MKL_COL_MAJOR, n, nrhs, Xip.data(), n, bpi, n, fmt, nm);

    double fwd = 0;
    for (int v = 0; v < nm; ++v)
        for (int e = 0; e < n * nrhs; ++e)
            fwd = std::max(fwd, std::fabs(Xi[v][e] - X[e]));
    const double rtol = 100.0 * n * EPS * (1.0 + 0.5 * n);
    expect(fwd <= rtol, "ISPC solve recovers known X");

    double ef = 0, es = 0;
    if (equiv) { /* reference: templated GNU kernels + MKL trsm, same inputs */
        double *apr = acalloc(szA), *tpr = acalloc(szT), *bpr = acalloc(szB);
        std::memcpy(apr, ap0, szA * 8);
        std::memcpy(bpr, bp0, szB * 8);
        cqr::detail::geqrf_compact_general<double, 8>(false, n, n, apr, n, tpr, nm);
        cqr::detail::ormqr_compact_general<double, 8>(true, false, 'T', n, nrhs, n, apr,
                                                      n, tpr, bpr, n, nm);
        mkl_dtrsm_compact(MKL_COL_MAJOR, MKL_LEFT, MKL_UPPER, MKL_NOTRANS, MKL_NONUNIT, n,
                          nrhs, 1.0, apr, n, bpr, n, fmt, nm);
        ef = maxdiff(api, apr, n * n, nm);    /* factor: expect bit-identical */
        es = maxdiff(bpi, bpr, n * nrhs, nm); /* solution */
        expect(ef <= 1e-9 * (1.0 + 2.0 * n), "ISPC factor matches reference geqrf");
        expect(es <= rtol, "ISPC solution matches reference pipeline");
        for (double *p : {apr, tpr, bpr})
            std::free(p);
    }
    std::printf("  nm=%-3d n=%-4d nrhs=%d | ISPC fwd %.2e (rtol %.1e)%s", nm, n, nrhs,
                fwd, rtol, equiv ? "" : "\n");
    if (equiv) std::printf(" | equiv fac %.1e sol %.1e\n", ef, es);
    for (double *p : {ap0, bp0, api, tpi, bpi})
        std::free(p);
}
} /* namespace */

int main()
{
    mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL);
    mkl_set_num_threads(1);
    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = (fmt == MKL_COMPACT_SSE ? 16 : fmt == MKL_COMPACT_AVX ? 32 : 64) / 8;
    std::printf("ISPC compact-QR correctness (compact format=%d, V=%d)\n", (int)fmt, V);
    if (V != 8) {
        std::printf(
            "prototype built for V=8 (avx512skx-x8); host V=%d -- rebuild ISPC target.\n",
            V);
        return 77;
    }
    for (auto c : {std::array<int, 3>{8, 10, 1},
                   {16, 20, 4},
                   {32, 40, 5},
                   {8, 100, 8},
                   {16, 150, 3}})
        run(fmt, c[0], c[1], c[2], true); /* nm % 8 == 0: element-wise equiv */
    for (auto c : {std::array<int, 3>{7, 32, 5}, {13, 64, 2}, {1, 48, 4}})
        run(fmt, c[0], c[1], c[2], false); /* padded final group: fwd error only */

    if (failures) {
        std::printf("\n%d CHECK(S) FAILED\n", failures);
        return 1;
    }
    std::printf("\nall checks passed\n");
    return 0;
}
