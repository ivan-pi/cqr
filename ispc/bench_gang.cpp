/* bench_gang.cpp -- ISPC gang-size sweep.
 *
 * The kernels are gang-generic (a compact pack of V matrices' element (i,j) is one
 * `varying double`), so the SAME source at a different --target gives a different
 * gang size -- the `-xN` suffix sets it. Build one tree per target and run each:
 *
 *   for w in 4 8 16 32 64; do
 *     cmake -S . -B build-x$w -DCMAKE_ISPC_INSTRUCTION_SETS=avx512skx-x$w \
 *           -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-march=native
 *     cmake --build build-x$w -j; ./build-x$w/bench_gang
 *   done
 *
 * Each binary packs at V = its gang width, runs the ISPC pipeline, validates
 * against a known solution, and reports throughput vs gang size. Uses ONLY the
 * ISPC kernels -- no MKL -- which is also proof the kernels carry no MKL
 * dependency (unlike the MKL-interop test/benchmark).
 *
 * Assisted-by: Claude:claude-opus-4.8
 */
#include "cqr_ispc.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using clk = std::chrono::steady_clock;

/* 64-byte-aligned buffer of n doubles. */
static double *al(std::size_t n)
{
    std::size_t b = ((n * sizeof(double) + 63) / 64) * 64;
    return static_cast<double *>(std::aligned_alloc(64, b));
}

int main(int argc, char **argv)
{
    const int G = cqr_ispc_gang_width(); /* interleave width V == gang size */
    const int nmat = argc > 1 ? std::atoi(argv[1]) : 1024;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 3;
    const int nrhs = argc > 3 ? std::atoi(argv[3]) : 4;
    const int sizes[] = {10, 20, 40, 60, 80, 100, 120, 150};
    const int ns = (int)(sizeof(sizes) / sizeof(*sizes));

    std::printf("ISPC gang-size sweep: gang=%d (pack V=%d)  nmat=%d reps=%d nrhs=%d\n", G,
                G, nmat, reps, nrhs);
    std::printf("   n |  GFLOP/s |  fwd err\n-----+----------+---------\n");

    double logsum = 0;
    for (int si = 0; si < ns; ++si) {
        const int n = sizes[si];
        const int ng = (nmat + G - 1) / G; /* groups of G interleaved matrices */
        const std::size_t szA = (std::size_t)ng * n * n * G;    /* ldap=n, ncol=n */
        const std::size_t szB = (std::size_t)ng * n * nrhs * G; /* ldbp=n         */
        const std::size_t szT = (std::size_t)ng * n * G;        /* k=n per matrix  */
        double *ap = al(szA), *bp = al(szB), *tau = al(szT), *ap0 = al(szA),
               *bp0 = al(szB);
        std::memset(ap0, 0, szA * 8);
        std::memset(bp0, 0, szB * 8);

        /* known solution X(i,j) = 1 + 0.1 i + 0.05 j; A diagonally dominant; B = A X.
         * Pack at V=G: matrix idx -> group idx/G, lane idx%G, element (i,j) at
         * (j*n+i)*G within the group. */
        std::vector<double> X((std::size_t)n * nrhs);
        for (int j = 0; j < nrhs; ++j)
            for (int i = 0; i < n; ++i)
                X[i + (std::size_t)j * n] = 1.0 + 0.1 * i + 0.05 * j;
        std::mt19937_64 rng(7);
        std::uniform_real_distribution<double> d(-1, 1);
        std::vector<double> A((std::size_t)n * n);
        for (int idx = 0; idx < nmat; ++idx) {
            const std::size_t g = idx / G, lane = idx % G;
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    A[i + (std::size_t)j * n] = d(rng);
            for (int i = 0; i < n; ++i)
                A[i + (std::size_t)i * n] += 2.0 * n;
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    ap0[g * n * n * G + ((std::size_t)j * n + i) * G + lane] =
                        A[i + (std::size_t)j * n];
            for (int j = 0; j < nrhs; ++j)
                for (int i = 0; i < n; ++i) {
                    double s = 0;
                    for (int l = 0; l < n; ++l)
                        s += A[i + (std::size_t)l * n] * X[l + (std::size_t)j * n];
                    bp0[g * n * nrhs * G + ((std::size_t)j * n + i) * G + lane] = s;
                }
        }
        for (int idx = nmat; idx < ng * G; ++idx) { /* pad lanes with I (tau=0, safe) */
            const std::size_t g = idx / G, lane = idx % G;
            for (int i = 0; i < n; ++i)
                ap0[g * n * n * G + ((std::size_t)i * n + i) * G + lane] = 1.0;
        }

        double best = 1e300;
        for (int r = -1; r < reps; ++r) { /* r = -1 is an untimed warm-up */
            std::memcpy(ap, ap0, szA * 8);
            std::memcpy(bp, bp0, szB * 8);
            auto t0 = clk::now();
            cqr_ispc_dgeqrf_compact(n, n, ap, n, tau, nmat);
            cqr_ispc_dormqr_compact(1, n, nrhs, n, ap, n, tau, bp, n, nmat);
            cqr_ispc_dtrsm_compact(n, nrhs, 1.0, ap, n, bp, n, nmat);
            std::chrono::duration<double> dt = clk::now() - t0;
            if (r >= 0) best = std::min(best, dt.count());
        }
        double err = 0; /* bp now holds the last pass's solution */
        for (int idx = 0; idx < nmat; ++idx) {
            const std::size_t g = idx / G, lane = idx % G;
            for (int j = 0; j < nrhs; ++j)
                for (int i = 0; i < n; ++i) {
                    double x = bp[g * n * nrhs * G + ((std::size_t)j * n + i) * G + lane];
                    err = std::max(err, std::fabs(x - X[i + (std::size_t)j * n]));
                }
        }
        const double flops = (4.0 / 3.0) * n * n * (double)n +
                             2.0 * nrhs * n * (n + 1.0) + 1.0 * nrhs * n * (double)n;
        const double gf = nmat * flops / best / 1e9;
        logsum += std::log(gf);
        std::printf("%4d | %8.2f | %.1e\n", n, gf, err);
        for (double *p : {ap, bp, tau, ap0, bp0})
            std::free(p);
    }
    std::printf("geomean GFLOP/s: %.2f  (gang=%d)\n", std::exp(logsum / ns), G);
    return 0;
}
