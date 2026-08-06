/* bench_gang.cpp -- ISPC gang-size sweep, vs the GNU vector_size kernels at the
 * same width.
 *
 * The kernels are gang-generic (a compact pack of V matrices' element (i,j) is one
 * `varying double`), so the SAME source at a different --target gives a different
 * gang size -- the `-xN` suffix sets it. The templated GNU vector_size kernels in
 * ../src take V as a template argument and support V in {2,4,8,16} (pack<T,V> is a
 * V*8-byte GNU vector). Both use the identical compact layout and neither needs
 * MKL, so this packs at V = the gang width and times BOTH at that V:
 *
 *   for w in 4 8 16 32 64; do
 *     cmake -S . -B build-x$w -DCMAKE_ISPC_INSTRUCTION_SETS=avx512skx-x$w \
 *           -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-march=native
 *     cmake --build build-x$w -j; ./build-x$w/bench_gang
 *   done
 *
 * Notably V=16 (a 1024-bit vector -> 2x ZMM) exists for both here but NOT for MKL
 * Compact (whose FP64 format tops out at V=8). Uses only the ISPC + GNU kernels,
 * no MKL. Both pipelines are validated against a known solution.
 *
 * Assisted-by: Claude:claude-opus-4.8
 */
#include "cqr_ispc.h"
#include "cqr_geqrf_compact.hpp" /* geqrf_compact_general<T,V>  (GNU vector_size) */
#include "cqr_compact.hpp"       /* ormqr_compact_general<T,V>                    */
#include "cqr_trsm_compact.hpp"  /* trsm_compact_general<T,V>   (../src)          */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

#if defined(__clang__)
static const char *CC = "clang";
#elif defined(__GNUC__)
static const char *CC = "GCC";
#else
static const char *CC = "cc";
#endif

using clk = std::chrono::steady_clock;

static double *al(std::size_t n) /* 64-byte-aligned buffer of n doubles */
{
    std::size_t b = ((n * sizeof(double) + 63) / 64) * 64;
    return static_cast<double *>(std::aligned_alloc(64, b));
}

/* One solve pipeline at compile-time width V, GNU vector_size kernels. */
template <int V>
static void gnu_pipe(int n, int nrhs, double *ap, double *tau, double *bp, int nmat)
{
    cqr::detail::geqrf_compact_general<double, V>(false, n, n, ap, n, tau, nmat);
    cqr::detail::ormqr_compact_general<double, V>(true, false, 'T', n, nrhs, n, ap, n,
                                                  tau, bp, n, nmat);
    /* R X = Q^T B: left, upper, no-trans, non-unit, col-major (R is n x n). */
    cqr::detail::trsm_compact_general<double, V>(true, true, false, false, false, n, nrhs,
                                                 1.0, ap, n, bp, n, nmat);
}

int main(int argc, char **argv)
{
    const int G = cqr_ispc_gang_width(); /* interleave width V == gang size */
    const int nmat = argc > 1 ? std::atoi(argv[1]) : 1024;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 3;
    const int nrhs = argc > 3 ? std::atoi(argv[3]) : 4;
    const int sizes[] = {10, 20, 40, 60, 80, 100, 120, 150};
    const int ns = (int)(sizeof(sizes) / sizeof(*sizes));
    const bool gnu_ok = (G == 2 || G == 4 || G == 8 || G == 16); /* pack<T,V> widths */

    std::printf("ISPC gang-size sweep vs GNU vector_size (native=%s): gang/V=%d  "
                "nmat=%d reps=%d nrhs=%d\n",
                CC, G, nmat, reps, nrhs);
    std::printf("   n |  ISPC     %-6s | ISPC/%-5s | fwd err (ISPC / %s)\n", CC, CC, CC);
    std::printf("-----+-----------------+------------+--------------------\n");

    double logi = 0, logg = 0;
    for (int si = 0; si < ns; ++si) {
        const int n = sizes[si];
        const int ng = (nmat + G - 1) / G;
        const std::size_t szA = (std::size_t)ng * n * n * G;
        const std::size_t szB = (std::size_t)ng * n * nrhs * G;
        const std::size_t szT = (std::size_t)ng * n * G;
        double *ap = al(szA), *bp = al(szB), *tau = al(szT), *ap0 = al(szA),
               *bp0 = al(szB);
        std::memset(ap0, 0, szA * 8);
        std::memset(bp0, 0, szB * 8);

        /* known X(i,j) = 1 + 0.1 i + 0.05 j; A diagonally dominant; B = A X. Pack at
         * V=G: matrix idx -> group idx/G, lane idx%G, element (i,j) at (j*n+i)*G. */
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

        auto err_of = [&](const double *b) {
            double e = 0;
            for (int idx = 0; idx < nmat; ++idx) {
                const std::size_t g = idx / G, lane = idx % G;
                for (int j = 0; j < nrhs; ++j)
                    for (int i = 0; i < n; ++i)
                        e = std::max(e, std::fabs(b[g * n * nrhs * G +
                                                    ((std::size_t)j * n + i) * G + lane] -
                                                  X[i + (std::size_t)j * n]));
            }
            return e;
        };
        /* best-of-reps seconds + forward error for a pipeline (reset each pass). */
        auto time_it = [&](auto &&pipe) {
            double best = 1e300;
            for (int r = -1; r < reps; ++r) {
                std::memcpy(ap, ap0, szA * 8);
                std::memcpy(bp, bp0, szB * 8);
                auto t0 = clk::now();
                pipe();
                std::chrono::duration<double> dt = clk::now() - t0;
                if (r >= 0) best = std::min(best, dt.count());
            }
            return std::pair<double, double>{best, err_of(bp)};
        };

        auto pr_i = time_it([&] {
            cqr_ispc_dgeqrf_compact(n, n, ap, n, tau, nmat);
            cqr_ispc_dormqr_compact(1, n, nrhs, n, ap, n, tau, bp, n, nmat);
            cqr_ispc_dtrsm_compact(n, nrhs, 1.0, ap, n, bp, n, nmat);
        });
        std::pair<double, double> pr_g{0, 0};
        if (gnu_ok)
            pr_g = time_it([&] {
                switch (G) {
                case 2: gnu_pipe<2>(n, nrhs, ap, tau, bp, nmat); break;
                case 4: gnu_pipe<4>(n, nrhs, ap, tau, bp, nmat); break;
                case 8: gnu_pipe<8>(n, nrhs, ap, tau, bp, nmat); break;
                case 16: gnu_pipe<16>(n, nrhs, ap, tau, bp, nmat); break;
                }
            });

        const double flops = (4.0 / 3.0) * n * n * (double)n +
                             2.0 * nrhs * n * (n + 1.0) + 1.0 * nrhs * n * (double)n;
        const double gi = nmat * flops / pr_i.first / 1e9;
        const double gg = gnu_ok ? nmat * flops / pr_g.first / 1e9 : 0.0;
        logi += std::log(gi);
        if (gnu_ok) logg += std::log(gg);
        if (gnu_ok)
            std::printf("%4d | %7.2f  %7.2f | %8.2fx  | %.1e / %.1e\n", n, gi, gg,
                        gg / gi, pr_i.second, pr_g.second);
        else
            std::printf("%4d | %7.2f       -- | (GNU has no V=%d) | %.1e / --\n", n, gi,
                        G, pr_i.second);
        for (double *p : {ap, bp, tau, ap0, bp0})
            std::free(p);
    }
    if (gnu_ok)
        std::printf("geomean GFLOP/s: ISPC %.2f  %s %.2f  (ISPC/%s %.2fx, gang/V=%d)\n",
                    std::exp(logi / ns), CC, std::exp(logg / ns), CC,
                    std::exp((logg - logi) / ns), G);
    else
        std::printf(
            "geomean GFLOP/s: ISPC %.2f  (gang=%d; GNU vector_size has no V=%d)\n",
            std::exp(logi / ns), G, G);
    return 0;
}
