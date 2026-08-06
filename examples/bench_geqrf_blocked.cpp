/* bench_geqrf_blocked.cpp
 *
 * Prototype benchmark of the BLOCKED (Level-3, WY-representation) compact QR
 * factorization -- cqr::detail::geqrf_blocked_compact (larft + larfb) -- against
 * the unblocked geqr2 kernel it is built from, across the square-size range.
 *
 * Motivation (cqr_geqrf_compact_cache_blocking.md): the unblocked kernel is a
 * Level-2 computation whose throughput rolls off once one interleaved group of V
 * matrices no longer fits in L2 (n ~ 120 on a 1 MiB-L2 host). The blocked path
 * streams the trailing matrix once per NB-column panel instead of once per column,
 * so it stays compute-bound past that point. This benchmark shows the roll-off
 * knee disappear.
 *
 * It is fully portable -- it needs neither MKL nor BLAS, only this repo's header
 * -- so it doubles as a correctness gate: the blocked factorization must match the
 * unblocked one (already validated against LAPACK by the MKL test suites) to
 * rounding. A norm-scaled elementwise difference above 1e-12 aborts non-zero.
 *
 * Usage:  bench_geqrf_blocked [nmat] [reps]      (defaults: 128 matrices, 4 reps)
 *
 * Build with host-tuned flags for a representative comparison, e.g.
 *   -DCMAKE_CXX_FLAGS="-O3 -march=native"
 * so the V=8 (AVX-512) interleave uses the full vector width. Single-threaded on
 * purpose: the outer batch loop is the intended parallel axis, but the cache
 * behaviour this measures is per-core.
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#include "cqr_geqrf_compact.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace {

using clk = std::chrono::steady_clock;
using cqr::detail::geqrf_blocked_compact;
using cqr::detail::geqrf_compact;

constexpr int V = 8;   /* interleave width (AVX-512 FP64; runs on any target) */
constexpr int NC = 64; /* trailing-column tile for larfb */

double *alloc64(std::size_t n)
{
    void *p = std::aligned_alloc(64, ((n * sizeof(double)) + 63) & ~std::size_t(63));
    if (!p) {
        std::perror("aligned_alloc");
        std::exit(1);
    }
    return static_cast<double *>(p);
}

/* Pack nmat square n x n matrices into the compact column-major layout (element
 * (i,c) of the V matrices in a group stored contiguously), well conditioned via a
 * 2n diagonal boost. A padded partial last group is filled with identities. */
void pack_pool(double *ap, int n, int nmat, unsigned seed)
{
    const std::size_t ldap = n, ncol = n;
    const int ngroups = (nmat + V - 1) / V;
    std::memset(ap, 0, (std::size_t)ngroups * ldap * ncol * V * sizeof(double));
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int mat = 0; mat < ngroups * V; ++mat) {
        const int g = mat / V, v = mat % V;
        double *base = ap + (std::size_t)g * ldap * ncol * V + v;
        const bool pad = (mat >= nmat);
        for (int c = 0; c < n; ++c)
            for (int i = 0; i < n; ++i) {
                double x = pad ? 0.0 : dist(rng);
                if (i == c) x += pad ? 1.0 : 2.0 * n; /* identity in padded lanes */
                base[((std::size_t)c * ldap + i) * V] = x;
            }
    }
}

/* max |a-b| scaled by the global max magnitude (the repo's factor_error metric;
 * per-element ratios blow up on the near-zero reflector/R entries). */
double reldiff(const double *a, const double *b, std::size_t count)
{
    double num = 0, den = 0;
    for (std::size_t i = 0; i < count; ++i) {
        num = std::max(num, std::fabs(a[i] - b[i]));
        den = std::max(den, std::fabs(a[i]));
    }
    return num / (den < 1e-300 ? 1e-300 : den);
}

double geqrf_gflop(int n)
{
    return (4.0 / 3.0) * n * (double)n * n * 1e-9;
}

template <typename F>
double best_time(int reps, const double *pristine, double *buf, std::size_t na, F &&run)
{
    std::memcpy(buf, pristine, na * sizeof(double));
    run(); /* warm-up */
    double best = std::numeric_limits<double>::infinity();
    for (int r = 0; r < reps; ++r) {
        std::memcpy(buf, pristine, na * sizeof(double));
        auto t0 = clk::now();
        run();
        best = std::min(best, std::chrono::duration<double>(clk::now() - t0).count());
    }
    return best;
}

} /* anonymous namespace */

int main(int argc, char **argv)
{
    const int nmat = argc > 1 ? std::atoi(argv[1]) : 128;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 4;
    if (nmat <= 0 || reps <= 0) {
        std::printf("usage: bench_geqrf_blocked [nmat>0] [reps>0]\n");
        return 2;
    }
    const int sizes[] = {32, 48, 64, 96, 128, 160, 192, 224, 256, 288, 320, 384};
    const int NBs[] = {8, 16, 32};
    const double gate = 1e-12; /* blocked must match unblocked to rounding */

    std::printf("Blocked (Level-3) vs unblocked (Level-2) compact QR -- V=%d, nmat=%d, "
                "reps=%d, NC=%d, single-thread\n",
                V, nmat, reps, NC);
    std::printf("GFLOP/s (higher is better); the unblocked column rolls off past L2, "
                "the blocked columns hold.\n\n");
    std::printf(
        "   n | unblk |  NB=8 |  NB=16 |  NB=32 | best blk/unblk | max reldiff\n");
    std::printf(
        "-----+-------+-------+--------+--------+----------------+------------\n");

    int failed = 0;
    for (int n : sizes) {
        const std::size_t ldap = n;
        const int ngroups = (nmat + V - 1) / V;
        const std::size_t na = (std::size_t)ngroups * ldap * n * V;
        const std::size_t nt_ = (std::size_t)ngroups * n * V;

        double *pristine = alloc64(na);
        double *bufU = alloc64(na);
        double *bufB = alloc64(na);
        double *tauU = alloc64(nt_);
        double *tauB = alloc64(nt_);
        pack_pool(pristine, n, nmat, 2025u);

        const double tU = best_time(reps, pristine, bufU, na, [&] {
            geqrf_compact<double, V, int>(n, n, bufU, (int)ldap, tauU, nmat);
        });
        const double gU = nmat * geqrf_gflop(n) / tU;

        double gB[3], worst = 0, bestG = 0;
        for (int b = 0; b < 3; ++b) {
            const int NB = NBs[b];
            const double tB = best_time(reps, pristine, bufB, na, [&] {
                geqrf_blocked_compact<double, V, int>(n, n, bufB, (int)ldap, tauB, nmat,
                                                      NB, NC);
            });
            gB[b] = nmat * geqrf_gflop(n) / tB;
            bestG = std::max(bestG, gB[b]);
            worst = std::max(worst,
                             std::max(reldiff(bufU, bufB, na), reldiff(tauU, tauB, nt_)));
        }

        std::printf("%4d | %5.1f | %5.1f | %6.1f | %6.1f | %13.2fx | %.1e%s\n", n, gU,
                    gB[0], gB[1], gB[2], bestG / gU, worst, worst > gate ? "  FAIL" : "");
        if (worst > gate) failed = 1;

        std::free(pristine);
        std::free(bufU);
        std::free(bufB);
        std::free(tauU);
        std::free(tauB);
    }
    std::printf(
        "-----+-------+-------+--------+--------+----------------+------------\n");
    if (failed) {
        std::printf("FAILED: blocked factorization diverged from the unblocked kernel\n");
        return 1;
    }
    std::printf("OK: blocked matches unblocked to rounding at every size\n");
    return 0;
}
