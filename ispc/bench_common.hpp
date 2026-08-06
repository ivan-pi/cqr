/* bench_common.hpp -- shared harness for the ISPC prototype benchmark/test.
 * Pool of small systems with known solution X==1, aligned compact buffers, a
 * best-of-reps timer, pack/unpack helpers, and GFLOP/geomean utilities.
 * Assisted-by: Claude:claude-opus-4.8 */
#ifndef BENCH_COMMON_HPP
#define BENCH_COMMON_HPP

#include <mkl.h>
#include <mkl_compact.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace bench {

using clk = std::chrono::steady_clock;
inline const double EPS = std::numeric_limits<double>::epsilon();

inline void die(const char *w)
{
    std::printf("FATAL: %s\n", w);
    std::exit(1);
}

/* 64-byte-aligned, zeroed buffer of n doubles (deterministic padding lanes). */
inline double *aligned(std::size_t n)
{
    std::size_t b = ((n * sizeof(double) + 63) / 64) * 64;
    double *p = static_cast<double *>(std::aligned_alloc(64, b));
    if (!p) die("aligned_alloc");
    std::memset(p, 0, b);
    return p;
}

/* Best (minimum) wall seconds over `reps`; `reset` runs untimed before each. */
template <class Reset, class Timed>
double best_time(int reps, Reset &&reset, Timed &&timed)
{
    reset();
    timed(); /* warm-up */
    double best = std::numeric_limits<double>::infinity();
    for (int r = 0; r < reps; ++r) {
        reset();
        auto t0 = clk::now();
        timed();
        std::chrono::duration<double> dt = clk::now() - t0;
        best = std::min(best, dt.count());
    }
    return best;
}

/* nmat column-major order-n matrices (diagonally dominant) + nrhs RHS, with
 * exact solution X==1 so each RHS column is the row sums of A. */
struct Pool {
    int n, nrhs, nmat;
    std::vector<double> a, b;
    Pool(int n_, int nrhs_, int nmat_)
        : n(n_), nrhs(nrhs_), nmat(nmat_), a((std::size_t)nmat_ * n_ * n_),
          b((std::size_t)nmat_ * n_ * nrhs_)
    {
        std::mt19937_64 rng(2024);
        std::uniform_real_distribution<double> d(-1, 1);
        for (int v = 0; v < nmat; ++v) {
            double *A = a.data() + (std::size_t)v * n * n;
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    A[i + (std::size_t)j * n] = d(rng);
            for (int i = 0; i < n; ++i)
                A[i + (std::size_t)i * n] += 2.0 * n;
            double *B = b.data() + (std::size_t)v * n * nrhs;
            for (int j = 0; j < nrhs; ++j)
                for (int i = 0; i < n; ++i) {
                    double s = 0;
                    for (int l = 0; l < n; ++l)
                        s += A[i + (std::size_t)l * n];
                    B[i + (std::size_t)j * n] = s;
                }
        }
    }
    /* pack A (n x n) and B (n x nrhs) into the given compact buffers. */
    void pack(MKL_COMPACT_PACK fmt, double *ap, double *bp) const
    {
        std::vector<double *> Ap(nmat), Bp(nmat);
        for (int v = 0; v < nmat; ++v) {
            Ap[v] = const_cast<double *>(a.data()) + (std::size_t)v * n * n;
            Bp[v] = const_cast<double *>(b.data()) + (std::size_t)v * n * nrhs;
        }
        mkl_dgepack_compact(MKL_COL_MAJOR, n, n, Ap.data(), n, ap, n, fmt, nmat);
        mkl_dgepack_compact(MKL_COL_MAJOR, n, nrhs, Bp.data(), n, bp, n, fmt, nmat);
    }
};

/* Unpack a compact solution and return max|x - 1| (forward error vs X==1). */
inline double unpack_err(int n, int nrhs, MKL_COMPACT_PACK fmt, int nmat, double *bp)
{
    std::vector<double> X((std::size_t)nmat * n * nrhs);
    std::vector<double *> Xp(nmat);
    for (int v = 0; v < nmat; ++v)
        Xp[v] = X.data() + (std::size_t)v * n * nrhs;
    mkl_dgeunpack_compact(MKL_COL_MAJOR, n, nrhs, Xp.data(), n, bp, n, fmt, nmat);
    double e = 0;
    for (double x : X)
        e = std::max(e, std::fabs(x - 1.0));
    return e;
}

/* Reporting: GFLOP/s, geomean of a per-size ratio, leading-order flop counts. */
inline double gflops(int nmat, double t, double fpm)
{
    return t > 0 ? nmat * fpm / t / 1e9 : 0;
}
template <class Pick> double geomean(int nsizes, Pick pick)
{
    double s = 0;
    for (int i = 0; i < nsizes; ++i)
        s += std::log(pick(i));
    return std::exp(s / nsizes);
}
inline double fl_geqrf(int n)
{
    return (4.0 / 3.0) * n * n * (double)n;
}
inline double fl_ormqr(int n, int r)
{
    return 2.0 * r * n * (n + 1.0);
}
inline double fl_trsm(int n, int r)
{
    return 1.0 * r * n * (double)n;
}
inline double fl_potrf(int n) /* Cholesky ~ n^3/3 (leading term, add+mul each 1) */
{
    return (1.0 / 3.0) * n * n * (double)n;
}

} /* namespace bench */
#endif
