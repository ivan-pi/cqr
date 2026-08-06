/* bench_geqrf_collect.cpp
 *
 * Publication-grade collection driver for the compact-QR *factorization*,
 * built on Google Benchmark. It measures the same three implementations as the
 * repo's examples/bench_geqrf_compact.cpp --
 *
 *   cqr     cqr_mkl_dgeqrf_compact   (this project's batched SIMD kernel)
 *   mkl     mkl_dgeqrf_compact       (Intel MKL's batched compact kernel)
 *   lapack  LAPACKE_dgeqrf           (per-matrix, one at a time)
 *
 * -- but hands warm-up, iteration-count tuning, and statistical repetition to
 * Google Benchmark, and emits machine-readable JSON. report/bench/json_to_dat.py
 * turns that JSON into the .dat tables the gnuplot scripts plot.
 *
 * Why a second driver? bench_geqrf_compact stays the fast CTest correctness gate
 * (min-of-reps, prints a table). This one is for the final numbers on a quiet,
 * frequency-pinned node: repetitions with mean/median/stddev/CV, and counters
 * reporting matrices/s and GFLOP/s directly.
 *
 * Timing model (identical intent to the example):
 *   - the pool is packed into compact form ONCE, outside the timed region;
 *   - each iteration restores the destroyed input (untimed) and factorizes,
 *     with manual timing (SetIterationTime) around the factorization only, so
 *     neither packing nor the restore is charged to the measurement;
 *   - the compact paths are driven from an OpenMP outer loop over the V-groups
 *     (the intended usage), with MKL's own threading pinned to 1; the per-matrix
 *     path parallelizes over matrices the same way.
 *
 * Batch size: set the environment variable CQR_NMAT (default 512). Everything
 * else -- repetitions, output format -- is a standard Google Benchmark flag,
 * e.g.  --benchmark_repetitions=15 --benchmark_report_aggregates_only=true
 *       --benchmark_format=json --benchmark_out=results.json
 *
 * Build: enable -DCQR_BUILD_REPORT_BENCH=ON (needs Google Benchmark), or see
 * report/bench/README.md for a standalone compile line. OpenMP is used when
 * available.
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#include <benchmark/benchmark.h>

#include <mkl.h>
#include <mkl_compact.h>

#include "cqr_mkl_ext.h"
#include "cqr_mkl_alloc.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <new>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using clk = std::chrono::steady_clock;
using cqr::detail::vlen_for_format;

/* Batch size (matrices per pool), from CQR_NMAT or the default. Read once in
 * main so every registered benchmark sees the same value. */
int g_nmat = 512;

/* Standard LAPACK ?geqrf flop count (m >= n), in GFLOP. */
double geqrf_gflop(int m, int n)
{
    return (2.0 * m * n * (double)n - (2.0 / 3.0) * n * (double)n * n) * 1e-9;
}

/* std::vector storage aligned to the compact pack width (64 B covers every
 * format), matching the example so the dense pool and its LAPACK copy start
 * pack-aligned -- no cache-line splits in the packing reads. */
template <typename T> struct aligned_allocator {
    using value_type = T;
    T *allocate(std::size_t n)
    {
        void *p = std::aligned_alloc(64, (n * sizeof(T) + 63) & ~std::size_t(63));
        if (!p) throw std::bad_alloc();
        return static_cast<T *>(p);
    }
    void deallocate(T *p, std::size_t) noexcept { std::free(p); }
    bool operator==(const aligned_allocator &) const noexcept { return true; }
    bool operator!=(const aligned_allocator &) const noexcept { return false; }
};
template <typename T> using aligned_vector = std::vector<T, aligned_allocator<T>>;

/* A pool of `nmat` dense column-major m x n matrices, well conditioned
 * (diagonal-boosted), fixed seed so every run factors the identical batch.
 * Templated on the scalar type so it serves both float and double precision. */
template <typename T> struct Pool {
    int m, n, nmat;
    aligned_vector<T> a;

    Pool(int m_, int n_, int nmat_)
        : m(m_), n(n_), nmat(nmat_), a((size_t)nmat_ * m_ * n_)
    {
        std::mt19937_64 rng(2026);
        std::uniform_real_distribution<T> dist(-1.0, 1.0);
        for (int v = 0; v < nmat; ++v) {
            T *A = a.data() + (size_t)v * m * n;
            for (int i = 0; i < m * n; ++i)
                A[i] = dist(rng);
            for (int i = 0; i < std::min(m, n); ++i)
                A[i + (size_t)i * m] += T(2 * n);
        }
    }
};

/* Factor a pre-packed compact pool in place, one V-group per OpenMP iteration
 * (the intended outer-loop usage). `use_cqr` selects our kernel or MKL's. */
void factor_compact(bool use_cqr, double *ap, double *taup, int m, int n, int ngroups,
                    int V, MKL_COMPACT_PACK fmt, MKL_INT lwork)
{
    const int k = std::min(m, n);
#pragma omp parallel
    {
        std::vector<double> work((size_t)std::max<MKL_INT>(lwork, 1));
        MKL_INT info;
#pragma omp for schedule(static)
        for (int g = 0; g < ngroups; ++g) {
            double *apg = ap + (size_t)g * m * n * V;
            double *taupg = taup + (size_t)g * k * V;
            if (use_cqr)
                cqr_mkl_dgeqrf_compact(MKL_COL_MAJOR, m, n, apg, m, taupg, work.data(),
                                       lwork, &info, fmt, V);
            else
                mkl_dgeqrf_compact(MKL_COL_MAJOR, m, n, apg, m, taupg, work.data(), lwork,
                                   &info, fmt, V);
        }
    }
}

/* Per-matrix LAPACK factorization of a standard-layout pool copy in place. */
void factor_unbatched(double *a, int m, int n, int nmat)
{
    const int k = std::min(m, n);
#pragma omp parallel
    {
        std::vector<double> tau(k);
#pragma omp for schedule(static)
        for (int v = 0; v < nmat; ++v)
            LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, a + (size_t)v * m * n, m, tau.data());
    }
}

/* Elementwise (H, tau) error of the cqr compact path vs per-matrix LAPACK,
 * scaled by the matrix L1 norm. Untimed correctness gate (see the example). */
double factor_error(const Pool<double> &P, MKL_COMPACT_PACK fmt, int V)
{
    const int m = P.m, n = P.n, nmat = P.nmat, k = std::min(m, n);
    const size_t sA = (size_t)m * n;

    MKL_INT sz_a = mkl_dget_size_compact(m, n, fmt, nmat);
    MKL_INT sz_t = mkl_dget_size_compact(k, 1, fmt, nmat);
    auto ap = cqr::detail::mkl_alloc_bytes<double>(sz_a);
    auto tp = cqr::detail::mkl_alloc_bytes<double>(sz_t);

    std::vector<double *> Ap(nmat);
    for (int v = 0; v < nmat; ++v)
        Ap[v] = const_cast<double *>(P.a.data()) + v * sA;
    mkl_dgepack_compact(MKL_COL_MAJOR, m, n, Ap.data(), m, ap.get(), m, fmt, nmat);

    const int ngroups = (nmat + V - 1) / V;
    MKL_INT info;
    double wq;
    cqr_mkl_dgeqrf_compact(MKL_COL_MAJOR, m, n, ap.get(), m, tp.get(), &wq, -1, &info, fmt,
                           V);
    factor_compact(true, ap.get(), tp.get(), m, n, ngroups, V, fmt, (MKL_INT)wq);

    std::vector<double> H(nmat * sA), tau(nmat * (size_t)k);
    std::vector<double *> Hp(nmat), Tp(nmat);
    for (int v = 0; v < nmat; ++v) {
        Hp[v] = H.data() + v * sA;
        Tp[v] = tau.data() + v * (size_t)k;
    }
    mkl_dgeunpack_compact(MKL_COL_MAJOR, m, n, Hp.data(), m, ap.get(), m, fmt, nmat);
    mkl_dgeunpack_compact(MKL_COL_MAJOR, k, 1, Tp.data(), k, tp.get(), k, fmt, nmat);

    double worst = 0;
    std::vector<double> Href(sA), tref(k);
    for (int v = 0; v < nmat; ++v) {
        const double *Av = P.a.data() + v * sA;
        std::copy(Av, Av + sA, Href.begin());
        LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, Href.data(), m, tref.data());
        double num = 0, den = 0;
        for (size_t i = 0; i < sA; ++i) {
            num = std::max(num, std::abs(H[v * sA + i] - Href[i]));
            den = std::max(den, std::abs(Href[i]));
        }
        for (int i = 0; i < k; ++i)
            num = std::max(num, std::abs(tau[v * (size_t)k + i] - tref[i]));
        worst = std::max(worst, num / std::max(den, 1e-300));
    }
    return worst;
}

enum class Impl { CQR, MKL, LAPACK };

/* One benchmark: factor a batch of `g_nmat` square n x n matrices with the
 * implementation `I`. state.range(0) == n. Manual timing charges only the
 * factorization; counters report the batch size, matrices/s, GFLOP/s, and (for
 * cqr) the correctness gate. */
template <Impl I>
void BM_geqrf(benchmark::State &state)
{
    const int n = (int)state.range(0);
    const int m = n, k = n, nmat = g_nmat;

    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = vlen_for_format<double>(fmt);
    const int ngroups = (nmat + V - 1) / V;

    Pool<double> P(m, n, nmat);

    if constexpr (I == Impl::LAPACK) {
        aligned_vector<double> work = P.a; /* standard-layout working copy */
        for (auto _ : state) {
            std::memcpy(work.data(), P.a.data(), P.a.size() * sizeof(double)); /* untimed */
            auto t0 = clk::now();
            factor_unbatched(work.data(), m, n, nmat);
            auto t1 = clk::now();
            state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
        }
    }
    else {
        constexpr bool use_cqr = (I == Impl::CQR);
        MKL_INT sz_a = mkl_dget_size_compact(m, n, fmt, nmat);
        MKL_INT sz_t = mkl_dget_size_compact(k, 1, fmt, nmat);
        auto pristine = cqr::detail::mkl_alloc_bytes<double>(sz_a);
        auto work_ap = cqr::detail::mkl_alloc_bytes<double>(sz_a);
        auto taup = cqr::detail::mkl_alloc_bytes<double>(sz_t);
        std::vector<double *> Ap(nmat);
        for (int v = 0; v < nmat; ++v)
            Ap[v] = P.a.data() + (size_t)v * m * n;
        mkl_dgepack_compact(MKL_COL_MAJOR, m, n, Ap.data(), m, pristine.get(), m, fmt, nmat);

        /* Each routine reports its own optimal lwork (MKL needs scratch; cqr
         * needs none), so query per path. */
        double wq;
        MKL_INT info;
        if (use_cqr)
            cqr_mkl_dgeqrf_compact(MKL_COL_MAJOR, m, n, work_ap.get(), m, taup.get(), &wq,
                                   -1, &info, fmt, V);
        else
            mkl_dgeqrf_compact(MKL_COL_MAJOR, m, n, work_ap.get(), m, taup.get(), &wq, -1,
                               &info, fmt, V);
        const MKL_INT lwork = (MKL_INT)wq;

        for (auto _ : state) {
            std::memcpy(work_ap.get(), pristine.get(), sz_a); /* restore (untimed) */
            auto t0 = clk::now();
            factor_compact(use_cqr, work_ap.get(), taup.get(), m, n, ngroups, V, fmt, lwork);
            auto t1 = clk::now();
            state.SetIterationTime(std::chrono::duration<double>(t1 - t0).count());
        }
    }

    /* kIsIterationInvariantRate divides the per-iteration value by the (manual)
     * iteration time -> matrices/s and GFLOP/s. `n` is a plain reported value. */
    state.counters["n"] = n;
    state.counters["mat_per_s"] =
        benchmark::Counter(nmat, benchmark::Counter::kIsIterationInvariantRate);
    state.counters["gflops"] = benchmark::Counter(
        nmat * geqrf_gflop(m, n), benchmark::Counter::kIsIterationInvariantRate);
    if constexpr (I == Impl::CQR)
        state.counters["relerr"] = factor_error(P, fmt, V);
}

/* Square sizes spanning the target range: powers plus the non-power stencil
 * sizes (30,45,60,105,168) so the SIMD-remainder staircase is visible. Matches
 * the example's default list. */
const std::vector<int> &sizes()
{
    static const std::vector<int> s = {8,   16,  24,  30,  32,  45,  48,  60, 64,
                                       96,  105, 128, 168, 170, 256, 384, 500};
    return s;
}

void register_all()
{
    for (int n : sizes()) {
        benchmark::RegisterBenchmark("cqr", BM_geqrf<Impl::CQR>)
            ->Arg(n)->UseManualTime()->Unit(benchmark::kMicrosecond);
        benchmark::RegisterBenchmark("mkl", BM_geqrf<Impl::MKL>)
            ->Arg(n)->UseManualTime()->Unit(benchmark::kMicrosecond);
        benchmark::RegisterBenchmark("lapack", BM_geqrf<Impl::LAPACK>)
            ->Arg(n)->UseManualTime()->Unit(benchmark::kMicrosecond);
    }
}

} /* anonymous namespace */

int main(int argc, char **argv)
{
    if (const char *e = std::getenv("CQR_NMAT")) {
        int v = std::atoi(e);
        if (v > 0) g_nmat = v;
    }

    /* Pin MKL to a single thread; the batch parallelism is the OpenMP outer
     * loop over V-groups. Disable LAPACK's NaN check (the pool is finite). */
    mkl_set_num_threads(1);
    LAPACKE_set_nancheck(0);

    register_all();
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
