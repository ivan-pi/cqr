/* bench_geqrf_compact.cpp
 *
 * Throughput benchmark of the QR *factorization* over pools of many small
 * matrices, comparing three implementations of the same LAPACK ?geqrf math:
 *
 *   cqr-compact  cqr_mkl_dgeqrf_compact   (this project's batched SIMD kernel)
 *   mkl-compact  mkl_dgeqrf_compact       (Intel MKL's batched compact kernel)
 *   per-matrix   LAPACKE_dgeqrf           (conventional one-matrix-at-a-time)
 *
 * This is benchmark 1 of the design document (section 9): the compact batched
 * factorization against the standard per-matrix layout, with MKL's own compact
 * kernel as a second yardstick. To measure the factorization kernels rather than
 * data movement, the pool is packed into compact form once, up front; only the
 * factorization is timed, and the destroyed input is restored (untimed) before
 * each pass. The two compact paths are driven from an OpenMP outer loop over the
 * groups of V interleaved matrices -- the intended "outer multi-threaded loop"
 * usage -- with MKL's own threading pinned to 1; the per-matrix path parallelizes
 * over matrices the same way. The factorization is checked (untimed) against
 * per-matrix LAPACK, so the benchmark doubles as an integration test.
 *
 * Usage:  bench_geqrf_compact [--size-sweep=nmin:nmax[:stride]] [--simdlen=2|4|8]
 *         [nmat] [reps]      (defaults: 512 matrices, 3 reps)
 *
 * With no --size-sweep it runs the 3-way comparison (cqr vs mkl_dgeqrf_compact
 * vs per-matrix LAPACK); with it, a cqr-only throughput scan over the size range.
 * --simdlen forces the interleave width (2/4/8) instead of the host's widest.
 *
 * Build: needs Intel MKL plus this repo's cqr_mkl_ext; wired up by CMakeLists.txt
 * as the `bench_geqrf_compact` target. OpenMP is used when available.
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#include <mkl.h>
#include <mkl_compact.h>

#include "cqr_mkl_ext.h"
#include "cqr_mkl_alloc.h"
#include "cqr_geqrf_compact_omp.hpp" /* omp-simd kernels (outer- and inner-loop) */

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <random>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using clk = std::chrono::steady_clock;
using cqr::detail::vlen_for_format;

void check(bool cond, const char *what)
{
    if (!cond) {
        std::printf("FAILED: %s\n", what);
        std::exit(1);
    }
}

/* Standard LAPACK ?geqrf flop count (m >= n), in GFLOP. */
double geqrf_gflop(int m, int n)
{
    return (2.0 * m * n * (double)n - (2.0 / 3.0) * n * (double)n * n) * 1e-9;
}

const char *compact_format_name(MKL_COMPACT_PACK format)
{
    switch (format) {
    case MKL_COMPACT_SSE: return "SSE";
    case MKL_COMPACT_AVX: return "AVX";
    case MKL_COMPACT_AVX512: return "AVX512";
    default: return "unknown";
    }
}

/* The compact pack format whose double interleave width is v (2/4/8 ->
 * SSE/AVX/AVX512); MKL_COMPACT_SSE for anything else (rejected before use). */
MKL_COMPACT_PACK format_for_vlen(int v)
{
    switch (v) {
    case 4: return MKL_COMPACT_AVX;
    case 8: return MKL_COMPACT_AVX512;
    default: return MKL_COMPACT_SSE;
    }
}

/* std::vector storage aligned to the compact pack width (64 B covers every
 * format), so the dense pool and its LAPACK working copy start pack-aligned like
 * the compact buffers -- no cache-line splits in the packing reads or the
 * per-matrix LAPACK path. Keeping the allocator a stateless, type-only template
 * is what keeps it small: allocator_traits then defaults rebind, construct, and
 * the rest, and std::vector's copy-assign is the only reason equality is spelled
 * out (aligned_alloc needs the size rounded up to the alignment). */
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

/* A pool of `nmat` dense column-major m x n matrices, back to back in `a`
 * (m*n per matrix), well conditioned (diagonal-boosted). */
struct Pool {
    int m, n, nmat;
    aligned_vector<double> a; /* nmat * m*n, 64 B-aligned */

    Pool(int m_, int n_, int nmat_)
        : m(m_), n(n_), nmat(nmat_), a((size_t)nmat_ * m_ * n_)
    {
        std::mt19937_64 rng(2025);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (int v = 0; v < nmat; ++v) {
            double *A = a.data() + (size_t)v * m * n;
            for (int i = 0; i < m * n; ++i)
                A[i] = dist(rng);
            for (int i = 0; i < std::min(m, n); ++i)
                A[i + (size_t)i * m] += 2.0 * n;
        }
    }
};

/* Best (minimum) wall time over `reps` timed passes, in seconds; `reset` runs
 * untimed before every pass to restore the input the factorization destroys. */
template <typename Reset, typename Timed>
double best_time(int reps, Reset &&reset, Timed &&timed)
{
    reset();
    timed(); /* warm-up (untimed) */
    double best = std::numeric_limits<double>::infinity();
    for (int r = 0; r < reps; ++r) {
        reset();
        auto t0 = clk::now();
        timed();
        best = std::min(best, std::chrono::duration<double>(clk::now() - t0).count());
    }
    return best;
}

/* The four compact factorizations compared here. cqr-vec is the GNU
 * vector-types kernel (via the MKL-style entry point); cqr-omp-outer and
 * cqr-omp-inner are the two `#pragma omp simd` kernels; mkl is Intel's own. */
enum class Kernel { Vec, OmpOuter, OmpInner, Mkl };

/* Factor one group of V matrices with an omp-simd kernel, dispatching the
 * runtime interleave width V to a compile-time instantiation. */
template <bool Inner> void omp_group(int m, int n, double *apg, double *taupg, int V)
{
    auto run = [&](auto W) {
        constexpr int VV = decltype(W)::value;
        if constexpr (Inner)
            cqr::detail::omp_simd::geqrf_compact_general_omp_inner<double, VV>(
                false, m, n, apg, m, taupg, VV);
        else
            cqr::detail::omp_simd::geqrf_compact_general_omp<double, VV>(false, m, n, apg,
                                                                         m, taupg, VV);
    };
    switch (V) {
    case 2: run(std::integral_constant<int, 2>{}); break;
    case 4: run(std::integral_constant<int, 4>{}); break;
    case 8: run(std::integral_constant<int, 8>{}); break;
    case 16: run(std::integral_constant<int, 16>{}); break;
    }
}

/* Factor a pre-packed compact pool in place, one group of V per OpenMP
 * iteration (the intended outer-loop usage). `which` selects the kernel; all
 * see the identical compact buffer. */
void factor_compact(Kernel which, double *ap, double *taup, int m, int n, int ngroups,
                    int V, MKL_COMPACT_PACK fmt, MKL_INT lwork)
{
    const int k = std::min(m, n);
#pragma omp parallel
    {
        std::vector<double> work((size_t)std::max<MKL_INT>(lwork, 1));
        MKL_INT info;
#pragma omp for schedule(static)
        for (int g = 0; g < ngroups; ++g) {
            double *apg = ap + (size_t)g * m * n * V; /* group stride m*n*V */
            double *taupg = taup + (size_t)g * k * V;
            switch (which) {
            case Kernel::Vec:
                cqr_mkl_dgeqrf_compact(MKL_COL_MAJOR, m, n, apg, m, taupg, work.data(),
                                       lwork, &info, fmt, V);
                break;
            case Kernel::OmpOuter: omp_group<false>(m, n, apg, taupg, V); break;
            case Kernel::OmpInner: omp_group<true>(m, n, apg, taupg, V); break;
            case Kernel::Mkl:
                mkl_dgeqrf_compact(MKL_COL_MAJOR, m, n, apg, m, taupg, work.data(), lwork,
                                   &info, fmt, V);
                break;
            }
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

/* Relative factor error of a compact kernel vs per-matrix LAPACK: unpack the
 * compact (H, tau) and compare elementwise to a fresh LAPACKE_dgeqrf, scaled by
 * the matrix L1 norm. Untimed correctness gate. */
double factor_error(Kernel which, const Pool &P, MKL_COMPACT_PACK fmt, int V)
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
    MKL_INT lwork = -1, info;
    double wq;
    cqr_mkl_dgeqrf_compact(MKL_COL_MAJOR, m, n, ap.get(), m, tp.get(), &wq, -1, &info,
                           fmt, V);
    lwork = (MKL_INT)wq;
    factor_compact(which, ap.get(), tp.get(), m, n, ngroups, V, fmt, lwork);

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

/* Single-kernel size sweep: factor a pre-packed pool with cqr at each n in
 * [nmin, nmax] (step stride) and print throughput only -- no MKL/LAPACK
 * cross-check, so it stays cheap and isolates the kernel. The point is the
 * staircase: n that is / is not a multiple of the interleave width V. The raw
 * best-pass time is printed next to the derived rates: it is the quantity they
 * come from (rate = nmat / time), so a total near the timer granularity flags a
 * noisy row -- raise nmat until it is comfortably above the clock resolution. */
void run_sweep(int nmat, int reps, int nmin, int nmax, int stride, MKL_COMPACT_PACK fmt,
               int V, int nthreads)
{
    std::printf("QR factorization size sweep: cqr_mkl_dgeqrf_compact only "
                "(throughput, no cross-check)\n");
    std::printf("matrices=%d  reps=%d  simdlen=%d (%s)  OpenMP threads=%d  (square, "
                "pre-packed)\n\n",
                nmat, reps, V, compact_format_name(fmt), nthreads);
    std::printf("   n |  total (s) | cqr GFLOP/s |   cqr mat/s\n");
    std::printf("-----+------------+-------------+-------------\n");

    for (int n = nmin; n <= nmax; n += stride) {
        const int m = n, k = n;
        Pool P(m, n, nmat);
        const int ngroups = (nmat + V - 1) / V;

        MKL_INT sz_a = mkl_dget_size_compact(m, n, fmt, nmat);
        MKL_INT sz_t = mkl_dget_size_compact(k, 1, fmt, nmat);
        auto pristine = cqr::detail::mkl_alloc_bytes<double>(sz_a);
        auto work_ap = cqr::detail::mkl_alloc_bytes<double>(sz_a);
        auto taup = cqr::detail::mkl_alloc_bytes<double>(sz_t);
        std::vector<double *> Ap(nmat);
        for (int v = 0; v < nmat; ++v)
            Ap[v] = P.a.data() + (size_t)v * m * n;
        mkl_dgepack_compact(MKL_COL_MAJOR, m, n, Ap.data(), m, pristine.get(), m, fmt,
                            nmat);

        double wq;
        MKL_INT info;
        cqr_mkl_dgeqrf_compact(MKL_COL_MAJOR, m, n, work_ap.get(), m, taup.get(), &wq, -1,
                               &info, fmt, V);
        const MKL_INT lwork = (MKL_INT)wq;

        auto restore = [&] { std::memcpy(work_ap.get(), pristine.get(), sz_a); };
        const double t = best_time(reps, restore, [&] {
            factor_compact(Kernel::Vec, work_ap.get(), taup.get(), m, n, ngroups, V, fmt,
                           lwork);
        });
        std::printf("%4d | %10.3e | %11.2f | %11.2e\n", n, t,
                    nmat * geqrf_gflop(m, n) / t, nmat / t);
    }
    std::printf("-----+------------+-------------+-------------\n");
}

/* Parsed command line: positional [nmat] [reps], plus the optional flags
 * --size-sweep=nmin:nmax[:stride] (cqr-only scan) and --simdlen=2|4|8 (force the
 * interleave width instead of the host default). The constructor parses and
 * validates; hold the object const so the values cannot change afterward.
 * simdlen == 0 means "use the host's widest". */
struct CmdArgs {
    int nmat = 512;
    int reps = 3;
    int simdlen = 0; /* forced interleave width, or 0 for the host default */
    bool sweep = false;
    int sweep_min = 0, sweep_max = 0, sweep_step = 1;

    CmdArgs(int argc, char **argv)
    {
        std::vector<const char *> pos;
        for (int i = 1; i < argc; ++i) {
            if (std::strncmp(argv[i], "--size-sweep=", 13) == 0) {
                int got = std::sscanf(argv[i] + 13, "%d:%d:%d", &sweep_min, &sweep_max,
                                      &sweep_step);
                check(got >= 2, "usage: --size-sweep=nmin:nmax[:stride]");
                if (got == 2) sweep_step = 1;
                sweep = true;
            }
            else if (std::strncmp(argv[i], "--simdlen=", 10) == 0)
                simdlen = std::atoi(argv[i] + 10);
            else
                pos.push_back(argv[i]);
        }
        if (pos.size() > 0) nmat = std::atoi(pos[0]);
        if (pos.size() > 1) reps = std::atoi(pos[1]);
        check(nmat > 0 && reps > 0,
              "usage: bench_geqrf_compact [--size-sweep=nmin:nmax[:stride]] "
              "[--simdlen=2|4|8] [nmat>0] [reps>0]");
        check(!sweep || (sweep_min > 0 && sweep_max >= sweep_min && sweep_step > 0),
              "usage: --size-sweep needs 0 < nmin <= nmax and stride > 0");
        /* Double compact widths are 2/4/8 (SSE/AVX/AVX512); 16 is float's AVX512
         * width and has no double format, so it is rejected here. */
        check(simdlen == 0 || simdlen == 2 || simdlen == 4 || simdlen == 8,
              "usage: --simdlen must be 2, 4, or 8 (16 is float-only; this is double)");
    }
};

} /* anonymous namespace */

int main(int argc, char **argv)
{
    const CmdArgs args(argc, argv);
    const int nmat = args.nmat, reps = args.reps;

    /* Use the host's widest compact format unless --simdlen forces a narrower one.
     * A wider interleave than the host's native SIMD cannot execute, so reject it
     * (mkl_get_format_compact reports the widest the architecture supports). */
    const MKL_COMPACT_PACK native = mkl_get_format_compact();
    const MKL_COMPACT_PACK fmt = args.simdlen ? format_for_vlen(args.simdlen) : native;
    const int V = vlen_for_format<double>(fmt);
    check(V > 0 && V <= vlen_for_format<double>(native),
          "requested --simdlen exceeds the host's native SIMD width");

    mkl_set_num_threads(1);
    LAPACKE_set_nancheck(0);

    int nthreads = 1;
#ifdef _OPENMP
#pragma omp parallel
#pragma omp single
    nthreads = omp_get_num_threads();
#endif

    if (args.sweep) {
        run_sweep(nmat, reps, args.sweep_min, args.sweep_max, args.sweep_step, fmt, V,
                  nthreads);
        return 0;
    }

    /* Square sizes across the project's small-matrix target range (the design's
     * "emphasis on sizes below 170"). Deliberately mixes sizes that are not
     * multiples of the SIMD width V -- 30, 45, 60, 105, 168, from 2-D/3-D RBF-FD
     * stencils -- with the round powers, so the remainder handling (the staircase
     * SIMD effect) is visible. The list stops at 168: past there the compact
     * format loses to per-matrix LAPACK, and the scalar omp-outer path (GCC/Clang
     * cannot vectorize it) makes a single large-n pass minutes long. Use
     * --size-sweep for a finer cqr-only scan over any range. */
    constexpr std::array sizes = {8, 16, 24, 30, 32, 45, 48, 60, 64, 96, 105, 128, 168};

    std::printf("QR factorization throughput vs mkl_dgeqrf_compact: GNU vector-types and "
                "OpenMP-SIMD (outer/inner) compact kernels\n");
    std::printf("matrices=%d  reps=%d  simdlen=%d (%s)  OpenMP threads=%d  (square, "
                "pre-packed)\n\n",
                nmat, reps, V, compact_format_name(fmt), nthreads);
    /* GFLOP/s for the three cqr kernels and MKL, then each cqr kernel's speedup
     * over MKL compact (the headline). vec = GNU vector types; oOut/oIn = the
     * omp-simd outer-/inner-loop kernels. err is the worst elementwise (H, tau)
     * vs per-matrix LAPACKE_dgeqrf over all three cqr kernels. */
    std::printf("   n |  vec GF/s |  oOut GF/s |  oIn GF/s |  mkl GF/s | vec/mkl | "
                "oOut/mkl | oIn/mkl |   err\n");
    std::printf("-----+-----------+------------+-----------+-----------+---------+"
                "----------+---------+--------\n");

    double log_vec_mkl = 0.0, log_out_mkl = 0.0, log_in_mkl = 0.0, log_vec_lap = 0.0;
    for (int n : sizes) {
        const int m = n, k = n;
        Pool P(m, n, nmat);
        const int ngroups = (nmat + V - 1) / V;

        MKL_INT sz_a = mkl_dget_size_compact(m, n, fmt, nmat);
        MKL_INT sz_t = mkl_dget_size_compact(k, 1, fmt, nmat);
        auto pristine = cqr::detail::mkl_alloc_bytes<double>(sz_a);
        auto work_ap = cqr::detail::mkl_alloc_bytes<double>(sz_a);
        auto taup = cqr::detail::mkl_alloc_bytes<double>(sz_t);
        std::vector<double *> Ap(nmat);
        for (int v = 0; v < nmat; ++v)
            Ap[v] = P.a.data() + (size_t)v * m * n;
        mkl_dgepack_compact(MKL_COL_MAJOR, m, n, Ap.data(), m, pristine.get(), m, fmt,
                            nmat);

        /* Each routine reports its own optimal lwork (MKL's compact geqrf needs
         * real scratch; ours needs none). The cqr kernels share lwork=1. */
        double wq;
        MKL_INT info;
        cqr_mkl_dgeqrf_compact(MKL_COL_MAJOR, m, n, work_ap.get(), m, taup.get(), &wq, -1,
                               &info, fmt, V);
        const MKL_INT lwork_cqr = (MKL_INT)wq;
        mkl_dgeqrf_compact(MKL_COL_MAJOR, m, n, work_ap.get(), m, taup.get(), &wq, -1,
                           &info, fmt, V);
        const MKL_INT lwork_mkl = (MKL_INT)wq;

        /* every compact path factors in place, so restore the packed input
         * (untimed) before each timed pass */
        auto restore = [&] { std::memcpy(work_ap.get(), pristine.get(), sz_a); };

        double t_vec = best_time(reps, restore, [&] {
            factor_compact(Kernel::Vec, work_ap.get(), taup.get(), m, n, ngroups, V, fmt,
                           lwork_cqr);
        });
        double t_out = best_time(reps, restore, [&] {
            factor_compact(Kernel::OmpOuter, work_ap.get(), taup.get(), m, n, ngroups, V,
                           fmt, lwork_cqr);
        });
        double t_in = best_time(reps, restore, [&] {
            factor_compact(Kernel::OmpInner, work_ap.get(), taup.get(), m, n, ngroups, V,
                           fmt, lwork_cqr);
        });
        double t_mkl = best_time(reps, restore, [&] {
            factor_compact(Kernel::Mkl, work_ap.get(), taup.get(), m, n, ngroups, V, fmt,
                           lwork_mkl);
        });

        /* correctness gate: every cqr kernel must match per-matrix LAPACK */
        double rel = 0;
        for (Kernel kk : {Kernel::Vec, Kernel::OmpOuter, Kernel::OmpInner})
            rel = std::max(rel, factor_error(kk, P, fmt, V));
        check(rel <= 1e-9, "compact factorizations match LAPACK");

        const double g_vec = nmat * geqrf_gflop(m, n) / t_vec;
        const double g_out = nmat * geqrf_gflop(m, n) / t_out;
        const double g_in = nmat * geqrf_gflop(m, n) / t_in;
        const double g_mkl = nmat * geqrf_gflop(m, n) / t_mkl;
        log_vec_mkl += std::log(t_mkl / t_vec);
        log_out_mkl += std::log(t_mkl / t_out);
        log_in_mkl += std::log(t_mkl / t_in);
        {
            aligned_vector<double> pool_work = P.a;
            double t_lap = best_time(
                reps, [&] { pool_work = P.a; },
                [&] { factor_unbatched(pool_work.data(), m, n, nmat); });
            log_vec_lap += std::log(t_lap / t_vec);
        }
        std::printf("%4d | %9.2f | %10.2f | %9.2f | %9.2f | %6.2fx | %7.2fx | %6.2fx | "
                    "%.0e\n",
                    n, g_vec, g_out, g_in, g_mkl, t_mkl / t_vec, t_mkl / t_out,
                    t_mkl / t_in, rel);
    }

    std::printf("-----+-----------+------------+-----------+-----------+---------+"
                "----------+---------+--------\n");
    const double ns = (double)sizes.size();
    std::printf("geometric-mean speedup vs mkl_dgeqrf_compact:  vec-types %.2fx | "
                "omp-outer %.2fx | omp-inner %.2fx\n",
                std::exp(log_vec_mkl / ns), std::exp(log_out_mkl / ns),
                std::exp(log_in_mkl / ns));
    std::printf("geometric-mean speedup vs per-matrix LAPACK:   vec-types %.2fx\n",
                std::exp(log_vec_lap / ns));
    return 0;
}
