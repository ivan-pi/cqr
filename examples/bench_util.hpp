/* bench_util.hpp
 *
 * The harness shared by the benchmark programs: abort-on-failure checks, the
 * MKL compact-format lookups, pack-aligned std::vector storage, best-of-N
 * timing, the OpenMP thread count, and the factorization benchmarks' command
 * line (--size-sweep, --simdlen, [nmat] [reps]). Needs MKL headers only.
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#ifndef CQR_BENCH_UTIL_HPP
#define CQR_BENCH_UTIL_HPP

#include "cqr_mkl_ext.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cqr {
namespace bench {

using cqr::detail::compact_format_name; /* format -> "SSE"/"AVX"/"AVX512" */
using cqr::detail::format_for_vlen;     /* interleave width -> pack format */
using cqr::detail::vlen_for_format;     /* pack format -> interleave width */

/* Report and abort on the spot if cond is false. */
inline void check(bool cond, const char *what)
{
    if (!cond) {
        std::printf("FAILED: %s\n", what);
        std::exit(1);
    }
}

/* std::vector storage aligned to the compact pack width (64 B covers every
 * format), so a dense pool and its LAPACK working copy start pack-aligned like
 * the compact buffers -- no cache-line splits in the packing reads or the
 * per-matrix LAPACK path. A stateless allocator: allocator_traits defaults the
 * rest; aligned_alloc needs the size rounded up to the alignment. */
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

/* Best (minimum) wall time over `reps` timed passes, in seconds. `reset` runs
 * untimed before every pass (e.g. to restore input the timed work destroys);
 * only `timed` is clocked, after one untimed warm-up. */
template <typename Reset, typename Timed>
double best_time(int reps, Reset &&reset, Timed &&timed)
{
    using clk = std::chrono::steady_clock;
    reset();
    timed();
    double best = std::numeric_limits<double>::infinity();
    for (int r = 0; r < reps; ++r) {
        reset();
        auto t0 = clk::now();
        timed();
        best = std::min(best, std::chrono::duration<double>(clk::now() - t0).count());
    }
    return best;
}

/* Number of OpenMP threads the outer loops will run on (1 without OpenMP). */
inline int omp_threads()
{
    int nthreads = 1;
#ifdef _OPENMP
#pragma omp parallel
#pragma omp single
    nthreads = omp_get_num_threads();
#endif
    return nthreads;
}

/* Command line of the factorization benchmarks: positional [nmat] [reps], plus
 * --size-sweep=nmin:nmax[:stride] (cqr-only scan) and --simdlen=2|4|8 (force the
 * interleave width instead of the host default). The constructor parses and
 * validates and resolves the pack format; hold the object const. */
struct CmdArgs {
    int nmat = 512;
    int reps = 3;
    bool sweep = false;
    int sweep_min = 0, sweep_max = 0, sweep_step = 1;
    MKL_COMPACT_PACK fmt; /* the host's widest, or the --simdlen one */
    int V;                /* its interleave width for double */

    CmdArgs(int argc, char **argv, const char *prog)
    {
        int simdlen = 0;
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
        if (!(nmat > 0 && reps > 0)) {
            std::printf("usage: %s [--size-sweep=nmin:nmax[:stride]] [--simdlen=2|4|8] "
                        "[nmat>0] [reps>0]\n",
                        prog);
            std::exit(1);
        }
        check(!sweep || (sweep_min > 0 && sweep_max >= sweep_min && sweep_step > 0),
              "usage: --size-sweep needs 0 < nmin <= nmax and stride > 0");
        /* Double compact widths are 2/4/8 (SSE/AVX/AVX512); 16 is float's AVX512
         * width and has no double format. */
        check(simdlen == 0 || simdlen == 2 || simdlen == 4 || simdlen == 8,
              "usage: --simdlen must be 2, 4, or 8 (16 is float-only; this is double)");

        /* A wider interleave than the host's native SIMD cannot execute
         * (mkl_get_format_compact reports the widest the architecture supports). */
        const MKL_COMPACT_PACK native = mkl_get_format_compact();
        fmt = simdlen ? format_for_vlen<double>(simdlen) : native;
        V = vlen_for_format<double>(fmt);
        check(V > 0 && V <= vlen_for_format<double>(native),
              "requested --simdlen exceeds the host's native SIMD width");
    }
};

} /* namespace bench */
} /* namespace cqr */

#endif /* CQR_BENCH_UTIL_HPP */
