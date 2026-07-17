/* bench_geqrf_batmat.cpp
 *
 * Benchmark 2 of the design document (section 9): the compact batched QR
 * factorization of this project against the open-source batmat project
 * (https://github.com/tttapa/batmat), which stores batches of small matrices in
 * the same interleaved "compact" format and factors them with SIMD
 * micro-kernels. This pits cqr_mkl_dgeqrf_compact head to head with
 * batmat::linalg::geqrf on identical problem sizes and batch depth, timing both
 * with the same clock and reporting GFLOP/s.
 *
 * The batmat side mirrors batmat's own benchmarks/geqrf.cpp (matrix<> +
 * deduced_abi + geqrf_size_W + geqrf(A,B,W)); see that file for the reference
 * usage. batmat is GPL-licensed and is used here only as an external yardstick
 * inside our own benchmark -- it is neither modified nor redistributed, so this
 * imposes no licensing obligation on cqr.
 *
 * Build: OPT-IN. batmat pulls a C++23 toolchain plus its own dependencies
 * (guanaqo, the SIMD backend, ...), normally provisioned through batmat's Conan
 * flow, so this benchmark is gated behind the CMake option CQR_WITH_BATMAT
 * (default OFF) and is not part of the standard build or CTest. Provide a batmat
 * install to CMake (e.g. a Conan toolchain or CMAKE_PREFIX_PATH) and configure
 * with -DCQR_WITH_BATMAT=ON. It links batmat::batmat, guanaqo::blas, and this
 * repo's cqr_mkl_ext (plus MKL for packing).
 *
 * Usage:  bench_geqrf_batmat [depth]        (default depth: 512 matrices)
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#include <batmat/linalg/geqrf.hpp>
#include <batmat/linalg/copy.hpp>
#include <batmat/matrix/matrix.hpp>

#include <mkl.h>
#include <mkl_compact.h>

#include "cqr_mkl_ext.h"
#include "cqr_mkl_alloc.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <ranges>
#include <vector>
#include <algorithm>

namespace {

using clk = std::chrono::steady_clock;
using batmat::index_t;
using batmat::real_t;                       /* double in batmat's default config */
using batmat::linalg::StorageOrder;
using batmat::matrix::Matrix;
using cqr::detail::vlen_for_format;

/* AVX-512 double lane width (8); batmat and cqr both interleave 8 matrices. */
using abi8 = batmat::datapar::deduced_abi<real_t, 8>;
template <StorageOrder O>
using bmat = Matrix<real_t, index_t, typename abi8::size_type, index_t, O>;

static_assert(std::is_same_v<real_t, double>,
              "this benchmark compares double precision; build batmat with real_t=double");

double geqrf_gflop(int m, int n)   /* standard ?geqrf flop count, m >= n */
{
    return (2.0 * m * n * (double)n - (2.0 / 3.0) * n * (double)n * n) * 1e-9;
}

template <typename Timed>
double best_time(int reps, Timed &&timed)
{
    timed();                                /* warm-up */
    double best = std::numeric_limits<double>::infinity();
    for (int r = 0; r < reps; ++r) {
        auto t0 = clk::now();
        timed();
        best = std::min(best, std::chrono::duration<double>(clk::now() - t0).count());
    }
    return best;
}

/* ---- batmat path: factor a depth-d batch of n x n matrices, ColMajor ---- */
double run_batmat(int n, int depth, int reps)
{
    std::mt19937 rng{12345};
    std::uniform_real_distribution<real_t> uni{-1, 1};

    bmat<StorageOrder::ColMajor> A{{.depth = depth, .rows = n, .cols = n}};
    bmat<StorageOrder::ColMajor> B{{.depth = depth, .rows = n, .cols = n}};
    auto [rw, cw] = batmat::linalg::geqrf_size_W(A.batch(0));
    Matrix<real_t, index_t, typename abi8::size_type, index_t> W{{.depth = depth, .rows = rw, .cols = cw}};
    std::ranges::generate(A, [&] { return uni(rng); });

    return best_time(reps, [&] {
        for (index_t l = 0; l < A.num_batches(); ++l)
            batmat::linalg::geqrf(A.batch(l), B.batch(l), W.batch(l));
    });
}

/* ---- cqr path: pack a depth-d batch and factor with cqr_mkl_dgeqrf_compact - */
double run_cqr(int n, int depth, int reps, MKL_COMPACT_PACK fmt, int V)
{
    const int m = n, k = n;
    std::mt19937 rng{12345};
    std::uniform_real_distribution<double> uni{-1, 1};
    std::vector<double> pool((size_t)depth * m * n);
    std::ranges::generate(pool, [&] { return uni(rng); });

    MKL_INT sz_a = mkl_dget_size_compact(m, n, fmt, depth);
    MKL_INT sz_t = mkl_dget_size_compact(k, 1, fmt, depth);
    auto pristine = cqr::detail::mkl_alloc_bytes<double>(sz_a);
    auto work_ap  = cqr::detail::mkl_alloc_bytes<double>(sz_a);
    auto taup     = cqr::detail::mkl_alloc_bytes<double>(sz_t);
    std::vector<double *> Ap(depth);
    for (int v = 0; v < depth; ++v) Ap[v] = pool.data() + (size_t)v * m * n;
    mkl_dgepack_compact(MKL_COL_MAJOR, m, n, Ap.data(), m, pristine.get(), m, fmt, depth);

    const int ngroups = (depth + V - 1) / V;
    return best_time(reps, [&] {
        std::memcpy(work_ap.get(), pristine.get(), sz_a);       /* restore input */
        MKL_INT info;
        double wq;
        for (int g = 0; g < ngroups; ++g)
            cqr_mkl_dgeqrf_compact(MKL_COL_MAJOR, m, n,
                                   work_ap.get() + (size_t)g * m * n * V, m,
                                   taup.get() + (size_t)g * k * V,
                                   &wq, 1, &info, fmt, V);
    });
}

} /* anonymous namespace */

int main(int argc, char **argv)
{
    const int depth = (argc > 1) ? std::atoi(argv[1]) : 512;
    const int reps  = 3;
    if (depth <= 0) { std::printf("usage: bench_geqrf_batmat [depth>0]\n"); return 1; }

    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = vlen_for_format<double>(fmt);
    mkl_set_num_threads(1);

    std::printf("QR factorization: cqr_mkl_dgeqrf_compact vs batmat::linalg::geqrf "
                "(double, depth=%d, single-thread)\n\n", depth);
    std::printf("   n | cqr GFLOP/s | batmat GFLOP/s | cqr/batmat\n");
    std::printf("-----+-------------+----------------+-----------\n");

    const int sizes[] = {16, 30, 60, 100, 150, 250};
    for (int n : sizes) {
        const double gtot = depth * geqrf_gflop(n, n);
        const double t_cqr = run_cqr(n, depth, reps, fmt, V);
        const double t_bat = run_batmat(n, depth, reps);
        std::printf("%4d | %11.2f | %14.2f | %8.2fx\n",
                    n, gtot / t_cqr, gtot / t_bat, t_bat / t_cqr);
    }
    return 0;
}
