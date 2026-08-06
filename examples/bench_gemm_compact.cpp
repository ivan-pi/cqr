/* bench_gemm_compact.cpp
 *
 * Where the compact (interleaved-batch) GEMM wins and where it does not -- the
 * context for the compact QR's large-n behaviour (cqr_geqrf_compact_cache_blocking.md).
 * Single-thread, three throughputs of the same batched work:
 *
 *   compact       mkl_dgemm_compact over the whole batch (SIMD across the V matrices)
 *   dgemm-loop    a per-matrix cblas_dgemm loop over the batch (register-blocked)
 *   dgemm-1       one dense cblas_dgemm (the compute-bound per-matrix ceiling)
 *
 * Two facts fall out:
 *   1. The compact format wins only for *very small* matrices (n <~ 14 here), where
 *      a single matrix cannot fill a SIMD vector and per-call overhead dominates --
 *      at n=4 it is ~12x the per-matrix loop. Past the crossover the register-blocked
 *      per-matrix GEMM pulls away.
 *   2. The compact GEMM plateaus several-fold below a regular GEMM and stays flat in
 *      n: interleaving spends the registers on the batch dimension, so it cannot also
 *      register-block within a matrix. This is the ceiling the blocked compact QR's
 *      larfb reaches -- and why routing larfb through mkl_dgemm_compact cannot beat
 *      the portable kernel, and why matching per-matrix LAPACK at large n needs the
 *      unpack-to-dense hybrid, not more compact tuning.
 *
 * Usage:  bench_gemm_compact [nmat] [reps]      (defaults: 512 matrices, 15 reps)
 * Build with -O3 -march=native. Needs Intel MKL (compact GEMM + CBLAS).
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#include <mkl.h>
#include <mkl_compact.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

using clk = std::chrono::steady_clock;

template <typename F> double best_time(int reps, F &&run)
{
    run();
    double b = 1e30;
    for (int r = 0; r < reps; ++r) {
        auto t0 = clk::now();
        run();
        double dt = std::chrono::duration<double>(clk::now() - t0).count();
        if (dt < b) b = dt;
    }
    return b;
}

} /* anonymous namespace */

int main(int argc, char **argv)
{
    const int nmat = argc > 1 ? std::atoi(argv[1]) : 512;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 15;
    mkl_set_num_threads(1);
    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();

    std::printf("Compact GEMM vs per-matrix dgemm for a batch of many square matrices "
                "(single-thread, nmat=%d)\n",
                nmat);
    std::printf("aggregate GFLOP/s = 2 N^3 nmat / t;  dgemm-1 is one matrix (the "
                "compute-bound ceiling)\n\n");
    std::printf("   N | compact | dgemm-loop | compact/loop | dgemm-1 (ceiling)\n");
    std::printf("-----+---------+------------+--------------+------------------\n");

    std::mt19937_64 rng(3);
    std::uniform_real_distribution<double> d(-1, 1);

    for (int N : {4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384}) {
        const int m = N, n = N, k = N;
        std::vector<double> da((size_t)nmat * m * k), db((size_t)nmat * k * n),
            dc((size_t)nmat * m * n, 0.0);
        for (auto &x : da)
            x = d(rng);
        for (auto &x : db)
            x = d(rng);

        MKL_INT sza = mkl_dget_size_compact(m, k, fmt, nmat);
        MKL_INT szb = mkl_dget_size_compact(k, n, fmt, nmat);
        MKL_INT szc = mkl_dget_size_compact(m, n, fmt, nmat);
        std::vector<double> A(sza), B(szb), C(szc, 0.0);
        std::vector<double *> Ap(nmat), Bp(nmat);
        for (int v = 0; v < nmat; ++v) {
            Ap[v] = da.data() + (size_t)v * m * k;
            Bp[v] = db.data() + (size_t)v * k * n;
        }
        mkl_dgepack_compact(MKL_COL_MAJOR, m, k, Ap.data(), m, A.data(), m, fmt, nmat);
        mkl_dgepack_compact(MKL_COL_MAJOR, k, n, Bp.data(), k, B.data(), k, fmt, nmat);

        const double tc = best_time(reps, [&] {
            mkl_dgemm_compact(MKL_COL_MAJOR, MKL_NOTRANS, MKL_NOTRANS, m, n, k, 1.0,
                              A.data(), m, B.data(), k, 0.0, C.data(), m, fmt, nmat);
        });
        const double tl = best_time(reps, [&] {
            for (int v = 0; v < nmat; ++v)
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0,
                            da.data() + (size_t)v * m * k, m,
                            db.data() + (size_t)v * k * n, k, 0.0,
                            dc.data() + (size_t)v * m * n, m);
        });
        const double t1 = best_time(reps, [&] {
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0,
                        da.data(), m, db.data(), k, 0.0, dc.data(), m);
        });

        const double gf = 2.0 * m * n * k * 1e-9;
        std::printf("%4d | %7.1f | %10.1f | %11.2fx | %14.1f\n", N, gf * nmat / tc,
                    gf * nmat / tl, tl / tc, gf / t1);
    }
    std::printf("-----+---------+------------+--------------+------------------\n");
    std::printf(
        "compact wins for tiny n (batch amortizes); per-matrix dgemm wins past "
        "the crossover and\nsets a ceiling the compact format -- MKL's or ours -- "
        "cannot reach. Hence larfb's plateau.\n");
    return 0;
}
