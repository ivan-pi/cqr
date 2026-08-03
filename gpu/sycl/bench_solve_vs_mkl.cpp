/* bench_solve_vs_mkl.cpp -- batched AX=b solve throughput, three paths:
 *   (1) non-batch LAPACK (MKL): per-matrix LAPACKE_dgeqrf/dormqr + cblas_dtrsm
 *   (2) compact batch (MKL): mkl_dgeqrf_compact + cqr_mkl_dormqr_compact + mkl_dtrsm_compact
 *   (3) fused SYCL: one kernel geqrf_slot+ormqr_slot+trsm_upper_slot (this project)
 * Same pool of nm random well-conditioned n x n systems, known solution X == 1.
 * All paths use all cores (OpenMP over the pool/groups for MKL; the OpenCL
 * runtime for SYCL). End-to-end host-in/host-out, so the SYCL path's timed
 * region includes its host<->device copies.
 *
 * CPU-DEVICE CAVEAT: no Intel GPU here, so path (3) runs on the OpenCL CPU
 * device -- this is three ways to use the SAME CPU, NOT a GPU-vs-CPU result.
 * MKL's compact kernels are hand-tuned CPU code; the SYCL "device" copies are
 * host memcpys. On a real GPU path (3) would be a different machine entirely,
 * and path (2) would not exist at all (MKL Compact is CPU-only).
 *
 * Needs MKL + a SYCL compiler + OpenMP, so it is NOT wired into CMake (the repo's
 * MKL discovery targets a oneAPI/MKLROOT setup, not system MKL under icpx). Build
 * it by hand, e.g. with Debian libmkl-dev headers/libs and icpx:
 *
 *   icpx -fsycl -O3 -march=native -qopenmp -std=c++17 \
 *        -I/usr/include/mkl -I../../src -I. \
 *        bench_solve_vs_mkl.cpp ../../src/cqr_mkl_ext.cpp \
 *        -L/usr/lib/x86_64-linux-gnu -lmkl_intel_lp64 -lmkl_sequential -lmkl_core \
 *        -lpthread -lm -ldl -o bench_solve_vs_mkl
 *   OMP_NUM_THREADS=4 ./bench_solve_vs_mkl [nmat] [reps]
 *
 * Assisted-by: Claude:claude-opus-4.8
 */
#include <mkl.h>
#include <mkl_compact.h>
#include <sycl/sycl.hpp>

#include "cqr_mkl_ext.h"       /* cqr_mkl_dormqr_compact, vlen_for_format */
#include "cqr_compact_sycl.hpp" /* cqr::gpu primitives */

#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

using clk = std::chrono::steady_clock;
static double secs(clk::duration d) { return std::chrono::duration<double>(d).count(); }
constexpr int V = 8; /* AVX-512 double compact width = sub-group size */

struct Pool {
    int n, nmat;
    std::vector<double> a, b; /* nmat*n*n , nmat*n */
    Pool(int n_, int nmat_) : n(n_), nmat(nmat_), a((size_t)nmat_ * n_ * n_), b((size_t)nmat_ * n_)
    {
        std::mt19937_64 rng(42);
        std::uniform_real_distribution<double> dist(-1, 1);
        for (int v = 0; v < nmat; ++v) {
            double *A = a.data() + (size_t)v * n * n;
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) A[i + (size_t)j * n] = dist(rng);
            for (int i = 0; i < n; ++i) A[i + (size_t)i * n] += 2.0 * n;
            double *B = b.data() + (size_t)v * n;
            for (int i = 0; i < n; ++i) {
                double s = 0;
                for (int j = 0; j < n; ++j) s += A[i + (size_t)j * n];
                B[i] = s; /* B = A * 1 */
            }
        }
    }
};
static double sol_err(const double *x, int n) { double e = 0; for (int i = 0; i < n; ++i) e = std::max(e, std::fabs(x[i] - 1.0)); return e; }

/* (1) per-matrix LAPACK, factoring in place (caller refreshes a,b each pass). */
static double run_unbatched(int n, int nmat, double *a, double *b)
{
    double maxerr = 0;
#pragma omp parallel reduction(max : maxerr)
    {
        std::vector<double> tau(n);
#pragma omp for schedule(static)
        for (int v = 0; v < nmat; ++v) {
            double *A = a + (size_t)v * n * n, *B = b + (size_t)v * n;
            LAPACKE_dgeqrf(LAPACK_COL_MAJOR, n, n, A, n, tau.data());
            LAPACKE_dormqr(LAPACK_COL_MAJOR, 'L', 'T', n, 1, n, A, n, tau.data(), B, n);
            cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                        n, 1, 1.0, A, n, B, n);
            maxerr = std::max(maxerr, sol_err(B, n));
        }
    }
    return maxerr;
}

/* (2) compact batch: pack -> geqrf_compact -> cqr ormqr -> trsm_compact -> unpack. */
static double run_compact(const Pool &P, MKL_COMPACT_PACK fmt)
{
    const int n = P.n, nmat = P.nmat, ng = (nmat + V - 1) / V;
    double maxerr = 0;
#pragma omp parallel reduction(max : maxerr)
    {
        std::vector<double> ap((size_t)mkl_dget_size_compact(n, n, fmt, V) / 8 + 8),
            taup((size_t)mkl_dget_size_compact(n, 1, fmt, V) / 8 + 8),
            bp((size_t)mkl_dget_size_compact(n, 1, fmt, V) / 8 + 8);
        MKL_INT info;
        double wq;
        mkl_dgeqrf_compact(MKL_COL_MAJOR, n, n, ap.data(), n, taup.data(), &wq, -1, &info, fmt, V);
        std::vector<double> work((size_t)std::max<MKL_INT>((MKL_INT)wq, 1));
        std::vector<double *> Aptr(V), Bptr(V), Xptr(V);
        std::vector<double> xout((size_t)V * n);
#pragma omp for schedule(static)
        for (int g = 0; g < ng; ++g) {
            const int base = g * V;
            const MKL_INT cnt = std::min(V, nmat - base);
            for (int s = 0; s < cnt; ++s) {
                Aptr[s] = const_cast<double *>(P.a.data()) + (size_t)(base + s) * n * n;
                Bptr[s] = const_cast<double *>(P.b.data()) + (size_t)(base + s) * n;
                Xptr[s] = xout.data() + (size_t)s * n;
            }
            mkl_dgepack_compact(MKL_COL_MAJOR, n, n, Aptr.data(), n, ap.data(), n, fmt, cnt);
            mkl_dgepack_compact(MKL_COL_MAJOR, n, 1, Bptr.data(), n, bp.data(), n, fmt, cnt);
            mkl_dgeqrf_compact(MKL_COL_MAJOR, n, n, ap.data(), n, taup.data(), work.data(), (MKL_INT)wq, &info, fmt, cnt);
            double dummy;
            cqr_mkl_dormqr_compact(MKL_COL_MAJOR, 'L', 'T', n, 1, n, ap.data(), n, taup.data(), bp.data(), n, &dummy, 1, &info, fmt, cnt);
            mkl_dtrsm_compact(MKL_COL_MAJOR, MKL_LEFT, MKL_UPPER, MKL_NOTRANS, MKL_NONUNIT, n, 1, 1.0, ap.data(), n, bp.data(), n, fmt, cnt);
            mkl_dgeunpack_compact(MKL_COL_MAJOR, n, 1, Xptr.data(), n, bp.data(), n, fmt, cnt);
            for (int s = 0; s < cnt; ++s) maxerr = std::max(maxerr, sol_err(Xptr[s], n));
        }
    }
    return maxerr;
}

/* (3) fused SYCL: host-pack -> device -> one kernel -> device -> host-unpack. */
static double run_fused_sycl(sycl::queue &q, const Pool &P, double *ap_d, double *tp_d, double *bp_d,
                             std::vector<double> &ap_h, std::vector<double> &bp_h,
                             std::vector<double> &xh)
{
    const int n = P.n, nmat = P.nmat, ng = (nmat + V - 1) / V;
    const size_t na = (size_t)ng * n * n * V, nb = (size_t)ng * n * V;
    /* pack pool -> compact host buffers (padded slots = identity / 0) */
#pragma omp parallel for schedule(static)
    for (int g = 0; g < ng; ++g)
        for (int v = 0; v < V; ++v) {
            const int idx = g * V + v;
            const double *A = (idx < nmat) ? P.a.data() + (size_t)idx * n * n : nullptr;
            const double *B = (idx < nmat) ? P.b.data() + (size_t)idx * n : nullptr;
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    ap_h[(size_t)g * n * n * V + ((size_t)j * n + i) * V + v] =
                        A ? A[i + (size_t)j * n] : (i == j ? 1.0 : 0.0);
            for (int i = 0; i < n; ++i)
                bp_h[(size_t)g * n * V + (size_t)i * V + v] = B ? B[i] : 0.0;
        }
    q.memcpy(ap_d, ap_h.data(), na * 8).wait();
    q.memcpy(bp_d, bp_h.data(), nb * 8).wait();
    auto nd = sycl::nd_range<1>(sycl::range<1>((size_t)ng * V), sycl::range<1>(V));
    q.submit([=](sycl::handler &h) {
         h.parallel_for(nd, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(V)]] {
             int g = it.get_group(0), v = it.get_local_id(0);
             cqr::gpu::geqrf_slot<double, V>(g, v, ap_d, n, tp_d, n, n);
             cqr::gpu::ormqr_slot<double, V>(g, v, ap_d, n, tp_d, bp_d, n, n, 1, n, true);
             cqr::gpu::trsm_upper_slot<double, V>(g, v, ap_d, n, bp_d, n, n, 1);
         });
     }).wait();
    q.memcpy(bp_h.data(), bp_d, nb * 8).wait();
    double maxerr = 0;
    for (int idx = 0; idx < nmat; ++idx) {
        const int g = idx / V, v = idx % V;
        for (int i = 0; i < n; ++i) xh[i] = bp_h[(size_t)g * n * V + (size_t)i * V + v];
        maxerr = std::max(maxerr, sol_err(xh.data(), n));
    }
    return maxerr;
}

template <class Reset, class Timed> static double best_time(int reps, Reset reset, Timed timed)
{
    reset(); timed();
    double best = std::numeric_limits<double>::infinity();
    for (int r = 0; r < reps; ++r) { reset(); auto t0 = clk::now(); timed(); best = std::min(best, secs(clk::now() - t0)); }
    return best;
}

int main(int argc, char **argv)
{
    const int nmat = (argc > 1) ? std::atoi(argv[1]) : 512;
    const int reps = (argc > 2) ? std::atoi(argv[2]) : 5;
    mkl_set_num_threads(1);          /* OpenMP over the pool is the only parallelism */
    LAPACKE_set_nancheck(0);
    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    if (cqr::detail::vlen_for_format<double>(fmt) != V) {
        std::printf("compact double width != %d on this CPU; benchmark assumes AVX-512\n", V);
        return 77;
    }
    sycl::queue q{sycl::default_selector_v};
    auto sgs = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
    int nth = 1;
#pragma omp parallel
#pragma omp single
    nth = omp_get_num_threads();
    std::fprintf(stderr, "SYCL device: %s | OpenMP threads: %d\n",
                 q.get_device().get_info<sycl::info::device::name>().c_str(), nth);
    if (std::find(sgs.begin(), sgs.end(), (size_t)V) == sgs.end()) { std::printf("sub-group %d unsupported\n", V); return 77; }

    std::printf("batched AX=b solve throughput (matrices/s), nmat=%d, reps=%d, %d cores\n", nmat, reps, nth);
    std::printf("  (1) per-matrix MKL   (2) compact-batch MKL   (3) fused SYCL [OpenCL CPU device]\n\n");
    std::printf("   n | (1) Mmat/s | (2) Mmat/s | (3) Mmat/s | (3)vs(1) | (3)vs(2) | max err\n");
    std::printf("-----+------------+------------+------------+----------+----------+--------\n");

    for (int n : {16, 24, 32, 48, 64, 96, 128}) {
        Pool P(n, nmat);
        const int ng = (nmat + V - 1) / V;
        double *ap_d = sycl::malloc_device<double>((size_t)ng * n * n * V, q);
        double *tp_d = sycl::malloc_device<double>((size_t)ng * n * V, q);
        double *bp_d = sycl::malloc_device<double>((size_t)ng * n * V, q);
        std::vector<double> ap_h((size_t)ng * n * n * V), bp_h((size_t)ng * n * V), xh(n), wa, wb;

        double e1 = 0, e2 = 0, e3 = 0;
        double t1 = best_time(reps, [&] { wa = P.a; wb = P.b; }, [&] { e1 = run_unbatched(n, nmat, wa.data(), wb.data()); });
        double t2 = best_time(reps, [] {}, [&] { e2 = run_compact(P, fmt); });
        double t3 = best_time(reps, [] {}, [&] { e3 = run_fused_sycl(q, P, ap_d, tp_d, bp_d, ap_h, bp_h, xh); });

        const double m1 = nmat / t1 / 1e6, m2 = nmat / t2 / 1e6, m3 = nmat / t3 / 1e6;
        std::printf("%4d | %10.2f | %10.2f | %10.2f | %7.2fx | %7.2fx | %.1e\n",
                    n, m1, m2, m3, t1 / t3, t2 / t3, std::max({e1, e2, e3}));
        sycl::free(ap_d, q); sycl::free(tp_d, q); sycl::free(bp_d, q);
    }
    return 0;
}
