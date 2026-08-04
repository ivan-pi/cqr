/* test_cqr_ispc.cpp
 *
 * Correctness test for the ISPC compact-QR prototype (cqr_ispc.ispc), exercised
 * through the full batched AX = B solve and validated two ways:
 *
 *   1. Forward error vs a KNOWN exact solution X (independent of any library):
 *      build B = A X, run the ISPC pipeline
 *          cqr_ispc_dgeqrf_compact -> cqr_ispc_dormqr_compact('T')
 *                                  -> cqr_ispc_dtrsm_compact
 *      unpack, and require max|Xhat - X| within 100 n eps.
 *
 *   2. Drop-in equivalence vs the existing GNU-vector kernels (../src): run the
 *      reference pipeline (dgeqrf_compact -> dormqr_compact -> mkl_dtrsm_compact)
 *      on the SAME packed inputs and require the ISPC factor buffer and the ISPC
 *      solution to agree with it to a tight tolerance. This is what makes the
 *      extern "C" ISPC routines genuine drop-in replacements.
 *
 * MKL supplies only the pack/unpack (guaranteeing the exact Compact layout the
 * kernels assume) and the reference trsm; the ISPC kernels themselves use no MKL.
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#include <mkl.h>
#include <mkl_compact.h>

#include "cqr_ispc.h"    /* the ISPC prototype (extern "C")            */
#include "cqr_compact.h" /* reference GNU-vector kernels (../src)      */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <vector>

namespace {

std::mt19937_64 rng(12345);
double frand()
{
    static std::uniform_real_distribution<double> dist(-1.0, 1.0);
    return dist(rng);
}

int failures = 0;
void expect(bool cond, const char *what)
{
    if (!cond) {
        std::printf("  FAIL: %s\n", what);
        ++failures;
    }
}

/* 64-byte-aligned double buffer (pack width for every compact format). */
struct AlignedBuf {
    double *p = nullptr;
    explicit AlignedBuf(size_t n)
    {
        /* aligned_alloc needs a size that is a multiple of the alignment */
        size_t bytes = ((n * sizeof(double) + 63) / 64) * 64;
        p = static_cast<double *>(std::aligned_alloc(64, bytes));
        std::memset(p, 0, bytes); /* deterministic padding lanes */
    }
    ~AlignedBuf() { std::free(p); }
    AlignedBuf(const AlignedBuf &) = delete;
    AlignedBuf &operator=(const AlignedBuf &) = delete;
};

/* Column-major dense matrix owning its storage. */
struct Dense {
    int rows, cols;
    std::vector<double> a;
    Dense(int r, int c) : rows(r), cols(c), a((size_t)r * c, 0.0) {}
    double &operator()(int i, int j) { return a[(size_t)i + (size_t)j * rows]; }
    double operator()(int i, int j) const { return a[(size_t)i + (size_t)j * rows]; }
    double *data() { return a.data(); }
};

std::vector<double *> base_ptrs(std::vector<Dense> &batch)
{
    std::vector<double *> p;
    p.reserve(batch.size());
    for (Dense &m : batch)
        p.push_back(m.data());
    return p;
}

/* max_v max_ij |P_v - Q_v| over the first `nm` matrices of two compact buffers
 * holding (rows x cols) matrices at width V (no padding when nm % V == 0). */
double compact_maxdiff(const double *P, const double *Q, int rows, int cols, int V,
                       int nm)
{
    double d = 0.0;
    const int ngroups = (nm + V - 1) / V;
    for (int g = 0; g < ngroups; ++g) {
        const size_t base = (size_t)g * rows * cols * V; /* ldap == rows here */
        for (int e = 0; e < rows * cols; ++e)
            for (int v = 0; v < V; ++v) {
                const int mat = g * V + v;
                if (mat >= nm) continue;
                const size_t off = base + (size_t)e * V + v;
                d = std::max(d, std::fabs(P[off] - Q[off]));
            }
    }
    return d;
}

const double EPS = std::numeric_limits<double>::epsilon();

/* One configuration: nm matrices of order n with nrhs right-hand sides. */
void run_case(MKL_COMPACT_PACK fmt, int V, int nm, int n, int nrhs, bool check_equiv)
{
    /* Known exact solution X(i,j) = (j+1) + 0.5 i, shared across the batch. */
    Dense X(n, nrhs);
    for (int j = 0; j < nrhs; ++j)
        for (int i = 0; i < n; ++i)
            X(i, j) = double(j + 1) + 0.5 * i;

    std::vector<Dense> A(nm, Dense(n, n)), B(nm, Dense(n, nrhs));
    for (int v = 0; v < nm; ++v) {
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                A[v](i, j) = frand();
        for (int i = 0; i < n; ++i)
            A[v](i, i) += 2.0 * n; /* diagonally dominant: tame conditioning */
        /* B_v = A_v X */
        for (int j = 0; j < nrhs; ++j)
            for (int i = 0; i < n; ++i) {
                double s = 0.0;
                for (int l = 0; l < n; ++l)
                    s += A[v](i, l) * X(l, j);
                B[v](i, j) = s;
            }
    }

    /* Pack the dense batch into two pristine compact copies (A and B). */
    const size_t szA = (size_t)mkl_dget_size_compact(n, n, fmt, nm);
    const size_t szT = (size_t)mkl_dget_size_compact(n, 1, fmt, nm);
    const size_t szB = (size_t)mkl_dget_size_compact(n, nrhs, fmt, nm);

    AlignedBuf ap0(szA), bp0(szB);
    {
        auto Ap = base_ptrs(A), Bp = base_ptrs(B);
        mkl_dgepack_compact(MKL_COL_MAJOR, n, n, Ap.data(), n, ap0.p, n, fmt, nm);
        mkl_dgepack_compact(MKL_COL_MAJOR, n, nrhs, Bp.data(), n, bp0.p, n, fmt, nm);
    }

    /* ---- ISPC pipeline ------------------------------------------------- */
    AlignedBuf api(szA), tpi(szT), bpi(szB);
    std::memcpy(api.p, ap0.p, szA * sizeof(double));
    std::memcpy(bpi.p, bp0.p, szB * sizeof(double));
    cqr_ispc_dgeqrf_compact(n, n, api.p, n, tpi.p, nm);
    cqr_ispc_dormqr_compact(/*trans=Q^T*/ 1, n, nrhs, n, api.p, n, tpi.p, bpi.p, n, nm);
    cqr_ispc_dtrsm_compact(n, nrhs, 1.0, api.p, n, bpi.p, n, nm);

    std::vector<Dense> Xi(nm, Dense(n, nrhs));
    {
        auto Xp = base_ptrs(Xi);
        mkl_dgeunpack_compact(MKL_COL_MAJOR, n, nrhs, Xp.data(), n, bpi.p, n, fmt, nm);
    }

    /* forward error of the ISPC solve vs the known X */
    double fwd_ispc = 0.0;
    for (int v = 0; v < nm; ++v)
        for (int j = 0; j < nrhs; ++j)
            for (int i = 0; i < n; ++i)
                fwd_ispc = std::max(fwd_ispc, std::fabs(Xi[v](i, j) - X(i, j)));

    const double rtol = 100.0 * n * EPS * (1.0 + 0.5 * n); /* scale by |X| ~ 0.5 n */
    expect(fwd_ispc <= rtol, "ISPC solve recovers known X");

    double equiv_fac = 0.0, equiv_sol = 0.0;
    if (check_equiv) {
        /* ---- reference pipeline: cqr GNU-vector kernels + MKL trsm ------ */
        AlignedBuf apr(szA), tpr(szT), bpr(szB);
        std::memcpy(apr.p, ap0.p, szA * sizeof(double));
        std::memcpy(bpr.p, bp0.p, szB * sizeof(double));
        int info = 0;
        info = dgeqrf_compact('C', n, n, apr.p, n, tpr.p, V, nm);
        expect(info == 0, "reference dgeqrf_compact info==0");
        info = dormqr_compact('T', n, nrhs, n, apr.p, n, tpr.p, bpr.p, n, V, nm);
        expect(info == 0, "reference dormqr_compact info==0");
        mkl_dtrsm_compact(MKL_COL_MAJOR, MKL_LEFT, MKL_UPPER, MKL_NOTRANS, MKL_NONUNIT, n,
                          nrhs, 1.0, apr.p, n, bpr.p, n, fmt, nm);

        /* factor buffers must match: same unblocked geqr2, same order. Allow a
         * few ulp for FMA-contraction differences between ISPC and GCC. */
        equiv_fac = compact_maxdiff(api.p, apr.p, n, n, V, nm);
        equiv_sol = compact_maxdiff(bpi.p, bpr.p, n, nrhs, V, nm);
        const double ftol = 1e-9 * (1.0 + 2.0 * n); /* relative to |R| ~ 2n */
        expect(equiv_fac <= ftol, "ISPC factor matches reference geqrf");
        expect(equiv_sol <= rtol, "ISPC solution matches reference pipeline");
    }

    std::printf("  nm=%-3d n=%-4d nrhs=%d | ISPC fwd %.2e (rtol %.1e)%s\n", nm, n, nrhs,
                fwd_ispc, rtol,
                check_equiv ? [&] {
                    static char buf[80];
                    std::snprintf(buf, sizeof buf, " | equiv fac %.1e sol %.1e",
                                  equiv_fac, equiv_sol);
                    return buf;
                }()
                            : "");
}

} /* namespace */

int main()
{
    /* Sequential MKL, pinned before the first MKL call (loads libmkl_sequential,
     * no OpenMP runtime needed) -- no MKL_THREADING_LAYER env var required. */
    mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL);
    mkl_set_num_threads(1);

    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = (fmt == MKL_COMPACT_SSE ? 16 : fmt == MKL_COMPACT_AVX ? 32 : 64) / 8;
    std::printf("ISPC compact-QR correctness (compact format=%d, V=%d)\n", (int)fmt, V);
    if (V != 8) {
        std::printf("NOTE: ISPC prototype is built for V=8 (avx512skx-x8); this host's\n"
                    "MKL format is V=%d. Rebuild ISPC with a matching --target.\n",
                    V);
        return 77; /* CTest "skipped" */
    }

    /* Equivalence-checked cases: nm a multiple of V so there is no padding to
     * complicate the element-wise factor comparison. */
    run_case(fmt, V, 8, 10, 1, true);
    run_case(fmt, V, 16, 20, 4, true);
    run_case(fmt, V, 32, 40, 5, true);
    run_case(fmt, V, 8, 100, 8, true);
    run_case(fmt, V, 16, 150, 3, true);

    /* Padded final group (nm not a multiple of V): forward error only, since the
     * padding lanes are never unpacked. */
    run_case(fmt, V, 7, 32, 5, false);
    run_case(fmt, V, 13, 64, 2, false);
    run_case(fmt, V, 1, 48, 4, false);

    if (failures) {
        std::printf("\n%d CHECK(S) FAILED\n", failures);
        return 1;
    }
    std::printf("\nall checks passed\n");
    return 0;
}
