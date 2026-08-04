/* bench_geqrf_omp_simd.cpp
 *
 * Head-to-head throughput of two vectorization strategies for the *same*
 * compact batched QR factorization, over pools of many small matrices:
 *
 *   vec-types   cqr::detail::geqrf_compact_general       (GNU __attribute__((vector_size)))
 *   omp-simd    cqr::detail::omp_simd::geqrf_compact_general_omp  (#pragma omp simd)
 *
 * Both are the unblocked geqr2 on identical compact buffers; only the SIMD
 * mechanism differs, so this isolates "hand-written vector types" vs "let the
 * compiler vectorize the outer lane loop under -fopenmp-simd". No MKL and no
 * BLAS: the two kernel templates are called directly (they carry the same C
 * symbol when wrapped, so they cannot both be linked through the C API -- here
 * we call the templates), and correctness is gated against a scalar geqr2
 * reference so the benchmark doubles as a portable integration test.
 *
 * The batch is packed once, up front; only the factorization is timed, and the
 * destroyed input is restored (untimed) before each pass. Single-threaded on
 * purpose -- the point is the SIMD kernel, not thread scaling.
 *
 * Usage:  bench_geqrf_omp_simd [--simdlen=2|4|8|16] [nmat] [reps]
 *         (defaults: host-native width for double, 256 matrices, 5 reps)
 *
 * Build with -O3 -march=native -fopenmp-simd (both compilers). Wired up by
 * CMakeLists.txt as the `bench_geqrf_omp_simd` target.
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#include "cqr_geqrf_compact.hpp"     /* vector-types kernel  */
#include "cqr_geqrf_compact_omp.hpp" /* omp-simd kernel      */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <utility>
#include <vector>

namespace {

using clk = std::chrono::steady_clock;

void die(const char *msg)
{
    std::printf("FAILED: %s\n", msg);
    std::exit(1);
}

/* Standard LAPACK ?geqrf flop count (m >= n), in GFLOP. */
double geqrf_gflop(int m, int n)
{
    return (2.0 * m * n * (double)n - (2.0 / 3.0) * n * (double)n * n) * 1e-9;
}

/* ----------------------- scalar geqr2 reference ---------------------- */

void ref_larfg(int m, double *alpha, double *x, double *tau)
{
    double xnorm = 0;
    for (int i = 0; i < m - 1; ++i)
        xnorm = std::hypot(xnorm, x[i]);
    if (xnorm == 0.0) {
        *tau = 0;
        return;
    }
    double beta = -std::copysign(std::hypot(*alpha, xnorm), *alpha);
    *tau = (beta - *alpha) / beta;
    double scal = 1.0 / (*alpha - beta);
    for (int i = 0; i < m - 1; ++i)
        x[i] *= scal;
    *alpha = beta;
}

void ref_geqr2(int m, int n, double *A, int lda, double *tau)
{
    int k = std::min(m, n);
    for (int kk = 0; kk < k; ++kk) {
        ref_larfg(m - kk, &A[kk + kk * lda], &A[(kk + 1) + kk * lda], &tau[kk]);
        for (int j = kk + 1; j < n; ++j) {
            double w = A[kk + j * lda];
            for (int i = kk + 1; i < m; ++i)
                w += A[i + kk * lda] * A[i + j * lda];
            A[kk + j * lda] -= tau[kk] * w;
            for (int i = kk + 1; i < m; ++i)
                A[i + j * lda] -= tau[kk] * A[i + kk * lda] * w;
        }
    }
}

/* --------------------------- compact pool --------------------------- */
/* Column-major m x n matrices; element (i,j) of matrix idx = g*V+v lives at
 * p[g*ldp*n*V + (j*ldp+i)*V + v]; padded slots (idx >= nm) carry the identity. */

struct Pool {
    int m, n, nmat, V, ng;
    std::vector<double> dense;           /* nmat * m*n, column-major per matrix */
    std::vector<double> Href, tref;      /* scalar-reference factor + tau       */
    std::vector<double> packed_pristine; /* packed input, restored each pass    */

    Pool(int m_, int n_, int nmat_, int V_)
        : m(m_), n(n_), nmat(nmat_), V(V_), ng((nmat_ + V_ - 1) / V_)
    {
        const size_t sA = (size_t)m * n, k = std::min(m, n);
        dense.resize((size_t)nmat * sA);
        Href.resize((size_t)nmat * sA);
        tref.resize((size_t)nmat * k);

        std::mt19937_64 rng(2025);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (int idx = 0; idx < nmat; ++idx) {
            double *A = dense.data() + (size_t)idx * sA;
            for (size_t e = 0; e < sA; ++e)
                A[e] = dist(rng);
            for (int i = 0; i < (int)k; ++i)
                A[i + (size_t)i * m] += 2.0 * n; /* diagonal boost: well conditioned */
            /* scalar reference factorization of this matrix */
            double *Hr = Href.data() + (size_t)idx * sA;
            std::copy(A, A + sA, Hr);
            ref_geqr2(m, n, Hr, m, tref.data() + (size_t)idx * k);
        }
        pack();
    }

    size_t packed_a() const { return (size_t)ng * m * n * V; }
    size_t packed_t() const { return (size_t)ng * std::min(m, n) * V; }

    void pack()
    {
        packed_pristine.assign(packed_a(), 0.0);
        double *p = packed_pristine.data();
        const int ldp = m;
        for (int g = 0; g < ng; ++g)
            for (int v = 0; v < V; ++v) {
                int idx = g * V + v;
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < m; ++i) {
                        double val = (idx < nmat)
                                         ? dense[(size_t)idx * m * n + i + (size_t)j * m]
                                         : (i == j ? 1.0 : 0.0);
                        p[(size_t)g * ldp * n * V + ((size_t)j * ldp + i) * V + v] = val;
                    }
            }
    }

    /* worst elementwise (H, tau) error of a packed result vs the scalar
     * reference, scaled by the matrix magnitude. */
    double error(const double *ap, const double *tp) const
    {
        const int ldp = m, k = std::min(m, n);
        double worst = 0;
        for (int g = 0; g < ng; ++g)
            for (int v = 0; v < V; ++v) {
                int idx = g * V + v;
                if (idx >= nmat) continue;
                const double *Hr = Href.data() + (size_t)idx * m * n;
                const double *tr = tref.data() + (size_t)idx * k;
                double num = 0, den = 0;
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < m; ++i) {
                        double got =
                            ap[(size_t)g * ldp * n * V + ((size_t)j * ldp + i) * V + v];
                        double ref = Hr[i + (size_t)j * m];
                        num = std::max(num, std::abs(got - ref));
                        den = std::max(den, std::abs(ref));
                    }
                for (int kk = 0; kk < k; ++kk) {
                    double got = tp[(size_t)g * k * V + (size_t)kk * V + v];
                    num = std::max(num, std::abs(got - tr[kk]));
                }
                worst = std::max(worst, num / std::max(den, 1e-300));
            }
        return worst;
    }
};

/* Best (minimum) wall time over `reps` timed passes, in seconds. */
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

/* The three kernels under test, selected at runtime. */
enum class Kernel { VecTypes, OmpOuter, OmpInner };

/* Factor the whole packed pool with the selected kernel; V chosen at compile
 * time so every template is instantiated for the same width. */
template <int V> void factor(Kernel which, double *ap, double *tp, int m, int n, int nm)
{
    switch (which) {
    case Kernel::VecTypes:
        cqr::detail::geqrf_compact_general<double, V>(false, m, n, ap, m, tp, nm);
        break;
    case Kernel::OmpOuter:
        cqr::detail::omp_simd::geqrf_compact_general_omp<double, V>(false, m, n, ap, m,
                                                                    tp, nm);
        break;
    case Kernel::OmpInner:
        cqr::detail::omp_simd::geqrf_compact_general_omp_inner<double, V>(false, m, n, ap,
                                                                          m, tp, nm);
        break;
    }
}

/* Time one kernel over the pool, gate its result against the scalar reference,
 * and return (GFLOP/s, wall time). */
template <int V>
std::pair<double, double> measure(Kernel which, const char *name, const Pool &P,
                                  std::vector<double> &ap, std::vector<double> &tp,
                                  int reps)
{
    const int m = P.m, n = P.n, nmat = P.nmat;
    auto restore = [&] {
        std::copy(P.packed_pristine.begin(), P.packed_pristine.end(), ap.begin());
    };
    double t = best_time(reps, restore,
                         [&] { factor<V>(which, ap.data(), tp.data(), m, n, nmat); });
    double e = P.error(ap.data(), tp.data());
    const double tol = 1e5 * std::numeric_limits<double>::epsilon() * std::max(1, m);
    if (e > tol) {
        std::printf("kernel '%s' disagrees with scalar reference: %.2e > %.2e\n", name, e,
                    tol);
        std::exit(1);
    }
    return {nmat * geqrf_gflop(m, n) / t, t};
}

/* Run one size for a compile-time width V and print a three-way comparison row.
 * Returns log(omp-inner / vec) for the geometric mean. */
template <int V> double run_size(int n, int nmat, int reps)
{
    const int m = n;
    Pool P(m, n, nmat, V);
    std::vector<double> ap(P.packed_a()), tp(P.packed_t(), 0.0);

    const double g_vec = measure<V>(Kernel::VecTypes, "vec-types", P, ap, tp, reps).first;
    const double g_out = measure<V>(Kernel::OmpOuter, "omp-outer", P, ap, tp, reps).first;
    const double g_in = measure<V>(Kernel::OmpInner, "omp-inner", P, ap, tp, reps).first;

    std::printf("%4d | %10.2f | %10.2f | %10.2f | %7.2fx | %7.2fx\n", n, g_vec, g_out,
                g_in, g_out / g_vec, g_in / g_vec);
    return std::log(g_in / g_vec);
}

/* Dispatch a compile-time V from the runtime width (kept in sync with the C
 * API's supported set). */
template <typename F> void with_width(int V, F &&f)
{
    switch (V) {
    case 2: f(std::integral_constant<int, 2>{}); break;
    case 4: f(std::integral_constant<int, 4>{}); break;
    case 8: f(std::integral_constant<int, 8>{}); break;
    case 16: f(std::integral_constant<int, 16>{}); break;
    default: die("simdlen must be 2, 4, 8, or 16");
    }
}

/* Host-native double interleave width from the widest compiled-in ISA. Mirrors
 * MKL's format->V mapping (AVX-512 -> 8, AVX -> 4, else SSE -> 2 doubles). */
int native_vlen_double()
{
#if defined(__AVX512F__)
    return 8;
#elif defined(__AVX__)
    return 4;
#else
    return 2;
#endif
}

const char *isa_name()
{
#if defined(__AVX512F__)
    return "AVX-512 (V=8)";
#elif defined(__AVX2__)
    return "AVX2 (V=4)";
#elif defined(__AVX__)
    return "AVX (V=4)";
#else
    return "SSE2 (V=2)";
#endif
}

} /* anonymous namespace */

int main(int argc, char **argv)
{
    int nmat = 256, reps = 5, simdlen = 0;
    std::vector<const char *> pos;
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--simdlen=", 10) == 0)
            simdlen = std::atoi(argv[i] + 10);
        else
            pos.push_back(argv[i]);
    }
    if (pos.size() > 0) nmat = std::atoi(pos[0]);
    if (pos.size() > 1) reps = std::atoi(pos[1]);
    if (nmat <= 0 || reps <= 0)
        die("usage: bench_geqrf_omp_simd [--simdlen=V] [nmat>0] [reps>0]");
    if (simdlen == 0) simdlen = native_vlen_double();

    std::printf(
        "compact QR: GNU vector-types vs OpenMP-SIMD (#pragma omp simd simdlen(V))\n");
    std::printf("built for %s | simdlen=%d | matrices=%d | reps=%d | single-threaded\n",
                isa_name(), simdlen, nmat, reps);
    std::printf("  vec-types = GNU vector_size | omp-outer = simd on lane loop | "
                "omp-inner = simd innermost\n\n");
    std::printf("   n |  vec GF/s |  omp-outer |  omp-inner | out/vec |  in/vec\n");
    std::printf("-----+-----------+------------+------------+---------+---------\n");

    constexpr std::array<int, 13> sizes = {8,  16, 24, 30,  32,  45, 48,
                                           60, 64, 96, 105, 128, 168};

    double logsum = 0;
    int cnt = 0;
    with_width(simdlen, [&](auto W) {
        constexpr int V = decltype(W)::value;
        for (int n : sizes) {
            logsum += run_size<V>(n, nmat, reps);
            ++cnt;
        }
    });

    std::printf("-----+-----------+------------+------------+---------+---------\n");
    std::printf("GFLOP/s columns; ratios are throughput vs vec-types. "
                "geomean omp-inner/vec-types: %.2fx\n",
                std::exp(logsum / cnt));
    return 0;
}
