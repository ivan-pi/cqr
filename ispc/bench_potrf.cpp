/* bench_potrf.cpp -- ISPC batched Cholesky vs mkl_dpotrf_compact.
 *
 * A random SPD batch A = M^T M + n I is packed in MKL Compact format (V from the
 * gang, e.g. avx512skx-x8 -> V=8) and factored by cqr_ispc_dpotrf_compact and by
 * MKL's mkl_dpotrf_compact; reports GFLOP/s for each and the ISPC speedup over
 * MKL, per size, sequential. Both uplo: lower is the contiguous ISPC path, upper
 * the strided one. Each factor is spot-checked by its reconstruction residual
 * (||R^T R - A|| / ||A||). Indicative only -- see README measurement caveats.
 *
 * Usage: bench_potrf [nmat] [reps]   (defaults 1024, 5)
 * Assisted-by: Claude:claude-opus-4.8
 */
#include "bench_common.hpp"
#include "cqr_ispc.h"

using namespace bench;

/* nmat dense column-major SPD matrices A = M^T M + n I; the dense originals are
 * kept so a factor can be checked by reconstruction after the timed runs. */
struct SPDPool {
    int n, nmat;
    std::vector<double> a;
    SPDPool(int n_, int nmat_) : n(n_), nmat(nmat_), a((std::size_t)nmat_ * n_ * n_)
    {
        std::mt19937_64 rng(2024);
        std::uniform_real_distribution<double> d(-1, 1);
        std::vector<double> M((std::size_t)n * n);
        for (int v = 0; v < nmat; ++v) {
            for (double &x : M)
                x = d(rng);
            double *A = a.data() + (std::size_t)v * n * n;
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) {
                    double s = 0;
                    for (int l = 0; l < n; ++l)
                        s += M[l + (std::size_t)i * n] * M[l + (std::size_t)j * n];
                    A[i + (std::size_t)j * n] = s + (i == j ? (double)n : 0.0);
                }
        }
    }
    void pack(MKL_COMPACT_PACK fmt, double *ap) const
    {
        std::vector<double *> Ap(nmat);
        for (int v = 0; v < nmat; ++v)
            Ap[v] = const_cast<double *>(a.data()) + (std::size_t)v * n * n;
        mkl_dgepack_compact(MKL_COL_MAJOR, n, n, Ap.data(), n, ap, n, fmt, nmat);
    }
};

/* Relative reconstruction residual of a compact factor: max over the batch of
 * ||R^T R - A||_max / ||A||_max, R the named triangle (L for lower, U for upper). */
static double recon_err(const SPDPool &P, MKL_COMPACT_PACK fmt, double *ap, bool upper)
{
    const int n = P.n, nmat = P.nmat;
    std::vector<double> H((std::size_t)nmat * n * n);
    std::vector<double *> Hp(nmat);
    for (int v = 0; v < nmat; ++v)
        Hp[v] = H.data() + (std::size_t)v * n * n;
    mkl_dgeunpack_compact(MKL_COL_MAJOR, n, n, Hp.data(), n, ap, n, fmt, nmat);
    double e = 0, an = 0;
    for (int v = 0; v < nmat; ++v) {
        const double *A = P.a.data() + (std::size_t)v * n * n;
        const double *F = H.data() + (std::size_t)v * n * n;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                double s = 0;
                const int lmax = std::min(i, j);
                for (int l = 0; l <= lmax; ++l)
                    s += upper ? F[l + (std::size_t)i * n] * F[l + (std::size_t)j * n]
                               : F[i + (std::size_t)l * n] * F[j + (std::size_t)l * n];
                e = std::max(e, std::fabs(s - A[i + (std::size_t)j * n]));
                an = std::max(an, std::fabs(A[i + (std::size_t)j * n]));
            }
    }
    return e / std::max(an, 1e-300);
}

int main(int argc, char **argv)
{
    const int nmat = argc > 1 ? std::atoi(argv[1]) : 1024;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 5;
    if (nmat <= 0 || reps <= 0) die("usage: bench_potrf [nmat>0] [reps>0]");

    mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL); /* sequential, no threading */
    mkl_set_num_threads(1);
    LAPACKE_set_nancheck(0);
    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();
    const int V = (fmt == MKL_COMPACT_SSE ? 16 : fmt == MKL_COMPACT_AVX ? 32 : 64) / 8;
    if (cqr_ispc_gang_width() != V) { /* the kernel needs gang width == MKL's V */
        std::printf("ISPC gang width %d != MKL compact V %d; rebuild the ISPC target "
                    "with a width-%d gang (e.g. avx512skx-x%d).\n",
                    cqr_ispc_gang_width(), V, V, V);
        return 77;
    }

    const int sizes[] = {10, 20, 30, 40, 50, 60, 80, 100, 120, 150};
    const int ns = (int)(sizeof(sizes) / sizeof(*sizes));

    std::printf(
        "ISPC potrf vs mkl_dpotrf_compact  nmat=%d reps=%d V=%d sequential\n"
        "(GFLOP/s; ISPC speedup vs MKL. Indicative only -- see README caveats.)\n",
        nmat, reps, V);

    for (int up = 0; up <= 1; ++up) {
        const bool upper = up;
        std::printf("\n[potrf %s]\n   n |    ISPC     MKL | ISPC/MKL | resid ISPC / MKL\n"
                    "-----+-----------------+----------+------------------\n",
                    upper ? "U (strided path)" : "L (contiguous path)");
        double gi[16], gm[16];
        for (int si = 0; si < ns; ++si) {
            const int n = sizes[si];
            SPDPool P(n, nmat);
            const std::size_t szA = mkl_dget_size_compact(n, n, fmt, nmat);
            double *ap0 = aligned(szA); /* pristine packed SPD A */
            P.pack(fmt, ap0);
            double *ai = aligned(szA), *am = aligned(szA);
            auto RI = [&] { std::memcpy(ai, ap0, szA * 8); };
            auto RM = [&] { std::memcpy(am, ap0, szA * 8); };
            MKL_INT info[1];

            const double ti = best_time(reps, RI, [&] {
                cqr_ispc_dpotrf_compact(0, upper ? 1 : 0, n, ai, n, nmat);
            });
            const double tm = best_time(reps, RM, [&] {
                mkl_dpotrf_compact(MKL_COL_MAJOR, upper ? MKL_UPPER : MKL_LOWER, n, am, n,
                                   info, fmt, nmat);
            });

            const double ri = recon_err(P, fmt, ai, upper);
            const double rm = recon_err(P, fmt, am, upper);
            gi[si] = gflops(nmat, ti, fl_potrf(n));
            gm[si] = gflops(nmat, tm, fl_potrf(n));
            std::printf("%4d | %7.2f %7.2f | %7.2fx | %.1e / %.1e\n", n, gi[si], gm[si],
                        gi[si] / gm[si], ri, rm);
            if (std::max(ri, rm) > 100.0 * n * EPS)
                std::printf("  WARN n=%d residual ISPC %.1e MKL %.1e\n", n, ri, rm);
            for (double *p : {ap0, ai, am})
                std::free(p);
        }
        std::printf("     geomean ISPC/MKL: %.2fx\n",
                    geomean(ns, [&](int i) { return gi[i] / gm[i]; }));
    }
    return 0;
}
