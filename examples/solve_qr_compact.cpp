/* solve_qr_compact.cpp
 *
 * Worked example: solving a *batch* of square systems A_v X_v = B_v with the
 * Intel MKL Compact (interleaved) QR pipeline, cross-checked against the naive
 * per-matrix LAPACKE_dgels forward driver.
 *
 * The compact pipeline is the batched analogue of the textbook QR solve, and
 * is exactly the sequence the design document validates (section 7.2):
 *
 *     mkl_dgeqrf_compact      A = Q R                    (factor the batch)
 *     cqr_mkl_dormqr_compact  B <- Q^T B                 (apply Q^T -- the
 *                                                         routine this repo
 *                                                         adds to MKL)
 *     mkl_dtrsm_compact       R X = (Q^T B)              (triangular solve)
 *
 * For a square, full-rank A this recovers X = R^{-1} Q^T B. The naive baseline
 * runs LAPACKE_dgels('N') on each matrix separately (which reduces to the same
 * QR solve when m == n). We confirm both the compact batch and the per-matrix
 * driver recover the known exact solution.
 *
 * Build: needs Intel MKL (the compact API is an MKL extension) plus this
 * repo's cqr_mkl_ormqr_compact; wired up by CMakeLists.txt as the
 * `solve_qr_compact` target.
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#include <mkl.h>
#include <mkl_compact.h>

#include "cqr_mkl_ext.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <random>
#include <vector>
#include <algorithm>

namespace {

/* Deterministic uniform reals in [-1, 1), seeded once for reproducibility. */
std::mt19937_64 rng(42);
double frand()
{
    static std::uniform_real_distribution<double> dist(-1.0, 1.0);
    return dist(rng);
}

/* Report and abort on the spot if cond is false. */
void check(bool cond, const char *what)
{
    if (!cond) { std::printf("FAILED: %s\n", what); std::exit(1); }
}

/* The compact routines report a single scalar status (MKL leaves the compact
 * info reserved; this project writes one value, 0 on success). */
void check_info(MKL_INT info, const char *what)
{
    check(info == 0, what);
}

/* Minimal column-major dense matrix: owns its storage and hands raw pointers
 * (data(), ld()) to BLAS/LAPACK and the compact pack/unpack routines.
 * Leading dimension == row count (contiguous storage, no padding). Indices are
 * assumed to stay in int32 range. */
class Matrix {
public:
    Matrix(int rows, int cols)
        : rows_(rows), cols_(cols), a_(rows * cols) {}

    /* Plain value type (rule of five, all defaulted): std::vector already
     * manages the storage, so no destructor is needed -- but once we spell
     * out the copy operations we also spell out the moves, otherwise declaring
     * the copies would suppress the implicit move members. */
    Matrix(const Matrix &)            = default;
    Matrix(Matrix &&)                 = default;
    Matrix &operator=(const Matrix &) = default;
    Matrix &operator=(Matrix &&)      = default;

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int ld()   const { return rows_; }
    double       *data()       { return a_.data(); }
    const double *data() const { return a_.data(); }
    double &operator()(int i, int j)       { return a_[i + j * rows_]; }
    double  operator()(int i, int j) const { return a_[i + j * rows_]; }

private:
    int rows_, cols_;
    std::vector<double> a_;
};

/* 1-norm of a matrix, ||A||_1 (max column sum). */
double norm1(const Matrix &A)
{
    return LAPACKE_dlange(LAPACK_COL_MAJOR, '1', A.rows(), A.cols(), A.data(), A.ld());
}

/* Relative 1-norm difference ||A - B||_1 / ||B||_1 (A, B same shape). */
double rel_diff(const Matrix &A, const Matrix &B)
{
    Matrix D = A;                                 /* D <- A */
    cblas_daxpy(D.rows() * D.cols(), -1.0, B.data(), 1, D.data(), 1);  /* D <- A - B */
    return norm1(D) / std::max(norm1(B), 1e-300);
}

/* Per-matrix base pointers the compact pack/unpack routines expect, one per
 * matrix in the batch. A single non-const overload covers both directions:
 * double** qualification-converts to the packer's `const double* const*` and to
 * the unpacker's `double* const*`. */
std::vector<double *> base_ptrs(std::vector<Matrix> &batch)
{
    std::vector<double *> p;
    p.reserve(batch.size());
    for (Matrix &M : batch) p.push_back(M.data());
    return p;
}

void batch_solve(int nm, int n, int nrhs)
{
    const MKL_COMPACT_PACK fmt = mkl_get_format_compact();

    /* Known exact solution X(:,j) = j+1, shared across the batch; per matrix
     * B_v = A_v X so the solve must recover X (mirrors the compact suite-2
     * construction in the design doc). */
    Matrix X(n, nrhs);
    for (int j = 0; j < nrhs; ++j)
        for (int i = 0; i < n; ++i) X(i, j) = double(j + 1);

    /* The batch is an array of independently-allocated matrices, not one
     * contiguous block -- that is fine, the compact pack routines take an
     * array of per-matrix base pointers. */
    std::vector<Matrix> A(nm, Matrix(n, n)), B(nm, Matrix(n, nrhs));
    for (int v = 0; v < nm; ++v) {
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) A[v](i, j) = frand();
        for (int i = 0; i < n; ++i) A[v](i, i) += 2.0;     /* tame conditioning */
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, nrhs, n,
                    1.0, A[v].data(), A[v].ld(), X.data(), X.ld(),
                    0.0, B[v].data(), B[v].ld());           /* B_v = A_v X */
    }

    /* ===== Path 1: compact batch pipeline ============================== *
     * geqrf_compact -> dormqr_compact -> dtrsm_compact, all on the         *
     * interleaved buffers ap / taup / bp.                                  */
    const int compact_align = 64;   /* byte alignment for the compact buffers */
    auto ap_buf   = cqr::detail::mkl_alloc_bytes<double>(mkl_dget_size_compact(n, n,    fmt, nm), compact_align);
    auto taup_buf = cqr::detail::mkl_alloc_bytes<double>(mkl_dget_size_compact(n, 1,    fmt, nm), compact_align);
    auto bp_buf   = cqr::detail::mkl_alloc_bytes<double>(mkl_dget_size_compact(n, nrhs, fmt, nm), compact_align);
    double *ap = ap_buf.get(), *taup = taup_buf.get(), *bp = bp_buf.get();

    /* pack the dense batches into compact (interleaved) layout */
    {
        auto Aptr = base_ptrs(A);
        mkl_dgepack_compact(MKL_COL_MAJOR, n, n,    Aptr.data(), n, ap, n, fmt, nm);
        auto Bptr = base_ptrs(B);
        mkl_dgepack_compact(MKL_COL_MAJOR, n, nrhs, Bptr.data(), n, bp, n, fmt, nm);
    }

    MKL_INT info[1];   /* compact status: a single scalar (MKL convention) */

    /* 1. compact QR: ap <- (H, R), taup <- tau. geqrf needs real workspace,
     *    so query the optimal size (lwork = -1) and allocate it. */
    double wq;
    mkl_dgeqrf_compact(MKL_COL_MAJOR, n, n, ap, n, taup, &wq, -1, info, fmt, nm);
    MKL_INT lwork = (MKL_INT)wq;
    std::vector<double> work((size_t)std::max<MKL_INT>(lwork, 1));
    mkl_dgeqrf_compact(MKL_COL_MAJOR, n, n, ap, n, taup, work.data(), lwork, info, fmt, nm);
    check_info(info[0], "mkl_dgeqrf_compact");

    /* 2. apply Q^T to the RHS: bp <- Q^T B   (this repo's extension). The
     *    apply-Q kernel needs no workspace, so its minimum (and optimal)
     *    lwork is 1 -- the same quick-return value reference LAPACK dormqr
     *    reports -- and no query is required. */
    double dummy;
    cqr_mkl_dormqr_compact(MKL_COL_MAJOR, 'L', 'T', n, nrhs, n,
                           ap, n, taup, bp, n, &dummy, 1, info, fmt, nm);
    check_info(info[0], "cqr_mkl_dormqr_compact");

    /* 3. triangular solve: bp <- R^{-1} (Q^T B) = Xhat */
    mkl_dtrsm_compact(MKL_COL_MAJOR, MKL_LEFT, MKL_UPPER, MKL_NOTRANS, MKL_NONUNIT,
                      n, nrhs, 1.0, ap, n, bp, n, fmt, nm);

    std::vector<Matrix> Xc(nm, Matrix(n, nrhs));
    {
        auto Xcptr = base_ptrs(Xc);
        mkl_dgeunpack_compact(MKL_COL_MAJOR, n, nrhs, Xcptr.data(), n, bp, n, fmt, nm);
    }

    /* ap/taup/bp stay live until their RAII owners go out of scope when
     * batch_solve returns; no manual mkl_free needed. */

    /* ===== Path 2: naive per-matrix forward driver LAPACKE_dgels =======
     * Xd starts as a copy of the RHS, which dgels overwrites in place. */
    std::vector<Matrix> Xd = B;
    for (int v = 0; v < nm; ++v) {
        Matrix Acopy = A[v];
        lapack_int info1 = LAPACKE_dgels(LAPACK_COL_MAJOR, 'N', n, n, nrhs,
                                         Acopy.data(), Acopy.ld(), Xd[v].data(), Xd[v].ld());
        check(info1 == 0, "LAPACKE_dgels");
    }

    /* ===== Compare both paths to the known exact solution X. Agreement
     * between them is implied: each is within rtol of X. ================ */
    double fwd_compact = 0, fwd_dgels = 0;
    for (int v = 0; v < nm; ++v) {
        fwd_compact = std::max(fwd_compact, rel_diff(Xc[v], X));
        fwd_dgels   = std::max(fwd_dgels,   rel_diff(Xd[v], X));
    }

    const double eps  = std::numeric_limits<double>::epsilon();
    const double rtol = 100.0 * n * eps;
    bool ok = (fwd_compact <= rtol && fwd_dgels <= rtol);
    std::printf("  nm=%-3d n=%-4d nrhs=%d | compact fwd %.2e  dgels fwd %.2e  "
                "(rtol %.2e) %s\n",
                nm, n, nrhs, fwd_compact, fwd_dgels, rtol, ok ? "OK" : "FAIL");
    check(ok, "compact QR solve accuracy within rtol");
}

} /* anonymous namespace */

int main()
{
    std::printf("Compact batch QR solve: mkl_dgeqrf_compact -> cqr_mkl_dormqr_compact "
                "-> mkl_dtrsm_compact  vs  per-matrix LAPACKE_dgels\n");
    std::printf("(compact format = %d)\n", (int)mkl_get_format_compact());

    batch_solve(8,  32,  5);
    batch_solve(16, 64,  4);
    batch_solve(7,  128, 3);   /* nm not a multiple of the SIMD width: padded last pack */
    batch_solve(4,  256, 1);

    std::printf("\nall checks passed\n");
    return 0;
}
