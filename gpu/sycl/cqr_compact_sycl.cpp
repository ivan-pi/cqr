/* cqr_compact_sycl.cpp
 *
 * SYCL (Intel oneAPI DPC++) prototype backend for the portable compact QR C API.
 *
 * It re-implements the SAME four entry points the portable CPU library exports
 * (dgeqrf_compact / sgeqrf_compact / dormqr_compact / sormqr_compact, declared
 * in ../../src/cqr_compact.h) with identical signatures and identical
 * LAPACK-style argument validation -- but the numerical work runs as a SYCL
 * kernel on the default device. Because the signatures and the info=-j contract
 * match exactly, the project's existing self-contained test suites link against
 * this object instead of the CPU dispatch and exercise the GPU kernels
 * unchanged (see gpu/sycl/CMake wiring): "the same test suite" in the literal
 * sense for geqrf, and a C-API-routed mirror for ormqr.
 *
 * Mapping -- design doc gpu_batched_compact_design.md, Strategy A:
 *   one work-item == one matrix (compact slot v of group g). Consecutive
 *   work-items own consecutive slots, so with the compact layout their reads of
 *   element (i,j) are contiguous in memory -- coalesced on a GPU. The scalar
 *   per-lane body is the SAME unblocked geqr2 / dorm2r the CPU kernel runs V
 *   lanes at a time; here the "lane" is the work-item, so the branch-free larfg
 *   select collapses to an ordinary scalar branch.
 *
 * This is a correctness-first prototype: a plain range parallel_for (no
 * sub-group tuning, no register blocking, USM copy in/out per call). It is
 * meant to validate the port on any SYCL device -- including the OpenCL CPU
 * device, which needs no GPU -- against the project's numerical gates. Xe
 * performance work (sub-group = V, blocking, device-resident pipeline) is the
 * design doc's Phase 2.
 *
 * Column-major is the tuned path exercised by the tests; geqrf also accepts
 * row-major (LAPACK layout 'R') via an element-stride swap, matching the CPU
 * geqrf_compact_general. ormqr follows the C API's side='L', column-major
 * contract.
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdio>

#include "cqr_compact.h" /* the four C signatures this file implements */

namespace {

/* One shared queue on the default device (the OpenCL CPU device when no GPU is
 * present). Announced once so the test log records where the kernels ran. */
sycl::queue &cqr_sycl_queue()
{
    static sycl::queue q{sycl::default_selector_v};
    static bool announced = [&] {
        std::fprintf(stderr, "[cqr SYCL backend] device: %s\n",
                     q.get_device().get_info<sycl::info::device::name>().c_str());
        return true;
    }();
    (void)announced;
    return q;
}

/* ----------------------------------------------------------------------------
 * geqrf: one work-item factors one matrix (unblocked geqr2 + scalar larfg).
 *
 * Element (i,j) of this slot is base[i*istep + j*jstep] with the strides (in
 * scalar units, V folded in) selecting the layout:
 *   column-major: istep = V,       jstep = ldap*V
 *   row-major:    istep = ldap*V,   jstep = V
 * Group strides: str_a scalars per V-matrix group of A, str_t of tau.
 * -------------------------------------------------------------------------- */
template <typename T>
void geqrf_launch(bool rowmajor, int m, int n, T *ap_host, int ldap, T *taup_host, int V,
                  int nm)
{
    sycl::queue &q = cqr_sycl_queue();
    const int k = (m < n) ? m : n;
    const int ng = (nm + V - 1) / V;

    const std::size_t str_a =
        static_cast<std::size_t>(ldap) * (rowmajor ? m : n) * V;
    const std::size_t str_t = static_cast<std::size_t>(k) * V;
    const std::size_t a_count = static_cast<std::size_t>(ng) * str_a;
    const std::size_t t_count = static_cast<std::size_t>(ng) * str_t;

    const std::size_t istep = (rowmajor ? static_cast<std::size_t>(ldap) : 1) * V;
    const std::size_t jstep = (rowmajor ? 1 : static_cast<std::size_t>(ldap)) * V;

    T *ap = sycl::malloc_device<T>(a_count, q);
    T *tp = sycl::malloc_device<T>(t_count, q);
    q.memcpy(ap, ap_host, a_count * sizeof(T)).wait(); /* A is the input */

    q.parallel_for(sycl::range<1>(static_cast<std::size_t>(ng) * V),
                   [=](sycl::id<1> idx) {
        const int wi = static_cast<int>(idx[0]);
        const int g = wi / V;
        const int v = wi % V;
        T *As = ap + static_cast<std::size_t>(g) * str_a + v; /* slot (g,v) of A */
        T *Ts = tp + static_cast<std::size_t>(g) * str_t + v; /* slot (g,v) of tau */
        auto A = [=](int i, int j) -> T & {
            return As[static_cast<std::size_t>(i) * istep + static_cast<std::size_t>(j) * jstep];
        };
        for (int kk = 0; kk < k; ++kk) {
            /* build reflector H(kk) from column kk, rows kk..m-1 (scalar larfg) */
            const T x0 = A(kk, kk);
            T tail = T(0);
            for (int i = kk + 1; i < m; ++i) {
                const T a = A(i, kk);
                tail += a * a;
            }
            const T norm = sycl::sqrt(x0 * x0 + tail);
            const T beta = (x0 >= T(0)) ? -norm : norm; /* -copysign(norm, x0) */
            T t, inv, rdiag;
            if (tail > T(0)) {           /* is there a sub-diagonal to zero? */
                t = (beta - x0) / beta;
                inv = T(1) / (x0 - beta);
                rdiag = beta;
            } else {
                t = T(0);
                inv = T(0);
                rdiag = x0;
            }
            Ts[static_cast<std::size_t>(kk) * V] = t;
            for (int i = kk + 1; i < m; ++i)
                A(i, kk) = A(i, kk) * inv; /* reflector body */
            A(kk, kk) = rdiag;            /* R diagonal */

            /* apply H(kk) = I - t v v^T (v(kk)=1 implicit) to trailing columns */
            for (int j = kk + 1; j < n; ++j) {
                T w = A(kk, j);
                for (int i = kk + 1; i < m; ++i)
                    w += A(i, kk) * A(i, j);
                A(kk, j) -= t * w;
                w *= t;
                for (int i = kk + 1; i < m; ++i)
                    A(i, j) -= A(i, kk) * w;
            }
        }
    }).wait();

    q.memcpy(ap_host, ap, a_count * sizeof(T)).wait();
    q.memcpy(taup_host, tp, t_count * sizeof(T)).wait();
    sycl::free(ap, q);
    sycl::free(tp, q);
}

/* ----------------------------------------------------------------------------
 * ormqr: one work-item applies op(Q) to one matrix's RHS block (scalar dorm2r,
 * side='L', column-major -- the portable C API's contract).
 *   A(i,j) = As[(j*ldap + i)*V]   reflectors, (ldap, k) per matrix
 *   tau(kk)= Ts[kk*V]
 *   B(i,j) = Bs[(j*ldbp + i)*V]   RHS, (ldbp, nrhs) per matrix
 * Direction: Q^T applies ascending (fwd), Q descending -- the LAPACK dorm2r
 * table for side='L'.
 * -------------------------------------------------------------------------- */
template <typename T>
void ormqr_launch(char trans, int m, int nrhs, int k, const T *ap_host, int ldap,
                  const T *taup_host, T *bp_host, int ldbp, int V, int nm)
{
    sycl::queue &q = cqr_sycl_queue();
    const int ng = (nm + V - 1) / V;
    const std::size_t str_a = static_cast<std::size_t>(ldap) * k * V;
    const std::size_t str_t = static_cast<std::size_t>(k) * V;
    const std::size_t str_b = static_cast<std::size_t>(ldbp) * nrhs * V;
    const std::size_t a_count = static_cast<std::size_t>(ng) * str_a;
    const std::size_t t_count = static_cast<std::size_t>(ng) * str_t;
    const std::size_t b_count = static_cast<std::size_t>(ng) * str_b;
    const bool fwd = (trans == 'T' || trans == 't');

    T *ap = sycl::malloc_device<T>(a_count, q);
    T *tp = sycl::malloc_device<T>(t_count, q);
    T *bp = sycl::malloc_device<T>(b_count, q);
    q.memcpy(ap, ap_host, a_count * sizeof(T)).wait();
    q.memcpy(tp, taup_host, t_count * sizeof(T)).wait();
    q.memcpy(bp, bp_host, b_count * sizeof(T)).wait();

    q.parallel_for(sycl::range<1>(static_cast<std::size_t>(ng) * V),
                   [=](sycl::id<1> idx) {
        const int wi = static_cast<int>(idx[0]);
        const int g = wi / V;
        const int v = wi % V;
        const T *As = ap + static_cast<std::size_t>(g) * str_a + v;
        const T *Ts = tp + static_cast<std::size_t>(g) * str_t + v;
        T *Bs = bp + static_cast<std::size_t>(g) * str_b + v;
        auto A = [=](int i, int j) -> const T & {
            return As[(static_cast<std::size_t>(j) * ldap + i) * V];
        };
        auto B = [=](int i, int j) -> T & {
            return Bs[(static_cast<std::size_t>(j) * ldbp + i) * V];
        };
        for (int s = 0; s < k; ++s) {
            const int kk = fwd ? s : k - 1 - s;
            const T t = Ts[static_cast<std::size_t>(kk) * V];
            for (int j = 0; j < nrhs; ++j) {
                T w = B(kk, j);
                for (int i = kk + 1; i < m; ++i)
                    w += A(i, kk) * B(i, j);
                B(kk, j) -= t * w;
                w *= t;
                for (int i = kk + 1; i < m; ++i)
                    B(i, j) -= A(i, kk) * w;
            }
        }
    }).wait();

    q.memcpy(bp_host, bp, b_count * sizeof(T)).wait();
    sycl::free(ap, q);
    sycl::free(tp, q);
    sycl::free(bp, q);
}

/* Argument validation, byte-for-byte the CPU dispatch contract (info = -j). */
template <typename T>
int validate_geqrf(char layout, int m, int n, int ldap, int V, int nm, bool &row)
{
    const bool col = (layout == 'C' || layout == 'c');
    row = (layout == 'R' || layout == 'r');
    const int ldmin = row ? (n < 1 ? 1 : n) : (m < 1 ? 1 : m);
    if (!col && !row) return -1;
    if (m < 0) return -2;
    if (n < 0) return -3;
    if (ldap < ldmin) return -5;
    if (V != 2 && V != 4 && V != 8 && V != 16) return -7;
    if (nm < 0) return -8;
    return 0;
}

template <typename T>
int validate_ormqr(char trans, int m, int nrhs, int k, int ldap, int ldbp, int V, int nm)
{
    const bool trans_ok =
        (trans == 'T' || trans == 't' || trans == 'N' || trans == 'n');
    const int ldmin = (m < 1 ? 1 : m);
    if (!trans_ok) return -1;
    if (m < 0) return -2;
    if (nrhs < 0) return -3;
    if (k < 0 || k > m) return -4;
    if (ldap < ldmin) return -6;
    if (ldbp < ldmin) return -9;
    if (V != 2 && V != 4 && V != 8 && V != 16) return -10;
    if (nm < 0) return -11;
    return 0;
}

} /* anonymous namespace */

extern "C" int dgeqrf_compact(char layout, int m, int n, double *ap, int ldap,
                              double *taup, int V, int nm)
{
    bool row;
    const int info = validate_geqrf<double>(layout, m, n, ldap, V, nm, row);
    if (info != 0) return info;
    if (m == 0 || n == 0 || nm == 0) return 0;
    geqrf_launch<double>(row, m, n, ap, ldap, taup, V, nm);
    return 0;
}

extern "C" int sgeqrf_compact(char layout, int m, int n, float *ap, int ldap, float *taup,
                              int V, int nm)
{
    bool row;
    const int info = validate_geqrf<float>(layout, m, n, ldap, V, nm, row);
    if (info != 0) return info;
    if (m == 0 || n == 0 || nm == 0) return 0;
    geqrf_launch<float>(row, m, n, ap, ldap, taup, V, nm);
    return 0;
}

extern "C" int dormqr_compact(char trans, int m, int nrhs, int k, const double *ap,
                              int ldap, const double *taup, double *bp, int ldbp, int V,
                              int nm)
{
    const int info = validate_ormqr<double>(trans, m, nrhs, k, ldap, ldbp, V, nm);
    if (info != 0) return info;
    if (m == 0 || nrhs == 0 || k == 0 || nm == 0) return 0;
    ormqr_launch<double>(trans, m, nrhs, k, ap, ldap, taup, bp, ldbp, V, nm);
    return 0;
}

extern "C" int sormqr_compact(char trans, int m, int nrhs, int k, const float *ap,
                              int ldap, const float *taup, float *bp, int ldbp, int V,
                              int nm)
{
    const int info = validate_ormqr<float>(trans, m, nrhs, k, ldap, ldbp, V, nm);
    if (info != 0) return info;
    if (m == 0 || nrhs == 0 || k == 0 || nm == 0) return 0;
    ormqr_launch<float>(trans, m, nrhs, k, ap, ldap, taup, bp, ldbp, V, nm);
    return 0;
}
