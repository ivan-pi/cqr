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
 * unchanged (see the gpu/sycl CMake wiring).
 *
 * Mapping -- design doc gpu_batched_compact_design.md, Strategy A:
 *   one work-item == one matrix (compact slot v of group g). The sub-group is
 *   pinned to the interleave width V (reqd_sub_group_size(V)), so the V
 *   work-items of a group are the V lanes of one hardware SIMD vector: their
 *   reads of element (i,j) are contiguous (coalesced on a GPU; one vector load
 *   on a CPU). V is the architecture vector size -- that is the whole point of
 *   the compact layout -- so this is the shipped kernel.
 *
 *   Fallback: reqd_sub_group_size(V) requires V to be a sub-group size the
 *   device actually supports. The compact format also admits widths that are
 *   NOT hardware sub-group sizes (V=2 anywhere; V=4 on Intel GPUs, whose minimum
 *   is 8) -- the portable test suite uses them to exercise the layout. For those
 *   the kernel falls back to a plain range parallel_for (same body, correct,
 *   just not sub-group-pinned). Real GPU deployment uses V in {8,16,32}, which
 *   always take the sub-group path.
 *
 * The per-lane body is the SAME unblocked geqr2 / dorm2r the CPU kernel runs V
 * lanes at a time, including the JB=4 register blocking of trailing / RHS
 * columns (each reflector entry A(i,kk) is loaded once and reused across 4
 * columns). Here the "lane" is the work-item, so the branch-free larfg select
 * collapses to an ordinary scalar branch.
 *
 * This is a correctness-first prototype: USM copy in/out per call, no
 * device-resident pipeline. geqrf handles column- and row-major (LAPACK
 * layout); ormqr follows the C API's side='L', column-major contract. Complex
 * types are out of scope, matching the CPU library.
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <vector>

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

/* Whether the device supports V as a required sub-group size (cached). If so, V
 * is a real hardware vector width and the kernel pins the sub-group to it. */
bool device_has_sub_group_size(sycl::queue &q, int V)
{
    static const std::vector<std::size_t> sizes =
        q.get_device().get_info<sycl::info::device::sub_group_sizes>();
    return std::find(sizes.begin(), sizes.end(), static_cast<std::size_t>(V)) !=
           sizes.end();
}

/* ----------------------------------------------------------------------------
 * Per-slot device bodies (one matrix). V is a compile-time template argument so
 * the *V index scaling folds to a shift and the sub-group size is a constant.
 * -------------------------------------------------------------------------- */

/* geqrf: unblocked geqr2 + scalar larfg, JB=4 trailing-column update.
 * Element (i,j) = As[i*istep + j*jstep] with the strides (scalar units, V
 * folded in) selecting the layout (col-major: istep=V, jstep=ldap*V). */
template <typename T, int V>
void geqrf_slot(int g, int v, int m, int n, int k, T *ap, std::size_t str_a,
                std::size_t istep, std::size_t jstep, T *tp, std::size_t str_t)
{
    T *As = ap + static_cast<std::size_t>(g) * str_a + v;
    T *Ts = tp + static_cast<std::size_t>(g) * str_t + v;
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
        if (tail > T(0)) {
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
            A(i, kk) = A(i, kk) * inv;
        A(kk, kk) = rdiag;

        /* apply H(kk) = I - t v v^T to trailing columns, 4 at a time */
        int j = kk + 1;
        for (; j + 4 <= n; j += 4) {
            T w0 = A(kk, j), w1 = A(kk, j + 1), w2 = A(kk, j + 2), w3 = A(kk, j + 3);
            for (int i = kk + 1; i < m; ++i) {
                const T av = A(i, kk);
                w0 += av * A(i, j);
                w1 += av * A(i, j + 1);
                w2 += av * A(i, j + 2);
                w3 += av * A(i, j + 3);
            }
            A(kk, j) -= t * w0;
            A(kk, j + 1) -= t * w1;
            A(kk, j + 2) -= t * w2;
            A(kk, j + 3) -= t * w3;
            w0 *= t;
            w1 *= t;
            w2 *= t;
            w3 *= t;
            for (int i = kk + 1; i < m; ++i) {
                const T av = A(i, kk);
                A(i, j) -= av * w0;
                A(i, j + 1) -= av * w1;
                A(i, j + 2) -= av * w2;
                A(i, j + 3) -= av * w3;
            }
        }
        for (; j < n; ++j) {
            T w = A(kk, j);
            for (int i = kk + 1; i < m; ++i)
                w += A(i, kk) * A(i, j);
            A(kk, j) -= t * w;
            w *= t;
            for (int i = kk + 1; i < m; ++i)
                A(i, j) -= A(i, kk) * w;
        }
    }
}

/* ormqr: scalar dorm2r, side='L', column-major, JB=4 RHS-column blocking.
 * A(i,j)=As[(j*ldap+i)*V] (reflectors), B(i,j)=Bs[(j*ldbp+i)*V]. */
template <typename T, int V>
void ormqr_slot(int g, int v, bool fwd, int m, int nrhs, int k, const T *ap, int ldap,
                std::size_t str_a, const T *tp, std::size_t str_t, T *bp, int ldbp,
                std::size_t str_b)
{
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
        int j = 0;
        for (; j + 4 <= nrhs; j += 4) { /* 4 RHS columns; A(i,kk) reused 4x */
            T w0 = B(kk, j), w1 = B(kk, j + 1), w2 = B(kk, j + 2), w3 = B(kk, j + 3);
            for (int i = kk + 1; i < m; ++i) {
                const T av = A(i, kk);
                w0 += av * B(i, j);
                w1 += av * B(i, j + 1);
                w2 += av * B(i, j + 2);
                w3 += av * B(i, j + 3);
            }
            B(kk, j) -= t * w0;
            B(kk, j + 1) -= t * w1;
            B(kk, j + 2) -= t * w2;
            B(kk, j + 3) -= t * w3;
            w0 *= t;
            w1 *= t;
            w2 *= t;
            w3 *= t;
            for (int i = kk + 1; i < m; ++i) {
                const T av = A(i, kk);
                B(i, j) -= av * w0;
                B(i, j + 1) -= av * w1;
                B(i, j + 2) -= av * w2;
                B(i, j + 3) -= av * w3;
            }
        }
        for (; j < nrhs; ++j) {
            T w = B(kk, j);
            for (int i = kk + 1; i < m; ++i)
                w += A(i, kk) * B(i, j);
            B(kk, j) -= t * w;
            w *= t;
            for (int i = kk + 1; i < m; ++i)
                B(i, j) -= A(i, kk) * w;
        }
    }
}

/* ----------------------------------------------------------------------------
 * Launches. reqd_sub_group_size(V) when the device supports V (the shipped
 * path); a plain range parallel_for otherwise (V not a hardware sub-group size).
 * -------------------------------------------------------------------------- */

template <typename T, int V>
void geqrf_launch(sycl::queue &q, bool use_sg, bool rowmajor, int m, int n, T *ap_host,
                  int ldap, T *taup_host, int nm)
{
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
    q.memcpy(ap, ap_host, a_count * sizeof(T)).wait();

    if (use_sg)
        q.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(ng) * V),
                                         sycl::range<1>(V)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(V)]] {
                           geqrf_slot<T, V>((int)it.get_group(0), (int)it.get_local_id(0),
                                            m, n, k, ap, str_a, istep, jstep, tp, str_t);
                       })
            .wait();
    else
        q.parallel_for(sycl::range<1>(static_cast<std::size_t>(ng) * V),
                       [=](sycl::id<1> id) {
                           geqrf_slot<T, V>((int)id[0] / V, (int)id[0] % V, m, n, k, ap,
                                            str_a, istep, jstep, tp, str_t);
                       })
            .wait();

    q.memcpy(ap_host, ap, a_count * sizeof(T)).wait();
    q.memcpy(taup_host, tp, t_count * sizeof(T)).wait();
    sycl::free(ap, q);
    sycl::free(tp, q);
}

template <typename T, int V>
void ormqr_launch(sycl::queue &q, bool use_sg, char trans, int m, int nrhs, int k,
                  const T *ap_host, int ldap, const T *taup_host, T *bp_host, int ldbp,
                  int nm)
{
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

    if (use_sg)
        q.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(ng) * V),
                                         sycl::range<1>(V)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(V)]] {
                           ormqr_slot<T, V>((int)it.get_group(0), (int)it.get_local_id(0),
                                            fwd, m, nrhs, k, ap, ldap, str_a, tp, str_t,
                                            bp, ldbp, str_b);
                       })
            .wait();
    else
        q.parallel_for(sycl::range<1>(static_cast<std::size_t>(ng) * V),
                       [=](sycl::id<1> id) {
                           ormqr_slot<T, V>((int)id[0] / V, (int)id[0] % V, fwd, m, nrhs, k,
                                            ap, ldap, str_a, tp, str_t, bp, ldbp, str_b);
                       })
            .wait();

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

template <typename T>
void geqrf_dispatch(bool row, int m, int n, T *ap, int ldap, T *taup, int V, int nm)
{
    sycl::queue &q = cqr_sycl_queue();
    const bool sg = device_has_sub_group_size(q, V);
    switch (V) {
    case 2:  geqrf_launch<T, 2>(q, sg, row, m, n, ap, ldap, taup, nm); break;
    case 4:  geqrf_launch<T, 4>(q, sg, row, m, n, ap, ldap, taup, nm); break;
    case 8:  geqrf_launch<T, 8>(q, sg, row, m, n, ap, ldap, taup, nm); break;
    case 16: geqrf_launch<T, 16>(q, sg, row, m, n, ap, ldap, taup, nm); break;
    }
}

template <typename T>
void ormqr_dispatch(char trans, int m, int nrhs, int k, const T *ap, int ldap,
                    const T *taup, T *bp, int ldbp, int V, int nm)
{
    sycl::queue &q = cqr_sycl_queue();
    const bool sg = device_has_sub_group_size(q, V);
    switch (V) {
    case 2:  ormqr_launch<T, 2>(q, sg, trans, m, nrhs, k, ap, ldap, taup, bp, ldbp, nm); break;
    case 4:  ormqr_launch<T, 4>(q, sg, trans, m, nrhs, k, ap, ldap, taup, bp, ldbp, nm); break;
    case 8:  ormqr_launch<T, 8>(q, sg, trans, m, nrhs, k, ap, ldap, taup, bp, ldbp, nm); break;
    case 16: ormqr_launch<T, 16>(q, sg, trans, m, nrhs, k, ap, ldap, taup, bp, ldbp, nm); break;
    }
}

} /* anonymous namespace */

extern "C" int dgeqrf_compact(char layout, int m, int n, double *ap, int ldap,
                              double *taup, int V, int nm)
{
    bool row;
    const int info = validate_geqrf<double>(layout, m, n, ldap, V, nm, row);
    if (info != 0) return info;
    if (m == 0 || n == 0 || nm == 0) return 0;
    geqrf_dispatch<double>(row, m, n, ap, ldap, taup, V, nm);
    return 0;
}

extern "C" int sgeqrf_compact(char layout, int m, int n, float *ap, int ldap, float *taup,
                              int V, int nm)
{
    bool row;
    const int info = validate_geqrf<float>(layout, m, n, ldap, V, nm, row);
    if (info != 0) return info;
    if (m == 0 || n == 0 || nm == 0) return 0;
    geqrf_dispatch<float>(row, m, n, ap, ldap, taup, V, nm);
    return 0;
}

extern "C" int dormqr_compact(char trans, int m, int nrhs, int k, const double *ap,
                              int ldap, const double *taup, double *bp, int ldbp, int V,
                              int nm)
{
    const int info = validate_ormqr<double>(trans, m, nrhs, k, ldap, ldbp, V, nm);
    if (info != 0) return info;
    if (m == 0 || nrhs == 0 || k == 0 || nm == 0) return 0;
    ormqr_dispatch<double>(trans, m, nrhs, k, ap, ldap, taup, bp, ldbp, V, nm);
    return 0;
}

extern "C" int sormqr_compact(char trans, int m, int nrhs, int k, const float *ap,
                              int ldap, const float *taup, float *bp, int ldbp, int V,
                              int nm)
{
    const int info = validate_ormqr<float>(trans, m, nrhs, k, ldap, ldbp, V, nm);
    if (info != 0) return info;
    if (m == 0 || nrhs == 0 || k == 0 || nm == 0) return 0;
    ormqr_dispatch<float>(trans, m, nrhs, k, ap, ldap, taup, bp, ldbp, V, nm);
    return 0;
}
