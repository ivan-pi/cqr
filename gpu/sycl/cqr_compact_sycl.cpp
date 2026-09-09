/* cqr_compact_sycl.cpp
 *
 * Thin REFERENCE LAUNCHERS over the SYCL device primitives in
 * cqr_compact_sycl.hpp. The building blocks are the primitives (one work-item
 * per matrix, fused by the caller inside a single kernel -- see the header and
 * example_fused_qr_solve.cpp). These launchers are a convenience batched API
 * for callers that do NOT need fusion, and the vehicle that lets the project's
 * existing self-contained test suites validate the primitives unchanged: they
 * implement the SAME four portable C entry points (dgeqrf_compact /
 * sgeqrf_compact / dormqr_compact / sormqr_compact, ../../src/cqr_compact.h)
 * with identical signatures and identical LAPACK-style info=-j validation, then
 * do the one thing a fused caller would do themselves -- own the batch
 * parallel_for -- around the primitive.
 *
 * Each launcher: USM alloc + copy in, one parallel_for calling the primitive
 * (nd_range with reqd_sub_group_size(V) when the device supports V as a
 * sub-group size -- the shipped path, V being the architecture vector width; a
 * plain range parallel_for otherwise, for widths that are not a hardware
 * sub-group size such as V=2, which the portable test suite exercises but real
 * GPU deployment never uses), copy out + free.
 *
 * Assisted-by: Claude:claude-opus-4.8
 */

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <vector>

#include "cqr_compact.h"      /* the four C signatures this file implements */
#include "cqr_compact_sycl.hpp" /* the primitives */

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

/* Whether V is a required sub-group size the device supports (cached). */
bool device_has_sub_group_size(sycl::queue &q, int V)
{
    static const std::vector<std::size_t> sizes =
        q.get_device().get_info<sycl::info::device::sub_group_sizes>();
    return std::find(sizes.begin(), sizes.end(), static_cast<std::size_t>(V)) !=
           sizes.end();
}

template <typename T, int V>
void geqrf_launch(sycl::queue &q, bool use_sg, bool rowmajor, int m, int n, T *ap_host,
                  int ldap, T *taup_host, int nm)
{
    const int k = (m < n) ? m : n;
    const int ng = (nm + V - 1) / V;
    const std::size_t a_count =
        static_cast<std::size_t>(ng) * ldap * (rowmajor ? m : n) * V;
    const std::size_t t_count = static_cast<std::size_t>(ng) * k * V;

    T *ap = sycl::malloc_device<T>(a_count, q);
    T *tp = sycl::malloc_device<T>(t_count, q);
    q.memcpy(ap, ap_host, a_count * sizeof(T)).wait();

    const auto range = sycl::range<1>(static_cast<std::size_t>(ng) * V);
    if (use_sg)
        q.parallel_for(sycl::nd_range<1>(range, sycl::range<1>(V)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(V)]] {
#ifdef CQR_SYCL_EXPLICIT_SG
                           cqr::gpu::geqrf_slot_sg<T, V>(it.get_sub_group(),
                                                         (int)it.get_group(0), ap, ldap,
                                                         tp, m, n, rowmajor);
#else
                           cqr::gpu::geqrf_slot<T, V>((int)it.get_group(0),
                                                      (int)it.get_local_id(0), ap, ldap,
                                                      tp, m, n, rowmajor);
#endif
                       })
            .wait();
    else
        q.parallel_for(range, [=](sycl::id<1> id) {
             cqr::gpu::geqrf_slot<T, V>((int)id[0] / V, (int)id[0] % V, ap, ldap, tp, m, n,
                                       rowmajor);
         }).wait();

    q.memcpy(ap_host, ap, a_count * sizeof(T)).wait();
    q.memcpy(taup_host, tp, t_count * sizeof(T)).wait();
    sycl::free(ap, q);
    sycl::free(tp, q);
}

template <typename T, int V>
void ormqr_launch(sycl::queue &q, bool use_sg, bool trans, int m, int nrhs, int k,
                  const T *ap_host, int ldap, const T *taup_host, T *bp_host, int ldbp,
                  int nm)
{
    const int ng = (nm + V - 1) / V;
    const std::size_t a_count = static_cast<std::size_t>(ng) * ldap * k * V;
    const std::size_t t_count = static_cast<std::size_t>(ng) * k * V;
    const std::size_t b_count = static_cast<std::size_t>(ng) * ldbp * nrhs * V;

    T *ap = sycl::malloc_device<T>(a_count, q);
    T *tp = sycl::malloc_device<T>(t_count, q);
    T *bp = sycl::malloc_device<T>(b_count, q);
    q.memcpy(ap, ap_host, a_count * sizeof(T)).wait();
    q.memcpy(tp, taup_host, t_count * sizeof(T)).wait();
    q.memcpy(bp, bp_host, b_count * sizeof(T)).wait();

    const auto range = sycl::range<1>(static_cast<std::size_t>(ng) * V);
    if (use_sg)
        q.parallel_for(sycl::nd_range<1>(range, sycl::range<1>(V)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(V)]] {
#ifdef CQR_SYCL_EXPLICIT_SG
                           cqr::gpu::ormqr_slot_sg<T, V>(it.get_sub_group(),
                                                         (int)it.get_group(0), ap, ldap,
                                                         tp, bp, ldbp, m, nrhs, k, trans);
#else
                           cqr::gpu::ormqr_slot<T, V>((int)it.get_group(0),
                                                      (int)it.get_local_id(0), ap, ldap,
                                                      tp, bp, ldbp, m, nrhs, k, trans);
#endif
                       })
            .wait();
    else
        q.parallel_for(range, [=](sycl::id<1> id) {
             cqr::gpu::ormqr_slot<T, V>((int)id[0] / V, (int)id[0] % V, ap, ldap, tp, bp,
                                       ldbp, m, nrhs, k, trans);
         }).wait();

    q.memcpy(bp_host, bp, b_count * sizeof(T)).wait();
    sycl::free(ap, q);
    sycl::free(tp, q);
    sycl::free(bp, q);
}

/* Argument validation, byte-for-byte the CPU dispatch contract (info = -j).
 * Not templated on the element type: the checks are all on shape and layout,
 * so a type parameter would only force every call site to pick one for
 * nothing. */
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
void ormqr_dispatch(bool trans, int m, int nrhs, int k, const T *ap, int ldap,
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
    const int info = validate_geqrf(layout, m, n, ldap, V, nm, row);
    if (info != 0) return info;
    if (m == 0 || n == 0 || nm == 0) return 0;
    geqrf_dispatch<double>(row, m, n, ap, ldap, taup, V, nm);
    return 0;
}

extern "C" int sgeqrf_compact(char layout, int m, int n, float *ap, int ldap, float *taup,
                              int V, int nm)
{
    bool row;
    const int info = validate_geqrf(layout, m, n, ldap, V, nm, row);
    if (info != 0) return info;
    if (m == 0 || n == 0 || nm == 0) return 0;
    geqrf_dispatch<float>(row, m, n, ap, ldap, taup, V, nm);
    return 0;
}

extern "C" int dormqr_compact(char trans, int m, int nrhs, int k, const double *ap,
                              int ldap, const double *taup, double *bp, int ldbp, int V,
                              int nm)
{
    const int info = validate_ormqr(trans, m, nrhs, k, ldap, ldbp, V, nm);
    if (info != 0) return info;
    if (m == 0 || nrhs == 0 || k == 0 || nm == 0) return 0;
    const bool tr = (trans == 'T' || trans == 't');
    ormqr_dispatch<double>(tr, m, nrhs, k, ap, ldap, taup, bp, ldbp, V, nm);
    return 0;
}

extern "C" int sormqr_compact(char trans, int m, int nrhs, int k, const float *ap,
                              int ldap, const float *taup, float *bp, int ldbp, int V,
                              int nm)
{
    const int info = validate_ormqr(trans, m, nrhs, k, ldap, ldbp, V, nm);
    if (info != 0) return info;
    if (m == 0 || nrhs == 0 || k == 0 || nm == 0) return 0;
    const bool tr = (trans == 'T' || trans == 't');
    ormqr_dispatch<float>(tr, m, nrhs, k, ap, ldap, taup, bp, ldbp, V, nm);
    return 0;
}
