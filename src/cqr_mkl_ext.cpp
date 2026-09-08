/* cqr_mkl_ext.cpp
 *
 * The MKL-style API (cqr_mkl_ext.h): thin C-linkage adapters that unwrap the
 * MKL enums to the kernels' flags and MKL_COMPACT_PACK to the interleave width
 * V (vlen_for_format + for_vlen), then forward to the templated kernels
 * instantiated on MKL_INT so ILP64 dimensions are not narrowed.
 *
 * Following the MKL Compact convention the routines validate NO arguments:
 * compact routines skip error checking for vectorization and make the caller
 * responsible for consistent parameters. MKL leaves the compact `info` reserved;
 * here it is a single scalar status, 0 on success, or -1 for an unrecognized
 * format (the one failure dispatch can detect). ?geqrf and ?ormqr answer the
 * lwork = -1 workspace query with 1: the kernels need no scratch. ?trsm has no
 * info and no workspace, like the BLAS ?trsm it batches. ?gels does use work --
 * as the tau scratch of its factorization, one slot per group -- so its query
 * returns the size of a compact tau buffer for the batch.
 *
 * Assisted-by: Claude:claude-fable-5 Claude:claude-opus-4.8
 */

#include "cqr_mkl_ext.h"
#include "cqr_geqrf_compact.hpp"
#include "cqr_ormqr_compact.hpp"
#include "cqr_potrf_compact.hpp"
#include "cqr_trsm_compact.hpp"
#include "cqr_gels_compact.hpp"

namespace {

using cqr::detail::for_vlen;
using cqr::detail::vlen_for_format;

/* Run f<V> for the interleave width of `format`; the scalar status. */
template <typename T, typename F> MKL_INT run_format(MKL_COMPACT_PACK format, F &&f)
{
    return for_vlen(vlen_for_format<T>(format), f) ? 0 : -1;
}

inline void set_info(MKL_INT *info, MKL_INT status)
{
    if (info) *info = status;
}

/* Workspace query (lwork = -1): the kernels need no scratch, so the optimal and
 * minimum lwork is 1. Returns true when the call was a query. */
template <typename T> bool workspace_query(T *work, MKL_INT lwork, MKL_INT *info)
{
    if (lwork != -1) return false;
    if (work) work[0] = T(1);
    set_info(info, 0);
    return true;
}

template <typename T>
void geqrf(MKL_LAYOUT layout, MKL_INT m, MKL_INT n, T *ap, MKL_INT ldap, T *taup, T *work,
           MKL_INT lwork, MKL_INT *info, MKL_COMPACT_PACK format, MKL_INT nm)
{
    if (workspace_query(work, lwork, info)) return;
    if (m == 0 || n == 0 || nm == 0) return set_info(info, 0);

    const bool rowmajor = (layout == MKL_ROW_MAJOR);
    set_info(info, run_format<T>(format, [&](auto v) {
                 cqr::detail::geqrf_compact<T, decltype(v)::value, MKL_INT>(
                     rowmajor, m, n, ap, ldap, taup, nm);
             }));
}

template <typename T>
void ormqr(MKL_LAYOUT layout, char side, char trans, MKL_INT m, MKL_INT n, MKL_INT k,
           const T *ap, MKL_INT ldap, const T *taup, T *cp, MKL_INT ldcp, T *work,
           MKL_INT lwork, MKL_INT *info, MKL_COMPACT_PACK format, MKL_INT nm)
{
    if (workspace_query(work, lwork, info)) return;
    if (m == 0 || n == 0 || k == 0 || nm == 0) return set_info(info, 0);

    const bool rowmajor = (layout == MKL_ROW_MAJOR);
    const bool left = (side == 'L' || side == 'l');
    set_info(info, run_format<T>(format, [&](auto v) {
                 cqr::detail::ormqr_compact<T, decltype(v)::value, MKL_INT>(
                     left, rowmajor, trans, m, n, k, ap, ldap, taup, cp, ldcp, nm);
             }));
}

template <typename T>
void potrf(MKL_LAYOUT layout, MKL_UPLO uplo, MKL_INT n, T *ap, MKL_INT ldap,
           MKL_INT *info, MKL_COMPACT_PACK format, MKL_INT nm)
{
    if (n == 0 || nm == 0) return set_info(info, 0);

    const bool rowmajor = (layout == MKL_ROW_MAJOR);
    const bool upper = (uplo == MKL_UPPER);
    set_info(info, run_format<T>(format, [&](auto v) {
                 cqr::detail::potrf_compact<T, decltype(v)::value, MKL_INT>(
                     rowmajor, upper, n, ap, ldap, nm);
             }));
}

template <typename T>
void trsm(MKL_LAYOUT layout, MKL_SIDE side, MKL_UPLO uplo, MKL_TRANSPOSE transa,
          MKL_DIAG diag, MKL_INT m, MKL_INT n, T alpha, const T *ap, MKL_INT ldap, T *bp,
          MKL_INT ldbp, MKL_COMPACT_PACK format, MKL_INT nm)
{
    /* alpha == 0 is NOT empty -- it means B := 0 -- so it flows to the kernel. */
    if (m == 0 || n == 0 || nm == 0) return;

    const bool rowmajor = (layout == MKL_ROW_MAJOR);
    const bool left = (side == MKL_LEFT);
    const bool upper = (uplo == MKL_UPPER);
    const bool tran = (transa != MKL_NOTRANS); /* MKL_CONJTRANS folds to T (real) */
    const bool unit = (diag == MKL_UNIT);
    /* An unrecognized format selects no kernel; with no info that is a no-op. */
    run_format<T>(format, [&](auto v) {
        cqr::detail::trsm_compact<T, decltype(v)::value, MKL_INT>(
            left, upper, rowmajor, tran, unit, m, n, alpha, ap, ldap, bp, ldbp, nm);
    });
}

template <typename T>
void gels(MKL_LAYOUT layout, char trans, MKL_INT m, MKL_INT n, MKL_INT nrhs, T *ap,
          MKL_INT ldap, T *bp, MKL_INT ldbp, T *work, MKL_INT lwork, MKL_INT *info,
          MKL_COMPACT_PACK format, MKL_INT nm)
{
    if (lwork == -1) { /* workspace query: the tau scratch, one slot per group */
        const int V = vlen_for_format<T>(format);
        if (V == 0) return set_info(info, -1);
        if (work) work[0] = T(cqr::detail::gels_lwork(m, n, nm, V));
        return set_info(info, 0);
    }
    if (nrhs == 0 || nm == 0) return set_info(info, 0);

    const bool rowmajor = (layout == MKL_ROW_MAJOR);
    set_info(info, run_format<T>(format, [&](auto v) {
                 cqr::detail::gels_compact<T, decltype(v)::value, MKL_INT>(
                     rowmajor, trans, m, n, nrhs, ap, ldap, bp, ldbp, work, nm);
             }));
}

} /* anonymous namespace */

/* The C entry points: one definition per routine, instantiated for double (d)
 * and float (s). T is a type name and p a token to paste, so neither can be
 * parenthesized. */
// NOLINTBEGIN(bugprone-macro-parentheses)
#define CQR_DEFINE_MKL_ENTRY_POINTS(T, p)                                                \
    void cqr_mkl_##p##geqrf_compact(MKL_LAYOUT layout, MKL_INT m, MKL_INT n, T *ap,      \
                                    MKL_INT ldap, T *taup, T *work, MKL_INT lwork,       \
                                    MKL_INT *info, MKL_COMPACT_PACK format, MKL_INT nm)  \
    {                                                                                    \
        geqrf(layout, m, n, ap, ldap, taup, work, lwork, info, format, nm);              \
    }                                                                                    \
    void cqr_mkl_##p##ormqr_compact(                                                     \
        MKL_LAYOUT layout, char side, char trans, MKL_INT m, MKL_INT n, MKL_INT k,       \
        const T *ap, MKL_INT ldap, const T *taup, T *cp, MKL_INT ldcp, T *work,          \
        MKL_INT lwork, MKL_INT *info, MKL_COMPACT_PACK format, MKL_INT nm)               \
    {                                                                                    \
        ormqr(layout, side, trans, m, n, k, ap, ldap, taup, cp, ldcp, work, lwork, info, \
              format, nm);                                                               \
    }                                                                                    \
    void cqr_mkl_##p##potrf_compact(MKL_LAYOUT layout, MKL_UPLO uplo, MKL_INT n, T *ap,  \
                                    MKL_INT ldap, MKL_INT *info,                         \
                                    MKL_COMPACT_PACK format, MKL_INT nm)                 \
    {                                                                                    \
        potrf(layout, uplo, n, ap, ldap, info, format, nm);                              \
    }                                                                                    \
    void cqr_mkl_##p##trsm_compact(MKL_LAYOUT layout, MKL_SIDE side, MKL_UPLO uplo,      \
                                   MKL_TRANSPOSE transa, MKL_DIAG diag, MKL_INT m,       \
                                   MKL_INT n, T alpha, const T *ap, MKL_INT ldap, T *bp, \
                                   MKL_INT ldbp, MKL_COMPACT_PACK format, MKL_INT nm)    \
    {                                                                                    \
        trsm(layout, side, uplo, transa, diag, m, n, alpha, ap, ldap, bp, ldbp, format,  \
             nm);                                                                        \
    }                                                                                    \
    void cqr_mkl_##p##gels_compact(MKL_LAYOUT layout, char trans, MKL_INT m, MKL_INT n,  \
                                   MKL_INT nrhs, T *ap, MKL_INT ldap, T *bp,             \
                                   MKL_INT ldbp, T *work, MKL_INT lwork, MKL_INT *info,  \
                                   MKL_COMPACT_PACK format, MKL_INT nm)                  \
    {                                                                                    \
        gels(layout, trans, m, n, nrhs, ap, ldap, bp, ldbp, work, lwork, info, format,   \
             nm);                                                                        \
    }
// NOLINTEND(bugprone-macro-parentheses)

extern "C" {
CQR_DEFINE_MKL_ENTRY_POINTS(double, d)
CQR_DEFINE_MKL_ENTRY_POINTS(float, s)
} /* extern "C" */

#undef CQR_DEFINE_MKL_ENTRY_POINTS
