/* cqr_compact.cpp
 *
 * The portable C API (cqr_compact.h): extern "C" wrappers around the templated
 * kernels. Each validates its arguments LAPACK-style -- returning -j for an
 * illegal j-th argument, never inspecting pointers, never aborting the host
 * process -- and dispatches on the runtime interleave width V to a compile-time
 * instantiation via for_vlen.
 *
 * Assisted-by: Claude:claude-fable-5 Claude:claude-opus-4.8
 */

#include "cqr_compact.h"
#include "cqr_geqrf_compact.hpp"
#include "cqr_ormqr_compact.hpp"
#include "cqr_potrf_compact.hpp"
#include "cqr_sytrfnp_compact.hpp"
#include "cqr_sytrsnp_compact.hpp"
#include "cqr_sysvnp_compact.hpp"
#include "cqr_trsm_compact.hpp"

#include <cassert>

namespace {

using cqr::detail::for_vlen;

/* Case-insensitive match of a LAPACK option character. */
inline bool opt(char c, char upper)
{
    return c == upper || c == upper + ('a' - 'A');
}

inline int max1(int x)
{
    return x < 1 ? 1 : x;
}

inline bool vlen_ok(int V)
{
    return V == 2 || V == 4 || V == 8 || V == 16;
}

template <typename T>
int geqrf(char layout, int m, int n, T *ap, int ldap, T *taup, int V, int nm)
{
    const bool col = opt(layout, 'C'), row = opt(layout, 'R');
    if (!col && !row) return -1;
    if (m < 0) return -2;
    if (n < 0) return -3;
    if (ldap < max1(row ? n : m)) return -5;
    if (!vlen_ok(V)) return -7;
    if (nm < 0) return -8;
    if (m == 0 || n == 0 || nm == 0) return 0; /* empty: nothing to compute */
    assert(ap != nullptr && taup != nullptr);

    for_vlen(V, [&](auto v) {
        cqr::detail::geqrf_compact<T, decltype(v)::value>(row, m, n, ap, ldap, taup, nm);
    });
    return 0;
}

template <typename T>
int ormqr(char trans, int m, int nrhs, int k, const T *ap, int ldap, const T *taup, T *bp,
          int ldbp, int V, int nm)
{
    if (!opt(trans, 'T') && !opt(trans, 'N')) return -1;
    if (m < 0) return -2;
    if (nrhs < 0) return -3;
    if (k < 0 || k > m) return -4;
    if (ldap < max1(m)) return -6;
    if (ldbp < max1(m)) return -9;
    if (!vlen_ok(V)) return -10;
    if (nm < 0) return -11;
    if (m == 0 || nrhs == 0 || k == 0 || nm == 0) return 0;
    assert(ap != nullptr && taup != nullptr && bp != nullptr);

    for_vlen(V, [&](auto v) {
        cqr::detail::ormqr_compact<T, decltype(v)::value>(true, false, trans, m, nrhs, k,
                                                          ap, ldap, taup, bp, ldbp, nm);
    });
    return 0;
}

template <typename T>
int potrf(char layout, char uplo, int n, T *ap, int ldap, int V, int nm)
{
    const bool col = opt(layout, 'C'), row = opt(layout, 'R');
    const bool lo = opt(uplo, 'L'), up = opt(uplo, 'U');
    if (!col && !row) return -1;
    if (!lo && !up) return -2;
    if (n < 0) return -3;
    if (ldap < max1(n)) return -5;
    if (!vlen_ok(V)) return -6;
    if (nm < 0) return -7;
    if (n == 0 || nm == 0) return 0;
    assert(ap != nullptr);

    for_vlen(V, [&](auto v) {
        cqr::detail::potrf_compact<T, decltype(v)::value>(row, up, n, ap, ldap, nm);
    });
    return 0;
}

template <typename T>
int sytrfnp(char layout, char uplo, int n, T *ap, int ldap, int V, int nm)
{
    const bool col = opt(layout, 'C'), row = opt(layout, 'R');
    const bool lo = opt(uplo, 'L'), up = opt(uplo, 'U');
    if (!col && !row) return -1;
    if (!lo && !up) return -2;
    if (n < 0) return -3;
    if (ldap < max1(n)) return -5;
    if (!vlen_ok(V)) return -6;
    if (nm < 0) return -7;
    if (n == 0 || nm == 0) return 0;
    assert(ap != nullptr);

    for_vlen(V, [&](auto v) {
        cqr::detail::sytrfnp_compact<T, decltype(v)::value>(row, up, n, ap, ldap, nm);
    });
    return 0;
}

/* ?sytrsnp_compact and ?sysvnp_compact share one signature (layout, uplo, n,
 * nrhs, ap, ldap, bp, ldbp, V, nm) and hence one validation, in argument order:
 * -1 layout, -2 uplo, -3 n, -4 nrhs, -6 ldap, -8 ldbp, -9 V, -10 nm. Sets the
 * kernel flags. */
inline int sytrs_args(char layout, char uplo, int n, int nrhs, int ldap, int ldbp, int V,
                      int nm, bool &row, bool &up)
{
    const bool col = opt(layout, 'C'), lo = opt(uplo, 'L');
    row = opt(layout, 'R');
    up = opt(uplo, 'U');
    if (!col && !row) return -1;
    if (!lo && !up) return -2;
    if (n < 0) return -3;
    if (nrhs < 0) return -4;
    if (ldap < max1(n)) return -6;
    if (ldbp < max1(row ? nrhs : n)) return -8;
    if (!vlen_ok(V)) return -9;
    if (nm < 0) return -10;
    return 0;
}

template <typename T>
int sytrsnp(char layout, char uplo, int n, int nrhs, const T *ap, int ldap, T *bp,
            int ldbp, int V, int nm)
{
    bool row, up;
    if (int e = sytrs_args(layout, uplo, n, nrhs, ldap, ldbp, V, nm, row, up)) return e;
    if (n == 0 || nrhs == 0 || nm == 0) return 0;
    assert(ap != nullptr && bp != nullptr);

    for_vlen(V, [&](auto v) {
        cqr::detail::sytrsnp_compact<T, decltype(v)::value>(row, up, n, nrhs, ap, ldap,
                                                            bp, ldbp, nm);
    });
    return 0;
}

template <typename T>
int sysvnp(char layout, char uplo, int n, int nrhs, T *ap, int ldap, T *bp, int ldbp,
           int V, int nm)
{
    bool row, up;
    if (int e = sytrs_args(layout, uplo, n, nrhs, ldap, ldbp, V, nm, row, up)) return e;
    if (n == 0 || nrhs == 0 || nm == 0) return 0;
    assert(ap != nullptr && bp != nullptr);

    for_vlen(V, [&](auto v) {
        cqr::detail::sysvnp_compact<T, decltype(v)::value>(row, up, n, nrhs, ap, ldap, bp,
                                                           ldbp, nm);
    });
    return 0;
}

template <typename T>
int trsm(char layout, char side, char uplo, char transa, char diag, int m, int n, T alpha,
         const T *ap, int ldap, T *bp, int ldbp, int V, int nm)
{
    const bool col = opt(layout, 'C'), row = opt(layout, 'R');
    const bool left = opt(side, 'L'), right = opt(side, 'R');
    const bool up = opt(uplo, 'U'), lo = opt(uplo, 'L');
    const bool tran = opt(transa, 'T') || opt(transa, 'C');
    const bool unit = opt(diag, 'U');
    const int s = left ? m : n; /* A is the order-s triangular factor */

    if (!col && !row) return -1;
    if (!left && !right) return -2;
    if (!up && !lo) return -3;
    if (!tran && !opt(transa, 'N')) return -4;
    if (!unit && !opt(diag, 'N')) return -5;
    if (m < 0) return -6;
    if (n < 0) return -7;
    /* -8 alpha: every value (including 0, which means B := 0) is legal */
    if (ldap < max1(s)) return -10;
    if (ldbp < max1(row ? n : m)) return -12;
    if (!vlen_ok(V)) return -13;
    if (nm < 0) return -14;
    if (m == 0 || n == 0 || nm == 0) return 0;
    assert(ap != nullptr && bp != nullptr);

    for_vlen(V, [&](auto v) {
        cqr::detail::trsm_compact<T, decltype(v)::value>(left, up, row, tran, unit, m, n,
                                                         alpha, ap, ldap, bp, ldbp, nm);
    });
    return 0;
}

} /* anonymous namespace */

/* The C entry points: one definition per routine, instantiated for double (d)
 * and float (s). T is a type name and p a token to paste, so neither can be
 * parenthesized. */
// NOLINTBEGIN(bugprone-macro-parentheses)
#define CQR_DEFINE_COMPACT_ENTRY_POINTS(T, p)                                            \
    int p##geqrf_compact(char layout, int m, int n, T *ap, int ldap, T *taup, int V,     \
                         int nm)                                                         \
    {                                                                                    \
        return geqrf(layout, m, n, ap, ldap, taup, V, nm);                               \
    }                                                                                    \
    int p##ormqr_compact(char trans, int m, int nrhs, int k, const T *ap, int ldap,      \
                         const T *taup, T *bp, int ldbp, int V, int nm)                  \
    {                                                                                    \
        return ormqr(trans, m, nrhs, k, ap, ldap, taup, bp, ldbp, V, nm);                \
    }                                                                                    \
    int p##potrf_compact(char layout, char uplo, int n, T *ap, int ldap, int V, int nm)  \
    {                                                                                    \
        return potrf(layout, uplo, n, ap, ldap, V, nm);                                  \
    }                                                                                    \
    int p##sytrfnp_compact(char layout, char uplo, int n, T *ap, int ldap, int V,        \
                           int nm)                                                       \
    {                                                                                    \
        return sytrfnp(layout, uplo, n, ap, ldap, V, nm);                                \
    }                                                                                    \
    int p##sytrsnp_compact(char layout, char uplo, int n, int nrhs, const T *ap,         \
                           int ldap, T *bp, int ldbp, int V, int nm)                     \
    {                                                                                    \
        return sytrsnp(layout, uplo, n, nrhs, ap, ldap, bp, ldbp, V, nm);                \
    }                                                                                    \
    int p##sysvnp_compact(char layout, char uplo, int n, int nrhs, T *ap, int ldap,      \
                          T *bp, int ldbp, int V, int nm)                                \
    {                                                                                    \
        return sysvnp(layout, uplo, n, nrhs, ap, ldap, bp, ldbp, V, nm);                 \
    }                                                                                    \
    int p##trsm_compact(char layout, char side, char uplo, char transa, char diag,       \
                        int m, int n, T alpha, const T *ap, int ldap, T *bp, int ldbp,   \
                        int V, int nm)                                                   \
    {                                                                                    \
        return trsm(layout, side, uplo, transa, diag, m, n, alpha, ap, ldap, bp, ldbp,   \
                    V, nm);                                                              \
    }
// NOLINTEND(bugprone-macro-parentheses)

extern "C" {
CQR_DEFINE_COMPACT_ENTRY_POINTS(double, d)
CQR_DEFINE_COMPACT_ENTRY_POINTS(float, s)
} /* extern "C" */

#undef CQR_DEFINE_COMPACT_ENTRY_POINTS
