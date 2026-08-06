# API reference

`cqr` exposes two C-callable API surfaces. They compute the same factorizations
and solves; they differ in how you describe the batch and whether arguments are
validated.

| Header | Symbols | Batch descriptor | Argument checking |
|--------|---------|------------------|-------------------|
| [`cqr_mkl_ext.h`](#mkl-style-api) | `cqr_mkl_?*_compact` | `MKL_LAYOUT` + `MKL_COMPACT_PACK` (drop-in for MKL) | none — Compact convention |
| [`cqr_compact.h`](#portable-c-api) | `?*_compact` | explicit interleave width `V` | LAPACK-style `info = -j` |

Both are `extern "C"` and **real-precision only**: every routine comes in double
(`d…`) and single (`s…`). Complex (`c`/`z`) is out of scope. Throughout, `?`
stands for the precision prefix.

Every routine is documented in full in its [design note](design/index.md); this
page is the calling reference.

## The Compact (interleaved) format

Both APIs operate on matrices stored in MKL's Compact layout: the batch is cut
into groups of `V` matrices, and element `(i, j)` of the `V` matrices in a group
is stored contiguously (interleaved), so one SIMD load pulls the same element
from all `V` matrices at once. `V` is the SIMD width — the register byte width
divided by `sizeof(T)`:

| Format | Register | `V` (FP64) | `V` (FP32) |
|--------|----------|-----------|-----------|
| `MKL_COMPACT_SSE` | 16 B | 2 | 4 |
| `MKL_COMPACT_AVX` | 32 B | 4 | 8 |
| `MKL_COMPACT_AVX512` | 64 B | 8 | 16 |

For group `g = idx/V` and slot `v = idx%V`, the column-major addressing is:

```text
A_v(i,j)  = ap [ g*ldap*ncol*V + (j*ldap + i)*V + v ]
tau_v(kk) = taup[ g*k*V         +  kk*V          + v ]
B_v(i,j)  = bp [ g*ldbp*nrhs*V  + (j*ldbp + i)*V + v ]
```

where `ncol` is `n` for `?geqrf` (the full matrix) and `k` for `?ormqr` (`A` is
the `(ldap, k)` reflector batch). Row-major swaps the in-matrix index roles
(`i → i*ldap + j`). A partial final group is **padded with identity
factorizations** so the kernels run it unmasked.

!!! note "Buffer alignment"
    Compact buffers are correct at any `T` alignment, but align each buffer's
    base to the pack width in bytes — **64 covers every format** — so the SIMD
    kernels avoid cache-line splits (`mkl_malloc(bytes, 64)` or
    `std::aligned_alloc(64, …)`). Only performance rides on it, up to ~40% on
    small, cache-resident sizes.

---

## MKL-style API {#mkl-style-api}

```c
#include "cqr_mkl_ext.h"   // needs the MKL headers for MKL_LAYOUT etc.; adds no MKL link dependency
```

These mirror MKL's native compact routines (`MKL_LAYOUT` + `MKL_COMPACT_PACK`),
so they drop into the MKL compact ecosystem and mix freely with MKL's own
compact calls — but the computation is `cqr`'s portable SIMD kernels.

Following the Compact convention, **these routines do not validate their
arguments** (compact routines skip checks for vectorization). `info` is a single
scalar, set to `0` on success. For validated calls, use the
[portable C API](#portable-c-api).

### cqr_mkl_?geqrf_compact — QR factorization

Factorizes each `m × n` matrix as `A = Q R`. On exit `ap` holds `R` (on and
above the diagonal) and the Householder vectors (below); `taup` holds the
`k = min(m,n)` reflector scalars — the LAPACK `?geqrf` storage convention.
Drop-in for `mkl_?geqrf_compact` (identical signature).

```c
void cqr_mkl_dgeqrf_compact(MKL_LAYOUT layout, MKL_INT m, MKL_INT n, double *ap,
                            MKL_INT ldap, double *taup, double *work, MKL_INT lwork,
                            MKL_INT *info, MKL_COMPACT_PACK format, MKL_INT nm);
```

| Parameter | Meaning |
|-----------|---------|
| `layout` | `MKL_COL_MAJOR` (tuned) or `MKL_ROW_MAJOR`. |
| `m`, `n` | rows and columns of each `A`. |
| `ap` | compact `A` (`m × n`); overwritten with `(R, Householder vectors)`. |
| `ldap` | compact leading dimension (`≥ m` col-major, `≥ n` row-major). |
| `taup` | compact `tau` output, `k = min(m,n)` per matrix. |
| `work`, `lwork` | workspace; with `lwork = -1` the call is a query returning the optimal `lwork` in `work[0]` (this kernel needs none, so `1`). |
| `info` | scalar status, `0` on success. |
| `format`, `nm` | Compact pack format; total number of matrices. |

### cqr_mkl_?ormqr_compact — apply Q or Qᵀ

Multiplies a compact batch of general matrices `C` by the orthogonal factor `Q`
(or `Qᵀ`) from a `?geqrf` factorization — the missing `mkl_?ormqr_compact`. `C`
is overwritten with `op(Q)·C`.

```c
void cqr_mkl_dormqr_compact(MKL_LAYOUT layout, char side, char trans, MKL_INT m,
                            MKL_INT n, MKL_INT k, const double *ap, MKL_INT ldap,
                            const double *taup, double *cp, MKL_INT ldcp, double *work,
                            MKL_INT lwork, MKL_INT *info, MKL_COMPACT_PACK format,
                            MKL_INT nm);
```

| Parameter | Meaning |
|-----------|---------|
| `layout` | `MKL_COL_MAJOR` (tuned, `side='L'`) or `MKL_ROW_MAJOR`. |
| `side` | `'L'/'l'` for `op(Q)·C`, `'R'/'r'` for `C·op(Q)`. |
| `trans` | `'N'/'n'` applies `Q`; `'T'/'t'/'C'/'c'` applies `Qᵀ` (`C` folds to `T` for real types). |
| `m`, `n` | rows and columns of each `C`. |
| `k` | number of reflectors; `A` is the `(ldap, k)` reflector batch — the buffer **must be packed with exactly `k` columns**. |
| `ap`, `ldap`, `taup` | the reflectors and `tau` from `?geqrf_compact` (consumed directly for square/tall factors). |
| `cp`, `ldcp` | compact `C`; overwritten with `op(Q)·C`. |
| `work`, `lwork`, `info` | as `geqrf`; query returns `1`. |
| `format`, `nm` | Compact pack format; total number of matrices. |

!!! note "Wide factors"
    For a wide factor (`k <` columns), pack only its `k` reflector columns, or
    use the portable `dormqr_compact` / `sormqr_compact`, which takes an explicit
    packed-column count for the non-conforming layout.

### cqr_mkl_?potrf_compact — Cholesky

Factorizes each SPD `n × n` matrix as `A = L Lᵀ` (`MKL_LOWER`) or `A = Uᵀ U`
(`MKL_UPPER`). On exit the named triangle holds the factor; the other is
untouched. No workspace, no argument checking, and **no SPD test** — a non-SPD
lane poisons itself with `NaN`/`Inf`, not `info = j`. Drop-in for
`mkl_?potrf_compact`.

```c
void cqr_mkl_dpotrf_compact(MKL_LAYOUT layout, MKL_UPLO uplo, MKL_INT n, double *ap,
                            MKL_INT ldap, MKL_INT *info, MKL_COMPACT_PACK format,
                            MKL_INT nm);
```

| Parameter | Meaning |
|-----------|---------|
| `layout` | `MKL_COL_MAJOR` (tuned when lower) or `MKL_ROW_MAJOR`. |
| `uplo` | `MKL_LOWER` factor `L`, or `MKL_UPPER` factor `U`. |
| `n` | order of each `A`. |
| `ap`, `ldap` | compact `A` (`n × n`); the named triangle is overwritten with `L`/`U`. |
| `info` | scalar status: `0` ok, `-1` for an unrecognized format. |
| `format`, `nm` | Compact pack format; total number of matrices. |

### cqr_mkl_?trsm_compact — triangular solve

Solves, in place for every matrix in the batch, `op(A) X = alpha·B`
(`MKL_LEFT`) or `X op(A) = alpha·B` (`MKL_RIGHT`), with `A` triangular and `B`
(`m × n`) overwritten by `X`. Drop-in for `mkl_?trsm_compact`: like the BLAS
`?trsm` it batches, it takes no workspace and reports no `info`.

```c
void cqr_mkl_dtrsm_compact(MKL_LAYOUT layout, MKL_SIDE side, MKL_UPLO uplo,
                           MKL_TRANSPOSE transa, MKL_DIAG diag, MKL_INT m, MKL_INT n,
                           double alpha, const double *ap, MKL_INT ldap, double *bp,
                           MKL_INT ldbp, MKL_COMPACT_PACK format, MKL_INT nm);
```

| Parameter | Meaning |
|-----------|---------|
| `layout` | `MKL_COL_MAJOR` (tuned) or `MKL_ROW_MAJOR`. |
| `side` | `MKL_LEFT` (`op(A) X = αB`) or `MKL_RIGHT` (`X op(A) = αB`). |
| `uplo` | `MKL_UPPER` or `MKL_LOWER` triangle of `A`. |
| `transa` | `MKL_NOTRANS` (`A`) or `MKL_TRANS`/`MKL_CONJTRANS` (`Aᵀ`, folded for real types). |
| `diag` | `MKL_UNIT` (diagonal not read) or `MKL_NONUNIT`. |
| `m`, `n` | rows and columns of `B` (`A` is `s × s`, `s = m` for LEFT, `n` for RIGHT). |
| `alpha` | scalar on `B`; `alpha = 0` sets `B := 0` (`A` not referenced). |
| `ap`, `ldap` | compact triangular `A`. |
| `bp`, `ldbp` | compact `B`; overwritten with `X`. |
| `format`, `nm` | Compact pack format; total number of matrices. |

---

## Portable C API {#portable-c-api}

```c
#include "cqr_compact.h"   // no MKL dependency
```

The same kernels with an explicit interleave width `V` (2, 4, 8, or 16) and no
MKL dependency. Unlike the MKL-style API, these **validate their arguments**
LAPACK-style: the return value is `0` on success or `-j` for an illegal `j`-th
argument. An empty problem is a valid no-op returning `0`; the routines never
abort the calling process.

`V` is the interleave width — SSE `d=2`/`s=4`, AVX `d=4`/`s=8`, AVX-512
`d=8`/`s=16` (any of these also run on NEON/SVE as unrolled bursts). `layout` is
`'C'/'c'` (column-major, tuned) or `'R'/'r'` (row-major).

### dgeqrf_compact / sgeqrf_compact — QR factorization

```c
int dgeqrf_compact(char layout, int m, int n, double *ap, int ldap, double *taup,
                   int V, int nm);
int sgeqrf_compact(char layout, int m, int n, float  *ap, int ldap, float  *taup,
                   int V, int nm);
```

On exit `ap` holds `R` and the Householder vectors, `taup` the `k = min(m,n)`
reflector scalars. Returns `0`, or `-j` for an illegal argument:

`-1` `layout`  · `-2` `m` (<0)  · `-3` `n` (<0)  · `-5` `ldap`  · `-7` `V` (not 2/4/8/16)  · `-8` `nm` (<0)

### dormqr_compact / sormqr_compact — apply Q or Qᵀ

```c
int dormqr_compact(char trans, int m, int nrhs, int k, const double *ap, int ldap,
                   const double *taup, double *bp, int ldbp, int V, int nm);
int sormqr_compact(char trans, int m, int nrhs, int k, const float  *ap, int ldap,
                   const float  *taup, float  *bp, int ldbp, int V, int nm);
```

Applies `Q` (`trans='N'`) or `Qᵀ` (`trans='T'`) **from the left**,
`B := op(Q) B` — the reflector-application step between `?geqrf_compact` and the
triangular solve. `k` is the reflector count; `A` is `(ldap, k)` per matrix.
Returns `0`, or `-j`:

`-1` `trans` · `-2` `m` (<0) · `-3` `nrhs` (<0) · `-4` `k` (<0 or >`m`) · `-6` `ldap` · `-9` `ldbp` · `-10` `V` · `-11` `nm` (<0)

### dpotrf_compact / spotrf_compact — Cholesky

```c
int dpotrf_compact(char layout, char uplo, int n, double *ap, int ldap, int V, int nm);
int spotrf_compact(char layout, char uplo, int n, float  *ap, int ldap, int V, int nm);
```

`uplo` is `'L'/'l'` (factor/store lower `L`) or `'U'/'u'` (upper `U`); the
opposite triangle is neither referenced nor modified. Positive-definiteness is
assumed, not checked — a non-SPD lane yields `NaN`/`Inf`. Returns `0`, or `-j`:

`-1` `layout` · `-2` `uplo` · `-3` `n` (<0) · `-5` `ldap` (< max(1,`n`)) · `-6` `V` · `-7` `nm` (<0)

### dtrsm_compact / strsm_compact — triangular solve

```c
int dtrsm_compact(char layout, char side, char uplo, char transa, char diag,
                  int m, int n, double alpha, const double *ap, int ldap,
                  double *bp, int ldbp, int V, int nm);
int strsm_compact(char layout, char side, char uplo, char transa, char diag,
                  int m, int n, float  alpha, const float  *ap, int ldap,
                  float  *bp, int ldbp, int V, int nm);
```

Solves `op(A) X = alpha·B` (`side='L'`) or `X op(A) = alpha·B` (`side='R'`).
`transa` is `'N'` (`A`) or `'T'`/`'C'` (`Aᵀ`); `diag` is `'U'` (unit, not read)
or `'N'`. `B` (`m × n`) is overwritten with `X`; `alpha = 0` sets `B := 0`.
Returns `0`, or `-j`:

`-1` `layout` · `-2` `side` · `-3` `uplo` · `-4` `transa` · `-5` `diag` · `-6` `m` (<0) · `-7` `n` (<0) · `-10` `ldap` · `-12` `ldbp` · `-13` `V` · `-14` `nm` (<0)

*(`alpha`, `ap`, and `bp` are never inspected, matching LAPACK/BLAS.)*
