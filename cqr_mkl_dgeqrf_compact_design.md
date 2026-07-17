# API Design Document: `cqr_mkl_dgeqrf_compact`

> Assisted-by: Claude:claude-opus-4.8

## 1. Overview

This extension provides a fast batched **QR factorization** of a set of general
`m x n` matrices stored in Intel MKL's Compact (interleaved-batch) format. It
mirrors `mkl_?geqrf_compact` in signature and semantics but is a fully portable,
open implementation built on GNU vector types -- specialized for SSE, AVX, and
AVX-512 double-precision registers -- so it can be used, studied, and tuned
without depending on MKL's closed compact kernels.

It is the natural companion to the `cqr_mkl_?ormqr_compact` routine this project
already ships: `?geqrf` produces the reflectors `(H, tau)`, `?ormqr` applies
them, and `?trsm` finishes the solve. Together they complete an all-open
Compact-format QR pipeline:

```
cqr_mkl_dgeqrf_compact(A -> H, tau);          // A = Q R
cqr_mkl_dormqr_compact('L','T', H, tau, B);   // B := Q^T B
mkl_dtrsm_compact      (U, R, B);             // B := R^{-1} Q^T B = X
```

The primary target is many small-to-medium matrices: both dimensions in `3..500`,
with the emphasis on `n < 170` and common sizes such as 30 and 60. Square and
rectangular shapes are supported; the tuned path is column-major, matching
LAPACK and the natural QR data flow.

## 2. Syntax

```c
void cqr_mkl_dgeqrf_compact (
    MKL_LAYOUT layout, MKL_INT m, MKL_INT n,
    double * ap, MKL_INT ldap,
    double * taup,
    double * work, MKL_INT lwork, MKL_INT * info,
    MKL_COMPACT_PACK format, MKL_INT nm
);
```

The signature is identical to `mkl_dgeqrf_compact` (see attached reference), so
the routine is a drop-in alternative: existing packing (`mkl_?gepack_compact`),
format discovery (`mkl_get_format_compact`), and unpacking
(`mkl_?geunpack_compact`) all remain valid and reusable. A single-precision
`cqr_mkl_sgeqrf_compact` is provided for symmetry; double precision is the
focus.

## 3. Description

The routine forms the QR factorization of each `m x n` matrix `A` in the batch,

```
A = Q R,   Q = H(0) H(1) ... H(k-1),   k = min(m, n),
H(kk) = I - tau(kk) * v(kk) * v(kk)^T,
```

where each `v(kk)` is a unit-lower Householder vector (implicit 1 on the
diagonal, sub-diagonal entries stored in column `kk` of `A`). `Q` is never formed
explicitly. On exit each matrix is overwritten exactly as `mkl_?geqrf_compact`
(and LAPACK `?geqrf`) leave it:

* the elements on and above the diagonal hold the `min(m,n) x n` upper
  trapezoidal factor `R` (upper triangular when `m >= n`);
* the elements below the diagonal hold the Householder vectors;
* `taup` holds the `k` scalar factors `tau` per matrix.

**Constraint note.** As with all Compact routines, every matrix in the call
shares the same dimensions (`m`, `n`), leading dimension (`ldap`), storage
`layout`, and `format`. The batch is processed one *pack* (group of `V`
interleaved matrices) at a time; `V` is derived from `format`.

## 4. Input Parameters

* **`layout`** (`MKL_LAYOUT`): `MKL_COL_MAJOR` (tuned path) or `MKL_ROW_MAJOR`.
* **`m`** (`MKL_INT`): rows of each `A` (`m >= 0`).
* **`n`** (`MKL_INT`): columns of each `A` (`n >= 0`).
* **`ap`** (`double *`): the compact buffer of `nm` matrices `A`, packed with
  `mkl_?gepack_compact`. Overwritten in place with the factorization.
* **`ldap`** (`MKL_INT`): leading dimension of each matrix within the compact
  buffer (column stride for column-major, row stride for row-major), `>= m`
  (col-major) / `>= n` (row-major).
* **`taup`** (`double *`): output buffer of `k = min(m,n)` scalars per matrix,
  in Compact format. Sized with `mkl_?get_size_compact(min(m,n), 1, format, nm)`.
* **`work`, `lwork`** (`double *`, `MKL_INT`): workspace. This unblocked kernel
  needs no scratch, so `work[0]` on a query (`lwork = -1`) returns `1`.
* **`format`** (`MKL_COMPACT_PACK`): the pack format from
  `mkl_get_format_compact()`; selects the interleave width `V`
  (SSE/AVX/AVX-512 -> 2/4/8 for FP64, 4/8/16 for FP32).
* **`nm`** (`MKL_INT`): total number of matrices in the batch.

## 5. Output Parameters

* **`ap`**: overwritten with `R` (on/above the diagonal) and the Householder
  vectors (below), in Compact format.
* **`taup`**: the reflector scalars `tau`, in Compact format.
* **`work[0]`**: minimum `lwork` on a workspace query (`1`).
* **`info`** (`MKL_INT *`): MKL leaves the compact `info` reserved, so we define
  it as a single scalar status, `0` on success. The routine performs no
  argument checking (section 6.5).

## 6. Design Considerations & Compatibility

### 6.1 The algorithm: vectorized unblocked Householder QR (`geqr2`)

The batch is factored with the unblocked LAPACK algorithm (`dgeqr2`: `dlarfg`
to build each reflector, `dlarf` to apply it to the trailing columns), executed
`V` matrices at a time. Because Compact format interleaves the `V` matrices so
that element `(i,j)` of all `V` is contiguous, the scalar algorithm lifts almost
verbatim with `double -> V`-wide vector: every `+`, `-`, `*`, `/` becomes a
lane-wise SIMD operation over `V` independent matrices. Blocked (`geqrf`)
factorization is deliberately *not* used -- for the target sizes the reflector
panels are short, and the interleaved batch already saturates the vector units
without the extra `larft`/`larfb` bookkeeping.

### 6.2 The one hard part: a branch-free `larfg`

Scalar `dlarfg` contains a data-dependent branch (`if xnorm == 0: tau = 0`) and
the scalar special functions `hypot`, `copysign`, and a division. Across a pack
the branch would diverge per lane, so it is rewritten branch-free with a mask
and a select -- exactly the structure the reference GPU submission and the
Gemini prototype use, but keyed on the LAPACK-correct quantity (the
*below-diagonal* norm, so an already-triangular column yields `tau = 0` with the
diagonal unchanged, identical to `dlarfg`):

```
tail  = sum_{i>kk} A(i,kk)^2                 // vector reduction (below diagonal)
x0    = A(kk,kk)
norm  = sqrt(x0*x0 + tail)                   // == hypot(x0, ||tail||)
beta  = (x0 >= 0) ? -norm : norm             // -copysign(norm, x0), branch-free
has   = tail > 0                             // lane mask: is there anything to zero?
tau   = has ? (beta - x0)/beta : 0
inv   = has ? 1/(x0 - beta)     : 0          // masked select kills 0/0 and 1/0
A(kk,kk)          = has ? beta : x0          // R diagonal (unchanged if no tail)
A(i>kk,kk)       *= inv                       // reflector body (0 * inv = 0 when !has)
```

Only `sqrt` needs a helper: a short lane loop `for v: r[v] = sqrt(x[v])` that
GCC and Clang both lower to a single `vsqrtpd`. Comparisons yield a native mask
(`vcmppd`); the select is a bit-blend on the may-alias integer view. All of this
is `O(1)` per column -- the `O(n^2)` reductions and the `O(n^3)` trailing update
are pure vector arithmetic. This was validated against scalar `dlarfg` before
committing to the design.

### 6.3 Layouts: tuned column-major, strided row-major

Column-major is the tuned path: a matrix column is contiguous in the compact
buffer (consecutive rows are one `V`-wide pack apart), so the `larfg` reduction,
the reflector scaling, and the trailing-column update all walk contiguous
pointers, register-blocked `JB = 4` trailing columns at a time so each reflector
load is reused. Row-major is supported for MKL compatibility through a
stride-generalized kernel over the same math (correctness-first; the strided
inner sweep is not separately SIMD-tuned), mirroring how `cqr_mkl_?ormqr_compact`
handles its non-contiguous side/layout combinations.

### 6.4 Padding and SIMD semantics

When `nm` is not a multiple of `V`, `mkl_?gepack_compact` fills the unused slots
of the last pack with identity matrices. The QR factorization of the identity is
`R = I`, all `tau = 0`, no Householder vectors -- so the padded lanes contribute
a mathematical no-op and the kernel can run unmasked across the whole final pack
at full width without corrupting real data. Rank-deficient or already-triangular
real columns are handled by the very same `has = tail > 0` mask that neutralizes
the padding (`tau = 0`, diagonal preserved), so no lane ever produces a NaN.

### 6.5 No argument checking (Compact convention)

Like MKL's own compact routines -- which "skip error checking for performance
reasons" and make "the user responsible for passing correct parameters" -- the
MKL-style `cqr_mkl_?geqrf_compact` validates nothing and writes a single scalar
`info = 0`. Callers who want defensive checking use the portable
`dgeqrf_compact` / `sgeqrf_compact` C API (section 8), which performs LAPACK-style
`info = -j` validation and never aborts the process.

### 6.6 Numerical scope

The column norm is the direct `sqrt(sum of squares)` -- fast and vectorizable,
and accurate to working precision across the target range, including the
competition's stress structures (column-scaled/`logspace` dynamic range,
rank-deficient, near-collinear, banded, upper-triangular, clustered-scale): all
of those stay well inside the FP64 exponent range, and rank deficiency degrades
gracefully through the `has` mask. The overflow/underflow-safe rescaling of
LAPACK's `dlarfg` (triggered only near `1e+/-150`) is intentionally omitted, as
is column pivoting; both are outside the stated small-well-scaled regime and are
noted here as the scoped limitations, consistent with MKL's own "Numerical
Limitations for Compact ... Routines".

## 7. Testing and Validation Methodology

Correctness is a hard gate against standard dense LAPACK. SIMD, blocking, and
lane-masking are internal strategies only: the returned `(H, tau, R)` must
satisfy the same invariants as an unbatched `?geqrf`.

### 7.1 Suite 1 -- Factorization invariants vs dense LAPACK

For each `(V, shape)` a batch of random `A` (with the competition's `cond`
column-scaling and a diagonal boost to set conditioning) is factored by the
routine under test, unpacked, and checked per matrix against the LAPACK-style
QR contract used by the GPU competition checker:

* **Factorization residual.** Materialize `Q` from `(H, tau)` (via the project's
  own `?ormqr` applied to `I`, and independently via dense `LAPACKE_dorgqr`),
  take `R = triu(H)`, and gate `|| R - Q^T A ||_1 / ||A||_1 <= 20 * n * eps`.
  Because `R` is the strict upper triangle, this simultaneously gates
  lower-triangular leakage (triangularity).
* **Orthogonality.** Gate `|| Q^T Q - I ||_1 <= 100 * n * eps`.
* **Elementwise vs LAPACK (diagnostic).** For well-conditioned inputs the
  reflectors are essentially unique, so `(H, tau)` are additionally compared
  elementwise to `LAPACKE_dgeqrf` at a loose relative tolerance, as a sharper
  regression signal than the residual alone.

### 7.2 Suite 2 -- Cross-check vs `mkl_dgeqrf_compact`

The same packed batch is factored by both `cqr_mkl_dgeqrf_compact` and the
native `mkl_dgeqrf_compact`; the two compact `ap`/`taup` buffers are compared
elementwise at a tolerance scaled by `eps`. This confirms bit-for-bit-level
agreement with MKL's own compact factorization on the shared well-conditioned
inputs (same sign convention, same unblocked math).

### 7.3 Suite 3 -- End-to-end solve `AX = B`

Closing the pipeline: `B = A X` for a known `X`, then
`cqr_mkl_dgeqrf_compact -> cqr_mkl_dormqr_compact('L','T') -> mkl_dtrsm_compact`
must recover `X`. Gate the forward error `Xhat - X` and the residual
`A Xhat - B` at `100 * n * eps` (relative to the matrix L1 norm), exactly as the
existing `?ormqr` Suite 2. This validates `?geqrf` in situ with the rest of the
compact toolkit.

### 7.4 Portable self-test (no BLAS)

A BLAS-free test (like `test_cqr_compact.cpp`) validates the templated kernel
directly against a scalar reference (`ref_geqr2` / `ref_larfg`) across
`(T, V)` combinations and partial (padded) final packs, plus the LAPACK-style
argument validation of the portable C API.

## 8. Implementation Strategy

Modern C++ (C++17) templated on scalar type `T` and interleave width `V`,
exposed through `extern "C"` for the FFI-stable surfaces, reusing the existing
`cqr::detail::pack<T,V>` GNU-vector machinery.

### 8.1 API boundary

* **Portable C API** (`cqr_geqrf_compact.h`): `dgeqrf_compact` /
  `sgeqrf_compact`, taking an explicit interleave width `V` and no MKL
  dependency, with LAPACK-style `info = -j` argument validation. Mirrors
  `dormqr_compact` in `cqr_compact.h`.
* **MKL-style API** (`cqr_mkl_ext.h`): `cqr_mkl_dgeqrf_compact` /
  `cqr_mkl_sgeqrf_compact`, unwrapping `MKL_COMPACT_PACK -> V`, instantiated on
  `MKL_INT` so ILP64 dimensions are not narrowed. Mirrors
  `cqr_mkl_dormqr_compact`.
* **Templated kernel** (`cqr_geqrf_compact.hpp`): `geqrf_compact<T,V>` over all
  packs; `geqrf_compact_group<T,V>` (tuned col-major) and
  `geqrf_compact_group_strided<T,V>` (general, via `BatchView`).

### 8.2 Phases

1. **Skeleton + tests first (TDD).** Land the headers, dispatch, and a
   deliberately-incomplete kernel body, then the Suite 1-3 tests, so the suite
   compiles and fails (red).
2. **Vectorized `geqr2`.** Implement the branch-free `larfg` + register-blocked
   trailing update until all suites pass (green).
3. **Benchmarks & robustness.** Two benchmarks (section 9), then a correctness,
   API-robustness, and readability pass.

## 9. Benchmarks

1. **vs standard per-matrix LAPACK.** The compact batched factorization against
   a one-matrix-at-a-time `LAPACKE_dgeqrf` loop (and, when available, MKL's own
   `mkl_dgeqrf_compact`) over pools of small matrices across the target size
   range, reporting a geometric-mean speedup and accuracy-gated so it doubles as
   an integration test. Outer batch loop parallelized with OpenMP.
2. **vs batmat.** Against the open-source `batmat` project's `geqrf` benchmark
   (GPL; used only as an external yardstick in our own benchmark, not
   distributed). Best-effort, gated on the dependency being fetchable.
