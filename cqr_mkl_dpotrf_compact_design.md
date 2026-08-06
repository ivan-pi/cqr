# API Design Document: `cqr_mkl_dpotrf_compact`

> Assisted-by: Claude:claude-opus-4.8

## 1. Overview

This extension provides a fast batched **Cholesky factorization** of a set of
symmetric positive-definite `n x n` matrices stored in Intel MKL's Compact
(interleaved-batch) format. It mirrors `mkl_?potrf_compact` in signature and
semantics but is a fully portable, open implementation built on GNU vector types
-- specialized for SSE, AVX, and AVX-512 registers -- so it can be used,
studied, and tuned without depending on MKL's closed compact kernels.

It joins the QR routines already in this project (`cqr_mkl_?geqrf_compact`,
`cqr_mkl_?ormqr_compact`), sharing the same `pack<T,V>` GNU-vector machinery and
the same MKL-style API surface (`MKL_LAYOUT` + `MKL_COMPACT_PACK`). Paired with
MKL's own `mkl_?trsm_compact`, the factorization solves batched
symmetric-positive-definite systems in the compact format:

```
cqr_mkl_dpotrf_compact('L', A -> L);          // A = L L^T
mkl_dtrsm_compact('L','L','N', L, B);        // B := L^{-1} B
mkl_dtrsm_compact('L','L','T', L, B);        // B := L^{-T} (L^{-1} B) = A^{-1} B = X
```

The primary target is many small-to-medium matrices, with order in `3..500` and
the emphasis on sizes below 170. The tuned path is column-major, lower triangle
-- matching LAPACK and the natural Cholesky data flow.

## 2. Syntax

```c
void cqr_mkl_dpotrf_compact (
    MKL_LAYOUT layout, MKL_UPLO uplo, MKL_INT n,
    double * ap, MKL_INT ldap, MKL_INT * info,
    MKL_COMPACT_PACK format, MKL_INT nm
);
```

The signature is identical to
[`mkl_dpotrf_compact`](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-2/mkl-potrf-compact.html),
so the routine is a drop-in alternative within the MKL Compact ecosystem. Both
real precisions are provided: `cqr_mkl_dpotrf_compact` (double) and
`cqr_mkl_spotrf_compact` (single). Unlike `?geqrf`, `?potrf` needs no workspace,
so -- like MKL's compact `potrf` -- there are no `work`/`lwork` arguments.

## 3. Description

The routine forms the Cholesky factorization of each symmetric positive-definite
`n x n` matrix `A` in the batch,

```
A = U^T U,   if uplo = MKL_UPPER  (U upper triangular),
A = L L^T,   if uplo = MKL_LOWER  (L lower triangular),
```

with `L` (or `U`) having a positive diagonal. On exit each matrix is overwritten
exactly as `mkl_?potrf_compact` (and LAPACK `?potrf`) leave it:

* if `uplo = MKL_LOWER`, the lower triangle of `A` is overwritten by `L`; the
  strictly upper triangle is neither referenced nor modified;
* if `uplo = MKL_UPPER`, the upper triangle of `A` is overwritten by `U`; the
  strictly lower triangle is neither referenced nor modified.

`A` must be symmetric; only the triangle named by `uplo` is read, so the caller
need only populate that triangle. Before calling this routine, pack the matrices
with `mkl_?gepack_compact`; after it, call `mkl_?geunpack_compact` unless another
compact routine (e.g. `mkl_?trsm_compact`) will consume the factors first.

**Constraint note.** As with all Compact routines, every matrix in the call
shares the same order `n`, leading dimension `ldap`, storage `layout`, and
`format`. The batch is processed one *pack* (group of `V` interleaved matrices)
at a time; `V` is derived from `format`.

Like ArmPL's `armpl_?potrf_interleave_batch`, this is an interleave-batch
Cholesky that -- unlike LAPACK -- does not verify that the inputs are SPD (section
6.2). Where ArmPL exposes the interleaved layout through explicit
`ninter`/`bstrd`/`istrd`/`jstrd` strides, this API abstracts it behind the
`MKL_COMPACT_PACK` parameter and `MKL_LAYOUT`, preserving symmetry with MKL's
native compact API: the batch is moved into the compact layout by
`mkl_?gepack_compact` and back out by `mkl_?geunpack_compact`, and this routine
consumes the packed buffer directly in between (as does `mkl_?trsm_compact` when
the factor feeds a solve, so no unpack is needed until the final result is read).

## 4. Input Parameters

* **`layout`** (`MKL_LAYOUT`): the in-memory storage order of each matrix --
  `MKL_COL_MAJOR` (tuned path) or `MKL_ROW_MAJOR`.
* **`uplo`** (`MKL_UPLO`): `MKL_LOWER` (factor and store the lower triangle `L`;
  tuned path) or `MKL_UPPER` (upper triangle `U`). The other triangle is not
  referenced.
* **`n`** (`MKL_INT`): the order of each `A` (`n >= 0`).
* **`ap`** (`double *`): the compact buffer of `nm` matrices `A`, packed with
  `mkl_?gepack_compact`. Overwritten in place with the Cholesky factor.
* **`ldap`** (`MKL_INT`): leading dimension of each matrix within the compact
  buffer (column stride for column-major, row stride for row-major), `>= n`.
* **`format`** (`MKL_COMPACT_PACK`): the pack format from
  `mkl_get_format_compact()`; selects the interleave width `V`
  (SSE/AVX/AVX-512 -> 2/4/8 for FP64, 4/8/16 for FP32).
* **`nm`** (`MKL_INT`): total number of matrices in the batch (`nm >= 0`).

**Buffer alignment.** Any base alignment of `ap` is correct. For full speed,
align the base to the pack width (64 B covers every format) so each SIMD access
stays on one cache line instead of splitting across two -- worth up to ~40% on
small, cache-resident sizes. `mkl_malloc(bytes, 64)` (the default of this
project's `mkl_alloc_bytes`) already does this.

## 5. Output Parameters

* **`ap`**: the named triangle is overwritten with its Cholesky factor `L` or
  `U`, in Compact format; the other triangle is left untouched.
* **`info`** (`MKL_INT *`): a single scalar status, `0` on success. MKL leaves
  the compact `info` reserved rather than reporting a non-SPD leading minor the
  way LAPACK `?potrf` does, and this routine does the same (section 6.2). The one
  value it can set is dispatch-level: an unrecognized `format` selects no kernel
  and sets `info = -1`.

## 6. Design Considerations & Compatibility

### 6.1 The algorithm: vectorized unblocked Cholesky (`potf2`)

The batch is factored with the unblocked LAPACK algorithm (`dpotf2`), executed
`V` matrices at a time. Because Compact format interleaves the `V` matrices so
that element `(i,j)` of all `V` is contiguous, the scalar algorithm lifts almost
verbatim with `double -> V`-wide vector: every `+`, `-`, `*`, `/`, and `sqrt`
becomes a lane-wise SIMD operation over `V` independent matrices. The
right-looking form is used, which for the lower triangle is, per pivot column
`j`:

```
d       = sqrt(A(j,j))                          // pivot (vector sqrt)
A(j,j)  = d
invd    = 1 / d
A(i,j) *= invd            for i > j             // scale the pivot column
A(i,jj)-= A(i,j)*A(jj,j)  for jj > j, i >= jj   // symmetric rank-1 trailing update
```

Only the lower trapezoid is ever touched, so the strictly-upper triangle passes
through untouched as `?potrf` requires. The trailing update -- the `O(n^3)` bulk
of the work -- is register-blocked `JB = 4` trailing columns at a time so each
pivot-column entry `A(i,j)` load is reused across four columns, exactly as the
`geqrf` trailing update. Blocked (`potrf`) factorization with `syrk`/`trsm`
panels is deliberately *not* used: for the target sizes the panels are short, and
the interleaved batch is already likely to saturate the vector units without the
extra blocking bookkeeping. `vsqrt<T,V>` (a short lane loop that GCC and Clang
lower to a single `vsqrt*`) is the only special function needed; it runs once per
column, negligible next to the `O(n^2)` scaling and `O(n^3)` update.

### 6.2 The pivot: `sqrt`, and no positive-definiteness check

Cholesky needs no branch-free trickery: unlike QR's `larfg`, its math has no
data-dependent branch. Scalar `dpotf2` has exactly one test -- `if (ajj <= 0 ||
isnan(ajj))` set `info = j` and stop, flagging a non-SPD leading minor -- and that
is the only thing that would diverge per lane across a pack. Both vendors'
interleave-batch Cholesky drop it: MKL leaves `info` "reserved for future use"
(its compact routines "skip error checking for performance reasons"), and ArmPL
"does not check that the input matrices are SPD; no error will be returned if any
`A_i` are not SPD." This routine does the same: it computes `d = sqrt(A(j,j))`
unconditionally.

The consequence is graceful "garbage in, garbage out": a genuinely non-SPD lane
has some pivot `A(j,j) <= 0`, so `sqrt` yields `NaN` (or the following `1/d`
yields `Inf`), and the poison propagates through that lane's factor. The caller
detects it by inspecting the unpacked diagonal, exactly as with MKL's compact
`potrf`. Early-exit with `info = j` is intentionally not provided -- it is
precisely the per-lane branch that does not vectorize across a pack.

### 6.3 Layouts and triangles: one tuned contiguous kernel, one strided

There are four `(layout, uplo)` combinations, and they pair up by transpose
duality into two contiguous cases and two strided ones:

* **Contiguous (tuned).** Column-major + `MKL_LOWER`: a factor column is
  contiguous in the compact buffer (consecutive rows are one `V`-wide pack
  apart), so the pivot `sqrt`, the column scaling, and the rank-1 update all walk
  contiguous packs. Its transpose dual, row-major + `MKL_UPPER`, is the same
  memory access pattern with `(i,j)` roles swapped, so it routes to the same
  tuned kernel by index transposition (`A = U^T U` on the row-major upper triangle
  is `A^T = L L^T` on the reinterpreted column-major lower triangle, and `A` is
  symmetric).
* **Strided.** Column-major + `MKL_UPPER` and row-major + `MKL_LOWER` walk the
  factor across the non-contiguous axis; they are supported for MKL compatibility
  through a stride-generalized kernel over the same `potf2` math
  (correctness-first; the strided inner sweep is not separately SIMD-tuned),
  reusing the existing `BatchView` addressing.

### 6.4 Padding and SIMD semantics

When `nm` is not a multiple of `V`, `mkl_?gepack_compact` fills the unused slots
of the last pack with identity matrices. The Cholesky factor of the identity is
the identity (`L = I`, all pivots `1`, no off-diagonal fill), so the padded lanes
compute a mathematical no-op and the kernel runs the whole final pack unmasked at
full width without corrupting real data. Because the pivot path is
unconditional, the identity flows through it with no lane mask -- unlike `geqrf`,
whose `larfg` needs a mask to neutralize padded columns.

### 6.5 No argument checking (Compact convention)

Like MKL's own compact routines -- which "skip error checking for performance
reasons" and make "the user responsible for passing correct parameters" --
`cqr_mkl_?potrf_compact` validates no arguments and writes a single scalar
`info = 0`. The only failure it can report is dispatch-level: an unrecognized
`format` has no kernel to run and sets `info = -1` (section 5).

### 6.6 Numerical scope

The unblocked `potf2` is backward stable for symmetric positive-definite input to
working precision across the target range, matching LAPACK `?potf2` element for
element (the SPD Cholesky factor with positive diagonal is unique, so agreement
is expected far below the backward-error bound -- see 7.1). Three limits are the
deliberate scope of this routine:

* **Positive definiteness is assumed, not enforced.** Non-SPD and
  borderline-semidefinite inputs -- where rounding can drive a true-zero pivot
  slightly negative -- poison their lane with `NaN`/`Inf` instead of taking a
  safeguarded path or reporting `info = j`.
* **Matrices scaled near underflow/overflow.** Following Intel's stated
  [numerical limitations for Compact BLAS and Compact LAPACK
  routines](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-2/numerical-limits-compact-blas-compact-lapack.html),
  the compact factorization does not provide safe handling of values near
  underflow/overflow; such inputs are out of scope.
* **Real precisions only; no pivoting.** Single and double (`s`/`d`); the
  Hermitian complex variants (`c`/`z`, `A = U^H U`) are out of scope, as they are
  for the QR routines. The factorization is unpivoted -- LAPACK `?potrf` is
  unpivoted too, and symmetric (diagonal) pivoting (`?pstrf`) needs comparisons
  that do not vectorize across a pack.

## 7. Testing and Validation Methodology

Correctness is checked against standard dense LAPACK. SIMD, blocking, and the
tuned/strided split are internal strategies only: the returned factor must
satisfy the same invariants as an unbatched `?potrf`.

### 7.1 Suite 1 -- Factorization invariants vs dense LAPACK

For each `(V, uplo, layout)`, a batch of random SPD `A` is generated with a
controlled condition number (`A = Q diag(logspace(0, -cond, n)) Q^T`, or the
cheaper `A = M^T M + n*I` with a `cond` diagonal-scaling knob), factored by the
routine under test, unpacked, and checked per matrix against the LAPACK Cholesky
contract:

* **Reconstruction residual.** Form the factor product (`L L^T` for lower,
  `U^T U` for upper) and gate `|| L L^T - A ||_1 / ||A||_1 <= 20 * n * eps`.
* **Triangularity / untouched triangle.** Confirm the strictly-opposite triangle
  of `ap` is bit-for-bit unchanged from the input (the routine must not reference
  or write it), which simultaneously gates that the factor is triangular.
* **Elementwise vs LAPACK (gated).** Unlike QR reflectors, the SPD Cholesky
  factor is *unique* (positive diagonal), so the elementwise difference of the
  factor vs `LAPACKE_dpotrf` is a sharp, meaningful signal and is gated, not just
  printed: `|| L_cqr - L_lapack ||_1 / ||L_lapack||_1 <= 20 * n * eps`.

### 7.2 Suite 2 -- Cross-check vs `mkl_dpotrf_compact`

The same packed batch is factored by both `cqr_mkl_dpotrf_compact` and the native
`mkl_dpotrf_compact`; the two compact `ap` buffers are compared elementwise at a
small fixed tolerance (`1e-9`; agreement on well-conditioned SPD inputs is
expected at the `1e-14` level). This confirms the two implementations match far
beyond the backward-error gate -- same factor, same unblocked math -- without
requiring bit-identical arithmetic.

### 7.3 Suite 3 -- End-to-end solve `AX = B`

Closing the pipeline: `B = A X` for a known SPD `A` and known `X`, then
`cqr_mkl_dpotrf_compact('L') -> mkl_dtrsm_compact('L','L','N') ->
mkl_dtrsm_compact('L','L','T')` (forward then back substitution) must recover
`X`. Gate the forward error `Xhat - X` and the residual `A Xhat - B` at
`100 * n * eps` (relative to the matrix L1 norm). This validates `?potrf` in situ
with the rest of the compact toolkit.

### 7.4 Portable self-test

A self-contained test validates the templated kernel directly against a scalar
reference (`ref_potf2`) across `(T, V)` combinations, both `uplo`, and partial
(padded) final packs, plus the LAPACK-style argument validation of the portable C
API. It needs no external libraries at all; only the suites above require an MKL
installation (for the Compact API). BLAS and LAPACK themselves are assumed
available, as they are on most platforms -- it is the MKL Compact extension that
must be installed separately.

## 8. Implementation Strategy

Modern C++ (C++17) templated on scalar type `T` and interleave width `V`, exposed
through `extern "C"` for the FFI-stable surfaces, reusing the existing
`cqr::detail::pack<T,V>` GNU-vector machinery and the `vsqrt<T,V>` helper already
defined for `geqrf`.

### 8.1 API boundary

* **MKL-style API** (`cqr_mkl_ext.h`, the primary surface):
  `cqr_mkl_dpotrf_compact` / `cqr_mkl_spotrf_compact`, unwrapping
  `MKL_COMPACT_PACK -> V` and `MKL_UPLO`/`MKL_LAYOUT`, instantiated on `MKL_INT`
  so ILP64 dimensions are not narrowed.
* **Portable C API** (`cqr_compact.h`): `dpotrf_compact` / `spotrf_compact`,
  taking `char layout` (`'C'`/`'R'`), `char uplo` (`'L'`/`'U'`), an explicit
  interleave width `V`, and no MKL dependency, with LAPACK-style `info = -j`
  argument validation.
* **Templated kernel** (`cqr_potrf_compact.hpp`): per-group kernels
  `potrf_compact_group<T,V>` (tuned contiguous, column-major lower) and
  `potrf_compact_group_strided<T,V>` (general, via `BatchView`), driven over all
  packs by `potrf_compact_general<T,V>` (any `layout`/`uplo`; the entry point both
  C adapters call) and a col-major-lower convenience driver `potrf_compact<T,V>`.

### 8.2 Suggested source layout

Mirroring the `geqrf` files, so the split is familiar:

| File | Role |
|------|------|
| `src/cqr_potrf_compact.hpp` | Templated SIMD Cholesky kernel (vectorized `potf2`; scalar `T`, width `V`). |
| `src/cqr_potrf_compact_dispatch.cpp` | Portable `?potrf_compact` C entry points (runtime `V` -> compile-time dispatch, `info = -j`). |
| `src/cqr_mkl_potrf.cpp` | Unwraps `MKL_COMPACT_PACK` -> `V` and `MKL_UPLO`/`MKL_LAYOUT`, calls the kernel. |
| `src/test_cqr_potrf_compact.cpp` | Self-contained correctness test vs a scalar `potf2` reference (no BLAS). |
| `src/test_cqr_potrf_mkl.cpp` | MKL + dense-LAPACK validation (residual, uniqueness, cross-check, solve). |

The `cqr_mkl_?potrf_compact` prototypes are added to `cqr_mkl_ext.h` and the
portable `?potrf_compact` prototypes to `cqr_compact.h`, alongside the existing
QR entry points.

## 9. Benchmark

The compact batched Cholesky is benchmarked against a one-matrix-at-a-time
`LAPACKE_dpotrf` loop (the standard layout) and against MKL's own
`mkl_dpotrf_compact`, over pools of small SPD matrices across the target size
range. It reports per-size throughput (GFLOP/s, using the `~(1/3) n^3` real
Cholesky flop count) and a geometric-mean speedup, and checks the compact factor
against per-matrix LAPACK so it doubles as an integration test. The outer batch
loop is parallelized with OpenMP.

As in `bench_geqrf_compact`, the default size list mixes sizes that are not
multiples of the interleave width (30, 45, 60, 105, 168 from 2-D/3-D RBF-FD
stencils) with the round powers, so the SIMD remainder handling is visible, and
the `--simdlen=2|4|8` and `--size-sweep=nmin:nmax[:stride]` flags carry over for
forcing a narrower width and for a fine cqr-only throughput scan.
