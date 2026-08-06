# API Design Document: `cqr_mkl_dsyrk_compact`

> Assisted-by: Claude:claude-opus-4.8

## 1. Overview

This extension provides a batched **symmetric rank-k update** for a set of
matrices stored in Intel MKL's Compact (interleaved-batch) format. It is the
Compact-format counterpart of BLAS `?syrk`, completing the compact BLAS-3 set
alongside MKL's own `mkl_?gemm_compact` and this project's
`cqr_mkl_?trsm_compact`. Unlike forming the product through a general
`mkl_?gemm_compact`, it exploits the symmetry of the result -- only one triangle
of the symmetric `n x n` output is referenced -- for roughly half the flops and
half the writes. It mirrors an `mkl_?syrk_compact` in signature and semantics,
but MKL ships no such routine, so like `cqr_mkl_?ormqr_compact` this fills a gap
in the compact ecosystem, as a fully portable, open implementation built on GNU
vector types.

The symmetric rank-k update is the defining kernel of **Cholesky QR** -- the
factorization this project is named for. For a tall matrix `A` (`m x n`,
`m >= n`), Cholesky QR forms the Gram matrix, factors it, and recovers `Q`:

```
cqr_mkl_dsyrk_compact('U','T', A -> G);       // G = A^T A         (rank-k update)
cqr_mkl_dpotrf_compact('U', G -> R);          // G = R^T R         (Cholesky)
cqr_mkl_dtrsm_compact ('R','U','N', R, A->Q); // Q = A R^{-1}      (triangular solve)
```

so `A = Q R` with `Q^T Q = I`. With `cqr_mkl_?syrk_compact` in place, this entire
pipeline runs in the compact format with no MKL compute kernel, exactly as the
QR-based `geqrf -> ormqr -> trsm` solver already does. MKL's compact pack/unpack
helpers (`mkl_?gepack_compact`, `mkl_get_format_compact`, ...) are still used to
move data in and out of the interleaved layout; only the *computation* is cqr's.

The primary target is many small-to-medium matrices (both dimensions in `3..500`,
emphasis below 170), matching the rest of the toolkit; the tuned path is
column-major, `trans = 'T'` (`C = A^T A`) -- the Gram matrix above, in the
column-major layout LAPACK and MKL produce their factors in.

## 2. Syntax

```c
void cqr_mkl_dsyrk_compact (
    MKL_LAYOUT layout, MKL_UPLO uplo, MKL_TRANSPOSE trans,
    MKL_INT n, MKL_INT k, double alpha,
    const double * ap, MKL_INT ldap,
    double beta, double * cp, MKL_INT ldcp,
    MKL_COMPACT_PACK format, MKL_INT nm
);
```

The signature is exactly BLAS
[`?syrk`](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-2/syrk.html)
plus the three arguments MKL's compact routines add (`layout`, `format`, `nm`) --
and, like the other compact BLAS-3 routines (`mkl_?gemm_compact`,
`mkl_?trsm_compact`), no `work`/`lwork`/`info` (there is no workspace and, per the
compact convention, no argument checking). Both real precisions are provided:
`cqr_mkl_dsyrk_compact` (double) and `cqr_mkl_ssyrk_compact` (single).

## 3. Description

For each matrix in the batch the routine forms, in place, one of the symmetric
rank-k updates

```
C := alpha * A * A^T + beta * C     (trans = MKL_NOTRANS, A is n x k)
C := alpha * A^T * A + beta * C      (trans = MKL_TRANS,   A is k x n)
```

where `alpha` and `beta` are scalars, `A` is a general rectangular matrix, and
`C` is the symmetric `n x n` result. Only the triangle of `C` named by `uplo` is
referenced and updated; the opposite triangle is neither read nor written, so on
exit it holds exactly its entry value (the caller may leave it uninitialised).

`op(A) = A^H` (`MKL_CONJTRANS`) is accepted and, for the real types handled here,
is identical to `A^T`; it is folded to the transpose path. (The genuinely
conjugated update `C := alpha A A^H + beta C` is `?herk`, a distinct routine, and
is out of scope here.) When `beta = 0`, `C` is not read on entry -- its prior
contents, even `NaN`/`Inf`, are overwritten -- exactly as reference BLAS `?syrk`
defines it. `k = 0` and `alpha = 0` are *not* special-cased: both reduce the
update to `C := beta * C` (an empty contraction contributes the zero rank-k
term), which the kernel computes directly.

**Constraint note.** As with all Compact routines, every matrix in the call
shares the same dimensions (`n`, `k`), leading dimensions (`ldap`, `ldcp`),
storage `layout`, `format`, and the same `uplo`/`trans`. The batch is processed
one *pack* (group of `V` interleaved matrices) at a time; `V` is derived from
`format`.

**Relationship to MKL and ArmPL.** The behavior captured here is the intersection
of the two production batched rank-k updates, expressed through MKL's compact
interface:

* **Intel MKL** exposes the rank-k update only as the general
  `mkl_?gemm_compact` (there is no `mkl_?syrk_compact`); `cqr_mkl_?syrk_compact`
  adds the symmetry-exploiting specialization over the same Compact
  (interleaved) format, selected by an opaque `MKL_COMPACT_PACK`, with the BLAS
  `?syrk` parameter set (`uplo`, `trans`, `n`, `k`, `alpha`, `beta`).
* **Arm Performance Libraries** provide `armpl_?syrk_interleave_batch` -- the same
  mathematical operation over Arm's interleave-batch format, with the interleaving
  exposed explicitly through `ninter`/`nbatch` and arbitrary strides. As with the
  other cqr routines, this API keeps MKL's abstraction instead, hiding the
  interleave width and strides behind `MKL_COMPACT_PACK` + `MKL_LAYOUT` + the
  compact leading dimensions, for symmetry with MKL's native compact ecosystem.
  ArmPL's real interface has no conjugate transpose; neither do the real types
  here.

## 4. Input Parameters

* **`layout`** (`MKL_LAYOUT`): the in-memory storage order of every matrix in the
  batch, `MKL_COL_MAJOR` (tuned path) or `MKL_ROW_MAJOR`.
* **`uplo`** (`MKL_UPLO`): `MKL_UPPER` -- reference and update the upper triangle
  of `C`; `MKL_LOWER` -- the lower triangle. The opposite triangle is untouched.
* **`trans`** (`MKL_TRANSPOSE`): `MKL_NOTRANS` -> `C := alpha A A^T + beta C`
  (`A` is `n x k`); `MKL_TRANS` -> `C := alpha A^T A + beta C` (`A` is `k x n`);
  `MKL_CONJTRANS` -> `A^H` (== `A^T` for the real types, folded to the transpose
  path).
* **`n`** (`MKL_INT`): the order of the symmetric matrix `C`, and the
  non-contracted dimension of `A` (`n >= 0`).
* **`k`** (`MKL_INT`): the contracted dimension -- the number of columns of `A`
  for `trans = MKL_NOTRANS`, or rows for `MKL_TRANS` (`k >= 0`).
* **`alpha`** (`double`): scalar multiplying the rank-k product.
* **`ap`** (`const double *`): the compact buffer of `nm` matrices `A`
  (`n x k` for `MKL_NOTRANS`, `k x n` for `MKL_TRANS`), packed with
  `mkl_?gepack_compact`.
* **`ldap`** (`MKL_INT`): leading dimension of each `A` within the compact buffer,
  `>= max(1, rows(A))` for column-major or `>= max(1, cols(A))` for row-major.
* **`beta`** (`double`): scalar multiplying `C`. When `beta = 0`, `C` is not read.
* **`cp`** (`double *`): the compact buffer of `nm` symmetric matrices `C`
  (`n x n`); the `uplo` triangle is overwritten in place.
* **`ldcp`** (`MKL_INT`): leading dimension of each `C` within the compact buffer,
  `>= max(1, n)`.
* **`format`** (`MKL_COMPACT_PACK`): the pack format from
  `mkl_get_format_compact()`; selects the interleave width `V`
  (SSE/AVX/AVX-512 -> 2/4/8 for FP64, 4/8/16 for FP32).
* **`nm`** (`MKL_INT`): total number of matrices in the batch.

**Buffer alignment.** Any base alignment of `ap`/`cp` is correct. For full speed,
align each buffer's base to the pack width (64 B covers every format) so each
SIMD access stays on one cache line -- worth up to ~40% on small, cache-resident
sizes. `mkl_malloc(bytes, 64)` (the default of this project's `mkl_alloc_bytes`)
already does this.

## 5. Output Parameters

* **`cp`**: the `uplo` triangle of each `C` is overwritten with
  `alpha op(A) op(A)^T + beta C`, in Compact format; the opposite triangle is
  left untouched.

An unrecognized `format` selects no kernel, so the call is a silent no-op (as for
`cqr_mkl_?trsm_compact`, `?syrk` has no `info` to report a dispatch-level status).

## 6. Design Considerations & Compatibility

### 6.1 The algorithm: vectorized dot-product rank-k update

Each update is formed by the dot-product (inner-product) variant of `?syrk`,
executed `V` matrices at a time. Because Compact format interleaves the `V`
matrices so that element `(i,j)` of all `V` is contiguous, the scalar algorithm
lifts verbatim with `double -> V`-wide vector: every `*`, `+`, and the final
`alpha * . + beta * .` becomes a lane-wise SIMD operation over `V` independent
matrices. There is no data-dependent branch to mask, so the lift is direct.

Each output `C(i,j)` is a length-`k` inner product of two "vectors" of `A` along
the contraction axis:

```
C(i,j) = alpha * sum_{p=0..k-1} A_i(p) * A_j(p)  [ + beta * C(i,j) ]
```

where `A_i` is row `i` of `A` for `trans = 'N'` (contracting over columns) or
column `i` of `A` for `trans = 'T'` (contracting over rows). The dot-product form
is preferred over the rank-1 (outer-product) form for three reasons: `C` is
written exactly once (minimal `C` traffic, and the `beta` fold happens on that
single store); the arithmetic is branch-free; and it **blocks cleanly against the
triangle** -- for a fixed row `i`, every column `j` in that row's triangle is a
*full* length-`k` dot, with no diagonal-corner peeling.

### 6.2 Register blocking (`JB = 4`)

The output columns are register-blocked four at a time (`JB = 4`): for a fixed
row `i`, one load of the `A`-vector `A_i(p)` is reused across the four columns
`A_{j..j+3}(p)`, halving the dominant `A` traffic -- the same reuse trick the
`ormqr` kernel applies across its RHS columns and `trsm` across its RHS block. The
one-, two-, and three-column tails of the triangle are handled by a scalar
remainder loop (the triangle length varies per row, so unlike `trsm` there is no
single fixed tail to specialize; the remainder is at most three columns per row).

### 6.3 Layouts and transpose: one tuned contiguous kernel, one strided

There are four `(trans, layout)` combinations. In the dot-product form the
contraction axis of `A` should be unit-stride for the inner loop to stream
contiguously, and exactly one combination achieves that in the natural layout:

* **Contiguous (tuned).** Column-major + `trans = 'T'` (`C = A^T A`): the
  contracted vectors are the *columns* of `A` (`A` is `k x n`), which are
  contiguous in the column-major compact buffer, so the length-`k` dot streams
  both operands down contiguous packs. This is the Cholesky-QR Gram matrix in the
  layout LAPACK/MKL produce factors in, so it is the case the toolkit most wants
  fast -- mirroring how `trsm` tunes the `side='L'` back-substitution and `potrf`
  the column-major lower factor.
* **Strided.** Column-major + `trans = 'N'` (the contracted vectors are *rows* of
  `A`, strided by `ldap`) and both row-major cases are supported for MKL
  compatibility through a stride-generalized kernel over the same dot-product math
  (correctness-first; the non-contiguous inner sweep is not separately
  SIMD-tuned), reusing the existing `BatchView` addressing. `trans = 'N'`
  row-major is itself a contiguous case (rows of `A` are contiguous row-major) and
  is a natural candidate for a future second tuned path, exactly as `potrf`'s
  row-major-upper dual is; it is left on the strided path here to keep the tuned
  surface small, as `trsm` does.

The `uplo` triangle is defined on the mathematical indices `(i,j)` of the
symmetric `C` regardless of `trans` or `layout`; only the strides of the two
`BatchView`s differ between the strided cases.

### 6.4 Padding and SIMD semantics

When `nm` is not a multiple of `V`, `mkl_?gepack_compact` fills the unused slots
of the last pack with identity matrices. The rank-k update needs no special
handling for them: the kernel runs the whole final pack unmasked at full width,
computing some (garbage) rank-k update in the padded lanes, and those lanes are
simply never unpacked or read back. Because every arithmetic step is
unconditional (no pivot, no `larfg`-style mask), the padding cannot produce a
`NaN` that contaminates a real lane -- each lane is independent. This is the
`?syrk` analogue of the `potrf` observation that the identity flows through the
unconditional pivot path unmasked.

### 6.5 No argument checking (Compact convention)

Like MKL's own compact routines -- which "skip error checking for performance
reasons" and make "the user responsible for passing correct parameters"
(Intel MKL, *Numerical Limitations for Compact BLAS and Compact LAPACK
Routines*) -- `cqr_mkl_?syrk_compact` validates nothing. Because `?syrk` has no
`info`, there is not even a dispatch-level status: an unrecognized `format` is a
silent no-op. It is the caller's responsibility to pass valid parameters, and to
validate them beforehand if defensive checking is wanted -- for which the portable
`dsyrk_compact` / `ssyrk_compact` (`cqr_compact.h`) provide LAPACK/BLAS-style
`info = -j` argument validation.

### 6.6 Numerical scope

The rank-k update is a sequence of fused multiply-adds, backward stable to working
precision, matching reference BLAS `?syrk` element for element up to the
accumulation order (the dot-product summation rounds differently from, e.g., a
rank-1 accumulation, but both are backward stable). As with the rest of the
toolkit, no overflow/underflow-safe scaling is attempted, and complex precisions
(`c`/`z`, whose symmetric update is `?herk`) are out of scope. When
`cqr_mkl_?syrk_compact` feeds a Cholesky QR (section 1), the well-known
conditioning caveat of Cholesky QR applies: the orthogonality of the recovered
`Q` degrades like `cond(A)^2 * eps`, so for ill-conditioned `A` a reorthogonaliza-
tion pass (CholeskyQR2) or the QR-based pipeline is preferable. The rank-k update
itself is unaffected -- that is a property of the algorithm it feeds, not of
`?syrk`.

## 7. Testing and Validation Methodology

Matching standard BLAS `?syrk` to working precision is the minimum bar. SIMD,
blocking, and layout handling are internal strategies only: the returned `C` must
satisfy the same numerical invariants as an unbatched `?syrk`. Tolerances are
purely relative to the working precision, scaled by the accumulation length `k`.
All suites below are CTest-registered.

### 7.1 Suite A -- vs dense per-matrix `cblas_?syrk`

`test_cqr_syrk_mkl.cpp` packs a random `A` and a random (non-symmetric) `C` with
the genuine MKL Compact API, runs `cqr_mkl_?syrk_compact`, and compares the
unpacked result against a per-matrix `cblas_?syrk` from the same seed, element for
element over the *whole* `n x n` `C`. Because `?syrk` writes only the `uplo`
triangle, this simultaneously gates the active-triangle result and that the
opposite triangle is left untouched. Run over the full feature matrix
(`precision x layout x uplo x trans`), a spread of `alpha`/`beta` (including
`beta = 0` overwrite and `alpha = 0`), and shapes covering wide/tall factors,
`n = 1`, `k = 1`, `nm = 1`, and padded partial last groups.

### 7.2 Suite B -- vs `mkl_?gemm_compact` (triangle only)

The same packed inputs are run through `cqr_mkl_?syrk_compact` and, independently,
`mkl_?gemm_compact` forming the full product (`(NoTrans, Trans)` for `A A^T`,
`(Trans, NoTrans)` for `A^T A`). Since gemm writes the entire matrix while syrk
writes one triangle, the comparison is restricted to the active `uplo` triangle.
This cross-checks against a second, independent MKL kernel over the full feature
matrix -- the direct analogue of the `trsm`/`geqrf` suites' cross-checks against
their native MKL counterparts.

### 7.3 Suite C -- end-to-end Cholesky QR, no MKL compute kernel

The capstone: for a tall, well-conditioned random `A`, the fully open pipeline

```
cqr_mkl_dsyrk_compact('U','T')   // G = A^T A
cqr_mkl_dpotrf_compact('U')      // G = R^T R
cqr_mkl_dtrsm_compact('R','U','N') // Q = A R^{-1}
```

must recover a valid QR factorization. Gate the reconstruction
`|| Q R - A ||_1 / ||A||_1` at `50 * n * eps` (backward stable, independent of
conditioning) and the orthogonality `|| Q^T Q - I ||` at `1e3 * n * eps` (small
for the tall, well-conditioned inputs used, per the `cond(A)^2` caveat of
section 6.6). This exercises `cqr_mkl_?syrk_compact` closing a complete batched
Cholesky QR with no MKL compute kernel -- the syrk analogue of the `trsm` suite's
end-to-end QR solve.

### 7.4 Portable self-test (no BLAS)

`test_cqr_syrk_compact.cpp` validates the templated kernel directly against a
scalar `?syrk` reference over the full `uplo x trans x layout` matrix, across
precisions, interleave widths, and padded final packs, plus the LAPACK/BLAS-style
argument validation of the portable C API. The reference accumulates the rank-k
product in a **different summation order** (rank-1 outer products, contraction
index outermost) than the kernel's inner-product form, so a bug common to both
cannot pass unseen while a correct kernel still agrees to working precision. All
four `(trans, layout)` combinations are covered so the tuned (`trans='T'`,
column-major) and strided kernels are both exercised. It needs no external
libraries at all; only Suites A-C require an MKL installation (for the Compact
API).

## 8. Implementation Strategy

Modern C++ (C++17) templated on scalar type `T` and interleave width `V`, exposed
through `extern "C"` for the FFI-stable surfaces, reusing the existing
`cqr::detail::pack<T,V>` / `BatchView` GNU-vector machinery.

### 8.1 API boundary

Two C-linkage interfaces wrap the same templated kernel:

* **MKL-style API** (`cqr_mkl_ext.h`): `cqr_mkl_dsyrk_compact` /
  `cqr_mkl_ssyrk_compact` -- taking the MKL enums and `MKL_COMPACT_PACK`, with no
  argument checking (Compact convention). It **dispatches on the
  `MKL_COMPACT_PACK` format enum** (`MKL_COMPACT_SSE`/`AVX`/`AVX512` ->
  `V = 16/32/64 bytes / sizeof(T)`), a compile-time constant per case, exactly as
  `cqr_mkl_?trsm_compact` / `cqr_mkl_?potrf_compact` do.
* **Portable C API** (`cqr_compact.h`): `dsyrk_compact` / `ssyrk_compact` -- an
  MKL-independent surface taking an explicit interleave width `V` and `char`
  selectors, with LAPACK/BLAS-style `info = -j` argument validation.

Both instantiate the kernel on `MKL_INT` (or `int`) so ILP64 dimensions are not
narrowed; the algorithm and its internal routines are described in section 6.

### 8.2 Source layout

| File | Role |
|------|------|
| `src/cqr_syrk_compact.hpp` | Templated SIMD rank-k kernel: `syrk_compact_group` (tuned `trans='T'` column-major), `syrk_compact_group_strided` (general via `BatchView`), and the batch driver `syrk_compact_general` (scalar `T`, width `V`). |
| `src/cqr_syrk_compact_dispatch.cpp` | Portable `?syrk_compact` C entry points (runtime `V` -> compile-time dispatch, `info = -j`). |
| `src/cqr_mkl_syrk.cpp` | Unwraps the MKL enums + dispatches on `MKL_COMPACT_PACK` -> `V`, calls the kernel (drop-in style; no `work`/`info`). |
| `src/test_cqr_syrk_compact.cpp` | Self-contained correctness test vs a scalar `?syrk` reference (no BLAS). |
| `src/test_cqr_syrk_mkl.cpp` | MKL validation: vs `cblas_?syrk`, vs `mkl_?gemm_compact`, and the end-to-end Cholesky QR. |

The `cqr_mkl_?syrk_compact` prototypes are added to `cqr_mkl_ext.h` and the
portable `?syrk_compact` prototypes to `cqr_compact.h`.
