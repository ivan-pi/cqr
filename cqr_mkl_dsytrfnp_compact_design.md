# API Design Document: `cqr_mkl_dsytrfnp_compact` / `cqr_mkl_dsytrsnp_compact`

> Assisted-by: Claude

## 1. Overview

This extension provides a fast batched **LDL^T factorization without pivoting**
of a set of symmetric `n x n` matrices stored in Intel MKL's Compact
(interleaved-batch) format, together with the **solve** that completes a batched
symmetric linear system from that factor. It follows the API style of
`mkl_?potrf_compact`, but MKL itself ships no compact `sytrf` of any kind, so --
like `cqr_mkl_?ormqr_compact` -- these routines fill a gap in the compact
ecosystem rather than shadowing an MKL routine. The `np` (no-pivoting) suffix
follows MKL's own naming for its unpivoted compact LU, `mkl_?getrfnp_compact`.

The factorization is the square-root-free sibling of `cqr_mkl_?potrf_compact`:
because there is no `sqrt`, **symmetric indefinite matrices factor too**
(negative pivots are legal), which is the point of LDL^T over Cholesky. The
solve companion pairs with it the way LAPACK pairs `?sytrs` with `?sytrf`:

```
cqr_mkl_dsytrfnp_compact('L', A -> (L, D));    // A = L D L^T
cqr_mkl_dsytrsnp_compact('L', A, B);           // B := A^{-1} B = X
                                               //   (L z = B; D w = z; L^T X = w)
```

Both are portable, open implementations built on the project's GNU-vector
`pack<T,V>` machinery, sharing the `MKL_LAYOUT` + `MKL_COMPACT_PACK` API surface
with the existing routines. The primary target is many small-to-medium matrices,
with order in `3..500` and the emphasis on sizes below 170. The tuned path is
column-major, lower triangle.

## 2. Syntax

```c
void cqr_mkl_dsytrfnp_compact (
    MKL_LAYOUT layout, MKL_UPLO uplo, MKL_INT n,
    double * ap, MKL_INT ldap, MKL_INT * info,
    MKL_COMPACT_PACK format, MKL_INT nm
);

void cqr_mkl_dsytrsnp_compact (
    MKL_LAYOUT layout, MKL_UPLO uplo, MKL_INT n, MKL_INT nrhs,
    const double * ap, MKL_INT ldap, double * bp, MKL_INT ldbp,
    MKL_INT * info, MKL_COMPACT_PACK format, MKL_INT nm
);
```

The factorization signature is `mkl_dpotrf_compact`'s (there being no MKL
compact `sytrf` to mirror); the solve is LAPACK `?sytrs` plus the three
arguments MKL's compact routines add (`layout`, `format`, `nm`), with `info` as
a scalar status. Both real precisions are provided (`d`/`s`). Neither routine
needs workspace, so there are no `work`/`lwork` arguments.

## 3. Description

`cqr_mkl_?sytrfnp_compact` forms, for each symmetric `n x n` matrix `A` in the
batch, the factorization

```
A = L D L^T,   if uplo = MKL_LOWER  (L unit lower triangular, D diagonal),
A = U^T D U,   if uplo = MKL_UPPER  (U unit upper triangular, D diagonal),
```

**without pivoting**. On exit the diagonal of each matrix holds `D` and the
strict off-diagonal of the named triangle holds `L` (or `U`); the unit diagonal
of the triangular factor is implied, not stored -- the same storage convention
as an unpivoted LU (`mkl_?getrfnp_compact`), restricted to one triangle. The
strictly-opposite triangle is neither referenced nor modified. `A` must be
symmetric; only the named triangle is read.

**Note the upper-triangle convention.** `MKL_UPPER` computes `A = U^T D U`,
the transpose dual of the lower factorization -- consistent with `?potrf`'s
`A = U^T U` -- and **not** LAPACK `?sytrf`'s `A = U D U^T`. The two conventions
differ in elimination order (top-down vs bottom-up); the dual keeps all four
`(layout, uplo)` cases on one kernel (section 6.4). Since no MKL compact
routine fixes the convention, the choice follows the existing compact Cholesky.

`cqr_mkl_?sytrsnp_compact` then solves `A X = B` for each matrix from that
factor, overwriting the `n x nrhs` RHS block `B` with `X` via three in-place
substitution sweeps:

```
L z = B;    D w = z;    L^T X = w      (MKL_LOWER)
U^T z = B;  D w = z;    U   X = w      (MKL_UPPER)
```

The triangular sweeps are the compact `trsm` with a unit diagonal (which never
reads the stored diagonal, i.e. `D`); the diagonal solve between them is the
one operation `trsm` cannot express (section 6.6).

**Constraint note.** As with all Compact routines, every matrix in the call
shares the same order `n`, leading dimensions, storage `layout`, and `format`.
The batch is processed one *pack* (group of `V` interleaved matrices) at a time;
`V` is derived from `format`. Pack the matrices with `mkl_?gepack_compact`
before, and unpack results with `mkl_?geunpack_compact` after -- or feed the
factor straight to `cqr_mkl_?sytrsnp_compact`, which consumes it in place.

## 4. Input Parameters

Factorization (`cqr_mkl_?sytrfnp_compact`):

* **`layout`** (`MKL_LAYOUT`): the in-memory storage order of each matrix --
  `MKL_COL_MAJOR` (tuned path) or `MKL_ROW_MAJOR`.
* **`uplo`** (`MKL_UPLO`): `MKL_LOWER` (factor/store the lower triangle,
  `A = L D L^T`; tuned path) or `MKL_UPPER` (`A = U^T D U`). The other triangle
  is not referenced.
* **`n`** (`MKL_INT`): the order of each `A` (`n >= 0`).
* **`ap`** (`double *`): the compact buffer of `nm` matrices `A`, packed with
  `mkl_?gepack_compact`. Overwritten in place with `(D, L|U)`.
* **`ldap`** (`MKL_INT`): leading dimension of each matrix within the compact
  buffer, `>= n`.
* **`format`** (`MKL_COMPACT_PACK`): the pack format from
  `mkl_get_format_compact()`; selects the interleave width `V`
  (SSE/AVX/AVX-512 -> 2/4/8 for FP64, 4/8/16 for FP32).
* **`nm`** (`MKL_INT`): total number of matrices in the batch (`nm >= 0`).

Solve (`cqr_mkl_?sytrsnp_compact`), in addition:

* **`nrhs`** (`MKL_INT`): number of right-hand sides (columns of `B`,
  `nrhs >= 0`).
* **`ap`** (`const double *`): the factored compact batch from
  `cqr_mkl_?sytrfnp_compact` (not modified); `uplo` must match the
  factorization call.
* **`bp`** (`double *`): the compact RHS batch `B` (`n x nrhs` per matrix),
  overwritten with `X`.
* **`ldbp`** (`MKL_INT`): leading dimension of each `B` (`>= n` column-major,
  `>= nrhs` row-major).

**Buffer alignment.** Any base alignment is correct. For full speed, align the
bases to the pack width (64 B covers every format) so each SIMD access stays on
one cache line -- worth up to ~40% on small, cache-resident sizes.
`mkl_malloc(bytes, 64)` (the default of this project's `mkl_alloc_bytes`)
already does this.

## 5. Output Parameters

* **`ap`** (factorization): the named triangle is overwritten with the factor --
  `D` on the diagonal, the unit-diagonal `L`/`U` strictly off it, in Compact
  format; the other triangle is left untouched.
* **`bp`** (solve): overwritten with the solution `X`, in Compact format.
* **`info`** (`MKL_INT *`): a single scalar status, `0` on success. Neither
  routine reports a zero pivot the way LAPACK reports `info = j` (section 6.2).
  The one value either can set is dispatch-level: an unrecognized `format`
  selects no kernel and sets `info = -1`.

## 6. Design Considerations & Compatibility

### 6.1 The algorithm: vectorized unblocked LDL^T (square-root-free `potf2`)

The batch is factored with the unblocked right-looking sweep -- `?potf2` with
the `sqrt` pivot replaced by a reciprocal -- executed `V` matrices at a time.
Because Compact format interleaves the `V` matrices so that element `(i,j)` of
all `V` is contiguous, the scalar algorithm lifts verbatim with `double ->`
V-wide vector; like Cholesky (and unlike QR's `larfg`) the math has no
data-dependent branch, so no lane mask is needed. Per pivot column `j` (lower):

```
d       = A(j,j)                                 // pivot, stays in place as D(j,j)
invd    = 1 / d
A(i,j) *= invd             for i > j             // scale -> L(i,j)
A(i,jj)-= A(i,j)*(A(jj,j)*d) for jj > j, i >= jj // rank-1 update: L(i,j) D(j,j) L(jj,j)
```

Only the lower trapezoid is touched, so the strictly-upper triangle passes
through untouched. The trailing update -- the `O(n^3)` bulk -- is
register-blocked `JB = 4` trailing columns at a time so each pivot-column entry
`A(i,j)` load is reused across four columns, exactly as in `potrf`/`geqrf`; the
extra multiply by `d` folds into the per-column scalar `w`, one op per block
column. Compared with Cholesky at the same size the flop count is the same to
`O(n^2)` (the `n` square roots are replaced by `n` reciprocals).

### 6.2 Zero pivots: what "no pivoting" does and does not tolerate

Two situations must be distinguished:

* **A zero on the *input* diagonal is not necessarily a problem.** The pivots
  are the *updated* (Schur-complement) diagonal entries, not the original ones:
  `A = [[1, 2], [2, 0]]` has `A(1,1) = 0` yet factors cleanly
  (`d = 1, -4`). Likewise **negative pivots are fine** -- there is no `sqrt` --
  so genuinely indefinite matrices are in scope; Cholesky would fail on them.
* **A zero *pivot* is fatal, by construction.** The unpivoted factorization
  exists iff every leading principal minor is nonsingular; a zero pivot
  (`d_j = 0`, equivalently a singular leading `j x j` minor, e.g.
  `A = [[0, 1], [1, 0]]` at the first step) meets `invd = 1/0 = Inf` and the
  lane's factor fills with `Inf`/`NaN`. Near-zero pivots are the gradual version
  of the same cliff: without pivoting the element growth is unbounded and no
  backward-stability guarantee exists (section 6.7).

Following the Compact convention (and `cqr_mkl_?potrf_compact`'s treatment of
non-SPD input), this is **graceful garbage-in, garbage-out**: no test, no
safeguarded path, no `info = j` early exit -- the per-lane branch is precisely
what does not vectorize across a pack. A poisoned lane never contaminates its
pack siblings, which are computed by the same unmasked SIMD instructions; the
caller detects the poison by inspecting the unpacked diagonal.

The cure for zero/near-zero pivots -- Bunch-Kaufman `1x1`/`2x2` pivoting
(LAPACK `?sytrf`), or even symmetric diagonal pivoting -- requires per-matrix
comparisons and row/column interchanges that diverge across lanes, destroying
the interleaved execution. That is why "no pivoting" is the natural batch
variant, and it is the same trade MKL itself makes in the compact API: its
compact LU is `mkl_?getrfnp_compact`, unpivoted, with no pivoted alternative.

### 6.3 The upper convention: `U^T D U`, by transpose duality

For `MKL_UPPER` the routine computes `A = U^T D U` -- elimination proceeding
top-down, `U`'s row `j` produced at step `j` -- rather than LAPACK `?sytrf`'s
bottom-up `A = U D U^T`. `A = U^T D U` is exactly the lower algorithm applied
through transposed indexing (`L = U^T`, same `D`, same pivot order, and `A` is
symmetric), which is what lets all four `(layout, uplo)` combinations reuse one
kernel pair (section 6.4); `?potrf` sets the same precedent with `A = U^T U`.
Callers migrating from LAPACK `?sytrf`'s upper factor should note the factors
are *not* elementwise comparable (different pivot sequence); the lower
convention `A = L D L^T` matches LAPACK's orientation (though LAPACK's is
pivoted, so factors still differ elementwise unless no pivoting occurred).

### 6.4 Layouts and triangles: one tuned contiguous kernel, one strided

As for `potrf`, the four `(layout, uplo)` combinations pair up by transpose
duality into two contiguous cases and two strided ones:

* **Contiguous (tuned).** Column-major + `MKL_LOWER`: a factor column is
  contiguous in the compact buffer, so the column scaling and the rank-1 update
  walk contiguous packs. Its transpose dual, row-major + `MKL_UPPER`, is the
  same memory access pattern with `(i,j)` roles swapped and routes to the same
  tuned kernel.
* **Strided.** Column-major + `MKL_UPPER` and row-major + `MKL_LOWER` walk the
  factor across the non-contiguous axis; they are supported through a
  stride-generalized kernel over the same math (correctness-first; not
  separately SIMD-tuned), reusing the `BatchView` addressing.

### 6.5 Padding and SIMD semantics

When `nm` is not a multiple of `V`, `mkl_?gepack_compact` fills the unused slots
of the last pack with identity matrices. The unpivoted LDL^T of the identity is
`L = I, D = I` (all pivots `1`, no fill), so the padded lanes compute a
mathematical no-op and the final pack runs unmasked at full width. The same
holds through the solve: padded factor slots are identities, so the padded
columns of `B` pass through the sweeps unchanged (and are never read back).

### 6.6 The solve: two unit `trsm` sweeps around a diagonal solve

`cqr_mkl_?sytrsnp_compact` completes the batched solve from the factor. Its two
triangular sweeps are the existing compact `trsm` kernels invoked with a *unit*
diagonal -- `?trsm` with `diag = 'U'` never reads the stored diagonal, which is
exactly what lets `D` live there -- so column-major runs on `trsm`'s tuned
side-left path. Between them sits the diagonal solve `B(i,:) *= 1/D(i)`, the
one step `trsm` cannot express: an `O(n*nrhs)` row scaling (one reciprocal per
factor row, reused across the `nrhs` columns), negligible next to the
`O(n^2*nrhs)` sweeps and therefore implemented once, stride-generally, for all
layouts. No workspace is needed and `B` is overwritten in place.

A caller who prefers to compose the solve manually can equally run
`cqr_mkl_?trsm_compact(diag = MKL_UNIT)` twice and scale the middle step --
`sytrsnp` packages exactly that composition (and the packaged diagonal solve is
the operation the public `trsm` API does not offer).

### 6.7 Numerical scope

* **Existence, not stability.** For any symmetric `A` whose leading principal
  minors are all nonsingular the factorization exists and is unique, and the
  routine computes it. Unlike Cholesky on SPD input, *unpivoted* LDL^T on an
  indefinite matrix carries **no general backward-stability guarantee**: element
  growth is bounded only by the pivot ratios, so accuracy degrades as pivots
  approach zero. It is the caller's contract that the batch is
  factorable-without-pivoting and reasonably conditioned that way -- e.g.
  quasi-definite matrices (as in interior-point KKT systems), diagonally
  dominant symmetric matrices, or matrices whitened by a prior transformation.
  For general indefinite input needing rank-revealing robustness, per-matrix
  LAPACK `?sytrf` (Bunch-Kaufman) is the right tool, not a batch kernel.
* **Zero/near-zero pivots** poison their lane (section 6.2); no check, no
  `info = j`.
* **Matrices scaled near underflow/overflow.** Following Intel's stated
  numerical limitations for Compact BLAS/LAPACK routines, no safe scaling is
  provided; such inputs are out of scope.
* **Real precisions only.** Single and double (`s`/`d`); complex Hermitian
  (`c`/`z`) variants are out of scope, as for the other routines. `D` is
  diagonal with `1x1` pivots only -- there are no `2x2` pivot blocks, which
  exist in LAPACK's `?sytrf` precisely to compensate for the pivoting this
  routine omits.

## 7. Testing and Validation Methodology

Standard dense LAPACK has no unpivoted LDL^T to diff against (`?sytrf` is
Bunch-Kaufman-pivoted, so its factors differ elementwise), so validation rests
on the algebraic contract, a scalar reference, and MKL's own unpivoted compact
LU:

### 7.1 Suite 1 -- Factorization invariants

For each `(uplo, layout)`, a batch of random symmetric *indefinite* matrices
with a known-good unpivoted factorization -- built as `A = L D L^T` from random
unit-lower `L` (small off-diagonals, so element growth is mild) and mixed-sign
`D` bounded away from zero -- is factored, unpacked, and checked per matrix:

* **Reconstruction residual.** Rebuild `L D L^T` (`U^T D U` for upper) from the
  unpacked factor (unit diagonal implied) and gate
  `|| L D L^T - A ||_1 / ||A||_1 <= 20 * n * eps`.
* **Untouched triangle.** The strictly-opposite triangle of the compact buffer
  is bit-for-bit unchanged from the input.

### 7.2 Suite 2 -- Cross-check vs `mkl_dgetrfnp_compact`

Unpivoted LU of a symmetric matrix satisfies `A = L * (D L^T)`: it shares the
unit-lower `L`, and its `U` carries `D` on the diagonal. The same batch is
factored by `cqr_mkl_dsytrfnp_compact` (col-major, lower) and by MKL's
`mkl_dgetrfnp_compact`, and the strict lower triangles and diagonals are
compared elementwise at a small fixed tolerance (`1e-9`) -- two independent
implementations agreeing on the same pivots (observed agreement is at the
`1e-15` level).

### 7.3 Suite 3 -- End-to-end indefinite solve `A X = B`

`B = A X` for known `X` on indefinite batches, then `cqr_mkl_dsytrfnp_compact
-> cqr_mkl_dsytrsnp_compact` must recover `X`, over both `uplo`, both layouts,
and padded partial groups. The system residual `A Xhat - B` is gated at
`100 * n * eps` (what the backward-stable sweeps control); the forward error
`Xhat - X` additionally carries `cond(A)` and is gated with conditioning
headroom (`500 * n * eps` on the tamely generated batches).

### 7.4 Portable self-test

A self-contained test (no BLAS) validates the templated kernels directly
against a scalar reference across `(T, V)` combinations, both `uplo`, both
layouts, and padded final packs, plus:

* the **end-to-end portable solve** (`?sytrfnp_compact` + `?sytrsnp_compact`),
* the **zero-diagonal** semantics of section 6.2 (a handcrafted matrix with
  `A(1,1) = 0` but nonsingular leading minors must factor cleanly and match
  the reference exactly),
* **zero-pivot lane isolation** (a lane with `A(0,0) = 0, A(0,1) != 0` must
  poison itself with `Inf`/`NaN` while its pack siblings stay finite and
  bit-comparable to the reference, with `info = 0`),
* LAPACK-style `info = -j` argument validation of both portable C APIs.

## 8. Implementation Strategy

Modern C++ (C++17) templated on scalar type `T` and interleave width `V`,
exposed through `extern "C"` for the FFI-stable surfaces, reusing the
`cqr::detail::pack<T,V>` machinery, the `BatchView` addressing, and (for the
solve) the `trsm_compact_general` kernels.

### 8.1 API boundary

* **MKL-style API** (`cqr_mkl_ext.h`, the primary surface):
  `cqr_mkl_?sytrfnp_compact` / `cqr_mkl_?sytrsnp_compact`, unwrapping
  `MKL_COMPACT_PACK -> V` and `MKL_UPLO`/`MKL_LAYOUT`, instantiated on `MKL_INT`
  so ILP64 dimensions are not narrowed.
* **Portable C API** (`cqr_compact.h`): `?sytrfnp_compact` / `?sytrsnp_compact`,
  taking `char layout`/`char uplo`, an explicit interleave width `V`, and no MKL
  dependency, with LAPACK-style `info = -j` argument validation.
* **Templated kernels**: `sytrfnp_compact_group<T,V>` (tuned contiguous) and
  `sytrfnp_compact_group_strided<T,V>` (general, via `BatchView`), driven over
  all packs by `sytrfnp_compact_general<T,V>`; `sytrsnp_compact_general<T,V>`
  composes two `trsm_compact_general` sweeps around the strided diagonal solve.

### 8.2 Source layout

| File | Role |
|------|------|
| `src/cqr_sytrfnp_compact.hpp` | Templated SIMD unpivoted-LDL^T kernel (scalar `T`, width `V`). |
| `src/cqr_sytrsnp_compact.hpp` | Templated solve driver (unit `trsm` sweeps + diagonal solve). |
| `src/cqr_sytrfnp_compact_dispatch.cpp` | Portable `?sytrfnp_compact` C entry points (runtime `V` dispatch, `info = -j`). |
| `src/cqr_sytrsnp_compact_dispatch.cpp` | Portable `?sytrsnp_compact` C entry points. |
| `src/cqr_mkl_sytrfnp.cpp` | MKL-style factorization adapter (`MKL_COMPACT_PACK` -> `V`). |
| `src/cqr_mkl_sytrsnp.cpp` | MKL-style solve adapter. |
| `src/test_cqr_sytrfnp_compact.cpp` | Self-contained correctness test vs a scalar reference (no BLAS). |
| `src/test_cqr_sytrfnp_mkl.cpp` | MKL validation (invariants, `getrfnp` cross-check, indefinite solve). |

The `cqr_mkl_?sytrfnp_compact` / `cqr_mkl_?sytrsnp_compact` prototypes are added
to `cqr_mkl_ext.h` and the portable prototypes to `cqr_compact.h`.
