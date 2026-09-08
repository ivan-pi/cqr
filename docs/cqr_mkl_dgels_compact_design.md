# API Design Document: `cqr_mkl_dgels_compact`

> Assisted-by: Claude:claude-fable-5

## 1. Overview

This extension provides a batched **least-squares / minimum-norm solve** for a
set of full-rank systems `op(A) X = B` whose matrices are stored in Intel MKL's
Compact (interleaved-batch) format: the compact form of LAPACK `?gels`. MKL ships
no `mkl_?gels_compact`; this routine is the one-call equivalent of the three-step
compact pipeline the rest of the toolkit provides,

```
cqr_mkl_dgeqrf_compact(A -> H, tau);          // A = Q R
cqr_mkl_dormqr_compact('L','T', H, tau, B);   // B := Q^T B
cqr_mkl_dtrsm_compact ('L','U','N','N', R, B) // B := R^{-1} Q^T B = X
```

generalized from square systems to over- and underdetermined ones (any `m x n`
shape, `A` or `A^T`), and run per *group* of `V` interleaved matrices from
factorization to solution while the group's buffers are cache-resident. The
three-step chain is three whole-batch calls that stream the batch three times;
`bench_qr_compact` measured that 15-55% slower than keeping a group's pipeline
together, which is why its batched paths drive the chain group by group from the
caller's loop. `cqr_mkl_?gels_compact` does that inside the library, so the
library's own threading over groups (`for_each_group`) covers the whole solve.

The primary target is many small-to-medium matrices, both dimensions in `3..500`
with the emphasis below 170, matching the rest of the toolkit; the tuned path is
column-major.

## 2. Syntax

```c
void cqr_mkl_dgels_compact (
    MKL_LAYOUT layout, char trans,
    MKL_INT m, MKL_INT n, MKL_INT nrhs,
    double * ap, MKL_INT ldap,
    double * bp, MKL_INT ldbp,
    double * work, MKL_INT lwork, MKL_INT * info,
    MKL_COMPACT_PACK format, MKL_INT nm
);
```

The argument list is LAPACK
[`?gels`](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-2/gels.html)
(`trans, m, n, nrhs, a, lda, b, ldb, work, lwork, info`) with MKL's compact
arguments added exactly as `cqr_mkl_?ormqr_compact` adds them to `?ormqr`:
`layout` in front, `format` and `nm` at the back. Both real precisions are
provided: `cqr_mkl_dgels_compact` (double) and `cqr_mkl_sgels_compact` (single).

## 3. Description

For each `m x n` matrix `A` in the batch and its right-hand sides `B`, the
routine solves one of the four problems `?gels` solves, assuming `op(A)` has full
rank:

1. `trans = 'N'`, `m >= n`: the **least-squares** solution of the overdetermined
   `A X = B`, `min || B - A X ||_F`.
2. `trans = 'N'`, `m < n`: the **minimum-norm** solution of the underdetermined
   `A X = B`.
3. `trans = 'T'`, `m >= n`: the minimum-norm solution of the underdetermined
   `A^T X = B`.
4. `trans = 'T'`, `m < n`: the least-squares solution of the overdetermined
   `A^T X = B`.

`B` is `max(m, n) x nrhs` per matrix, as `?gels` declares it (`ldb >= max(m, n)`):
on entry its first `(rows of op(A))` rows hold the right-hand sides, on exit its
first `(columns of op(A))` rows hold `X`. In the least-squares cases the
remaining rows hold the residual: the sum of squares of rows `n..m-1` (case 1) or
`m..n-1` (case 4) of a column is that column's residual sum of squares, exactly
as `?gels` reports it. On exit `ap` holds the QR factorization of `A` (`m >= n`,
`?geqrf` storage) or its LQ factorization (`m < n`, `?gelqf` storage), and `work`
the reflector scalars `tau`, in compact format.

**Constraint note.** As with all Compact routines, every matrix in the call
shares the same `trans`, dimensions (`m`, `n`, `nrhs`), leading dimensions
(`ldap`, `ldbp`), storage `layout`, and `format`. The batch is processed one
*pack* (group of `V` interleaved matrices) at a time; `V` is derived from
`format`.

## 4. Input Parameters

* **`layout`** (`MKL_LAYOUT`): `MKL_COL_MAJOR` (tuned path) or `MKL_ROW_MAJOR`.
* **`trans`** (`char`): `'N'` solves `A X = B`; `'T'` (or `'C'`, the same for the
  real types) solves `A^T X = B`.
* **`m`**, **`n`** (`MKL_INT`): rows and columns of each `A` (`>= 0`).
* **`nrhs`** (`MKL_INT`): columns of each `B` and `X` (`>= 0`).
* **`ap`** (`double *`): the compact buffer of `nm` matrices `A`, packed with
  `mkl_?gepack_compact`. Overwritten with the factorization (section 5). Any
  alignment is correct; a pack-aligned base (64 B) keeps the SIMD sweeps off
  cache-line splits, as for every routine here.
* **`ldap`** (`MKL_INT`): leading dimension of each `A` within the compact
  buffer, `>= m` (column-major) or `>= n` (row-major).
* **`bp`** (`double *`): the compact buffer of `nm` matrices `B`, each
  `max(m, n) x nrhs`, packed with `mkl_?gepack_compact(layout, max(m,n), nrhs,
  ...)`; sized with `mkl_?get_size_compact(max(m,n), nrhs, format, nm)`. Rows
  beyond `(rows of op(A))` need not be set on entry.
* **`ldbp`** (`MKL_INT`): leading dimension of each `B` within the compact
  buffer, `>= max(m, n)` (column-major) or `>= nrhs` (row-major).
* **`work`, `lwork`** (`double *`, `MKL_INT`): workspace, `lwork >=
  max(1, min(m, n) * V * ceil(nm / V))` -- the size in scalars of a compact
  `tau` buffer for the batch, `mkl_?get_size_compact(min(m,n), 1, format, nm) /
  sizeof(double)`. With `lwork = -1` the call is a workspace query: `work[0]`
  receives the required `lwork` and nothing else is touched. Unlike the other
  routines here, whose queries return `1`, `?gels` does use `work`: it is the
  `tau` scratch of the factorization, one slot per group so the groups can run
  in parallel (section 6.5). Size it from *this routine's* query.
* **`format`** (`MKL_COMPACT_PACK`): the pack format from
  `mkl_get_format_compact()`; selects the interleave width `V`
  (SSE/AVX/AVX-512 -> 2/4/8 for FP64, 4/8/16 for FP32).
* **`nm`** (`MKL_INT`): total number of matrices in the batch.

## 5. Output Parameters

* **`bp`**: the solution `X` in the first `(columns of op(A))` rows of each `B`;
  in the least-squares cases the residual in the remaining rows (section 3).
* **`ap`**: the factorization of `A` -- QR (`m >= n`) or LQ (`m < n`) -- in
  the LAPACK `?geqrf` / `?gelqf` storage convention: `R` on and above (`L` on
  and below) the diagonal, the Householder vectors below (to the right of) it.
* **`work`**: the `min(m, n)` reflector scalars `tau` per matrix, in compact
  format, one group after another. `(ap, work)` is therefore the `(H, tau)`
  that `cqr_mkl_?ormqr_compact` accepts, so further right-hand sides can be
  solved without refactoring (for `m < n`, the LQ reflectors are the QR
  reflectors of `A^T`, i.e. of the same buffer in the *other* layout with the
  same `ldap`).
* **`work[0]`**: the required `lwork` on a workspace query.
* **`info`** (`MKL_INT *`): a single scalar status, `0` on success (MKL leaves
  the compact `info` reserved). The routine performs no argument checking
  (section 6.6) and does not detect rank deficiency (no `info > 0`; section
  6.7); the one failure it reports is an unrecognized `format`, which selects no
  kernel and sets `info = -1`.

## 6. Design Considerations & Compatibility

### 6.1 Four cases, one tall QR over a view

`?gels` factors `A` by QR when `m >= n` and by LQ when `m < n`, and then handles
`trans` with the appropriate combination of `Q` application and triangular
solve. In the compact kernels this collapses to a single code path, because the
LQ factorization of `A` *is* the QR factorization of `A^T`, and a `BatchView`
transposes for free (its strides swap). Let `F` be the *tall* orientation of `A`:

```
F = A     if m >= n       F is p x q,  p = max(m, n),  q = min(m, n)
F = A^T   if m <  n
```

In the compact buffer `F` is `A`'s view, transposed when `m < n`, and
`geqrf_compact_group` over that view produces exactly `?gelqf`'s storage: `L`
on and below `A`'s diagonal, the reflector rows to its right, the same `tau`.
With `F = Q [R; 0]` (`Q` is `p x p`, `R` is `q x q` upper triangular), the four
cases become two:

| case | system | solved as |
|------|--------|-----------|
| `m >= n, 'N'` and `m < n, 'T'` | `F X = B` (overdetermined) | `B := Q^T B`; `R X = B(0:q)`; rows `q..p-1` of `B` are the residual |
| `m >= n, 'T'` and `m < n, 'N'` | `F^T X = B` (underdetermined) | `R^T Y = B(0:q)`; `B(q:p) := 0`; `B := Q B` (`X = Q [Y; 0]`, the minimum-norm solution) |

Every step is an existing kernel over views: `geqrf_compact_group` for the
factorization, `ormqr_compact_group` (the backward sweep) for `Q B`, and the
triangular solve for `R X = B` / `R^T Y = B` -- the tuned column-major
`trsm_left_dot` when both `F` and `B` have unit row stride (column-major, `m >=
n`), the strided `trsm_compact_group_strided` otherwise. `?gels` itself also
scales `A` and `B` when their norms are near overflow or underflow; that
rescaling is out of scope here, as it is for every routine in the toolkit
(section 6.7).

### 6.2 The fused `[F | B]` reduction

In the overdetermined case `Q^T B` is not a separate sweep. `geqrf_compact_group`
takes an optional panel: every reflector `H(kk)`, right after it is built from
column `kk` of `F`, is applied to the trailing columns of `F` *and* to all
columns of `B` -- the QR factorization of the fused matrix `[F | B]`, truncated
to `F`'s `q` reflectors. This is exactly what `ormqr('L','T')` computes (the
same reflectors in the same, ascending order, so the arithmetic is identical
and the result matches the separate sweep to rounding), but each reflector is
loaded once for both panels, while it is still in cache. It closes the "one-pass
solve on the fused system `[A | B]`" item of `PLANS.md`, without a
reflector-count argument on `geqrf` or explicit group strides on `trsm`: the
panel is a second view, so `B` keeps its own buffer and leading dimension.

The underdetermined case cannot fuse: `Q` is applied *after* the triangular
solve, and in descending order, so the factorization must be complete first.
It runs the three steps in sequence -- still per group, still cache-resident.

What the fusion is worth depends on `nrhs`: the factorization is `O(p q^2)`
and the apply-`Q^T` sweep it absorbs is `O(p q nrhs)`. `bench_qr_compact`, with
its single right-hand side, measures the one-call `gels` at parity with the
three-step chain driven group by group (`1.00x` geometric mean over
`n = 10..100`, 4 threads, AVX-512) -- the fused sweep is a small fraction of
the work there. The gain grows with `nrhs`; at `nrhs = 1` the routine's value is
the interface, the rectangular cases, and threading the whole solve inside the
library.

### 6.3 One call per batch, threaded over groups

The driver is a single `for_each_group` over the batch; the body runs the whole
solve of one group. The threading gate is the toolkit's (`README`, "Threading"):
a static-schedule OpenMP loop on at most one thread per group, active only for
two or more groups and enough work, with the per-group estimate the sum of the
three steps' flops (`2 p q^2 + 4 p q nrhs + q^2 nrhs`, times `V`). This is the
"fused per-group solve driver" `PLANS.md` asked for: the library threads the
whole solve, and no group is streamed through separate passes.

### 6.4 Layouts

Column-major is the tuned path: `F`'s columns are contiguous packs, the fused
reduction and the apply-`Q` sweep are `larf`'s register-blocked kernel, and the
triangular solve is `trsm`'s 4/2/1-blocked row-dot. Row-major, and the
transposed-view (`m < n`) cases in either layout, run the same code through
strided views (correctness-first, as for `geqrf`, `ormqr` and `trsm`).

### 6.5 Workspace

`?gels` needs `min(m, n)` scalars of scratch per matrix for `tau` (LAPACK's
`?gels` keeps them in `work` too). Because the groups run in parallel, each group
gets its own slot: `lwork >= min(m, n) * V * ceil(nm / V)`, one compact `tau`
buffer for the batch. The routine does not allocate; it never has (no routine
here does), and a per-call allocation inside a threaded loop would be the wrong
place to start. The scratch is not thrown away: on exit it *is* the `tau` of the
factorization left in `ap` (section 5).

### 6.6 Padding, SIMD semantics, and no argument checking

When `nm` is not a multiple of `V`, `mkl_?gepack_compact` fills the unused slots
of the last pack with identity matrices, of `A` and of `B` alike. An identity
`A` factors to `R = I`, `tau = 0` (no reflectors, so the fused reduction and the
apply-`Q` sweep are no-ops in those lanes), and the triangular solve with `R = I`
divides by `1` -- so the kernel runs unmasked across the whole final pack at full
width without a `NaN` or a corrupted real lane. As with every compact routine
here, and with MKL's own (which "skip error checking for performance reasons"),
nothing is validated: an unrecognized `format` is the only failure reported
(`info = -1`). The portable `?gels_compact` of `cqr_compact.h` is the checked
surface (section 8.1).

### 6.7 Numerical scope

The solve is backward stable for a full-rank, well-conditioned `op(A)`, exactly
as the unblocked `?gels` path is. `?gels` returns `info = i > 0` when the `i`-th
diagonal element of `R` is exactly zero, i.e. `op(A)` is detectably rank
deficient; that test does not vectorize across a pack, and the compact `info` is
a scalar, so it is not performed: a zero diagonal of `R` divides through to
`Inf`/`NaN` in that lane, as in `?trsm`. A *nearly* rank-deficient `op(A)`
produces the same inaccurate solution it would from `?gels` (no pivoting: use a
rank-revealing method for such problems). The overflow/underflow rescaling of
`?gels` (`?lascl` on `A` and `B` when their norms are outside `[smlnum,
bignum]`) and of `?larfg` is omitted, as throughout the toolkit; complex
precisions are out of scope.

## 7. Testing and Validation Methodology

Matching LAPACK `?gels` to working precision is the bar. Fusion, views, and the
group-at-a-time driver are internal strategies only: the returned `X` must
satisfy the same invariants as an unbatched `?gels`. All suites are
CTest-registered and run in FP64 and FP32.

### 7.1 Suite 1 -- Solution vs dense `LAPACKE_?gels`

For every `(layout, trans)` over square, tall and wide `A` -- so all four cases
of section 3 run in both layouts -- with random `A` (diagonal-boosted, so
`op(A)` is well conditioned) and random `B`, the batch is solved by
`cqr_mkl_?gels_compact` and, per matrix, by `LAPACKE_?gels` on the same input.
Gated: the forward error of `X` against LAPACK's (`100 * max(m,n) * eps`,
relative to `||X||_1`); the defining property, formed independently -- in the
least-squares cases the residual sums of squares in the trailing rows of `B`
against `||B - op(A) X||^2` (the `?gels` contract), in the minimum-norm cases
the residual `op(A) X - B`; the workspace query; and the factorization left in
`ap` and `work`, elementwise against `LAPACKE_?geqrf` (`m >= n`) or
`LAPACKE_?gelqf` (`m < n`), which pins the `?gelqf` storage convention of the
transposed-view case.

### 7.2 Suite 2 -- Cross-check vs the three-step compact pipeline

On the same packed square batch with `B = A X` for a known `X`,
`cqr_mkl_?gels_compact` and `mkl_?geqrf_compact -> cqr_mkl_?ormqr_compact ->
mkl_?trsm_compact` must agree: the factorization in `ap` and the `tau` (`gels`'s
`work` against `geqrf`'s `taup`) elementwise at the cross-check tolerance, and
the two solutions with each other and with `X` at the solve gate (`100 * n *
eps`). This confirms the one-call routine is the pipeline it replaces -- same
storage, same `tau`, same solution.

### 7.3 Portable self-test

A self-contained test (no BLAS) validates the templated kernel across `(T, V,
layout, trans, shape)`, padded final packs, single right-hand sides and the
smallest sizes, against a scalar `ref_gels` -- the same unblocked steps
(`geqr2` of the tall orientation, `orm2r`, back substitution) in scalar form, so
`X`, the factorization and `tau` are gated elementwise at `~eps` -- and, formed
without the reference, the properties that define the two solutions: the normal
equations `op(A)^T (B - op(A) X) = 0` and the residual-sum-of-squares rows for
least squares; `op(A) X = B` and agreement with the minimum-norm solution built
the other way, `X = op(A)^T Z` with `(op(A) op(A)^T) Z = B`, for minimum norm.
It also covers the LAPACK-style argument validation of the portable C API, the
workspace query, and `min(m, n) = 0` (`B := 0`, the `?gels` quick return).

## 8. Implementation Strategy

### 8.1 API boundary

* **MKL-style API** (`cqr_mkl_ext.h`, the primary surface):
  `cqr_mkl_dgels_compact` / `cqr_mkl_sgels_compact`, unwrapping
  `MKL_COMPACT_PACK -> V` and instantiated on `MKL_INT`. No argument checking;
  `lwork = -1` answers the workspace query.
* **Portable C API** (`cqr_compact.h`): `dgels_compact` / `sgels_compact`, with
  an explicit interleave width `V`, no MKL dependency, and LAPACK-style
  `info = -j` validation (`-11` for an `lwork` below the requirement; `V` and
  `nm` are checked first since the requirement depends on them).
* **Templated kernel** (`src/cqr_gels_compact.hpp`): the per-group
  `gels_compact_group<T,V>` over the tall view `F` and the right-hand-side view
  `B`, driven over all groups by `gels_compact<T,V>` (one `for_each_group`),
  which both C adapters call. It owns no arithmetic of its own: the
  factorization (with the fused panel) is `geqrf_compact_group`, the apply-`Q`
  is `ormqr_compact_group`, the triangular step `trsm`'s group kernels;
  `gels_lwork` is the workspace rule the two adapters and the tests share.

### 8.2 Relationship to `?gels` and to the toolkit

The routine is `?gels` minus its scaling and rank test (section 6.7), over the
unblocked kernels the toolkit already validates -- so the numerical behaviour is
that of `?geqrf`/`?gelqf` + `?ormqr`/`?ormlq` + `?trsm` for these sizes, and the
`(ap, work)` it leaves behind interoperate with `cqr_mkl_?ormqr_compact` and
`cqr_mkl_?trsm_compact` for anything the one call does not cover (further
right-hand sides, `Q` applied on the right).
