# PLANS

## geqrf (`cqr_mkl_dgeqrf_compact`)

The compact QR *factorization* (`cqr_mkl_dgeqrf_compact_design.md`): a portable,
vectorized `mkl_?geqrf_compact`. Status vs. its design document:

- **Implemented (design sections 6-8):** the vectorized unblocked `geqr2` with a
  branch-free masked `larfg` (FP64 + FP32), the portable C API
  `dgeqrf_compact`/`sgeqrf_compact` (LAPACK-style `info=-j` validation) and the
  MKL-style `cqr_mkl_?geqrf_compact` (no checking, scalar `info`, `lwork`
  query). Column-major is the tuned contiguous path; row-major routes through
  the stride-generalized kernel.
- **Validated (design section 7):** a BLAS-free test vs. a scalar `geqr2`
  reference, and an MKL/LAPACK test gating the factorization residual
  (`20 n eps`) and orthogonality (`100 n eps`) against dense
  `LAPACKE_dorgqr`/`dgeqrf`, cross-checking vs. `mkl_dgeqrf_compact` (both
  layouts), closing the `AX=B` solve, and covering rank-deficient / near-collinear
  stress structures. All match LAPACK/MKL to machine precision.
- **Benchmarked (design section 9):** `bench_geqrf_compact` (vs.
  `mkl_dgeqrf_compact` and per-matrix `LAPACKE_dgeqrf`) is built and CTest-gated,
  reporting GFLOP/s and a geometric-mean speedup.
- **Known gaps / scoped out (design section 6.6):** no overflow/underflow-safe
  `dlarfg` rescaling (matters only near `1e+/-150`), no column pivoting, and the
  row-major sweep is correctness-first, not separately SIMD-tuned (mirroring
  `ormqr`). Blocked (`larft`/`larfb`) factorization is intentionally not used at
  the target sizes. Complex precisions are out of scope, as for `ormqr`.
- **Deferred:** a comparison benchmark against the open-source `batmat` project's
  `geqrf` (same interleaved format) is left for a future change.

## potrf (`cqr_mkl_dpotrf_compact`)

The compact Cholesky *factorization* (`cqr_mkl_dpotrf_compact_design.md`): a
portable, vectorized `mkl_?potrf_compact` for symmetric positive-definite
batches. Status vs. its design document:

- **Implemented (design sections 6-8):** the vectorized unblocked `potf2` with a
  branch-free unconditional-`sqrt` pivot (FP64 + FP32) and a `JB = 4`
  register-blocked rank-1 trailing update, the portable C API
  `dpotrf_compact`/`spotrf_compact` (LAPACK-style `info = -j` validation) and the
  MKL-style `cqr_mkl_?potrf_compact` (no checking, scalar `info`, no `work`).
  Column-major lower is the tuned contiguous path; row-major upper folds onto it
  by transpose duality, and the other two `(layout, uplo)` combinations route
  through the stride-generalized kernel. The shared `vsqrt<T,V>` helper now lives
  in `cqr_compact.hpp` alongside `pack`/`BatchView`.
- **Validated (design section 7):** a BLAS-free test vs. a scalar `potf2`
  reference (both `uplo`, all layouts, padded final packs, and a non-SPD
  lane-isolation case gating that a poisoned lane never contaminates its
  siblings), and an MKL/LAPACK test gating the reconstruction residual
  (`20 n eps`), the untouched triangle (bit-for-bit), and -- since the SPD factor
  is unique -- the elementwise factor vs. `LAPACKE_dpotrf` (`20 n eps`),
  cross-checking vs. `mkl_dpotrf_compact` (both layouts, both `uplo`; observed
  bit-exact), and closing the SPD `AX = B` solve (`potrf` + two
  `mkl_dtrsm_compact`). All match LAPACK/MKL to machine precision.
- **Known gaps / scoped out (design section 6.6):** positive-definiteness is
  assumed, not enforced (a non-SPD lane poisons itself with `NaN`/`Inf` instead
  of `info = j`, mirroring MKL's reserved `info`); no overflow/underflow-safe
  scaling; no pivoting; and the strided (column-major upper / row-major lower)
  inner sweep is correctness-first, not separately SIMD-tuned. Blocked
  (`syrk`/`trsm`) factorization is intentionally not used at the target sizes.
  Complex (`c`/`z`) Hermitian variants are out of scope, as for the QR routines.
- **Deferred:** a throughput benchmark (`cqr_mkl_?potrf_compact` vs.
  `mkl_?potrf_compact` vs. per-matrix `LAPACKE_?potrf`), mirroring
  `bench_geqrf_compact`, is left for a future change.

## trsm (`cqr_mkl_dtrsm_compact`)

The compact batched triangular solve (`cqr_mkl_dtrsm_compact_design.md`): a
portable, vectorized `mkl_?trsm_compact`, the step that closes the batched
`AX = B` solve so it needs no MKL compute kernel. Status vs. its design document:

- **Implemented (design sections 2-6, 8.1):** both API surfaces --
  the MKL-style `cqr_mkl_?trsm_compact` (drop-in for `mkl_?trsm_compact`, no
  `work`/`info`, `src/cqr_mkl_trsm.cpp`) and the portable `dtrsm_compact` /
  `strsm_compact` with LAPACK/BLAS-style `info = -j` validation
  (`src/cqr_trsm_compact_dispatch.cpp`) -- backed by the vectorized substitution
  in `src/cqr_trsm_compact.hpp`. The tuned column-major/`side='L'` path is
  templated on the RHS block width and on `uplo/trans/diag` (all compile-time):
  `trsm_dot_block<JB,...>` (register-blocked row-dot of a `JB`-column block),
  `trsm_left_dot_tb` (sweeps the RHS in descending 4/2/1 blocks so the 1-3
  leftover columns still reuse `A`), and `trsm_axpy_col` (contiguous column-axpy
  for the single `op(A)=A` leftover column); `trsm_left_dot` dispatches the
  runtime config to these. A stride-generalized kernel
  (`trsm_compact_group_strided`) handles the other side/layout combinations, all
  driven over the packs by `trsm_compact_general`. Full
  `side x uplo x transa x diag` support in FP64/FP32, `alpha = 0` handled as the
  BLAS `B := 0` fast path.
- **Validated (design section 7):** `test_cqr_trsm_compact` (no BLAS) runs the
  argument-validation gate plus a numerical suite vs a scalar `?trsm` reference
  (itself cross-checked against `cblas_?trsm`) -- the row-dot kernels match it to
  the last bit, the column-axpy kernel to working precision (~1e-16, a different
  but backward-stable accumulation order). `test_cqr_trsm_mkl` cross-checks
  `cqr_mkl_?trsm_compact` vs
  `mkl_?trsm_compact` over the full `layout x side x uplo x transa x diag` matrix
  (agreement ~1e-16) and runs the end-to-end `AX = B` solve
  (`mkl_dgeqrf_compact -> cqr_mkl_dormqr_compact -> cqr_mkl_dtrsm_compact`). Both
  are CTest-registered.
- **Wired into the pipeline:** `examples/solve_qr_compact.cpp`
  and the `bench_qr_compact` benchmark now call `cqr_mkl_dtrsm_compact` in the cqr
  workflow, so the batched solve path uses no MKL compute kernel (MKL is used only
  for pack/unpack).
- **Known gaps / scoped out (design section 6.6):** no singularity check (a zero
  diagonal of a non-unit factor divides to Inf/NaN, as in BLAS `?trsm`), no
  overflow/underflow-safe scaling, and the strided (right-side / row-major) inner
  sweep is correctness-first, not separately SIMD-tuned (mirroring `ormqr`).
  Complex precisions are out of scope.

### trsm performance vs `mkl_?trsm_compact`

Head-to-head timing of `cqr_mkl_dtrsm_compact` against `mkl_?trsm_compact` on
identical packed data, single thread, `L/U/N/N`, AVX-512 (`V=8`), matrix orders
10-148, verified on **both GCC and Clang** (isolated micro-benchmark, not
in-tree). Speedup = MKL time / cqr time, `> 1` means cqr is faster:

- **Single RHS (`nrhs = 1`, the QR-solve `R x = Q^T b`):** ~`1.0x` across the
  whole size range. Solved by the column-axpy 1-tail (`trsm_axpy_col`): the
  row-dot alone fell to ~`0.85x` at `n = 148` because it read `A` across a row
  (stride `ldap`) and prefetched poorly out of cache; the axpy streams `A` down
  columns contiguously, matching MKL (same formulation -- they agree bit-for-bit).
- **`nrhs` a multiple of 4:** ~`1.3-1.5x` (the `JB = 4` row-dot block reuses each
  strided `A` load four times, beating MKL).
- **`nrhs = 2, 3, 5, 6, 7, ...` (mixed / awkward counts):** the 4/2/1 tail
  blocking lifts these to parity-or-better (~`1.0-1.3x`) -- e.g. `nrhs = 2` went
  from ~`0.77x` to ~`1.0-1.2x` and `nrhs = 6` from ~`0.85x` to ~`1.1-1.3x`. The
  compile-time `uplo/trans/diag` specialization also nudged the clean multiples
  up (Clang `nrhs = 8` at `n = 148`: `0.96x -> 1.23x`). One residual dip: Clang
  `nrhs = 2` at `n = 148` (~`0.9x`) -- there a 2-wide strided block trails
  contiguous axpy, but the small/mid-size gains dominate, so `n >= 2` stays on
  the blocked row-dot.

**Note -- this access-pattern issue is specific to `trsm`.** `geqrf` and `ormqr`
apply Householder reflectors, whose inner loops already sweep *down columns*
(`akk[i]`, contiguous), so they have no strided-`A` problem and are untouched.

Remaining `trsm` optimization opportunities (not yet done; performance-only, no
correctness impact):

1. **Small-`n` overhead (`n ~ 10`): ~`0.6-0.9x`.** Fixed per-group cost swamps
   the tiny flop count. The compile-time `uplo/trans/diag` specialization (above)
   removed the runtime config branches and helped, but a dedicated small-`n` code
   path (and/or amortizing per-group setup) would cut it further.
2. **Reciprocal-multiply the diagonal** in the register-blocked paths: compute
   `1/A(k,k)` once and multiply the `JB` columns, instead of `JB` vector
   divisions (`divpd` is ~5-8x a multiply and poorly pipelined). Helps the wider
   blocks; irrelevant at `nrhs = 1`. Costs ~1 ULP vs the current division.
3. **SIMD-tune the strided kernel** (right-side / row-major), currently
   correctness-first.

## Known gaps

Gaps between the `ormqr` implementation and its design document
(`cqr_mkl_dormqr_compact_design.md`), verified against the source tree.

- **Complex precisions (`cunmqr`/`zunmqr`).** Only real precisions exist; the
  family is real-only and `trans='C'` is folded to `'T'`
  (in `src/cqr_mkl_ext.cpp`). Document the real-only scope, or
  add genuine complex specializations.
- **Stress-test matrix (sections 7.3/7.4) absent.** Tests use only
  well-conditioned `frand` + diagonal boost. Missing: the `cond` scaling knob
  (`logspace(0,-cond,n)`), the rank-deficient / near-rank-deficient / banded /
  row-scaled / near-collinear / clustered-scale structures, and the
  geometric-mean benchmark-ranking harness.
- **No install/export.** `CMakeLists.txt` defines no `install()`/package-config
  rules, so the project isn't consumable via `find_package(cqr)`.

## Accordance with the design document

What the implementation provides, mapped to the design document:

- **API (sections 2-5):** `cqr_mkl_dormqr_compact` matches the signature, the
  `MKL_COMPACT_PACK` format abstraction, and the `lwork = -1` workspace query.
  It performs no argument checking and writes a single scalar `info` (0 on
  success). Both are implementation choices favoring performance -- mirroring
  MKL's own compact routines, which skip checks and leave `info` reserved --
  and may change in future.
- **Dispatcher (section 8.1):** unwraps the format to the interleave width `V`
  (SSE/AVX/AVX-512 -> 2/4/8 for FP64, 4/8/16 for FP32) and forwards to the
  templated kernel.
- **Validation (section 7):** `test_cqr_mkl_ext` runs Suite 1 (isolated
  `op(Q)*C` vs dense LAPACK, gate `20*s*eps`) over the full feature matrix --
  `side in {L,R} x layout in {col,row} x trans in {N,T}` -- and Suite 2
  (end-to-end `AX=B`: `mkl_dgeqrf_compact -> cqr_mkl_dormqr_compact ->
  mkl_dtrsm_compact`, gates on forward error and system residual at
  `100*n*eps`) against real MKL.
- **Padding (section 6.4):** padded slots of the last compact pack carry
  identity factorizations (`tau=0`), so applying them is a no-op; exercised by
  the partial-group test cases.

**Feature coverage:** `side in {'L','R'}`, `layout in {MKL_COL_MAJOR,
MKL_ROW_MAJOR}`, `trans in {N,T}` (`C` folds to `T` for the real types) in
FP64/FP32. The tuned contiguous kernel serves the `side='L'`, column-major
solver path; the other three side/layout combinations run through a
stride-generalized kernel (same unblocked `dorm2r` math, correctness-first --
the non-contiguous inner sweep is not yet SIMD-tuned). Real types only
(`d`/`s`); complex (`c`/`z`) is out of scope. The MKL extension does not
validate arguments; the portable `dormqr_compact`/`sormqr_compact` C API does.
