# PLANS

Status of each routine against its design document (`docs/`), plus the open
items. All four routines ship both API surfaces -- the MKL-style
`cqr_mkl_?*_compact` (no argument checking, scalar `info`) and the portable
`?*_compact` C API (LAPACK-style `info = -j` validation) -- in FP64 and FP32,
over one `BatchView`-based kernel each that covers every layout (and, for
`ormqr`/`trsm`, every side). Complex precisions, pivoting, and overflow/
underflow-safe scaling are out of scope for all of them.

## geqrf (`cqr_mkl_dgeqrf_compact`)

- **Implemented (design 6-8):** vectorized unblocked `geqr2` with a branch-free
  masked `larfg`; the trailing update is `ormqr`'s `larf`. Column-major is the
  contiguous case (unit row stride); row-major is the same kernel with the
  strides swapped.
- **Validated (design 7):** a BLAS-free test vs a scalar `geqr2`, and an
  MKL/LAPACK test gating the factorization residual (`20 n eps`) and
  orthogonality (`100 n eps`) against dense `LAPACKE_dorgqr`/`dgeqrf`,
  cross-checking vs `mkl_dgeqrf_compact` (both layouts), closing the `AX = B`
  solve, and covering rank-deficient / near-collinear structures.
- **Benchmarked:** `bench_geqrf_compact` (see `examples/BENCHMARKS.md`).
- **Scoped out (design 6.6):** no `dlarfg` rescaling near `1e+/-150`, no column
  pivoting; blocked (`larft`/`larfb`) factorization is deliberately not used at
  the target sizes.
- **Open: one-pass solve on the fused system `[A | B]`.** Factoring the
  `m x (n + nrhs)` matrix `[A | B]` with `geqrf` applies every reflector to the
  `B` block as it is built, so `Q^T B` comes out of the factorization and the
  separate `ormqr` pass disappears. The current API only partly supports this:
  `geqrf` always builds `min(m, ncols)` reflectors, which is exactly `n` for a
  square `A` (correct, one pass) but `min(m, n + nrhs) > n` for a tall `A`
  (extra reflectors over the `B` block -- harmless for the least-squares
  solution, wasted work); and `trsm` cannot address the `R` and `Q^T B` blocks
  inside the fused buffer, because it derives the group stride from its own
  `ldap*s` / `ldbp*n`, not from the fused column count `n + nrhs`, so it only
  works while the batch is a single group (`nm <= V`). Closing this needs either
  a reflector-count argument on `geqrf` plus explicit group strides on `trsm`,
  or a fused driver (`?gels`-style: per group, factor `[A | B]` to `n`
  reflectors and back-substitute the `B` block in place) built on the existing
  view-based kernels.
- **Deferred:** a benchmark against the open-source `batmat` `geqrf` (same
  interleaved format).

## ormqr (`cqr_mkl_dormqr_compact`)

- **Implemented (design 2-6, 8.1):** vectorized unblocked `dorm2r`;
  `side in {L, R}`, both layouts, `trans in {N, T}` (`C` folds to `T`). The
  reflector sweep is the shared `larf`, register-blocked four slices at a time;
  `side = 'R'` is the same kernel over the transposed view of `C`.
- **Validated (design 7):** a BLAS-free test vs a scalar `dorm2r` (including a
  column-pivoted-QR + back-permutation solve), and an MKL test running
  `op(Q) C` vs dense `LAPACKE_dormqr` over the full `layout x side x trans`
  matrix (gate `20 s eps`) plus the end-to-end `mkl_dgeqrf_compact ->
  cqr_mkl_dormqr_compact -> mkl_dtrsm_compact` solve (`100 n eps`).
- **Padding (design 6.4):** padded slots carry identity factorizations
  (`tau = 0`), so applying them is a no-op.
- **Known gaps:** the stress-test structures of design 7.3/7.4 (the `cond`
  scaling knob, banded / row-scaled / clustered-scale inputs) are only partly
  covered -- `geqrf`'s suite has the rank-deficient and near-collinear cases.

## potrf (`cqr_mkl_dpotrf_compact`)

- **Implemented (design 6-8):** vectorized unblocked `potf2` with an
  unconditional `sqrt` pivot and a `JB = 4` register-blocked rank-1 trailing
  update. The four `(layout, uplo)` cases are one kernel over transposed views
  (design 6.3).
- **Validated (design 7):** a BLAS-free test vs a scalar `potf2` (both `uplo`,
  both layouts, padded groups, and a non-SPD lane-isolation case), and an
  MKL/LAPACK test gating the reconstruction residual (`20 n eps`), the untouched
  triangle (bit-for-bit), the elementwise factor vs `LAPACKE_dpotrf`
  (`20 n eps`; the SPD factor is unique), the cross-check vs `mkl_dpotrf_compact`
  (observed bit-exact), and the SPD `AX = B` solve.
- **Benchmarked:** `bench_potrf_compact`.
- **Scoped out (design 6.6):** positive-definiteness is assumed (a non-SPD lane
  poisons itself with `NaN`/`Inf` instead of `info = j`); blocked
  (`syrk`/`trsm`) factorization is not used at the target sizes.

## trsm (`cqr_mkl_dtrsm_compact`)

- **Implemented (design 2-6, 8.1):** full `side x uplo x transa x diag`,
  `alpha = 0` as the BLAS `B := 0` fast path. Column-major `side = 'L'` is the
  tuned path (4/2/1 register-blocked row-dot, contiguous column-axpy for the
  single `op(A) = A` leftover column); the other side/layout combinations use
  the strided kernel.
- **Validated (design 7):** a BLAS-free test vs a scalar `?trsm` gating the
  forward error and the solve's own residual `||op(A) X - alpha B||`, and an
  MKL test cross-checking vs `mkl_?trsm_compact` over the full feature matrix
  plus the end-to-end MKL-compute-free solve.
- **Performance vs `mkl_?trsm_compact`** (single thread, AVX-512, orders
  10-148): `nrhs = 1` ~`1.0x`, `nrhs` a multiple of 4 ~`1.3-1.5x`, mixed counts
  ~`1.0-1.3x`. Remaining opportunities: the small-`n` (~10) per-group overhead
  (~`0.6-0.9x`), reciprocal-multiplying the diagonal in the blocked paths, and
  tuning the strided kernel.
- **Scoped out:** no singularity check (a zero non-unit diagonal divides to
  `Inf`/`NaN`, as in BLAS).

## Project-wide

- **Precision coverage.** Every suite is templated on the scalar type and runs
  in FP64 and FP32; the MKL-backed ones reach MKL and LAPACK through the
  `cqr_mkl<T>` / `mkl<T>` / `lapack<T>` dispatch structs of
  `tests/test_mkl_util.hpp`. The FP32 cross-checks agree with
  `mkl_s*_compact` to ~1e-6 (potrf bit-exact), gated at 1e-4.
- **Threading.** Each routine's group loop is an OpenMP `parallel for`
  (static schedule, at most one thread per group) gated on two or more groups
  and a per-call work estimate above the measured fork/join break-even
  (`2e5` flops), so it stays serial for small calls and inside a caller's own
  parallel region unless nested parallelism is enabled (`OMP_NUM_THREADS=8,2`).
  The factorization benchmarks' cqr paths hand the whole pool to one call; the
  MKL and LAPACK reference paths keep an outer OpenMP loop (sequential MKL is
  not threaded). The solve benchmark keeps its pipeline per group in the
  caller's loop: whole-pool geqrf/ormqr/trsm calls stream the pool three times
  and measured 15-55% slower than the cache-resident per-group pipeline. A
  fused per-group solve driver (the `?gels`-style entry above) is the way to
  get library-side threading for the whole solve without that penalty.
- **No install/export.** `CMakeLists.txt` defines no `install()`/package-config
  rules, so the project is not consumable via `find_package(cqr)`.
- **Alignment contract.** Compact buffers are correct at any `T` alignment on
  GCC and clang alike now that the kernels' view is templated on `(T, V)` rather
  than on the pack type (issue #34); pack-width alignment remains a performance
  recommendation only.
