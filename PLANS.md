# PLANS

## Known gaps

Gaps between the implementation and the design document
(`cqr_mkl_dormqr_compact_design.md`), verified against the source tree.

- **Complex precisions (`cunmqr`/`zunmqr`).** Only real precisions exist; the
  family is real-only and `trans='C'` is folded to `'T'`
  (in `src/cqr_mkl_ext.cpp`). Document the real-only scope, or
  add genuine complex specializations.
- **`A` packed column count assumed equal to `k`.** `ncols_a = k` is hardcoded
  (in `src/cqr_mkl_ext.cpp`), correct for square/tall
  (`m >= n` so `k = n`) but mis-addresses groups for wide factorizations
  (`m < n`, packed with `n` columns). Document the precondition or add a
  packed-ncols parameter.
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
