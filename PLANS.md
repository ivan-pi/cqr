# PLANS

Known gaps between the implementation and the design document
(`cqr_mkl_dormqr_compact_design.md`), verified against the source tree.

- **Complex precisions (`cunmqr`/`zunmqr`).** Only real precisions exist; the
  family is real-only and `trans='C'` is folded to `'T'`
  (`src/cqr_mkl_ext.cpp:53-54,82`). Document the real-only scope, or
  add genuine complex specializations.
- **`A` packed column count assumed equal to `k`.** `ncols_a = k` is hardcoded
  (`src/cqr_mkl_ext.cpp:84-87`), correct for square/tall
  (`m >= n` so `k = n`) but mis-addresses groups for wide factorizations
  (`m < n`, packed with `n` columns). Document the precondition or add a
  packed-ncols parameter.
- **`work[0]` not set on the compute path.** Design section 5 requires `work[0]`
  to hold the minimum `lwork` on successful exit; it is only set during the
  `lwork = -1` query (`src/cqr_mkl_ext.cpp:73-76`). Trivial fix.
- **Stress-test matrix (sections 7.3/7.4) absent.** Tests use only
  well-conditioned `frand` + diagonal boost. Missing: the `cond` scaling knob
  (`logspace(0,-cond,n)`), the rank-deficient / near-rank-deficient / banded /
  row-scaled / near-collinear / clustered-scale structures, and the
  geometric-mean benchmark-ranking harness.
- **No install/export.** `CMakeLists.txt` defines no `install()`/package-config
  rules, so the project isn't consumable via `find_package(cqr)`.
