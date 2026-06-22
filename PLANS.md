# PLANS

Roadmap of known gaps between the current implementation and the design
document (`ext_mkl_dormqr_compact_design.md`). Each item below has been
verified against the source tree; file/line references point at the relevant
code as of this writing.

## Functional / API

### 1. Complex precisions (`cunmqr`/`zunmqr`) — currently real-only
A "complete `?ormqr`" family normally includes the complex variants
`cunmqr`/`zunmqr`. Only the real precisions are provided
(`ext_mkl_dormqr_compact`, `ext_mkl_sormqr_compact` in
`src/ext_mkl_ormqr_compact.cpp`). `trans='C'` is accepted but folded to `'T'`
(`src/ext_mkl_ormqr_compact.cpp:53-54,82`), which is correct only for real
matrices.

- **Short term:** document that the family is real-only and that `'C'` is
  treated as `'T'`.
- **Long term:** add complex specializations with a genuine conjugate-transpose
  path.

### 2. `A` packed column count is assumed equal to `k`
The dispatcher passes `k` as the A-group column stride
(`ncols_a = k`, `src/ext_mkl_ormqr_compact.cpp:84-87`; addressing uses
`g*ldap*k*V`). This is correct for square/tall factorizations
(`m ≥ n ⇒ k = n`), i.e. the entire solver use-case. But for a *wide*
factorization (`m < n`, so `k = m < n`), the buffer is packed with `n` columns,
so the inter-group stride should be `ldap·n·V`, not `ldap·k·V` — every group
after the first would be mis-addressed. This is a valid `ormqr` call
(`k ≤ m` holds) that the current API cannot express and that is untested.

- **Option A:** document the "A packed with exactly `k` columns" precondition.
- **Option B:** add a packed-ncols parameter so wide factorizations are
  expressible.

### 3. `work[0]` not set on the compute path
Design §5 states `work[0]` holds the minimum required `lwork` on a successful
(non-query) exit. The implementation only sets `work[0]` during the
`lwork = -1` workspace query (`src/ext_mkl_ormqr_compact.cpp:73-76`); on the
compute path it is left untouched. Trivial fix: set `work[0]` on successful
non-query exit as well.

## Testing / Validation

### 4. Stress-test matrix from design §7.3 is absent
This is the biggest gap relative to the documented validation plan. The tests
(`src/test_ormqr_compact.cpp`, `src/test_ext_mkl_ormqr_compact.cpp`,
`examples/solve_qr_compact.cpp`) use only well-conditioned `frand` inputs with a
diagonal boost ("tame cond"). Missing:

- The documented `cond` input-scaling knob (columns scaled by
  `logspace(0, -cond, n)`).
- Structured stress cases: rank-deficient, near-rank-deficient, banded,
  row-scaled, near-collinear, upper-triangular, and clustered-scale inputs.
- The geometric-mean benchmark-ranking harness (§7.4) — currently only an
  ad-hoc micro-bench exists in the portable test.

## Packaging

### 5. No install/export rules
`CMakeLists.txt` builds the libraries and tests but defines no
`install()` / package-config (`*Config.cmake`) rules, so the project is not yet
consumable as an installed dependency. Add `install()` targets,
`GNUInstallDirs`, header installation, and an exported CMake package config so
downstream projects can `find_package(cqr)`.
