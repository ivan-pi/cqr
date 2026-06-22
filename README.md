# cqr — compact-format apply-Q (`ext_mkl_dormqr_compact`)

A portable kernel and an Intel MKL–style public API for applying the
orthogonal factor `Q` (or `Qᵀ`) of a **Compact-format** QR factorization
(`mkl_?geqrf_compact`) to a batch of general matrices — the routine missing
between `mkl_?geqrf_compact` and the application of its reflectors. See the
design document [`ext_mkl_dormqr_compact_design.md`](ext_mkl_dormqr_compact_design.md).

## Layout

| File | Role |
|------|------|
| `src/ormqr_compact.hpp` | Templated SIMD kernel: `B := op(Q)·B`, templated on scalar `T` and interleave width `V` (design §8.2/§8.3, tier-2 GNU vector types). |
| `src/ormqr_compact_dispatch.cpp` / `.h` | Portable C entry points `dormqr_compact` / `sormqr_compact` (runtime `V` → compile-time dispatch). |
| `src/ext_mkl_ormqr_compact.cpp` / `.h` | **The design-document public API** `ext_mkl_dormqr_compact` (§2): a C-linkage dispatcher that unwraps `MKL_COMPACT_PACK` → `V` and calls the kernel (§8.1). |
| `src/test_ormqr_compact.cpp` | Self-contained correctness/bench test (no BLAS). |
| `src/test_ext_mkl_ormqr_compact.cpp` | MKL-backed validation through the real compact pipeline (design §7). |
| `src/xcheck_armpl_ormqr.cpp` | Cross-check vs ArmPL interleave-batch (real ArmPL or `src/armpl_stub/`). |
| `examples/solve_qr_dense.cpp` | Worked dense `AX=B` QR solve (`dgeqrf`→`dormqr`→`dtrsm`) cross-checked against `LAPACKE_dgels`; the single-matrix analogue of the compact pipeline. |

## Build

The compact API is an Intel MKL extension: the `*_compact` symbols are reached
through the BLAS link line, so the BLAS is selected with CMake's standard
`BLA_VENDOR` mechanism. Configuring against a non-Intel BLAS stops with a clear
fatal error (only MKL provides the compact API).

```sh
cmake -S . -B build -DBLA_VENDOR=Intel10_64lp_seq -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Useful options: `-DCQR_ENABLE_NATIVE=ON` (host-tuned codegen),
`-DCQR_WITH_MKL=OFF` (build only the portable kernel + ArmPL cross-check),
`-DCQR_BUILD_XCHECK=OFF`.

## Example: a full QR solve in plain LAPACK

[`examples/solve_qr_dense.cpp`](examples/solve_qr_dense.cpp) shows the complete
`AX = B` QR solve as the explicit three-call LAPACK sequence — the dense,
single-matrix analogue of the Compact-format batch pipeline:

```
dgeqrf   A -> (H, R, tau)      A = Q R
dormqr   B <- Qᵀ B             multiply both sides by Qᵀ:  R X = Qᵀ B
dtrsm    R X = (Qᵀ B)          back-substitute:            X = R⁻¹(Qᵀ B)
```

It cross-checks this manual path against the one-call `LAPACKE_dgels` "naive
forward" driver (which performs exactly this QR solve when `m == n`), confirming
both recover the known exact solution to machine precision. The `dormqr` step is
the dense counterpart of `ext_mkl_dormqr_compact`, the routine missing from the
MKL Compact API. Built (with MKL) as the `solve_qr_dense` target and registered
as the `example_solve_qr_dense` CTest:

```sh
./build/solve_qr_dense
```

## Accordance with the design document

* **API (§2–§5):** `ext_mkl_dormqr_compact` is implemented with the exact
  signature, `MKL_COMPACT_PACK` format abstraction, `lwork = -1` workspace
  query, and per-matrix `info[]` reporting (`info[i] = -j` for an illegal
  `j`-th argument).
* **Dispatcher (§8.1):** unwraps the format to the interleave width `V`
  (SSE/AVX/AVX-512 → 2/4/8 for FP64, 4/8/16 for FP32) and dispatches to the
  templated kernel.
* **Validation (§7):** `test_ext_mkl_ormqr_compact` runs Suite 1 (isolated
  `Qᵀ B` vs dense LAPACK, gate `20·n·ε`) and Suite 2 (end-to-end `AX=B`:
  `mkl_dgeqrf_compact → ext_mkl_dormqr_compact → mkl_dtrsm_compact`, gates on
  forward error and system residual at `100·n·ε`) against real MKL.
* **Padding (§6.4):** padded slots of the last compact pack carry identity
  factorizations (`τ=0`), so applying them is a no-op; exercised by the
  partial-group test cases.

**Phase-1 scope / limitations (honest):** the kernel implements `side='L'`,
`layout=MKL_COL_MAJOR`, `trans ∈ {N,T}` for FP64/FP32 — the Compact-format
solver case the document validates. `side='R'` and `MKL_ROW_MAJOR` are reported
as an illegal argument through `info[]` (never silently miscomputed) pending a
future phase, consistent with LAPACK error conventions.
