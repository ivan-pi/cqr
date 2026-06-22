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
`-DCQR_WITH_MKL=OFF` (build and test only the portable kernel, no MKL).

## Accordance with the design document

* **API (§2–§5):** `ext_mkl_dormqr_compact` is implemented with the exact
  signature, `MKL_COMPACT_PACK` format abstraction, `lwork = -1` workspace
  query, and per-matrix `info[]` reporting (`info[i] = -j` for an illegal
  `j`-th argument).
* **Dispatcher (§8.1):** unwraps the format to the interleave width `V`
  (SSE/AVX/AVX-512 → 2/4/8 for FP64, 4/8/16 for FP32) and dispatches to the
  templated kernel.
* **Validation (§7):** `test_ext_mkl_ormqr_compact` runs Suite 1 (isolated
  `op(Q)·C` vs dense LAPACK, gate `20·s·ε`) over the full feature matrix —
  `side ∈ {L,R} × layout ∈ {col,row} × trans ∈ {N,T}` — and Suite 2 (end-to-end
  `AX=B`: `mkl_dgeqrf_compact → ext_mkl_dormqr_compact → mkl_dtrsm_compact`,
  gates on forward error and system residual at `100·n·ε`) against real MKL.
* **Padding (§6.4):** padded slots of the last compact pack carry identity
  factorizations (`τ=0`), so applying them is a no-op; exercised by the
  partial-group test cases.

**Feature coverage:** `side ∈ {'L','R'}`, `layout ∈ {MKL_COL_MAJOR,
MKL_ROW_MAJOR}`, `trans ∈ {N,T}` (`C` folds to `T` for the real types) in
FP64/FP32. The tuned contiguous kernel serves the `side='L'`, column-major
solver path; the other three side/layout combinations run through a
stride-generalized kernel (same unblocked `dorm2r` math, correctness-first —
the non-contiguous inner sweep is not yet SIMD-tuned). Real types only
(`d`/`s`); complex (`c`/`z`) is out of scope. Unsupported argument values are
reported through `info[]` (never silently miscomputed), per LAPACK convention.
