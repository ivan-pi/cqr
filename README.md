# cqr - Compact QR extensions

Batched QR for many small matrices, stored in Intel MKL's **Compact**
(interleaved) format. **cqr** provides portable, SIMD-vectorized kernels behind
an Intel MKL-style API:

* **`cqr_mkl_?geqrf_compact`** - the QR *factorization* itself: an open,
  vectorized alternative to `mkl_?geqrf_compact`, built on GNU vector types and
  specialized for SSE/AVX/AVX-512. On this project's AVX-512 test machine it
  outruns MKL's own compact `geqrf` and per-matrix LAPACK across the small-size
  range. See its [design document](cqr_mkl_dgeqrf_compact_design.md).
* **`cqr_mkl_?ormqr_compact`** - the *apply-Q* step MKL omits: MKL ships
  `mkl_?geqrf_compact` and `mkl_?trsm_compact` but no `?ormqr_compact`, so there
  is no supported way to apply `Q` (or `Q^T`) to a batch. cqr fills that gap. See
  its [design document](cqr_mkl_dormqr_compact_design.md).

Together (`?geqrf` -> `?ormqr` -> `?trsm`) they form an all-open Compact-format
QR pipeline. FP64 is the focus; FP32 is provided for symmetry.

## Prerequisites

* **CMake >= 3.18**, a **C++17 compiler** (GCC/Clang) and a build tool (Make/Ninja).
* **Intel MKL** - provides the Compact-format extension (`mkl_compact.h`,
  `mkl_*geqrf_compact`, ...) that this project builds on. Any MKL works:
  * oneAPI MKL - `source /opt/intel/oneapi/setvars.sh` (sets `MKLROOT`), or
  * Debian/Ubuntu - `sudo apt-get install libmkl-dev` (headers in
    `/usr/include/mkl`, LP64 libs in the default library path).

## Build

The `*_compact` symbols are reached through the BLAS link line, so the BLAS is
selected with CMake's standard `BLA_VENDOR` mechanism (only MKL provides the
compact API; other vendors stop with a fatal error).

```sh
cmake -S . -B build -DBLA_VENDOR=Intel10_64lp_seq -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Useful options: `-DCQR_ENABLE_NATIVE=ON` (host-tuned codegen),
`-DCQR_WITH_MKL=OFF` (portable kernel only, no MKL).

## Examples

* `solve_qr_compact` - a batch of square systems `A_v X_v = B_v` solved end to
  end with the compact pipeline (`mkl_dgeqrf_compact` -> `cqr_mkl_dormqr_compact`
  -> `mkl_dtrsm_compact`), cross-checked against per-matrix `LAPACKE_dgels`.
* `bench_qr_compact [nmat] [reps]` - throughput of the batched *solve* pipeline
  vs. the one-matrix-at-a-time LAPACK path over pools of small matrices (order
  10-100), reporting a geometric-mean speedup. Accuracy-gated.
* `bench_geqrf_compact [nmat] [reps]` - throughput of the *factorization*:
  `cqr_mkl_dgeqrf_compact` vs `mkl_dgeqrf_compact` vs per-matrix
  `LAPACKE_dgeqrf`, across the target square-size range, reporting GFLOP/s and a
  geometric-mean speedup. Accuracy-gated against LAPACK. Build with
  `-DCQR_ENABLE_NATIVE=ON` for a fair comparison (so the compact kernel uses the
  full vector width, as MKL's runtime dispatch does).

All are registered with CTest (`example_solve_qr_compact`,
`bench_qr_compact_integration`, `bench_geqrf_compact_integration`).

## Layout

### Public interface

These headers are the project's API - the only files most users need to
include:

| File | Role |
|------|------|
| `src/cqr_mkl_ext.h` | The MKL-style public API: `cqr_mkl_?geqrf_compact` (QR factorization, drop-in for `mkl_?geqrf_compact`) and `cqr_mkl_?ormqr_compact` (apply `Q`/`Q^T`, the missing `mkl_?ormqr_compact`). Takes `MKL_COMPACT_PACK` formats. |
| `src/cqr_geqrf_compact.h` | The portable C API `dgeqrf_compact` / `sgeqrf_compact`: the QR factorization with an explicit interleave width `V` and no MKL dependency. |
| `src/cqr_compact.h` | The portable C API `dormqr_compact` / `sormqr_compact`: the apply-`Q` operation from the left (`B := op(Q)*B`), explicit `V`, no MKL dependency. |

Everything else under `src/` is internal - implementation details and tests,
not part of the supported interface:

| File | Role |
|------|------|
| `src/cqr_geqrf_compact.hpp` | Templated SIMD QR-factorization kernel (vectorized `geqr2`; scalar `T`, interleave width `V`). |
| `src/cqr_compact.hpp` | Templated SIMD kernel `B := op(Q)*B` (scalar `T`, interleave width `V`). |
| `src/cqr_geqrf_compact_dispatch.cpp` | Portable geqrf C entry points (runtime `V` -> compile-time dispatch). |
| `src/cqr_compact_dispatch.cpp` | Portable ormqr C entry points (runtime `V` -> compile-time dispatch). |
| `src/cqr_mkl_geqrf.cpp` | Unwraps `MKL_COMPACT_PACK` -> `V` and calls the geqrf kernel. |
| `src/cqr_mkl_ext.cpp` | Unwraps `MKL_COMPACT_PACK` -> `V` and calls the ormqr kernel. |
| `src/cqr_mkl_alloc.h` | Optional RAII buffer helpers (`mkl_alloc_bytes`, `mkl_buffer`) wrapping `mkl_malloc`/`mkl_free`. |
| `src/test_cqr_geqrf_compact.cpp` | Self-contained geqrf correctness test vs a scalar `geqr2` reference (no BLAS). |
| `src/test_cqr_geqrf_mkl.cpp` | MKL + dense-LAPACK validation of `cqr_mkl_dgeqrf_compact` (residual, orthogonality, solve). |
| `src/test_cqr_compact.cpp` | Self-contained ormqr correctness/bench test (no BLAS). |
| `src/test_cqr_mkl_ext.cpp` | MKL-backed validation through the real compact pipeline. |

### Examples

| File | Role |
|------|------|
| `examples/solve_qr_compact.cpp` | Worked batched `AX=B` solve, cross-checked against `LAPACKE_dgels`. |
| `examples/bench_qr_compact.cpp` | Throughput benchmark of the batched *solve* vs. per-matrix LAPACK. |
| `examples/bench_geqrf_compact.cpp` | Throughput benchmark of the *factorization* vs. `mkl_dgeqrf_compact` and per-matrix `LAPACKE_dgeqrf`. |

## Related work

Batched / compact dense linear algebra for many small matrices:

* [Batched BLAS (BBLAS)](https://icl.utk.edu/bblas/) - the proposed standard interface for batched BLAS.
* [Intel oneMKL Compact BLAS and LAPACK functions](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-2/compact-blas-and-lapack-functions.html) - the compact (interleaved) format this project extends.
* [Arm Performance Libraries interleave-batch functions](https://developer.arm.com/documentation/101004/2507/Interleave-batch-functions/Interleave-batch-introduction?lang=en) - Arm's equivalent interleaved-batch API.
* [Kokkos Kernels batched API](https://kokkos.org/kokkos-kernels/docs/API/batched-index.html) - portable batched kernels.
* [batmat](https://github.com/tttapa/batmat) - batched small-matrix linear algebra.
