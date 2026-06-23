# cqr - compact-format apply-Q (`cqr_mkl_dormqr_compact`)

A portable SIMD kernel and an Intel MKL-style API for applying the orthogonal
factor `Q` (or `Q^T`) of a **Compact-format** QR factorization
(`mkl_?geqrf_compact`) to a batch of matrices - the routine missing between
`mkl_?geqrf_compact` and the use of its reflectors. See the
[design document](cqr_mkl_dormqr_compact_design.md).

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
* `bench_qr_compact [nmat] [reps]` - throughput of the batched pipeline vs. the
  one-matrix-at-a-time LAPACK path over pools of small matrices (order 20-100),
  reporting a geometric-mean speedup. Accuracy-gated, so it doubles as an
  integration test.

Both are registered with CTest (`example_solve_qr_compact`,
`bench_qr_compact_integration`).

## Layout

| File | Role |
|------|------|
| `src/cqr_compact.hpp` | Templated SIMD kernel `B := op(Q)*B` (scalar `T`, interleave width `V`). |
| `src/cqr_compact_dispatch.{cpp,h}` | Portable C entry points `dormqr_compact` / `sormqr_compact` (runtime `V` -> compile-time dispatch). |
| `src/cqr_mkl_ext.{cpp,h}` | The public API `cqr_mkl_dormqr_compact`: unwraps `MKL_COMPACT_PACK` -> `V` and calls the kernel. |
| `src/test_cqr_compact.cpp` | Self-contained correctness/bench test (no BLAS). |
| `src/test_cqr_mkl_ext.cpp` | MKL-backed validation through the real compact pipeline. |
| `examples/solve_qr_compact.cpp` | Worked batched `AX=B` solve, cross-checked against `LAPACKE_dgels`. |
| `examples/bench_qr_compact.cpp` | Throughput benchmark vs. per-matrix LAPACK. |

## Related work

Batched / compact dense linear algebra for many small matrices:

* [Intel oneMKL Compact BLAS and LAPACK functions](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-2/compact-blas-and-lapack-functions.html) - the compact (interleaved) format this project extends.
* [Arm Performance Libraries interleave-batch functions](https://developer.arm.com/documentation/101004/2507/Interleave-batch-functions/Interleave-batch-introduction?lang=en) - Arm's equivalent interleaved-batch API.
* [Batched BLAS (BBLAS)](https://icl.utk.edu/bblas/) - the proposed standard interface for batched BLAS.
* [Kokkos Kernels batched API](https://kokkos.org/kokkos-kernels/docs/API/batched-index.html) - portable batched kernels.
* [batmat](https://github.com/tttapa/batmat) - batched small-matrix linear algebra.
