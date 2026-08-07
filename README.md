# cqr - Compact QR extensions

Batched QR for many small matrices, stored in Intel MKL's **Compact**
(interleaved) format. **cqr** provides portable, SIMD-vectorized kernels behind
an Intel MKL-style API:

* **`cqr_mkl_?geqrf_compact`** - the QR *factorization* itself: an open,
  vectorized alternative to `mkl_?geqrf_compact`. On this project's AVX-512 test
  machine it outruns MKL's own compact `geqrf` and per-matrix LAPACK across the
  small-size range. See its [design document](cqr_mkl_dgeqrf_compact_design.md).
* **`cqr_mkl_?ormqr_compact`** - the *apply-Q* step MKL omits: MKL ships
  `mkl_?geqrf_compact` and `mkl_?trsm_compact` but no `?ormqr_compact`, so there
  is no supported way to apply `Q` (or `Q^T`) to a batch. cqr fills that gap. See
  its [design document](cqr_mkl_dormqr_compact_design.md).
* **`cqr_mkl_?potrf_compact`** - the batched **Cholesky factorization**
  (`A = L L^T` / `U^T U`) of symmetric-positive-definite matrices: a portable,
  vectorized alternative to `mkl_?potrf_compact`. Paired with
  `cqr_mkl_?trsm_compact` it factors and solves batched SPD systems. See its
  [design document](cqr_mkl_dpotrf_compact_design.md).
* **`cqr_mkl_?trsm_compact`** - an open drop-in for
  `mkl_?trsm_compact` (the batched triangular solve), so the whole `AX = B`
  pipeline runs with no MKL compute kernel. See its
  [design document](cqr_mkl_dtrsm_compact_design.md).

All routines come in single and double precision. Together they factor and
solve batched systems entirely in the compact format, with no MKL compute kernel.

The kernels are written with GNU vector types (`__attribute__((vector_size))`),
which the compiler lowers to SSE, AVX, or AVX-512 -- one portable source for
every width. That is the project's central SIMD decision.

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

For a performance build, pass host-tuned optimization flags through
`CMAKE_CXX_FLAGS` so the SIMD kernels target the machine's widest vectors (the
same AVX-512 MKL selects at runtime):

```sh
cmake -S . -B build -DBLA_VENDOR=Intel10_64lp_seq -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-O3 -march=native"
```

Useful option: `-DCQR_WITH_MKL=OFF` (portable kernel only, no MKL).

## Examples

* `solve_qr_compact` - a batch of square systems `A_v X_v = B_v` solved end to
  end with the compact pipeline (`mkl_dgeqrf_compact` -> `cqr_mkl_dormqr_compact`
  -> `cqr_mkl_dtrsm_compact`), cross-checked against per-matrix `LAPACKE_dgels`.
* `bench_qr_compact [nmat] [reps]` - throughput of the fully open compact *solve*
  pipeline vs. MKL's batched pipeline and the one-matrix-at-a-time LAPACK path,
  over pools of small matrices (order 10-100), reporting geometric-mean speedups.
  As for `bench_geqrf_compact`, build with host-tuned flags (`-march=native`) for
  a fair comparison against MKL. All paths are checked against the known solution.
* `bench_geqrf_compact [nmat] [reps]` - throughput of the QR *factorization*:
  `cqr_mkl_dgeqrf_compact` vs `mkl_dgeqrf_compact` vs per-matrix
  `LAPACKE_dgeqrf`, across the target square-size range, reporting GFLOP/s and a
  geometric-mean speedup, checked against LAPACK. For a fair comparison, build
  with host-tuned flags (e.g. `-DCMAKE_CXX_FLAGS="-O3 -march=native"`) so the
  compact kernel uses the full vector width, as MKL's runtime dispatch does.
* `bench_potrf_compact [nmat] [reps]` - the Cholesky counterpart: throughput of
  the SPD *factorization* `cqr_mkl_dpotrf_compact` vs `mkl_dpotrf_compact` vs
  per-matrix `LAPACKE_dpotrf`, over the same square-size range (tuned col-major
  lower, `A = L L^T`), reporting GFLOP/s and a geometric-mean speedup, checked
  elementwise against LAPACK (the SPD factor is unique). Same `--size-sweep` /
  `--simdlen` flags and the same `-march=native` caveat as `bench_geqrf_compact`.

All are registered with CTest (`example_solve_qr_compact`,
`bench_qr_compact_integration`, `bench_geqrf_compact_integration`,
`bench_potrf_compact_integration`). The three benchmarks -- what they measure,
how to run them, the flags, and the `-march=native` caveat -- are documented in
detail in [`examples/BENCHMARKS.md`](examples/BENCHMARKS.md).

## Layout

### Public interface

These headers are the project's API - the only files most users need to
include:

| File | Role |
|------|------|
| `src/cqr_mkl_ext.h` | The MKL-style public API: `cqr_mkl_?geqrf_compact` (QR factorization, drop-in for `mkl_?geqrf_compact`), `cqr_mkl_?ormqr_compact` (apply `Q`/`Q^T`, the missing `mkl_?ormqr_compact`), `cqr_mkl_?potrf_compact` (Cholesky, drop-in for `mkl_?potrf_compact`), and `cqr_mkl_?trsm_compact` (triangular solve, drop-in for `mkl_?trsm_compact`). Takes `MKL_COMPACT_PACK` formats. |
| `src/cqr_compact.h` | The portable C API, all eight exported functions: `dgeqrf_compact` / `sgeqrf_compact` (QR factorization), `dormqr_compact` / `sormqr_compact` (apply `Q` / `Q^T`), `dpotrf_compact` / `spotrf_compact` (Cholesky), and `dtrsm_compact` / `strsm_compact` (triangular solve), with an explicit interleave width `V` and no MKL dependency. |

Everything else under `src/` is internal - implementation details and tests,
not part of the supported interface:

| File | Role |
|------|------|
| `src/cqr_geqrf_compact.hpp` | Templated SIMD QR-factorization kernel (vectorized `geqr2`; scalar `T`, interleave width `V`). |
| `src/cqr_potrf_compact.hpp` | Templated SIMD Cholesky-factorization kernel (vectorized `potf2`; scalar `T`, interleave width `V`). |
| `src/cqr_compact.hpp` | Templated SIMD kernel `B := op(Q)*B` (scalar `T`, interleave width `V`); also hosts the shared `pack<T,V>` / `BatchView` / `vsqrt` / `broadcast` machinery. |
| `src/cqr_trsm_compact.hpp` | Templated compact triangular-solve kernels (tuned column-major/left + general strided) and group driver (scalar `T`, interleave width `V`). |
| `src/cqr_geqrf_compact_dispatch.cpp` | Portable geqrf C entry points (runtime `V` -> compile-time dispatch). |
| `src/cqr_potrf_compact_dispatch.cpp` | Portable potrf C entry points (runtime `V` -> compile-time dispatch). |
| `src/cqr_compact_dispatch.cpp` | Portable ormqr C entry points (runtime `V` -> compile-time dispatch). |
| `src/cqr_trsm_compact_dispatch.cpp` | Portable trsm C entry points with LAPACK/BLAS-style `info = -j` validation (runtime `V` -> compile-time dispatch). |
| `src/cqr_mkl_geqrf.cpp` | Unwraps `MKL_COMPACT_PACK` -> `V` and calls the geqrf kernel. |
| `src/cqr_mkl_potrf.cpp` | Dispatches on `MKL_COMPACT_PACK` directly and maps `MKL_UPLO`/`MKL_LAYOUT`, then calls the potrf kernel. |
| `src/cqr_mkl_ext.cpp` | Unwraps `MKL_COMPACT_PACK` -> `V` and calls the ormqr kernel. |
| `src/cqr_mkl_trsm.cpp` | Unwraps the MKL enums + `MKL_COMPACT_PACK` -> `V` and calls the trsm kernel (drop-in for `mkl_?trsm_compact`; no `work`/`info`). |
| `src/cqr_mkl_alloc.h` | Optional RAII buffer helpers (`mkl_alloc_bytes`, `mkl_buffer`) wrapping `mkl_malloc`/`mkl_free`. |
| `src/test_compact_util.hpp` | Shared test helpers (seeded RNG, error metrics, SPD generation, Compact pack/unpack); header-only, no MKL. |
| `src/test_cqr_geqrf_compact.cpp` | Self-contained geqrf correctness test vs a scalar `geqr2` reference (no BLAS). |
| `src/test_cqr_geqrf_mkl.cpp` | MKL + dense-LAPACK validation of `cqr_mkl_dgeqrf_compact` (residual, orthogonality, solve). |
| `src/test_cqr_potrf_compact.cpp` | Self-contained potrf correctness test vs a scalar `potf2` reference (no BLAS). |
| `src/test_cqr_potrf_mkl.cpp` | MKL + dense-LAPACK validation of `cqr_mkl_dpotrf_compact` (residual, untouched triangle, uniqueness, cross-check, solve). |
| `src/test_cqr_compact.cpp` | Self-contained ormqr correctness/bench test (no BLAS). |
| `src/test_cqr_mkl_ext.cpp` | MKL-backed validation through the real compact pipeline. |
| `src/test_cqr_trsm_compact.cpp` | Self-contained trsm test (no BLAS): C API validation + numerical vs a scalar `?trsm` reference. |
| `src/test_cqr_trsm_mkl.cpp` | MKL-backed cross-check of `cqr_mkl_?trsm_compact` vs `mkl_?trsm_compact` + an end-to-end MKL-compute-free solve. |

### Examples

| File | Role |
|------|------|
| `examples/solve_qr_compact.cpp` | Worked batched `AX=B` solve, cross-checked against `LAPACKE_dgels`. |
| `examples/bench_qr_compact.cpp` | Throughput benchmark of the batched *solve* vs. per-matrix LAPACK. |
| `examples/bench_geqrf_compact.cpp` | Throughput benchmark of the QR *factorization* vs. `mkl_dgeqrf_compact` and per-matrix `LAPACKE_dgeqrf`. |
| `examples/bench_potrf_compact.cpp` | Throughput benchmark of the SPD Cholesky *factorization* vs. `mkl_dpotrf_compact` and per-matrix `LAPACKE_dpotrf`. |

## Related work

Batched / compact dense linear algebra for many small matrices:

* [Batched BLAS (BBLAS)](https://icl.utk.edu/bblas/) - the proposed standard interface for batched BLAS.
* [Intel oneMKL Compact BLAS and LAPACK functions](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-2/compact-blas-and-lapack-functions.html) - the compact (interleaved) format this project extends.
* [Arm Performance Libraries interleave-batch functions](https://developer.arm.com/documentation/101004/2507/Interleave-batch-functions/Interleave-batch-introduction?lang=en) - Arm's equivalent interleaved-batch API.
* [Kokkos Kernels batched API](https://kokkos.org/kokkos-kernels/docs/API/batched-index.html) - portable batched kernels.
* [batmat](https://github.com/tttapa/batmat) - batched small-matrix linear algebra.
