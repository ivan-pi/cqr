# cqr - Compact QR extensions

Batched QR for many small matrices, stored in Intel MKL's **Compact**
(interleaved) format. **cqr** provides portable, SIMD-vectorized kernels behind
an Intel MKL-style API:

* **`cqr_mkl_?geqrf_compact`** - the QR *factorization* itself: an open,
  vectorized alternative to `mkl_?geqrf_compact`. On this project's AVX-512 test
  machine it outruns MKL's own compact `geqrf` and per-matrix LAPACK across the
  small-size range. See its [design document](docs/cqr_mkl_dgeqrf_compact_design.md).
* **`cqr_mkl_?ormqr_compact`** - the *apply-Q* step MKL omits: MKL ships
  `mkl_?geqrf_compact` and `mkl_?trsm_compact` but no `?ormqr_compact`, so there
  is no supported way to apply `Q` (or `Q^T`) to a batch. cqr fills that gap. See
  its [design document](docs/cqr_mkl_dormqr_compact_design.md).
* **`cqr_mkl_?potrf_compact`** - the batched **Cholesky factorization**
  (`A = L L^T` / `U^T U`) of symmetric-positive-definite matrices: a portable,
  vectorized alternative to `mkl_?potrf_compact`. Paired with
  `cqr_mkl_?trsm_compact` it factors and solves batched SPD systems. See its
  [design document](docs/cqr_mkl_dpotrf_compact_design.md).
* **`cqr_mkl_?trsm_compact`** - an open drop-in for
  `mkl_?trsm_compact` (the batched triangular solve), so the whole `AX = B`
  pipeline runs with no MKL compute kernel. See its
  [design document](docs/cqr_mkl_dtrsm_compact_design.md).
* **`cqr_mkl_?gels_compact`** - the batched **least-squares / minimum-norm
  solve** `op(A) X = B` in one call: LAPACK `?gels` for the compact format
  (square, over- and underdetermined systems, `A` or `A^T`), running the
  factorization, the apply-`Q` step (fused into the factorization) and the
  triangular solve per group of `V` matrices while they are cache-resident, so
  the library's own threading covers the whole solve. See its
  [design document](docs/cqr_mkl_dgels_compact_design.md).

All routines come in single and double precision. Together they factor and
solve batched systems entirely in the compact format, with no MKL compute kernel
-- as the `geqrf -> ormqr -> trsm` chain, or as the one-call `gels`.

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

Useful options: `-DCQR_WITH_MKL=OFF` (portable kernel only, no MKL) and
`-DCQR_WITH_OPENMP=OFF` (single-threaded routines; see below).

## Threading

Every routine threads its loop over the *groups* of `V` interleaved matrices
with OpenMP (static schedule over a team of at most one thread per group; the
groups are independent and equal-sized), but only when the call has at least two
groups and enough work to pay for the fork/join -- about `2e5` flops, the
measured break-even (`-DCQR_OMP_MIN_FLOPS=...` overrides it). Because the
thread count is OpenMP's team size *at the current nesting level*, the routines
compose with a caller's own parallel loop: called from inside it they run
serially by default (no competing thread pools), and the standard per-level
thread list enables nested splitting when wanted, e.g.

```sh
OMP_NUM_THREADS=8,2 OMP_MAX_ACTIVE_LEVELS=2 ./my_app   # 8 outer x 2 inner threads
```

Results are independent of the thread count.

## Examples

* `solve_qr_compact` - a batch of square systems `A_v X_v = B_v` solved end to
  end with the compact pipeline (`mkl_dgeqrf_compact` -> `cqr_mkl_dormqr_compact`
  -> `cqr_mkl_dtrsm_compact`) and with the one-call `cqr_mkl_dgels_compact`,
  both cross-checked against per-matrix `LAPACKE_dgels`.
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

```
include/   public headers
src/       kernels and the two adapter sources
tests/     portable (no BLAS) and MKL-backed test suites
examples/  worked solve + benchmarks (see examples/BENCHMARKS.md)
docs/      one design document per routine
```

### Public interface

The headers under `include/` are the project's API, the only files users need:

| File | Role |
|------|------|
| `include/cqr_mkl_ext.h` | The MKL-style API: `cqr_mkl_?geqrf_compact`, `cqr_mkl_?ormqr_compact`, `cqr_mkl_?potrf_compact`, `cqr_mkl_?trsm_compact`, taking `MKL_COMPACT_PACK` formats. Also the C++ helpers `vlen_for_format` / `format_for_vlen` / `compact_format_name`. |
| `include/cqr_compact.h` | The portable C API: `?geqrf_compact`, `?ormqr_compact`, `?potrf_compact`, `?trsm_compact` (`d`/`s`), with an explicit interleave width `V`, LAPACK-style `info = -j` validation, and no MKL dependency. |
| `include/cqr_mkl_alloc.h` | Optional RAII buffer helpers (`mkl_alloc_bytes`, `mkl_buffer`) wrapping `mkl_malloc`/`mkl_free`. |

### Internals

| File | Role |
|------|------|
| `src/cqr_compact_common.hpp` | The `pack<T,V>` SIMD element, the `BatchView` strided group view every kernel addresses its operands through, `vsqrt`/`broadcast`, and the runtime-`V` dispatch helper `for_vlen`. |
| `src/cqr_geqrf_compact.hpp` | QR factorization kernel: vectorized `geqr2` with a branch-free `larfg`, reusing ormqr's `larf` for the trailing update. |
| `src/cqr_ormqr_compact.hpp` | Apply-Q kernel (vectorized `dorm2r`) and the shared one-reflector update `larf`. |
| `src/cqr_potrf_compact.hpp` | Cholesky kernel (vectorized `potf2`); the four `(layout, uplo)` cases are one kernel over transposed views. |
| `src/cqr_trsm_compact.hpp` | Triangular-solve kernels: the tuned column-major `side='L'` row-dot path and the general strided kernel. |
| `src/cqr_compact.cpp` | The portable C API: argument validation and `V` dispatch for all eight entry points. |
| `src/cqr_mkl_ext.cpp` | The MKL-style API: MKL enum / `MKL_COMPACT_PACK` unwrapping for all eight entry points. |
| `tests/test_compact_util.hpp` | Shared test helpers: RNG, error metrics, input generation, scalar reference kernels, Compact pack/unpack, and `compact<T>` (the portable C API dispatched on the scalar type); header-only, no MKL. |
| `tests/test_mkl_util.hpp` | Scalar-type dispatch for the MKL-backed suites: `cqr_mkl<T>` (routines under test), `mkl<T>` (MKL's compact API and kernels), `lapack<T>` (LAPACKE/CBLAS references). |
| `tests/test_cqr_*_compact.cpp` | Portable self-contained suites (no BLAS): each kernel vs its scalar reference, plus C API validation, in FP64 and FP32. |
| `tests/test_cqr_*_mkl.cpp` | MKL + dense-LAPACK validation, templated on the scalar type and run in FP64 and FP32: invariants vs LAPACK, cross-checks vs MKL's compact kernels, end-to-end solves. |
| `examples/bench_util.hpp` | The benchmarks' shared harness (timing, aligned storage, command line). |

## Contributing

The C++ is formatted with clang-format and linted with clang-tidy, both driven
by [pre-commit](https://pre-commit.com/) (`pip install pre-commit && pre-commit
install`); CI checks the same hooks. [AGENTS.md](AGENTS.md) has the details,
including the clang-configured tree clang-tidy needs.

## Related work

Batched / compact dense linear algebra for many small matrices:

* [Batched BLAS (BBLAS)](https://icl.utk.edu/bblas/) - the proposed standard interface for batched BLAS.
* [Intel oneMKL Compact BLAS and LAPACK functions](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-2/compact-blas-and-lapack-functions.html) - the compact (interleaved) format this project extends.
* [Arm Performance Libraries interleave-batch functions](https://developer.arm.com/documentation/101004/2507/Interleave-batch-functions/Interleave-batch-introduction?lang=en) - Arm's equivalent interleaved-batch API.
* [Kokkos Kernels batched API](https://kokkos.org/kokkos-kernels/docs/API/batched-index.html) - portable batched kernels.
* [batmat](https://github.com/tttapa/batmat) - batched small-matrix linear algebra.
