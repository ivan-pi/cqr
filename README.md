# cqr - compact-format apply-Q (`ext_mkl_dormqr_compact`)

A portable kernel and an Intel MKL-style public API for applying the
orthogonal factor `Q` (or `Q^T`) of a **Compact-format** QR factorization
(`mkl_?geqrf_compact`) to a batch of general matrices - the routine missing
between `mkl_?geqrf_compact` and the application of its reflectors. See the
design document [`ext_mkl_dormqr_compact_design.md`](ext_mkl_dormqr_compact_design.md).

## Layout

| File | Role |
|------|------|
| `src/ormqr_compact.hpp` | Templated SIMD kernel: `B := op(Q)*B`, templated on scalar `T` and interleave width `V` (design sections 8.2/8.3, tier-2 GNU vector types). |
| `src/ormqr_compact_dispatch.cpp` / `.h` | Portable C entry points `dormqr_compact` / `sormqr_compact` (runtime `V` -> compile-time dispatch). |
| `src/ext_mkl_ormqr_compact.cpp` / `.h` | **The design-document public API** `ext_mkl_dormqr_compact` (section 2): a C-linkage dispatcher that unwraps `MKL_COMPACT_PACK` -> `V` and calls the kernel (section 8.1). |
| `src/test_ormqr_compact.cpp` | Self-contained correctness/bench test (no BLAS). |
| `src/test_ext_mkl_ormqr_compact.cpp` | MKL-backed validation through the real compact pipeline (design section 7). |
| `examples/solve_qr_compact.cpp` | Worked Compact-format batch `AX=B` solve (`mkl_dgeqrf_compact` -> `ext_mkl_dormqr_compact` -> `mkl_dtrsm_compact`) cross-checked against the naive per-matrix `LAPACKE_dgels`. |
| `examples/bench_qr_compact.cpp` | Throughput benchmark over a pool of small matrices (order 20-100): compact batched pipeline vs per-matrix LAPACK (`LAPACKE_dgeqrf`/`dormqr` + `cblas_dtrsm`), OpenMP over the pool, geometric-mean speedup. |

## Prerequisites

* **CMake >= 3.18**, a **C++17 compiler** (GCC/Clang) and a build tool (Make/Ninja).
* **Intel MKL** - provides the Compact-format extension (`mkl_compact.h`,
  `mkl_*geqrf_compact`, ...) that this project builds on. Any MKL works:
  * oneAPI MKL - `source /opt/intel/oneapi/setvars.sh` (sets `MKLROOT`), or
  * Debian/Ubuntu - `sudo apt-get install libmkl-dev` (headers in
    `/usr/include/mkl`, LP64 libs in the default library path).

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

## Example: a batched QR solve with the Compact API

[`examples/solve_qr_compact.cpp`](examples/solve_qr_compact.cpp) solves a batch
of square systems `A_v X_v = B_v` end to end with the Compact-format (interleaved)
pipeline - the routine this repo adds (`ext_mkl_dormqr_compact`) is the middle
step:

```
mkl_dgeqrf_compact      A = Q R                       factor the batch
ext_mkl_dormqr_compact  B <- Q^T B                    apply Q^T  (the added routine)
mkl_dtrsm_compact       R X = (Q^T B)  ->  X = R^-1 Q^T B   triangular solve
```

The dense batches are packed with `mkl_dgepack_compact`, run through the three
compact calls, and unpacked with `mkl_dgeunpack_compact`. It cross-checks the
result against the naive baseline - `LAPACKE_dgels('N')` on each matrix
separately (which reduces to the same QR solve when `m == n`) - confirming the
compact batch agrees with the per-matrix driver and with the known exact
solution. One batch size is deliberately not a multiple of the SIMD width to
exercise the padded last pack. Built (with MKL) as the `solve_qr_compact`
target and registered as the `example_solve_qr_compact` CTest:

```sh
./build/solve_qr_compact
```

## Benchmark: batched vs. per-matrix throughput

[`examples/bench_qr_compact.cpp`](examples/bench_qr_compact.cpp) measures how
much the Compact (interleaved, batched) pipeline buys over the conventional
one-matrix-at-a-time LAPACK path on the small-matrix regime the compact API
targets. For each order `n in {20, 40, 60, 80, 100}` it builds a pool of
`nmat` well-conditioned matrices with known solution `X == 1` and times two
solves of the same data:

```
batched      mkl_dgeqrf_compact -> ext_mkl_dormqr_compact -> mkl_dtrsm_compact
non-batched  LAPACKE_dgeqrf     -> LAPACKE_dormqr          -> cblas_dtrsm
```

The batched path compacts/uncompacts each group of `V` matrices (the SIMD
vector length of the active compact format) on the fly, as the intended
application would; the non-batched path turns LAPACKE NaN-checking off
(`LAPACKE_set_nancheck(0)`) so the per-matrix driver is timed clean. The outer
loop over the pool is parallelised with OpenMP (MKL's own threading is pinned to
1 so the outer loop is the only parallelism), each size is timed `reps` times
keeping the best, both paths are accuracy-gated against the known solution, and
a geometric-mean speedup across the sizes is printed at the end.

```sh
./build/bench_qr_compact [nmat] [reps]    # defaults: 1000 matrices, 3 reps
```

## Accordance with the design document

* **API (sections 2-5):** `ext_mkl_dormqr_compact` is implemented with the exact
  signature, `MKL_COMPACT_PACK` format abstraction, `lwork = -1` workspace
  query, and per-matrix `info[]` reporting (`info[i] = -j` for an illegal
  `j`-th argument).
* **Dispatcher (section 8.1):** unwraps the format to the interleave width `V`
  (SSE/AVX/AVX-512 -> 2/4/8 for FP64, 4/8/16 for FP32) and dispatches to the
  templated kernel.
* **Validation (section 7):** `test_ext_mkl_ormqr_compact` runs Suite 1 (isolated
  `op(Q)*C` vs dense LAPACK, gate `20*s*eps`) over the full feature matrix -
  `side in {L,R} x layout in {col,row} x trans in {N,T}` - and Suite 2 (end-to-end
  `AX=B`: `mkl_dgeqrf_compact -> ext_mkl_dormqr_compact -> mkl_dtrsm_compact`,
  gates on forward error and system residual at `100*n*eps`) against real MKL.
* **Padding (section 6.4):** padded slots of the last compact pack carry identity
  factorizations (`tau=0`), so applying them is a no-op; exercised by the
  partial-group test cases.

**Feature coverage:** `side in {'L','R'}`, `layout in {MKL_COL_MAJOR,
MKL_ROW_MAJOR}`, `trans in {N,T}` (`C` folds to `T` for the real types) in
FP64/FP32. The tuned contiguous kernel serves the `side='L'`, column-major
solver path; the other three side/layout combinations run through a
stride-generalized kernel (same unblocked `dorm2r` math, correctness-first -
the non-contiguous inner sweep is not yet SIMD-tuned). Real types only
(`d`/`s`); complex (`c`/`z`) is out of scope. Unsupported argument values are
reported through `info[]` (never silently miscomputed), per LAPACK convention.
