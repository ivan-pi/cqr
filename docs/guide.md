# User guide

## Prerequisites

- **CMake ≥ 3.18**, a **C++17 compiler** (GCC or Clang), and a build tool
  (Make or Ninja).
- **Intel MKL** — provides the Compact-format extension (`mkl_compact.h`,
  `mkl_?geqrf_compact`, …) that `cqr` builds on, plus the LAPACK/LAPACKE the
  test and example programs validate against. Any MKL works:
    - **oneAPI MKL** — `source /opt/intel/oneapi/setvars.sh` (sets `MKLROOT`), or
    - **Debian/Ubuntu** — `sudo apt-get install libmkl-dev` (headers in
      `/usr/include/mkl`, LP64 libraries on the default search path).

`cqr`'s own kernels need no MKL; the dependency is for the Compact pack/unpack
helpers and for the LAPACK used in validation. To build only the portable
kernels, see [MKL-free builds](#mkl-free-builds).

## Build and test

The `*_compact` symbols are reached through the BLAS link line, so the BLAS is
selected with CMake's standard `BLA_VENDOR` mechanism (only MKL provides the
compact API; other vendors stop with a fatal error).

```sh
cmake -S . -B build -DBLA_VENDOR=Intel10_64lp_seq -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Performance builds

The library sets no `-march` of its own; optimization flags are the caller's to
choose. For a fast build, pass them through `CMAKE_CXX_FLAGS` so the SIMD
kernels target the host's widest vectors (the same AVX-512 MKL selects at
runtime):

```sh
cmake -S . -B build -DBLA_VENDOR=Intel10_64lp_seq -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-O3 -march=native"
```

Correctness is independent of these flags; only throughput changes.

!!! tip "Benchmarks need host-tuned flags"
    The benchmark examples compare against MKL, whose runtime dispatch already
    picks the widest vectors on the machine. Build with `-march=native` (or an
    explicit `-mavx512f …`) for a fair comparison — otherwise the `cqr` kernels
    are pinned to the baseline ISA while MKL uses AVX-512.

## MKL-free builds

`-DCQR_WITH_MKL=OFF` builds only the portable kernels — no MKL, and the
MKL-backed tests are skipped:

```sh
cmake -S . -B build -DCQR_WITH_MKL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

You still reach every kernel through the
[portable C API](reference.md#portable-c-api) (`cqr_compact.h`), which takes an
explicit interleave width `V` and carries no MKL dependency.

## Examples

Three programs are built and registered with CTest:

| Example | What it shows |
|---------|---------------|
| `solve_qr_compact` | A batch of square systems `Aᵥ Xᵥ = Bᵥ` solved end to end with the compact pipeline (`mkl_dgeqrf_compact` → `cqr_mkl_dormqr_compact` → `cqr_mkl_dtrsm_compact`), cross-checked against per-matrix `LAPACKE_dgels`. |
| `bench_qr_compact [nmat] [reps]` | Throughput of the fully open compact *solve* pipeline vs. MKL's batched pipeline and the one-matrix-at-a-time LAPACK path, over pools of small matrices (order 10–100), reporting geometric-mean speedups. |
| `bench_geqrf_compact [nmat] [reps]` | Throughput of the *factorization*: `cqr_mkl_dgeqrf_compact` vs. `mkl_dgeqrf_compact` vs. per-matrix `LAPACKE_dgeqrf`, reporting GFLOP/s and a geometric-mean speedup. |

They are registered as the CTest cases `example_solve_qr_compact`,
`bench_qr_compact_integration`, and `bench_geqrf_compact_integration`. Build
the benchmarks with host-tuned flags (see above) for a fair comparison against
MKL.

## Calling the routines

The worked pattern is the batched `AX = B` solve — factor, apply `Qᵀ`, then
back-substitute:

```c
#include "cqr_mkl_ext.h"   // MKL-style API; needs the MKL headers for its types

cqr_mkl_dgeqrf_compact(layout, m, n, ap, ldap, taup,
                       work, lwork, &info, format, nm);   // A = Q R
cqr_mkl_dormqr_compact(layout, 'L', 'T', m, nrhs, k, ap, ldap, taup,
                       bp, ldbp, work, lwork, &info, format, nm); // B := Qᵀ B
cqr_mkl_dtrsm_compact (layout, MKL_LEFT, MKL_UPPER, MKL_NOTRANS, MKL_NONUNIT,
                       m, nrhs, 1.0, ap, ldap, bp, ldbp, format, nm); // B := X
```

The matrices ride in the same Compact buffers used elsewhere in the MKL compact
API: pack with `mkl_?gepack_compact`, obtain the opaque `format` from
`mkl_get_format_compact()`, and pass the total batch size `nm`.

!!! warning "Workspace is per-routine"
    Size each routine's `work` array from **its own** `lwork = -1` query, and
    give each routine its own buffer. `cqr`'s kernels need no scratch (their
    query returns `1`), but MKL's `mkl_?geqrf_compact` needs `~n·V`. Because
    compact routines skip argument checking, handing one routine a `work` sized
    for another — or sharing a buffer across `geqrf`/`ormqr` — is undefined
    behavior.

See the [API reference](reference.md) for every routine's parameters, the two
API surfaces, and the compact storage layout, and the
[design notes](design/index.md) for the algorithms and numerical scope.
