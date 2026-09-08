# AGENTS.md

Working notes for contributors (human or agent) to **cqr**, a batched QR library
for many small matrices in Intel MKL's Compact (interleaved) format.

## Build and test

Intel MKL supplies the Compact API and the LAPACK/LAPACKE used for validation:

```sh
# Debian/Ubuntu: sudo apt-get install libmkl-dev
cmake -S . -B build -DBLA_VENDOR=Intel10_64lp_seq -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

`-DCQR_WITH_MKL=OFF` builds only the portable kernels (no MKL, no MKL tests).

## Performance builds

The library sets no `-march` of its own; optimization flags are the caller's to
choose. For a fast build, pass them through `CMAKE_CXX_FLAGS` so the SIMD kernels
target the host's widest vectors:

```sh
cmake -S . -B build -DBLA_VENDOR=Intel10_64lp_seq -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-O3 -march=native"
```

Correctness is independent of these flags; only throughput changes.

## Tree

```
include/   public headers: cqr_compact.h (portable C API), cqr_mkl_ext.h
           (MKL-style API), cqr_mkl_alloc.h (optional RAII mkl_malloc helpers)
src/       the templated kernels (cqr_*_compact.hpp, one per routine, on the
           shared cqr_compact_common.hpp) and the two adapter sources that
           implement the public headers: cqr_compact.cpp, cqr_mkl_ext.cpp
tests/     portable (no BLAS) and MKL-backed suites, on test_compact_util.hpp
examples/  the worked solve and the benchmarks (BENCHMARKS.md), on bench_util.hpp
docs/      one design document per routine
```

## Code style

Layout follows C++ Core Guidelines **NL.17** (K&R-derived / "Stroustrup"),
enforced by `.clang-format`. Format changed C++ before committing:

```sh
clang-format -i include/*.h src/*.hpp src/*.cpp tests/*.hpp tests/*.cpp examples/*.hpp examples/*.cpp
```

Run it before committing so changes land already formatted. Hand-aligned tables
and compact one-liners that clang-format would expand are fenced with
`// clang-format off` / `// clang-format on`; leave those fences in place.

## Conventions worth knowing

- **Compact layout.** Element `(i,j)` of the `V` interleaved matrices in a group
  is stored contiguously; `V` is the SIMD width (FP64: 2/4/8 for SSE/AVX/AVX-512).
  A partial final group is padded with identities, so kernels run it unmasked.
- **SIMD via GNU vector types.** The kernels use
  `__attribute__((vector_size))` vectors, which the compiler lowers to the target
  ISA -- one portable source for every width. The pack type carries a relaxed
  `aligned(alignof(T))`; always name it as `typename pack<T,V>::type` and never
  pass it as a template *argument* (clang strips typedef alignment there and
  emits aligned loads that fault on 16-byte-aligned buffers -- issue #34).
- **Threading over groups.** Every all-groups driver wraps its group loop in
  `CQR_OMP_PARALLEL_GROUPS(ngroups)` (a static-schedule `omp parallel for`
  gated by `parallel_groups`, both in `cqr_compact_common.hpp`). Keep new
  drivers on that macro; do not add threading inside a group kernel. The gate
  is `ngroups >= omp_get_max_threads()` at the current nesting level, which is
  what makes the library compose with a caller's outer parallel loop.
- **One kernel per routine.** Every kernel addresses its operands through
  `BatchView` (strides `si`, `sj`), so column-major, row-major, and ormqr's
  `side='R'` are the same code with different strides. Register blocking is
  written as a `JB`-templated block helper with `for (c < JB)` loops the
  compiler unrolls, not as hand-expanded `w0..w3` copies.
- **Argument checking.** The MKL-style API (`cqr_mkl_*`) skips validation like
  MKL's own compact routines (`info` is a scalar, `0` on success). The portable C
  API (`cqr_compact.h`) validates LAPACK-style, returning `-j` for a bad j-th
  argument.
- **Workspace (`lwork`).** Size each routine's `work` from *its own* `lwork = -1`
  query, and give each routine its own buffer. The compact kernels here need no
  scratch (their query returns `1`), but MKL's `mkl_?geqrf_compact` needs `~n*V`.
  Because compact routines skip argument checking, handing one routine a `work`
  sized for another -- or sharing a buffer across `geqrf`/`ormqr` -- is undefined
  behavior: harmless on some MKL builds, silent heap corruption on others (this
  is exactly the bug that aborted the MKL test with "unaligned tcache chunk").
- **Buffer alignment.** Compact buffers are correct at any `T` alignment, but
  align the base to the pack width (64 B covers every format) so the SIMD kernels
  avoid cache-line splits -- `mkl_malloc(bytes, 64)`, which is what
  `mkl_alloc_bytes` does by default, or `std::aligned_alloc(64, ...)`. Only
  performance, not correctness, rides on it (up to ~40% on small sizes).
- **Scope.** Real precisions (`s`/`d`) only; no column pivoting; no overflow/
  underflow-safe reflector rescaling (see the design documents).
