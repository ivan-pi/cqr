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

## Code style

Layout follows C++ Core Guidelines **NL.17** (K&R-derived / "Stroustrup"),
enforced by `.clang-format`. Format changed C++ before committing:

```sh
clang-format -i src/*.h src/*.hpp src/*.cpp examples/*.cpp
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
  ISA -- one portable source for every width.
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
- **Scope.** Real precisions (`s`/`d`) only; no column pivoting; no overflow/
  underflow-safe reflector rescaling (see the design documents).
