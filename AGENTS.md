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
tests/     portable (no BLAS) and MKL-backed suites, templated on the scalar
           type; test_compact_util.hpp / test_mkl_util.hpp hold the helpers and
           the compact<T> / cqr_mkl<T> / mkl<T> / lapack<T> dispatch structs
examples/  the worked solve and the benchmarks (BENCHMARKS.md), on bench_util.hpp
docs/      one design document per routine
```

## Formatting and linting

Layout follows C++ Core Guidelines **NL.17** (K&R-derived / "Stroustrup"), the
style of `.clang-format`; `.clang-tidy` lints (the bug finders, the performance
checks, and the parts of modernize and readability that are not a matter of
taste). Both run through [pre-commit](https://pre-commit.com/), whose
`.pre-commit-config.yaml` pins the clang-format version CI checks with
(`.github/workflows/style.yml`):

```sh
pip install pre-commit
pre-commit install                 # format on every commit from now on
pre-commit run --all-files         # or by hand, over the whole tree
```

clang-tidy wants the compile database of a tree configured with **clang** (it
parses with clang's front end, which cannot read g++'s `omp.h`; `libomp-dev`
supplies clang's) and is a manual stage:

```sh
CXX=clang++ cmake -S . -B build-tidy -DBLA_VENDOR=Intel10_64lp_seq
pre-commit run --hook-stage manual clang-tidy --all-files
```

Use the clang-tidy of the same LLVM release as that clang++. Only the
translation units are listed; the headers are checked through them.

Hand-aligned tables and compact one-liners that clang-format would expand are
fenced with `// clang-format off` / `// clang-format on`; leave those fences in
place. The dispatch macros (`CQR_TEST_*_DISPATCH` in the test headers,
`CQR_DEFINE_*_ENTRY_POINTS` in the two adapter sources) take a type name as an
argument, which cannot be parenthesized, so they also sit between
`// NOLINTBEGIN(bugprone-macro-parentheses)` and the matching `NOLINTEND`.

## Claude Code hooks

`.claude/settings.json` wires up two hooks:

- `.claude/hooks/session-start.sh` provisions a fresh remote session: Intel MKL,
  clang's OpenMP runtime, and pre-commit with its hook environments. It does
  nothing on a developer's own machine.
- `.claude/hooks/format.sh` runs after every `Edit` or `Write`: the pre-commit
  hooks on that one file, so Claude's edits come out the way a commit would.
  clang-format fixes silently; a finding the hooks cannot fix is fed back to
  Claude to correct. Without pre-commit it falls back to the system
  clang-format.

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
- **Threading over groups.** Every all-groups driver is a call to
  `for_each_group<V>(nm, flops_per_group, body)` (`cqr_compact_common.hpp`):
  a static-schedule `omp parallel for` on at most one thread per group, gated
  by an if-clause. Keep new drivers on it; do not add threading inside a group
  kernel. The gate refuses when nesting is exhausted (that is what makes the
  library compose with a caller's outer parallel loop), when there is a single
  group, or below `parallel_min_flops`.
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
