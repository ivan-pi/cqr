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
           shared cqr_compact_common.hpp; sysvnp's is a driver over the sytrfnp
           and sytrsnp group kernels), the two adapter sources that implement
           the public headers (cqr_compact.cpp, cqr_mkl_ext.cpp), and
           cqr_matrix_view.hpp, the dense MatrixView the tests, benchmarks and
           examples share (internal: src/ is on their include path, but the
           public API stays include/)
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
translation units are listed; the headers are checked through them. CI runs
this only on manual dispatch (`.github/workflows/clang-tidy.yml`), since the
runner has to install MKL and an LLVM toolchain first; run it locally before
pushing changes to the kernels.

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
- **Never pass a pack across a call by reference.** The second face of the
  same clang behavior (issue #34 fixed the template-argument face in PR #35):
  a `const typename pack<T,V>::type &` parameter is loaded with the *natural*
  vector alignment once the call is not inlined -- the typedef's relaxed
  alignment does not survive on the referent -- and faults on a buffer or stack
  local that is only `T`-aligned. The `JB = 4` block helper of `sytrfnp` was
  first written that way (the pivot `d` passed by reference from the caller's
  local) and segfaulted on a `movapd` under clang in the plain Release
  configuration CI uses, while the same source passed every test under gcc.
  Passing by value is no escape (`-Wpsabi`, see `cqr_compact_common.hpp`).
  Instead give the helper the view and the indices and let it load what it
  needs (`potrf_update_block` / `sytrfnp_update_block` do exactly that), or
  pass the scalar the pack was broadcast from. The tiny lane-wise helpers
  (`vsqrt`, `broadcast`, `trsm_dot_block`'s `va`) get away with references only
  because they always inline.
- **Build and test with both gcc and clang before pushing.** CI runs both, and
  the packs' alignment is exactly the kind of contract only one of them
  enforces: both alignment faults so far (issue #34 and the one above) were
  invisible to gcc. The portable tree needs nothing but the compiler:

  ```sh
  CXX=clang++ cmake -S . -B build-clang -DCQR_WITH_MKL=OFF && cmake --build build-clang && ctest --test-dir build-clang
  ```

- **Threading over groups.** Every all-groups driver is a call to
  `for_each_group<V>(nm, flops_per_group, body)` (`cqr_compact_common.hpp`):
  a static-schedule `omp parallel for` on at most one thread per group, gated
  by an if-clause. Keep new drivers on it; do not add threading inside a group
  kernel. The gate refuses when nesting is exhausted (that is what makes the
  library compose with a caller's outer parallel loop), when there is a single
  group, or below `parallel_min_flops`.
- **Two views, one idea.** Compact (packed) operands are addressed through
  `BatchView` (`src/cqr_compact_common.hpp`), dense host-side ones through
  `MatrixView` (`src/cqr_matrix_view.hpp`). Both carry the layout as runtime
  strides `(si, sj)`, so one body serves column-major and row-major and a
  transpose is a stride swap. Do not hand-write `A[i + (size_t)j * lda]` in
  new tests, benchmarks or examples -- take a view. `MatrixView` asserts its
  bounds, so run the suites once in a `Debug` build when adding indexing code.
  The tests use both: the pack/unpack helpers in `test_compact_util.hpp` write
  the interleaved side through `BatchView` (`for_vlen` turns their runtime `V`
  into its compile-time one) and the dense side through `MatrixBatch`. The
  library and its tests are one internal codebase and share these views on
  purpose; what keeps the suites honest is that they compute the *answers*
  independently -- scalar LAPACK references, dense LAPACK/MKL cross-checks.
  The benchmarks' batch is `MatrixPool` (`examples/bench_util.hpp`): storage,
  the per-matrix view, and the base-pointer array the compact pack routines
  take; each benchmark only fills it.
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
