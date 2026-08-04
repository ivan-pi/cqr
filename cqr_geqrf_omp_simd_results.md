# Experiment: OpenMP SIMD vs GNU vector types for compact `dgeqrf`

> Assisted-by: Claude:claude-opus-4.8

## Question

The compact QR kernels vectorize the batch with **GNU vector types**
(`__attribute__((vector_size))`): element `(i,j)` of the `V` interleaved
matrices is a `V`-wide vector, and arithmetic operators lift the scalar `geqr2`
over the whole pack for free (`cqr_geqrf_compact.hpp`). Can we get the same batch
(outer-loop) vectorization from the compiler instead, by writing the ordinary
scalar per-element `geqr2` and annotating the lane loop with
`#pragma omp simd simdlen(V)` under `-fopenmp-simd`?

Two placements of that pragma are compared against the vector-types kernel:

| kernel | where `#pragma omp simd` goes | source |
|--------|-------------------------------|--------|
| **vec-types** | (none -- GNU vector types) | `cqr_geqrf_compact.hpp` |
| **omp-outer** | on the **outer** lane loop; scalar nested `geqr2` inside, element access via the `CQR_A(i,j)` macro that closes over the lane `v` | `cqr_geqrf_compact_omp.hpp`, `geqrf_compact_group_omp` |
| **omp-inner** | on the **innermost** lane loop, one contiguous stride-1 sweep per element operation, with small per-column stack temporaries | `cqr_geqrf_compact_omp.hpp`, `geqrf_compact_group_omp_inner` |

`omp-outer` is the literal "outer-loop vectorization" idiom requested: the batch
loop is the vectorized dimension, the inner code reads as a single-matrix
algorithm over `A(i,j)`. `omp-inner` is the fallback the compilers actually
vectorize (see results).

## Correctness (drop-in for the C entry points)

`omp-outer` is wired behind the **same** `dgeqrf_compact` / `sgeqrf_compact` C
symbols as the vector-types kernel (`cqr_geqrf_compact_dispatch_omp.cpp`, library
`cqr_compact_ompsimd`). The existing portable test
(`src/test_cqr_geqrf_compact.cpp`) is compiled a second time against that library
(`test_cqr_geqrf_compact_omp`, CTest `portable_geqrf_ompsimd`): a literal drop-in
swap of the backend. Both backends pass identically -- `(H, tau, R)` match the
scalar `geqr2` reference and MKL-free reconstruction/solve to working precision
-- under GCC and Clang. `omp-inner` is gated the same way inside the benchmark
(every kernel's result is checked against the scalar reference each run).

Correctness does **not** depend on `-fopenmp-simd`: without the flag the pragma
is ignored and each group runs as a plain scalar loop over its `V` matrices, still
a valid factorization. The flag only changes throughput.

## Benchmark

`examples/bench_geqrf_omp_simd.cpp` (portable, no MKL): the three kernel
templates are called directly on identical compact buffers, single-threaded, at
the host's native double width. Only the factorization is timed (best of `reps`
passes); the destroyed input is restored untimed between passes. Standard
`?geqrf` flop count; `nmat = 512`.

Machine: Intel Xeon (AVX-512), `-O3 -march=native -fopenmp-simd`, GCC 13.3 /
Clang 18.1. `-march=native` selected AVX-512, so the native double width is
**V = 8**. Numbers are GFLOP/s; the ratio columns are throughput relative to
vec-types (higher = closer to the hand-written kernel).

All indexing inside the kernels is plain `int` (matrix dims for this batch are in
the low hundreds, so an in-group index never overflows): a widening to `size_t`
on the lane term forces 64-bit index vectors and blocks the compiler from
proving the lane loop is unit-stride. Per-group base pointers are advanced by
accumulation rather than a `g * stride` multiply, so large batches stay correct
without a wider type.

### GCC 13.3, V = 8 (AVX-512)

```
   n |  vec GF/s |  omp-outer |  omp-inner | out/vec |  in/vec
-----+-----------+------------+------------+---------+---------
   8 |       8.92 |       2.35 |       6.54 |    0.26x |    0.73x
  16 |      13.26 |       2.65 |      10.23 |    0.20x |    0.77x
  24 |      16.21 |       2.78 |      11.66 |    0.17x |    0.72x
  30 |      16.54 |       2.79 |      11.63 |    0.17x |    0.70x
  32 |      13.97 |       2.81 |      11.34 |    0.20x |    0.81x
  45 |      14.95 |       2.74 |      10.95 |    0.18x |    0.73x
  48 |      13.54 |       2.68 |      11.25 |    0.20x |    0.83x
  60 |      13.81 |       2.49 |      10.85 |    0.18x |    0.79x
  64 |      11.40 |       2.45 |      10.49 |    0.21x |    0.92x
  96 |      12.44 |       2.27 |      10.77 |    0.18x |    0.87x
 105 |      12.96 |       2.22 |      10.88 |    0.17x |    0.84x
 128 |      11.66 |       2.13 |      10.97 |    0.18x |    0.94x
 168 |      12.02 |       1.98 |      10.85 |    0.16x |    0.90x
-----+-----------+------------+------------+---------+---------
geomean omp-inner / vec-types: 0.81x
```

### Clang 18.1, V = 8 (AVX-512)

```
   n |  vec GF/s |  omp-outer |  omp-inner | out/vec |  in/vec
-----+-----------+------------+------------+---------+---------
   8 |       9.56 |       1.54 |       6.88 |    0.16x |    0.72x
  16 |      13.96 |       2.09 |       9.85 |    0.15x |    0.70x
  24 |      14.99 |       2.29 |      10.61 |    0.15x |    0.71x
  30 |      14.15 |       2.33 |      10.66 |    0.16x |    0.75x
  32 |      13.49 |       2.36 |      10.42 |    0.17x |    0.77x
  45 |      12.94 |       2.20 |      10.58 |    0.17x |    0.82x
  48 |      13.45 |       2.15 |      10.65 |    0.16x |    0.79x
  60 |      12.51 |       1.93 |       9.85 |    0.15x |    0.79x
  64 |      11.37 |       1.90 |      10.18 |    0.17x |    0.90x
  96 |      12.07 |       1.78 |      10.56 |    0.15x |    0.87x
 105 |      11.99 |       1.77 |      10.43 |    0.15x |    0.87x
 128 |      11.17 |       1.76 |      10.49 |    0.16x |    0.94x
 168 |      11.76 |       1.76 |      10.54 |    0.15x |    0.90x
-----+-----------+------------+------------+---------+---------
geomean omp-inner / vec-types: 0.81x
```

A narrower interleave (`--simdlen=4`, V = 4 / 256-bit) shows the same ordering:
omp-outer ~0.15--0.20x, omp-inner ~0.67x (GCC) / 0.60x (Clang) geomean. (Absolute
throughput on this shared VM is noisy at the ~10% level; the ordering and the
~5x outer/inner gap are the stable signal.)

## Findings

**1. The outer-loop `omp simd` idiom is not vectorized by either compiler.**
`omp-outer` runs at ~0.15--0.23x of the vector-types kernel -- i.e. essentially
scalar (a factor of roughly `V` slower, as expected when the batch is not
vectorized at all). The compilers say so directly:

* **GCC** (`-fopt-info-vec-all`): at the `#pragma omp simd` line,
  `not vectorized: loop nest containing two or more consecutive inner loops
  cannot be vectorized`. GCC's vectorizer only handles innermost loops or perfect
  nests; the `geqr2` body is a sequence of sibling inner loops (the norm
  reduction, the reflector scaling, and the trailing-column update, which itself
  nests two more), so outer-loop vectorization is refused outright.
* **Clang** (`-Rpass-analysis=loop-vectorize`): at the same line,
  `loop not vectorized: the optimizer was unable to perform the requested
  transformation` (`-Wpass-failed=transform-warning`). Its attempts on the inner
  loops then fail on real per-lane loop-carried dependences and FP-reassociation
  it will not assume.

The takeaway: `#pragma omp simd` on a loop asserts that *that* loop's iterations
are independent, but it does **not** buy loop interchange. When the annotated
loop is an outer loop wrapping a nontrivial nest, GCC's and Clang's
*auto-vectorizers* do not sink the vectorization inward -- the pragma is dropped
and the code stays scalar. (A dedicated OpenMP SIMD code generator can, though:
icpx's `-qopenmp-simd` path vectorizes exactly this loop -- see the icpx section.
It is the lowering path that decides, not that outer-loop vectorization is
impossible in principle.)

**2. Moving the pragma to the innermost lane loop recovers most of the speed.**
`omp-inner` reaches **0.79x** (GCC) and **0.88x** (Clang) of the hand-written
vector-types kernel on geomean, and **0.96--0.99x** on Clang for n >= 64. Each
`#pragma omp simd` there is a straight-line, contiguous (stride-1 in `v`) sweep
over the `V` lanes -- exactly the shape auto-vectorization is built for -- so the
compiler packs the batch just as the vector types do. The residual gap is the
split-loop overhead (per-column stack temporaries `tail`/`tau`/`w`, reloaded
between the separated simd loops) that the vector-types kernel keeps in
registers across a single fused loop body, plus its hand JB=4 column blocking.

**3. Practical guidance.** With `-fopenmp-simd` (GCC, Clang, or icpx -- all three
route it through the LLVM/GCC auto-vectorizer), put `simd` on the **innermost**
lane loop, not the outer batch loop: the outer-loop form reads better (it is the
single-element algorithm verbatim) but compiles to scalar there. With **icpx and
`-qopenmp-simd`** the outer-loop form vectorizes and the distinction disappears
-- both reach ~0.9x of the hand-written vector types (see the icpx section). So
the clean outer-loop idiom *is* viable today, on the compiler path that has a
real OpenMP SIMD code generator behind the pragma.

## icpx (Intel oneAPI) -- the outer loop *does* vectorize

Intel's oneAPI DPC++/C++ compiler (**icpx 2026.1.1**) settles the open question,
and the answer turns on *which SIMD lowering path* `#pragma omp simd` takes --
not on the compiler brand. icpx exposes two:

| flag | `#pragma omp simd` lowered by | outer kernel |
|------|-------------------------------|--------------|
| `-fopenmp-simd` | the LLVM auto-vectorizer (icpx is LLVM-based, so the *same* path as Clang) | **not vectorized** -- `-Wpass-failed` at the pragma, scalar codegen, ~0.08x |
| `-qopenmp-simd` (or `-fiopenmp`) | **Intel's own OpenMP SIMD code generator** | **vectorized** -- outer lane loop packed + AVX-512 predication masks, ~0.91x |

Evidence from the generated assembly for the outer-loop kernel
(`geqrf_compact_group_omp`, all V instantiations, `-O3 -xHost`):

| | scalar `*sd` ops | packed `*pd` ops | mask (`k`) regs |
|-|------------------|------------------|-----------------|
| `-fopenmp-simd` | 236 | 36 | 0 |
| `-qopenmp-simd` | **0** | **600** | 156 |

So Intel's OpenMP SIMD path performs exactly the outer-loop (batch)
vectorization GCC's and Clang's auto-vectorizers refuse: the clean single-element
`geqr2` under `#pragma omp simd simdlen(V)` becomes real SIMD, no restructuring.
`-qopenmp-simd` is simd-only -- it links no OpenMP runtime (`ldd` shows no
`libiomp5`), the direct counterpart of `-fopenmp-simd`.

### icpx 2026.1.1, `-qopenmp-simd -xHost`, V = 8 (AVX-512)

```
   n |  vec GF/s |  omp-outer |  omp-inner | out/vec |  in/vec
-----+-----------+------------+------------+---------+---------
   8 |       7.86 |       8.02 |       7.31 |    1.02x |    0.93x
  16 |      11.12 |       9.79 |       9.55 |    0.88x |    0.86x
  24 |      11.52 |       9.57 |       9.68 |    0.83x |    0.84x
  30 |      10.29 |       9.25 |       9.37 |    0.90x |    0.91x
  32 |       9.61 |       9.23 |       9.28 |    0.96x |    0.97x
  45 |      10.67 |       8.93 |       9.21 |    0.84x |    0.86x
  48 |      10.50 |       8.82 |       9.20 |    0.84x |    0.88x
  60 |      10.22 |       9.20 |       9.32 |    0.90x |    0.91x
  64 |       8.02 |       9.14 |       9.06 |    1.14x |    1.13x
  96 |      10.15 |       9.35 |       9.50 |    0.92x |    0.94x
 105 |      10.81 |       9.20 |       9.35 |    0.85x |    0.87x
 128 |       7.73 |       8.81 |       9.01 |    1.14x |    1.17x
 168 |       8.30 |       7.87 |       8.08 |    0.95x |    0.97x
-----+-----------+------------+------------+---------+---------
geomean: omp-outer 0.91x, omp-inner 0.94x
```

Under `-qopenmp-simd` the *outer-loop* form -- the one written as the plain
single-element algorithm, which reads best -- reaches parity with the
hand-written GNU vector types (0.91x), and matches the inner-loop form (0.94x).
The choice between outer and inner stops mattering once the compiler's OpenMP
SIMD path can vectorize the outer loop. (Intel's codegen here used 256-bit `ymm`
packs even on the AVX-512 host, so vec-types -- which the compiler lowers to
`zmm` -- keeps a small edge at some sizes; passing a narrower `simdlen` or
`-qopt-zmm-usage=high` can shift that.)

The CMake build selects `-qopenmp-simd` automatically for `IntelLLVM`;
`scripts/icpx_outer_simd_check.sh` reproduces the two-path comparison and the
op-count evidence from a bare checkout (installing the compiler if needed).

## Reproduce

```sh
# GCC
cmake -S . -B build-gcc   -DCQR_WITH_MKL=OFF -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-O3 -march=native"
cmake --build build-gcc -j
./build-gcc/bench_geqrf_omp_simd 512 7            # native width
./build-gcc/bench_geqrf_omp_simd --simdlen=4 512 7

# Clang
cmake -S . -B build-clang -DCQR_WITH_MKL=OFF -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_FLAGS="-O3 -march=native"
cmake --build build-clang -j
./build-clang/bench_geqrf_omp_simd 512 7

# icpx (Intel oneAPI) -- CMake auto-selects -qopenmp-simd, which vectorizes the
# outer-loop kernel. scripts/icpx_outer_simd_check.sh also prints the
# -fopenmp-simd vs -qopenmp-simd instruction-count comparison (installs the
# compiler if absent).
. /opt/intel/oneapi/setvars.sh
cmake -S . -B build-icpx  -DCQR_WITH_MKL=OFF -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=icpx -DCMAKE_CXX_FLAGS="-O3 -xHost"
cmake --build build-icpx -j
./build-icpx/bench_geqrf_omp_simd 512 7
./scripts/icpx_outer_simd_check.sh

# drop-in correctness (all backends), any config
ctest --test-dir build-gcc -R portable_geqrf
```

## Note: a pre-existing crash unrelated to this experiment

While benchmarking under `clang -march=native` (AVX-512) I hit a **segfault in
the existing vector-types kernel**, not in any code added here:
`portable_geqrf` (the vector-types `test_cqr_geqrf_compact`) crashes inside
`geqrf_compact_group_strided<double,4>` (`cqr_geqrf_compact.hpp:239`) on the very
first, valid **row-major** validation call `dgeqrf_compact('R', 8, 6, ...)`. In
the crashing frame the buffer pointer has become null and the forwarded
dimensions are garbage, i.e. it presents as a Clang miscompilation of the strided
row-major path at `-O2`/`-O3` with AVX-512. It reproduces with the stock kernel
sources alone (no OpenMP files involved).

Scope of the issue:

* **Only** `clang` **+** `-march=native` (AVX-512). GCC `-march=native` (full
  AVX-512) is fine; Clang without `-march=native` is fine; the omp-simd drop-in
  test passes even under `clang -march=native`.
* The project's default / CI configuration does **not** pass `-march=native`, so
  CI is unaffected and all tests pass there under both compilers.

It is left unfixed here because it is outside this experiment and in
performance-tuned code the maintainer owns; flagging it for a separate look. The
column-major path (the tuned one, and everything these OpenMP kernels exercise)
is unaffected.
