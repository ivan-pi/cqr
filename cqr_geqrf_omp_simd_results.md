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

### GCC 13.3, V = 8 (AVX-512)

```
   n |  vec GF/s |  omp-outer |  omp-inner | out/vec |  in/vec
-----+-----------+------------+------------+---------+---------
   8 |       9.87 |       2.28 |       6.63 |    0.23x |    0.67x
  16 |      15.60 |       2.57 |      10.69 |    0.17x |    0.69x
  24 |      18.03 |       2.57 |      11.17 |    0.14x |    0.62x
  30 |      17.26 |       2.67 |      12.97 |    0.15x |    0.75x
  32 |      16.24 |       2.80 |      12.30 |    0.17x |    0.76x
  45 |      15.61 |       2.72 |      12.22 |    0.17x |    0.78x
  48 |      15.01 |       2.71 |      12.03 |    0.18x |    0.80x
  60 |      14.72 |       2.59 |      11.62 |    0.18x |    0.79x
  64 |      13.18 |       2.50 |      11.31 |    0.19x |    0.86x
  96 |      12.98 |       2.28 |      11.55 |    0.18x |    0.89x
 105 |      13.35 |       2.21 |      11.64 |    0.17x |    0.87x
 128 |      12.12 |       2.10 |      11.45 |    0.17x |    0.94x
 168 |      12.51 |       1.93 |      11.41 |    0.15x |    0.91x
-----+-----------+------------+------------+---------+---------
geomean omp-inner / vec-types: 0.79x
```

### Clang 18.1, V = 8 (AVX-512)

```
   n |  vec GF/s |  omp-outer |  omp-inner | out/vec |  in/vec
-----+-----------+------------+------------+---------+---------
   8 |       9.52 |       1.89 |       7.04 |    0.20x |    0.74x
  16 |      13.67 |       2.82 |      10.54 |    0.21x |    0.77x
  24 |      15.21 |       3.05 |      11.59 |    0.20x |    0.76x
  30 |      14.62 |       3.00 |      11.46 |    0.20x |    0.78x
  32 |      13.16 |       2.85 |      11.34 |    0.22x |    0.86x
  45 |      12.83 |       2.73 |      11.67 |    0.21x |    0.91x
  48 |      13.22 |       2.54 |      11.52 |    0.19x |    0.87x
  60 |      12.19 |       2.16 |      10.96 |    0.18x |    0.90x
  64 |      11.26 |       2.08 |      11.16 |    0.18x |    0.99x
  96 |      11.68 |       1.88 |      11.55 |    0.16x |    0.99x
 105 |      12.15 |       1.85 |      11.69 |    0.15x |    0.96x
 128 |      11.17 |       1.80 |      11.00 |    0.16x |    0.99x
 168 |      11.68 |       1.76 |      11.40 |    0.15x |    0.98x
-----+-----------+------------+------------+---------+---------
geomean omp-inner / vec-types: 0.88x
```

A narrower interleave (`--simdlen=4`, V = 4 / 256-bit) shows the same ordering:
omp-outer ~0.2x, omp-inner ~0.70x (GCC) / 0.70x (Clang) geomean.

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
loop is an outer loop wrapping a nontrivial nest, neither GCC nor Clang sinks the
vectorization inward -- the pragma is dropped and the code stays scalar.

**2. Moving the pragma to the innermost lane loop recovers most of the speed.**
`omp-inner` reaches **0.79x** (GCC) and **0.88x** (Clang) of the hand-written
vector-types kernel on geomean, and **0.96--0.99x** on Clang for n >= 64. Each
`#pragma omp simd` there is a straight-line, contiguous (stride-1 in `v`) sweep
over the `V` lanes -- exactly the shape auto-vectorization is built for -- so the
compiler packs the batch just as the vector types do. The residual gap is the
split-loop overhead (per-column stack temporaries `tail`/`tau`/`w`, reloaded
between the separated simd loops) that the vector-types kernel keeps in
registers across a single fused loop body, plus its hand JB=4 column blocking.

**3. Practical guidance.** To match GNU vector types with `-fopenmp-simd` on this
kind of batched kernel, put `simd` on the **innermost** lane loop, not the outer
batch loop. The outer-loop form reads better (it is the single-element algorithm
verbatim) but compiles to scalar on both mainstream compilers today. Clang closes
the gap to the hand-written kernel more than GCC here.

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

# drop-in correctness (both backends), any config
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
