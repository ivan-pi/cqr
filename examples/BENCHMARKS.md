# Benchmarks

> Assisted-by: Claude:claude-opus-4.8

Throughput benchmarks for the compact batched kernels, living alongside the
programs they drive in `examples/`. Each one runs the *same* linear-algebra math
three ways -- this project's open compact kernels, MKL's own compact kernels, and
the conventional one-matrix-at-a-time LAPACK path -- over pools of many small
matrices, and reports per-size throughput plus a geometric-mean speedup. Every
benchmark also cross-checks its result against per-matrix LAPACK, so each doubles
as an integration test (and is registered with CTest on a small, quick pool).

| Program | Measures | Compares |
|---------|----------|----------|
| [`bench_geqrf_compact`](#bench_geqrf_compact) | QR *factorization* | `cqr_mkl_dgeqrf_compact` vs `mkl_dgeqrf_compact` vs `LAPACKE_dgeqrf` |
| [`bench_potrf_compact`](#bench_potrf_compact) | Cholesky *factorization* (SPD) | `cqr_mkl_dpotrf_compact` vs `mkl_dpotrf_compact` vs `LAPACKE_dpotrf` |
| [`bench_qr_compact`](#bench_qr_compact) | end-to-end QR *solve* `AX = B` | fully-open compact pipeline vs MKL's compact pipeline vs per-matrix LAPACK |

The worked, self-validating solver `solve_qr_compact` (not a benchmark) lives in
the same folder; see the top-level [README](../README.md) for it.

## How to run them

The benchmarks are built by the standard MKL build (they need the compact API);
see the top-level [README](../README.md) / [AGENTS.md](../AGENTS.md) for the full
build. From a configured build tree:

```sh
cmake --build build -j
./build/bench_geqrf_compact          # QR factorization
./build/bench_potrf_compact          # Cholesky (SPD) factorization
./build/bench_qr_compact             # end-to-end QR solve
```

### Build for a fair comparison (`-march=native`)

This library sets no `-march` of its own, so a default build emits only the
baseline ISA while MKL's compact kernels dispatch to the host's widest vectors
(AVX-512) at runtime -- an apples-to-oranges comparison in which the open kernels
look unfairly slow. For a fair cqr-vs-MKL measurement, build with host-tuned
flags so the compact kernels emit the full vector width:

```sh
cmake -S . -B build -DBLA_VENDOR=Intel10_64lp_seq -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-O3 -march=native"
```

Correctness is independent of these flags; only throughput changes. (The
correctness gate each benchmark carries holds either way.)

### Threading

The outer loop over the matrix pool is parallelized with OpenMP (when available),
which is the intended "outer multi-threaded loop" usage of the compact kernels;
MKL's own internal threading is pinned to 1 so the two levels do not
oversubscribe. Build without OpenMP for a sequential run.

## `bench_geqrf_compact`

Throughput of the QR *factorization* itself over pools of small square matrices,
comparing three implementations of the same LAPACK `?geqrf` math:

* `cqr_mkl_dgeqrf_compact` -- this project's batched SIMD kernel,
* `mkl_dgeqrf_compact` -- Intel MKL's batched compact kernel,
* `LAPACKE_dgeqrf` -- conventional one matrix at a time.

To measure the factorization kernels rather than data movement, the pool is
packed into compact form once, up front; only the factorization is timed, and the
destroyed input is restored (untimed) before each pass. It reports GFLOP/s (the
standard `2mn^2 - (2/3)n^3` `?geqrf` flop count), matrices/s for each path, the
three pairwise speedups, and a geometric-mean speedup of cqr over per-matrix
LAPACK. The error column is the compact `(H, tau)` compared elementwise against a
fresh `LAPACKE_dgeqrf`.

```
Usage:  bench_geqrf_compact [--size-sweep=nmin:nmax[:stride]] [--simdlen=2|4|8]
        [nmat] [reps]                          (defaults: 512 matrices, 3 reps)
```

## `bench_potrf_compact`

The Cholesky counterpart of `bench_geqrf_compact`: throughput of the batched
factorization of symmetric positive-definite matrices, on the tuned path --
column-major, lower triangle (`A = L L^T`) -- comparing

* `cqr_mkl_dpotrf_compact` -- this project's batched SIMD kernel,
* `mkl_dpotrf_compact` -- Intel MKL's batched compact kernel,
* `LAPACKE_dpotrf` -- conventional one matrix at a time.

As with `geqrf`, the pool is packed once and only the factorization is timed. It
reports GFLOP/s (the `n^3/3 + n^2/2 + n/6` LAPACK Cholesky flop count; the `n`
square roots are not counted, as in LAPACK's own timing), matrices/s, the pairwise
speedups, and a geometric-mean speedup over per-matrix LAPACK. Because the SPD
Cholesky factor is *unique* (positive diagonal), the error column is the compact
factor compared *elementwise* against `LAPACKE_dpotrf` (over the lower triangle) --
a sharper signal than a reconstruction residual. Unlike `geqrf`, `potrf` needs no
workspace, so there is no `lwork` query and no per-thread work array. The SPD pool
uses the cheap diagonally dominant `A_ii = 2n` form (SPD and well conditioned
without an `O(n^3)` `M^T M` product).

```
Usage:  bench_potrf_compact [--size-sweep=nmin:nmax[:stride]] [--simdlen=2|4|8]
        [nmat] [reps]                          (defaults: 512 matrices, 3 reps)
```

## `bench_qr_compact`

Throughput of solving many small square systems `A_v X_v = B_v` with the QR
pipeline (`X = R^-1 Q^T B`), comparing three ways to run the same solve:

* **MKL batched** -- `mkl_dgeqrf_compact` -> `cqr_mkl_dormqr_compact` -> `mkl_dtrsm_compact`,
* **cqr batched** -- `cqr_mkl_dgeqrf_compact` -> `cqr_mkl_dormqr_compact` -> `cqr_mkl_dtrsm_compact`,
* **unbatched** -- `LAPACKE_dgeqrf` -> `LAPACKE_dormqr` -> `cblas_dtrsm`.

The two batched paths run the identical compact pipeline from different libraries
(MKL has no compact `ormqr`, so `cqr_mkl_dormqr_compact` is shared by both), so
the cqr path runs the whole solve with no MKL compute kernel and the cqr-vs-MKL
ratio is the end-to-end open-vs-MKL comparison. There is a single right-hand side
per system (`nrhs = 1`); every path is checked against the known solution
`X == 1`, so the reported error is a forward error, not a comparison to LAPACK.
It reports matrices/s for each path plus the cqr-over-unbatched and cqr-over-MKL
geometric-mean speedups.

```
Usage:  bench_qr_compact [nmat] [reps]         (defaults: 1000 matrices, 3 reps)
```

## Common notes

* **Optional flags (the factorization benchmarks).** `--simdlen=2|4|8` forces a
  narrower interleave width than the host default (a wider-than-native or
  unsupported width is rejected); `--size-sweep=nmin:nmax[:stride]` switches to a
  cqr-only throughput scan -- raw best-pass time, GFLOP/s, and matrices/s per
  size, no cross-check -- to resolve the SIMD "staircase" finely.
* **The default size list** deliberately mixes sizes that are *not* multiples of
  the interleave width `V` (30, 45, 60, 105, 168, from 2-D/3-D RBF-FD stencils)
  with the round powers, so the remainder handling -- the staircase SIMD effect --
  is visible across the target small-to-medium range.
* **Reading the numbers.** On a host-tuned (`-march=native`) build over the target
  small-size range, the compact paths outrun per-matrix LAPACK and are competitive
  with MKL's compact kernels; per-matrix LAPACK's cache-blocked algorithm crosses
  ahead only at larger orders, where the unblocked compact kernels stop being the
  right tool. Throughput is reported as matrices/second (scientific notation) so
  it stays legible across the whole size range.
* **CTest.** Each benchmark is registered as an integration test on a small,
  padded pool (`33 1`), so the accuracy gate runs in CI:
  `bench_geqrf_compact_integration`, `bench_potrf_compact_integration`,
  `bench_qr_compact_integration`.
