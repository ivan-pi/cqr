# Benchmarks

> Assisted-by: Claude:claude-opus-4.8

Throughput benchmarks for the compact batched kernels, alongside the programs
they drive. Each runs the *same* math three ways -- this project's open compact
kernels, MKL's own compact kernels, and the conventional one-matrix-at-a-time
LAPACK path -- over pools of many small matrices, reporting per-size throughput
and a geometric-mean speedup. Each also cross-checks its result against per-matrix
LAPACK, so it doubles as an integration test (CTest-registered on a small pool).

| Program | Measures | Compares |
|---------|----------|----------|
| [`bench_geqrf_compact`](#bench_geqrf_compact) | QR *factorization* | `cqr_mkl_dgeqrf_compact` vs `mkl_dgeqrf_compact` vs `LAPACKE_dgeqrf` |
| [`bench_potrf_compact`](#bench_potrf_compact) | Cholesky *factorization* (SPD) | `cqr_mkl_dpotrf_compact` vs `mkl_dpotrf_compact` vs `LAPACKE_dpotrf` |
| [`bench_qr_compact`](#bench_qr_compact) | end-to-end QR *solve* `AX = B` | fully-open compact pipeline vs MKL's pipeline vs per-matrix LAPACK |

The worked, self-validating solver `solve_qr_compact` (not a benchmark) lives in
the same folder; see the top-level [README](../README.md).

## Running them

The benchmarks are built by the standard MKL build (see the
[README](../README.md) / [AGENTS.md](../AGENTS.md)). From a configured tree:

```sh
cmake --build build -j
./build/bench_geqrf_compact      # QR factorization
./build/bench_potrf_compact      # Cholesky (SPD) factorization
./build/bench_qr_compact         # end-to-end QR solve
```

**Build with `-march=native` for a fair comparison.** This library sets no
`-march` of its own, so a default build emits only the baseline ISA while MKL's
compact kernels dispatch to the host's widest vectors (AVX-512) at runtime -- an
unfair matchup. Pass host-tuned flags so the open kernels emit the full width:

```sh
cmake -S . -B build -DBLA_VENDOR=Intel10_64lp_seq -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-O3 -march=native"
```

Correctness (the gate each benchmark carries) is independent of these flags; only
throughput changes. The outer loop over the pool is parallelized with OpenMP when
available -- the intended usage -- with MKL's own threading pinned to 1.

## `bench_geqrf_compact`

Throughput of the QR factorization over pools of small square matrices. To
measure the kernels rather than data movement, the pool is packed once and only
the factorization is timed (the destroyed input restored, untimed, between
passes). Reports GFLOP/s (the standard `2mn^2 - (2/3)n^3` count), matrices/s per
path, the pairwise speedups, and a geometric-mean speedup over per-matrix LAPACK;
the error column is the compact `(H, tau)` elementwise vs a fresh `LAPACKE_dgeqrf`.

```
bench_geqrf_compact [--size-sweep=nmin:nmax[:stride]] [--simdlen=2|4|8] [nmat] [reps]
```

## `bench_potrf_compact`

The Cholesky counterpart, on the tuned column-major lower path (`A = L L^T`).
Same shape as `bench_geqrf_compact`, with three differences: `potrf` needs no
workspace; GFLOP/s uses the `n^3/3 + n^2/2 + n/6` Cholesky count (the `n` square
roots uncounted, as in LAPACK's own timing); and, since the SPD factor is *unique*
(positive diagonal), the error column is the compact factor compared elementwise
over the lower triangle vs `LAPACKE_dpotrf` -- sharper than a residual. The SPD
pool uses the cheap diagonally dominant `A_ii = 2n` form (SPD and well conditioned
without an `O(n^3)` `M^T M`).

```
bench_potrf_compact [--size-sweep=nmin:nmax[:stride]] [--simdlen=2|4|8] [nmat] [reps]
```

## `bench_qr_compact`

Throughput of the end-to-end solve of many systems `A_v X_v = B_v` via QR
(`X = R^-1 Q^T B`, single RHS), three ways:

* **MKL batched** -- `mkl_dgeqrf_compact` -> `cqr_mkl_dormqr_compact` -> `mkl_dtrsm_compact`
* **cqr batched** -- `cqr_mkl_dgeqrf_compact` -> `cqr_mkl_dormqr_compact` -> `cqr_mkl_dtrsm_compact`
* **unbatched** -- `LAPACKE_dgeqrf` -> `LAPACKE_dormqr` -> `cblas_dtrsm`

The two batched paths share `cqr_mkl_dormqr_compact` (MKL ships no compact
`ormqr`), so the cqr path runs the whole solve with *no* MKL compute kernel and
the cqr-vs-MKL ratio is the end-to-end open-vs-MKL comparison. Every path is
checked against the known solution `X == 1`, so the reported error is a forward
error, not a comparison to LAPACK.

```
bench_qr_compact [nmat] [reps]
```

## Notes

* **Defaults:** 512 matrices / 3 reps for the factorization benchmarks, 1000 / 3
  for `bench_qr_compact`.
* **Flags (factorization benchmarks).** `--simdlen=2|4|8` forces a narrower
  interleave width than the host default (a wider-than-native width is rejected);
  `--size-sweep=nmin:nmax[:stride]` switches to a cqr-only throughput scan (no
  cross-check) to resolve the SIMD "staircase" finely.
* **Size list.** The default deliberately mixes sizes that are *not* multiples of
  the interleave width `V` (30, 45, 60, 105, 168) with round powers, so the SIMD
  remainder handling stays visible across the target small-to-medium range.
* **Reading the numbers.** On a `-march=native` build over the small-size range,
  the compact paths outrun per-matrix LAPACK and are competitive with MKL's
  compact kernels; LAPACK's cache-blocked algorithm crosses ahead only at larger
  orders, where the unblocked compact kernels stop being the right tool.
