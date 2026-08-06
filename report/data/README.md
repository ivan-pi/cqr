# Benchmark data for the report figures

Each figure is driven by a whitespace-separated `.dat` table. **The committed
files are synthetic placeholders** produced by `make_placeholder_data.py` so the
gnuplot → LaTeX pipeline renders while the draft is written. Replace them with
measured output before the numbers mean anything.

## Column contract

`geqrf_x86.dat` (Intel MKL Compact, x86):

```
n  cqr_gflops  cqr_mats_s  mkl_mats_s  lapack_mats_s  sp_cqr_lap  sp_mkl_lap  sp_cqr_mkl  relerr
```

`geqrf_arm.dat` (Arm Performance Libraries interleave-batch, Arm) — same layout
with the vendor columns renamed:

```
n  cqr_gflops  cqr_mats_s  armpl_mats_s  lapack_mats_s  sp_cqr_lap  sp_armpl_lap  sp_cqr_armpl  relerr
```

* `n` — square matrix order.
* `*_mats_s` — throughput in matrices/second (higher is better).
* `cqr_gflops` — cqr sustained rate, GFLOP/s (`nmat · geqrf_flops(n) / time`).
* `sp_*` — speedup ratios (see the header comment written into each file).
* `relerr` — elementwise `(H, tau)` error of cqr vs per-matrix LAPACK (a
  correctness gate, not performance).

The gnuplot scripts reference columns by index, so **keep the column order**.
Lines beginning with `#` are ignored by gnuplot.

## Producing real x86 numbers

`bench_geqrf_compact` (built from the repo root) already prints exactly these
columns. From a release build with host-tuned flags:

```sh
# from the repo root, after: cmake --build build
./build/bench_geqrf_compact 512 5 | sed -n '/^ *[0-9]/p' \
    | awk '{print $1, $2, $3, $5, $7, $9, $11, $13, $15}' > report/data/geqrf_x86.dat
```

Adjust the `awk` field map to the exact table your MKL build prints (the header
row lists the columns); the goal is the nine columns above, in order. Prepend a
`#`-comment line recording the CPU, MKL version, compiler flags, `nmat`, and
thread count so the figure is reproducible.

## Producing real Arm numbers

There is no ArmPL harness in the tree yet (see the paper's *Future work*). It
requires (i) building cqr for AArch64 so the GNU vector types lower to NEON/SVE,
and (ii) a driver that calls `armpl_?geqrf_interleave_batch` on the same pools,
converting ArmPL's `(nintl, ...)` interleave strides to/from the compact packs.
Emit the same nine columns into `geqrf_arm.dat`.
