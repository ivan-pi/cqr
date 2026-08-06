# Benchmark collection for the report

Publication-grade measurement of the compact-QR factorization, built on
[Google Benchmark](https://github.com/google/benchmark). This is separate from
the repo's `examples/bench_geqrf_compact` (which stays the fast, self-checking
CTest gate): this driver adds warm-up, automatic iteration tuning, statistical
repetitions (mean/median/stddev/CV), and machine-readable JSON — what you want
for the final numbers on a quiet, frequency-pinned node.

```
bench_geqrf_collect.cpp   Google Benchmark driver (cqr vs MKL vs per-matrix LAPACK)
json_to_dat.py            GBench JSON  ->  report/data/*.dat  (the 9-column contract)
run_x86.sh                build + run (repetitions) + provenance + convert, for x86/MKL
results/                  raw JSON + provenance records (created on first run)
```

## Build

Google Benchmark and (for the x86 driver) Intel MKL are required. Enable the
CMake option from the repo root:

```sh
cmake -S . -B build -DBLA_VENDOR=Intel10_64lp_seq -DCMAKE_BUILD_TYPE=Release \
      -DCQR_BUILD_REPORT_BENCH=ON -DCMAKE_CXX_FLAGS="-O3 -march=native"
cmake --build build -j --target bench_geqrf_collect
```

`-march=native` matters: it lets the portable SIMD kernels use the host's widest
vectors, the same width MKL selects at runtime. Standalone compile line, if you
prefer not to touch the build:

```sh
g++ -std=c++17 -O3 -march=native -fopenmp -I src -I /usr/include/mkl \
    report/bench/bench_geqrf_collect.cpp \
    src/cqr_mkl_ext.cpp src/cqr_mkl_geqrf.cpp \
    src/cqr_compact_dispatch.cpp src/cqr_geqrf_compact_dispatch.cpp \
    -lbenchmark -lmkl_rt -lpthread -o build/bench_geqrf_collect
```

## Run

The scripted path (build if needed, record provenance, run, convert):

```sh
REPS=15 CQR_NMAT=512 OMP_NUM_THREADS=$(nproc) \
  OMP_PROC_BIND=close OMP_PLACES=cores \
  report/bench/run_x86.sh
```

It writes `results/geqrf_x86_<stamp>.json` (+ a `.provenance.txt`) and updates
`report/data/geqrf_x86.dat`. Then rebuild the figures: `(cd report && make figures)`.

Running the collector by hand instead:

```sh
CQR_NMAT=512 ./build/bench_geqrf_collect \
  --benchmark_repetitions=15 --benchmark_report_aggregates_only=true \
  --benchmark_format=json --benchmark_out=results.json
python3 report/bench/json_to_dat.py results.json --vendor mkl -o report/data/geqrf_x86.dat
```

Batch size is `CQR_NMAT` (env var, default 512); everything else is a standard
Google Benchmark flag. The driver pins MKL to one thread and takes its batch
parallelism from the OpenMP outer loop over `V`-groups.

## Stable measurements

For the numbers that go in the paper, on the empty/pinned node:

* `sudo cpupower frequency-set -g performance` and disable turbo
  (`echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo`);
* `export OMP_PROC_BIND=close OMP_PLACES=cores` and set `OMP_NUM_THREADS`;
* raise `--benchmark_repetitions` until the reported CV is small (single-digit %);
* keep the `results/*.provenance.txt` next to the data so the run is reproducible.

## Arm / ArmPL (future work)

There is no AArch64 driver yet — it is the paper's stated future work. The x86
driver leans on MKL's compact helpers (`mkl_?gepack_compact`,
`mkl_get_format_compact`, …) for packing and the format token, which do not
exist on Arm. An Arm collector needs:

1. **cqr built for AArch64**, so the GNU vector types lower to NEON/SVE (the
   kernel source is already ISA-agnostic; only the build target changes).
2. **A packing layer** that lays the pools out in ArmPL's interleave-batch
   convention (inner/outer/batch strides) and calls
   `armpl_?geqrf_interleave_batch`, with a matching driver for cqr's portable
   `dgeqrf_compact(V, …)` C API on the same buffers.
3. Emit the **same nine columns** into `geqrf_arm.dat` (use `--vendor armpl`);
   the figures and paper then re-render with no edits.
