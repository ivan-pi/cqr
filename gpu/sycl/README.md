# SYCL (oneAPI DPC++) GPU prototype

A first cut at running the compact batched QR on an Intel GPU with SYCL, per
[`gpu_batched_compact_design.md`](../../gpu_batched_compact_design.md)
(Strategy A: **one work-item per matrix**). It prototypes **both** compact
routines:

* `dgeqrf_compact` / `sgeqrf_compact` — the QR *factorization* (unblocked
  `geqr2` + scalar `larfg` per work-item), and
* `dormqr_compact` / `sormqr_compact` — the *apply-Q* step (scalar `dorm2r`).

`cqr_compact_sycl.cpp` re-implements the **same four C entry points** as the
portable CPU library (`src/cqr_compact.h`), with identical signatures and the
same LAPACK-style `info = -j` validation, but the arithmetic runs as a SYCL
kernel on the default device. Because the ABI matches exactly, the project's own
test suites validate the GPU kernels:

| Test | Source | What it drives |
|------|--------|----------------|
| `test_geqrf_sycl` | **unchanged** `src/test_cqr_geqrf_compact.cpp` | SYCL `geqrf` (H/tau/reconstruction) **and** SYCL `ormqr` (the check-3 solve), through the C API — the same suite, verbatim |
| `test_ormqr_sycl` | `test_ormqr_sycl.cpp` | SYCL `ormqr`: `QᵀB` vs scalar `dorm2r`, solve, and the `Q Qᵀ = I` round-trip. A C-API-routed mirror of `src/test_cqr_compact.cpp` (that CPU test calls the internal template directly, so it can't be relinked) |

Both gate against the project's numerical tolerances — the SYCL kernels pass the
same bars as the CPU kernels.

## Why this validates without an Intel GPU

The kernel logic is device-agnostic. A DPC++ install ships an **OpenCL CPU
device**, so `test_geqrf_sycl` / `test_ormqr_sycl` compile and run — and prove
**correctness** — on a machine with no GPU. Only *performance* on Xe needs real
hardware; that (sub-group = `V`, register blocking, a device-resident
`geqrf → ormqr → trsm` pipeline) is the design doc's Phase 2.

## Build & run

Needs a SYCL compiler (Intel oneAPI DPC++, `icpx`). If you don't have one,
`scripts/install-sycl-toolchain.sh` provisions it from conda-forge (no root, no
Intel apt repo). Then:

```sh
# from the repo root, inside the toolchain env (icpx on PATH)
CXX=icpx cmake -S . -B build-sycl -DCMAKE_BUILD_TYPE=Release \
      -DCQR_WITH_SYCL=ON -DCQR_WITH_MKL=OFF
cmake --build build-sycl --target test_geqrf_sycl test_ormqr_sycl -j
ctest --test-dir build-sycl -R sycl --output-on-failure
```

`sycl-ls` lists the available devices; set `ONEAPI_DEVICE_SELECTOR` (e.g.
`level_zero:gpu` or `opencl:cpu`) to pick one. Each run prints the device it
used on the first kernel call (`[cqr SYCL backend] device: ...`).

The `CQR_WITH_SYCL` option is **OFF by default**, so ordinary GCC/Clang builds
are untouched. `-DCQR_WITH_MKL=OFF` is optional but avoids requiring MKL for a
SYCL-only build.

## Benchmark: SYCL vs the vector-types kernel

`bench_ormqr_sycl_vs_vec.cpp` times apply-`Qᵀ` throughput of the SYCL kernel
against the CPU vector-types kernel (`ormqr_compact_group`) — same compact
layout, same batch, all cores for both — sweeping `n` up to 150 and
`nrhs ∈ {1,4,8}`. Built with `-DCQR_WITH_SYCL=ON` when OpenMP is found:

```sh
cmake --build build-sycl --target bench_ormqr_sycl_vs_vec
./build-sycl/bench_ormqr_sycl_vs_vec        # self-checks; non-zero exit on disagreement
```

**With no GPU present this runs on the OpenCL CPU device — a CPU-vs-CPU codegen
comparison, not the Xe story and not predictive of GPU throughput.** What it
pins down (one Xeon, AVX-512, 4 cores, `-O3 -march=native`, double):

| SYCL kernel | GFLOP/s | vs vector-types |
|-------------|---------|-----------------|
| plain `parallel_for` (no sub-group hint) | ~3 (flat) | **0.03–0.2×** |
| `reqd_sub_group_size(V)` + `JB=4` blocking (**shipped**) | ~4–63 | **≈ parity for `n≥30`** (0.6–1.3×) |

Two levers matter, both now in `cqr_compact_sycl.cpp`:

* **Sub-group = `V`.** A plain `parallel_for` runs essentially scalar on the CPU
  device (~3 GFLOP/s); pinning the sub-group to the interleave width `V` lets
  the runtime pack `V` work-items into one AVX-512 op — the batch-in-SIMD the
  vector-types kernel does by hand — a ~5–10× jump. `V` is the architecture
  vector size, so this is the shipped path.
* **`JB=4` RHS-column register blocking.** Reuse each reflector entry `A(i,kk)`
  across 4 RHS columns; this closes most of the remaining `nrhs=4/8` gap.

Two regimes explain the numbers:

* **Small-to-mid `n` (cache-resident, compute-bound):** the hand-tuned kernel
  wins, up to ~2×. Its AVX-512 + register blocking is exactly what pays off when
  the data fits in cache and the limit is arithmetic. Below that, fixed
  kernel-launch overhead sinks SYCL further (~0.2–0.3× at `n=10`).
* **Large `n` (working set spills cache, memory-bandwidth-bound):** the two
  converge to parity (~0.8–1.2×, noisy). Apply-`Q` has low arithmetic intensity
  — it streams the reflector panel and the RHS — so once it is bandwidth-bound
  the SIMD-compute edge stops mattering, and both kernels, touching the
  *identical* compact bytes, hit the same ceiling. Both GFLOP/s figures fall
  from ~40–90 (mid `n`) to ~20–50 (`n=150`), the memory-bound signature. Where
  SYCL edges slightly ahead here it is streaming codegen from the OpenCL JIT
  (most visible at `nrhs=1`, whose vec path is not register-blocked), not a
  compute win.

Both paths are built `-O3 -march=native`: the vector-types kernel via the host
compiler, the SYCL kernel via the OpenCL JIT (which targets the host ISA
regardless). The benchmark CMake target forces `-march=native` so the vec kernel
is never compared with one hand tied — the library itself still leaves `-march`
to the caller.

**Backend.** With no GPU present this runs on the OpenCL CPU device. On a real
Intel GPU the Level Zero backend is preferred (`ONEAPI_DEVICE_SELECTOR=level_zero:gpu`)
— lower kernel-launch latency, which is exactly what the small-`n` regime needs;
it has no device to bind to without a GPU (`sycl-ls` shows no `level_zero:*`
here).

`reqd_sub_group_size(V)` requires `V` to be a device-supported sub-group size.
The backend uses it whenever it is, and falls back to a plain `parallel_for`
only for widths that are not a hardware sub-group size (`V=2` anywhere; `V=4` on
Intel GPUs, whose minimum is 8) — which the portable test suite exercises but
real GPU deployment (`V ∈ {8,16,32}`) never hits.

## Scope of the prototype

The kernel pins the sub-group to the interleave width `V` (`reqd_sub_group_size`,
with the plain-`range` fallback noted above) and register-blocks 4 trailing /
RHS columns, matching the CPU kernel. Still correctness-first in its data
movement: a USM copy in/out per call, no device-resident
`geqrf → ormqr → trsm` pipeline (the Phase 2 performance work). `geqrf` handles
column- and row-major (LAPACK `layout`); `ormqr` follows the C API's `side='L'`,
column-major contract. Complex types are out of scope, matching the CPU library.
See the design doc for the remaining Phase 2 plan.

<!-- Assisted-by: Claude:claude-opus-4.8 -->
