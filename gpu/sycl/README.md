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

## Scope of the prototype

Correctness-first, deliberately unoptimized: a plain `range` `parallel_for` (no
sub-group pinning), no RHS-column register blocking, and a USM copy in/out per
call. `geqrf` handles column- and row-major (LAPACK `layout`); `ormqr` follows
the C API's `side='L'`, column-major contract. Complex types are out of scope,
matching the CPU library. See the design doc for the Phase 2 performance plan.

<!-- Assisted-by: Claude:claude-opus-4.8 -->
