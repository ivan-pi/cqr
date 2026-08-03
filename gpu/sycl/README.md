# SYCL (oneAPI DPC++) GPU prototype

Compact batched QR on Intel GPUs with SYCL, per
[`gpu_batched_compact_design.md`](../../gpu_batched_compact_design.md)
(Strategy A: **one work-item per matrix**, the sub-group pinned to the compact
interleave width `V`). The feature is a set of **device primitives** you compose
inside **your own** batch kernel — not a batched API that owns the launch.

## Primitives — the building blocks

[`cqr_compact_sycl.hpp`](cqr_compact_sycl.hpp) (namespace `cqr::gpu`,
header-only) gives one per-lane primitive per operation, each the work for **one
matrix** (slot `v` of compact group `g`):

| Primitive | Does |
|-----------|------|
| `geqrf_slot<T,V>(g, v, ap, lda, tau, m, n)` | QR-factorize one matrix in place |
| `ormqr_slot<T,V>(g, v, ap, lda, tau, bp, ldb, m, nrhs, k, trans)` | apply `Q`/`Qᵀ` to one RHS block |
| `trsm_upper_slot<T,V>(g, v, ap, lda, bp, ldb, n, nrhs)` | upper-triangular solve `R X = B` |
| `col_major` / `row_major` / `tau_at` | build a slot view for filling / reading |

**You own the launch:** one `parallel_for` over the batch, `V` work-items per
group with `reqd_sub_group_size(V)` so the `V` lanes of a sub-group are the `V`
interleaved matrices (coalesced accesses; one hardware SIMD vector). Inside, you
compose the primitives and — the point — **fuse your own steps around them**
(fill/pack, unpack, RHS build) in a single kernel, with no intermediate
global-memory passes and no per-step host↔device copies.

[`example_fused_qr_solve.cpp`](example_fused_qr_solve.cpp) does a full batched
`AX=B` solve in **one kernel** — fill → `geqrf` → `Qᵀ` → `trsm` → unpack:

```cpp
q.parallel_for(sycl::nd_range<1>{ngroups*V, V},
    [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(V)]] {
        const int g = it.get_group(0), v = it.get_local_id(0);
        auto A = cqr::gpu::col_major<double,V>(ap, g, v, n, n);
        auto B = cqr::gpu::col_major<double,V>(bp, g, v, n, nrhs);
        for (...) A(i,j) = fill(...);                       // your pack, fused in
        for (...) B(i,c) = build_rhs(...);
        cqr::gpu::geqrf_slot<double,V>(g, v, ap, n, tau, n, n);
        cqr::gpu::ormqr_slot<double,V>(g, v, ap, n, tau, bp, n, n, nrhs, n, true);
        cqr::gpu::trsm_upper_slot<double,V>(g, v, ap, n, bp, n, n, nrhs);
        for (...) xout[...] = B(i,c);                       // your unpack, fused in
    });
```

The only host↔device transfer is copying the result out; `A` and `B` are
generated on the device and consumed in place. For many small matrices this
fusion — one launch, data resident across steps — is where the throughput is,
and only the caller can orchestrate it. That is why the batch `parallel_for` is
the user's responsibility.

`bench_fused_solve.cpp` measures the payoff on the full `AX=B` workflow: the
fused single kernel vs the same solve run as three separate operations. On the
OpenCL CPU device (where launches are cheap and "copies" are host memcpys — a
**lower bound** on the GPU win) fusing beats three separate launches by up to
~1.6× at small `n` (falling to parity by `n≈64` as compute dominates the fixed
launch cost), and beats the naive three-separate-calls-with-copies pattern (what
the batched entry points do) by **~1.3–8×**, largest for small matrices. A real
GPU — kernel-dispatch latency plus PCIe transfers — amplifies both, precisely in
the small-matrix regime compact batching targets.

## Convenience launchers (and how the suites validate the primitives)

For callers that do **not** need fusion, [`cqr_compact_sycl.cpp`](cqr_compact_sycl.cpp)
wraps the primitives in the **same four portable C entry points** as the CPU
library (`src/cqr_compact.h`) — `d/sgeqrf_compact`, `d/sormqr_compact`, identical
signatures and `info = -j` validation — each owning one `parallel_for` plus a USM
copy in/out. That ABI match is also what lets the project's own suites validate
the primitives unchanged:

| Test | Source | What it drives |
|------|--------|----------------|
| `test_geqrf_sycl` | **unchanged** `src/test_cqr_geqrf_compact.cpp` | `geqrf_slot` (H/tau/reconstruction) **and** `ormqr_slot` (the check-3 solve), through the C API — the same suite, verbatim |
| `test_ormqr_sycl` | `test_ormqr_sycl.cpp` | `ormqr_slot`: `QᵀB` vs scalar `dorm2r`, solve, and the `Q Qᵀ = I` round-trip (a C-API-routed mirror of `src/test_cqr_compact.cpp`, which calls the internal template directly and can't be relinked) |
| `example_fused_qr_solve` | `example_fused_qr_solve.cpp` | all three primitives, fused, against a known `AX=B` solution |

All gate against the project's numerical tolerances — the SYCL primitives pass
the same bars as the CPU kernels.

## Why this validates without an Intel GPU

The kernel logic is device-agnostic. A DPC++ install ships an **OpenCL CPU
device**, so the suites compile and run — proving **correctness** — on a machine
with no GPU. Only *performance* on Xe needs real hardware.

## Build & run

Needs a SYCL compiler (Intel oneAPI DPC++, `icpx`). If you don't have one,
`scripts/install-sycl-toolchain.sh` provisions it from conda-forge (no root, no
Intel apt repo). Then:

```sh
# from the repo root, inside the toolchain env (icpx on PATH)
CXX=icpx cmake -S . -B build-sycl -DCMAKE_BUILD_TYPE=Release \
      -DCQR_WITH_SYCL=ON -DCQR_WITH_MKL=OFF
cmake --build build-sycl -j
ctest --test-dir build-sycl -R "sycl|fused" --output-on-failure
```

`sycl-ls` lists the devices; set `ONEAPI_DEVICE_SELECTOR` (e.g. `level_zero:gpu`
or `opencl:cpu`) to pick one. The `CQR_WITH_SYCL` option is **OFF by default**,
so ordinary GCC/Clang builds are untouched.

## Benchmark: SYCL vs the vector-types kernel

`bench_ormqr_sycl_vs_vec.cpp` times apply-`Qᵀ` throughput of the SYCL kernel
against the CPU vector-types kernel (`ormqr_compact_group`) — same compact
layout, same batch, all cores for both — sweeping `n` up to 150 and
`nrhs ∈ {1,4,8}`:

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

Two levers matter, both in the primitives and launchers:

* **Sub-group = `V`.** A plain `parallel_for` runs essentially scalar on the CPU
  device (~3 GFLOP/s); pinning the sub-group to the interleave width `V` lets
  the runtime pack `V` work-items into one AVX-512 op — the batch-in-SIMD the
  vector-types kernel does by hand — a ~5–10× jump. `V` is the architecture
  vector size, so this is the shipped path.
* **`JB=4` register blocking.** Reuse each reflector entry `A(i,kk)` across 4
  columns; closes most of the remaining `nrhs=4/8` gap.

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
  SYCL edges slightly ahead it is streaming codegen from the OpenCL JIT (most
  visible at `nrhs=1`, whose vec path is not register-blocked), not a compute
  win.

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

## Scope

`reqd_sub_group_size(V)` requires `V` to be a device-supported sub-group size.
The launchers use it whenever it is, and fall back to a plain `parallel_for` only
for widths that are not a hardware sub-group size (`V=2` anywhere; `V=4` on Intel
GPUs, whose minimum is 8) — which the portable test suite exercises but real GPU
deployment (`V ∈ {8,16,32}`) never hits. The primitives themselves are
launch-agnostic; you pick the sub-group size in your kernel.

`geqrf` handles column- and row-major (LAPACK `layout`); `ormqr`/`trsm` follow
the `side='L'`, column-major contract. Real types only (`float`/`double`),
matching the CPU library. The remaining Phase 2 performance work — keeping a
whole `geqrf → ormqr → trsm` pipeline device-resident (the fused example is the
template) and Xe tuning — is in the design doc.

<!-- Assisted-by: Claude:claude-opus-4.8 -->
