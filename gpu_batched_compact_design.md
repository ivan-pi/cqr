# Design Document: Batched Compact QR on Intel GPUs

> Assisted-by: Claude:claude-opus-4.8

## 1. Scope and Goal

This document explores taking the compact (interleaved-batch) apply-`Q`
kernel -- `cqr_mkl_?ormqr_compact` and the portable `?ormqr_compact`
(`cqr_compact.h`) -- and the surrounding QR-solve pipeline
(`?geqrf_compact` -> apply `Q^T` -> `?trsm_compact`) onto **Intel GPUs**
(the Xe family: integrated Iris Xe, Arc discrete, and Data Center GPU Max /
Ponte Vecchio).

The workload -- many *small* matrices (order ~10-100) solved as a batch -- is
a throughput problem, which is what a GPU's wide array of execution units is
built for. The central question this document answers is *how* to place that
batch onto the hardware, and *which* programming model and toolchain to reach
for first.

**In scope (phase 1):** the `side='L'`, column-major apply-`Q^T` step -- the
solver hot path -- in FP64/FP32.

**Out of scope (phase 1):** complex precisions, `side='R'`, and row-major
layout (the CPU kernel already covers these via the stride-generalized path;
the GPU port starts narrow and widens later).

The key finding, developed in §2, is that **the compact layout this library
already uses is the memory layout a GPU wants** -- so the port is largely a
change of execution model, not of data format.

## 2. The Enabling Property: Compact Layout Is Already Coalesced

The compact format stores element `(i,j)` of `V` consecutive matrices
contiguously. From the addressing (`cqr_compact.hpp`, "Compact storage
convention"):

```
A_v(i,j)  = ap [ g*ldap*k*V    + (j*ldap + i)*V + v ]   group g = idx/V, slot v = idx%V
tau_v(kk) = taup[ g*k*V         +  kk*V          + v ]
B_v(i,j)  = bp [ g*ldbp*nrhs*V  + (j*ldbp + i)*V + v ]
```

For a fixed group `g` and a fixed element `(i,j)`, walking the slot
`v = 0 .. V-1` walks **contiguous** addresses (`... + v`). This one fact has
two different payoffs on the two kinds of hardware:

* **On the CPU** (what the library does today) the contiguity lets one SIMD
  register hold the `(i,j)` element of all `V` matrices -- the GNU
  `vector_size` pack. The batch is the *vector lane* dimension.

* **On the GPU**, under the single-work-item-per-matrix ("me-style" / SPMD)
  model, if work-item `v` in a sub-group owns matrix-slot `v`, then all `V`
  work-items reading `A_v(i,j)` for the same `(i,j)` touch **contiguous
  memory** -- a fully **coalesced** load, the single most important factor for
  GPU memory throughput. The batch is the *work-item* dimension, and the
  hardware SIMD (the sub-group) re-vectorizes it.

The alternative batched layout -- *strided-batch*, each matrix stored
contiguously and matrices laid back to back (what oneMKL's GPU
`geqrf_batch` consumes) -- is the opposite:

| Layout | Address of `A_v(i,j)` vs `A_{v+1}(i,j)` | Work-item-per-matrix access |
|--------|------------------------------------------|-----------------------------|
| **Compact (interleaved)** | 1 element apart | **coalesced** |
| Strided-batch | `ldm*n` elements apart | **uncoalesced** (huge stride) |

So the interleaved format is not a CPU-only trick to be undone on the GPU --
it is precisely the layout that makes the natural GPU mapping coalesce.

**Consequence for `V`.** On the GPU, the compact interleave width `V` stops
meaning "CPU SIMD register width" (2/4/8 for FP64 on SSE/AVX/AVX-512) and
starts meaning **sub-group size** -- 8, 16, or 32 on Xe. The portable
`?ormqr_compact` C API already takes `V` as an explicit runtime argument
(2/4/8/16), so nothing in the interface has to change; the GPU path simply
packs with a GPU-appropriate `V`. (Widths 2 and 4 are below the Xe minimum
sub-group size and are not useful there; `V=32` for FP32 / SIMD32 would need
the one-line relaxation of the kernel's `V==2||4||8||16` `static_assert`.)

## 3. The Design Space: Placing the Batch on the Hardware

There is one primary axis of choice, orthogonal to the programming model:
**how a matrix maps to the execution hierarchy.**

### 3.1 Strategy A -- work-item = matrix (batch-parallel)

A sub-group of `V` work-items processes `V` matrices; lane `v` runs the entire
unblocked `dorm2r` sweep for its own matrix, scalar-style. The compact layout
(§2) makes every load/store coalesced.

* **Pros:** no barriers, no cross-lane reductions; reuses the existing scalar
  kernel body almost verbatim (`double` in place of the `V`-wide pack, indices
  gain a `*V + v`); the compact *uniform-dimensions* rule means **zero control
  divergence** across the sub-group. Register-blocking of RHS columns (`JB=4`
  in the CPU kernel) carries over as per-work-item register reuse.
* **Cons:** each work-item's working set is one matrix; for large `m` this
  raises register pressure and can cap occupancy.
* **Verdict:** the right first target for the compact regime (many small
  matrices). It is the most direct lift of the current code and the least
  likely to harbor an untestable indexing bug.

### 3.2 Strategy B -- sub-group / work-group = matrix (cooperative)

A sub-group (or work-group) cooperates on *one* matrix: the lanes split the
reflector's row range, the dot product `w = v^T c` becomes a **sub-group
reduction**, and the axpy `c -= tau*w*v` is lane-parallel over rows.

* **Pros:** more parallelism per matrix; the way to keep a big GPU busy when
  matrices grow (hundreds of rows) and the batch alone no longer supplies
  enough independent work.
* **Cons:** reductions and (for work-group scope) barriers and shared-local
  memory; the coalescing story is different and the code diverges from the CPU
  kernel.
* **Verdict:** a later option, gated on measured occupancy for larger `m`.

Strategy A is the recommendation for phase 1; B is a documented escalation
path.

### 3.3 Interface: primitives, not a batched call

A second design axis is *who owns the launch*. A batched library call
(`dgeqrf_compact(...)` that allocates, copies, launches its own `parallel_for`,
and copies back) is convenient but it forecloses the optimization that actually
matters for many small matrices on a GPU: **fusion**. Each such call is a launch
plus a host<->device round trip; a real solve (`geqrf -> apply Q^T -> trsm`, with
a pack before and an unpack after) becomes several launches and a stack of
copies, dominated by launch latency and PCIe traffic for small `n`.

So the shipped shape is a set of **device-callable primitives** -- the per-lane
work for one matrix (`geqrf_slot`, `ormqr_slot`, `trsm_upper_slot`) -- and the
batch `parallel_for` is the **caller's** responsibility. The user writes one
kernel, `V` work-items per group with `reqd_sub_group_size(V)`, and composes the
primitives *together with their own fill/pack/unpack*, so the whole pipeline runs
in a single launch with the data resident across steps and no intermediate
global-memory passes. This is the SPMD-native way to express the workload and
the realistic usage pattern; the convenience batched entry points are kept as
thin wrappers over the same primitives (and as the vehicle for validating them
against the existing test suites).

The primitives use no sub-group collectives -- each lane owns an independent
matrix (Strategy A), so the sub-group is purely the SIMD packing. Strategy B
would instead make the primitives cooperative (sub-group reductions over one
matrix); the interface -- caller-owned launch, composable per-tile primitives --
is the same.

## 4. Programming Models and Compiler Toolchains

**Common substrate.** Regardless of the model, on an Intel GPU everything
lowers through the **Intel Graphics Compiler (IGC)** to Xe ISA and runs on the
**Intel Compute Runtime (NEO)** via either the **Level Zero** or the
**OpenCL** backend. `ocloc` is the offline (ahead-of-time) front-end to IGC.
So the four models differ mostly in the *authoring* language and *host* API,
not in the ultimate device code path.

A second axis cuts across all four: **JIT vs AOT.** JIT ships SPIR-V and lets
the driver specialize for the device at run time (portable across Xe
generations); AOT (`ocloc`, `-fsycl-targets=spir64_gen`, ISPC's device
targets) bakes in Xe ISA for a named device (faster first launch, pinned to
that part).

### 4.1 SYCL / DPC++ -- recommended

* **Style:** SPMD "me-style" -- write one work-item's kernel; the sub-group is
  the hardware SIMD. Maps directly onto Strategy A.
* **Kernel mapping:** `nd_range` with `[[sycl::reqd_sub_group_size(V)]]`; the
  existing scalar `dorm2r` body ports line for line. Coalesced by construction.
* **Toolchain:** Intel oneAPI DPC++/C++ compiler, `icpx -fsycl`. JIT to SPIR-V
  by default (`-fsycl-targets=spir64`), executed on Level Zero; AOT with
  `-fsycl-targets=spir64_gen -Xs "-device <pvc|dg2|acm-g10|...>"` (or the
  `intel_gpu_*` shorthands). Device picked at run time with
  `sycl::gpu_selector_v` / `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`.
* **Open-source variants:** the upstream `intel/llvm` compiler (same `-fsycl`),
  and AdaptiveCpp (formerly hipSYCL / Open SYCL) targeting Intel via Level Zero.
* **Maturity on Intel GPU:** **best.** This is Intel's flagship model, with
  first-class tooling (VTune, Advisor, `onetrace`, `unitrace`) and oneMKL's
  own GPU interface.

### 4.2 OpenCL

* **Style:** the same SPMD model as SYCL, one abstraction layer lower.
* **Kernel mapping:** essentially the SYCL kernel body as an OpenCL C string;
  `reqd_work_group_size` / `intel_reqd_sub_group_size` express the same intent.
* **Toolchain:** host code with any C/C++ compiler linked against the ICD
  loader (`-lOpenCL`); kernels JIT-compiled at run time by the Intel driver
  (IGC), or AOT via `ocloc compile`. Intel exposes OpenCL 3.0 on Xe.
* **Maturity on Intel GPU:** solid and stable, but you pay in host boilerplate
  (context/queue/program/buffer plumbing) that SYCL hides. Best reserved for a
  non-Intel portability fallback or a C-only host; SYCL supersedes it for
  Intel-first work.

### 4.3 OpenMP Target Offload

* **Style:** directive / explicit loop. You keep the `for g` / `for kk` loops
  and annotate the batch loop; the compiler maps iterations to work-items. It
  is "explicit loop, implicit lanes" -- the batch loop is the *parallel*
  (thread) dimension, not a hand-packed SIMD dimension.
* **Kernel mapping:** `#pragma omp target teams distribute parallel for` over
  the groups, with the compact buffers `map`-ed to the device. Lowest code
  churn from the CPU version; the compiler owns coalescing and occupancy, which
  usually means the lowest performance ceiling of the three GPU-native options
  unless carefully tuned.
* **Toolchain:** Intel `icx`/`icpx` with `-fiopenmp -fopenmp-targets=spir64`
  (JIT) or `spir64_gen` (AOT); runs on Level Zero via `libomptarget`. Upstream
  LLVM/Clang also offers Level Zero/SPIR-V offload
  (`-fopenmp --offload-arch=...`), less mature than Intel's. **Note:** GCC's
  `libgomp` offloads to NVPTX and AMD GCN, **not** to Intel Xe -- so GCC is not
  an Intel-GPU OpenMP path; use `icx`.
* **Maturity on Intel GPU:** good and improving; the pragmatic choice if the
  goal is to keep a single directive-annotated source shared with the CPU.

### 4.4 ISPC (including experimental Intel Xe GPU support)

* **Style:** SPMD "me-style" -- and this is worth stating plainly, because it
  is easy to file ISPC alongside OpenMP as a "write the loop" model. It is not:
  ISPC is SPMD-on-SIMD. You write one program instance's scalar-looking body
  and `foreach` / `programIndex` maps instances onto lanes -- the *same* mental
  model as SYCL/OpenCL. What distinguishes ISPC is its *target*, not its style.
  Its compact-batch kernel (`foreach` over the batch, `programIndex` = slot) is
  arguably a cleaner expression of the current CPU kernel than the
  `vector_size` packs.
* **Experimental Xe GPU support:** ISPC has an experimental **"ISPC for GPU"
  (Xe)** backend. Device code is compiled to SPIR-V / zebin for Xe targets
  (e.g. `--target=gen9-x16`, `--target=xelp-x16`, `--target=xelpg-x16`,
  `--emit-spirv` / `--emit-zebin`), lowered through IGC like the others, and
  launched from the host through the **ISPC Run Time (`ispcrt`)**, which
  abstracts device allocation and kernel launch over oneAPI Level Zero (and a
  CPU fallback). This makes a *single* ISPC source targetable at both CPU SIMD
  and Intel GPU.
* **Toolchain:** the `ispc` compiler plus `libispcrt`; host code in any C/C++
  compiler.
* **Maturity on Intel GPU:** **experimental** -- usable and interesting for a
  shared CPU/GPU kernel, but the least battle-tested of the four for Xe. Best
  treated as a portability/ergonomics spike rather than the phase-1 path.

### 4.5 Summary

| Model | Style | Intel-GPU toolchain | Runtime | Xe maturity |
|-------|-------|---------------------|---------|-------------|
| **SYCL / DPC++** | SPMD ("me") | `icpx -fsycl` (`spir64` / `spir64_gen`) | Level Zero / OpenCL | **best** (flagship) |
| **OpenCL** | SPMD ("me") | `-lOpenCL` + JIT/`ocloc` | OpenCL (NEO) | solid, verbose host |
| **OpenMP target** | directive / loop | `icx -fiopenmp -fopenmp-targets=spir64` (not GCC) | Level Zero | good, lower ceiling |
| **ISPC** | SPMD ("me") | `ispc --target=xe* --emit-spirv` + `ispcrt` | Level Zero | **experimental** |

All four converge on IGC -> Xe ISA over Level Zero / NEO; `ocloc` is the shared
AOT front-end.

## 5. Cross-Cutting Choices: `V`, Precision, Occupancy

* **`V` = sub-group size.** Pick `V` from `{8, 16, 32}` to match a whole Xe
  sub-group so each sub-group is exactly one compact group and every access
  coalesces. FP32 can use SIMD32 (`V=32`); FP64 typically SIMD8/16.
* **FP64 availability is a hardware gate, not a given.** Native, fast FP64 is a
  **Data Center GPU Max (Ponte Vecchio)** feature. On client parts -- Arc
  discrete and integrated Iris Xe -- FP64 is limited or software-emulated and
  can be dramatically slower (or unavailable). Any FP64 batched-solve ambition
  must name the target part; on client GPUs, plan for FP32 (or a
  single-precision-with-correction scheme) as the default.
* **Work-group shape.** The simplest correct mapping is one work-group = one
  compact group (`local = V`). For occupancy, a work-group can hold several
  sub-groups (`local = c*V`), each sub-group still one compact group so
  coalescing is preserved; the multiplier `c` is a tuning knob.
* **Register blocking.** Keep the CPU kernel's `JB=4` RHS-column blocking: it
  reuses each loaded reflector entry `A(i,kk)` across 4 columns, halving the
  dominant `A` traffic -- a per-work-item register optimization on the GPU too.

## 6. Correctness Strategy

The repo's culture is a hard correctness gate against dense LAPACK
equivalents. The GPU port keeps that, cheaply:

* **CPU kernel as oracle.** The portable `?ormqr_compact` (`cqr_compact.h`) is
  already validated against a scalar `dorm2r` and, through the MKL suite,
  against dense LAPACK. Feeding the *same* packed `(A, tau, B)` to the CPU
  kernel and the GPU kernel and comparing element-wise is a self-contained test
  that needs only a SYCL compiler -- **no MKL**. (This mirrors the structure of
  `src/test_cqr_compact.cpp`: `pack_compact` -> run -> `unpack_compact` ->
  `max_abs_diff`.)
* **Oracle-free invariant.** Applying `'T'` then `'N'` must recover the input
  (`Q Q^T = I`) -- a strong check independent of any reference.
* **Tolerances.** Reuse the existing gates -- the isolated application-residual
  gate (`~20 * eps` scaled by the reflector-axis length) and the end-to-end
  forward-error / residual gate (`~100 * n * eps`) -- but expect host-vs-device
  differences from FMA contraction and reassociation; compare with a *relative*
  gate rather than bit-exactness.
* **Padding.** Padded slots of a partial last group carry identity
  factorizations (`tau=0`), so applying them is a no-op -- the sub-group runs
  fully unmasked with no special-casing, exactly as on the CPU.

## 7. Phased Plan

* **Phase 0 -- baseline.** Measure the current OpenMP CPU pipeline
  (`bench_qr_compact`) as the reference number. Survey oneMKL's GPU batched
  LAPACK (`geqrf_batch`, strided/group batch) as an interop option -- and note
  that it exposes **no compact-format apply-`Q`** on the GPU, so the very gap
  this project fills on the CPU persists on the GPU.
* **Phase 1 -- SYCL primitives + fused example (done).** `gpu/sycl/` provides
  device-callable **primitives** -- `geqrf_slot`, `ormqr_slot`,
  `trsm_upper_slot` (header `cqr_compact_sycl.hpp`; Strategy A, one work-item per
  matrix, sub-group pinned to `V` per §5, `JB=4` register-blocked) -- that the
  caller composes inside a single batch `parallel_for`, fusing fill / pack /
  unpack around them (see §3.3 and `example_fused_qr_solve.cpp`, a full `AX=B`
  solve in one kernel). Thin C launchers wrap the primitives in the same four
  portable entry points, which lets the project's own suites validate them
  unchanged -- the unchanged `src/test_cqr_geqrf_compact.cpp` plus a
  C-API-routed `ormqr` mirror, gating at the CPU tolerances. Validated on the
  OpenCL CPU device (no Intel GPU required); there the sub-group + blocked kernel
  reaches rough parity with the hand-tuned vector-types kernel for `n >= 30`
  (see `gpu/sycl/README.md`). OFF-by-default `CQR_WITH_SYCL`, so the existing
  build is untouched.
* **Phase 2 -- full pipeline + tuning.** Get `geqrf` and `trsm` onto the GPU,
  either by porting the compact versions or by interop with oneMKL's batched
  GPU LAPACK on a strided layout (accepting a repack). Tune `V`, work-group
  shape, and register blocking; profile with VTune / `unitrace`.
* **Phase 3 -- portability spikes.** Re-express the same kernel in OpenCL
  (near-identical body), OpenMP target (directive), and ISPC-for-Xe (SPMD) to
  compare ergonomics and IGC codegen, informing which model to standardize on.

## 8. Toolchain Quick Reference

```sh
# --- SYCL / DPC++ (recommended) ---
icpx -fsycl -O3 -std=c++17 kernel.cpp -o app                 # JIT (SPIR-V, Level Zero)
icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device pvc" ...  # AOT for GPU Max
ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./app                  # pick the GPU at run time

# --- OpenCL ---
cc host.c -lOpenCL -o app                                    # host; kernels JIT at run time
ocloc compile -file k.cl -device pvc                         # optional AOT via IGC

# --- OpenMP target offload (Intel; NOT GCC for Xe) ---
icx -fiopenmp -fopenmp-targets=spir64 -O3 src.c -o app

# --- ISPC (experimental Xe GPU) ---
ispc --target=xelp-x16 --emit-spirv kernel.ispc -o kernel.spv   # device module
#   + host code using the ISPC Run Time (ispcrt) over Level Zero
```

## 9. Risks and Open Questions

* **FP64 target part.** Is the intended device a GPU Max (native FP64) or a
  client Arc/iGPU (emulated/limited)? This decides whether FP64 batched solves
  are viable at all, and whether FP32 must be the default.
* **Port all three, or reuse oneMKL for two?** The CPU story is "MKL supplies
  `geqrf`/`trsm`; this project supplies the missing `ormqr`." The cleanest GPU
  analogue may be the same division of labor -- but oneMKL's GPU batch is
  strided, not compact, so bridging it means a layout repack. Porting all three
  compact kernels keeps one layout end to end.
* **Sub-group size portability.** A `reqd_sub_group_size(V)` pins the kernel to
  a width the device must support; `V=32` is not available on every Xe
  generation. A small dispatch over supported widths (as the CPU code already
  does over `V`) keeps it portable.
* **Numerical reproducibility.** Host-vs-device and run-to-run FP differences
  from FMA/reassociation; relative gates and, if bit-reproducibility is ever
  required, `-ffp-model=precise`-style controls.
