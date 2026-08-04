# ISPC prototype — compact QR pipeline

A prototype reimplementation of the **cqr** batched QR pipeline in
[Intel ISPC](https://ispc.github.io/) (the Implicit SPMD Program Compiler),
built to answer one question: **what does ISPC buy us over the hand-written GNU
`vector_size` kernels** in `../src`, on the very same algorithm, data layout, and
hardware?

It implements the full `AX = B` solve — `DGEQRF` → `DORMQR` → `DTRSM` — over
Intel MKL's Compact (interleaved) format, exposes the kernels as `extern "C"`
drop-in siblings of the portable C API (`../src/cqr_compact.h`), validates them
against the existing kernels and MKL, and benchmarks all of them head to head.

## The idea: a compact pack *is* a `varying`

The MKL Compact format stores element `(i,j)` of the `V` interleaved matrices of
a group **contiguously**:

```
ap[ g*ldap*ncol*V + (j*ldap + i)*V + v ]     v = 0..V-1   (lane = matrix)
```

In ISPC a `varying double` is, by definition, `programCount` consecutive doubles
in memory — one per gang lane. So when `V == programCount`, **a compact pack of
`V` matrices' element `(i,j)` is exactly one `varying double`**. Casting a
group's base pointer to `varying double * uniform` and indexing by the linear
element offset `(j*ldap + i)` loads or stores that whole pack in a single aligned
vector op, lane `v` holding matrix `v`:

```c
varying double * uniform A = (varying double * uniform)(ap + group_offset);
varying double aij = A[j*ldap + i];     // one vmovupd: element (i,j) of V matrices
```

The kernel body is then the ordinary **scalar** `geqr2` / `dorm2r` /
back-substitution written *once*; ISPC vectorizes it across the `V` matrices of
the group. Compare the GNU version, which spells the vector out by hand with
`__attribute__((vector_size))` and manual mask/select. Same machine code target,
very different source.

The gang width fixes the interleave width, so this prototype targets
`avx512skx-x8` → `programCount == 8 == V`, the AVX-512 FP64 compact format MKL
selects on an AVX-512 host (`mkl_get_format_compact() == MKL_COMPACT_AVX512`).
Column-major, double precision — the tuned path the batched solver uses.

## Installing ISPC

Any of these gives a working `ispc`:

```sh
# Debian/Ubuntu (used here): the distro package
sudo apt-get install ispc            # 1.22.0 on Ubuntu 24.04

# or the official prebuilt binary (newest releases)
#   https://github.com/ispc/ispc/releases  -> ispc-vX.Y.Z-linux.tar.gz
# or via PyPI
pip install ispc
```

This prototype was developed and measured with **ISPC 1.22.0 (LLVM 17)**.

## Compiler flags

**ISPC** (`cqr_ispc.ispc` → `cqr_ispc.o` + a generated `extern "C"` header):

```sh
ispc --target=avx512skx-x8 --arch=x86-64 -O3 --pic \
     --opt=disable-assertions -o cqr_ispc.o -h cqr_ispc_generated.h cqr_ispc.ispc
```

- `--target=avx512skx-x8` — **the key flag.** `x8` sets `programCount = 8`, so a
  `varying double` maps onto one 512-bit ZMM = one compact pack (`V = 8`). Use
  `avx2-i64x4` for `V = 4` (AVX2), `sse4.2-i32x4`/... otherwise. One target = one
  interleave width; ISPC bakes the gang width in at compile time (unlike the GNU
  API, which takes `V` at runtime and dispatches).
- `-O3`, `--pic` — optimize; position-independent so it links into the driver.
- `--opt=disable-assertions` — drop ISPC's internal bounds asserts in the hot path.
- *Not* used: `--opt=fast-math`. Left off so ISPC's FP matches the reference; with
  it left off the ISPC factor comes out **bit-for-bit identical** to the GNU
  kernel (see below). Turn it on for a few percent more if you can accept
  reassociation.

Verify the layout mapped to vector loads (no gather/scatter) with the perf
warnings — this prototype compiles with **zero**:

```sh
ispc --target=avx512skx-x8 -O3 cqr_ispc.ispc -o /dev/null   # (no "Performance Warning")
```

**Reference GNU kernels** (`../src/*_dispatch.cpp`) — compiled `-O3
-march=native -std=c++17` so the `vector_size` kernels see the host's full
AVX-512 width, a fair opponent (the README's performance-build guidance).

## Build and run

Needs Intel MKL (apt `libmkl-dev`, or oneAPI) for pack/unpack, the reference
`trsm`, and per-matrix LAPACK. The drivers pin MKL to its **sequential** layer at
startup (`mkl_set_threading_layer`), so the benchmark is single-threaded and no
`MKL_THREADING_LAYER` env var or threading runtime is needed.

```sh
make                        # builds test + benchmark
./test_cqr_ispc             # correctness
./bench_cqr_ispc 1024 3 4   # nmat reps nrhs   (sequential, 1 thread)
```

## Files

| File | Role |
|------|------|
| `cqr_ispc.ispc` | The three ISPC kernels: `cqr_ispc_dgeqrf_compact`, `cqr_ispc_dormqr_compact`, `cqr_ispc_dtrsm_compact`. |
| `cqr_ispc.h` | `extern "C"` declarations — drop-in siblings of `../src/cqr_compact.h`. |
| `test_cqr_ispc.cpp` | Correctness: full solve vs known `X`, plus element-wise equivalence vs the GNU kernels + MKL `trsm`. |
| `bench_cqr_ispc.cpp` | Per-kernel and full-pipeline GFLOP/s vs GNU, MKL, and per-matrix LAPACK. |
| `cqr_trsm_compact.hpp` | Templated GNU-vector compact `trsm` (so GCC/clang have a counterpart to `mkl_dtrsm_compact`). |
| `bench_compilers.cpp` | Compiler shootout: GCC vs clang vs ISPC, each vs the MKL compact routines (`make shootout`). |
| `Makefile` | Standalone build (ISPC + reference sources + MKL link); targets `bench`, `shootout`. |

## Correctness

`test_cqr_ispc` builds batches with a known exact solution and runs the ISPC
pipeline, checking two things:

1. **Forward error vs the known `X`** — machine precision (`~1e-13`), inside the
   `100·n·eps` gate, across `n = 10..150`, `nrhs = 1..8`, and padded partial
   groups (`nm` not a multiple of 8).
2. **Drop-in equivalence vs the GNU kernels** — the ISPC factor buffer comes out
   `equiv fac 0.0e+00`, **bit-for-bit identical** to `dgeqrf_compact`, and the
   ISPC solution matches the reference pipeline (GNU + MKL `trsm`) to `~1e-14`.
   Same algorithm, same instruction selection → same bits.

```
ISPC compact-QR correctness (compact format=183, V=8)
  nm=8   n=10   nrhs=1 | ISPC fwd 2.66e-15 (rtol 1.3e-12) | equiv fac 0.0e+00 sol 6.7e-16
  nm=16  n=20   nrhs=4 | ISPC fwd 1.42e-14 (rtol 4.9e-12) | equiv fac 0.0e+00 sol 4.4e-15
  ...
  nm=16  n=150  nrhs=3 | ISPC fwd 2.27e-13 (rtol 2.5e-10) | equiv fac 0.0e+00 sol 7.1e-14
all checks passed
```

## Benchmark results

`bench_cqr_ispc` times each stage in isolation and the full solve, on
`nmat = 1024` matrices, best of 3, **sequential (1 thread)**, over `n = 10..150`.
Throughput is GFLOP/s (size-normalized, so efficiency reads across the range).
Machine: virtualized Intel Xeon @ 2.8 GHz, AVX-512, ISPC 1.22 (LLVM 17) vs
GCC 13.3 `-O3 -march=native`.

**Full `AX = B` solve pipeline** (`geqrf → ormqr(Q^T) → trsm`), `nrhs = 4`:

```
   n |   ISPC    GNU+MKL  MKLgeqrf  perMatrix   |  ISPC speedup vs GNU / MKL / perMat
-----+------------------------------------------+-------------------------------------
  10 |   15.35    11.18     7.70      1.03   |    1.37x /  1.99x /  14.91x
  20 |   11.36    11.15     9.31      2.31   |    1.02x /  1.22x /   4.92x
  30 |   12.57    12.52     9.80      3.50   |    1.00x /  1.28x /   3.60x
  40 |   12.95    12.72    10.42      4.75   |    1.02x /  1.24x /   2.73x
  50 |   14.16    12.95    10.81      4.58   |    1.09x /  1.31x /   3.09x
  60 |   14.42    11.95    10.81      5.55   |    1.21x /  1.33x /   2.60x
  80 |   15.33    13.59    11.20      6.58   |    1.13x /  1.37x /   2.33x
 100 |   16.14    13.16    11.66      7.01   |    1.23x /  1.38x /   2.30x
 120 |   15.51    12.21    10.88      8.88   |    1.27x /  1.43x /   1.75x
 150 |   15.56    12.77    10.43     12.57   |    1.22x /  1.49x /   1.24x
-----+------------------------------------------+-------------------------------------
geomean full-solve speedup  ISPC vs GNU 1.15x   vs MKL 1.39x   vs per-matrix 3.04x
```

- **ISPC ≈ GNU (slightly ahead):** the plain-SPMD ISPC pipeline matches and edges
  out the hand-written GNU `vector_size` pipeline — **1.15× geomean** (nrhs=4),
  **1.12×** (nrhs=1). It is faster than the MKL-`geqrf` pipeline (**1.39×**) and
  **~3×** faster than per-matrix LAPACK (which only closes the gap past `n≈150`,
  where the batch spills cache and dense BLAS-3 catches up).

**Per-kernel, where the difference actually lives** (geomean over `n = 10..150`):

| kernel | ISPC vs GNU | ISPC vs MKL | notes |
|--------|:-----------:|:-----------:|-------|
| `geqrf` (factor) | **1.17×** | **1.44×** | ISPC's clear win; LLVM schedules the branch-free `larfg` + blocked trailing update better than GCC. |
| `ormqr` (apply Q^T), nrhs=4 | 0.97× | — | parity: the 4-wide blocked loop dominates. |
| `ormqr`, nrhs=1 | 0.78× | — | ISPC's scalar single-column remainder loop is weaker than GCC's; only shows when nrhs<4. |
| `trsm` (solve), nrhs=4 | — | 1.07× | parity with MKL; MKL pulls ahead only at `n≥100`. |

The full-solve win is carried by **`geqrf`**, which dominates cost at these sizes
(`O(n³)` vs `O(n²·nrhs)`); `ormqr`/`trsm` are secondary and roughly at parity.

## Compiler shootout: GCC vs clang vs ISPC vs MKL

`make shootout` builds the *same* kernels three ways — the templated GNU
`vector_size` source compiled by **GCC** and by **clang**, and the **ISPC**
kernels — and reports each as a speedup over the matching MKL compact routine
(`bench_compilers.cpp`, built once per compiler via `bench_gcc`/`bench_clang` so
each column is genuinely that compiler's codegen; ISPC and MKL appear in both as
a consistency check). Geomean over `n = 10..150`, `nrhs = 4`, sequential:

| routine (speedup vs MKL compact) | GCC | clang | ISPC | fastest |
|---|:---:|:---:|:---:|---|
| `geqrf` vs `mkl_dgeqrf_compact` | 1.24× | **1.48×** | 1.47× | clang ≈ ISPC |
| `trsm` vs `mkl_dtrsm_compact` | **1.40×** | 1.05× | 1.10× | GCC, by a lot |
| full solve vs the MKL pipeline | 1.20× | **1.35×** | 1.30× | clang |

(`ormqr` has no MKL compact counterpart; the three land within ~5% of each other,
ISPC marginally ahead. ISPC's `vs MKL` geomeans agree to ±0.02× across the two
builds — the consistency check.)

**Which is fastest? It depends on the kernel — and all three beat MKL:**

- **Factorization (`geqrf`)**, the dominant cost: **clang and ISPC tie** at ~1.47×
  over MKL; GCC trails at 1.24×. LLVM's scheduler (shared by clang and ISPC)
  handles the branch-free `larfg` + blocked trailing update better than GCC's.
- **Triangular solve (`trsm`)**: **GCC wins decisively** (1.40× vs MKL) — its
  autovectorizer handles the back-substitution recurrence markedly better; clang
  and ISPC sit near parity with MKL and fall behind it past `n≈100`.
- **Full pipeline**: **clang is fastest** (1.35×), ISPC just behind (1.30×), GCC
  third (1.20×) — GCC's strong `trsm` doesn't offset its weaker `geqrf`, which
  dominates the flop count.

Bottom line: **clang ≳ ISPC > GCC > MKL** on the end-to-end solve, but the lead
changes hands per kernel — GCC owns `trsm`, clang/ISPC own `geqrf`. ISPC is never
the worst and never far from the best, from far simpler source; no single toolchain
wins everything.

### Why the per-kernel lead flips: 256-bit vs 512-bit

The GNU `pack<double,8>` is a 64-byte (512-bit) vector type, and the toolchains
disagree on how wide to run it — this is the *whole* story on Cascade Lake:

- **GCC → 512-bit ZMM** (honors the 64-byte vector type; `-mprefer-vector-width`
  doesn't override it for *explicit* vectors).
- **clang → 2×256-bit YMM** (splits it; respects `-mprefer-vector-width`).
- **ISPC → 2×256-bit YMM.** `avx512skx-x8` is 256-bit *by design* — only
  `avx512skx-x16` emits ZMM, and that means gang/`V`=16, which breaks the MKL
  `V`=8 compact layout. ISPC 1.22 has no ZMM-at-gang-8 switch, and neither
  `--opt=fast-math` nor `--device=skx` changes it (verified: still 0 `zmm`).

Forcing clang to each width on the *same source* (`-mprefer-vector-width=256/512`)
isolates it — width alone flips which kernel wins (speedup vs MKL, n=100):

| n=100 | 256-bit (ymm) | 512-bit (zmm) |
|---|:---:|:---:|
| `geqrf` (FMA-dense) | **1.54×** | 1.17× |
| `trsm` (division / latency-bound) | 0.85× | **1.11×** |

- **`trsm` likes 512-bit:** half the instructions and half the `vdivpd`, and its
  low sustained-FMA density doesn't trip the AVX-512 frequency license — so the
  wider path just wins. That is exactly why GCC (always ZMM here) leads `trsm`.
- **`geqrf` likes 256-bit:** it *is* FMA-dense, so 512-bit triggers the AVX-512
  downclock and loses — why clang/ISPC (256-bit) beat GCC on the factorization.

**Matching GCC on `trsm` needs 512-bit, which ISPC can't emit at gang-8** — but a
"mix by kernel" build (ISPC `geqrf`/`ormqr` + a 512-bit `trsm`) would be an
*overfit to this exact CPU and compiler set, not a portable strategy.* The
256-vs-512 crossover moves — and can invert — with the microarchitecture: the
AVX-512 frequency license that penalizes wide FMA on Skylake-SP / Cascade Lake is
much milder on Ice Lake / Sapphire Rapids and absent off-x86, and it shifts with
compiler version too. Even the *direction* here is only inferred: this shared VM
pins its reported clock at 2.8 GHz and exposes no PMU, so the downclock cannot be
measured — at a genuinely fixed frequency, 512-bit might win `geqrf` as well,
collapsing the flip. All builds are release + native (C++ `-O3 -march=native`,
ISPC `-O3` on the host AVX-512 ISA); the gap is *width*, not a missing flag — but
**which width wins is machine-specific, so this is an observation, not a
recommendation.** The portable conclusion is the opposite: let the target pick the
width and take the one-source ~parity.

## Takeaways

1. **Correct and drop-in.** The ISPC kernels recover the known solution to machine
   precision and produce a **bit-for-bit identical** factorization to the existing
   GNU kernels — genuine `extern "C"` drop-in replacements over the same MKL
   Compact buffers.

2. **A compact pack is a `varying` — the model fits the format exactly.** Because
   `V = programCount`, the interleaved batch layout maps onto ISPC's SPMD lanes
   with zero shuffling: one aligned vector load per element, no gather/scatter
   (verified: zero ISPC performance warnings). The kernels are the *scalar*
   `geqr2`/`dorm2r`/back-substitution written once — no hand-rolled vector types,
   masks, or `vselect`, which the GNU version spells out explicitly.

3. **Same performance as hand-written GNU vectors, from far simpler source.** The
   compiler shootout reframes the earlier "ISPC beats GCC on `geqrf`" as an *LLVM*
   effect — clang compiling the GNU kernels matches ISPC there (both ~1.47× vs MKL;
   GCC 1.24×). ISPC's headline is **maintainability at no throughput cost**: it ties
   the fastest compiler on the dominant factorization, is never the worst on any
   kernel, and comes from far simpler source than the hand-rolled vectors.

4. **Where ISPC gives ground:** the scalar (non-blocked) inner loops — `ormqr` at
   `nrhs=1`, `trsm` at large `n` — where GCC/MKL's hand-tuning wins. Enough panel
   width (nrhs ≥ 4) erases the gap.

5. **The one structural cost:** ISPC bakes the gang width in at compile time, so
   the interleave width `V` is fixed per target (one `.o` per `V`). The GNU API
   dispatches `V` at runtime from one source. For a batch pipeline pinned to the
   host's native compact width (here AVX-512, `V=8`) that is a non-issue; for a
   library shipping every width it means one ISPC object per target.

*Measurement caveats — read every number here as indicative only, not a
benchmark result.* These runs are on a **shared, virtualized 4-vCPU node with no
frequency control**: the governor/turbo are inaccessible, the guest reports a
**fixed 2.8 GHz and exposes no PMU**, so the achieved core clock — and any
AVX-512 downclock — is neither pinned nor even observable here. Best-of-3 wall
time hides some neighbor noise but nothing about DVFS. A result you could trust
needs a **non-shared / bare-metal node at pinned frequency** (turbo off,
`performance` governor or a fixed P-state), isolated and pinned cores, PMU cycle
counts, repeated with reported variance/CI, across **several microarchitectures**
and compiler versions — none of which this environment provides. Also: one
toolchain set (GCC 13.3, clang 18.1.3, ISPC 1.22/LLVM 17), double precision,
`V=8`. The one conclusion that survives all of that is the portable one — ISPC
gives a **correct, drop-in, ~parity** batched pipeline from far simpler source,
ISA/width delegated to the target; the absolute speedups are not a portable
claim. Reproduce with `make bench` / `make shootout`.

---
*Assisted-by: Claude:claude-opus-4.8*
