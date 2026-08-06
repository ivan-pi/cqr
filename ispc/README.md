# ISPC prototype — compact QR pipeline

A prototype of the **cqr** batched QR solve (`DGEQRF` → `DORMQR` → `DTRSM` over
Intel MKL's Compact/interleaved format) reimplemented in
[Intel ISPC](https://ispc.github.io/), to measure ISPC against the hand-written
GNU `vector_size` kernels in `../src` — and, since ISPC uses the LLVM backend,
against GCC and clang compiling those same kernels — all normalized to MKL. A
batched Cholesky (`DPOTRF`, the SPD counterpart to the QR path) rounds out the
kernel set.

## The idea: SPMD across the SIMD lanes

ISPC runs a *gang* of program instances concurrently, mapped onto the SIMD lanes
of one core — you write the scalar computation for a single instance and the gang
executes it in lockstep across the lanes (the same model as a CUDA warp or an
OpenCL work-item). That is exactly the batched-vectorization we want: **one
program instance per matrix, one gang per interleaved group.**

It lands on the compact layout with no shuffling. MKL Compact stores element
`(i,j)` of the `V` interleaved matrices of a group contiguously —
`ap[g*ldap*ncol*V + (j*ldap+i)*V + v]` — and an ISPC `varying double` is one value
per lane, laid out as `V` consecutive doubles. So when the gang width equals `V`,
a compact pack *is* one `varying double`:

```c
varying double * uniform A = (varying double * uniform)(ap + group_offset);
varying double aij = A[j*ldap + i];   // one aligned vector load: (i,j) of V matrices
```

Each kernel is then the ordinary **scalar** `geqr2` / `dorm2r` / `potf2` /
back-substitution, which ISPC vectorizes across the `V` matrices — no hand-rolled
vector types or masks, no gather/scatter (zero ISPC perf warnings). Column-major,
double. The gang width is set by the `--target`'s `-xN` suffix; the kernels work
at any width as long as the data is packed at `V` = the gang width. The MKL-interop
test and benchmark use MKL's AVX-512 FP64 format (`V = 8`, `avx512skx-x8`); the
gang-size sweep (below) varies it.

## Install, flags, build

Built with CMake's native ISPC language (`project(... LANGUAGES CXX ISPC)`),
CMake >= 3.19 and Intel MKL (`libmkl-dev`). The build imposes no ISA, build type,
or optimization flags — those are the caller's, as in the parent project:

```sh
sudo apt-get install ispc            # 1.22 on Ubuntu 24.04 (or pip install ispc,
                                     # or a github.com/ispc/ispc release tarball)
cmake -S . -B build -DCMAKE_ISPC_INSTRUCTION_SETS=avx512skx-x8 \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-march=native
cmake --build build -j
ctest --test-dir build --output-on-failure     # per-kernel correctness tests
./build/bench_cqr_ispc                           # the benchmark
```

- **ISPC target:** `-DCMAKE_ISPC_INSTRUCTION_SETS=avx512skx-x8` sets the ISA and
  the gang width (`-x8`). The MKL-interop test/benchmark need the gang to equal
  MKL's compact `V` (8 on AVX-512) and check it at runtime — ISPC's default here is
  `avx512spr-x16` (width 16), which they reject with a clear message.
  `-DCMAKE_BUILD_TYPE=Release` gives ISPC `-O3`; add ISPC flags via
  `CMAKE_ISPC_FLAGS`, and `CMAKE_ISPC_HEADER_DIRECTORY` selects ISPC's generated
  header over the shipped `cqr_ispc.h` if you prefer it.
- **Fair benchmark:** `-DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-march=native`
  so the GNU `vector_size` kernels use the host's full width (correctness needs
  neither). The drivers pin MKL sequential at startup — no env var.
- **Compiler shootout:** the benchmark's `native` column is whichever CXX compiler
  configured the tree, so compare with a second tree:
  `cmake -S . -B build-clang -DCMAKE_CXX_COMPILER=clang++ ...` and run both binaries.
- **Gang-size sweep:** `bench_gang` packs at `V` = the gang width (no MKL), so build
  a tree per target to sweep it — `for w in 4 8 16 32 64; do cmake -S . -B build-x$w
  -DCMAKE_ISPC_INSTRUCTION_SETS=avx512skx-x$w ...; done` (see Results).

## Files

| File | Role |
|------|------|
| `cqr_ispc.ispc` | The ISPC kernels (`geqrf`/`ormqr`/`trsm`/`potrf`) + `cqr_ispc_gang_width` (gang-generic). |
| `cqr_ispc.h` | `extern "C"` declarations — drop-ins for `../src/cqr_compact.h`. |
| `bench_gang.cpp` | Gang-size sweep (packs at `V` = gang width; ISPC-only, no MKL). |
| `bench_common.hpp` | Shared harness: pool, timer, pack/unpack, GFLOP helpers. |
| `test_cqr_ispc.cpp` | Per-kernel unit tests (vs LAPACK/MKL oracles) + end-to-end solve. |
| `bench_cqr_ispc.cpp` | The benchmark; the `native` column is the configuring CXX compiler. |
| `bench_potrf.cpp` | ISPC Cholesky vs `mkl_dpotrf_compact` (SPD batch, both uplo). |
| `bench_solve.cpp` | Full solve pipeline: MKL-compact vs CQR-GNU (GCC) vs ISPC. |
| `CMakeLists.txt` | Standalone build via CMake's ISPC language (>= 3.19). |

## Correctness

`test_cqr_ispc` is a set of **per-kernel unit tests**, each against an independent
oracle so a failure isolates to one routine, plus the end-to-end solve (all pass
at machine precision; square, tall, padded groups):

- **geqrf** — `A = Q R` residual and orthogonality of `Q` (formed by LAPACK
  `dorgqr` from the ISPC reflectors) — ~1e-16.
- **ormqr** — `Q B` *and* `Q^T B` vs `LAPACKE_dormqr` on a LAPACK factorization
  (independent of the ISPC geqrf), plus a `Q(Q^T B)=B` round-trip — ~1e-15.
- **trsm** — `R X = alpha B` vs `mkl_dtrsm_compact` and a known `X`, with
  `alpha != 1` and `nrhs = 1` and `6` — ~1e-16.
- **potrf** — `A = L L^T` (lower, the contiguous path) and `A = U^T U` (upper, the
  strided path) vs `mkl_dpotrf_compact` and `LAPACKE_dpotrf` (both to machine
  precision), with reconstruction, the untouched opposite triangle, a non-SPD lane
  that poisons only itself (NaN, siblings intact), and an SPD solve (`potrf` + two
  MKL `trsm`).
- **solve** — the full pipeline recovers a known `X`; the ISPC factor also matches
  the GNU kernel to a few ULP (often bit-identical, but that is input/compiler-
  dependent, not guaranteed).

## Results — GCC vs clang vs ISPC, normalized to MKL

Geomean speedup over the matching MKL compact routine, `n=10..150`, `nrhs=4`,
sequential (two build trees, GCC and clang). **On this shared, frequency-unpinned node the
numbers swing ~15–30% run to run** (two independent runs bracket each cell) — so
these are rough ranges, not point estimates:

| geomean vs MKL compact | GCC | clang | ISPC |
|---|:---:|:---:|:---:|
| `geqrf` | 1.2–1.4× | 1.3–1.5× | 1.3–1.5× |
| `trsm`  | 1.1–1.4× | 0.8–1.1× | 0.8–1.1× |
| full solve | 1.2–1.3× | 1.2–1.4× | 1.2–1.3× |

What is **stable across runs** (the defensible part):

- **ISPC ≈ the hand-written GNU vectors on every kernel** — same LLVM/GCC
  backend — and the three full-solve pipelines land within noise of each other,
  modestly over MKL and ~3× over per-matrix LAPACK. (`ormqr` has no MKL compact
  counterpart; the three are within ~5% there.)
- **`geqrf` beats MKL; `trsm` is a toss-up with MKL.** Which of GCC/clang/ISPC
  *leads* a given kernel is **not** stable enough here to rank — the per-kernel
  winner reshuffled between runs.

### The one thing that is a fact, not a measurement: 256- vs 512-bit

The `pack<double,8>` is a 512-bit vector type, and the toolchains disagree on how
wide to run it (verified in the asm, independent of any timing): **GCC emits
512-bit ZMM; clang and ISPC split it into 2×256-bit YMM.** ISPC's `avx512skx-x8`
is 256-bit by design — only `-x16` uses ZMM, which would mean `V=16` and break the
MKL `V=8` layout; 1.22 has no ZMM-at-gang-8 switch.

The performance *consequence* is where reliability runs out. A controlled
same-source A/B (forcing clang to each width) suggested 512-bit helps the
divide/latency-bound `trsm` (fewer instructions and `vdivpd`) and can hurt the
FMA-dense `geqrf` (AVX-512 downclock) — the classic Skylake-SP / Cascade Lake
tradeoff. But this box pins its clock readout and exposes no PMU, so the downclock
is *inferred, not observed*, and across full runs even the `geqrf` direction did
not hold. So: the width difference is real and fixed per toolchain; **which width
wins, and by how much, is machine-specific and unmeasured here.** A "mix by
kernel" build (which ISPC couldn't do at gang-8 anyway) would be an overfit, not a
portable strategy.

### Vector width sweep — and ISPC vs GNU vectors at V=16 (`bench_gang`)

The ISPC kernels are gang-generic, and the templated GNU `vector_size` kernels
take `V` as a template argument (`pack<T,V>`, `V ∈ {2,4,8,16}`). Both use the same
compact layout and neither needs MKL, so `bench_gang` packs at `V` = the gang
width and times **both** at that `V`. All widths validate to machine precision.

ISPC alone scales gracefully and peaks at gang 16 (geomean GFLOP/s, standalone):

| gang (`avx512skx-x`) | 4 | 8 | **16** | 32 | 64 |
|---|:---:|:---:|:---:|:---:|:---:|
| ISPC GFLOP/s | 13.8 | 16.4 | **17.3** | 14.2 | 11.8 |
| registers (asm) | ymm | ymm | **zmm** | zmm×2 | zmm×4 |

The gang size also picks the register width — this is the same 256-vs-512 knob as
above. `avx512skx-x8` emits **256-bit YMM** (a deliberate choice that dodges the
AVX-512 downclock — 0 `zmm`); only `-x16`+ emit **512-bit ZMM**. So gang 16 wins on
width (2× data/instruction) *and* ILP (16 matrices in flight hide the `larfg`/`trsm`
divide + sqrt latency, per ISPC's "2–4× native width" guidance); 32/64 then spill.

Head-to-head with the GNU vectors at the same `V` — a regime **MKL can't reach**,
its FP64 compact format tops out at `V=8` — the crossover is the story:

| geomean GFLOP/s | V=8 | V=16 |
|---|:---:|:---:|
| ISPC | ~11 | ~12 |
| GCC `pack<double,V>` | **~12** | ~6.6 |
| winner | GCC ~1.1× | **ISPC ~1.8×** |

At `V=8` (the native width) GCC's hand-lowered vectors edge ISPC. At `V=16` — a
1024-bit `pack<double,16>` = 2× ZMM — GCC **halves its own `V=8` throughput**: the
4-wide-blocked 1024-bit vectors thrash the register file. ISPC holds up, because
scheduling a gang *wider* than the native SIMD width is exactly what it is built
for. So the one width MKL doesn't offer is where ISPC's model earns its keep.
(Absolute numbers are noisy here; the crossover direction is stable across runs.)

### Full solve pipeline — MKL-compact vs CQR-GNU vs ISPC (`bench_solve`)

`bench_solve` times the whole `geqrf → ormqr → trsm` solve of a batch (known
`X == 1`) three ways on identical packed input — `n = 10..120`, `-O3 -march=native`,
`V = 8`, sequential:

- **MKL** — `mkl_dgeqrf_compact → cqr_mkl_dormqr_compact → mkl_dtrsm_compact`
- **CQR-GNU** — the templated GNU `vector_size` kernels, compiled by GCC
- **ISPC** — the three `cqr_ispc_*` kernels

MKL ships no compact `ormqr`, so both the MKL and CQR-GNU pipelines borrow this
project's `cqr_mkl_dormqr_compact` for that step; the MKL-vs-CQR gap is therefore
purely `geqrf + trsm`. All three recover `X` to ~1e-15. Geomean over `n=10..120`,
two runs bracketing each figure:

| geomean, n=10..120 | MKL | CQR-GNU (GCC) | ISPC |
|---|:--:|:--:|:--:|
| GFLOP/s | ~8.8–9.2 | ~11.0–11.6 | ~9.9–10.5 |
| speedup vs MKL | 1.00× | **1.25–1.30×** | **1.13–1.19×** |

Both open pipelines beat MKL's compact solve for these small batched systems, and
the ordering is stable across runs and sizes: **CQR-GNU (GCC) > ISPC > MKL**. The
CQR-GNU lead is the per-kernel result compounded — GCC's `geqrf` and `trsm` each
beat MKL's compact ones. ISPC lands between: its `geqrf` matches GCC and its `ormqr`
is at parity, but its `trsm` runs ~0.9× MKL (the 256-bit `avx512skx-x8` gang vs
GCC's 512-bit ZMM — see above), which is what pulls its full pipeline below GCC's.
Same shared-node caveats — the ratios are steady, the absolute GFLOP/s drift ~5–10%
run to run.

### Cholesky (`potrf`) vs MKL (`bench_potrf`)

`bench_potrf` factors an SPD batch (`A = MᵀM + nI`) with `cqr_ispc_dpotrf_compact`
and with `mkl_dpotrf_compact` and reports the ISPC speedup over MKL (both agree to
machine precision). The kernel is **blocked** (right-looking, panel width `NB = 8`):
each panel is factored unblocked, then its cross-panel contribution to the trailing
submatrix is applied as one symmetric rank-`NB` update (SYRK) — streaming the
trailing submatrix once per panel (`n/NB` times) instead of once per column, which
is what keeps a large batch cache-resident. Geomean ISPC/MKL ≈ **1.03×** (lower,
contiguous) and **0.98×** (upper, strided): **on par with MKL overall, and a clear
ISPC lead once the matrices are big enough to matter.**

| ISPC/MKL | n=10 | 20–40 | 50 | 60–80 | 100–150 | geomean |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| lower (contiguous) | 0.72× | 0.84–0.99× | 1.05× | 1.1–1.2× | **1.2–1.3×** | **1.03×** |
| upper (strided) | 0.69× | 0.74–0.96× | 0.98× | 1.05–1.12× | **1.1–1.3×** | **0.98×** |

MKL keeps a **small-`n` edge** (`n ≲ 40`, where an 8-matrix group is already
cache-resident so blocking only adds panel-factor overhead); ISPC crosses over
around `n ≈ 50` and leads by ~1.2–1.3× at the top of the range.

The blocking is what earned this. The earlier **unblocked** `potf2` re-streamed the
trailing submatrix `O(n)` times and went memory-bound past L2, plateauing at ~10
GFLOP/s — geomeans of 0.76× / 0.72×, with MKL winning everywhere for large `n`
(0.58× at `n=150`). The blocked kernel climbs to ~20–22 GFLOP/s at `n=150`, ~2×
faster right where it had been losing. Same lesson as the QR kernels from the other
side: match the memory-traffic structure (here, block for cache) and the plain SPMD
kernel is competitive with MKL. (Errors track MKL to machine precision throughout.)

> **Measurement caveats — indicative only, not benchmark-grade.** Shared,
> virtualized 4-vCPU node with **no frequency control** (governor/turbo
> inaccessible, a pinned 2.8 GHz readout, **no PMU**): the achieved clock and any
> AVX-512 downclock are neither controllable nor observable, and repeat runs move
> the geomeans 15–30%. A trustworthy study needs a bare-metal / non-shared node at
> pinned frequency, isolated+pinned cores, PMU cycle counts, variance/CI, and
> several microarchitectures. One toolchain set (GCC 13.3, clang 18.1.3,
> ISPC 1.22/LLVM 17), double, `V=8`.

## Takeaways

1. **Correct, drop-in, on par — from far simpler source.** Every kernel matches
   its LAPACK/MKL oracle to machine precision (isolated unit tests) and runs within
   run-to-run noise of the hand-written GNU vectors, written as plain scalar code.
2. **No ISPC-specific speed edge, and none needed.** Where ISPC looked faster than
   GCC it was an LLVM/width effect (clang on the same source tracks ISPC); the
   value is the one-source SPMD model at parity, not a throughput win.
3. **Absolute speedups here are not trustworthy** — shared node, no fixed
   frequency, no PMU. The width choice (256 vs 512) is a real, target-dependent
   knob; the portable move is to let the target pick it, not to hand-tune a mix.

---
*Assisted-by: Claude:claude-opus-4.8*
