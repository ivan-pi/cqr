# ISPC prototype — compact QR pipeline

A prototype of the **cqr** batched QR solve (`DGEQRF` → `DORMQR` → `DTRSM` over
Intel MKL's Compact/interleaved format) reimplemented in
[Intel ISPC](https://ispc.github.io/), to measure ISPC against the hand-written
GNU `vector_size` kernels in `../src` — and, since ISPC uses the LLVM backend,
against GCC and clang compiling those same kernels — all normalized to MKL.

## The idea: a compact pack *is* a `varying`

MKL Compact stores element `(i,j)` of the `V` interleaved matrices of a group
contiguously — `ap[g*ldap*ncol*V + (j*ldap+i)*V + v]` — and an ISPC
`varying double` *is* `programCount` consecutive doubles. So when `V ==
programCount`, a compact pack is exactly one `varying double`:

```c
varying double * uniform A = (varying double * uniform)(ap + group_offset);
varying double aij = A[j*ldap + i];   // one aligned vector load: (i,j) of V matrices
```

Each kernel is then the ordinary **scalar** `geqr2` / `dorm2r` /
back-substitution, which ISPC vectorizes across the `V` matrices — no hand-rolled
vector types or masks, no gather/scatter (zero ISPC perf warnings). The gang
width fixes `V`, so this targets `avx512skx-x8` → `programCount == 8 == V`, the
AVX-512 FP64 compact format MKL selects on this host. Column-major, double.

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

- **ISPC target — required, must be width 8** so `programCount == V == 8`:
  `-DCMAKE_ISPC_INSTRUCTION_SETS=avx512skx-x8` (AVX-512; `avx2-i32x8` for AVX2).
  ISPC's default here is `avx512spr-x16` (width 16), which `cqr_ispc.ispc` rejects
  with an `#error`. `-DCMAKE_BUILD_TYPE=Release` gives ISPC `-O3` too; add extra
  ISPC flags (e.g. `--opt=disable-assertions`) via `CMAKE_ISPC_FLAGS`, and
  `CMAKE_ISPC_HEADER_DIRECTORY` selects ISPC's generated header over the shipped
  `cqr_ispc.h` if you prefer it.
- **Fair benchmark:** `-DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-march=native`
  so the GNU `vector_size` kernels use the host's full width (correctness needs
  neither). The drivers pin MKL sequential at startup — no env var.
- **Compiler shootout:** the benchmark's `native` column is whichever CXX compiler
  configured the tree, so compare with a second tree:
  `cmake -S . -B build-clang -DCMAKE_CXX_COMPILER=clang++ ...` and run both binaries.

## Files

| File | Role |
|------|------|
| `cqr_ispc.ispc` | The three ISPC kernels (`cqr_ispc_d{geqrf,ormqr,trsm}_compact`). |
| `cqr_ispc.h` | `extern "C"` declarations — drop-ins for `../src/cqr_compact.h`. |
| `cqr_trsm_compact.hpp` | Templated GNU-vector `trsm` (a counterpart to `mkl_dtrsm_compact` for GCC/clang). |
| `bench_common.hpp` | Shared harness: pool, timer, pack/unpack, GFLOP helpers. |
| `test_cqr_ispc.cpp` | Per-kernel unit tests (vs LAPACK/MKL oracles) + end-to-end solve. |
| `bench_cqr_ispc.cpp` | The benchmark; the `native` column is the configuring CXX compiler. |
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
