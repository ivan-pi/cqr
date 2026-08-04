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

```sh
sudo apt-get install ispc            # 1.22 on Ubuntu 24.04 (or pip install ispc,
                                     # or a github.com/ispc/ispc release tarball)
make          # test + gcc/clang benchmarks   (needs Intel MKL: apt libmkl-dev)
make test     # correctness      make shootout # GCC vs clang vs ISPC vs MKL
```

- **ISPC:** `--target=avx512skx-x8 --arch=x86-64 -O3 --pic --opt=disable-assertions`.
  `avx512skx-x8` is the key flag (`programCount = 8 = V`). One target = one width.
- **C++ kernels:** `-O3 -march=native -std=c++17` so the `vector_size` kernels get
  the host's full width. The drivers pin MKL sequential at startup (no env var).

## Files

| File | Role |
|------|------|
| `cqr_ispc.ispc` | The three ISPC kernels (`cqr_ispc_d{geqrf,ormqr,trsm}_compact`). |
| `cqr_ispc.h` | `extern "C"` declarations — drop-ins for `../src/cqr_compact.h`. |
| `cqr_trsm_compact.hpp` | Templated GNU-vector `trsm` (a counterpart to `mkl_dtrsm_compact` for GCC/clang). |
| `bench_common.hpp` | Shared harness: pool, timer, pack/unpack, GFLOP helpers. |
| `test_cqr_ispc.cpp` | Correctness: known-`X` solve + bit-for-bit equivalence vs the GNU kernels. |
| `bench_cqr_ispc.cpp` | The benchmark; built by g++ and clang++ (`make shootout`). |
| `Makefile` | Standalone build. |

## Correctness

`test_cqr_ispc` recovers a known `X` to machine precision across `n=10..150`,
`nrhs=1..8`, and padded partial groups, and the ISPC factor comes out
**bit-for-bit identical** (`equiv fac 0.0e+00`) to the GNU kernel — same
algorithm, same FMA selection. Genuine drop-in replacements.

## Results — GCC vs clang vs ISPC, normalized to MKL

Geomean speedup over the matching MKL compact routine, `n=10..150`, `nrhs=4`,
sequential (`make shootout`):

| speedup vs MKL compact | GCC | clang | ISPC | fastest |
|---|:---:|:---:|:---:|---|
| `geqrf` vs `mkl_dgeqrf_compact` | 1.24× | **1.48×** | 1.47× | clang ≈ ISPC |
| `trsm` vs `mkl_dtrsm_compact` | **1.40×** | 1.05× | 1.10× | GCC, by a lot |
| full solve vs the MKL pipeline | 1.20× | **1.35×** | 1.30× | clang |

(ISPC ≈ 3× the per-matrix LAPACK path at these sizes; `ormqr` has no MKL compact
counterpart, and the three land within ~5% there. ISPC's `vs MKL` geomeans agree
across the GCC and clang builds — a consistency check.)

**No single winner; the lead flips per kernel — and it's a 256- vs 512-bit
story.** The `pack<double,8>` is a 512-bit vector type: **GCC emits 512-bit ZMM;
clang and ISPC split it into 2×256-bit YMM** (ISPC's `avx512skx-x8` is 256-bit by
design — only `-x16` uses ZMM, which would mean `V=16` and break the MKL `V=8`
layout; 1.22 has no ZMM-at-gang-8 switch). Forcing clang to each width on the
*same source* isolates it (speedup vs MKL, n=100):

| n=100 | 256-bit (ymm) | 512-bit (zmm) |
|---|:---:|:---:|
| `geqrf` (FMA-dense) | **1.54×** | 1.17× |
| `trsm` (division/latency-bound) | 0.85× | **1.11×** |

`trsm` likes 512-bit (fewer instructions/divides, low FMA density so no AVX-512
downclock) → GCC wins it; `geqrf` is FMA-dense so 512-bit downclocks and loses →
clang/ISPC win it. Matching GCC on `trsm` would need 512-bit, which ISPC can't
emit at gang-8 — so a "mix by kernel" build is an **overfit to this CPU/compiler,
not a portable strategy** (the crossover weakens or inverts on Ice Lake / Sapphire
Rapids / non-x86 and with compiler version).

> **Measurement caveats — indicative only, not benchmark-grade.** Shared,
> virtualized 4-vCPU node with **no frequency control**: governor/turbo
> inaccessible, a pinned 2.8 GHz readout, and **no PMU**, so the achieved clock and
> any AVX-512 downclock are neither controllable nor observable — the geqrf-256
> conclusion in particular is *inferred*, and at fixed frequency 512-bit might win
> geqrf too. A trustworthy study needs a bare-metal / non-shared node at pinned
> frequency, isolated+pinned cores, PMU cycle counts, variance/CI, and several
> microarchitectures. One toolchain set (GCC 13.3, clang 18.1.3, ISPC 1.22/LLVM 17),
> double, `V=8`.

## Takeaways

1. **Correct, drop-in, ~parity from far simpler source.** ISPC ties the fastest
   compiler on the dominant `geqrf` and is never worst on any kernel, written as
   plain scalar code instead of hand-rolled vectors.
2. **The `geqrf` edge over GCC is an LLVM/width effect, not ISPC-specific** — clang
   on the same GNU source matches ISPC (both 256-bit).
3. **Width, not toolchain, decides each kernel** — and which width wins is
   machine-specific, so the portable move is to let the target pick it and take the
   one-source parity, not to hand-assemble a per-kernel mix.

---
*Assisted-by: Claude:claude-opus-4.8*
