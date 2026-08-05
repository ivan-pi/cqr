# Fortran batched-QR: four ways to vectorize a fixed-width-8 batch

A self-contained Fortran experiment that writes the two workhorse batched-QR
kernels of this project -- the unblocked Householder factorization (`dgeqrf` /
`geqr2`) and the apply-`Q^T` step (`dormqr` / `orm2r`) -- **four different ways**,
checks that all four agree with a scalar reference, and measures which is
fastest under **both gfortran and Intel `ifx`**.

The batch is interleaved in groups of a **fixed, compile-time width `VW = 8`** --
exactly one 512-bit AVX register of doubles -- with layout `A(VW, m, n, ng)`: the
8-lane batch index is fastest-varying and `ng` groups are stacked behind it.
Every variant loops over the `ng` groups; within a group the batch extent is the
constant `8`, so the compiler can lower each lane operation to a single unmasked
8-wide instruction with **no remainder loop**. This is the Fortran counterpart of
the question the C++ library answered with GNU vector types: *what is the best way
to express SIMD over a batch of many small matrices?*

## The four variants

All four share the layout above; the **only** thing that differs is how each
expresses the width-8 batch SIMD:

| # | Variant | How the batch SIMD is written | Source |
|---|---------|-------------------------------|--------|
| 1 | **inner-loop** | explicit `DO` loops, **`do b = 1, VW` innermost** | `geqr2_inner`, `orm2r_inner` |
| 2 | **outer-loop** | one `!$omp simd` over **`do b = 1, VW`**, the whole scalar per-matrix algorithm in the body (one matrix per lane) | `geqr2_outer`, `orm2r_outer` |
| 3 | **array-ops** | Fortran whole-array syntax `A(:,i,j,g)` over the 8 lanes | `geqr2_array`, `orm2r_array` |
| 4 | **vector-type** | the `vec8` derived type (`real(dp) :: v(8)`) with overloaded `+ - * /`, `sqrt`, masked `merge` -- the Fortran analogue of the C++ GNU vector types | `geqr2_vtype`, `orm2r_vtype` |

The data-dependent branch in LAPACK's `dlarfg` (`if xnorm == 0: tau = 0`) is
replaced everywhere by the branch-free `tail > 0` mask, so every lane runs
unmasked with identical semantics to LAPACK `?geqr2` / `?orm2r`.

## Files

| File | Role |
|------|------|
| `vec8_mod.f90` | The `vec8` type (`v(8)`, `VW = 8`) with overloaded operators (`pure` functions), `vsqrt`, `vmerge`, `load8`/`store8`. |
| `batched_qr.f90` | Scalar reference (`ref_geqr2`, `ref_orm2r`) + the four `geqr2` and four `orm2r` variants. |
| `bench.f90` | Driver: correctness vs the reference at every size, then adaptive-rep timing. |
| `Makefile` | `make` (gfortran) or `make FC=ifx`; `make run`. |

## Build and run

```sh
cd fortran_bench
make run          # gfortran -O3 -march=native -ffast-math -fopenmp-simd -flto
make run FC=ifx   # ifx      -O3 -xHOST -qopenmp-simd -flto
```

Only `!$omp simd` is used (no threading); timing is via `SYSTEM_CLOCK`, so the
SIMD-only flags `-fopenmp-simd` / `-qopenmp-simd` suffice with no OpenMP-runtime
dependency. **`-flto` matters** -- see [The vec8 twist](#the-vec8-twist-a-cross-module-inlining-trap) below; without it the vector-type variant is ~4x slower for reasons that have nothing to do with the abstraction.

## Correctness

Every variant is checked elementwise against the scalar per-matrix reference, and
the reference itself is self-checked via `|| Q^T A - R ||`. All four pass at
machine precision (max error `~1e-14`) under both compilers, e.g. (gfortran):

```
   m    n nrhs     ng | ref-selfchk |  geqr2 (max err)  | orm2r (max err)
   8    8    8   1024 |   6.7E-16   |   3.6E-15         |   4.4E-16   [PASS]
  64   64    8     64 |   6.6E-16   |   1.4E-14         |   1.6E-15   [PASS]
```

## Results

Achieved **GFLOP/s** (higher is better), single core, AVX-512, default flags
(`-flto`). Numbers vary a few % run to run; the rankings are stable.

### gfortran 13.3

**dgeqrf**

| m×n | inner | outer | array | vtype |
|-----|------:|------:|------:|------:|
| 8×8   | 5.45 | 1.98 | **5.55** | 4.43 |
| 16×16 | **8.21** | 2.48 | 8.06 | 6.69 |
| 32×16 | 9.74 | 2.20 | **10.03** | 8.27 |
| 32×32 | 8.66 | 2.48 | **9.50** | 9.03 |
| 48×48 | 10.11 | 2.38 | **10.55** | 8.25 |
| 64×64 | 9.96 | 2.31 | **10.82** | 8.29 |

**dormqr** (nrhs = 8)

| m×n | inner | outer | array | vtype |
|-----|------:|------:|------:|------:|
| 8×8   | 7.76 | 2.04 | **8.67** | 6.75 |
| 16×16 | 8.07 | 2.56 | **8.10** | 6.78 |
| 32×16 | 8.62 | 2.58 | **8.81** | 8.12 |
| 32×32 | 8.01 | 2.50 | **8.84** | 7.12 |
| 48×48 | 8.94 | 2.37 | **9.31** | 6.89 |
| 64×64 | 9.04 | 2.26 | **9.67** | 7.60 |

### Intel `ifx` 2026.1

**dgeqrf**

| m×n | inner | outer | array | vtype |
|-----|------:|------:|------:|------:|
| 8×8   | 9.25 | **9.72** | 7.85 | 6.27 |
| 16×16 | 10.69 | **10.83** | 10.51 | 6.99 |
| 32×16 | **11.60** | 11.43 | 11.20 | 7.08 |
| 32×32 | 12.59 | **12.73** | 12.67 | 8.72 |
| 48×48 | 11.64 | **13.45** | 12.59 | 9.14 |
| 64×64 | 11.39 | 12.32 | **12.34** | 9.02 |

**dormqr** (nrhs = 8)

| m×n | inner | outer | array | vtype |
|-----|------:|------:|------:|------:|
| 8×8   | 8.40 | **9.79** | 7.47 | 5.09 |
| 16×16 | 9.10 | **10.09** | 8.32 | 5.20 |
| 32×16 | 9.46 | **11.16** | 10.08 | 5.79 |
| 32×32 | 7.29 | **8.55** | 7.46 | 4.91 |
| 48×48 | 7.63 | **9.09** | 7.88 | 4.97 |
| 64×64 | 8.12 | 6.41 | **8.59** | 5.19 |

## What the numbers say

Three findings, each with a mechanism confirmed by the compilers' vectorization
reports.

### 1. array-ops and inner-loop are the portable winners

Both express the batch as a clean width-8 loop that **both** compilers turn into
unmasked AVX-512. They are in the fastest tier everywhere, on gfortran and `ifx`
alike, and they are the most readable. `-fopt-info-vec` (gfortran) shows 8
vectorized loops each; array-syntax and an explicit `do b = 1, 8` compile to
effectively the same code. **If you write the kernels one way, write them this
way.**

### 2. outer-loop `!$omp simd` is a compiler gamble

This is the biggest cross-compiler swing in the benchmark:

| | gfortran | ifx |
|--|---------:|----:|
| outer-loop dgeqrf | ~2 GFLOP/s (**worst**) | ~10-13 GFLOP/s (**best/tied**) |

`ifx` genuinely SIMD-vectorizes the batch loop the pragma sits on -- its
`-qopt-report` prints, for that exact loop:

```
OMP SIMD BEGIN at batched_qr.f90 (174, 16)
    LOOP BEGIN at batched_qr.f90 (176, 13)
        remark #15301: SIMD LOOP WAS VECTORIZED
        remark #15305: vectorization support: vector length 8
```

gfortran gives the **same** loop *zero* vectorized loops: it cannot SIMD a batch
loop whose body has inner loops with data-dependent trip counts, pragma or not,
so it falls back to (honest, unrolled) scalar -- hence ~2 GFLOP/s. Same source,
5x difference, purely by compiler. Great when you know your toolchain; risky as
portable code.

### 3. The vec8 twist: a cross-module inlining trap

The vector type *looks* like a clear loser -- until you notice **why**. Built the
naive way (vec8 operators in their own module, no LTO), it runs at ~2 GFLOP/s on
*both* compilers. That is not the cost of the abstraction; it is the compiler
failing to inline the operators across the module boundary, leaving a real
function call per `+`/`*`. Fix the inlining and it collapses:

| vtype dgeqrf 64×64 | gfortran | ifx |
|--------------------|---------:|----:|
| separate module, **no** LTO | 2.3 | 2.4 |
| separate module, **`-flto`** (default here) | **8.3** | **9.0** |
| same file / one translation unit, no LTO | 6.8 | — |

So the whole `~4x` deficit was the module split, not the `vec8` type. With `-flto`
(or `-ipo` on `ifx`, or simply compiling the operators in the same translation
unit as the kernels), the vector type joins the competitive tier -- close behind
array/inner on gfortran, a bit further behind on `ifx`, but nowhere near the
cliff it first appeared to sit on. (`elemental` vs `pure` on the operators makes
**no** measurable difference -- the operators are only ever called on scalars;
the lever is inlining, and only inlining.)

## Bottom line

All four variants are correct to machine precision on both compilers; the rest is
throughput.

1. **`array-ops` (or the equivalent innermost `do b = 1, 8`) is the safe default.**
   Fastest-tier on both compilers, most readable, no special flags needed.
2. **`!$omp simd` over the batch is compiler-dependent:** near-peak on `ifx`,
   ~5x slower than everything else on gfortran. Use it only when you control the
   compiler.
3. **A custom `vec8` type is viable** -- readable *and* competitive -- **but only
   if its operators actually inline.** Keep them in the same translation unit as
   the kernels, or build with `-flto` / `-ipo`. Without that it silently costs
   ~4x, which is easy to misread as "the abstraction is slow."
4. Pinning the interleave width to a compile-time constant (8 = one AVX-512
   register) is worth ~2x over a runtime batch dimension (see git history),
   regardless of style.

Environment: single core, Intel AVX-512 (Cascade Lake), gfortran 13.3 and
`ifx` 2026.1, all builds with `-flto`.
