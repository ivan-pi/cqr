# Fortran batched-QR: four ways to vectorize a fixed-width-8 batch

A self-contained Fortran experiment that writes the two workhorse batched-QR
kernels of this project -- the unblocked Householder factorization (`dgeqrf` /
`geqr2`) and the apply-`Q^T` step (`dormqr` / `orm2r`) -- **four different ways**,
then checks that all four agree with a scalar reference and measures which is
fastest.

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
| `vec8_mod.f90` | The `vec8` type (`v(8)`, `VW = 8`) with overloaded operators, `vsqrt`, `vmerge`, `load8`/`store8`. |
| `batched_qr.f90` | Scalar reference (`ref_geqr2`, `ref_orm2r`) + the four `geqr2` and four `orm2r` variants. |
| `bench.f90` | Driver: correctness vs the reference at every size, then adaptive-rep timing. |
| `Makefile` | `make` (gfortran, host-tuned) or `make FC=ifx`; `make run`. |

## Build and run

```sh
cd fortran_bench
make run          # gfortran -O3 -march=native -ffast-math -fopenmp, then run
make run FC=ifx   # or the Intel compiler
```

Single-threaded by design (`make run` sets `OMP_NUM_THREADS=1`): the question is
vectorization of the batch, not threading.

## Correctness

Every variant is checked elementwise against the scalar per-matrix reference, and
the reference itself is self-checked via `|| Q^T A - R ||`. All four pass at
machine precision (max error `~1e-14`), e.g.:

```
   m    n nrhs     ng | ref-selfchk |  geqr2 (max err)  | orm2r (max err)
   8    8    8   1024 |   6.7E-16   |   3.6E-15         |   4.4E-16   [PASS]
  64   64    8     64 |   6.6E-16   |   1.4E-14         |   1.6E-15   [PASS]
```

## Results (gfortran 13.3, `-O3 -march=native -ffast-math`, single core, AVX-512)

Achieved **GFLOP/s** (higher is better). Numbers vary a few % run to run; the
ranking is stable at every size.

**dgeqrf (factorization)**

| m×n | inner | outer | array | vtype |
|-----|------:|------:|------:|------:|
| 8×8   | 5.36 | 1.94 | **5.60** | 1.48 |
| 16×16 | 6.93 | 2.64 | **8.30** | 1.91 |
| 32×16 | 7.92 | 2.19 | **8.24** | 1.85 |
| 32×32 | 7.48 | 2.46 | **8.66** | 2.00 |
| 48×48 | 8.19 | 2.36 | **8.89** | 2.33 |
| 64×64 | **8.75** | 2.27 | 8.61 | 2.30 |

**dormqr (apply Q^T, nrhs = 8)**

| m×n | inner | outer | array | vtype |
|-----|------:|------:|------:|------:|
| 8×8   | **9.32** | 1.98 | 8.53 | 1.74 |
| 16×16 | **8.61** | 2.45 | 8.14 | 1.82 |
| 32×16 | 8.17 | 2.47 | **9.00** | 1.98 |
| 32×32 | **7.85** | 2.50 | 7.70 | 1.94 |
| 48×48 | **8.29** | 2.33 | 8.04 | 1.98 |
| 64×64 | 8.13 | 2.20 | **9.06** | 1.98 |

**Fastest: array-ops and inner-loop, essentially tied** (~5-9 GFLOP/s). They trade
the lead by a few percent from size to size. The vector type and the outer-loop
`simd` trail at ~2 GFLOP/s -- **3-4x slower**.

### Effect of fixing the width to 8

Pinning the batch to a compile-time `8` (versus the earlier runtime `nb`
dimension, see git history) **roughly doubled throughput for every variant**
(fast ones went from ~3-5 to ~5-9 GFLOP/s): the compiler now knows each lane loop
is exactly one AVX-512 register wide, with no remainder and clean unrolling. But
it did **not** change the ranking.

### Why the ranking looks like this

The `-fopt-info-vec` report is decisive -- vectorized loops per `geqr2` variant:

| variant | vectorized loops | what that means |
|---------|-----------------:|-----------------|
| inner-loop  | 8 | every batch loop → real 8-wide SIMD |
| array-ops   | 8 | array sections → the same SIMD code |
| outer-loop  | **0** | the `!$omp simd` batch loop is **not** vectorized |
| vector-type | 2 | only the pack/unpack copies -- **not** the compute |

* **array-ops and inner-loop win** because both present the batch as a clean
  width-8 loop that gfortran turns into unmasked AVX-512 (`vfmadd…pd`, `vsqrtpd`),
  fully unrolled. Array syntax and an explicit `do b = 1, 8` compile to
  effectively the same thing.

* **outer-loop `!$omp simd` gets *zero* vectorized loops even at a fixed width
  of 8.** gfortran cannot SIMD-vectorize a batch loop whose body contains inner
  loops with data-dependent trip counts, pragma or not. It still runs ~4x faster
  than the old runtime-`nb` version only because the fixed trip count lets it
  cleanly *unroll to 8 scalar iterations* instead of emitting gathers -- honest
  scalar, but scalar, hence ~2 GFLOP/s.

* **vector-type is correct and reads beautifully but lands slowest.** The only
  loops that vectorize are the `load8`/`store8` copies; the `vec8` compute
  operators are inlined but gfortran does not fuse their length-8 bodies into
  single wide instructions here. The abstraction that maps straight onto a
  hardware register in C++ (`__attribute__((vector_size))`) does not get the same
  treatment from gfortran's derived-type path.

### Takeaways

1. **Express batch SIMD as array operations over the batch dimension** (or the
   equivalent innermost `do b = 1, 8`). Fastest *and* most readable.
2. Fixing the width to one register (8) is worth ~2x on its own -- but it does
   **not** rescue the two abstraction-heavy styles.
3. **`!$omp simd` over the batch does not vectorize** on gfortran when the body
   has inner loops -- not even at a constant width of 8.
4. A **custom `vec8` type** gives C++-like readability but costs ~3-4x on
   gfortran; it does not reproduce C++ GNU-vector-type performance.

Run `make run FC=ifx` to see how much of #3 and #4 is compiler-specific -- Intel's
`ifx` is generally far more aggressive at both `omp simd` outer-loop vectorization
and small-array / derived-type SIMD, so those gaps may narrow or reorder. (Not
installed in this environment; the numbers above are gfortran.)
