# Fortran batched-QR: four ways to vectorize the batch dimension

A self-contained Fortran experiment that writes the two workhorse batched-QR
kernels of this project -- the unblocked Householder factorization (`dgeqrf` /
`geqr2`) and the apply-`Q^T` step (`dormqr` / `orm2r`) -- **four different ways**,
then checks that all four agree with a scalar reference and measures which is
fastest.

It is the Fortran counterpart to the question the C++ library already answered
with GNU vector types: *what is the best way to express SIMD over a batch of
many small matrices?* Here the same math is written in four idiomatic Fortran
styles and handed to the compiler.

## The four variants

All four operate on **one shared memory layout** -- the compact / interleaved
layout `A(nb, m, n)` with the **batch index fastest-varying** (Fortran
column-major, so consecutive batch elements are contiguous). That makes SIMD over
the batch a unit-stride operation for every variant; the *only* thing that
differs is how each expresses it:

| # | Variant | How the batch SIMD is written | Source |
|---|---------|-------------------------------|--------|
| 1 | **inner-loop** | explicit `DO` loops, **batch index innermost** (`do b`); each micro-kernel is a unit-stride batch loop the compiler vectorizes | `geqr2_inner`, `orm2r_inner` |
| 2 | **outer-loop** | one `!$omp simd` loop **over the batch**, the whole scalar per-matrix algorithm in the body (one matrix per SIMD lane) | `geqr2_outer`, `orm2r_outer` |
| 3 | **array-ops** | Fortran whole-array syntax `A(:,i,j)` over the batch dimension; reads like scalar `geqr2` with the batch collapsed into a `(:)` | `geqr2_array`, `orm2r_array` |
| 4 | **vector-type** | a `VL`-wide derived type (`dvec`) with overloaded `+ - * /`, `sqrt`, masked `merge`, packing `VL` matrices per group -- the Fortran analogue of the C++ GNU vector types | `geqr2_vtype`, `orm2r_vtype` |

The data-dependent branch in LAPACK's `dlarfg` (`if xnorm == 0: tau = 0`) is
replaced everywhere by the branch-free `tail > 0` mask, so every lane runs
unmasked with identical semantics to LAPACK `?geqr2` / `?orm2r` (padding and
rank-deficient columns fall out as `tau = 0`, diagonal preserved).

## Files

| File | Role |
|------|------|
| `dvec_mod.f90` | The `dvec` vector type (`VL = 8` FP64 lanes) with overloaded operators, `vsqrt`, `dmerge`, and group pack/unpack. |
| `batched_qr.f90` | Scalar reference (`ref_geqr2`, `ref_orm2r`) + the four `geqr2` and four `orm2r` variants. |
| `bench.f90` | Driver: correctness vs the reference at every size, then adaptive-rep timing. |
| `Makefile` | `make` (gfortran, host-tuned) or `make FC=ifx`; `make run`. |

## Build and run

```sh
cd fortran_bench
make run          # gfortran -O3 -march=native -ffast-math -fopenmp, then run

# or with the Intel compiler:
make run FC=ifx
```

The comparison is deliberately **single-threaded** (`make run` sets
`OMP_NUM_THREADS=1`): the question is how each style vectorizes the batch, not
how it threads. `!$omp simd` in variant 2 is a *vectorization* directive, not a
threading one.

## Correctness

Every variant is checked elementwise against the scalar per-matrix reference
(same unblocked algorithm, same sign convention, so they must agree to rounding),
and the reference itself is semantically self-checked via `|| Q^T A - R ||`. All
four pass at machine precision:

```
   m    n nrhs     nb | ref-selfchk |  geqr2 (max err)  | orm2r (max err)
   8    8    8   8192 |   6.6E-16   |   3.6E-15         |   4.4E-16   [PASS]
  ...
  64   64    8    512 |   6.6E-16   |   2.8E-14         |   1.1E-15   [PASS]
```

## Results (gfortran 13.3, `-O3 -march=native -ffast-math`, single core, AVX-512)

Representative GFLOP/s (higher is better); the full sweep is in the program
output. Numbers vary a few % run to run, but the *ranking* is stable at every
size.

| variant | dgeqrf (n=16) | dgeqrf (n=64) | dormqr (n=64) |
|---------|--------------:|--------------:|--------------:|
| **array-ops**   | **3.5** | **4.2** | **5.6** |
| **inner-loop**  | 3.1 | 3.8 | 5.5 |
| vector-type     | 1.3 | 1.7 | 1.1 |
| outer-loop simd | 0.48 | 0.33 | 0.54 |

**Fastest: array-ops, with inner-loop essentially tied.** The two top variants
trade the lead by a few percent from size to size and are statistically
indistinguishable.

### Why the ranking looks like this

The `-fopt-info-vec` report explains all of it:

* **array-ops and inner-loop win** because both present the compiler with the
  batch as a long, unit-stride innermost loop. gfortran vectorizes it directly
  (AVX-512, with unrolling) -- the reduction, the reflector scaling, and the
  trailing update all become clean lane-wise vector loops. Array syntax and an
  explicit batch-innermost `DO` compile to effectively the same code.

* **outer-loop `!$omp simd` is the slowest by ~10x** because gfortran does **not**
  SIMD-vectorize the batch loop it is applied to. The report shows the `do b`
  loop (the `simd` loop) left scalar, while the compiler instead vectorized the
  *inner* `do i` loops -- which walk the `m` dimension at stride `nb`, i.e.
  gathers. So the one variant that asks for "one matrix per lane" gets neither
  the lane parallelism nor good memory access. Outer-loop vectorization of a body
  containing inner loops and control flow is a known weak spot for gfortran; this
  is exactly why the C++ side of the project uses explicit vector types rather
  than trusting a `simd` pragma.

* **vector-type is correct and reads beautifully but lands ~2-3x behind
  array-ops.** The `dvec` operators inline, but gfortran does not fuse the
  length-`VL` register loops into single wide vector instructions here (none of
  the kernel loops in `geqr2_vtype` show up as vectorized), and it processes only
  `VL = 8` matrices per group with pack/unpack overhead instead of streaming the
  whole batch. The abstraction that maps straight onto a hardware register in
  C++ (`__attribute__((vector_size))`) does **not** get the same treatment from
  gfortran's derived-type path.

### Takeaways

1. In Fortran, **express batch SIMD as array operations over the batch
   dimension** (or, equivalently, an explicit batch-innermost loop). It is the
   fastest *and* the most readable.
2. **Do not** reach for `!$omp simd` over the batch with the scalar algorithm in
   the body and expect one-matrix-per-lane vectorization from gfortran -- it
   regresses badly.
3. A **custom vector type** gives C++-like readability and is a reasonable
   portability tool, but on gfortran it costs a 2-3x throughput penalty versus
   plain array syntax; it does not reproduce the C++ GNU-vector-type performance.

Try `make run FC=ifx` to see how much of #2 and #3 is compiler-specific --
Intel's `ifx` is generally far more aggressive at both `omp simd` outer-loop
vectorization and small-array/derived-type SIMD, so the gaps may narrow or
reorder. (Not installed in this environment; gfortran numbers above.)
