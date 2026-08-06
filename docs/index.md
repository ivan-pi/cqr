# cqr — Compact QR extensions

**Batched QR, Cholesky, and triangular solve for many small matrices**, stored
in Intel MKL's **Compact** (interleaved) format. `cqr` provides portable,
SIMD-vectorized kernels behind an Intel MKL-style API — together they factor and
solve batched systems **entirely in the compact format, with no MKL compute
kernel**.

The kernels are written with GNU vector types
(`__attribute__((vector_size))`), which the compiler lowers to SSE, AVX, or
AVX-512 — one portable source for every width. That is the project's central
SIMD decision.

[Get started](guide.md){ .md-button .md-button--primary }
[API reference](reference.md){ .md-button }
[Download](download.md){ .md-button }

## The routines

<div class="grid cards" markdown>

-   :material-matrix: __`cqr_mkl_?geqrf_compact` — QR factorization__

    An open, vectorized alternative to `mkl_?geqrf_compact`. On this project's
    AVX-512 test machine it outruns MKL's own compact `geqrf` and per-matrix
    LAPACK across the small-size range.

    [Design notes →](design/geqrf.md)

-   :material-rotate-orbit: __`cqr_mkl_?ormqr_compact` — apply Q__

    The apply-`Q` step MKL omits. MKL ships `geqrf`/`trsm` compact but no
    `ormqr_compact`, so there is no supported way to apply `Q` (or `Qᵀ`) to a
    batch. `cqr` fills that gap.

    [Design notes →](design/ormqr.md)

-   :material-triangle-outline: __`cqr_mkl_?potrf_compact` — Cholesky__

    Batched Cholesky (`A = L Lᵀ` / `Uᵀ U`) of SPD matrices; a portable,
    vectorized alternative to `mkl_?potrf_compact`. Paired with `trsm` it
    factors and solves batched SPD systems.

    [Design notes →](design/potrf.md)

-   :material-function-variant: __`cqr_mkl_?trsm_compact` — triangular solve__

    An open drop-in for `mkl_?trsm_compact` (the batched triangular solve), so
    the whole `AX = B` pipeline runs with no MKL compute kernel.

    [Design notes →](design/trsm.md)

</div>

All routines come in single and double precision.

## The batched solve, end to end

Applying the QR routines in sequence solves a batch of systems `Aᵥ Xᵥ = Bᵥ`:

```c
cqr_mkl_dgeqrf_compact(..., A -> H, tau);        // A = Q R
cqr_mkl_dormqr_compact('L', 'T', ..., H, tau, B); // B := Qᵀ B
cqr_mkl_dtrsm_compact (..., R, B);               // B := R⁻¹ Qᵀ B = X
```

MKL's compact pack/unpack helpers still move data in and out of the interleaved
layout; only the *computation* is `cqr`'s.

## Two API surfaces

| Header | Audience |
|--------|----------|
| [`cqr_mkl_ext.h`](reference.md#mkl-style-api) | **MKL-style API** — drop-in for the `mkl_?*_compact` routines, taking `MKL_LAYOUT` + `MKL_COMPACT_PACK`. Mixes freely with MKL's native compact routines. |
| [`cqr_compact.h`](reference.md#portable-c-api) | **Portable C API** — no MKL dependency, explicit interleave width `V`, LAPACK-style `info = -j` argument checking. |

## Project status

`cqr` is pre-1.0 (`0.1.0`). The four routine families are implemented,
validated against MKL and dense LAPACK to machine precision, and benchmarked.
Real precisions (`s`/`d`) only; complex (`c`/`z`) is out of scope. See
[`PLANS.md`](https://github.com/ivan-pi/cqr/blob/main/PLANS.md) for the detailed
status against each design document.

## License

`cqr` is released under the
[Apache-2.0](https://github.com/ivan-pi/cqr/blob/main/LICENSE) license.
