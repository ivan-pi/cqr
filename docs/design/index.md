# Design notes

Each `cqr` routine has a full design document covering its algorithm, numerical
scope, layout handling, testing methodology, and implementation strategy. These
are the same documents that live at the repository root — published here
unchanged, so the site and the source tree never drift.

They are written for contributors and for readers who want to understand *how*
and *why* the kernels are built the way they are. If you only need to call the
routines, start with the [User guide](../guide.md) and the
[API reference](../reference.md).

<div class="grid cards" markdown>

-   :material-matrix: __[geqrf — QR factorization](geqrf.md)__

    Vectorized unblocked Householder QR (`geqr2`) with a branch-free masked
    `larfg`. The open, vectorized alternative to `mkl_?geqrf_compact`.

-   :material-rotate-orbit: __[ormqr — apply Q](ormqr.md)__

    Applies `Q` or `Qᵀ` to a compact batch — the `mkl_?ormqr_compact` MKL never
    shipped.

-   :material-triangle-outline: __[potrf — Cholesky](potrf.md)__

    Vectorized unblocked Cholesky (`potf2`) for SPD batches, with a
    register-blocked rank-1 trailing update.

-   :material-function-variant: __[trsm — triangular solve](trsm.md)__

    Vectorized batched substitution with tail-blocked register blocking — the
    step that closes the batched `AX = B` solve.

</div>
