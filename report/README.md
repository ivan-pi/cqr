# Report: the compact (interleaved) batched linear-algebra API

An arXiv-style short paper positioning the **compact / interleaved-batch format**
for many small matrices across vendors (Intel MKL Compact, Arm Performance
Libraries interleave-batch), with this repository's `cqr` as an open, portable
reference implementation — and a cross-vendor performance study.

## Status: initial draft

* **Prose** — drafted from the repo's design documents and README.
* **Figures** — the gnuplot → LaTeX pipeline is wired and renders, but the data
  in `data/*.dat` is **synthetic placeholder** (see `data/README.md`). Every
  measured claim is flagged in the source with `\draftnote{…}`.
* Open items are marked with `\draftnote{…}` in `paper.tex`; grep for them:
  `grep -n 'draftnote' paper.tex`. Set `\draftmode` to `false` in the preamble
  to hide them for a clean read.

## Build

```sh
cd report
make            # gnuplot figures, then latexmk -> paper.pdf
```

Targets: `make figures` (just the plots), `make data` (regenerate placeholders),
`make clean`, `make veryclean`. Requirements: `gnuplot` (with the `pdfcairo`
terminal), a TeX Live including `latexmk`, and `python3`.

## Layout

```
report/
├── paper.tex                     # the paper
├── refs.bib                      # bibliography
├── Makefile                      # figures -> PDF
├── figures/
│   ├── common.gp                 # shared gnuplot style (palette, axes)
│   ├── throughput_x86.gp         # matrices/s vs n, x86 (cqr / MKL / LAPACK)
│   ├── throughput_arm.gp         # matrices/s vs n, Arm (cqr / ArmPL / LAPACK)
│   ├── speedup_x86.gp            # speedup ratios vs n, x86
│   ├── gflops_x86.gp             # GFLOP/s vs n, x86 (cqr / MKL / LAPACK) + roofline
│   ├── gflops_arm.gp             # GFLOP/s vs n, Arm (cqr / ArmPL / LAPACK) + roofline
│   └── roofline.gp               # LIKWID-measured DP peak levels (edit these)
└── data/
    ├── make_placeholder_data.py  # generates the synthetic .dat files
    ├── geqrf_x86.dat             # PLACEHOLDER (MKL Compact, x86)
    ├── geqrf_arm.dat             # PLACEHOLDER (ArmPL interleave-batch, Arm)
    └── README.md                 # column contract + how to drop in real data
```

## Dropping in real numbers

Replace `data/*.dat` with measured benchmark output (same columns — see
`data/README.md`) and re-run `make`. No figure or paper edits are needed; the
plots and tables re-render from the data.

For the GFLOP/s figures, also set the two roofline ceilings in
`figures/roofline.gp` from a LIKWID measurement on the same node and core count
as the runs (the file documents the exact `likwid-bench` / `likwid-perfctr`
commands). The per-driver GFLOP/s curves are derived from the `mats_s` columns
already in the data, so they need no extra input.
