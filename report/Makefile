# Build the report:  gnuplot figures  ->  pdflatex/bibtex PDF.
#
#   make            # figures + paper.pdf
#   make figures    # just the gnuplot PDFs
#   make placeholder# (re)generate SYNTHETIC .dat files -- see the warning below
#   make clean      # remove LaTeX aux files
#   make veryclean  # also remove paper.pdf and the figure PDFs
#
# The .dat files are treated as source (checked-in data); `make` never
# regenerates them, so a real measured geqrf_x86.dat is safe from being
# clobbered. `make placeholder` re-creates the synthetic data deliberately.
#
# Requires: gnuplot (pdfcairo terminal), a TeX Live with latexmk, python3.

GP      := gnuplot
LATEXMK := latexmk -pdf -halt-on-error -interaction=nonstopmode

FIGS := figures/throughput_x86.pdf figures/speedup_x86.pdf \
        figures/throughput_arm.pdf figures/gflops_x86.pdf figures/gflops_arm.pdf

.PHONY: all figures placeholder clean veryclean
all: paper.pdf

paper.pdf: paper.tex refs.bib $(FIGS)
	$(LATEXMK) paper.tex

figures: $(FIGS)

# Each figure is produced by its own gnuplot script; scripts use paths relative
# to this directory, so gnuplot must be invoked from here (as Make does).
figures/%.pdf: figures/%.gp figures/common.gp
	$(GP) $<

# Each figure depends only on its own platform's data, so re-fitting one platform
# does not re-run the others; the GFLOP/s figures also read the roofline levels.
figures/throughput_x86.pdf figures/speedup_x86.pdf figures/gflops_x86.pdf: data/geqrf_x86.dat
figures/throughput_arm.pdf figures/gflops_arm.pdf: data/geqrf_arm.dat
figures/gflops_x86.pdf figures/gflops_arm.pdf: figures/roofline.gp

# (Re)generate the SYNTHETIC placeholder .dat files. Opt-in only: this OVERWRITES
# data/geqrf_x86.dat, which holds real measurements. To use real numbers, drop in
# measured output (same columns -- see data/README.md) instead of running this.
placeholder:
	python3 data/make_placeholder_data.py

clean:
	$(LATEXMK) -c paper.tex 2>/dev/null || true
	rm -f *.aux *.bbl *.blg *.log *.out *.fls *.fdb_latexmk

veryclean: clean
	rm -f paper.pdf $(FIGS)
