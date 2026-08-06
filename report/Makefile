# Build the report:  gnuplot figures  ->  pdflatex/bibtex PDF.
#
#   make            # figures + paper.pdf
#   make figures    # just the gnuplot PDFs
#   make data       # regenerate the (placeholder) .dat files
#   make clean      # remove LaTeX aux files
#   make veryclean  # also remove paper.pdf and the figure PDFs
#
# Requires: gnuplot (pdfcairo terminal), a TeX Live with latexmk, python3.

GP      := gnuplot
LATEXMK := latexmk -pdf -halt-on-error -interaction=nonstopmode

FIGS := figures/throughput_x86.pdf figures/speedup_x86.pdf \
        figures/throughput_arm.pdf figures/gflops.pdf
DATA := data/geqrf_x86.dat data/geqrf_arm.dat

.PHONY: all figures data clean veryclean
all: paper.pdf

paper.pdf: paper.tex refs.bib $(FIGS)
	$(LATEXMK) paper.tex

figures: $(FIGS)

# Each figure is produced by its own gnuplot script; scripts use paths relative
# to this directory, so gnuplot must be invoked from here (as Make does).
figures/%.pdf: figures/%.gp figures/common.gp $(DATA)
	$(GP) $<

# Regenerate the placeholder data. DELETE THIS ONCE REAL DATA LANDS: replace the
# .dat files with measured benchmark output and drop the dependency on the script.
data: $(DATA)
$(DATA): data/make_placeholder_data.py
	python3 data/make_placeholder_data.py

clean:
	$(LATEXMK) -c paper.tex 2>/dev/null || true
	rm -f *.aux *.bbl *.blg *.log *.out *.fls *.fdb_latexmk

veryclean: clean
	rm -f paper.pdf $(FIGS)
