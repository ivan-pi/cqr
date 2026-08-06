# Shared gnuplot style for the report figures (loaded by each *.gp script).
# Okabe-Ito colourblind-safe palette; each series gets a distinct colour, dash,
# and point type so the curves stay separable in greyscale print too.

set terminal pdfcairo size 4.3in,3.0in font "Sans,10" rounded lw 1.5

set border 3 lc rgb "#4d4d4d"          # draw left + bottom axes only
set tics nomirror out
set grid back lc rgb "#dddddd" lw 1.0

set key top right samplen 2.0 spacing 1.2 opaque

# Shared x-axis: every figure plots against matrix order on a log scale over the
# same range. Each figure adds its own y-axis (logscale y for throughput/speedup;
# gflops keeps a linear y).
set logscale x
set xlabel "matrix order {/:Italic n}"
set xrange [7:560]

# .dat column contract (see report/data/README.md), for the `using` specs:
#   1 n            2 cqr_gflops     3 cqr_mats_s     4 vendor_mats_s
#   5 lapack_mats_s  6 sp_cqr_lap   7 sp_vendor_lap  8 sp_cqr_vendor   9 relerr

# 1 = cqr (this work)   2 = vendor (MKL / ArmPL)   3 = per-matrix LAPACK
# 9 = reference / guide line
set style line 1 lc rgb "#0072B2" lw 2.5 pt 7 ps 0.6 dt 1       # blue,       filled circle, solid
set style line 2 lc rgb "#D55E00" lw 2.5 pt 5 ps 0.6 dt (8,4)   # vermillion, filled square, dashed
set style line 3 lc rgb "#009E73" lw 2.5 pt 9 ps 0.7 dt (2,3)   # green,      triangle,      dotted
set style line 9 lc rgb "#999999" lw 1.5 dt (4,4)               # grey guide line

# LAPACK ?geqrf flop count for a square n x n matrix, in GFLOP. Lets the GFLOP/s
# figures derive each driver's rate from its matrices/s column, since
# GFLOP/s = (matrices/s) * gflop(n). (Equal to the stored cqr_gflops column.)
gflop(n) = (2.0 * n * n * n - (2.0 / 3.0) * n * n * n) * 1e-9
