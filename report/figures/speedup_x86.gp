# Speedup over the per-matrix LAPACK baseline vs matrix order (x86 / AVX-512).
# The guide line at 1.0 marks parity with the baseline.
# Run from the report/ directory:  gnuplot figures/speedup_x86.gp
load 'figures/common.gp'
set output 'figures/speedup_x86.pdf'

set title "Speedup over per-matrix LAPACK (x86, AVX-512)"
set logscale x
set xlabel "matrix order {/:Italic n}"
set ylabel "speedup ({/:Italic x})"
set xrange [7:560]
set yrange [0.8:5]
set key top right

plot \
  1.0 with lines ls 9 title 'parity', \
  'data/geqrf_x86.dat' using 1:6 with linespoints ls 1 title 'cqr / LAPACK', \
  ''                   using 1:7 with linespoints ls 3 title 'MKL / LAPACK', \
  ''                   using 1:8 with linespoints ls 2 title 'cqr / MKL'
