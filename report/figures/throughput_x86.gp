# Throughput (matrices/s) vs matrix order on x86 / AVX-512.
# Run from the report/ directory:  gnuplot figures/throughput_x86.gp
load 'figures/common.gp'
set output 'figures/throughput_x86.pdf'

set title "QR factorisation throughput on x86 (AVX-512)"
set logscale xy
set xlabel "matrix order {/:Italic n}"
set ylabel "throughput (matrices / s)"
set format y "10^{%L}"
set xrange [7:560]

plot \
  'data/geqrf_x86.dat' using 1:3 with linespoints ls 1 title 'cqr (this work)', \
  ''                   using 1:4 with linespoints ls 2 title 'MKL Compact', \
  ''                   using 1:5 with linespoints ls 3 title 'per-matrix LAPACK'
