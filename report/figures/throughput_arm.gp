# Throughput (matrices/s) vs matrix order on Arm / SVE.
# Run from the report/ directory:  gnuplot figures/throughput_arm.gp
load 'figures/common.gp'
set output 'figures/throughput_arm.pdf'

set title "QR factorisation throughput on Arm (SVE)"
set logscale xy
set xlabel "matrix order {/:Italic n}"
set ylabel "throughput (matrices / s)"
set format y "10^{%L}"
set xrange [7:560]

plot \
  'data/geqrf_arm.dat' using 1:3 with linespoints ls 1 title 'cqr (this work)', \
  ''                   using 1:4 with linespoints ls 2 title 'ArmPL interleave-batch', \
  ''                   using 1:5 with linespoints ls 3 title 'per-matrix LAPACK'
