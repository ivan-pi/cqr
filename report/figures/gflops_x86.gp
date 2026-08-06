# Achieved rate (GFLOP/s) of all three x86 drivers vs matrix order, against the
# LIKWID-measured double-precision roofline. GFLOP/s = (matrices/s) * gflop(n),
# derived from the mats_s columns (cqr=3, mkl=4, lapack=5).
# Run from the report/ directory:  gnuplot figures/gflops_x86.gp
load 'figures/common.gp'
load 'figures/roofline.gp'
set output 'figures/gflops_x86.pdf'

set title "QR factorisation rate on x86 (AVX-512) vs LIKWID roofline"
set ylabel "performance (GFLOP/s)"
set yrange [0:*]
set key top left

plot \
  roof_x86 with lines ls 9 lw 2.5 title 'LIKWID DP peak (measured)', \
  'data/geqrf_x86.dat' using 1:($3*gflop($1)) with linespoints ls 1 title 'cqr (this work)', \
  ''                   using 1:($4*gflop($1)) with linespoints ls 2 title 'MKL Compact', \
  ''                   using 1:($5*gflop($1)) with linespoints ls 3 title 'per-matrix LAPACK'
