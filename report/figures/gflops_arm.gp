# Achieved rate (GFLOP/s) of all three Arm drivers vs matrix order, against the
# LIKWID-measured double-precision roofline. GFLOP/s = (matrices/s) * gflop(n),
# derived from the mats_s columns (cqr=3, armpl=4, lapack=5).
# Run from the report/ directory:  gnuplot figures/gflops_arm.gp
load 'figures/common.gp'
load 'figures/roofline.gp'
set output 'figures/gflops_arm.pdf'

set title "QR factorisation rate on Arm (SVE) vs LIKWID roofline"
set logscale x
set xlabel "matrix order {/:Italic n}"
set ylabel "performance (GFLOP/s)"
set xrange [7:560]
set yrange [0:*]
set key top left

plot \
  roof_arm with lines ls 9 lw 2.5 title 'LIKWID DP peak (measured)', \
  'data/geqrf_arm.dat' using 1:($3*gflop($1)) with linespoints ls 1 title 'cqr (this work)', \
  ''                   using 1:($4*gflop($1)) with linespoints ls 2 title 'ArmPL interleave-batch', \
  ''                   using 1:($5*gflop($1)) with linespoints ls 3 title 'per-matrix LAPACK'
