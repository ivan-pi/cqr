# Sustained rate (GFLOP/s) of cqr vs matrix order on both architectures.
# The wiggle at sizes that are not multiples of the interleave width is the
# SIMD-remainder "staircase" discussed in the text.
# Run from the report/ directory:  gnuplot figures/gflops.gp
load 'figures/common.gp'
set output 'figures/gflops.pdf'

set title "cqr sustained rate and the SIMD-remainder staircase"
set logscale x
set xlabel "matrix order {/:Italic n}"
set ylabel "performance (GFLOP/s)"
set xrange [7:560]
set yrange [0:55]
set key bottom right

plot \
  'data/geqrf_x86.dat' using 1:2 with linespoints ls 1 title 'cqr, x86 (AVX-512)', \
  'data/geqrf_arm.dat' using 1:2 with linespoints ls 2 title 'cqr, Arm (SVE)'
