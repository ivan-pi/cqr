# Measured arithmetic roofline (double-precision compute ceiling) for the
# GFLOP/s figures. Loaded by gflops_x86.gp / gflops_arm.gp.
#
#   !!!  PLACEHOLDER VALUES -- replace with LIKWID measurements.  !!!
#
# The ceiling must be measured on the SAME node and at the SAME core/thread count
# as the benchmark runs. The collector drives the batch from an OpenMP outer loop,
# so measure the multicore peak, not a single core:
#
#   x86 (AVX-512):  likwid-bench -t peakflops_avx512_fma -w S0:2GB:<ncores>
#   Arm (SVE)    :  likwid-bench -t peakflops_sve_fma    -w S0:2GB:<ncores>
#
# Alternatively, read the achieved rate of the kernel directly and report it
# against that same peak:
#
#   likwid-perfctr -C 0-<ncores-1> -g FLOPS_DP ./bench_geqrf_collect ...
#
# Units: GFLOP/s (aggregate over the cores used).
# x86 value below is a real measurement on the development VM:
#   likwid-bench -t peakflops_avx512_fma -w S0:100kB:4  ->  316899 MFlops/s
# (4 threads, ~88% of the 2.8 GHz AVX-512 FMA theoretical peak). Re-measure on
# the final node at the run's core count.
roof_x86 = 316.9   # MEASURED (dev VM, 4 threads); GFLOP/s
roof_arm = 90.0    # PLACEHOLDER -- LIKWID DP peak, Arm node
