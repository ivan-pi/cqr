# Batched-QR expression-style benchmark.
#
#   make            # build with gfortran, host-tuned
#   make FC=ifx     # build with the Intel compiler instead
#   make run        # build and run the expression-style benchmark
#   make run-mkl    # build and run the fastest-Fortran vs mkl_dgeqrf_compact
#                   # comparison (needs MKL; link vars below)
#
# SIMD-only: the comparison is about how each expression style vectorizes the
# batch dimension, not about threading. Only !$omp simd is used (no OpenMP
# runtime), so -fopenmp-simd / -qopenmp-simd suffice.

FC = gfortran

# -flto is important here: the vec8 operators live in a separate module, and
# without link-time optimization neither compiler inlines them across the module
# boundary -- which alone makes the vector-type variant ~4x slower (an artifact
# of the module split, not the abstraction). -flto (ifx also accepts -ipo)
# restores it. See README "The vec8 twist".
ifeq ($(FC),ifx)
  FFLAGS ?= -O3 -xHOST -qopenmp-simd -flto
else
  # gfortran (default). -fopenmp-simd enables the !$omp simd directives only
  # (no threading runtime); timing uses SYSTEM_CLOCK, so this is self-contained.
  FFLAGS ?= -O3 -march=native -funroll-loops -ffast-math -fopenmp-simd -flto
endif

OBJ = vec8_mod.o batched_qr.o bench.o

bench: $(OBJ)
	$(FC) $(FFLAGS) -o $@ $(OBJ)

# module dependency order
batched_qr.o: vec8_mod.o
bench.o: batched_qr.o vec8_mod.o

%.o: %.f90
	$(FC) $(FFLAGS) -c $<

.PHONY: run clean run-mkl
run: bench
	./bench

# --- MKL comparison (optional) -------------------------------------------
# Link against a sequential (single-threaded) MKL for a fair 1-core compare.
# Debian/Ubuntu libmkl-dev provides the GF-ABI LP64 libs below; oneAPI users
# can override MKL_LIBS on the command line.
MKL_LIBS ?= -lmkl_gf_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

bench_mkl: vec8_mod.o batched_qr.o mkl_compact.o bench_mkl.o
	$(FC) $(FFLAGS) -o $@ $^ $(MKL_LIBS)

mkl_compact.o: mkl_compact.f90
	$(FC) $(FFLAGS) -c $<
bench_mkl.o: batched_qr.o vec8_mod.o mkl_compact.o

run-mkl: bench_mkl
	MKL_NUM_THREADS=1 ./bench_mkl

clean:
	rm -f *.o *.mod *.optrpt bench bench_mkl
