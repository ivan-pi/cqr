# Batched-QR expression-style benchmark.
#
#   make            # build with gfortran, host-tuned
#   make FC=ifx     # build with the Intel compiler instead
#   make run        # build and run
#
# SIMD-only: the comparison is about how each expression style vectorizes the
# batch dimension, not about threading. Only !$omp simd is used (no OpenMP
# runtime), so -fopenmp-simd / -qopenmp-simd suffice.

FC = gfortran

ifeq ($(FC),ifx)
  FFLAGS ?= -O3 -xHOST -qopenmp-simd
else
  # gfortran (default). -fopenmp-simd enables the !$omp simd directives only
  # (no threading runtime); timing uses SYSTEM_CLOCK, so this is self-contained.
  FFLAGS ?= -O3 -march=native -funroll-loops -ffast-math -fopenmp-simd
endif

OBJ = vec8_mod.o batched_qr.o bench.o

bench: $(OBJ)
	$(FC) $(FFLAGS) -o $@ $(OBJ)

# module dependency order
batched_qr.o: vec8_mod.o
bench.o: batched_qr.o vec8_mod.o

%.o: %.f90
	$(FC) $(FFLAGS) -c $<

.PHONY: run clean
run: bench
	./bench

clean:
	rm -f *.o *.mod bench
