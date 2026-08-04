# Batched-QR expression-style benchmark.
#
#   make            # build with gfortran, host-tuned
#   make FC=ifx     # build with the Intel compiler instead
#   make run        # build and run
#
# Single-threaded by design: the comparison is about how each expression style
# vectorizes the batch dimension, not about threading. Run with OMP_NUM_THREADS=1.

FC = gfortran

ifeq ($(FC),ifx)
  FFLAGS ?= -O3 -xHost -qopenmp -qopenmp-simd -fp-model=precise
else
  # gfortran (default)
  FFLAGS ?= -O3 -march=native -funroll-loops -ffast-math -fopenmp
endif

OBJ = dvec_mod.o batched_qr.o bench.o

bench: $(OBJ)
	$(FC) $(FFLAGS) -o $@ $(OBJ)

# module dependency order
batched_qr.o: dvec_mod.o
bench.o: batched_qr.o dvec_mod.o

%.o: %.f90
	$(FC) $(FFLAGS) -c $<

.PHONY: run clean
run: bench
	OMP_NUM_THREADS=1 ./bench

clean:
	rm -f *.o *.mod bench
