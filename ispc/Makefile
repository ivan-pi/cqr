# Makefile -- standalone build for the ISPC compact-QR prototype.
#
# Builds three things and links them against the apt/oneAPI MKL (for pack/unpack,
# the reference trsm, and per-matrix LAPACK):
#   * the ISPC kernels            (cqr_ispc.ispc  -> cqr_ispc.o)
#   * the reference GNU kernels    (../src/*_dispatch.cpp, -O3 -march=native)
#   * the test / benchmark drivers (C++17)
#
# Usage:
#   make            # build test + benchmark
#   make test       # build and run the correctness test
#   make bench      # build and run the benchmark
#   make clean
#
# Override the compilers / MKL include dir on the command line if needed, e.g.
#   make MKL_INC=/opt/intel/oneapi/mkl/latest/include
#
# Assisted-by: Claude:claude-opus-4.8

ISPC        ?= ispc
CXX         ?= g++
MKL_INC     ?= /usr/include/mkl

# ISPC: one gang width == one interleave width. avx512skx-x8 => programCount 8 ==
# V for the AVX-512 FP64 compact format. --pic for linking into the driver, -O3.
ISPC_TARGET ?= avx512skx-x8
ISPC_FLAGS  ?= --target=$(ISPC_TARGET) --arch=x86-64 -O3 --pic --opt=disable-assertions

# The reference GNU-vector kernels must see the host's widest vectors to be a
# fair opponent for ISPC (README's performance-build guidance): -march=native.
CXX_OPT     ?= -O3 -march=native -std=c++17 -Wall
CXXFLAGS    ?= $(CXX_OPT) -I../src -I$(MKL_INC)
LDLIBS      ?= -lmkl_rt -lm

# Sequential only: no OpenMP. The drivers pin MKL to its sequential layer at
# startup (mkl_set_threading_layer), so no threading runtime is linked or loaded.

.PHONY: all test bench clean
all: test_cqr_ispc bench_cqr_ispc

cqr_ispc.o: cqr_ispc.ispc
	$(ISPC) $(ISPC_FLAGS) -o $@ -h cqr_ispc_generated.h $<

# Reference kernels compiled straight from the repo sources.
cqr_geqrf_compact_dispatch.o: ../src/cqr_geqrf_compact_dispatch.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<
cqr_compact_dispatch.o: ../src/cqr_compact_dispatch.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

REF_OBJS = cqr_geqrf_compact_dispatch.o cqr_compact_dispatch.o

test_cqr_ispc: test_cqr_ispc.cpp cqr_ispc.o $(REF_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

bench_cqr_ispc: bench_cqr_ispc.cpp cqr_ispc.o $(REF_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

test: test_cqr_ispc
	./test_cqr_ispc

bench: bench_cqr_ispc
	./bench_cqr_ispc

clean:
	rm -f *.o cqr_ispc_generated.h test_cqr_ispc bench_cqr_ispc
