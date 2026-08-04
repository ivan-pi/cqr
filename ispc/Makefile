# Makefile -- standalone build for the ISPC compact-QR prototype.
#
#   make            # test + benchmark (gcc and clang builds)
#   make test       # build + run the correctness test
#   make bench      # build + run the benchmark (gcc build)
#   make shootout   # run the gcc AND clang builds (the compiler shootout)
#   make clean
#
# Needs Intel MKL (apt libmkl-dev, or oneAPI) for pack/unpack, the reference
# trsm, and per-matrix LAPACK. The drivers pin MKL sequential at startup, so no
# threading runtime or MKL_THREADING_LAYER env var is needed.
# Assisted-by: Claude:claude-opus-4.8

ISPC      ?= ispc
CXX_GCC   ?= g++
CXX_CLANG ?= clang++
MKL_INC   ?= /usr/include/mkl

# --target=avx512skx-x8 => programCount 8 == V (AVX-512 FP64 compact format).
ISPC_FLAGS ?= --target=avx512skx-x8 --arch=x86-64 -O3 --pic --opt=disable-assertions
# Release + native so the GNU vector_size kernels see the host's full width.
CXXFLAGS   ?= -O3 -march=native -std=c++17 -Wall -I. -I../src -I$(MKL_INC)
LDLIBS     ?= -lmkl_rt -lm

.PHONY: all test bench shootout clean
all: test_cqr_ispc bench_gcc bench_clang

cqr_ispc.o: cqr_ispc.ispc
	$(ISPC) $(ISPC_FLAGS) -o $@ -h cqr_ispc_generated.h $<

test_cqr_ispc: test_cqr_ispc.cpp cqr_ispc.o
	$(CXX_GCC) $(CXXFLAGS) -o $@ $< cqr_ispc.o $(LDLIBS)

# The same benchmark built by g++ and by clang++: the native column is that
# compiler's codegen, so the two builds are the GCC and clang shootout entries.
BENCH_DEPS = bench_cqr_ispc.cpp bench_common.hpp cqr_trsm_compact.hpp cqr_ispc.o
bench_gcc: $(BENCH_DEPS)
	$(CXX_GCC) $(CXXFLAGS) -o $@ bench_cqr_ispc.cpp cqr_ispc.o $(LDLIBS)
bench_clang: $(BENCH_DEPS)
	$(CXX_CLANG) $(CXXFLAGS) -o $@ bench_cqr_ispc.cpp cqr_ispc.o $(LDLIBS)

test: test_cqr_ispc
	./test_cqr_ispc
bench: bench_gcc
	./bench_gcc
shootout: bench_gcc bench_clang
	./bench_gcc
	@echo
	./bench_clang

clean:
	rm -f *.o cqr_ispc_generated.h test_cqr_ispc bench_gcc bench_clang
