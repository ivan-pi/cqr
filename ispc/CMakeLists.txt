# ispc/CMakeLists.txt -- standalone build for the ISPC compact-QR prototype,
# using CMake's native ISPC language support (requires CMake >= 3.19).
#
#   cmake -S . -B build
#   cmake --build build -j
#   ctest --test-dir build --output-on-failure     # the correctness unit tests
#   ./build/bench_cqr_ispc                          # the benchmark
#
# Compiler shootout (GCC vs clang): the benchmark's "native" column is whatever
# CXX compiler configured the tree, so configure a second tree to compare, e.g.
#   cmake -S . -B build-clang -DCMAKE_CXX_COMPILER=clang++ && cmake --build build-clang
#   ./build/bench_cqr_ispc ; ./build-clang/bench_cqr_ispc
#
# Standalone (own project) so the ISPC/3.19 requirement stays out of the parent
# build (CMake 3.18, MKL-optional); it can also be add_subdirectory()'d.
# Assisted-by: Claude:claude-opus-4.8

cmake_minimum_required(VERSION 3.19) # enable_language(ISPC)
project(cqr_ispc_prototype LANGUAGES CXX ISPC)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# --- ISPC: pick the ISA. avx512skx-x8 => programCount 8 == V, the AVX-512 FP64
#     compact format MKL selects on an AVX-512 host. One target = one width. ---
set(CMAKE_ISPC_INSTRUCTION_SETS "avx512skx-x8" CACHE STRING "ISPC target ISA")
string(APPEND CMAKE_ISPC_FLAGS " -O3 --opt=disable-assertions")

# --- Intel MKL (pack/unpack, reference trsm, per-matrix LAPACK). ---
find_path(MKL_INCLUDE_DIR mkl_compact.h
  HINTS $ENV{MKLROOT}/include $ENV{MKLROOT}/include/mkl
  PATHS /usr/include/mkl /usr/include)
find_library(MKL_RT_LIB mkl_rt HINTS $ENV{MKLROOT}/lib/intel64 $ENV{MKLROOT}/lib)
if(NOT MKL_INCLUDE_DIR OR NOT MKL_RT_LIB)
  message(FATAL_ERROR "Intel MKL not found (need mkl_compact.h and libmkl_rt). "
    "Install libmkl-dev, or source oneAPI setvars.sh (sets MKLROOT).")
endif()

# --- ISPC kernels as their own library: keeping .ispc and .cpp in separate
#     targets avoids the generated-header cross-dependency, and the C++ uses the
#     hand-written cqr_ispc.h. (Set CMAKE_ISPC_HEADER_DIRECTORY instead if you
#     prefer ISPC's generated header.) ---
add_library(cqr_ispc_kernels STATIC cqr_ispc.ispc)
set_target_properties(cqr_ispc_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)

# --- C++ side: -march=native so the GNU vector_size kernels use the host's full
#     width (a fair opponent for ISPC). Shared settings via an INTERFACE target. ---
add_library(cqr_ispc_common INTERFACE)
target_include_directories(cqr_ispc_common INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../src ${MKL_INCLUDE_DIR})
target_compile_features(cqr_ispc_common INTERFACE cxx_std_17)
target_compile_options(cqr_ispc_common INTERFACE -march=native)
target_link_libraries(cqr_ispc_common INTERFACE cqr_ispc_kernels ${MKL_RT_LIB} m)

add_executable(test_cqr_ispc test_cqr_ispc.cpp)
target_link_libraries(test_cqr_ispc PRIVATE cqr_ispc_common)

add_executable(bench_cqr_ispc bench_cqr_ispc.cpp)
target_link_libraries(bench_cqr_ispc PRIVATE cqr_ispc_common)

enable_testing()
add_test(NAME ispc_correctness COMMAND test_cqr_ispc)
