cmake_minimum_required(VERSION 3.19)
project(cqr_ispc_prototype LANGUAGES CXX ISPC)

# Intel MKL (pack/unpack, reference trsm, per-matrix LAPACK).
find_path(MKL_INCLUDE_DIR mkl_compact.h
  HINTS $ENV{MKLROOT}/include $ENV{MKLROOT}/include/mkl
  PATHS /usr/include/mkl /usr/include)
find_library(MKL_RT_LIB mkl_rt HINTS $ENV{MKLROOT}/lib/intel64 $ENV{MKLROOT}/lib)
if(NOT MKL_INCLUDE_DIR OR NOT MKL_RT_LIB)
  message(FATAL_ERROR "Intel MKL not found (need mkl_compact.h and libmkl_rt). "
    "Install libmkl-dev, or source oneAPI setvars.sh (sets MKLROOT).")
endif()

# ISPC kernels as their own library: keeping .ispc and .cpp in separate targets
# avoids the generated-header cross-dependency, and the C++ uses the hand-written
# cqr_ispc.h. The ISPC target/optimization flags are the caller's to set
# (CMAKE_ISPC_INSTRUCTION_SETS, CMAKE_ISPC_FLAGS); the source #errors unless the
# gang width is 8 (see cqr_ispc.ispc / README).
add_library(cqr_ispc_kernels STATIC cqr_ispc.ispc)
set_target_properties(cqr_ispc_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Shared C++ settings. Optimization/-march are the caller's to choose (as in the
# parent project): pass e.g. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-march=native.
add_library(cqr_ispc_common INTERFACE)
target_include_directories(cqr_ispc_common INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../src ${MKL_INCLUDE_DIR})
target_compile_features(cqr_ispc_common INTERFACE cxx_std_17)
target_link_libraries(cqr_ispc_common INTERFACE cqr_ispc_kernels ${MKL_RT_LIB} m)

add_executable(test_cqr_ispc test_cqr_ispc.cpp)
target_link_libraries(test_cqr_ispc PRIVATE cqr_ispc_common)

add_executable(bench_cqr_ispc bench_cqr_ispc.cpp)
target_link_libraries(bench_cqr_ispc PRIVATE cqr_ispc_common)

enable_testing()
add_test(NAME ispc_correctness COMMAND test_cqr_ispc)
