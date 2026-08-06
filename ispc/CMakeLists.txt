cmake_minimum_required(VERSION 3.19)
project(cqr_ispc_prototype LANGUAGES CXX ISPC)

# Intel MKL is a dependency of the DRIVERS (pack/unpack, reference trsm, per-matrix
# LAPACK) -- not of the ISPC kernels. Prefer the official package config (oneAPI
# ships MKLConfig.cmake under $MKLROOT); fall back to finding mkl_rt directly, since
# Debian/apt libmkl-dev ships no config.
find_package(MKL CONFIG QUIET)
add_library(cqr_mkl INTERFACE)
if(MKL_FOUND)
  target_link_libraries(cqr_mkl INTERFACE MKL::MKL)
else()
  find_path(MKL_INCLUDE_DIR mkl_compact.h
    HINTS $ENV{MKLROOT}/include $ENV{MKLROOT}/include/mkl PATHS /usr/include/mkl /usr/include)
  find_library(MKL_RT_LIB mkl_rt HINTS $ENV{MKLROOT}/lib/intel64 $ENV{MKLROOT}/lib)
  if(NOT MKL_INCLUDE_DIR OR NOT MKL_RT_LIB)
    message(FATAL_ERROR "Intel MKL not found (need mkl_compact.h and libmkl_rt). "
      "Install libmkl-dev, or source oneAPI setvars.sh (sets MKLROOT).")
  endif()
  target_include_directories(cqr_mkl INTERFACE ${MKL_INCLUDE_DIR})
  target_link_libraries(cqr_mkl INTERFACE ${MKL_RT_LIB})
endif()

# The ISPC kernels: no external dependencies. The target/optimization flags are the
# caller's (CMAKE_ISPC_INSTRUCTION_SETS, CMAKE_ISPC_FLAGS); the kernels work at any
# gang width and the drivers check gang == the compact V they use (see README).
add_library(cqr_ispc_kernels cqr_ispc.ispc)

# What a C++ driver needs from the prototype: the headers and the ISPC kernels.
add_library(cqr_ispc_iface INTERFACE)
target_include_directories(cqr_ispc_iface INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_compile_features(cqr_ispc_iface INTERFACE cxx_std_17)
target_link_libraries(cqr_ispc_iface INTERFACE cqr_ispc_kernels)

# MKL-interop correctness test and benchmark (need MKL). Optimization/-march are the
# caller's, e.g. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-march=native.
add_executable(test_cqr_ispc test_cqr_ispc.cpp)
target_link_libraries(test_cqr_ispc PRIVATE cqr_ispc_iface cqr_mkl)

add_executable(bench_cqr_ispc bench_cqr_ispc.cpp)
target_link_libraries(bench_cqr_ispc PRIVATE cqr_ispc_iface cqr_mkl)

add_executable(bench_potrf bench_potrf.cpp)
target_link_libraries(bench_potrf PRIVATE cqr_ispc_iface cqr_mkl)

# Gang-size sweep: ISPC kernels only, no MKL.
add_executable(bench_gang bench_gang.cpp)
target_link_libraries(bench_gang PRIVATE cqr_ispc_iface)

enable_testing()
add_test(NAME test_cqr_ispc COMMAND test_cqr_ispc)
