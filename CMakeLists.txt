cmake_minimum_required(VERSION 3.18)

project(cqr
  VERSION 0.1.0
  DESCRIPTION "Compact QR extensions"
  LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# compile_commands.json, which clang-tidy reads (.clang-tidy and the
# manual-stage hook in .pre-commit-config.yaml)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CQR_WITH_MKL    "Build the Intel MKL compact extension, examples and MKL tests" ON)
option(CQR_WITH_OPENMP "Thread each routine's loop over compact groups with OpenMP"     ON)
option(CQR_BUILD_TESTS "Build tests and register them with CTest"                     ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# C++17, no GNU extensions (the kernels' vector_size attribute compiles under
# strict -std=c++17). Optimization flags are the caller's to choose; for a
# performance build pass them through CMAKE_CXX_FLAGS, e.g.
#   cmake -S . -B build -DCMAKE_CXX_FLAGS="-O3 -march=native"
# so the SIMD kernels target the host's widest vectors.

# OpenMP threads the group loop inside every routine (see the threading note
# in src/cqr_compact_common.hpp); the benchmarks use it for their outer loops.
# Linked PRIVATE: consumers get the runtime library, not the -fopenmp flag.
if(CQR_WITH_OPENMP)
  find_package(OpenMP QUIET)
endif()
function(cqr_link_openmp target)
  if(CQR_WITH_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
  endif()
endfunction()

# ---------------------------------------------------------------------------
# Portable kernel library: the C API of include/cqr_compact.h (no BLAS)
# ---------------------------------------------------------------------------
add_library(cqr_compact src/cqr_compact.cpp)
target_include_directories(cqr_compact
  PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_features(cqr_compact PUBLIC cxx_std_17)
cqr_link_openmp(cqr_compact)
add_library(cqr::compact ALIAS cqr_compact)

# ---------------------------------------------------------------------------
# Intel MKL compact extension: cqr_mkl_?{geqrf,ormqr,potrf,trsm,gels}_compact, the
# MKL-style drop-ins of include/cqr_mkl_ext.h (see docs/*_design.md)
# ---------------------------------------------------------------------------
if(CQR_WITH_MKL)
  find_package(MKLCompact REQUIRED)

  add_library(cqr_mkl_ext src/cqr_mkl_ext.cpp)
  target_include_directories(cqr_mkl_ext
    PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_compile_features(cqr_mkl_ext PUBLIC cxx_std_17)
  # MKL types appear in the public header (no link dependency).
  target_link_libraries(cqr_mkl_ext PUBLIC MKL::CompactHeaders)
  cqr_link_openmp(cqr_mkl_ext)
  add_library(cqr::mkl_ext ALIAS cqr_mkl_ext)

  # Examples: the worked batched solve and the three throughput benchmarks
  # (examples/BENCHMARKS.md). All link MKL for the pack/unpack routines and the
  # per-matrix LAPACK baselines; the benchmarks parallelize their outer loop
  # with OpenMP when available.
  foreach(prog solve_qr_compact bench_qr_compact bench_geqrf_compact bench_potrf_compact)
    add_executable(${prog} examples/${prog}.cpp)
    target_link_libraries(${prog} PRIVATE cqr_mkl_ext MKL::Compact)
    if(prog MATCHES "^bench_")
      cqr_link_openmp(${prog})
    endif()
  endforeach()
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(CQR_BUILD_TESTS)
  enable_testing()

  # Portable self-contained suites (no BLAS): each kernel vs a scalar reference
  # of the same unblocked algorithm, plus the C API's LAPACK-style validation.
  foreach(routine geqrf ormqr potrf trsm gels)
    add_executable(test_cqr_${routine}_compact tests/test_cqr_${routine}_compact.cpp)
    target_link_libraries(test_cqr_${routine}_compact PRIVATE cqr_compact)
    add_test(NAME portable_${routine} COMMAND test_cqr_${routine}_compact)
  endforeach()

  if(CQR_WITH_MKL)
    # MKL + dense-LAPACK validation of each cqr_mkl_?*_compact routine (the
    # design docs' section 7): invariants vs LAPACK, cross-checks vs MKL's own
    # compact kernels, and end-to-end solves.
    foreach(routine geqrf ormqr potrf trsm gels)
      add_executable(test_cqr_${routine}_mkl tests/test_cqr_${routine}_mkl.cpp)
      target_link_libraries(test_cqr_${routine}_mkl PRIVATE cqr_mkl_ext MKL::Compact)
      add_test(NAME mkl_${routine}_suites COMMAND test_cqr_${routine}_mkl)
    endforeach()

    # The example self-validates, and the benchmarks carry a correctness gate
    # against per-matrix LAPACK, so they double as integration tests. A small
    # pool with a single rep keeps CTest quick; 33 is not a multiple of any
    # SIMD width, so the padded last group is exercised too.
    add_test(NAME example_solve_qr_compact COMMAND solve_qr_compact)
    foreach(prog bench_qr_compact bench_geqrf_compact bench_potrf_compact)
      add_test(NAME ${prog}_integration COMMAND ${prog} 33 1)
      set_tests_properties(${prog}_integration PROPERTIES LABELS "integration")
    endforeach()
  endif()
endif()
