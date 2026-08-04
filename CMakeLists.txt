# Assisted-by: Claude:claude-opus-4.8

cmake_minimum_required(VERSION 3.18)

project(cqr
  VERSION 0.1.0
  DESCRIPTION "Compact QR extensions"
  LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(CQR_WITH_MKL        "Build the Intel MKL compact extension + MKL tests" ON)
option(CQR_BUILD_TESTS     "Build tests and register them with CTest"          ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# C++17 is required per-target via target_compile_features below; no GNU
# extensions are needed (the kernel's vector_size attribute compiles under
# strict -std=c++17). Optimization flags are the caller's to choose: for a
# performance build pass them through CMAKE_CXX_FLAGS, e.g.
#   cmake -S . -B build -DCMAKE_CXX_FLAGS="-O3 -march=native"
# so the SIMD kernels target the host's widest vectors.

# ---------------------------------------------------------------------------
# Core kernel library (portable; no BLAS dependency)
# ---------------------------------------------------------------------------
add_library(cqr_compact
  src/cqr_compact_dispatch.cpp
  src/cqr_geqrf_compact_dispatch.cpp)
target_include_directories(cqr_compact PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_features(cqr_compact PUBLIC cxx_std_17)
add_library(cqr::compact ALIAS cqr_compact)

# ---------------------------------------------------------------------------
# OpenMP SIMD experiment (portable; no BLAS/MKL)
#
# An alternative geqrf backend that vectorizes the outer batch loop with
# `#pragma omp simd simdlen(V)` instead of GNU vector types. -fopenmp-simd
# enables only the SIMD lowering (no OpenMP runtime, no threads); the kernel is
# correct without it -- the pragma is simply ignored and the loop runs scalar.
# ---------------------------------------------------------------------------
include(CheckCXXCompilerFlag)
set(CQR_OMP_SIMD_FLAGS "")
if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
  # icpx: -qopenmp-simd lowers `#pragma omp simd` through Intel's own SIMD code
  # generator, which vectorizes the *outer* lane loop -- unlike -fopenmp-simd
  # (the LLVM auto-vectorizer path, which every compiler including icpx routes
  # through and which cannot vectorize that nest). It is simd-only: no OpenMP
  # runtime is linked. See cqr_geqrf_omp_simd_results.md.
  check_cxx_compiler_flag("-qopenmp-simd" CQR_HAS_QOPENMP_SIMD)
  if(CQR_HAS_QOPENMP_SIMD)
    list(APPEND CQR_OMP_SIMD_FLAGS "-qopenmp-simd")
  else()
    message(WARNING "icpx rejects -qopenmp-simd; the omp-simd backend builds "
                    "correct but scalar")
  endif()
else()
  check_cxx_compiler_flag("-fopenmp-simd" CQR_HAS_FOPENMP_SIMD)
  if(CQR_HAS_FOPENMP_SIMD)
    list(APPEND CQR_OMP_SIMD_FLAGS "-fopenmp-simd")
  else()
    message(WARNING "compiler rejects -fopenmp-simd; the omp-simd geqrf backend "
                    "will build correct but scalar (no SIMD)")
  endif()
  # The outer-loop simd kernel (geqrf_compact_group_omp) keeps `#pragma omp simd`
  # on the lane loop even though neither GCC nor Clang can vectorize that nest --
  # the "requested but not vectorized" data point of the experiment. Clang
  # reports the failed transform via -Wpass-failed; silence it so the build stays
  # warning-clean. GCC does not know this flag but ignores unrecognised -Wno-*.
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    list(APPEND CQR_OMP_SIMD_FLAGS "-Wno-pass-failed")
  endif()
endif()

# Drop-in variant of the portable library: the ormqr entry points from the
# vector-types dispatch (shared, unchanged) plus the geqrf entry points routed
# to the omp-simd kernel. It defines the same dgeqrf_compact / sgeqrf_compact C
# symbols as cqr_compact, so a program links exactly one of the two.
add_library(cqr_compact_ompsimd
  src/cqr_compact_dispatch.cpp
  src/cqr_geqrf_compact_dispatch_omp.cpp)
target_include_directories(cqr_compact_ompsimd PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_features(cqr_compact_ompsimd PUBLIC cxx_std_17)
target_compile_options(cqr_compact_ompsimd PRIVATE ${CQR_OMP_SIMD_FLAGS})
add_library(cqr::compact_ompsimd ALIAS cqr_compact_ompsimd)

# Portable head-to-head benchmark: vector-types vs omp-simd geqrf, no MKL. The
# two kernel templates are called directly (header-only), so it needs the
# source include dir and -fopenmp-simd, nothing else.
add_executable(bench_geqrf_omp_simd examples/bench_geqrf_omp_simd.cpp)
target_include_directories(bench_geqrf_omp_simd PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_features(bench_geqrf_omp_simd PRIVATE cxx_std_17)
target_compile_options(bench_geqrf_omp_simd PRIVATE ${CQR_OMP_SIMD_FLAGS})

# ---------------------------------------------------------------------------
# Intel MKL compact extension: cqr_mkl_dormqr_compact (design doc section 2)
# ---------------------------------------------------------------------------
if(CQR_WITH_MKL)
  find_package(MKLCompact REQUIRED)

  add_library(cqr_mkl_ext
    src/cqr_mkl_ext.cpp
    src/cqr_mkl_geqrf.cpp)
  target_include_directories(cqr_mkl_ext
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_compile_features(cqr_mkl_ext PUBLIC cxx_std_17)
  target_link_libraries(cqr_mkl_ext
    PUBLIC  MKL::CompactHeaders   # MKL types in the public header (no link dep)
    PRIVATE cqr_compact)
  add_library(cqr::mkl_ext ALIAS cqr_mkl_ext)
endif()

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
# Worked example of the Compact-format batch QR solver
# (mkl_dgeqrf_compact -> cqr_mkl_dormqr_compact -> mkl_dtrsm_compact),
# cross-checked against the naive per-matrix LAPACKE_dgels. Needs Intel MKL
# (the compact API) and this repo's cqr_mkl_ext.
if(CQR_WITH_MKL)
  add_executable(solve_qr_compact examples/solve_qr_compact.cpp)
  target_link_libraries(solve_qr_compact
    PRIVATE cqr_mkl_ext MKL::Compact)

  # Throughput benchmark: compact batched vs per-matrix LAPACK QR solve, with
  # the outer loop over the matrix pool parallelised by OpenMP when available.
  add_executable(bench_qr_compact examples/bench_qr_compact.cpp)
  target_link_libraries(bench_qr_compact
    PRIVATE cqr_mkl_ext MKL::Compact)
  find_package(OpenMP QUIET)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(bench_qr_compact PRIVATE OpenMP::OpenMP_CXX)
  endif()

  # Throughput benchmark of the QR factorization itself: the GNU vector-types and
  # both OpenMP-SIMD compact kernels vs mkl_dgeqrf_compact and per-matrix
  # LAPACKE_dgeqrf (design doc benchmark 1). The omp-simd flag makes its
  # `#pragma omp simd` kernels vectorize (via Intel's codegen under icpx).
  add_executable(bench_geqrf_compact examples/bench_geqrf_compact.cpp)
  target_link_libraries(bench_geqrf_compact
    PRIVATE cqr_mkl_ext MKL::Compact)
  target_compile_options(bench_geqrf_compact PRIVATE ${CQR_OMP_SIMD_FLAGS})
  if(OpenMP_CXX_FOUND)
    target_link_libraries(bench_geqrf_compact PRIVATE OpenMP::OpenMP_CXX)
  endif()
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(CQR_BUILD_TESTS)
  enable_testing()

  # Portable self-contained kernel test (no BLAS).
  add_executable(test_cqr_compact src/test_cqr_compact.cpp)
  target_link_libraries(test_cqr_compact PRIVATE cqr_compact)
  add_test(NAME portable_kernel COMMAND test_cqr_compact)

  # Portable self-contained QR factorization test (no BLAS): the compact geqrf
  # kernel vs a scalar geqr2/larfg reference, plus C API validation.
  add_executable(test_cqr_geqrf_compact src/test_cqr_geqrf_compact.cpp)
  target_link_libraries(test_cqr_geqrf_compact PRIVATE cqr_compact)
  add_test(NAME portable_geqrf COMMAND test_cqr_geqrf_compact)

  # The same geqrf test source, linked against the omp-simd backend instead:
  # a literal drop-in swap of the dgeqrf_compact / sgeqrf_compact C entry
  # points, which must pass identically to the vector-types build above.
  add_executable(test_cqr_geqrf_compact_omp src/test_cqr_geqrf_compact.cpp)
  target_link_libraries(test_cqr_geqrf_compact_omp PRIVATE cqr_compact_ompsimd)
  add_test(NAME portable_geqrf_ompsimd COMMAND test_cqr_geqrf_compact_omp)

  # The head-to-head benchmark carries a correctness gate (both kernels vs a
  # scalar geqr2 reference), so a small quick run doubles as an integration
  # test. 33 exercises a padded final group.
  add_test(NAME bench_geqrf_omp_simd_integration COMMAND bench_geqrf_omp_simd 33 1)
  set_tests_properties(bench_geqrf_omp_simd_integration PROPERTIES LABELS "integration")

  # MKL-backed validation of cqr_mkl_dormqr_compact (design doc section 7).
  if(CQR_WITH_MKL)
    add_executable(test_cqr_mkl_ext src/test_cqr_mkl_ext.cpp)
    target_link_libraries(test_cqr_mkl_ext
      PRIVATE cqr_mkl_ext MKL::Compact)
    add_test(NAME mkl_compact_suites COMMAND test_cqr_mkl_ext)

    # MKL + dense LAPACK validation of cqr_mkl_dgeqrf_compact (design doc
    # section 7): factorization invariants, cross-check vs mkl_dgeqrf_compact,
    # and the end-to-end solve. Needs cqr_compact for the ormqr apply step.
    add_executable(test_cqr_geqrf_mkl src/test_cqr_geqrf_mkl.cpp)
    target_link_libraries(test_cqr_geqrf_mkl
      PRIVATE cqr_mkl_ext cqr_compact MKL::Compact)
    add_test(NAME mkl_geqrf_suites COMMAND test_cqr_geqrf_mkl)

    # The compact QR-solve example self-validates, so register it too.
    add_test(NAME example_solve_qr_compact COMMAND solve_qr_compact)

    # The benchmark carries a correctness gate (both paths must recover the
    # known solution), so it doubles as an end-to-end integration test of the
    # compact pipeline against per-matrix LAPACK. Run a small, quick pool
    # (few matrices, a single rep) to keep CTest fast. 33 is not a multiple of
    # the SIMD vector length, so the partial last group is exercised too.
    add_test(NAME bench_qr_compact_integration COMMAND bench_qr_compact 33 1)
    set_tests_properties(bench_qr_compact_integration PROPERTIES LABELS "integration")

    # The factorization benchmark is accuracy-gated against per-matrix LAPACK,
    # so register a small, quick run as an integration test too. 33 exercises a
    # padded final group.
    add_test(NAME bench_geqrf_compact_integration COMMAND bench_geqrf_compact 33 1)
    set_tests_properties(bench_geqrf_compact_integration PROPERTIES LABELS "integration")
  endif()
endif()
