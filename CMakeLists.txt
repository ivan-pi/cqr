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
option(CQR_WITH_SYCL       "Build the experimental SYCL (oneAPI DPC++) GPU prototype" OFF)

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

  # Throughput benchmark of the QR factorization itself: cqr_mkl_dgeqrf_compact
  # vs mkl_dgeqrf_compact vs per-matrix LAPACKE_dgeqrf (design doc benchmark 1).
  add_executable(bench_geqrf_compact examples/bench_geqrf_compact.cpp)
  target_link_libraries(bench_geqrf_compact
    PRIVATE cqr_mkl_ext MKL::Compact)
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

# ---------------------------------------------------------------------------
# SYCL (Intel oneAPI DPC++) GPU prototype -- design doc Phase 1.
#
# Optional and OFF by default. Re-implements the four portable C entry points on
# a SYCL device (Strategy A: one work-item per matrix, in gpu/sycl/) and
# validates them with the project's own suites: the UNCHANGED geqrf test
# (src/test_cqr_geqrf_compact.cpp -- it drives both kernels through the C API)
# and a C-API-routed ormqr test (gpu/sycl/test_ormqr_sycl.cpp). Requires a SYCL
# compiler (configure with CXX=icpx); needs neither MKL nor a GPU -- it runs on
# the OpenCL CPU device as well. See gpu/sycl/README.md and
# scripts/install-sycl-toolchain.sh.
# ---------------------------------------------------------------------------
if(CQR_WITH_SYCL)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-fsycl" CQR_HAS_FSYCL)
  if(NOT CQR_HAS_FSYCL)
    message(FATAL_ERROR
      "CQR_WITH_SYCL=ON requires a SYCL compiler. Configure with CXX=icpx "
      "(Intel oneAPI DPC++); see gpu/sycl/README.md and "
      "scripts/install-sycl-toolchain.sh.")
  endif()

  # When built inside a conda (conda-forge dpcpp) environment, bake the SYCL
  # runtime path into the binaries so they run without a manual LD_LIBRARY_PATH.
  set(_cqr_sycl_rpath "")
  if(DEFINED ENV{CONDA_PREFIX})
    set(_cqr_sycl_rpath "$ENV{CONDA_PREFIX}/lib")
  endif()

  # The backend source is compiled directly into each test so the four C symbols
  # resolve to the SYCL versions (not the CPU dispatch in cqr_compact).
  add_executable(test_geqrf_sycl
    src/test_cqr_geqrf_compact.cpp gpu/sycl/cqr_compact_sycl.cpp)
  add_executable(test_ormqr_sycl
    gpu/sycl/test_ormqr_sycl.cpp gpu/sycl/cqr_compact_sycl.cpp)
  foreach(tgt test_geqrf_sycl test_ormqr_sycl)
    target_include_directories(${tgt} PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}/gpu/sycl")
    target_compile_features(${tgt} PRIVATE cxx_std_17)
    target_compile_options(${tgt} PRIVATE -fsycl)
    target_link_options(${tgt} PRIVATE -fsycl)
    if(_cqr_sycl_rpath)
      target_link_options(${tgt} PRIVATE "-Wl,-rpath,${_cqr_sycl_rpath}")
    endif()
    if(CQR_BUILD_TESTS)
      add_test(NAME ${tgt} COMMAND ${tgt})
    endif()
  endforeach()

  # Fused-pattern example: ONE user-owned parallel_for composing the device
  # primitives (fill -> geqrf -> apply Q^T -> trsm -> unpack) in a single kernel.
  # Self-checking against a known solution, so it is also registered as a test
  # (exit 77 -> skipped when the device lacks the required sub-group size).
  add_executable(example_fused_qr_solve gpu/sycl/example_fused_qr_solve.cpp)
  target_include_directories(example_fused_qr_solve PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/gpu/sycl")
  target_compile_features(example_fused_qr_solve PRIVATE cxx_std_17)
  target_compile_options(example_fused_qr_solve PRIVATE -fsycl)
  target_link_options(example_fused_qr_solve PRIVATE -fsycl)
  if(_cqr_sycl_rpath)
    target_link_options(example_fused_qr_solve PRIVATE "-Wl,-rpath,${_cqr_sycl_rpath}")
  endif()
  if(CQR_BUILD_TESTS)
    add_test(NAME example_fused_qr_solve COMMAND example_fused_qr_solve)
    set_tests_properties(example_fused_qr_solve PROPERTIES SKIP_RETURN_CODE 77)
  endif()

  # Full-workflow benchmark: the fused single-kernel solve vs running geqrf /
  # apply-Q^T / trsm as separate operations (with and without per-call copies).
  # Pure SYCL (no OpenMP); self-checking.
  add_executable(bench_fused_solve gpu/sycl/bench_fused_solve.cpp)
  target_include_directories(bench_fused_solve PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/gpu/sycl")
  target_compile_features(bench_fused_solve PRIVATE cxx_std_17)
  target_compile_options(bench_fused_solve PRIVATE -fsycl)
  target_link_options(bench_fused_solve PRIVATE -fsycl)
  if(_cqr_sycl_rpath)
    target_link_options(bench_fused_solve PRIVATE "-Wl,-rpath,${_cqr_sycl_rpath}")
  endif()

  # Optional throughput benchmark: SYCL vs the vector-types ormqr kernel (needs
  # OpenMP for the multi-core vec path). CPU-vs-CPU on the OpenCL CPU device when
  # no GPU is present; self-checking (non-zero exit on numeric disagreement).
  find_package(OpenMP QUIET)
  if(OpenMP_CXX_FOUND)
    add_executable(bench_ormqr_sycl_vs_vec gpu/sycl/bench_ormqr_sycl_vs_vec.cpp)
    target_include_directories(bench_ormqr_sycl_vs_vec PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src")
    if(DEFINED ENV{CONDA_PREFIX})
      target_include_directories(bench_ormqr_sycl_vs_vec PRIVATE "$ENV{CONDA_PREFIX}/include")
    endif()
    target_compile_features(bench_ormqr_sycl_vs_vec PRIVATE cxx_std_17)
    target_compile_options(bench_ormqr_sycl_vs_vec PRIVATE -fsycl)
    # Fairness: the vector-types kernel is host code and must target the host's
    # widest SIMD, or it is compared against the SYCL JIT (which uses AVX-512
    # regardless) with one hand tied. The SYCL device code is unaffected by these
    # host flags. Force -O3 -march=native on THIS benchmark target (the library
    # itself still leaves -march to the caller via CMAKE_CXX_FLAGS).
    check_cxx_compiler_flag("-march=native" CQR_BENCH_HAS_MARCH_NATIVE)
    if(CQR_BENCH_HAS_MARCH_NATIVE)
      target_compile_options(bench_ormqr_sycl_vs_vec PRIVATE -O3 -march=native)
    endif()
    target_link_options(bench_ormqr_sycl_vs_vec PRIVATE -fsycl)
    if(_cqr_sycl_rpath)
      target_link_options(bench_ormqr_sycl_vs_vec PRIVATE "-Wl,-rpath,${_cqr_sycl_rpath}")
    endif()
    target_link_libraries(bench_ormqr_sycl_vs_vec PRIVATE OpenMP::OpenMP_CXX)
  endif()
endif()
