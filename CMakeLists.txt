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

  # Bake the SYCL runtime path into the binaries so they run without a manual
  # LD_LIBRARY_PATH -- ctest and CI do not inherit an activated shell.
  #
  # Ask the driver where its own runtime lives rather than guessing from
  # CONDA_PREFIX. That variable is set only on the conda-forge route, so the
  # guess left every other layout (/opt/intel/oneapi via setvars.sh, a tarball,
  # a module file, a source build) with no rpath at all -- and the omission is
  # invisible until load time, as "libsycl.so.N: cannot open shared object
  # file". -print-file-name works on any DPC++ layout.
  set(_cqr_sycl_rpath "")
  execute_process(
    COMMAND "${CMAKE_CXX_COMPILER}" -print-file-name=libsycl.so
    OUTPUT_VARIABLE _cqr_libsycl OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET RESULT_VARIABLE _cqr_libsycl_rc)
  # A driver that cannot find the library echoes the bare name back.
  if(_cqr_libsycl_rc EQUAL 0 AND IS_ABSOLUTE "${_cqr_libsycl}")
    get_filename_component(_cqr_sycl_rpath "${_cqr_libsycl}" DIRECTORY)
  endif()

  # One carrier for the usage requirements every SYCL target needs. Spelling
  # them out per target invited exactly the drift they had developed: two
  # targets were missing an include directory they relied on, and the rpath is
  # silently optional -- forgetting it configures and builds cleanly, then
  # fails only when ctest runs outside an activated shell.
  add_library(cqr_sycl_opts INTERFACE)
  target_include_directories(cqr_sycl_opts INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}/gpu/sycl")
  target_compile_features(cqr_sycl_opts INTERFACE cxx_std_17)
  target_compile_options(cqr_sycl_opts INTERFACE -fsycl)
  target_link_options(cqr_sycl_opts INTERFACE -fsycl)
  if(_cqr_sycl_rpath)
    target_link_options(cqr_sycl_opts INTERFACE "-Wl,-rpath,${_cqr_sycl_rpath}")
  endif()

  # The backend source is compiled directly into each test so the four C symbols
  # resolve to the SYCL versions (not the CPU dispatch in cqr_compact).
  add_executable(test_geqrf_sycl
    src/test_cqr_geqrf_compact.cpp gpu/sycl/cqr_compact_sycl.cpp)
  add_executable(test_ormqr_sycl
    gpu/sycl/test_ormqr_sycl.cpp gpu/sycl/cqr_compact_sycl.cpp)
  foreach(tgt test_geqrf_sycl test_ormqr_sycl)
    target_link_libraries(${tgt} PRIVATE cqr_sycl_opts)
    if(CQR_BUILD_TESTS)
      add_test(NAME ${tgt} COMMAND ${tgt})
    endif()
  endforeach()

  # Variant 2: the SAME two suites, built against the EXPLICIT sub-group
  # primitives (CQR_SYCL_EXPLICIT_SG routes the launchers' sub-group path through
  # geqrf_slot_sg/ormqr_slot_sg), so they validate the group_load/store variant
  # verbatim at the same tolerances.
  add_executable(test_geqrf_sycl_sg
    src/test_cqr_geqrf_compact.cpp gpu/sycl/cqr_compact_sycl.cpp)
  add_executable(test_ormqr_sycl_sg
    gpu/sycl/test_ormqr_sycl.cpp gpu/sycl/cqr_compact_sycl.cpp)
  foreach(tgt test_geqrf_sycl_sg test_ormqr_sycl_sg)
    target_link_libraries(${tgt} PRIVATE cqr_sycl_opts)
    target_compile_definitions(${tgt} PRIVATE CQR_SYCL_EXPLICIT_SG)
    if(CQR_BUILD_TESTS)
      add_test(NAME ${tgt} COMMAND ${tgt})
    endif()
  endforeach()

  # Fused-pattern example: ONE user-owned parallel_for composing the device
  # primitives (fill -> geqrf -> apply Q^T -> trsm -> unpack) in a single kernel.
  # Self-checking against a known solution, so it is also registered as a test
  # (exit 77 -> skipped when the device lacks the required sub-group size).
  add_executable(example_fused_qr_solve gpu/sycl/example_fused_qr_solve.cpp)
  target_link_libraries(example_fused_qr_solve PRIVATE cqr_sycl_opts)
  if(CQR_BUILD_TESTS)
    add_test(NAME example_fused_qr_solve COMMAND example_fused_qr_solve)
    set_tests_properties(example_fused_qr_solve PROPERTIES SKIP_RETURN_CODE 77)
  endif()

  # Full-workflow benchmark: the fused single-kernel solve vs running geqrf /
  # apply-Q^T / trsm as separate operations (with and without per-call copies).
  # Pure SYCL (no OpenMP); self-checking.
  add_executable(bench_fused_solve gpu/sycl/bench_fused_solve.cpp)
  target_link_libraries(bench_fused_solve PRIVATE cqr_sycl_opts)

  # Optional throughput benchmark: SYCL vs the vector-types ormqr kernel (needs
  # OpenMP for the multi-core vec path). CPU-vs-CPU on the OpenCL CPU device when
  # no GPU is present; self-checking (non-zero exit on numeric disagreement).
  find_package(OpenMP QUIET)
  if(OpenMP_CXX_FOUND)
    add_executable(bench_ormqr_sycl_vs_vec gpu/sycl/bench_ormqr_sycl_vs_vec.cpp)
    target_link_libraries(bench_ormqr_sycl_vs_vec PRIVATE cqr_sycl_opts)
    if(DEFINED ENV{CONDA_PREFIX})
      target_include_directories(bench_ormqr_sycl_vs_vec PRIVATE "$ENV{CONDA_PREFIX}/include")
    endif()
    # Fairness: the vector-types kernel is host code and must target the host's
    # widest SIMD, or it is compared against the SYCL JIT (which uses AVX-512
    # regardless) with one hand tied. The SYCL device code is unaffected by these
    # host flags. Force -O3 -march=native on THIS benchmark target (the library
    # itself still leaves -march to the caller via CMAKE_CXX_FLAGS).
    check_cxx_compiler_flag("-march=native" CQR_BENCH_HAS_MARCH_NATIVE)
    if(CQR_BENCH_HAS_MARCH_NATIVE)
      target_compile_options(bench_ormqr_sycl_vs_vec PRIVATE -O3 -march=native)
    endif()
    target_link_libraries(bench_ormqr_sycl_vs_vec PRIVATE OpenMP::OpenMP_CXX)
  endif()
endif()
