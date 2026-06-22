cmake_minimum_required(VERSION 3.18)

project(cqr
  VERSION 0.1.0
  DESCRIPTION "Compact-format apply-Q (ext_mkl_dormqr_compact) and its kernel"
  LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(CQR_WITH_MKL        "Build the Intel MKL compact extension + MKL tests" ON)
option(CQR_BUILD_TESTS     "Build tests and register them with CTest"          ON)
option(CQR_BUILD_XCHECK    "Build the ArmPL interleave-batch cross-check"      ON)
option(CQR_ENABLE_NATIVE   "Compile for the host CPU (-march/-mcpu=native)"    OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)   # ormqr_compact.hpp uses GNU vector_size types

# Host-tuned codegen for the SIMD kernel (opt-in; correctness is independent).
add_library(cqr_flags INTERFACE)
if(CQR_ENABLE_NATIVE)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-march=native" CQR_HAS_MARCH_NATIVE)
  if(CQR_HAS_MARCH_NATIVE)
    target_compile_options(cqr_flags INTERFACE -march=native)
  else()
    check_cxx_compiler_flag("-mcpu=native" CQR_HAS_MCPU_NATIVE)
    if(CQR_HAS_MCPU_NATIVE)
      target_compile_options(cqr_flags INTERFACE -mcpu=native)
    endif()
  endif()
endif()

# ---------------------------------------------------------------------------
# Core kernel library (portable; no BLAS dependency)
# ---------------------------------------------------------------------------
add_library(ormqr_compact src/ormqr_compact_dispatch.cpp)
target_include_directories(ormqr_compact PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(ormqr_compact PRIVATE cqr_flags)
add_library(cqr::ormqr_compact ALIAS ormqr_compact)

# ---------------------------------------------------------------------------
# Intel MKL compact extension: ext_mkl_dormqr_compact (design doc section 2)
# ---------------------------------------------------------------------------
if(CQR_WITH_MKL)
  find_package(MKLCompact REQUIRED)

  add_library(ext_mkl_ormqr_compact src/ext_mkl_ormqr_compact.cpp)
  target_include_directories(ext_mkl_ormqr_compact
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(ext_mkl_ormqr_compact
    PUBLIC  MKL::Compact          # MKL types in the public header
    PRIVATE ormqr_compact cqr_flags)
  add_library(cqr::ext_mkl_ormqr_compact ALIAS ext_mkl_ormqr_compact)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(CQR_BUILD_TESTS)
  enable_testing()

  # Portable self-contained kernel test (no BLAS).
  add_executable(test_ormqr_compact src/test_ormqr_compact.cpp)
  target_link_libraries(test_ormqr_compact PRIVATE ormqr_compact cqr_flags)
  add_test(NAME portable_kernel COMMAND test_ormqr_compact)

  # MKL-backed validation of ext_mkl_dormqr_compact (design doc section 7).
  if(CQR_WITH_MKL)
    add_executable(test_ext_mkl_ormqr_compact src/test_ext_mkl_ormqr_compact.cpp)
    target_link_libraries(test_ext_mkl_ormqr_compact
      PRIVATE ext_mkl_ormqr_compact MKL::Compact cqr_flags)
    add_test(NAME mkl_compact_suites COMMAND test_ext_mkl_ormqr_compact)
  endif()

  # ArmPL interleave-batch cross-check (real ArmPL if present, else the
  # bundled functional stub so the driver logic is still exercised).
  if(CQR_BUILD_XCHECK)
    find_path(ARMPL_INCLUDE_DIR armpl.h
      HINTS $ENV{ARMPL_DIR}/include /opt/arm/*/include)
    find_library(ARMPL_LIBRARY NAMES armpl_lp64 armpl
      HINTS $ENV{ARMPL_DIR}/lib /opt/arm/*/lib)

    add_executable(xcheck_armpl_ormqr src/xcheck_armpl_ormqr.cpp)
    target_link_libraries(xcheck_armpl_ormqr PRIVATE ormqr_compact cqr_flags)

    if(ARMPL_INCLUDE_DIR AND ARMPL_LIBRARY)
      message(STATUS "xcheck: using real ArmPL (${ARMPL_LIBRARY})")
      target_include_directories(xcheck_armpl_ormqr PRIVATE "${ARMPL_INCLUDE_DIR}")
      target_link_libraries(xcheck_armpl_ormqr PRIVATE "${ARMPL_LIBRARY}" m)
    else()
      message(STATUS "xcheck: ArmPL not found, using bundled stub headers "
                     "(logic-only cross-check)")
      target_include_directories(xcheck_armpl_ormqr
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/armpl_stub")
      target_link_libraries(xcheck_armpl_ormqr PRIVATE m)
    endif()
    add_test(NAME xcheck_armpl COMMAND xcheck_armpl_ormqr)
  endif()
endif()
