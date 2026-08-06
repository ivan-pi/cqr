# Standalone build for the report's benchmark executables.
#
# This is a SEPARATE CMake project, intentionally NOT part of the main cqr
# build. It reuses the repo's FindMKLCompact module and the one cqr kernel
# source the collector needs, and builds only the report artefacts.
#
# Configure/build (needs Intel MKL and Google Benchmark):
#
#   cmake -S report/bench -B report/bench/build \
#         -DBLA_VENDOR=Intel10_64lp_seq -DCMAKE_BUILD_TYPE=Release \
#         -DCMAKE_CXX_FLAGS="-O3 -march=native"
#   cmake --build report/bench/build -j
#
# -march=native lets the portable SIMD kernel use the host's widest vectors,
# the same width MKL selects at runtime.

cmake_minimum_required(VERSION 3.18)
project(cqr_report_bench
  DESCRIPTION "Benchmark executables for the cqr report (separate build)"
  LANGUAGES CXX)

# Repo root is two levels up; borrow its FindMKLCompact module.
get_filename_component(CQR_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
list(APPEND CMAKE_MODULE_PATH "${CQR_ROOT}/cmake")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(MKLCompact REQUIRED)     # MKL::Compact (honours BLA_VENDOR)
find_package(benchmark REQUIRED)      # Google Benchmark
find_package(OpenMP QUIET)

# Google Benchmark collection driver. cqr_mkl_dgeqrf_compact is defined in
# cqr_mkl_geqrf.cpp, which instantiates the header-only kernel directly, so that
# single source (plus its headers under src/) is all the collector needs from
# the main tree. This mirrors the geqrf subset of the root `cqr_mkl_ext` target
# (../../CMakeLists.txt): if that library gains sources the kernel depends on,
# mirror them here.
add_executable(bench_geqrf_collect
  bench_geqrf_collect.cpp
  "${CQR_ROOT}/src/cqr_mkl_geqrf.cpp")
target_include_directories(bench_geqrf_collect PRIVATE "${CQR_ROOT}/src")
target_compile_features(bench_geqrf_collect PRIVATE cxx_std_17)
target_link_libraries(bench_geqrf_collect
  PRIVATE MKL::Compact benchmark::benchmark)
if(OpenMP_CXX_FOUND)
  target_link_libraries(bench_geqrf_collect PRIVATE OpenMP::OpenMP_CXX)
endif()
