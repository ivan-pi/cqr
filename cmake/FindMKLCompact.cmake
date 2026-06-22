# FindMKLCompact.cmake
#
# Locate the Intel MKL Compact-format API and expose it as the imported target
#   MKL::Compact  (BLAS link libraries + the include dir with mkl_compact.h).
#
# The compact routines (mkl_get_format_compact, mkl_?geqrf_compact, ...) are an
# Intel MKL extension: they are reached through the ordinary BLAS link line, so
# this module selects the BLAS implementation with CMake's standard mechanism:
#
#   cmake -DBLA_VENDOR=Intel10_64lp_seq ...      (LP64, sequential MKL)
#
# After find_package(BLAS) it link-tests mkl_get_format_compact against the
# selected BLAS. Non-Intel BLAS implementations (OpenBLAS, reference, ...) do
# not provide the compact API, so the configure step stops with a FATAL_ERROR
# explaining how to select Intel MKL.

if(TARGET MKL::Compact)
  set(MKLCompact_FOUND TRUE)
  return()
endif()

# 1. Select the BLAS implementation (honours BLA_VENDOR).
find_package(BLAS REQUIRED)

if(TARGET BLAS::BLAS)
  set(_blas_link BLAS::BLAS)
else()
  set(_blas_link ${BLAS_LIBRARIES} ${BLAS_LINKER_FLAGS})
endif()

# 2. MKL headers (FindBLAS only resolves libraries, not the include dir).
find_path(MKLCompact_INCLUDE_DIR
  NAMES mkl_compact.h
  HINTS
    $ENV{MKLROOT}/include
    $ENV{MKLROOT}/include/mkl
  PATHS
    /usr/include/mkl
    /usr/include
    /opt/intel/oneapi/mkl/latest/include
)

# 3. Verify the selected BLAS actually provides the compact extension.
include(CheckCXXSourceCompiles)
set(_save_inc "${CMAKE_REQUIRED_INCLUDES}")
set(_save_lib "${CMAKE_REQUIRED_LIBRARIES}")
if(MKLCompact_INCLUDE_DIR)
  set(CMAKE_REQUIRED_INCLUDES "${MKLCompact_INCLUDE_DIR}")
endif()
set(CMAKE_REQUIRED_LIBRARIES ${_blas_link})
check_cxx_source_compiles("
#include <mkl_compact.h>
int main() { return (int) mkl_get_format_compact(); }
" MKLCompact_HAS_COMPACT_API)
set(CMAKE_REQUIRED_INCLUDES "${_save_inc}")
set(CMAKE_REQUIRED_LIBRARIES "${_save_lib}")

if(NOT MKLCompact_INCLUDE_DIR OR NOT MKLCompact_HAS_COMPACT_API)
  message(FATAL_ERROR
    "The compact-format extension (ext_mkl_dormqr_compact) requires Intel MKL, "
    "which is the only BLAS that provides the *_compact API "
    "(mkl_get_format_compact, mkl_dgeqrf_compact, ...).\n"
    "  BLA_VENDOR     = '${BLA_VENDOR}'\n"
    "  BLAS_LIBRARIES = '${BLAS_LIBRARIES}'\n"
    "  mkl_compact.h  = '${MKLCompact_INCLUDE_DIR}'\n"
    "The BLAS selected above does not provide it. Reconfigure with Intel MKL, "
    "e.g.  -DBLA_VENDOR=Intel10_64lp_seq , and make sure the MKL headers and "
    "libraries are installed.")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MKLCompact
  REQUIRED_VARS MKLCompact_INCLUDE_DIR MKLCompact_HAS_COMPACT_API)

if(MKLCompact_FOUND)
  find_package(Threads QUIET)
  add_library(MKL::Compact INTERFACE IMPORTED)
  set(_link ${_blas_link})
  if(Threads_FOUND)
    list(APPEND _link Threads::Threads)
  endif()
  if(CMAKE_DL_LIBS)
    list(APPEND _link ${CMAKE_DL_LIBS})
  endif()
  list(APPEND _link m)
  set_target_properties(MKL::Compact PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${MKLCompact_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "${_link}")
endif()

mark_as_advanced(MKLCompact_INCLUDE_DIR MKLCompact_HAS_COMPACT_API)
