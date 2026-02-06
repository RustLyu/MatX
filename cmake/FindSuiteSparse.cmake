# Minimal SuiteSparse + KLU finder (aggregate imported target).
#
# This is intentionally lightweight; for a full integration you may want
# SuiteSparse's own CMake config packages or vcpkg.
#
# Variables:
# - SuiteSparse_FOUND
#
# Imported target:
# - SuiteSparse::SuiteSparse

include(FindPackageHandleStandardArgs)

set(_SS_ROOT_HINTS "")
if(DEFINED SuiteSparse_ROOT)
  list(APPEND _SS_ROOT_HINTS "${SuiteSparse_ROOT}")
endif()
if(DEFINED SUITESPARSE_ROOT)
  list(APPEND _SS_ROOT_HINTS "${SUITESPARSE_ROOT}")
endif()

# Headers: SuiteSparse_config.h / klu.h are usually present.
find_path(SuiteSparse_INCLUDE_DIR
  NAMES SuiteSparse_config.h klu.h
  HINTS ${_SS_ROOT_HINTS}
  PATH_SUFFIXES include suitesparse include/suitesparse
)

# Core libraries we care about for KLU:
find_library(SuiteSparse_KLU_LIBRARY
  NAMES klu libklu
  HINTS ${_SS_ROOT_HINTS}
  PATH_SUFFIXES lib lib64
)
find_library(SuiteSparse_AMD_LIBRARY
  NAMES amd libamd
  HINTS ${_SS_ROOT_HINTS}
  PATH_SUFFIXES lib lib64
)
find_library(SuiteSparse_COLAMD_LIBRARY
  NAMES colamd libcolamd
  HINTS ${_SS_ROOT_HINTS}
  PATH_SUFFIXES lib lib64
)
find_library(SuiteSparse_SUITESPARSECONFIG_LIBRARY
  NAMES suitesparseconfig libsuitesparseconfig SuiteSparse_config
  HINTS ${_SS_ROOT_HINTS}
  PATH_SUFFIXES lib lib64
)
# BTF is optional but commonly linked with KLU.
find_library(SuiteSparse_BTF_LIBRARY
  NAMES btf libbtf
  HINTS ${_SS_ROOT_HINTS}
  PATH_SUFFIXES lib lib64
)

find_package_handle_standard_args(SuiteSparse
  REQUIRED_VARS
    SuiteSparse_INCLUDE_DIR
    SuiteSparse_KLU_LIBRARY
    SuiteSparse_AMD_LIBRARY
    SuiteSparse_COLAMD_LIBRARY
    SuiteSparse_SUITESPARSECONFIG_LIBRARY
)

if(SuiteSparse_FOUND)
  if(NOT TARGET SuiteSparse::SuiteSparse)
    add_library(SuiteSparse::SuiteSparse INTERFACE IMPORTED)
    target_include_directories(SuiteSparse::SuiteSparse INTERFACE "${SuiteSparse_INCLUDE_DIR}")
    target_link_libraries(SuiteSparse::SuiteSparse INTERFACE
      "${SuiteSparse_KLU_LIBRARY}"
      "${SuiteSparse_AMD_LIBRARY}"
      "${SuiteSparse_COLAMD_LIBRARY}"
      "${SuiteSparse_SUITESPARSECONFIG_LIBRARY}"
    )
    if(SuiteSparse_BTF_LIBRARY)
      target_link_libraries(SuiteSparse::SuiteSparse INTERFACE "${SuiteSparse_BTF_LIBRARY}")
    endif()
  endif()
endif()

