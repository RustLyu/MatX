# Minimal OpenBLAS finder for Windows/Linux.
#
# Variables:
# - OpenBLAS_FOUND
# - OpenBLAS_INCLUDE_DIRS
# - OpenBLAS_LIBRARIES
#
# Imported target:
# - OpenBLAS::OpenBLAS

include(FindPackageHandleStandardArgs)

set(_OPENBLAS_ROOT_HINTS "")
if(DEFINED OpenBLAS_ROOT)
  list(APPEND _OPENBLAS_ROOT_HINTS "${OpenBLAS_ROOT}")
endif()
if(DEFINED OPENBLAS_ROOT)
  list(APPEND _OPENBLAS_ROOT_HINTS "${OPENBLAS_ROOT}")
endif()

find_path(OpenBLAS_INCLUDE_DIR
  NAMES cblas.h
  HINTS ${_OPENBLAS_ROOT_HINTS}
  PATH_SUFFIXES include
)

find_library(OpenBLAS_LIBRARY
  NAMES openblas openblas64_ openblas64 libopenblas
  HINTS ${_OPENBLAS_ROOT_HINTS}
  PATH_SUFFIXES lib lib64
)

find_package_handle_standard_args(OpenBLAS
  REQUIRED_VARS OpenBLAS_LIBRARY OpenBLAS_INCLUDE_DIR
)

if(OpenBLAS_FOUND)
  set(OpenBLAS_LIBRARIES ${OpenBLAS_LIBRARY})
  set(OpenBLAS_INCLUDE_DIRS ${OpenBLAS_INCLUDE_DIR})

  if(NOT TARGET OpenBLAS::OpenBLAS)
    add_library(OpenBLAS::OpenBLAS UNKNOWN IMPORTED)
    set_target_properties(OpenBLAS::OpenBLAS PROPERTIES
      IMPORTED_LOCATION "${OpenBLAS_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${OpenBLAS_INCLUDE_DIR}"
    )
  endif()
endif()

