# Find CXSparse (SuiteSparse component) for sparse matrix ops.
# Variables: CXSparse_FOUND, CXSparse_INCLUDE_DIR, CXSparse_LIBRARY
# Target: CXSparse::CXSparse

include(FindPackageHandleStandardArgs)

set(_CX_ROOT "")
if(DEFINED CXSparse_ROOT)
  list(APPEND _CX_ROOT "${CXSparse_ROOT}")
endif()
if(DEFINED SuiteSparse_ROOT)
  list(APPEND _CX_ROOT "${SuiteSparse_ROOT}")
endif()
if(DEFINED SUITESPARSE_ROOT)
  list(APPEND _CX_ROOT "${SUITESPARSE_ROOT}")
endif()

find_path(CXSparse_INCLUDE_DIR
  NAMES cs.h
  HINTS ${_CX_ROOT}
  PATH_SUFFIXES include include/cxsparse include/suitesparse
)

find_library(CXSparse_LIBRARY
  NAMES cxsparse libcxsparse cs libcs
  HINTS ${_CX_ROOT}
  PATH_SUFFIXES lib lib64
)

find_package_handle_standard_args(CXSparse
  REQUIRED_VARS CXSparse_LIBRARY CXSparse_INCLUDE_DIR
)

if(CXSparse_FOUND)
  if(NOT TARGET CXSparse::CXSparse)
    add_library(CXSparse::CXSparse UNKNOWN IMPORTED)
    set_target_properties(CXSparse::CXSparse PROPERTIES
      IMPORTED_LOCATION "${CXSparse_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${CXSparse_INCLUDE_DIR}"
    )
  endif()
endif()
