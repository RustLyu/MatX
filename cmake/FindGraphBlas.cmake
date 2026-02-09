# Find GraphBlas (SuiteSparse component) for sparse matrix ops.
# Variables: GraphBlas_FOUND, GraphBlas_INCLUDE_DIR, GraphBlas_LIBRARY
# Target: GraphBlas::GraphBlas

include(FindPackageHandleStandardArgs)

set(_GB_ROOT "")
if(DEFINED GraphBlas_ROOT)
  list(APPEND _GB_ROOT "${GraphBlas_ROOT}")
endif()
if(DEFINED SuiteSparse_ROOT)
  list(APPEND _GB_ROOT "${SuiteSparse_ROOT}")
endif()
if(DEFINED SUITESPARSE_ROOT)
  list(APPEND _GB_ROOT "${SUITESPARSE_ROOT}")
endif()

find_path(GraphBlas_INCLUDE_DIR
  NAMES GraphBlas.h
  HINTS ${_GB_ROOT}
  PATH_SUFFIXES include include/suitesparse
)

find_library(GraphBlas_LIBRARY
  NAMES graphblas libgraphblas
  HINTS ${_GB_ROOT}
  PATH_SUFFIXES lib lib64
)

find_package_handle_standard_args(GraphBlas
  REQUIRED_VARS GraphBlas_LIBRARY GraphBlas_INCLUDE_DIR
)

if(GraphBlas_FOUND)
  if(NOT TARGET GraphBlas::GraphBlas)
    add_library(GraphBlas::GraphBlas UNKNOWN IMPORTED)
    set_target_properties(GraphBlas::GraphBlas PROPERTIES
      IMPORTED_LOCATION "${GraphBlas_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${GraphBlas_INCLUDE_DIR}"
    )
  endif()
endif()
