include(CMakeFindDependencyMacro)

# ---- OpenBLAS ----
set(MATX_OPENBLAS_FOUND OFF)
if(MATX_ENABLE_OPENBLAS)
  find_package(OpenBLAS QUIET)
  if(OpenBLAS_FOUND)
    set(MATX_OPENBLAS_FOUND ON)
  endif()
endif()

# ---- BLIS ----
set(MATX_BLIS_FOUND OFF)
if(MATX_ENABLE_BLIS)
  find_package(BLIS QUIET)
  if(BLIS_FOUND)
    set(MATX_BLIS_FOUND ON)
  endif()
endif()

# ---- SuiteSparse ----
set(MATX_SUITESPARSE_FOUND OFF)
if(MATX_ENABLE_SUITESPARSE)
  find_package(SuiteSparse QUIET)
  if(SuiteSparse_FOUND)
    set(MATX_SUITESPARSE_FOUND ON)
  endif()
endif()

# ---- CXSparse (sparse numerical ops; optional) ----
set(MATX_CXSPARSE_FOUND OFF)
if(MATX_ENABLE_CXSPARSE)
  find_package(CXSparse QUIET)
  if(CXSparse_FOUND)
    set(MATX_CXSPARSE_FOUND ON)
  endif()
endif()

# ---- GraphBlas (sparse numerical ops; optional) ----
set(MATX_GRAPHBLAS_FOUND OFF)
if(MATX_ENABLE_GRAPHBLAS)
  find_package(GraphBlas QUIET)
  if(GraphBlas_FOUND)
    set(MATX_GRAPHBLAS_FOUND ON)
  endif()
endif()

# ---- LIBFLAME ----
set(MATX_LIBFLAME_FOUND OFF)
if(MATX_ENABLE_LIBFLAME)
  find_package(LIBFLAME QUIET)
  if(LIBFLAME_FOUND)
    set(MATX_LIBFLAME_FOUND ON)
  endif()
endif()

# ---- MKL ----
set(MATX_MKL_FOUND OFF)
if(MATX_ENABLE_MKL)
  find_package(MKL QUIET)
  if(MKL_FOUND)
    set(MATX_MKL_FOUND ON)
  endif()
endif()
