# =========================================================
# MatXOptions.cmake
# Build options: backend selection, feature toggles
# =========================================================

set(MATX_BACKEND "OPENBLAS" CACHE STRING "Backend: OPENBLAS | AMD_AOCL | AUTO")
set_property(CACHE MATX_BACKEND PROPERTY STRINGS OPENBLAS AMD_AOCL AUTO)

message(STATUS "current backend:${MATX_BACKEND}")
if (MATX_BACKEND STREQUAL "OPENBLAS")
    set(MATX_ENABLE_OPENBLAS ON)
    set(MATX_ENABLE_BLIS OFF)
    set(MATX_ENABLE_LIBFLAME OFF)
    set(MATX_ENABLE_AOCL_SPARSE OFF)
elseif (MATX_BACKEND STREQUAL "AMD_AOCL")
    set(MATX_ENABLE_OPENBLAS OFF)
    set(MATX_ENABLE_BLIS ON)
    set(MATX_ENABLE_LIBFLAME ON)
    set(MATX_ENABLE_AOCL_SPARSE ON)
endif()

option(MATX_ENABLE_SUITESPARSE "Enable SuiteSparse backend if found/provided" ON)
option(MATX_ENABLE_UMFPACK "Enable SuiteSparse UMFPACK sparse solver backend if found/provided" ON)
option(MATX_ENABLE_CXSPARSE "Enable SuiteSparse CXSparse sparse solver backend if found/provided" ON)
option(MATX_ENABLE_SUPERLU "Enable SuperLU sparse solver backend (auto-fetch)" ON)
option(MATX_ENABLE_MUMPS "Enable MUMPS sparse solver backend if found/provided" ON)
option(MATX_ENABLE_GRAPHBLAS "Enable GraphBlas for sparse numerical ops if found" ON)

option(BUILD_SHARED_LIBS "Build shared library (ON) or static (OFF)" OFF)

set(MATX_DENSE_BLAS_BACKEND "AUTO" CACHE STRING "DENSE BLAS backend: AUTO|OPENBLAS|BLIS|REFERENCE")
set(MATX_SPARSE_BLAS_BACKEND "AUTO" CACHE STRING "SPARSE BLAS backend: AUTO|OPENBLAS|BLIS|REFERENCE")

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
endif()
