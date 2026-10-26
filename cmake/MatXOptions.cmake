# =========================================================
# MatXOptions.cmake
# Build options: backend selection, feature toggles
# =========================================================

set(MATX_BACKEND "OPENBLAS" CACHE STRING "Backend: OPENBLAS | AMD_AOCL | AUTO")
set_property(CACHE MATX_BACKEND PROPERTY STRINGS OPENBLAS AMD_AOCL AUTO)

set(MATX_BACKEND_REQUESTED "${MATX_BACKEND}")
if(MATX_BACKEND STREQUAL "AUTO")
    if(CPU_PLATFORM STREQUAL "AMD")
        set(MATX_BACKEND "AMD_AOCL")
    else()
        set(MATX_BACKEND "OPENBLAS")
    endif()
endif()

if(NOT MATX_BACKEND STREQUAL "OPENBLAS" AND NOT MATX_BACKEND STREQUAL "AMD_AOCL")
    message(FATAL_ERROR "Unsupported MATX_BACKEND='${MATX_BACKEND}'. Choose OPENBLAS, AMD_AOCL, or AUTO.")
endif()
message(STATUS "requested backend: ${MATX_BACKEND_REQUESTED}; resolved backend: ${MATX_BACKEND}")
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
option(MATX_ENABLE_AGMG "Enable AGMG algebraic multigrid solver (manual download)" OFF)

# Optional iterative sparse solver libraries. They are discovered only when
# explicitly enabled so the existing direct-solver build does not gain a new
# mandatory dependency.
option(MATX_ENABLE_GINKGO "Enable the Ginkgo iterative sparse solver backend" OFF)
option(MATX_ENABLE_PETSC "Enable the PETSc KSP iterative sparse solver backend" OFF)
option(MATX_ENABLE_HYPRE "Enable the hypre iterative sparse solver backend" OFF)
option(MATX_ENABLE_PETSC_HYPRE "Enable hypre BoomerAMG as a PETSc preconditioner (requires PETSc built with hypre)" OFF)

# Source builds are the default for the optional iterative dependencies. Set a
# corresponding option to OFF only when using an existing upstream install.
option(MATX_BUILD_GINKGO_FROM_SOURCE "Build Ginkgo from its upstream source" ON)
option(MATX_BUILD_PETSC_FROM_SOURCE "Build PETSc from its upstream source" ON)
option(MATX_BUILD_HYPRE_FROM_SOURCE "Build hypre from its upstream source" ON)

if(MATX_ENABLE_PETSC_HYPRE)
    set(MATX_ENABLE_PETSC ON)
endif()

if(NOT MATX_ENABLE_SUITESPARSE)
    if(MATX_ENABLE_UMFPACK OR MATX_ENABLE_CXSPARSE OR MATX_ENABLE_GRAPHBLAS)
        message(STATUS "Disabling UMFPACK, CXSparse, and GraphBLAS because MATX_ENABLE_SUITESPARSE=OFF")
    endif()
    set(MATX_ENABLE_UMFPACK OFF)
    set(MATX_ENABLE_CXSPARSE OFF)
    set(MATX_ENABLE_GRAPHBLAS OFF)
endif()

option(BUILD_SHARED_LIBS "Build shared library (ON) or static (OFF)" OFF)

set(MATX_DENSE_BLAS_BACKEND "AUTO" CACHE STRING "DENSE BLAS backend: AUTO|OPENBLAS|BLIS|REFERENCE")
set(MATX_SPARSE_BLAS_BACKEND "AUTO" CACHE STRING "SPARSE BLAS backend: AUTO|OPENBLAS|BLIS|REFERENCE")

option(MATX_BUILD_TESTS "Enable test project if found" ON)
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
endif()
