# =========================================================
# MatXBackendValidate.cmake
# BLAS/LAPACK backend validation + install prefix setup
# =========================================================

set(BLAS_LIST "")

if (MATX_ENABLE_OPENBLAS)
    list(APPEND BLAS_LIST "OpenBLAS")
endif()

if (MATX_ENABLE_BLIS)
    list(APPEND BLAS_LIST "BLIS")
endif()

list(LENGTH BLAS_LIST BLAS_COUNT)

if (BLAS_COUNT EQUAL 0)
    message(FATAL_ERROR
        "No BLAS backend enabled. Enable one of: OpenBLAS, BLIS, MKL.")
elseif (BLAS_COUNT GREATER 1)
    message(FATAL_ERROR
        "Multiple BLAS backends enabled (${BLAS_LIST}). Only one BLAS backend is allowed.")
endif()

if (MATX_BACKEND STREQUAL "OPENBLAS")
    set(DEPEND_LIB_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/deps_output_openblas")
elseif (MATX_BACKEND STREQUAL "AMD_AOCL")
    set(DEPEND_LIB_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/deps_output_amd_aocl")
endif()

set(AOCL_ROOT ${DEPEND_LIB_OUTPUT}/aocl_root)
