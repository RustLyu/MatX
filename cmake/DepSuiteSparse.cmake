# =========================================================
# DepSuiteSparse.cmake
# Fetch + build SuiteSparse (when MATX_ENABLE_SUITESPARSE=ON)
# =========================================================

if(MATX_ENABLE_SUITESPARSE)

    FetchContent_Declare(
        SuiteSparse
        GIT_REPOSITORY https://github.com/DrTimothyAldenDavis/SuiteSparse.git
        GIT_TAG v7.12.2
    )

    FetchContent_GetProperties(SuiteSparse)

    if(NOT SuiteSparse_POPULATED)
        FetchContent_Populate(SuiteSparse)
    endif()

    set(SUITESPARSE_BUILD_DIR
        ${CMAKE_BINARY_DIR}/suitesparse-build)

    file(MAKE_DIRECTORY
        ${SUITESPARSE_BUILD_DIR})

    set(OPENBLAS_INCLUDE_DIR
        ${OPENBLAS_BUILD_DIR}/include)

    message(STATUS "OPENBLAS:${BLAS_LIBRARIES}")
    message(STATUS "OPENBLAS:${LAPACK_LIBRARIES}")

    set(SUITESPARSE_PROJECTS "suitesparse_config;amd;btf;camd;ccolamd;colamd;cholmod;klu")
    if(MATX_ENABLE_UMFPACK)
        list(APPEND SUITESPARSE_PROJECTS umfpack)
    endif()
    if(MATX_ENABLE_CXSPARSE)
        list(APPEND SUITESPARSE_PROJECTS cxsparse)
    endif()
    if(MATX_ENABLE_GRAPHBLAS)
        list(APPEND SUITESPARSE_PROJECTS graphblas)
    endif()

if(MATX_BACKEND STREQUAL "OPENBLAS")
    execute_process(
        COMMAND
        ${CMAKE_COMMAND}
        -G ${CMAKE_GENERATOR}
        -S ${suitesparse_SOURCE_DIR}
        -B ${SUITESPARSE_BUILD_DIR}
        -DCMAKE_BUILD_TYPE=Release
        -DBLA_VENDOR=FLAME
        -DBLAS_LIBRARIES=${BLAS_LIBRARIES}
        -DLAPACK_LIBRARIES=${LAPACK_LIBRARIES}
        "-DSUITESPARSE_ENABLE_PROJECTS=${SUITESPARSE_PROJECTS}"
        -DSUITESPARSE_USE_64BIT_BLAS=ON
        -DCMAKE_INCLUDE_PATH=${OPENBLAS_INCLUDE_DIR}
        -DCMAKE_LIBRARY_PATH=${DEPEND_LIB_OUTPUT}/openblas/lib
        -DCMAKE_BUILD_RPATH=${DEPEND_LIB_OUTPUT}/openblas/lib
        -DCMAKE_INSTALL_RPATH=${DEPEND_LIB_OUTPUT}/openblas/lib
         -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath,${DEPS_OUTPUT_OPENBLAS}/openblas/lib"
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
        -DBUILD_SHARED_LIBS=ON
        -DBLAS64=ON
        -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON
        -DCMAKE_INSTALL_PREFIX=${DEPEND_LIB_OUTPUT}/suitesparse
        -DCMAKE_C_FLAGS=-fopenmp

        RESULT_VARIABLE SUITESPARSE_CONFIG_RESULT
    )
elseif(MATX_BACKEND STREQUAL "AMD_AOCL")
    execute_process(
        COMMAND
        ${CMAKE_COMMAND}
        -G ${CMAKE_GENERATOR}
        -S ${suitesparse_SOURCE_DIR}
        -B ${SUITESPARSE_BUILD_DIR}
        -DCMAKE_BUILD_TYPE=Release
        -DBLA_VENDOR=FLAME
        -DBLAS_LIBRARIES=${BLAS_LIBRARIES}\;${AOCL_ROOT}/lib/libaoclutils.so
        -DLAPACK_LIBRARIES=${LAPACK_LIBRARIES}
        "-DSUITESPARSE_ENABLE_PROJECTS=${SUITESPARSE_PROJECTS}"
        -DSUITESPARSE_USE_64BIT_BLAS=ON
        -DCMAKE_INCLUDE_PATH=${OPENBLAS_INCLUDE_DIR}
        -DCMAKE_LIBRARY_PATH=${OPENBLAS_LIB_DIR}\;${AOCL_ROOT}/lib
        -DCMAKE_INSTALL_RPATH=${AOCL_ROOT}/lib
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
        -DBUILD_SHARED_LIBS=ON
        -DCMAKE_INSTALL_PREFIX=${DEPEND_LIB_OUTPUT}/suitesparse

        RESULT_VARIABLE SUITESPARSE_CONFIG_RESULT
    )
endif()
if(NOT SUITESPARSE_CONFIG_RESULT EQUAL 0)

    message(FATAL_ERROR
        "SuiteSparse configure failed")
endif()
execute_process(

    COMMAND
    ${CMAKE_COMMAND}
    --build ${SUITESPARSE_BUILD_DIR}
    --parallel ${BUILD_JOBS}

    RESULT_VARIABLE SUITESPARSE_BUILD_RESULT
)
execute_process(
    COMMAND ${CMAKE_COMMAND}
    --install ${SUITESPARSE_BUILD_DIR}
    RESULT_VARIABLE SUITESPARSE_INSTALL_RESULT
)

if(NOT SUITESPARSE_BUILD_RESULT EQUAL 0)

    message(FATAL_ERROR
        "SuiteSparse build failed")

endif()

# Imported targets via find_package
set(SuiteSparse_config_DIR "${DEPEND_LIB_OUTPUT}/suitesparse/lib/cmake/SuiteSparse_config")
set(BTF_DIR "${DEPEND_LIB_OUTPUT}/suitesparse/lib/cmake/BTF")
set(AMD_DIR "${DEPEND_LIB_OUTPUT}/suitesparse/lib/cmake/AMD")
set(COLAMD_DIR "${DEPEND_LIB_OUTPUT}/suitesparse/lib/cmake/COLAMD")
set(CCOLAMD_DIR "${DEPEND_LIB_OUTPUT}/suitesparse/lib/cmake/CCOLAMD")
set(CHOLMOD_DIR "${DEPEND_LIB_OUTPUT}/suitesparse/lib/cmake/CHOLMOD")
set(CAMD_DIR "${DEPEND_LIB_OUTPUT}/suitesparse/lib/cmake/CAMD")
set(KLU_DIR "${DEPEND_LIB_OUTPUT}/suitesparse/lib/cmake/KLU")
if(MATX_ENABLE_GRAPHBLAS)
    set(GraphBLAS_DIR "${DEPEND_LIB_OUTPUT}/suitesparse/lib/cmake/GraphBLAS")
endif()
if(MATX_ENABLE_UMFPACK)
    set(UMFPACK_DIR "${DEPEND_LIB_OUTPUT}/suitesparse/lib/cmake/UMFPACK")
endif()
if(MATX_ENABLE_CXSPARSE)
    set(CXSparse_DIR "${DEPEND_LIB_OUTPUT}/suitesparse/lib/cmake/CXSparse")
endif()
find_package(SuiteSparse_config REQUIRED)
find_package(KLU REQUIRED)
if(MATX_ENABLE_GRAPHBLAS)
    find_package(GraphBLAS REQUIRED)
endif()
if(MATX_ENABLE_UMFPACK)
    find_package(UMFPACK REQUIRED)
endif()
if(MATX_ENABLE_CXSPARSE)
    find_package(CXSparse REQUIRED)
endif()
endif()
