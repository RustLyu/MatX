cmake_minimum_required(VERSION 3.20)

project(MatX LANGUAGES C CXX)


if (WIN32)
    add_definitions(-DOS_WINDOWS)
    message(STATUS "current os：Windows")
    #set(3Party_path "${CMAKE_CURRENT_SOURCE_DIR}/3party/win32")
elseif (LINUX)
    message(STATUS "current os：Linux")
    #set(3Party_path "${CMAKE_CURRENT_SOURCE_DIR}/3party/linux")
elseif (APPLE)
    message(STATUS "current os：macOS")
else ()
    message(WARNING "UNKNOWN OS！")
endif ()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    message(STATUS "ADD DEBUG MSG")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
    set(CMAKE_C_FLAGS_DEBUG "-O0 -g")
endif()

# =========================================================
# Build options
# =========================================================

option(MATX_BUILD_TESTS "Build tests" ON)

option(MATX_ENABLE_OPENBLAS "Enable OpenBLAS backend if found/provided" OFF)
option(MATX_ENABLE_BLIS "Enable BLIS backend if found/provided" OFF)
option(MATX_ENABLE_LIBFLAME "Enable LIBFLAME backend if found/provided" OFF)
option(MATX_ENABLE_SUITESPARSE "Enable SuiteSparse backend if found/provided" ON)
option(MATX_ENABLE_UMFPACK "Enable SuiteSparse UMFPACK sparse solver backend if found/provided" ON)
option(MATX_ENABLE_CXSPARSE "Enable SuiteSparse CXSparse sparse solver backend if found/provided" ON)
option(MATX_ENABLE_SUPERLU "Enable SuperLU sparse solver backend (auto-fetch)" OFF)
option(MATX_ENABLE_MUMPS "Enable MUMPS sparse solver backend if found/provided" OFF)
option(MATX_ENABLE_GRAPHBLAS "Enable GraphBlas for sparse numerical ops if found" OFF)
option(MATX_ENABLE_AOCL_SPARSE "Enable amd AOCL SPARSE for sparse numerical ops if found" OFF)

option(BUILD_SHARED_LIBS "Build shared library (ON) or static (OFF)" OFF)

set(MATX_DENSE_BLAS_BACKEND "AUTO" CACHE STRING "DENSE BLAS backend: AUTO|OPENBLAS|BLIS|REFERENCE")
set(MATX_SPARSE_BLAS_BACKEND "AUTO" CACHE STRING "SPARSE BLAS backend: AUTO|OPENBLAS|BLIS|REFERENCE")

# ---- Build type ----
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    #set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# =========================================================
# C / C++
# =========================================================

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(MATX_BLAS_AUTO_SELECTED ON)


# Allow manual override for cross-compilation
set(MATX_CPU_PLATFORM "" CACHE STRING "Override CPU platform: AMD | INTEL | OTHER (leave empty for auto-detect)")

if(MATX_CPU_PLATFORM STREQUAL "")

    set(CPU_DETECT_SOURCE ${CMAKE_BINARY_DIR}/detect_cpu.cpp)

    file(WRITE ${CPU_DETECT_SOURCE} "
    #include <iostream>
    #if defined(_MSC_VER)
    #include <intrin.h>
    #else
    #include <cpuid.h>
    #endif

    int main() {
    char vendor[13] = {0};

    #if defined(_MSC_VER)
    int regs[4];
    __cpuid(regs, 0);
    ((int*)vendor)[0] = regs[1];
    ((int*)vendor)[1] = regs[3];
    ((int*)vendor)[2] = regs[2];
    #else
    unsigned int eax, ebx, ecx, edx;
    __get_cpuid(0, &eax, &ebx, &ecx, &edx);
    ((unsigned int*)vendor)[0] = ebx;
    ((unsigned int*)vendor)[1] = edx;
    ((unsigned int*)vendor)[2] = ecx;
    #endif

    std::cout << vendor;
    return 0;
    }
    ")

    try_run(
        RUN_RESULT
        COMPILE_RESULT
        ${CMAKE_BINARY_DIR}
        ${CPU_DETECT_SOURCE}
        RUN_OUTPUT_VARIABLE CPU_VENDOR
    )

    string(STRIP "${CPU_VENDOR}" CPU_VENDOR)

    if(CPU_VENDOR STREQUAL "AuthenticAMD")
        set(CPU_PLATFORM "AMD")
    elseif(CPU_VENDOR STREQUAL "GenuineIntel")
        set(CPU_PLATFORM "INTEL")
    else()
        set(CPU_PLATFORM "OTHER")
    endif()

else()
    set(CPU_PLATFORM "${MATX_CPU_PLATFORM}")
    set(CPU_VENDOR "(manual override)")
endif()

set(CPU_PLATFORM "AMD")
message(STATUS "CPU vendor detected: ${CPU_VENDOR}")
message(STATUS "CPU platform: ${CPU_PLATFORM}")
# ---- auto detect cpu BLAS/LAPACK & Sparse ----
if(NOT DEFINED MATX_BLAS_AUTO_SELECTED)

    if(CPU_PLATFORM STREQUAL "AMD")
        set(MATX_ENABLE_BLIS ON CACHE BOOL "Enable BLIS backend" FORCE)
        set(MATX_ENABLE_LIBFLAME ON CACHE BOOL "Enable libFLAME LAPACK" FORCE)
        set(MATX_ENABLE_OPENBLAS OFF CACHE BOOL "" FORCE)
        set(MATX_ENABLE_GRAPHBLAS ON CACHE BOOL "" FORCE)
        set(MATX_ENABLE_AOCL_SPARSE ON CACHE BOOL "" FORCE)
    elseif(CPU_PLATFORM STREQUAL "INTEL")
        set(MATX_ENABLE_OPENBLAS ON CACHE BOOL "Enable OpenBLAS backend" FORCE)
        set(MATX_ENABLE_GRAPHBLAS ON CACHE BOOL "Enable GraphBLAS" FORCE)
        set(MATX_ENABLE_BLIS OFF CACHE BOOL "" FORCE)
        set(MATX_ENABLE_LIBFLAME OFF CACHE BOOL "" FORCE)
        set(MATX_ENABLE_AOCL_SPARSE OFF CACHE BOOL "" FORCE)
    else()
        set(MATX_ENABLE_OPENBLAS ON CACHE BOOL "Enable OpenBLAS backend" FORCE)
        set(MATX_ENABLE_BLIS OFF CACHE BOOL "" FORCE)
        set(MATX_ENABLE_LIBFLAME OFF CACHE BOOL "" FORCE)
        set(MATX_ENABLE_GRAPHBLAS OFF CACHE BOOL "" FORCE)
    endif()

    set(MATX_BLAS_AUTO_SELECTED ON CACHE BOOL "" FORCE)

endif()

# ---- Validate BLAS/LAPACK choices ----
set(BLAS_LIST "")

if(MATX_ENABLE_OPENBLAS)
    list(APPEND BLAS_LIST "OpenBLAS")
endif()

message(STATUS "44444444")
if(MATX_ENABLE_BLIS)
    message(STATUS "3333333333333333")
    list(APPEND BLAS_LIST "BLIS")
endif()

list(LENGTH BLAS_LIST BLAS_COUNT)

if(BLAS_COUNT EQUAL 0)
    message(FATAL_ERROR
        "No BLAS backend enabled. Enable one of: OpenBLAS, BLIS, MKL.")
elseif(BLAS_COUNT GREATER 1)
    message(FATAL_ERROR
        "Multiple BLAS backends enabled (${BLAS_LIST}). Only one BLAS backend is allowed.")
endif()

set(DEPEND_LIB_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/deps_output")

function(matx_spdlog)
    include(FetchContent)
    FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.13.0
    )

    FetchContent_GetProperties(spdlog)

    if(NOT blis_POPULATED)
        FetchContent_Populate(spdlog)
    endif()

    # =========================================================
    # spdlog build dirs
    # =========================================================

    execute_process(
        COMMAND ${CMAKE_COMMAND}
        -S ${spdlog_SOURCE_DIR}
        -B ${spdlog_BINARY_DIR}
        -DCMAKE_INSTALL_PREFIX=${DEPEND_LIB_OUTPUT}/spdlog
    )

    # =========================================================
    # build spdlog
    # =========================================================

    execute_process(
        COMMAND ${CMAKE_COMMAND}
        --build ${spdlog_BINARY_DIR}
        --parallel
        RESULT_VARIABLE BLIS_BUILD_RESULT
    )

    execute_process(
        COMMAND ${CMAKE_COMMAND}
        --install ${spdlog_BINARY_DIR}
        RESULT_VARIABLE spdlog_INSTALL_RESULT
    )
endfunction()
matx_spdlog()

set(spdlog_DIR ${DEPEND_LIB_OUTPUT}/spdlog/lib/cmake/spdlog)
find_package(spdlog REQUIRED)
# =========================================================
# OpenBLAS source build
# =========================================================
function(matx_add_openblas)

    include(FetchContent)

    FetchContent_Declare(
        OpenBLAS
        GIT_REPOSITORY https://github.com/OpenMathLib/OpenBLAS.git
        GIT_TAG v0.3.33
    )

    FetchContent_GetProperties(OpenBLAS)

    if(NOT OpenBLAS_POPULATED)
        FetchContent_Populate(OpenBLAS)
    endif()
    set(OPENBLAS_BUILD_DIR
        ${CMAKE_BINARY_DIR}/openblas-build)

    file(MAKE_DIRECTORY ${OPENBLAS_BUILD_DIR})

    execute_process(
        COMMAND
        ${CMAKE_COMMAND}
        -S ${openblas_SOURCE_DIR}
        -B ${OPENBLAS_BUILD_DIR}
        -DBUILD_SHARED_LIBS=ON
        -DCMAKE_INSTALL_PREFIX=${DEPEND_LIB_OUTPUT}/openblas

        RESULT_VARIABLE OPENBLAS_CONFIG_RESULT
    )

    #execute_process(

    if(NOT OPENBLAS_CONFIG_RESULT EQUAL 0)
        message(FATAL_ERROR "OpenBLAS configure failed")
    endif()

    execute_process(
        COMMAND
        ${CMAKE_COMMAND}
        --build ${OPENBLAS_BUILD_DIR}
        --parallel

        RESULT_VARIABLE OPENBLAS_BUILD_RESULT
    )
    execute_process(
        COMMAND ${CMAKE_COMMAND}
        --install ${OPENBLAS_BUILD_DIR}
        RESULT_VARIABLE MUMPS_INSTALL_RESULT
    )

    if(NOT OPENBLAS_BUILD_RESULT EQUAL 0)
        message(FATAL_ERROR "OpenBLAS build failed")
    endif()
    if(WIN32)

        if(EXISTS ${OPENBLAS_BUILD_DIR}/lib/libopenblas.dll.a)
            set(OPENBLAS_LIB
                ${OPENBLAS_BUILD_DIR}/lib/libopenblas.dll.a)

        elseif(EXISTS ${OPENBLAS_BUILD_DIR}/lib/libopenblas.a)
            set(OPENBLAS_LIB
                ${OPENBLAS_BUILD_DIR}/lib/libopenblas.a)

        else()
            message(FATAL_ERROR
                "Cannot find built OpenBLAS library")
        endif()

    else()

        set(OPENBLAS_LIB
            ${OPENBLAS_BUILD_DIR}/libopenblas.a)

    endif()
    add_library(openblas STATIC IMPORTED GLOBAL)

    set_target_properties(openblas PROPERTIES
        IMPORTED_LOCATION ${OPENBLAS_LIB}

        INTERFACE_INCLUDE_DIRECTORIES
        "${OPENBLAS_BUILD_DIR}/install/include"
    )

    add_library(OpenBLAS::OpenBLAS ALIAS openblas)
    set(BLAS_LIBRARIES
        ${OPENBLAS_LIB}
        PARENT_SCOPE)

    set(LAPACK_LIBRARIES
        ${OPENBLAS_LIB}
        PARENT_SCOPE)

    message(STATUS "OpenBLAS lib: ${OPENBLAS_LIB}")

endfunction()

set(AOCL_ROOT ${DEPEND_LIB_OUTPUT}/aocl_root)
if(CPU_PLATFORM STREQUAL "AMD")
    include(FetchContent)

    FetchContent_Declare(
        aocl-utils
        GIT_REPOSITORY https://github.com/amd/aocl-utils.git
        GIT_TAG 5.2.2
    )

    FetchContent_GetProperties(aocl-utils)

    if(NOT aocl-utils_POPULATED)
        FetchContent_Populate(aocl-utils)

        execute_process(
            COMMAND ${CMAKE_COMMAND}
            -S ${aocl-utils_SOURCE_DIR}
            -B ${aocl-utils_BINARY_DIR}
            -DENABLE_ILP64=ON
            -DCMAKE_BUILD_TYPE=RELEASE
            -DCMAKE_INSTALL_PREFIX=${AOCL_ROOT}
        )

        message(STATUS "Building AOCL Utils...")

        execute_process(
            COMMAND ${CMAKE_COMMAND}
            --build ${aocl-utils_BINARY_DIR}
            --parallel
            RESULT_VARIABLE AOCL_UTILS_BUILD_RESULT
        )

        if(NOT AOCL_UTILS_BUILD_RESULT EQUAL 0)
            message(FATAL_ERROR "AOCL Utils build failed")
        endif()

        message(STATUS "Installing AOCL Utils...")

        execute_process(
            COMMAND ${CMAKE_COMMAND}
            --install ${aocl-utils_BINARY_DIR}
            RESULT_VARIABLE AOCL_UTILS_INSTALL_RESULT
        )

        if(NOT AOCL_UTILS_INSTALL_RESULT EQUAL 0)
            message(FATAL_ERROR "AOCL Utils install failed")
        endif()

        # 可选：导出路径
        set(AOCL_UTILS_ROOT
            ${aocl-utils_BINARY_DIR}/install
            CACHE PATH "AOCL Utils install path")

        message(STATUS "AOCL Utils installed to: ${AOCL_UTILS_ROOT}")
        set(AOCL_UTILS_LIBRARY ${AOCL_ROOT}/lib/libaoclutils.so)
        set(AOCL_UTILS_INCLUDE_DIR ${AOCL_ROOT}/include)
    endif()
endif()

if(MATX_ENABLE_OPENBLAS)
    matx_add_openblas()
elseif(MATX_ENABLE_BLIS)

    # set(BLIS_CONFIG_FAMILY auto)
    include(FetchContent)

    FetchContent_Declare(
        blis
        GIT_REPOSITORY https://github.com/amd/blis.git
        GIT_TAG 5.2.2
    )

    FetchContent_GetProperties(blis)

    if(NOT blis_POPULATED)
        FetchContent_Populate(blis)
    endif()

    # =========================================================
    # BLIS build dirs
    # =========================================================

    execute_process(
        COMMAND ${CMAKE_COMMAND}
        -S ${blis_SOURCE_DIR}
        -B ${blis_BINARY_DIR}
        -DBLAS_INT_SIZE=64
        -DBLIS_CONFIG_FAMILY=auto
        -DBLIS_DISABLE_F77_TYPES=ON
        -DBLIS_ENABLE_NOFORTRAN=ON
        -DBLIS_ENABLE_OPENMP=ON
        -DENABLE_THREADING=openmp
        -DENABLE_CBLAS=ON
        -DCMAKE_INSTALL_PREFIX=${AOCL_ROOT}
    )

    # =========================================================
    # build BLIS
    # =========================================================

    execute_process(
        COMMAND ${CMAKE_COMMAND}
        --build ${blis_BINARY_DIR}
        --parallel
        RESULT_VARIABLE BLIS_BUILD_RESULT
    )

    execute_process(
        COMMAND ${CMAKE_COMMAND}
        --install ${blis_BINARY_DIR}
        RESULT_VARIABLE BLIS_INSTALL_RESULT
    )
    # =========================================================
    # imported target
    # =========================================================

    #   add_library(blis STATIC IMPORTED GLOBAL)
    #   set_target_properties(blis PROPERTIES
    #       IMPORTED_LOCATION
    #       ${blis_BINARY_DIR}/install/lib/libblis.so

    #       INTERFACE_INCLUDE_DIRECTORIES
    #       "${BLIS_INSTALL_DIR}/include"
    #   )

    set(BLAS_LIBRARY ${AOCL_ROOT}/lib/libblis-mt.so)
    set(BLAS_LIBRARIES ${BLAS_LIBRARY})
    set(AOCL_BLAS_INCLUDE_DIR ${AOCL_ROOT}/include)
    set(amdblis_DIR "${CMAKE_CURRENT_BINARY_DIR}/aocl_cmake/")
    find_package(amdblis REQUIRED)

endif()

# =========================================================
# AOCL libflame
# =========================================================
if(0)
    if(MATX_ENABLE_LIBFLAME)
        include(FetchContent)

        FetchContent_Declare(
            libflame
            GIT_REPOSITORY https://github.com/amd/libflame.git
            GIT_TAG 5.2.2
            BINARY_DIR ${CMAKE_BINARY_DIR}/libflame-build
        )

        FetchContent_GetProperties(libflame)
        if(NOT libflame_POPULATED)
            FetchContent_Populate(libflame)

            file(MAKE_DIRECTORY ${libflame_BINARY_DIR})
            execute_process(
                COMMAND ${CMAKE_COMMAND}
                -S ${libflame_SOURCE_DIR}
                -B ${libflame_BINARY_DIR}
                -DCMAKE_BUILD_TYPE=RELEASE
                -DBLIS_CONFIG_FAMILY=auto
                -DAOCL_ROOT=${AOCL_ROOT}
                -DENABLE_ILP64=ON
                -DCMAKE_C_STANDARD_LIBRARIES="-lm"
                -DCMAKE_SHARED_LINKER_FLAGS="-lm"
                -DBLAS_LIBRARY=${BLAS_LIBRARY}
                -DENABLE_AMD_FLAGS=ON
                -DENABLE_AOCL_BLAS=ON
                -DCMAKE_INSTALL_PREFIX=${AOCL_ROOT}
                -DBUILD_SHARED_LIBS=ON
            )

            # =========================================================
            # build BLIS
            # =========================================================

            execute_process(
                COMMAND ${CMAKE_COMMAND}
                --build ${libflame_BINARY_DIR}
                --parallel
                RESULT_VARIABLE libflame_BUILD_RESULT
            )

            execute_process(
                COMMAND ${CMAKE_COMMAND}
                --install ${libflame_BINARY_DIR}
                RESULT_VARIABLE LIBFLAME_INSTALL_RESULT
            )

            set(LIBFLAME_INCLUDE_DIR ${AOCL_ROOT}/include)
            set(LIBFLAME_LIBRARY ${AOCL_ROOT}/lib/libflame.so)

            add_library(libflame::libflame STATIC IMPORTED GLOBAL)
            set_target_properties(libflame::libflame PROPERTIES
                IMPORTED_LOCATION ${LIBFLAME_LIBRARY}
                INTERFACE_INCLUDE_DIRECTORIES ${LIBFLAME_INCLUDE_DIR}
            )
            set(LAPACK_LIBRARIES ${LIBFLAME_LIBRARY})
        endif()
    endif()
endif()

if(MATX_ENABLE_LIBFLAME)
    set(LIBFLAME_LIBRARY_PATH "${AOCL_ROOT}/lib/libflame.so")
    set(LIBFLAME_HEADER_PATH "${AOCL_ROOT}/include/FLAME.h")

    if(EXISTS ${LIBFLAME_LIBRARY_PATH} AND EXISTS ${LIBFLAME_HEADER_PATH})
        message(STATUS "✅ libflame exists，do not compile: ${LIBFLAME_LIBRARY_PATH}")

        set(LAPACK_LIBRARIES ${LIBFLAME_LIBRARY_PATH})
    else()
        message(STATUS "🔨 libflame do not exist，start compile and install...")

        include(FetchContent)
        FetchContent_Declare(
            libflame
            GIT_REPOSITORY https://github.com/amd/libflame.git
            GIT_TAG 5.2.2
            BINARY_DIR ${CMAKE_BINARY_DIR}/libflame-build
        )

        FetchContent_GetProperties(libflame)
        if(NOT libflame_POPULATED)
            FetchContent_Populate(libflame)

            file(MAKE_DIRECTORY ${libflame_BINARY_DIR})
            execute_process(
                COMMAND ${CMAKE_COMMAND}
                -S ${libflame_SOURCE_DIR}
                -B ${libflame_BINARY_DIR}
                -DCMAKE_BUILD_TYPE=RELEASE
                -DBLIS_CONFIG_FAMILY=auto
                -DAOCL_ROOT=${AOCL_ROOT}
                -DENABLE_ILP64=ON
                -DCMAKE_C_STANDARD_LIBRARIES="-lm"
                -DCMAKE_SHARED_LINKER_FLAGS="-lm"
                -DBLAS_LIBRARY=${BLAS_LIBRARY}
                -DENABLE_AMD_FLAGS=ON
                -DENABLE_AOCL_BLAS=ON
                -DCMAKE_INSTALL_PREFIX=${AOCL_ROOT}
                -DBUILD_SHARED_LIBS=ON
            )

            execute_process(
                COMMAND ${CMAKE_COMMAND}
                --build ${libflame_BINARY_DIR}
                --parallel
                RESULT_VARIABLE libflame_BUILD_RESULT
            )

            execute_process(
                COMMAND ${CMAKE_COMMAND}
                --install ${libflame_BINARY_DIR}
                RESULT_VARIABLE LIBFLAME_INSTALL_RESULT
            )

            set(LAPACK_LIBRARIES ${AOCL_ROOT}/lib/libflame.so)
        endif()
    endif()
    set(libflame_DIR "${CMAKE_CURRENT_BINARY_DIR}/aocl_cmake/")
    find_package(libflame REQUIRED)

    set(aoclutils_DIR "${CMAKE_CURRENT_BINARY_DIR}/aocl_cmake/")
    find_package(aoclutils REQUIRED)
endif()

# =========================================================
# SuiteSparse
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

    #
    # SuiteSparse build dir
    #

    set(SUITESPARSE_BUILD_DIR
        ${CMAKE_BINARY_DIR}/suitesparse-build)

    file(MAKE_DIRECTORY
        ${SUITESPARSE_BUILD_DIR})

    set(OPENBLAS_INCLUDE_DIR
        ${OPENBLAS_BUILD_DIR}/include)

    #
    # Configure SuiteSparse
    #
    message(STATUS "SU333 ${BLAS_LIBRARIES}")
    message(STATUS "SU222 ${LAPACK_LIBRARIES}")
    message(STATUS "SU222 ${aocl_root}")
    execute_process(
        COMMAND
        ${CMAKE_COMMAND}
        -S ${suitesparse_SOURCE_DIR}
        -B ${SUITESPARSE_BUILD_DIR}
        -DBLA_VENDOR=FLAME
        -DBLAS_LIBRARIES=${BLAS_LIBRARIES}\;${AOCL_ROOT}/lib/libaoclutils.so
        -DLAPACK_LIBRARIES=${LAPACK_LIBRARIES}
        -DSUITESPARSE_USE_64BIT_BLAS=ON
        -DCMAKE_INCLUDE_PATH=${OPENBLAS_INCLUDE_DIR}
        -DCMAKE_LIBRARY_PATH=${OPENBLAS_LIB_DIR}\;${AOCL_ROOT}/lib
        -DCMAKE_INSTALL_RPATH=${AOCL_ROOT}/lib
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
        -DBUILD_SHARED_LIBS=ON
        -DCMAKE_INSTALL_PREFIX=${DEPEND_LIB_OUTPUT}/suitesparse

        RESULT_VARIABLE SUITESPARSE_CONFIG_RESULT
    )

    if(NOT SUITESPARSE_CONFIG_RESULT EQUAL 0)

        message(FATAL_ERROR
            "SuiteSparse configure failed")

    endif()
    execute_process(

        COMMAND
        ${CMAKE_COMMAND}
        --build ${SUITESPARSE_BUILD_DIR}
        --parallel

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

    #
    # Example imported target
    #
    set(SuiteSparse_config_DIR "${DEPEND_LIB_OUTPUT}/suitesparse/lib/cmake/SuiteSparse_config")
    set(BTF_DIR "${DEPEND_LIB_OUTPUT}/suitesparse/lib/cmake/BTF")
    set(AMD_DIR "${DEPEND_LIB_OUTPUT}/suitesparse/lib/cmake/AMD")
    set(COLAMD_DIR "${DEPEND_LIB_OUTPUT}/suitesparse/lib/cmake/COLAMD")
    set(CCOLAMD_DIR "${DEPEND_LIB_OUTPUT}/suitesparse/lib/cmake/CCOLAMD")
    set(CHOLMOD_DIR "${DEPEND_LIB_OUTPUT}/suitesparse/lib/cmake/CHOLMOD")
    set(CAMD_DIR "${DEPEND_LIB_OUTPUT}/suitesparse/lib/cmake/CAMD")
    set(KLU_DIR "${DEPEND_LIB_OUTPUT}/suitesparse/lib/cmake/KLU")
    set(GraphBLAS_DIR "${DEPEND_LIB_OUTPUT}/suitesparse/lib/cmake/GraphBLAS")
    set(UMFPACK_DIR "${DEPEND_LIB_OUTPUT}/suitesparse/lib/cmake/UMFPACK")
    set(CXSparse_DIR "${DEPEND_LIB_OUTPUT}/suitesparse/lib/cmake/CXSparse")
    find_package(SuiteSparse_config REQUIRED)
    find_package(KLU REQUIRED)
    find_package(GraphBLAS REQUIRED)
    find_package(UMFPACK REQUIRED)
    find_package(CXSparse REQUIRED)
endif()

# =========================================================
# AOCL Sparse
# =========================================================

if(MATX_ENABLE_AOCL_SPARSE)

    FetchContent_Declare(
        aocl-sparse
        GIT_REPOSITORY https://github.com/amd/aocl-sparse.git
        GIT_TAG 5.2.2
    )
    FetchContent_GetProperties(aocl-sparse)

    if(NOT aocl-sparse_POPULATED)
        FetchContent_Populate(aocl-sparse)
    endif()

    execute_process(
        COMMAND ${CMAKE_COMMAND}
        -S ${aocl-sparse_SOURCE_DIR}
        -B ${aocl-sparse_BINARY_DIR}
        -DCMAKE_BUILD_TYPE=RELEASE
        -DBLIS_CONFIG_FAMILY=auto
        -DCMAKE_AOCL_ROOT=${AOCL_ROOT}
        -DSUPPORT_OMP=ON
        -DBUILD_ILP64=ON
        -DCMAKE_INSTALL_PREFIX=${AOCL_ROOT}
        -DBUILD_SHARED_LIBS=ON
    )

    # =========================================================
    # build aocl-sparse
    # =========================================================

    execute_process(
        COMMAND ${CMAKE_COMMAND}
        --build ${aocl-sparse_BINARY_DIR}
        --parallel
        RESULT_VARIABLE libflame_BUILD_RESULT
    )

    execute_process(
        COMMAND ${CMAKE_COMMAND}
        --install ${aocl-sparse_BINARY_DIR}
        RESULT_VARIABLE LIBFLAME_INSTALL_RESULT
    )

    set(AOCLSPARSE_INCLUDE_DIR ${AOCL_ROOT}/include)
    set(AOCLSPARSE_LIBRARY ${AOCL_ROOT}/lib/libflame.so)

    add_library(AOCLSPARSE::AOCLSPARSE STATIC IMPORTED GLOBAL)
    set_target_properties(AOCLSPARSE::AOCLSPARSE PROPERTIES
        IMPORTED_LOCATION ${AOCLSPARSE_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES ${AOCLSPARSE_INCLUDE_DIR}
    )
    set(aoclsparse_DIR "${CMAKE_CURRENT_BINARY_DIR}/aocl_cmake/")
    find_package(aoclsparse REQUIRED)
endif()

if(MATX_ENABLE_SUPERLU)

    FetchContent_Declare(
        superlu
        GIT_REPOSITORY https://github.com/xiaoyeli/superlu.git
        GIT_TAG v7.0.1
    )

    FetchContent_GetProperties(superlu)

    if(NOT superlu_POPULATED)
        FetchContent_Populate(superlu)
    endif()

    set(SUPERLU_BUILD_DIR
        ${CMAKE_BINARY_DIR}/superlu-build)

    file(MAKE_DIRECTORY
        ${SUPERLU_BUILD_DIR})

    #
    # Configure
    #

    execute_process(
        COMMAND
        ${CMAKE_COMMAND}
        -S ${superlu_SOURCE_DIR}
        -B ${SUPERLU_BUILD_DIR}
        -DTPL_BLAS_LIBRARIES=${OPENBLAS_LIB}
        -Denable_tests=OFF
        -Denable_examples=OFF
        -DBUILD_SHARED_LIBS=OFF
        -DCMAKE_INSTALL_PREFIX=${DEPEND_LIB_OUTPUT}/superlu_lib

        RESULT_VARIABLE SUPERLU_CONFIG_RESULT
    )

    if(NOT SUPERLU_CONFIG_RESULT EQUAL 0)

        message(FATAL_ERROR
            "SuperLU configure failed")

    endif()

    #
    # Build
    #

    execute_process(

        COMMAND
        ${CMAKE_COMMAND}
        --build ${SUPERLU_BUILD_DIR}
        --parallel

        RESULT_VARIABLE SUPERLU_BUILD_RESULT
    )

    if(NOT SUPERLU_BUILD_RESULT EQUAL 0)

        message(FATAL_ERROR
            "SuperLU build failed")

    endif()

    #
    # Imported target
    #
    execute_process(
        COMMAND ${CMAKE_COMMAND}
        --install ${SUPERLU_BUILD_DIR}
        RESULT_VARIABLE MUMPS_INSTALL_RESULT
    )

    add_library(SUPERLU::SUPERLU STATIC IMPORTED GLOBAL)

    set_target_properties(SUPERLU::SUPERLU PROPERTIES
        IMPORTED_LOCATION
        "${DEPEND_LIB_OUTPUT}/superlu_lib/lib/libsuperlu.a"

        INTERFACE_INCLUDE_DIRECTORIES
        "${DEPEND_LIB_OUTPUT}/superlu_lib/include"
    )

endif()

if(MATX_ENABLE_MUMPS)

    FetchContent_Declare(
        mumps
        GIT_REPOSITORY https://github.com/scivision/mumps-superbuild.git
        GIT_TAG v5.9.0.0
    )

    FetchContent_GetProperties(mumps)

    if(NOT mumps_POPULATED)
        FetchContent_Populate(mumps)
    endif()

    set(MUMPS_BUILD_DIR
        ${CMAKE_BINARY_DIR}/mumps-build)

    file(MAKE_DIRECTORY
        ${MUMPS_BUILD_DIR})

    #
    # Configure
    #
    message(STATUS "000000000000000000+${LAPACK_LIBRARIES}")
    message(STATUS "000000000000000000+${BLAS_LIBRARIES}")
    execute_process(

        COMMAND
        ${CMAKE_COMMAND}

        -S ${mumps_SOURCE_DIR}
        -B ${MUMPS_BUILD_DIR}

        -DBUILD_SHARED_LIBS=OFF
        -DMUMPS_parallel=OFF
        -DBLAS_LIBRARIES=${BLAS_LIBRARIES}
        -DMUMPS_intsize64=ON
        -DBUILD_COMPLEX16=ON
        -DBUILD_COMPLEX=ON
        -DLAPACK_LIBRARY=${LAPACK_LIBRARIES}\;${BLAS_LIBRARIES}
        -DBUILD_SHARED_LIBS=ON
        -DCMAKE_INSTALL_PREFIX=${DEPEND_LIB_OUTPUT}/mumps

        RESULT_VARIABLE MUMPS_CONFIG_RESULT
    )

    if(NOT MUMPS_CONFIG_RESULT EQUAL 0)

        message(FATAL_ERROR
            "MUMPS configure failed")

    endif()

    #
    # Build
    #

    execute_process(

        COMMAND
        ${CMAKE_COMMAND}
        --build ${MUMPS_BUILD_DIR}
        --parallel

        RESULT_VARIABLE MUMPS_BUILD_RESULT
    )
    execute_process(
        COMMAND ${CMAKE_COMMAND}
        --install ${MUMPS_BUILD_DIR}
        RESULT_VARIABLE MUMPS_INSTALL_RESULT
    )

    if(NOT MUMPS_BUILD_RESULT EQUAL 0)

        message(FATAL_ERROR
            "MUMPS build failed")

    endif()
    add_library(MUMPS::MUMPS STATIC IMPORTED GLOBAL)

    set_target_properties(MUMPS::MUMPS PROPERTIES
        IMPORTED_LOCATION
        "${DEPEND_LIB_OUTPUT}/mumps/lib/libdmumps.so"

        INTERFACE_INCLUDE_DIRECTORIES
        "${DEPEND_LIB_OUTPUT}/mumps/include"
    )

    set(mumps_lib 
        "${DEPEND_LIB_OUTPUT}/mumps/lib/libdmumps.so"
        "${DEPEND_LIB_OUTPUT}/mumps/lib/libzmumps.so"
        "${DEPEND_LIB_OUTPUT}/mumps/lib/libmumps_common.so"
        "${DEPEND_LIB_OUTPUT}/mumps/lib/libmpiseq_fortran.so"
        "${DEPEND_LIB_OUTPUT}/mumps/lib/libmpiseq_c.so"
        "${DEPEND_LIB_OUTPUT}/mumps/lib/libpord.so"
    )

endif()


add_subdirectory(types)
add_subdirectory(tools)
add_subdirectory(core)
add_subdirectory(io)
add_subdirectory(vec_blas)
add_subdirectory(dense_blas)
add_subdirectory(sparse_blas)
add_subdirectory(dense_solve)
add_subdirectory(sparse_solve)

if(MATX_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# ---- Export targets ----
install(EXPORT MatXConfig
  NAMESPACE MatX::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MatX
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/generate_dir/include/config.h
    @ONLY
)
