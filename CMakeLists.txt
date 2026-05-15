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

#message(STATUS "current 3Party path ${3Party_path}")

include(FetchContent)

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
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# =========================================================
# C / C++
# =========================================================

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(MATX_BLAS_AUTO_SELECTED ON)


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

if(MATX_ENABLE_BLIS)
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
          -DBUILD_SHARED_LIBS=OFF

      RESULT_VARIABLE OPENBLAS_CONFIG_RESULT
  )

  execute_process(

      COMMAND
          ${CMAKE_COMMAND}
          --install ${OPENBLAS_BUILD_DIR}

          --prefix ${OPENBLAS_BUILD_DIR}/install

      RESULT_VARIABLE OPENBLAS_INSTALL_RESULT
  )

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
              -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
              -DCMAKE_INSTALL_PREFIX=${aocl-utils_BINARY_DIR}/install
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
  endif()
endif()

if(MATX_ENABLE_OPENBLAS)
  matx_add_openblas()
elseif(MATX_ENABLE_BLIS)
  include(FetchContent)
  FetchContent_Declare(
    blis
    GIT_REPOSITORY https://github.com/amd/blis.git
    GIT_TAG 5.2.2
  )
  FetchContent_Populate(blis)
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
    execute_process(

        COMMAND
            ${CMAKE_COMMAND}

            -S ${suitesparse_SOURCE_DIR}
            -B ${SUITESPARSE_BUILD_DIR}

            -DBLA_VENDOR=OpenBLAS

            -DBLAS_LIBRARIES=${BLAS_LIBRARIES}
            -DLAPACK_LIBRARIES=${LAPACK_LIBRARIES}
            -DSUITESPARSE_USE_64BIT_BLAS=ON
            -DCMAKE_INCLUDE_PATH=${OPENBLAS_INCLUDE_DIR}
            -DCMAKE_LIBRARY_PATH=${OPENBLAS_LIB_DIR}

            -DBUILD_SHARED_LIBS=OFF

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

    if(NOT SUITESPARSE_BUILD_RESULT EQUAL 0)

        message(FATAL_ERROR
            "SuiteSparse build failed")

    endif()

    #
    # Example imported target
    #

    add_library(SuiteSparse::AMD STATIC IMPORTED GLOBAL)

    set_target_properties(SuiteSparse::AMD PROPERTIES
        IMPORTED_LOCATION
        "${SUITESPARSE_BUILD_DIR}/AMD/libamd.a"
        INTERFACE_INCLUDE_DIRECTORIES
        "${suitesparse_SOURCE_DIR}/AMD/Include"
    )

  add_library(SuiteSparse::COLAMD STATIC IMPORTED GLOBAL)

  set_target_properties(SuiteSparse::COLAMD PROPERTIES
      IMPORTED_LOCATION
      "${SUITESPARSE_BUILD_DIR}/COLAMD/libcolamd.a"
      INTERFACE_INCLUDE_DIRECTORIES
      "${suitesparse_SOURCE_DIR}/COLAMD/Include"
  )

  add_library(SuiteSparse::CHOLMOD STATIC IMPORTED GLOBAL)

  set_target_properties(SuiteSparse::CHOLMOD PROPERTIES
      IMPORTED_LOCATION
      "${SUITESPARSE_BUILD_DIR}/CHOLMOD/libcholmod.a"
      INTERFACE_INCLUDE_DIRECTORIES
      "${suitesparse_SOURCE_DIR}/CHOLMOD/Include"
  )

  add_library(SuiteSparse::BTF STATIC IMPORTED GLOBAL)

  set_target_properties(SuiteSparse::BTF PROPERTIES
      IMPORTED_LOCATION
      "${SUITESPARSE_BUILD_DIR}/BTF/libbtf.a"
      INTERFACE_INCLUDE_DIRECTORIES
      "${suitesparse_SOURCE_DIR}/BTF/Include"
  )

  add_library(SuiteSparse::CAMD STATIC IMPORTED GLOBAL)

  set_target_properties(SuiteSparse::CAMD PROPERTIES
      IMPORTED_LOCATION
      "${SUITESPARSE_BUILD_DIR}/CAMD/libcamd.a"
      INTERFACE_INCLUDE_DIRECTORIES
      "${suitesparse_SOURCE_DIR}/CAMD/Include"
  )

  add_library(SuiteSparse::CCOLAMD STATIC IMPORTED GLOBAL)

  set_target_properties(SuiteSparse::CCOLAMD PROPERTIES
      IMPORTED_LOCATION
      "${SUITESPARSE_BUILD_DIR}/CCOLAMD/libccolamd.a"
      INTERFACE_INCLUDE_DIRECTORIES
      "${suitesparse_SOURCE_DIR}/CCOLAMD/Include"
  )

  add_library(SuiteSparse::KLU STATIC IMPORTED GLOBAL)

  set_target_properties(SuiteSparse::KLU PROPERTIES
      IMPORTED_LOCATION
      "${SUITESPARSE_BUILD_DIR}/KLU/libklu.a"
      INTERFACE_INCLUDE_DIRECTORIES
      "${suitesparse_SOURCE_DIR}/KLU/Include"
  )

  add_library(SuiteSparse::UMFPACK STATIC IMPORTED GLOBAL)

  set_target_properties(SuiteSparse::UMFPACK PROPERTIES
      IMPORTED_LOCATION
      "${SUITESPARSE_BUILD_DIR}/UMFPACK/libumfpack.a"
      INTERFACE_INCLUDE_DIRECTORIES
      "${suitesparse_SOURCE_DIR}/UMFPACK/Include"
  )

  add_library(SuiteSparse::GraphBLAS STATIC IMPORTED GLOBAL)

  set_target_properties(SuiteSparse::GraphBLAS PROPERTIES
      IMPORTED_LOCATION
      "${SUITESPARSE_BUILD_DIR}/GraphBLAS/libgraphblas.a"
      INTERFACE_INCLUDE_DIRECTORIES
      "${suitesparse_SOURCE_DIR}/GraphBLAS/Include"
  )


  add_library(SuiteSparse::CXSparse STATIC IMPORTED GLOBAL)

  set_target_properties(SuiteSparse::CXSparse PROPERTIES
      IMPORTED_LOCATION
      "${SUITESPARSE_BUILD_DIR}/CXSparse/libcxsparse.a"
      INTERFACE_INCLUDE_DIRECTORIES
      "${suitesparse_SOURCE_DIR}/CXSparse/Include"
  )

add_library(SuiteSparse::SuiteSparse_config STATIC IMPORTED GLOBAL)

set_target_properties(SuiteSparse::SuiteSparse_config PROPERTIES
    IMPORTED_LOCATION
    "${SUITESPARSE_BUILD_DIR}/SuiteSparse_config/libsuitesparseconfig.a"
    INTERFACE_INCLUDE_DIRECTORIES
    "${suitesparse_SOURCE_DIR}/SuiteSparse_config"
)

endif()

# =========================================================
# AOCL libflame
# =========================================================
if(MATX_ENABLE_LIBFLAME)
 include(FetchContent)

 FetchContent_Declare(
   libflame
   GIT_REPOSITORY https://github.com/flame/libflame.git
   GIT_TAG 5.2.2
 )

 FetchContent_Populate(libflame)
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

  FetchContent_MakeAvailable(aocl-sparse)

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

  add_library(SUPERLU::SUPERLU STATIC IMPORTED GLOBAL)

  set_target_properties(SUPERLU::SUPERLU PROPERTIES
      IMPORTED_LOCATION
      "${SUPERLU_BUILD_DIR}/lib/libsuperlu.a"

      INTERFACE_INCLUDE_DIRECTORIES
      "${superlu_SOURCE_DIR}/SRC"
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
  execute_process(

      COMMAND
          ${CMAKE_COMMAND}

          -S ${mumps_SOURCE_DIR}
          -B ${MUMPS_BUILD_DIR}

          -DBUILD_SHARED_LIBS=OFF

          -DBLA_VENDOR=OpenBLAS

           -G Ninja
          -DCMAKE_C_COMPILER=C:/msys64/ucrt64/bin/gcc.exe
          -DCMAKE_CXX_COMPILER=C:/msys64/ucrt64/bin/g++.exe
          -DCMAKE_Fortran_COMPILER=C:/msys64/ucrt64/bin/gfortran.exe
          -DMUMPS_parallel=OFF
          -DBLAS_LIBRARIES=${LAPACK_LIBRARIES}

          -DLAPACK_LIBRARY=${LAPACK_LIBRARIES}

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

  if(NOT MUMPS_BUILD_RESULT EQUAL 0)

      message(FATAL_ERROR
          "MUMPS build failed")

  endif()

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
add_subdirectory(tests)