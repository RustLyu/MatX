cmake_minimum_required(VERSION 3.20)
project(MatX LANGUAGES C CXX)

# ---- Global options ----
option(MATX_BUILD_TESTS "Build unit tests (gtest)" ON)

option(MATX_ENABLE_OPENBLAS "Enable OpenBLAS backend if found/provided" OFF)
option(MATX_ENABLE_BLIS "Enable BLIS backend if found/provided" OFF)
option(MATX_ENABLE_LIBFLAME "Enable LIBFLAME backend if found/provided" OFF)
option(MATX_ENABLE_MKL "Enable MKL backend if found/provided" OFF)
option(MATX_ENABLE_SUITESPARSE "Enable SuiteSparse backend if found/provided" ON)
option(MATX_ENABLE_GRAPHBLAS "Enable GraphBlas for sparse numerical ops if found" OFF)

set(MATX_DENSE_BLAS_BACKEND "AUTO" CACHE STRING "DENSE BLAS backend: AUTO|OPENBLAS|BLIS|REFERENCE")
set(MATX_SPARSE_BLAS_BACKEND "AUTO" CACHE STRING "SPARSE BLAS backend: AUTO|OPENBLAS|BLIS|REFERENCE")

# ---- Build type ----
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---- C/C++ Standard ----
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- spdlog ----
set(spdlog_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3party/spdlog_1.17.0/lib/cmake/spdlog")
find_package(spdlog REQUIRED)
message(STATUS "spdlog found at ${spdlog_DIR}")

include(CheckCXXSourceRuns)

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

message(STATUS "CPU vendor detected: ${CPU_VENDOR}")
message(STATUS "CPU platform: ${CPU_PLATFORM}")

# ---- auto detect cpu BLAS/LAPACK & Sparse ----
if(NOT DEFINED MATX_BLAS_AUTO_SELECTED)

  if(CPU_PLATFORM STREQUAL "AMD")
    set(MATX_ENABLE_BLIS ON CACHE BOOL "Enable BLIS backend" FORCE)
    set(MATX_ENABLE_LIBFLAME ON CACHE BOOL "Enable libFLAME LAPACK" FORCE)
    set(MATX_ENABLE_OPENBLAS OFF CACHE BOOL "" FORCE)
    set(MATX_ENABLE_MKL OFF CACHE BOOL "" FORCE)
    set(MATX_ENABLE_GRAPHBLAS ON CACHE BOOL "" FORCE)
  elseif(CPU_PLATFORM STREQUAL "INTEL")
    set(MATX_ENABLE_OPENBLAS ON CACHE BOOL "Enable OpenBLAS backend" FORCE)
    set(MATX_ENABLE_GRAPHBLAS ON CACHE BOOL "Enable GraphBLAS" FORCE)
    set(MATX_ENABLE_BLIS OFF CACHE BOOL "" FORCE)
    set(MATX_ENABLE_LIBFLAME OFF CACHE BOOL "" FORCE)
    set(MATX_ENABLE_MKL OFF CACHE BOOL "" FORCE)
  else()
    set(MATX_ENABLE_OPENBLAS ON CACHE BOOL "Enable OpenBLAS backend" FORCE)
    set(MATX_ENABLE_BLIS OFF CACHE BOOL "" FORCE)
    set(MATX_ENABLE_LIBFLAME OFF CACHE BOOL "" FORCE)
    set(MATX_ENABLE_MKL OFF CACHE BOOL "" FORCE)
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

if(MATX_ENABLE_MKL)
  list(APPEND BLAS_LIST "MKL")
endif()

list(LENGTH BLAS_LIST BLAS_COUNT)

if(BLAS_COUNT EQUAL 0)
  message(FATAL_ERROR
    "No BLAS backend enabled. Enable one of: OpenBLAS, BLIS, MKL.")
elseif(BLAS_COUNT GREATER 1)
  message(FATAL_ERROR
    "Multiple BLAS backends enabled (${BLAS_LIST}). Only one BLAS backend is allowed.")
endif()

# ---- BLAS backend ----
if(MATX_ENABLE_OPENBLAS)
  set(openblas_DIR
      "${CMAKE_CURRENT_SOURCE_DIR}/3party/openblas-0.3.30/cmake")
  find_package(openblas REQUIRED)
elseif(MATX_ENABLE_BLIS)
  set(amdblis_DIR
      "${CMAKE_CURRENT_SOURCE_DIR}/3party/aocl-5.2.0/AOCL-Windows/amd-blis/cmake")
  find_package(amdblis REQUIRED)
elseif(MATX_ENABLE_MKL)
  set(mkl_DIR
      "${CMAKE_CURRENT_SOURCE_DIR}/3party/mkl-2025.3.0/cmake")
  find_package(mkl REQUIRED)
endif()

# ---- LAPACK backend ----
if(MATX_ENABLE_LIBFLAME)
  set(libflame_DIR
      "${CMAKE_CURRENT_SOURCE_DIR}/3party/aocl-5.2.0/AOCL-Windows/amd-libflame/cmake")
  find_package(libflame REQUIRED)
endif()

# ---- Sparse / Graph libraries ----
if(MATX_ENABLE_SUITESPARSE)
  set(suitesparse_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3party/suitesparse-v7.11.0/cmake")
  find_package(suitesparse REQUIRED)
endif()

if(MATX_ENABLE_GRAPHBLAS)
  set(graphblas_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3party/suitesparse-v7.11.0/cmake")
  find_package(graphblas REQUIRED)
endif()

# ---- Include MatX submodules ----
include(GNUInstallDirs)
add_subdirectory(tools)
add_subdirectory(core)
add_subdirectory(dense_blas)
add_subdirectory(sparse_blas)
add_subdirectory(dense_solve)
add_subdirectory(sparse_solve)

# ---- Unit tests ----
if(MATX_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# ---- Export targets ----
install(EXPORT MatXTargets
  NAMESPACE MatX::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MatX
)