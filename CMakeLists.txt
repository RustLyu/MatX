cmake_minimum_required(VERSION 3.20)
project(MatX LANGUAGES C CXX)

if (WIN32)
    message(STATUS "current os：Windows")
    set(3Party_path "${CMAKE_CURRENT_SOURCE_DIR}/3party/win32")
elseif (LINUX)
    message(STATUS "current os：Linux")
    set(3Party_path "${CMAKE_CURRENT_SOURCE_DIR}/3party/linux")
elseif (APPLE)
    message(STATUS "current os：macOS")
else ()
    message(WARNING "UNKNOWN OS！")
endif ()

message(STATUS "current 3Party path ${3Party_path}")

# ---- Global options ----
option(MATX_BUILD_TESTS "Build unit tests (gtest)" ON)

option(MATX_ENABLE_OPENBLAS "Enable OpenBLAS backend if found/provided" OFF)
option(MATX_ENABLE_BLIS "Enable BLIS backend if found/provided" OFF)
option(MATX_ENABLE_LIBFLAME "Enable LIBFLAME backend if found/provided" OFF)
option(MATX_ENABLE_MKL "Enable MKL backend if found/provided" OFF)
option(MATX_ENABLE_SUITESPARSE "Enable SuiteSparse backend if found/provided" ON)
option(MATX_ENABLE_GRAPHBLAS "Enable GraphBlas for sparse numerical ops if found" OFF)
option(MATX_ENABLE_AOCL_SPARSE "Enable amd AOCL SPARSE for sparse numerical ops if found" OFF)

option(BUILD_SHARED_LIBS "Build shared library (ON) or static (OFF)" OFF)

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
if (WIN32)
	set(spdlog_DIR "${3Party_path}/spdlog_1.17.0/lib/cmake/spdlog")
elseif (LINUX)
	set(spdlog_DIR "${3Party_path}/spdlog_1.17.0/lib64/cmake/spdlog")
endif()
message(STATUS "spdlog found at ${spdlog_DIR}")
find_package(spdlog REQUIRED)

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
    set(MATX_ENABLE_AOCL_SPARSE ON CACHE BOOL "" FORCE)
  elseif(CPU_PLATFORM STREQUAL "INTEL")
    set(MATX_ENABLE_OPENBLAS ON CACHE BOOL "Enable OpenBLAS backend" FORCE)
    set(MATX_ENABLE_GRAPHBLAS ON CACHE BOOL "Enable GraphBLAS" FORCE)
    set(MATX_ENABLE_BLIS OFF CACHE BOOL "" FORCE)
    set(MATX_ENABLE_LIBFLAME OFF CACHE BOOL "" FORCE)
    set(MATX_ENABLE_MKL OFF CACHE BOOL "" FORCE)
    set(MATX_ENABLE_AOCL_SPARSE ON CACHE BOOL "" FORCE)
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
   if (WIN32)
	  set(OpenBLAS_DIR
		  "${3Party_path}/openblas-0.3.30/lib/cmake/OpenBLAS")
		  message(STATUS "Looking for OpenBLAS in ${OpenBLAS_DIR}")
	  elseif (LINUX)
	  	  set(OpenBLAS_DIR
		  "${3Party_path}/openblas-0.3.30/lib64/cmake/OpenBLAS")
		  message(STATUS "Looking for OpenBLAS in ${OpenBLAS_DIR}")
	  endif()
  find_package(OpenBLAS REQUIRED)
elseif(MATX_ENABLE_BLIS)
	if (WIN32)
	  set(amdblis_DIR
		  "${3Party_path}/aocl-5.2.0/AOCL-Windows/amd-blis/cmake")
	elseif(LINUX)
	    set(amdblis_DIR
		      "${3Party_path}/aocl-5.2.0/lib_ILP64")
	endif()
  find_package(amdblis REQUIRED)
elseif(MATX_ENABLE_MKL)
  set(mkl_DIR
      "${3Party_path}/mkl-2025.3.0/cmake")
  find_package(mkl REQUIRED)
endif()

# ---- LAPACK backend ----
if(MATX_ENABLE_LIBFLAME)
	if (WIN32)
	  set(libflame_DIR
      "${3Party_path}/aocl-5.2.0/AOCL-Windows/amd-libflame/cmake")
	elseif(LINUX)
		set(libflame_DIR
			  "${3Party_path}/aocl-5.2.0/lib_ILP64")
	endif()

	  
  find_package(libflame REQUIRED)
endif()

# ---- Sparse / Graph libraries ----
if(LINUX)
    add_compile_options(-pthread)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread")
endif()

if(MATX_ENABLE_AOCL_SPARSE)

    if (WIN32)
      set(aoclsparse_DIR
            "${3Party_path}/aocl-5.2.0/AOCL-Windows/amd-sparse/cmake")
    elseif (LINUX)
      set(aoclsparse_DIR
            "${3Party_path}/aocl-5.2.0/lib_ILP64")
    endif()

  find_package(aoclsparse REQUIRED)
endif()

if(MATX_ENABLE_SUITESPARSE)

  if (WIN32)
			set(SuiteSparse_config_DIR "${3Party_path}/suitesparse-v7.11.0/lib/cmake/SuiteSparse_config")
			set(BTF_DIR "${3Party_path}/suitesparse-v7.11.0/lib/cmake/BTF")
			set(AMD_DIR "${3Party_path}/suitesparse-v7.11.0/lib/cmake/AMD")
			set(COLAMD_DIR "${3Party_path}/suitesparse-v7.11.0/lib/cmake/COLAMD")
			set(KLU_DIR "${3Party_path}/suitesparse-v7.11.0/lib/cmake/KLU")
  elseif (LINUX)
			set(SuiteSparse_config_DIR "${3Party_path}/suitesparse-v7.11.0/lib64/cmake/SuiteSparse_config")
			set(BTF_DIR "${3Party_path}/suitesparse-v7.11.0/lib64/cmake/BTF")
			set(AMD_DIR "${3Party_path}/suitesparse-v7.11.0/lib64/cmake/AMD")
			set(COLAMD_DIR "${3Party_path}/suitesparse-v7.11.0/lib64/cmake/COLAMD")
			set(KLU_DIR "${3Party_path}/suitesparse-v7.11.0/lib64/cmake/KLU")
	  endif()
  
  find_package(KLU REQUIRED)
endif()

if(MATX_ENABLE_GRAPHBLAS)
	if (WIN32)
	  set(GraphBLAS_DIR "${3Party_path}/suitesparse-v7.11.0/lib/cmake/GraphBLAS")
	elseif (LINUX)
	  set(GraphBLAS_DIR "${3Party_path}/suitesparse-v7.11.0/lib64/cmake/GraphBLAS")
	endif()
	find_package(GraphBLAS REQUIRED)
endif()

# ---- Include MatX submodules ----
include(GNUInstallDirs)
add_subdirectory(types)
add_subdirectory(tools)
add_subdirectory(core)
add_subdirectory(io)
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
install(EXPORT MatXConfig
  NAMESPACE MatX::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MatX
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/generate_dir/include/config.h
    @ONLY
)