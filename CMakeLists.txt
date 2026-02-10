cmake_minimum_required(VERSION 3.20)

project(MatX LANGUAGES C CXX)

# ---- Global options ----
option(MATX_BUILD_TESTS "Build unit tests (gtest)" ON)
option(MATX_ENABLE_OPENBLAS "Enable OpenBLAS backend if found/provided" ON)
option(MATX_ENABLE_BLIS "Enable BLIS backend if found/provided" ON)
option(MATX_ENABLE_SUITESPARSE "Enable SuiteSparse backend if found/provided" ON)
option(MATX_ENABLE_CXSPARSE "Enable CXSparse for sparse numerical ops if found" ON)
option(MATX_ENABLE_GRAPHBLAS "Enable GraphBlas for sparse numerical ops if found" ON)
set(MATX_BLAS_BACKEND "AUTO" CACHE STRING "BLAS backend: AUTO|OPENBLAS|BLIS|REFERENCE")

set_property(CACHE MATX_BLAS_BACKEND PROPERTY STRINGS AUTO OPENBLAS BLIS REFERENCE)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

#add_library(matx_options INTERFACE)
#target_compile_definitions(matx_options INTERFACE
#  $<$<CONFIG:Debug>:MATX_DEBUG=1>
#)
#if(MSVC)
#  target_compile_options(matx_options INTERFACE /W4 /permissive-)
#else()
#  target_compile_options(matx_options INTERFACE -Wall -Wextra -Wpedantic)
#endif()

#include(GNUInstallDirs)

add_subdirectory(core)
add_subdirectory(blas)
add_subdirectory(solve)

if(MATX_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

install(EXPORT MatXTargets
  NAMESPACE MatX::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MatX
)