cmake_minimum_required(VERSION 3.20)

project(MatX LANGUAGES C CXX)

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

include(GNUInstallDirs)

# Keep a default project installation local to the build tree. Users can still
# pass -DCMAKE_INSTALL_PREFIX=... when an external install is intended.
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH
      "Install path prefix, prepended onto install directories" FORCE)
endif()

include(ProcessorCount)
ProcessorCount(NPROC)
if(NPROC GREATER 0)
    math(EXPR BUILD_JOBS "${NPROC} / 2")
else()
    set(BUILD_JOBS 4)
endif()

# Add cmake/ module path so find_package() can discover aocl_cmake-style configs
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# ---- Platform & language standards ----
include(MatXPlatform)

# ---- CPU vendor auto-detection ----
include(MatXCPUDetect)

# ---- Build options (backend, feature toggles) ----
include(MatXOptions)

# ---- BLAS backend validation + install prefix setup ----
include(MatXBackendValidate)

# ---- Third-party dependencies (order matters: later deps reference earlier variables) ----
include(DepSpdlog)
include(DepOpenBLAS)
include(DepAOCL)
include(DepSuiteSparse)
include(DepSuperLU)
include(DepMUMPS)
include(DepGinkgo)
include(DepAGMG)
include(DepPETSc)
include(DepHYPRE)

# ---- MatX library subdirectories ----
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

option(MATX_BUILD_BENCHMARKS "Build MatX performance benchmarks" OFF)
if(MATX_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

option(MATX_BUILD_ITERATIVE_BENCHMARKS
  "Build the optional iterative and direct sparse solver comparison benchmark" OFF)
if(MATX_BUILD_ITERATIVE_BENCHMARKS)
  if(NOT MATX_ENABLE_GINKGO AND NOT MATX_ENABLE_PETSC AND NOT MATX_ENABLE_HYPRE)
    message(FATAL_ERROR
      "MATX_BUILD_ITERATIVE_BENCHMARKS requires at least one iterative backend")
  endif()
  add_executable(matx_sparse_iterative_benchmarks
    benchmarks/sparse_iterative_benchmarks.cpp)
  target_link_libraries(matx_sparse_iterative_benchmarks
    PRIVATE MatX::sparse_solve MatX::core)
endif()

# ---- Export targets ----
include(CMakePackageConfigHelpers)

install(EXPORT MatXConfig
  FILE MatXTargets.cmake
  NAMESPACE MatX::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MatX
)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/MatXConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/MatXConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MatX
)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/MatXConfig.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MatX
)

# Stage dependencies built from upstream sources so an installed MatX package
# can be consumed without access to this build tree.
if(MATX_ENABLE_OPENBLAS)
  install(DIRECTORY "${DEPEND_LIB_OUTPUT}/openblas/include/"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(DIRECTORY "${DEPEND_LIB_OUTPUT}/openblas/lib/"
    DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

if(MATX_ENABLE_SUITESPARSE)
  install(DIRECTORY "${DEPEND_LIB_OUTPUT}/suitesparse/include/"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(DIRECTORY "${DEPEND_LIB_OUTPUT}/suitesparse/lib/"
    DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

if(MATX_ENABLE_SUPERLU)
  install(DIRECTORY "${DEPEND_LIB_OUTPUT}/superlu_lib/include/"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(DIRECTORY "${DEPEND_LIB_OUTPUT}/superlu_lib/lib/"
    DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

if(MATX_ENABLE_MUMPS)
  install(DIRECTORY "${DEPEND_LIB_OUTPUT}/mumps/include/"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(DIRECTORY "${DEPEND_LIB_OUTPUT}/mumps/lib/"
    DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

install(DIRECTORY "${DEPEND_LIB_OUTPUT}/spdlog/include/"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY "${DEPEND_LIB_OUTPUT}/spdlog/lib/"
  DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(MATX_ENABLE_PETSC AND MATX_BUILD_PETSC_FROM_SOURCE)
  install(DIRECTORY "${MATX_PETSC_INSTALL_DIR}/include/"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/petsc)
  install(DIRECTORY "${MATX_PETSC_INSTALL_DIR}/lib/"
    DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/generate_dir/include/config.h
    @ONLY
)
