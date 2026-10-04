cmake_minimum_required(VERSION 3.20)

project(MatX LANGUAGES C CXX)

include(GNUInstallDirs)

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
