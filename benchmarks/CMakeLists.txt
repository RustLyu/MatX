project(matx_benchmarks CXX)

if(NOT MATX_ENABLE_GRAPHBLAS)
  message(FATAL_ERROR "MATX_BUILD_BENCHMARKS requires MATX_ENABLE_GRAPHBLAS=ON")
endif()

find_package(OpenMP REQUIRED)

add_executable(matx_benchmarks matx_benchmarks.cpp)
target_compile_definitions(matx_benchmarks
  PRIVATE
    MATX_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
    MATX_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
)
target_link_libraries(matx_benchmarks
  PRIVATE
    MatX::types
    MatX::core
    MatX::io
    MatX::tools
    MatX::vec_blas
    MatX::dense_blas
    MatX::sparse_blas
    MatX::dense_solve
    MatX::sparse_solve
    OpenMP::OpenMP_CXX
)
