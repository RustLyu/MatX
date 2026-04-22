set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(INSTALL_GMOCK OFF CACHE BOOL "" FORCE)

include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)

set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(INSTALL_GMOCK OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(BUILD_GTEST ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)
project (matx_tests CXX)

add_executable(${PROJECT_NAME}
  main.cpp
  test_core_dense.cpp
  test_core_vec.cpp
  test_compute_dgemm.cpp
  test_compute_vec.cpp
  test_compute_dense.cpp
  test_compute_sparse.cpp
  test_solve.cpp
  test_print_mtx.cpp
  test_read_mtx.cpp
)

target_link_libraries(${PROJECT_NAME}
  PRIVATE
    GTest::gtest_main
    MatX::types
    MatX::core
    MatX::dense_blas
    MatX::sparse_blas
    MatX::dense_solve
    MatX::sparse_solve
    MatX::tools
    MatX::io
)

include(GoogleTest)
gtest_discover_tests(matx_tests)
