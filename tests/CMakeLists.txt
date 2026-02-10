set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(INSTALL_GMOCK OFF CACHE BOOL "" FORCE)

include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)

# gtest has its own CRT flags; keep default on MSVC.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(matx_tests
  test_core_dense.cpp
  test_core_vec.cpp
  test_compute_dgemm.cpp
  test_compute_vec.cpp
  test_compute_dense.cpp
  test_compute_sparse.cpp
  test_solve.cpp
)

target_link_libraries(matx_tests
  PRIVATE
    GTest::gtest_main
    MatX::core
    MatX::blas
    MatX::solve
)

include(GoogleTest)
gtest_discover_tests(matx_tests)

