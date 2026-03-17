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
  test_core_dense.cpp
  test_core_vec.cpp
  test_compute_dgemm.cpp
  test_compute_vec.cpp
  test_compute_dense.cpp
  test_compute_sparse.cpp
  test_solve.cpp
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
)

if(MATX_ENABLE_OPENBLAS)
  target_link_libraries(${PROJECT_NAME} PRIVATE  ${openblas_lib})
endif()

if(MATX_ENABLE_BLIS)
  target_link_libraries(${PROJECT_NAME} PRIVATE  ${amdblis_lib})
endif()

if(MATX_ENABLE_LIBFLAME)
  target_link_libraries(${PROJECT_NAME} PRIVATE ${libflame_lib})
endif()

if(MATX_ENABLE_MKL)
  target_link_libraries(${PROJECT_NAME} PRIVATE ${mkl_lib})
endif()

if(MATX_ENABLE_SUITESPARSE)
  target_link_libraries(${PROJECT_NAME} PRIVATE ${suitesparse_lib})
endif()

if(MATX_ENABLE_GRAPHBLAS)
  target_link_libraries(${PROJECT_NAME} PRIVATE ${graphblas_lib})
endif()

include(GoogleTest)
gtest_discover_tests(matx_tests)

