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
  test_core_math.cpp
  test_core_sparse.cpp
  test_compute_dgemm.cpp
  test_compute_vec.cpp
  test_compute_vec_blas.cpp
  test_compute_dense.cpp
  test_compute_sparse.cpp
  test_solve.cpp
  test_print_mtx.cpp
  test_read_mtx.cpp
)
find_package(OpenMP REQUIRED)
target_link_libraries(${PROJECT_NAME}
  PRIVATE

    GTest::gtest_main

    MatX::types
    MatX::core
    MatX::vec_blas
    MatX::dense_blas
    MatX::sparse_blas
    MatX::dense_solve
    MatX::sparse_solve
    MatX::tools
    MatX::io

    OpenMP::OpenMP_C
)

include(GoogleTest)
if(WIN32)
  # Copy runtime DLLs so gtest_discover_tests can run during build
  add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${DEPEND_LIB_OUTPUT}/openblas/bin/libopenblas_64.dll"
      "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${DEPEND_LIB_OUTPUT}/suitesparse/bin/libsuitesparseconfig.dll"
      "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${DEPEND_LIB_OUTPUT}/suitesparse/bin/libgraphblas.dll"
      "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${DEPEND_LIB_OUTPUT}/suitesparse/bin/libklu.dll"
      "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${DEPEND_LIB_OUTPUT}/suitesparse/bin/libklu_cholmod.dll"
      "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${DEPEND_LIB_OUTPUT}/suitesparse/bin/libumfpack.dll"
      "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${DEPEND_LIB_OUTPUT}/suitesparse/bin/libcxsparse.dll"
      "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${DEPEND_LIB_OUTPUT}/suitesparse/bin/libamd.dll"
      "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${DEPEND_LIB_OUTPUT}/suitesparse/bin/libbtf.dll"
      "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${DEPEND_LIB_OUTPUT}/suitesparse/bin/libcamd.dll"
      "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${DEPEND_LIB_OUTPUT}/suitesparse/bin/libccolamd.dll"
      "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${DEPEND_LIB_OUTPUT}/suitesparse/bin/libcholmod.dll"
      "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${DEPEND_LIB_OUTPUT}/suitesparse/bin/libcolamd.dll"
      "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${DEPEND_LIB_OUTPUT}/superlu_lib/bin/libsuperlu.dll"
      "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${DEPEND_LIB_OUTPUT}/mumps/bin/libdmumps.dll"
      "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${DEPEND_LIB_OUTPUT}/mumps/bin/libzmumps.dll"
      "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${DEPEND_LIB_OUTPUT}/mumps/bin/libmumps_common.dll"
      "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${DEPEND_LIB_OUTPUT}/mumps/bin/libmpiseq_fortran.dll"
      "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${DEPEND_LIB_OUTPUT}/mumps/bin/libmpiseq_c.dll"
      "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${DEPEND_LIB_OUTPUT}/mumps/bin/libpord.dll"
      "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${DEPEND_LIB_OUTPUT}/mumps/bin/libcmumps.dll"
      "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${DEPEND_LIB_OUTPUT}/mumps/bin/libsmumps.dll"
      "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${DEPEND_LIB_OUTPUT}/mumps/bin/libmpiseq.dll"
      "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "C:/msys64/ucrt64/bin/libgcc_s_seh-1.dll"
      "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "C:/msys64/ucrt64/bin/libstdc++-6.dll"
      "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "C:/msys64/ucrt64/bin/libwinpthread-1.dll"
      "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "C:/msys64/ucrt64/bin/libgfortran-5.dll"
      "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "C:/msys64/ucrt64/bin/libgomp-1.dll"
      "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
  )
endif()
gtest_discover_tests(matx_tests)
