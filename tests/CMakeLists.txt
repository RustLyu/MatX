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
  test_sparse_conversion.cpp
  test_extended_numeric.cpp
  test_print_mtx.cpp
  test_read_mtx.cpp
  test_write_mtx.cpp
  test_csv.cpp
)
find_package(OpenMP REQUIRED)
target_link_libraries(${PROJECT_NAME}
  PRIVATE

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

if(WIN32)
  # Collect dependency and toolchain runtime DLLs at configure time.
  # Uses file(GLOB) so DLL names can vary across versions/toolchains.
  set(DLL_GLOB_PATTERNS "")

  # OpenBLAS
  file(GLOB openblas_dlls "${DEPEND_LIB_OUTPUT}/openblas/bin/*.dll")
  list(APPEND DLL_FILES ${openblas_dlls})

  # SuiteSparse
  file(GLOB suitesparse_dlls "${DEPEND_LIB_OUTPUT}/suitesparse/bin/*.dll")
  list(APPEND DLL_FILES ${suitesparse_dlls})

  # SuperLU (optional)
  if(MATX_ENABLE_SUPERLU)
    file(GLOB superlu_dlls "${DEPEND_LIB_OUTPUT}/superlu_lib/bin/*.dll")
    list(APPEND DLL_FILES ${superlu_dlls})
  endif()

  # MUMPS (optional)
  if(MATX_ENABLE_MUMPS)
    file(GLOB mumps_dlls "${DEPEND_LIB_OUTPUT}/mumps/bin/*.dll")
    list(APPEND DLL_FILES ${mumps_dlls})
  endif()

  # MinGW toolchain runtime DLLs — derived from compiler location, not
  # hardcoded to a specific MSYS2 installation path.
  get_filename_component(MINGW_BIN_DIR "${CMAKE_C_COMPILER}" DIRECTORY)
  file(GLOB mingw_runtime_dlls
    "${MINGW_BIN_DIR}/libgcc_s_seh-*.dll"
    "${MINGW_BIN_DIR}/libstdc++-*.dll"
    "${MINGW_BIN_DIR}/libwinpthread-*.dll"
    "${MINGW_BIN_DIR}/libgfortran-*.dll"
    "${MINGW_BIN_DIR}/libgomp-*.dll"
  )
  list(APPEND DLL_FILES ${mingw_runtime_dlls})

  # Build the add_custom_command argument list
  set(DLL_COPY_COMMANDS "")
  foreach(dll IN LISTS DLL_FILES)
    list(APPEND DLL_COPY_COMMANDS
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${dll}"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    )
  endforeach()

  add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    ${DLL_COPY_COMMANDS}
  )
endif()

include(CTest)
add_test(NAME matx_tests COMMAND matx_tests)
