include(${CMAKE_CURRENT_LIST_DIR}/../cmake/MatXDeps.cmake)

add_library(matx_solve
  src/matx_solve_backend.c
  src/matx_suitesparse_stub.c
)

add_library(MatX::solve ALIAS matx_solve)

target_link_libraries(matx_solve
  PUBLIC matx_options MatX::core
)

target_include_directories(matx_solve
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_definitions(matx_solve
  PUBLIC MATX_SOLVE_BUILD=1
  PRIVATE
    MATX_SUITESPARSE_FOUND=$<BOOL:${MATX_SUITESPARSE_FOUND}>
    MATX_OPENBLAS_FOUND=$<BOOL:${MATX_OPENBLAS_FOUND}>
    MATX_BLIS_FOUND=$<BOOL:${MATX_BLIS_FOUND}>
)

if(MATX_SUITESPARSE_FOUND)
  target_compile_definitions(matx_solve PRIVATE MATX_HAVE_SUITESPARSE=1)
  target_link_libraries(matx_solve PRIVATE SuiteSparse::SuiteSparse)
endif()

if(MATX_OPENBLAS_FOUND)
  target_compile_definitions(matx_solve PRIVATE MATX_HAVE_OPENBLAS=1)
  target_link_libraries(matx_solve PRIVATE OpenBLAS::OpenBLAS)
endif()

if(MATX_BLIS_FOUND)
  target_compile_definitions(matx_solve PRIVATE MATX_HAVE_BLIS=1)
  target_link_libraries(matx_solve PRIVATE BLIS::BLIS)
endif()

target_sources(matx_solve
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/matx/matx_solve.h
)

