include(${CMAKE_CURRENT_LIST_DIR}/../cmake/MatXDeps.cmake)

add_library(matx_compute
  src/matx_blas_backend.c
  src/matx_blas_ref.c
  src/sparse_f64_ops.c
  src/sparse_c64_ops.c
)

add_library(MatX::compute ALIAS matx_compute)

target_link_libraries(matx_compute
  PUBLIC matx_options MatX::core
)

target_include_directories(matx_compute
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_definitions(matx_compute
  PUBLIC MATX_COMPUTE_BUILD=1
  PRIVATE
    MATX_OPENBLAS_FOUND=$<BOOL:${MATX_OPENBLAS_FOUND}>
    MATX_BLIS_FOUND=$<BOOL:${MATX_BLIS_FOUND}>
)

if(MATX_BLAS_BACKEND STREQUAL "REFERENCE")
  target_compile_definitions(matx_compute PRIVATE MATX_BLAS_BACKEND_REFERENCE_ONLY=1)
endif()

if(MATX_OPENBLAS_FOUND)
  target_compile_definitions(matx_compute PRIVATE MATX_HAVE_OPENBLAS=1)
  target_link_libraries(matx_compute PRIVATE OpenBLAS::OpenBLAS)
endif()

if(MATX_BLIS_FOUND)
  target_compile_definitions(matx_compute PRIVATE MATX_HAVE_BLIS=1)
  target_link_libraries(matx_compute PRIVATE BLIS::BLIS)
endif()

if(MATX_CXSPARSE_FOUND)
  target_compile_definitions(matx_compute PRIVATE MATX_HAVE_CXSPARSE=1)
  target_link_libraries(matx_compute PRIVATE CXSparse::CXSparse)
endif()

if(MATX_GRAPHBLAS_FOUND)
  target_compile_definitions(matx_compute PRIVATE MATX_HAVE_GRAPHBLAS=1)
  target_link_libraries(matx_compute PRIVATE GraphBlas::GraphBlas)
endif()

target_sources(matx_compute
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/matx/matx_compute.h
)

