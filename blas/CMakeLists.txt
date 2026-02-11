include(${CMAKE_CURRENT_LIST_DIR}/../cmake/MatXDeps.cmake)

add_library(matx_blas
  src/matx_blas_backend.c
  src/matx_blas_ref.c
  src/matx_graphblas_backend.c
  src/matx_graphblas_ref.c
  src/tools.c
)

add_library(MatX::blas ALIAS matx_blas)

target_link_libraries(matx_blas
  PUBLIC MatX::core
)

target_include_directories(matx_blas
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_definitions(matx_blas
  PUBLIC MATX_COMPUTE_BUILD=1
  PRIVATE
    MATX_ENABLE_OPENBLAS=$<BOOL:${MATX_ENABLE_OPENBLAS}>
    MATX_ENABLE_BLIS=$<BOOL:${MATX_ENABLE_BLIS}>
)

if(MATX_BLAS_BACKEND STREQUAL "REFERENCE")
  target_compile_definitions(matx_blas PRIVATE MATX_BLAS_BACKEND_REFERENCE_ONLY=1)
endif()

if(MATX_ENABLE_OPENBLAS)
  target_compile_definitions(matx_blas PRIVATE MATX_HAVE_OPENBLAS=1)
  target_link_libraries(matx_blas PRIVATE OpenBLAS::OpenBLAS)
endif()

if(MATX_ENABLE_BLIS)
  target_compile_definitions(matx_blas PRIVATE MATX_HAVE_BLIS=1)
  target_link_libraries(matx_blas PRIVATE BLIS::BLIS)
endif()

if(MATX_CXSPARSE_FOUND)
  target_compile_definitions(matx_blas PRIVATE MATX_HAVE_CXSPARSE=1)
  target_link_libraries(matx_blas PRIVATE CXSparse::CXSparse)
endif()

if(MATX_GRAPHBLAS_FOUND)
  target_compile_definitions(matx_blas PRIVATE MATX_HAVE_GRAPHBLAS=1)
  target_link_libraries(matx_blas PRIVATE GraphBlas::GraphBlas)
endif()

target_sources(matx_blas
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/matx/matx_compute.h>
    $<INSTALL_INTERFACE:include/matx/matx_compute.h>
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS matx_blas
  EXPORT MatXTargets
)
