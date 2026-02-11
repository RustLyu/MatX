add_library(matx_blas
  src/matx_blas_backend.c
  src/matx_blas_ref.c
  src/matx_sparse_blas_backend.c
  src/matx_graphblas_ref.c
  src/matx_mkl_ref.c
  src/tools.c
)

add_library(MatX::blas ALIAS matx_blas)

target_link_libraries(matx_blas
  PUBLIC MatX::core
)

target_include_directories(matx_blas
  PUBLIC
    $<BUILD_INTERFACE:${openblas_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_definitions(matx_blas
  PUBLIC MATX_COMPUTE_BUILD=1
  PRIVATE
    MATX_ENABLE_OPENBLAS=$<BOOL:${MATX_ENABLE_OPENBLAS}>
    MATX_ENABLE_BLIS=$<BOOL:${MATX_ENABLE_BLIS}>
    MATX_ENABLE_MKL=$<BOOL:${MATX_ENABLE_MKL}>
    MATX_ENABLE_GRAPHBLAS=$<BOOL:${MATX_ENABLE_GRAPHBLAS}>
)

if(MATX_BLAS_BACKEND STREQUAL "REFERENCE")
  target_compile_definitions(matx_blas PRIVATE MATX_BLAS_BACKEND_REFERENCE_ONLY=1)
endif()

if(MATX_ENABLE_OPENBLAS)
  target_compile_definitions(matx_blas PRIVATE MATX_HAVE_OPENBLAS=1)
  target_link_libraries(matx_blas PRIVATE ${openblas_lib})
endif()

if(MATX_ENABLE_BLIS)
  target_compile_definitions(matx_blas PRIVATE MATX_HAVE_BLIS=1)
  target_link_libraries(matx_blas PRIVATE ${amdblis_lib})
endif()

if(MATX_ENABLE_MKL)
  target_compile_definitions(matx_blas PRIVATE MATX_HAVE_MKL=1)
  target_link_libraries(matx_blas PRIVATE ${mkl_lib})
endif()

if(MATX_ENABLE_GRAPHBLAS)
    target_compile_definitions(matx_blas PRIVATE MATX_HAVE_GRAPHBLAS=1)
    target_link_libraries(matx_blas PRIVATE ${graphblas_lib})
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
