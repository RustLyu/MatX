project (matx_sparse_blas C)

add_library(${PROJECT_NAME}
  src/matx_sparse_blas_backend.c
  src/matx_graphblas_ref.c
  src/matx_mkl_ref.c
  src/matx_aocl_cparse_ref.c
  src/tools.c
)

add_library(MatX::sparse_blas ALIAS ${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME}
  PUBLIC 
    MatX::core
    MatX::tools
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_definitions(${PROJECT_NAME}
  PUBLIC MATX_COMPUTE_BUILD=1
  PRIVATE
    MATX_ENABLE_MKL=$<BOOL:${MATX_ENABLE_MKL}>
    MATX_ENABLE_GRAPHBLAS=$<BOOL:${MATX_ENABLE_GRAPHBLAS}>
)

if(MATX_ENABLE_MKL)
    target_include_directories(${PROJECT_NAME}
        PUBLIC
            $<BUILD_INTERFACE:${mkl_INCLUDE_DIR}>
    )
endif()

if(MATX_SPARSE_BLAS_BACKEND STREQUAL "REFERENCE")
  target_compile_definitions(${PROJECT_NAME} PRIVATE MATX_SPARSE_BLAS_BACKEND_REFERENCE_ONLY=1)
endif()

if(MATX_ENABLE_MKL)
  target_compile_definitions(${PROJECT_NAME} PRIVATE MATX_HAVE_MKL=1)
  target_link_libraries(${PROJECT_NAME} PRIVATE ${mkl_lib})
endif()

if(MATX_ENABLE_GRAPHBLAS)
    target_compile_definitions(${PROJECT_NAME} PRIVATE MATX_HAVE_GRAPHBLAS=1)
    target_link_libraries(${PROJECT_NAME} PRIVATE SuiteSparse::GraphBLAS)
endif()

if(MATX_ENABLE_AOCL_SPARSE)
    target_compile_definitions(${PROJECT_NAME} PRIVATE MATX_HAVE_AOCL_SPARSE=1)
    target_link_libraries(${PROJECT_NAME} PRIVATE aoclsparse::aoclsparse)
endif()

target_sources(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/matx/matx_sparse_compute.h>
    $<INSTALL_INTERFACE:include/matx/matx_sparse_compute.h>
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS ${PROJECT_NAME}
  EXPORT MatXTargets
)
