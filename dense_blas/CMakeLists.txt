project (matx_dense_blas C)

add_library(${PROJECT_NAME}
  src/matx_blas_backend.c
  src/matx_blas_ref.c
)

add_library(MatX::dense_blas ALIAS ${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME}
  PUBLIC 
  MatX::types
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
    MATX_ENABLE_OPENBLAS=$<BOOL:${MATX_ENABLE_OPENBLAS}>
    MATX_ENABLE_BLIS=$<BOOL:${MATX_ENABLE_BLIS}>
    MATX_ENABLE_LIBFLAME=$<BOOL:${MATX_ENABLE_LIBFLAME}>
)

if(MATX_DENSE_BLAS_BACKEND STREQUAL "REFERENCE")
  target_compile_definitions(${PROJECT_NAME} PRIVATE MATX_DENSE_BLAS_BACKEND_REFERENCE_ONLY=1)
endif()

if(MATX_ENABLE_BLIS)
  target_compile_definitions(${PROJECT_NAME} PRIVATE MATX_HAVE_BLIS=1)
  target_link_libraries(${PROJECT_NAME} PRIVATE amdblis::amdblis)
  if(LINUX)
    target_link_libraries(${PROJECT_NAME} PRIVATE aoclutils::aoclutils)
  endif()
endif()

if(MATX_ENABLE_OPENBLAS)
  target_compile_definitions(${PROJECT_NAME} PRIVATE MATX_HAVE_OPENBLAS=1)
  target_link_libraries(${PROJECT_NAME} PRIVATE OpenBLAS::OpenBLAS)
endif()

if(MATX_ENABLE_LIBFLAME)
  target_compile_definitions(${PROJECT_NAME} PRIVATE MATX_HAVE_LIBFLAME=1)
  target_link_libraries(${PROJECT_NAME} PRIVATE libflame::libflame)
endif()

target_sources(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/matx/matx_dense_compute.h>
    $<INSTALL_INTERFACE:include/matx/matx_dense_compute.h>
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS ${PROJECT_NAME}
  EXPORT MatXConfig
)
