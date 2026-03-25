project (matx_sparse_solve C)

add_library(${PROJECT_NAME}
  src/matx_solve_backend.c
  src/matx_suitesparse_stub.c
  src/tools.c
)

add_library(MatX::sparse_solve ALIAS ${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME}
  PRIVATE
    MatX::types
    MatX::tools
)

target_link_libraries(${PROJECT_NAME}
  PUBLIC MatX::core
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_definitions(${PROJECT_NAME}
  PUBLIC MATX_SOLVE_BUILD=1
  PRIVATE
    MATX_ENABLE_SUITESPARSE=$<BOOL:${MATX_ENABLE_SUITESPARSE}>
)

if(MATX_ENABLE_SUITESPARSE)
  target_compile_definitions(${PROJECT_NAME} PRIVATE MATX_HAVE_SUITESPARSE=1)
  target_link_libraries(${PROJECT_NAME} PRIVATE SuiteSparse::KLU)
endif()

target_sources(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/matx/matx_sparse_solve.h>
    $<INSTALL_INTERFACE:include/matx/matx_sparse_solve.h>
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS ${PROJECT_NAME}
  EXPORT MatXConfig
)