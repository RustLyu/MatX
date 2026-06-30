project (matx_core C)

add_library(${PROJECT_NAME}
  src/matx_alloc.c
  src/matx_dense_d_i8.c
  src/matx_error.c
  src/matx_vec.c
  src/matx_dense_z_i8.c
  src/matx_sparse_d_i8.c
  src/matx_sparse_z_i8.c
)

add_library(MatX::core ALIAS ${PROJECT_NAME})


target_link_libraries(${PROJECT_NAME}
  PRIVATE
    MatX::types
    MatX::tools
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../generate_dir/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_sources(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/matx/matx_func.h>
    $<INSTALL_INTERFACE:include/matx/matx_func.h>
)

target_compile_definitions(${PROJECT_NAME} PUBLIC MATX_CORE_BUILD=1)

if(WIN32)
  target_compile_definitions(${PROJECT_NAME} PUBLIC MATX_PLATFORM_WINDOWS=1)
else()
  target_compile_definitions(${PROJECT_NAME} PUBLIC MATX_PLATFORM_POSIX=1)
endif()

install(DIRECTORY 
        ${CMAKE_CURRENT_SOURCE_DIR}/include/
        ${CMAKE_CURRENT_BINARY_DIR}/../generate_dir/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS ${PROJECT_NAME}
  EXPORT MatXConfig
)
