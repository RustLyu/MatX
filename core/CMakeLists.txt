add_library(matx_core
  src/matx_alloc.c
  src/matx_dense_f64.c
  src/matx_error.c
  src/matx_version.c
  src/matx_vec.c
  src/matx_dense_c64.c
  src/matx_sparse_f64.c
  src/matx_sparse_c64.c
)

add_library(MatX::core ALIAS matx_core)

#target_link_libraries(matx_core PUBLIC )

target_include_directories(matx_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_sources(matx_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/matx/matx.h>
    $<INSTALL_INTERFACE:include/matx/matx.h>
)

target_compile_definitions(matx_core PUBLIC MATX_CORE_BUILD=1)

if(WIN32)
  target_compile_definitions(matx_core PUBLIC MATX_PLATFORM_WINDOWS=1)
else()
  target_compile_definitions(matx_core PUBLIC MATX_PLATFORM_POSIX=1)
endif()

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS matx_core
  EXPORT MatXTargets
)
