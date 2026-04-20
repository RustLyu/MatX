project (matx_io C CXX)

add_library(${PROJECT_NAME}
  src/matx_print.cc
  src/matx_read.cc
)

add_library(MatX::io ALIAS ${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME}
  PRIVATE
    MatX::types
    MatX::core
    MatX::tools
)


target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_sources(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/matx/matx_print.h>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/matx/matx_read.h>
)

target_compile_definitions(${PROJECT_NAME} PUBLIC MATX_CORE_BUILD=1)

if(WIN32)
  target_compile_definitions(${PROJECT_NAME} PUBLIC MATX_PLATFORM_WINDOWS=1)
else()
  target_compile_definitions(${PROJECT_NAME} PUBLIC MATX_PLATFORM_POSIX=1)
endif()

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS ${PROJECT_NAME}
  EXPORT MatXConfig
)
