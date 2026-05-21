project (matx_tools C CXX)

add_library(${PROJECT_NAME}
  src/matx_log.cc
  src/matx_tm.cc
)

add_library(MatX::tools ALIAS ${PROJECT_NAME})
#
#add_library(spdlog::spdlog INTERFACE IMPORTED GLOBAL)
#target_include_directories(spdlog::spdlog INTERFACE
#    ${spdlog_SOURCE_DIR}/include
#)

target_link_libraries(${PROJECT_NAME}
  PRIVATE
    MatX::types
    spdlog::spdlog
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_sources(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/matx/matx_log.h>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/matx/matx_tm.h>
    $<INSTALL_INTERFACE:include/matx/matx_log.h>
    $<INSTALL_INTERFACE:include/matx/matx_tm.h>
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
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

install(EXPORT MatXConfig
  NAMESPACE MatX::
  DESTINATION lib/cmake/MatX
)
