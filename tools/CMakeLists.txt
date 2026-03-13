project (matx_toools C CXX)

add_library(${PROJECT_NAME}
  src/matx_log.cc
)

add_library(MatX::tools ALIAS ${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME}
  PRIVATE
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
    $<INSTALL_INTERFACE:include/matx/matx_log.h>
)

target_compile_definitions(${PROJECT_NAME} PUBLIC MATX_CORE_BUILD=1)

if(WIN32)
  target_compile_definitions(${PROJECT_NAME} PUBLIC MATX_PLATFORM_WINDOWS=1)
else()
  target_compile_definitions(${PROJECT_NAME} PUBLIC MATX_PLATFORM_POSIX=1)
endif()

message(STATUS ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS ${PROJECT_NAME}
  EXPORT MatXTargets
)
