project (matx_types C)

add_library(${PROJECT_NAME}
	src/matx_version.c
)

add_library(MatX::types ALIAS ${PROJECT_NAME})


target_link_libraries(${PROJECT_NAME}
  PRIVATE
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_sources(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/matx/matx_types.h>
    $<INSTALL_INTERFACE:include/matx/matx_types.h>
)

target_compile_definitions(${PROJECT_NAME} PUBLIC MATX_CORE_BUILD=1)

if(WIN32)
  target_compile_definitions(${PROJECT_NAME} PUBLIC MATX_PLATFORM_WINDOWS=1)
else()
  target_compile_definitions(${PROJECT_NAME} PUBLIC MATX_PLATFORM_POSIX=1)
endif()

if(BUILD_SHARED_LIBS)
    target_compile_definitions(${PROJECT_NAME} PUBLIC MATX_BUILD_SHARED)
endif()

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS ${PROJECT_NAME}
  EXPORT MatXConfig
)
