add_library(matx_solve
  src/matx_solve_backend.c
  src/matx_suitesparse_stub.c
)

add_library(MatX::solve ALIAS matx_solve)

target_link_libraries(matx_solve
  PUBLIC MatX::core
)

target_include_directories(matx_solve
  PUBLIC
    $<BUILD_INTERFACE:${openblas_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_definitions(matx_solve
  PUBLIC MATX_SOLVE_BUILD=1
  PRIVATE
    MATX_ENABLE_SUITESPARSE=$<BOOL:${MATX_ENABLE_SUITESPARSE}>
    MATX_ENABLE_OPENBLAS=$<BOOL:${MATX_ENABLE_OPENBLAS}>
    MATX_ENABLE_LIBFLAME=$<BOOL:${MATX_ENABLE_LIBFLAME}>
    MATX_ENABLE_BLIS=$<BOOL:${MATX_ENABLE_BLIS}>
)

if(MATX_ENABLE_SUITESPARSE)
  target_compile_definitions(matx_solve PRIVATE MATX_HAVE_SUITESPARSE=1)
  target_link_libraries(matx_solve PRIVATE ${suitesparse_lib})
endif()

if(MATX_ENABLE_OPENBLAS)
  target_compile_definitions(matx_solve PRIVATE MATX_HAVE_OPENBLAS=1)
  message(STATUS "Linking OpenBLAS library: ${openblas_lib}")
  target_link_libraries(matx_solve PRIVATE ${openblas_lib})
endif()

if(MATX_ENABLE_BLIS)
  target_compile_definitions(matx_solve PRIVATE MATX_HAVE_BLIS=1)
  target_link_libraries(matx_solve PRIVATE ${amdblis_lib})
endif()

if(MATX_ENABLE_LIBFLAME)
  target_compile_definitions(matx_solve PRIVATE MATX_HAVE_LIBFLAME=1)
  target_link_libraries(matx_solve PRIVATE ${libflame_lib})
endif()

target_sources(matx_solve
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/matx/matx_solve.h>
    $<INSTALL_INTERFACE:include/matx/matx_solve.h>
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS matx_solve
  EXPORT MatXTargets
)