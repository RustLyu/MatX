project (matx_sparse_solve C)

add_library(${PROJECT_NAME}
  src/matx_solve_backend.c
  src/matx_suitesparse_stub.c
  src/matx_umfpack_stub.c
  src/matx_cxsparse_stub.c
  src/matx_mumps_stub.c
  src/matx_superlu_stub.c
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
    MATX_ENABLE_UMFPACK=$<BOOL:${MATX_ENABLE_UMFPACK}>
    MATX_ENABLE_CXSPARSE=$<BOOL:${MATX_ENABLE_CXSPARSE}>
    MATX_ENABLE_SUPERLU=$<BOOL:${MATX_ENABLE_SUPERLU}>
    MATX_ENABLE_MUMPS=$<BOOL:${MATX_ENABLE_MUMPS}>
)

if(MATX_ENABLE_SUITESPARSE)
  target_compile_definitions(${PROJECT_NAME} PRIVATE MATX_HAVE_SUITESPARSE=1)
  target_link_libraries(${PROJECT_NAME} PRIVATE SuiteSparse::KLU
    SuiteSparse::AMD
    SuiteSparse::COLAMD
    SuiteSparse::BTF
    SuiteSparse::SuiteSparseConfig)
endif()

if(MATX_ENABLE_UMFPACK)
  target_compile_definitions(${PROJECT_NAME} PRIVATE MATX_HAVE_UMFPACK=1)

  target_link_libraries(${PROJECT_NAME} PRIVATE
      SuiteSparse::UMFPACK
  )
endif()

if(MATX_ENABLE_CXSPARSE)
  target_compile_definitions(${PROJECT_NAME} PRIVATE MATX_HAVE_CXSPARSE=1)
  target_link_libraries(${PROJECT_NAME} PRIVATE SuiteSparse::CXSparse)
endif()

if(MATX_ENABLE_SUPERLU)
  if(TARGET superlu)
    if(MATX_ENABLE_BLIS)
      target_link_libraries(${PROJECT_NAME} PRIVATE amdblis::amdblis)
    elseif(MATX_ENABLE_OPENBLAS)
      target_link_libraries(${PROJECT_NAME} PRIVATE OpenBLAS::OpenBLAS)
    endif()
    target_sources(${PROJECT_NAME} PRIVATE src/matx_superlu_stub.c)
    target_compile_definitions(${PROJECT_NAME} PRIVATE MATX_HAVE_SUPERLU=1)
    target_link_libraries(${PROJECT_NAME} PRIVATE superlu)
  elseif(TARGET SUPERLU::SUPERLU)
    if(MATX_ENABLE_BLIS)
      target_link_libraries(${PROJECT_NAME} PRIVATE amdblis::amdblis)
    elseif(MATX_ENABLE_OPENBLAS)
      target_link_libraries(${PROJECT_NAME} PRIVATE OpenBLAS64::OpenBLAS)
    endif()
    target_sources(${PROJECT_NAME} PRIVATE src/matx_superlu_stub.c)
    target_compile_definitions(${PROJECT_NAME} PRIVATE MATX_HAVE_SUPERLU=1)
    target_link_libraries(${PROJECT_NAME} PRIVATE SUPERLU::SUPERLU)
  else()
    message(WARNING "MATX_ENABLE_SUPERLU=ON but SuperLU target is unavailable; sparse_solve will build without SuperLU backend.")
  endif()
endif()

if(MATX_ENABLE_MUMPS)
  if(TARGET MUMPS::MUMPS)
    if(MATX_ENABLE_BLIS)
      target_link_libraries(${PROJECT_NAME} PRIVATE amdblis::amdblis)
    elseif(MATX_ENABLE_OPENBLAS)
      target_link_libraries(${PROJECT_NAME} PRIVATE OpenBLAS64::OpenBLAS)
    endif()
    target_compile_definitions(${PROJECT_NAME} PRIVATE MATX_HAVE_MUMPS=1)
    target_link_libraries(${PROJECT_NAME} PRIVATE MUMPS::MUMPS)
  elseif(TARGET dmumps)
    if(MATX_ENABLE_BLIS)
      target_link_libraries(${PROJECT_NAME} PRIVATE amdblis::amdblis)
    elseif(MATX_ENABLE_OPENBLAS)
      target_link_libraries(${PROJECT_NAME} PRIVATE OpenBLAS::OpenBLAS)
    endif()
    target_compile_definitions(${PROJECT_NAME} PRIVATE MATX_HAVE_MUMPS=1)
    target_link_libraries(${PROJECT_NAME} PRIVATE dmumps)
    if(TARGET mumps_common)
      target_link_libraries(${PROJECT_NAME} PRIVATE mumps_common)
    endif()
  else()
    message(WARNING "MATX_ENABLE_MUMPS=ON but MUMPS target is unavailable; sparse_solve will build without MUMPS backend.")
  endif()
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
