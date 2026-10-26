project (matx_sparse_solve C)

enable_language(CXX)

add_library(${PROJECT_NAME}
  src/matx_solve_backend.c
  src/matx_suitesparse_stub.c
  src/matx_umfpack_stub.c
  src/matx_cxsparse_stub.c
  src/matx_mumps_stub.c
  src/matx_superlu_stub.c
  src/tools.c
  src/matx_sparse_iterative.cpp
  src/matx_sparse_iterative_ginkgo.cpp
  src/matx_sparse_iterative_petsc.cpp
  src/matx_sparse_iterative_hypre.cpp
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

# SuiteSparse and MUMPS use the selected BLAS ABI. Put it before PETSc in the
# final link order so their BLAS symbols cannot bind to PETSc's bundled LP64
# f2cblaslapack when MatX uses an ILP64 OpenBLAS.
if(MATX_ENABLE_OPENBLAS)
  target_link_libraries(${PROJECT_NAME} PRIVATE OpenBLAS64::OpenBLAS)
endif()

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
    MATX_HAVE_GINKGO=$<BOOL:${MATX_ENABLE_GINKGO}>
    MATX_HAVE_PETSC=$<BOOL:${MATX_ENABLE_PETSC}>
    MATX_HAVE_HYPRE=$<BOOL:${MATX_ENABLE_HYPRE}>
    MATX_HAVE_PETSC_HYPRE=$<BOOL:${MATX_ENABLE_PETSC_HYPRE}>
    MATX_ENABLE_AGMG=$<BOOL:${MATX_ENABLE_AGMG}>
)

if(MATX_ENABLE_GINKGO)
  target_link_libraries(${PROJECT_NAME} PRIVATE Ginkgo::ginkgo)
endif()

if(MATX_ENABLE_PETSC)
  target_link_libraries(${PROJECT_NAME} PRIVATE MatXThirdParty::PETSc)
  if(MATX_PETSC_BUILD_TARGET)
    add_dependencies(${PROJECT_NAME} ${MATX_PETSC_BUILD_TARGET})
  endif()
endif()

if(MATX_ENABLE_HYPRE)
  target_link_libraries(${PROJECT_NAME} PRIVATE MatXThirdParty::HYPRE)
  if(MATX_HYPRE_SEQUENTIAL_BUILD)
    target_compile_definitions(${PROJECT_NAME} PRIVATE MATX_HYPRE_SEQUENTIAL_BUILD=1)
  else()
    target_link_libraries(${PROJECT_NAME} PRIVATE MPI::MPI_CXX)
  endif()
endif()

if(MATX_ENABLE_SUITESPARSE)
  target_compile_definitions(${PROJECT_NAME} PRIVATE MATX_HAVE_SUITESPARSE=1)
  target_link_libraries(${PROJECT_NAME} PRIVATE SuiteSparse::KLU
    SuiteSparse::CHOLMOD
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
      target_link_libraries(${PROJECT_NAME} PRIVATE OpenBLAS64::OpenBLAS)
    endif()
    target_sources(${PROJECT_NAME} PRIVATE src/matx_superlu_stub.c)
    target_compile_definitions(${PROJECT_NAME} PRIVATE MATX_HAVE_SUPERLU=1)
    target_link_libraries(${PROJECT_NAME} PRIVATE superlu)
  elseif(TARGET superlu::superlu)
    if(MATX_ENABLE_BLIS)
      target_link_libraries(${PROJECT_NAME} PRIVATE amdblis::amdblis)
    elseif(MATX_ENABLE_OPENBLAS)
      target_link_libraries(${PROJECT_NAME} PRIVATE OpenBLAS64::OpenBLAS)
    endif()
    target_sources(${PROJECT_NAME} PRIVATE src/matx_superlu_stub.c)
    target_compile_definitions(${PROJECT_NAME} PRIVATE MATX_HAVE_SUPERLU=1)
    target_link_libraries(${PROJECT_NAME} PRIVATE superlu::superlu)
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

if(MATX_ENABLE_AGMG)
  if(TARGET AGMG::AGMG)
    target_sources(${PROJECT_NAME} PRIVATE src/matx_agmg_stub.c)
    target_compile_definitions(${PROJECT_NAME} PRIVATE MATX_HAVE_AGMG=1)
    target_link_libraries(${PROJECT_NAME} PRIVATE AGMG::AGMG)
  else()
    message(WARNING "MATX_ENABLE_AGMG=ON but AGMG target is unavailable; sparse_solve will build without AGMG backend.")
  endif()
endif()

target_sources(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/matx/matx_sparse_solve.h>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/matx/matx_sparse_iterative.h>
    $<INSTALL_INTERFACE:include/matx/matx_sparse_solve.h>
    $<INSTALL_INTERFACE:include/matx/matx_sparse_iterative.h>
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS ${PROJECT_NAME}
  EXPORT MatXConfig
)
