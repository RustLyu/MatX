# PETSc's supported build system is its configure script plus make. Wrap the
# upstream workflow in ExternalProject so download, build, and install remain
# ordinary CMake build steps and stay below the MatX binary directory.
if(MATX_ENABLE_PETSC)
  if(MATX_BUILD_PETSC_FROM_SOURCE)
    include(ExternalProject)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    find_program(MATX_PETSC_MAKE_EXECUTABLE NAMES make gmake REQUIRED)

    set(MATX_PETSC_INSTALL_DIR "${CMAKE_BINARY_DIR}/iterative-deps/petsc")
    set(_matx_petsc_configure_args
      "--prefix=<INSTALL_DIR>"
      "PETSC_ARCH=arch-matx"
      "--with-cc=${CMAKE_C_COMPILER}"
      # PETSc's public headers define PETSC_FUNCTION_NAME_CXX when included
      # from C++. Keep PETSc's implementation in C, but let configure probe
      # the C++ compiler so those headers are usable by the MatX adapter.
      "--with-cxx=${CMAKE_CXX_COMPILER}"
      "--with-fc=0"
      "--with-mpi=0"
      "--with-shared-libraries=1"
      "--with-debugging=0"
      "--download-f2cblaslapack"
    )
    # MatX may link against an ILP64 OpenBLAS whose BLAS symbols have the same
    # ELF names as PETSc's bundled LP64 f2cblaslapack. Bind PETSc's own calls
    # locally so the application's BLAS cannot interpose incompatible symbols.
    if(UNIX AND NOT APPLE)
      list(APPEND _matx_petsc_configure_args
        "-LDFLAGS=-Wl,-Bsymbolic-functions")
    endif()
    if(MATX_ENABLE_PETSC_HYPRE)
      list(APPEND _matx_petsc_configure_args "--download-hypre")
    endif()

    set(_matx_petsc_source_dir "${CMAKE_BINARY_DIR}/_deps/petsc-src")
    set(_matx_petsc_archive_timestamp_args)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
      list(APPEND _matx_petsc_archive_timestamp_args DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    endif()

    ExternalProject_Add(matx_petsc_external
      URL https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-3.26.0.tar.gz
      ${_matx_petsc_archive_timestamp_args}
      PREFIX "${CMAKE_BINARY_DIR}/_deps/petsc"
      SOURCE_DIR "${CMAKE_BINARY_DIR}/_deps/petsc-src"
      BINARY_DIR "${CMAKE_BINARY_DIR}/_deps/petsc-build"
      INSTALL_DIR "${MATX_PETSC_INSTALL_DIR}"
      CONFIGURE_COMMAND
        "${CMAKE_COMMAND}" -E chdir "${_matx_petsc_source_dir}"
        "${Python3_EXECUTABLE}" ./configure
        ${_matx_petsc_configure_args}
      BUILD_COMMAND
        "${CMAKE_COMMAND}" -E chdir "${_matx_petsc_source_dir}"
        "${MATX_PETSC_MAKE_EXECUTABLE}" PETSC_DIR=${_matx_petsc_source_dir}
        PETSC_ARCH=arch-matx -j${BUILD_JOBS} all
      INSTALL_COMMAND
        "${CMAKE_COMMAND}" -E chdir "${_matx_petsc_source_dir}"
        "${MATX_PETSC_MAKE_EXECUTABLE}" PETSC_DIR=${_matx_petsc_source_dir}
        PETSC_ARCH=arch-matx install
      BUILD_BYPRODUCTS
        "${MATX_PETSC_INSTALL_DIR}/lib/libpetsc${CMAKE_SHARED_LIBRARY_SUFFIX}"
    )

    # Imported targets validate include directories during CMake generation;
    # create their future install locations before the ExternalProject builds.
    file(MAKE_DIRECTORY
      "${MATX_PETSC_INSTALL_DIR}/include"
      "${MATX_PETSC_INSTALL_DIR}/include/petsc"
      "${MATX_PETSC_INSTALL_DIR}/lib"
    )
    add_library(matx_petsc SHARED IMPORTED GLOBAL)
    set_target_properties(matx_petsc PROPERTIES
      IMPORTED_LOCATION
        "${MATX_PETSC_INSTALL_DIR}/lib/libpetsc${CMAKE_SHARED_LIBRARY_SUFFIX}"
      INTERFACE_INCLUDE_DIRECTORIES
        "${MATX_PETSC_INSTALL_DIR}/include;${MATX_PETSC_INSTALL_DIR}/include/petsc"
    )
    add_library(MatXThirdParty::PETSc ALIAS matx_petsc)
    set(MATX_PETSC_BUILD_TARGET matx_petsc_external)
    message(STATUS "MatX iterative dependency: PETSc v3.26.0 (source build, real scalar, serial)")
  else()
    find_package(PETSc CONFIG QUIET)
    if(NOT PETSc_FOUND AND NOT PETSC_FOUND)
      find_package(PETSc QUIET)
    endif()
    if(NOT TARGET PETSc::PETSc AND NOT TARGET PETSc::petsc AND
       NOT PETSC_LIBRARIES AND NOT PETSc_LIBRARIES)
      find_package(PkgConfig QUIET)
      if(PkgConfig_FOUND)
        pkg_check_modules(MATX_PETSC QUIET IMPORTED_TARGET PETSc)
        if(NOT MATX_PETSC_FOUND)
          pkg_check_modules(MATX_PETSC_LOWER QUIET IMPORTED_TARGET petsc)
        endif()
      endif()
    endif()

    if(TARGET PETSc::PETSc)
      add_library(MatXThirdParty::PETSc ALIAS PETSc::PETSc)
    elseif(TARGET PETSc::petsc)
      add_library(MatXThirdParty::PETSc ALIAS PETSc::petsc)
    elseif(TARGET PkgConfig::MATX_PETSC)
      add_library(MatXThirdParty::PETSc ALIAS PkgConfig::MATX_PETSC)
    elseif(TARGET PkgConfig::MATX_PETSC_LOWER)
      add_library(MatXThirdParty::PETSc ALIAS PkgConfig::MATX_PETSC_LOWER)
    else()
      add_library(matx_petsc INTERFACE)
      if(PETSC_INCLUDES)
        target_include_directories(matx_petsc INTERFACE ${PETSC_INCLUDES})
      elseif(PETSc_INCLUDE_DIRS)
        target_include_directories(matx_petsc INTERFACE ${PETSc_INCLUDE_DIRS})
      endif()
      if(PETSC_LIBRARIES)
        target_link_libraries(matx_petsc INTERFACE ${PETSC_LIBRARIES})
      elseif(PETSc_LIBRARIES)
        target_link_libraries(matx_petsc INTERFACE ${PETSc_LIBRARIES})
      else()
        message(FATAL_ERROR
          "MATX_ENABLE_PETSC=ON but PETSc was not found. Set MATX_BUILD_PETSC_FROM_SOURCE=ON, "
          "or provide a PETSc CMake package/pkg-config file.")
      endif()
      add_library(MatXThirdParty::PETSc ALIAS matx_petsc)
    endif()
    message(STATUS "MatX iterative dependency: PETSc (installed package)")
  endif()
endif()
