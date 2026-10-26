# =========================================================
# DepAGMG.cmake
# Locate AGMG (Algebraic Multigrid) library on the system.
#
# AGMG is distributed from https://agmg.eu (free for academic
# use, requires manual download + license acceptance).
#
# Usage:
#   1. Download AGMG source (Fortran 90 + C wrappers) from
#      https://agmg.eu
#   2. Build with gfortran:
#        gfortran -c -O3 dagmg.f90 zagmg.f90
#        ar rcs libagmg.a *.o
#   3. Set AGMG_ROOT to the install prefix, or ensure the
#      library and headers are in standard search paths.
#
# Alternatively, point AGMG_LIBRARY and AGMG_INCLUDE_DIR directly.
# =========================================================

if(MATX_ENABLE_AGMG)

  # Let the user point to a pre-built AGMG tree
  if(NOT AGMG_ROOT AND DEFINED ENV{AGMG_ROOT})
    set(AGMG_ROOT "$ENV{AGMG_ROOT}")
  endif()

  find_library(AGMG_LIBRARY
    NAMES agmg
    PATHS ${AGMG_ROOT} ${AGMG_ROOT}/lib
    PATH_SUFFIXES lib
    DOC "AGMG static/shared library"
  )

  find_path(AGMG_INCLUDE_DIR
    NAMES dagmg.h
    PATHS ${AGMG_ROOT} ${AGMG_ROOT}/include
    PATH_SUFFIXES include
    DOC "AGMG C header directory"
  )

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(AGMG
    REQUIRED_VARS AGMG_LIBRARY AGMG_INCLUDE_DIR
  )

  if(AGMG_FOUND)
    message(STATUS "AGMG found: lib=${AGMG_LIBRARY}, include=${AGMG_INCLUDE_DIR}")

    if(NOT TARGET AGMG::AGMG)
      add_library(AGMG::AGMG UNKNOWN IMPORTED)
      set_target_properties(AGMG::AGMG PROPERTIES
        IMPORTED_LOCATION "${AGMG_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${AGMG_INCLUDE_DIR}"
      )
      # AGMG is Fortran — always link the Fortran runtime
      if(CMAKE_Fortran_COMPILER)
        set_target_properties(AGMG::AGMG PROPERTIES
          INTERFACE_LINK_LIBRARIES "${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES}"
        )
      else()
        # Fortran not enabled in project(), pull in gfortran manually
        target_link_libraries(AGMG::AGMG INTERFACE gfortran)
      endif()
    endif()

    set(MATX_HAVE_AGMG 1)

    # Find LP64 (32-bit int) BLAS for AGMG's Fortran BLAS symbols.
    # AGMG is Fortran and requires the standard 32-bit-int BLAS ABI.
    # This is SEPARATE from MatX's ILP64 OpenBLAS and will be isolated
    # inside a bridge DLL to avoid symbol conflicts with MatX's BLAS.
    find_library(AGMG_LP64_BLAS
      NAMES openblas
      PATHS /ucrt64/lib /mingw64/lib /usr/lib
      DOC "LP64 OpenBLAS for AGMG Fortran BLAS symbols"
    )
    if(AGMG_LP64_BLAS)
      message(STATUS "AGMG LP64 BLAS: ${AGMG_LP64_BLAS}")
    else()
      message(WARNING "LP64 BLAS not found for AGMG — bridge DLL will have unresolved symbols")
    endif()
  else()
    message(WARNING
      "MATX_ENABLE_AGMG=ON but AGMG was not found. "
      "Download AGMG from https://agmg.eu, build it, and set AGMG_ROOT to the install prefix."
    )
    set(MATX_ENABLE_AGMG OFF)
  endif()

endif()