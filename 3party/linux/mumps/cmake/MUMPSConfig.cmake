
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

include(CMakeFindDependencyMacro)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})

set(MUMPS_UPSTREAM_VERSION 5.9.0)
set(MUMPS_ACTUAL_VERSION 5.9.0)

set(MUMPS_intsize64 OFF)
set(MUMPS_parallel OFF)

set(MUMPS_LAPACK_VENDOR )

set(MUMPS_SCALAPACK_VENDOR )
set(MUMPS_scalapack ON)

set(MUMPS_s_FOUND on)
set(MUMPS_SINGLE on)

set(MUMPS_d_FOUND on)
set(MUMPS_DOUBLE on)

set(MUMPS_c_FOUND on)
set(MUMPS_COMPLEX on)

set(MUMPS_z_FOUND on)
set(MUMPS_COMPLEX16 on)

if(MUMPS_parallel)
  find_dependency(MPI COMPONENTS C Fortran)

  if(NOT MUMPS_LAPACK_VENDOR MATCHES "^MKL")
    find_dependency(LAPACK COMPONENTS ${MUMPS_LAPACK_VENDOR})
  endif()

  if(DEFINED MUMPS_SCALAPACK_VENDOR)
    find_dependency(SCALAPACK COMPONENTS ${MUMPS_SCALAPACK_VENDOR})
  elseif(MUMPS_scalapack)
    find_dependency(SCALAPACK)
  else()
    find_dependency(SCALAPACK CONFIG)
  endif()
else()
  find_dependency(LAPACK COMPONENTS ${MUMPS_LAPACK_VENDOR})

  # This is used in place of find_dependency(MPI) in MUMPS when MUMPS_parallel is OFF
  # may not have ALIAS of ALIAS so we do INTERFACE IMPORTED of ALIAS
  add_library(MPI::MPI_C INTERFACE IMPORTED)
  target_link_libraries(MPI::MPI_C INTERFACE MUMPS::mpiseq_c)

  add_library(MPI::MPI_Fortran INTERFACE IMPORTED)
  target_link_libraries(MPI::MPI_Fortran INTERFACE MUMPS::mpiseq_fortran)
endif()

set(MUMPS_scotch OFF)
set(MUMPS_find_scotch )
set(SCOTCH_COMPONENTS )
if(MUMPS_scotch)
  find_dependency(SCOTCH COMPONENTS ${SCOTCH_COMPONENTS})
endif()

set(MUMPS_metis OFF)
if(MUMPS_metis)
  find_dependency(METIS)
endif()

set(MUMPS_gpu OFF)
if(MUMPS_gpu)
  find_dependency(CUDAToolkit)
endif()

set(MUMPS_parmetis OFF)
if(MUMPS_parmetis)
  find_dependency(METIS COMPONENTS ParMETIS)
endif()

set(MUMPS_openmp OFF)
if(MUMPS_openmp)
  find_dependency(OpenMP COMPONENTS C Fortran)
endif()

# all dependencies must be determined before this project's targets
include(${CMAKE_CURRENT_LIST_DIR}/MUMPS-targets.cmake)

check_required_components(MUMPS)
