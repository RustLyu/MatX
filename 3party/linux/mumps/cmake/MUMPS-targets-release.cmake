#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "MUMPS::MPISEQ" for configuration "Release"
set_property(TARGET MUMPS::MPISEQ APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MUMPS::MPISEQ PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C;Fortran"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib64/libmpiseq.a"
  )

list(APPEND _cmake_import_check_targets MUMPS::MPISEQ )
list(APPEND _cmake_import_check_files_for_MUMPS::MPISEQ "${_IMPORT_PREFIX}/lib64/libmpiseq.a" )

# Import target "MUMPS::mpiseq_c" for configuration "Release"
set_property(TARGET MUMPS::mpiseq_c APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MUMPS::mpiseq_c PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib64/libmpiseq_c.a"
  )

list(APPEND _cmake_import_check_targets MUMPS::mpiseq_c )
list(APPEND _cmake_import_check_files_for_MUMPS::mpiseq_c "${_IMPORT_PREFIX}/lib64/libmpiseq_c.a" )

# Import target "MUMPS::mpiseq_fortran" for configuration "Release"
set_property(TARGET MUMPS::mpiseq_fortran APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MUMPS::mpiseq_fortran PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C;Fortran"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib64/libmpiseq_fortran.a"
  )

list(APPEND _cmake_import_check_targets MUMPS::mpiseq_fortran )
list(APPEND _cmake_import_check_files_for_MUMPS::mpiseq_fortran "${_IMPORT_PREFIX}/lib64/libmpiseq_fortran.a" )

# Import target "MUMPS::PORD" for configuration "Release"
set_property(TARGET MUMPS::PORD APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MUMPS::PORD PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib64/libpord.a"
  )

list(APPEND _cmake_import_check_targets MUMPS::PORD )
list(APPEND _cmake_import_check_files_for_MUMPS::PORD "${_IMPORT_PREFIX}/lib64/libpord.a" )

# Import target "MUMPS::COMMON" for configuration "Release"
set_property(TARGET MUMPS::COMMON APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MUMPS::COMMON PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C;Fortran"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib64/libmumps_common.a"
  )

list(APPEND _cmake_import_check_targets MUMPS::COMMON )
list(APPEND _cmake_import_check_files_for_MUMPS::COMMON "${_IMPORT_PREFIX}/lib64/libmumps_common.a" )

# Import target "MUMPS::SMUMPS" for configuration "Release"
set_property(TARGET MUMPS::SMUMPS APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MUMPS::SMUMPS PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C;Fortran"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib64/libsmumps.a"
  )

list(APPEND _cmake_import_check_targets MUMPS::SMUMPS )
list(APPEND _cmake_import_check_files_for_MUMPS::SMUMPS "${_IMPORT_PREFIX}/lib64/libsmumps.a" )

# Import target "MUMPS::DMUMPS" for configuration "Release"
set_property(TARGET MUMPS::DMUMPS APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MUMPS::DMUMPS PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C;Fortran"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib64/libdmumps.a"
  )

list(APPEND _cmake_import_check_targets MUMPS::DMUMPS )
list(APPEND _cmake_import_check_files_for_MUMPS::DMUMPS "${_IMPORT_PREFIX}/lib64/libdmumps.a" )

# Import target "MUMPS::CMUMPS" for configuration "Release"
set_property(TARGET MUMPS::CMUMPS APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MUMPS::CMUMPS PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C;Fortran"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib64/libcmumps.a"
  )

list(APPEND _cmake_import_check_targets MUMPS::CMUMPS )
list(APPEND _cmake_import_check_files_for_MUMPS::CMUMPS "${_IMPORT_PREFIX}/lib64/libcmumps.a" )

# Import target "MUMPS::ZMUMPS" for configuration "Release"
set_property(TARGET MUMPS::ZMUMPS APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(MUMPS::ZMUMPS PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C;Fortran"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib64/libzmumps.a"
  )

list(APPEND _cmake_import_check_targets MUMPS::ZMUMPS )
list(APPEND _cmake_import_check_files_for_MUMPS::ZMUMPS "${_IMPORT_PREFIX}/lib64/libzmumps.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
