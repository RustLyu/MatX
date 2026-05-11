#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SuiteSparse::ParU" for configuration "Debug"
set_property(TARGET SuiteSparse::ParU APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SuiteSparse::ParU PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/libparu.dll.a"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_DEBUG "SuiteSparse::SuiteSparseConfig;SuiteSparse::UMFPACK"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/libparu.dll"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::ParU )
list(APPEND _cmake_import_check_files_for_SuiteSparse::ParU "${_IMPORT_PREFIX}/lib/libparu.dll.a" "${_IMPORT_PREFIX}/bin/libparu.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
