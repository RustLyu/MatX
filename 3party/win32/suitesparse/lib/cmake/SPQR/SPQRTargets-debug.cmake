#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SuiteSparse::SPQR" for configuration "Debug"
set_property(TARGET SuiteSparse::SPQR APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SuiteSparse::SPQR PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/libspqr.dll.a"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_DEBUG "SuiteSparse::SuiteSparseConfig;SuiteSparse::CHOLMOD"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/libspqr.dll"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::SPQR )
list(APPEND _cmake_import_check_files_for_SuiteSparse::SPQR "${_IMPORT_PREFIX}/lib/libspqr.dll.a" "${_IMPORT_PREFIX}/bin/libspqr.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
