#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SuiteSparse::SPEX" for configuration "Release"
set_property(TARGET SuiteSparse::SPEX APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SuiteSparse::SPEX PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "SuiteSparse::SuiteSparseConfig;SuiteSparse::AMD;SuiteSparse::COLAMD"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libspex.so.3.2.4"
  IMPORTED_SONAME_RELEASE "libspex.so.3"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::SPEX )
list(APPEND _cmake_import_check_files_for_SuiteSparse::SPEX "${_IMPORT_PREFIX}/lib/libspex.so.3.2.4" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
