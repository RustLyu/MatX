#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SuiteSparse::RBio" for configuration "Debug"
set_property(TARGET SuiteSparse::RBio APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SuiteSparse::RBio PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/rbio.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_DEBUG "SuiteSparse::SuiteSparseConfig"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/rbio.dll"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::RBio )
list(APPEND _cmake_import_check_files_for_SuiteSparse::RBio "${_IMPORT_PREFIX}/lib/rbio.lib" "${_IMPORT_PREFIX}/bin/rbio.dll" )

# Import target "SuiteSparse::RBio_static" for configuration "Debug"
set_property(TARGET SuiteSparse::RBio_static APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SuiteSparse::RBio_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/rbio_static.lib"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::RBio_static )
list(APPEND _cmake_import_check_files_for_SuiteSparse::RBio_static "${_IMPORT_PREFIX}/lib/rbio_static.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
