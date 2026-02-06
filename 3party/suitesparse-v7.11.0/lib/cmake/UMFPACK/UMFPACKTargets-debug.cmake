#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SuiteSparse::UMFPACK" for configuration "Debug"
set_property(TARGET SuiteSparse::UMFPACK APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SuiteSparse::UMFPACK PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/umfpack.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_DEBUG "SuiteSparse::SuiteSparseConfig;SuiteSparse::AMD;SuiteSparse::CHOLMOD"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/umfpack.dll"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::UMFPACK )
list(APPEND _cmake_import_check_files_for_SuiteSparse::UMFPACK "${_IMPORT_PREFIX}/lib/umfpack.lib" "${_IMPORT_PREFIX}/bin/umfpack.dll" )

# Import target "SuiteSparse::UMFPACK_static" for configuration "Debug"
set_property(TARGET SuiteSparse::UMFPACK_static APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SuiteSparse::UMFPACK_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/umfpack_static.lib"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::UMFPACK_static )
list(APPEND _cmake_import_check_files_for_SuiteSparse::UMFPACK_static "${_IMPORT_PREFIX}/lib/umfpack_static.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
