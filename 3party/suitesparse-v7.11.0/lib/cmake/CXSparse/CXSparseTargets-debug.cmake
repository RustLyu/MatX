#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SuiteSparse::CXSparse" for configuration "Debug"
set_property(TARGET SuiteSparse::CXSparse APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SuiteSparse::CXSparse PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/cxsparse.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_DEBUG "SuiteSparse::SuiteSparseConfig"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/cxsparse.dll"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::CXSparse )
list(APPEND _cmake_import_check_files_for_SuiteSparse::CXSparse "${_IMPORT_PREFIX}/lib/cxsparse.lib" "${_IMPORT_PREFIX}/bin/cxsparse.dll" )

# Import target "SuiteSparse::CXSparse_static" for configuration "Debug"
set_property(TARGET SuiteSparse::CXSparse_static APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SuiteSparse::CXSparse_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/cxsparse_static.lib"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::CXSparse_static )
list(APPEND _cmake_import_check_files_for_SuiteSparse::CXSparse_static "${_IMPORT_PREFIX}/lib/cxsparse_static.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
