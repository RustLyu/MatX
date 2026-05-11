#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SuiteSparse::CXSparse" for configuration "Debug"
set_property(TARGET SuiteSparse::CXSparse APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SuiteSparse::CXSparse PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/libcxsparse.dll.a"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_DEBUG "SuiteSparse::SuiteSparseConfig"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/libcxsparse.dll"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::CXSparse )
list(APPEND _cmake_import_check_files_for_SuiteSparse::CXSparse "${_IMPORT_PREFIX}/lib/libcxsparse.dll.a" "${_IMPORT_PREFIX}/bin/libcxsparse.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
