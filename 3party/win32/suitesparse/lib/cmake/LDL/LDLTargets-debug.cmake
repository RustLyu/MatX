#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SuiteSparse::LDL" for configuration "Debug"
set_property(TARGET SuiteSparse::LDL APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SuiteSparse::LDL PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/libldl.dll.a"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/libldl.dll"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::LDL )
list(APPEND _cmake_import_check_files_for_SuiteSparse::LDL "${_IMPORT_PREFIX}/lib/libldl.dll.a" "${_IMPORT_PREFIX}/bin/libldl.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
