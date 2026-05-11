#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SuiteSparse::Mongoose" for configuration "Debug"
set_property(TARGET SuiteSparse::Mongoose APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SuiteSparse::Mongoose PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/libsuitesparse_mongoose.dll.a"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_DEBUG "SuiteSparse::SuiteSparseConfig"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/libsuitesparse_mongoose.dll"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::Mongoose )
list(APPEND _cmake_import_check_files_for_SuiteSparse::Mongoose "${_IMPORT_PREFIX}/lib/libsuitesparse_mongoose.dll.a" "${_IMPORT_PREFIX}/bin/libsuitesparse_mongoose.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
