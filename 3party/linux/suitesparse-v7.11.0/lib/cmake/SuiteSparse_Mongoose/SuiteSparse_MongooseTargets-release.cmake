#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SuiteSparse::Mongoose" for configuration "Release"
set_property(TARGET SuiteSparse::Mongoose APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SuiteSparse::Mongoose PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "SuiteSparse::SuiteSparseConfig"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libsuitesparse_mongoose.so.3"
  IMPORTED_SONAME_RELEASE "libsuitesparse_mongoose.so.3"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::Mongoose )
list(APPEND _cmake_import_check_files_for_SuiteSparse::Mongoose "${_IMPORT_PREFIX}/lib/libsuitesparse_mongoose.so.3" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
