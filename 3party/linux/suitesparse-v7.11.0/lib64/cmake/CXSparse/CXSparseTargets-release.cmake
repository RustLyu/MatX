#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SuiteSparse::CXSparse" for configuration "Release"
set_property(TARGET SuiteSparse::CXSparse APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SuiteSparse::CXSparse PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "SuiteSparse::SuiteSparseConfig"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib64/libcxsparse.so.4.4.2"
  IMPORTED_SONAME_RELEASE "libcxsparse.so.4"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::CXSparse )
list(APPEND _cmake_import_check_files_for_SuiteSparse::CXSparse "${_IMPORT_PREFIX}/lib64/libcxsparse.so.4.4.2" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
