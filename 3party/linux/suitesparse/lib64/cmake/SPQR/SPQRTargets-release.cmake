#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SuiteSparse::SPQR" for configuration "Release"
set_property(TARGET SuiteSparse::SPQR APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SuiteSparse::SPQR PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "SuiteSparse::SuiteSparseConfig;SuiteSparse::CHOLMOD"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib64/libspqr.so.4.3.6"
  IMPORTED_SONAME_RELEASE "libspqr.so.4"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::SPQR )
list(APPEND _cmake_import_check_files_for_SuiteSparse::SPQR "${_IMPORT_PREFIX}/lib64/libspqr.so.4.3.6" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
