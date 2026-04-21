#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SuiteSparse::ParU" for configuration "Release"
set_property(TARGET SuiteSparse::ParU APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SuiteSparse::ParU PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "SuiteSparse::SuiteSparseConfig;SuiteSparse::UMFPACK"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libparu.so.1.0.1"
  IMPORTED_SONAME_RELEASE "libparu.so.1"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::ParU )
list(APPEND _cmake_import_check_files_for_SuiteSparse::ParU "${_IMPORT_PREFIX}/lib/libparu.so.1.0.1" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
