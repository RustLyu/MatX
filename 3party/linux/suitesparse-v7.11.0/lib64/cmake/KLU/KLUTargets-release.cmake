#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SuiteSparse::KLU" for configuration "Release"
set_property(TARGET SuiteSparse::KLU APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SuiteSparse::KLU PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "SuiteSparse::SuiteSparseConfig;SuiteSparse::AMD;SuiteSparse::COLAMD;SuiteSparse::BTF"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib64/libklu.so.2.3.6"
  IMPORTED_SONAME_RELEASE "libklu.so.2"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::KLU )
list(APPEND _cmake_import_check_files_for_SuiteSparse::KLU "${_IMPORT_PREFIX}/lib64/libklu.so.2.3.6" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
