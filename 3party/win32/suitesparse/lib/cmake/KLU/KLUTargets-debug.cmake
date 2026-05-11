#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SuiteSparse::KLU" for configuration "Debug"
set_property(TARGET SuiteSparse::KLU APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SuiteSparse::KLU PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/libklu.dll.a"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_DEBUG "SuiteSparse::SuiteSparseConfig;SuiteSparse::AMD;SuiteSparse::COLAMD;SuiteSparse::BTF"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/libklu.dll"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::KLU )
list(APPEND _cmake_import_check_files_for_SuiteSparse::KLU "${_IMPORT_PREFIX}/lib/libklu.dll.a" "${_IMPORT_PREFIX}/bin/libklu.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
