#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SuiteSparse::KLU" for configuration "Debug"
set_property(TARGET SuiteSparse::KLU APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SuiteSparse::KLU PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/klu.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_DEBUG "SuiteSparse::SuiteSparseConfig;SuiteSparse::AMD;SuiteSparse::COLAMD;SuiteSparse::BTF"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/klu.dll"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::KLU )
list(APPEND _cmake_import_check_files_for_SuiteSparse::KLU "${_IMPORT_PREFIX}/lib/klu.lib" "${_IMPORT_PREFIX}/bin/klu.dll" )

# Import target "SuiteSparse::KLU_static" for configuration "Debug"
set_property(TARGET SuiteSparse::KLU_static APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SuiteSparse::KLU_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/klu_static.lib"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::KLU_static )
list(APPEND _cmake_import_check_files_for_SuiteSparse::KLU_static "${_IMPORT_PREFIX}/lib/klu_static.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
