#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SuiteSparse::ParU" for configuration "Debug"
set_property(TARGET SuiteSparse::ParU APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SuiteSparse::ParU PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/paru.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_DEBUG "SuiteSparse::SuiteSparseConfig;SuiteSparse::UMFPACK"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/paru.dll"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::ParU )
list(APPEND _cmake_import_check_files_for_SuiteSparse::ParU "${_IMPORT_PREFIX}/lib/paru.lib" "${_IMPORT_PREFIX}/bin/paru.dll" )

# Import target "SuiteSparse::ParU_static" for configuration "Debug"
set_property(TARGET SuiteSparse::ParU_static APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SuiteSparse::ParU_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/ParU_static.lib"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::ParU_static )
list(APPEND _cmake_import_check_files_for_SuiteSparse::ParU_static "${_IMPORT_PREFIX}/lib/ParU_static.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
