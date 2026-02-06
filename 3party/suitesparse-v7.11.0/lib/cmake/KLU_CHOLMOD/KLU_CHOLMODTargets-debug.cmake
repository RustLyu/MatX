#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SuiteSparse::KLU_CHOLMOD" for configuration "Debug"
set_property(TARGET SuiteSparse::KLU_CHOLMOD APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SuiteSparse::KLU_CHOLMOD PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/klu_cholmod.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_DEBUG "SuiteSparse::CHOLMOD;SuiteSparse::KLU"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/klu_cholmod.dll"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::KLU_CHOLMOD )
list(APPEND _cmake_import_check_files_for_SuiteSparse::KLU_CHOLMOD "${_IMPORT_PREFIX}/lib/klu_cholmod.lib" "${_IMPORT_PREFIX}/bin/klu_cholmod.dll" )

# Import target "SuiteSparse::KLU_CHOLMOD_static" for configuration "Debug"
set_property(TARGET SuiteSparse::KLU_CHOLMOD_static APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SuiteSparse::KLU_CHOLMOD_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/klu_cholmod_static.lib"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::KLU_CHOLMOD_static )
list(APPEND _cmake_import_check_files_for_SuiteSparse::KLU_CHOLMOD_static "${_IMPORT_PREFIX}/lib/klu_cholmod_static.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
