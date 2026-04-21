#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SuiteSparse::LAGraph_static" for configuration "Release"
set_property(TARGET SuiteSparse::LAGraph_static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SuiteSparse::LAGraph_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/liblagraph.a"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::LAGraph_static )
list(APPEND _cmake_import_check_files_for_SuiteSparse::LAGraph_static "${_IMPORT_PREFIX}/lib/liblagraph.a" )

# Import target "SuiteSparse::LAGraphX_static" for configuration "Release"
set_property(TARGET SuiteSparse::LAGraphX_static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SuiteSparse::LAGraphX_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/liblagraphx.a"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::LAGraphX_static )
list(APPEND _cmake_import_check_files_for_SuiteSparse::LAGraphX_static "${_IMPORT_PREFIX}/lib/liblagraphx.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
