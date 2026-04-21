#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SuiteSparse::LAGraph" for configuration "Release"
set_property(TARGET SuiteSparse::LAGraph APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SuiteSparse::LAGraph PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "SuiteSparse::GraphBLAS"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/liblagraph.so.1.2.0"
  IMPORTED_SONAME_RELEASE "liblagraph.so.1"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::LAGraph )
list(APPEND _cmake_import_check_files_for_SuiteSparse::LAGraph "${_IMPORT_PREFIX}/lib/liblagraph.so.1.2.0" )

# Import target "SuiteSparse::LAGraphX" for configuration "Release"
set_property(TARGET SuiteSparse::LAGraphX APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SuiteSparse::LAGraphX PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "SuiteSparse::LAGraph;SuiteSparse::GraphBLAS"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/liblagraphx.so.1.2.0"
  IMPORTED_SONAME_RELEASE "liblagraphx.so.1"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::LAGraphX )
list(APPEND _cmake_import_check_files_for_SuiteSparse::LAGraphX "${_IMPORT_PREFIX}/lib/liblagraphx.so.1.2.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
