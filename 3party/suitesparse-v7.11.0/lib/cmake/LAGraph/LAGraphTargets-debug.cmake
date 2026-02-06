#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SuiteSparse::LAGraph" for configuration "Debug"
set_property(TARGET SuiteSparse::LAGraph APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SuiteSparse::LAGraph PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/lagraph.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_DEBUG "SuiteSparse::GraphBLAS"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/lagraph.dll"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::LAGraph )
list(APPEND _cmake_import_check_files_for_SuiteSparse::LAGraph "${_IMPORT_PREFIX}/lib/lagraph.lib" "${_IMPORT_PREFIX}/bin/lagraph.dll" )

# Import target "SuiteSparse::LAGraphX" for configuration "Debug"
set_property(TARGET SuiteSparse::LAGraphX APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SuiteSparse::LAGraphX PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/lagraphx.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_DEBUG "SuiteSparse::LAGraph;SuiteSparse::GraphBLAS"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/lagraphx.dll"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::LAGraphX )
list(APPEND _cmake_import_check_files_for_SuiteSparse::LAGraphX "${_IMPORT_PREFIX}/lib/lagraphx.lib" "${_IMPORT_PREFIX}/bin/lagraphx.dll" )

# Import target "SuiteSparse::LAGraph_static" for configuration "Debug"
set_property(TARGET SuiteSparse::LAGraph_static APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SuiteSparse::LAGraph_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/lagraph_static.lib"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::LAGraph_static )
list(APPEND _cmake_import_check_files_for_SuiteSparse::LAGraph_static "${_IMPORT_PREFIX}/lib/lagraph_static.lib" )

# Import target "SuiteSparse::LAGraphX_static" for configuration "Debug"
set_property(TARGET SuiteSparse::LAGraphX_static APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SuiteSparse::LAGraphX_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/lagraphx_static.lib"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::LAGraphX_static )
list(APPEND _cmake_import_check_files_for_SuiteSparse::LAGraphX_static "${_IMPORT_PREFIX}/lib/lagraphx_static.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
