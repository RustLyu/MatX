#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SuiteSparse::GraphBLAS" for configuration "Debug"
set_property(TARGET SuiteSparse::GraphBLAS APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SuiteSparse::GraphBLAS PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/libgraphblas.dll.a"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/libgraphblas.dll"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::GraphBLAS )
list(APPEND _cmake_import_check_files_for_SuiteSparse::GraphBLAS "${_IMPORT_PREFIX}/lib/libgraphblas.dll.a" "${_IMPORT_PREFIX}/bin/libgraphblas.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
