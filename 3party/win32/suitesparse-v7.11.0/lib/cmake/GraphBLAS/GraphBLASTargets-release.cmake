#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SuiteSparse::GraphBLAS" for configuration "Release"
set_property(TARGET SuiteSparse::GraphBLAS APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SuiteSparse::GraphBLAS PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/graphblas.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/graphblas.dll"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::GraphBLAS )
list(APPEND _cmake_import_check_files_for_SuiteSparse::GraphBLAS "${_IMPORT_PREFIX}/lib/graphblas.lib" "${_IMPORT_PREFIX}/bin/graphblas.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
