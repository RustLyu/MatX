#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SuiteSparse::BTF" for configuration "Debug"
set_property(TARGET SuiteSparse::BTF APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SuiteSparse::BTF PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/btf.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/btf.dll"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::BTF )
list(APPEND _cmake_import_check_files_for_SuiteSparse::BTF "${_IMPORT_PREFIX}/lib/btf.lib" "${_IMPORT_PREFIX}/bin/btf.dll" )

# Import target "SuiteSparse::BTF_static" for configuration "Debug"
set_property(TARGET SuiteSparse::BTF_static APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(SuiteSparse::BTF_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/btf_static.lib"
  )

list(APPEND _cmake_import_check_targets SuiteSparse::BTF_static )
list(APPEND _cmake_import_check_files_for_SuiteSparse::BTF_static "${_IMPORT_PREFIX}/lib/btf_static.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
