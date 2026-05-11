#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "AoclDlp::aocl-dlp" for configuration "Release"
set_property(TARGET AoclDlp::aocl-dlp APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(AoclDlp::aocl-dlp PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libaocl-dlp.so"
  IMPORTED_SONAME_RELEASE "libaocl-dlp.so"
  )

list(APPEND _cmake_import_check_targets AoclDlp::aocl-dlp )
list(APPEND _cmake_import_check_files_for_AoclDlp::aocl-dlp "${_IMPORT_PREFIX}/lib/libaocl-dlp.so" )

# Import target "AoclDlp::aocl-dlp_static" for configuration "Release"
set_property(TARGET AoclDlp::aocl-dlp_static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(AoclDlp::aocl-dlp_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C;CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libaocl-dlp.a"
  )

list(APPEND _cmake_import_check_targets AoclDlp::aocl-dlp_static )
list(APPEND _cmake_import_check_files_for_AoclDlp::aocl-dlp_static "${_IMPORT_PREFIX}/lib/libaocl-dlp.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
