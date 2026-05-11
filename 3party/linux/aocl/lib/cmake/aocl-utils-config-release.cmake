#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "au::aoclutils" for configuration "Release"
set_property(TARGET au::aoclutils APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(au::aoclutils PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libaoclutils.a"
  )

list(APPEND _cmake_import_check_targets au::aoclutils )
list(APPEND _cmake_import_check_files_for_au::aoclutils "${_IMPORT_PREFIX}/lib/libaoclutils.a" )

# Import target "au::aoclutils_shared" for configuration "Release"
set_property(TARGET au::aoclutils_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(au::aoclutils_shared PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libaoclutils.so"
  IMPORTED_SONAME_RELEASE "libaoclutils.so"
  )

list(APPEND _cmake_import_check_targets au::aoclutils_shared )
list(APPEND _cmake_import_check_files_for_au::aoclutils_shared "${_IMPORT_PREFIX}/lib/libaoclutils.so" )

# Import target "au::au_cpuid" for configuration "Release"
set_property(TARGET au::au_cpuid APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(au::au_cpuid PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libau_cpuid.a"
  )

list(APPEND _cmake_import_check_targets au::au_cpuid )
list(APPEND _cmake_import_check_files_for_au::au_cpuid "${_IMPORT_PREFIX}/lib/libau_cpuid.a" )

# Import target "au::au_cpuid_shared" for configuration "Release"
set_property(TARGET au::au_cpuid_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(au::au_cpuid_shared PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libau_cpuid.so"
  IMPORTED_SONAME_RELEASE "libau_cpuid.so"
  )

list(APPEND _cmake_import_check_targets au::au_cpuid_shared )
list(APPEND _cmake_import_check_files_for_au::au_cpuid_shared "${_IMPORT_PREFIX}/lib/libau_cpuid.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
