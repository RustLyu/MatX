#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "au::libaoclutils" for configuration "Release"
set_property(TARGET au::libaoclutils APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(au::libaoclutils PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libaoclutils_static.lib"
  )

list(APPEND _cmake_import_check_targets au::libaoclutils )
list(APPEND _cmake_import_check_files_for_au::libaoclutils "${_IMPORT_PREFIX}/lib/libaoclutils_static.lib" )

# Import target "au::libaoclutils_shared" for configuration "Release"
set_property(TARGET au::libaoclutils_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(au::libaoclutils_shared PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/bin/libaoclutils.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/libaoclutils.dll"
  )

list(APPEND _cmake_import_check_targets au::libaoclutils_shared )
list(APPEND _cmake_import_check_files_for_au::libaoclutils_shared "${_IMPORT_PREFIX}/bin/libaoclutils.lib" "${_IMPORT_PREFIX}/bin/libaoclutils.dll" )

# Import target "au::au_cpuid" for configuration "Release"
set_property(TARGET au::au_cpuid APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(au::au_cpuid PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/au_cpuid_static.lib"
  )

list(APPEND _cmake_import_check_targets au::au_cpuid )
list(APPEND _cmake_import_check_files_for_au::au_cpuid "${_IMPORT_PREFIX}/lib/au_cpuid_static.lib" )

# Import target "au::au_cpuid_shared" for configuration "Release"
set_property(TARGET au::au_cpuid_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(au::au_cpuid_shared PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/bin/au_cpuid.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/au_cpuid.dll"
  )

list(APPEND _cmake_import_check_targets au::au_cpuid_shared )
list(APPEND _cmake_import_check_files_for_au::au_cpuid_shared "${_IMPORT_PREFIX}/bin/au_cpuid.lib" "${_IMPORT_PREFIX}/bin/au_cpuid.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
