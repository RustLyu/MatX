# Minimal Mkl finder for Windows/Linux.
#
# Variables:
# - Mkl_FOUND
# - Mkl_INCLUDE_DIRS
# - Mkl_LIBRARIES
#
# Imported target:
# - Mkl::Mkl

include(FindPackageHandleStandardArgs)

set(_Mkl_ROOT_HINTS "")
if(DEFINED Mkl_ROOT)
  list(APPEND _Mkl_ROOT_HINTS "${Mkl_ROOT}")
endif()

find_path(Mkl_INCLUDE_DIR
  NAMES FLAME.h
  HINTS ${_Mkl_ROOT_HINTS}
  PATH_SUFFIXES include
)

find_library(Mkl_LIBRARY
  NAMES Mkl mkl_core_dll.lib
  HINTS ${_BLIS_ROOT_HINTS}
  PATH_SUFFIXES lib lib64
)

find_package_handle_standard_args(Mkl
  REQUIRED_VARS Mkl_LIBRARY Mkl_INCLUDE_DIR
)

if(Mkl_FOUND)
  set(Mkl_LIBRARIES ${Mkl_LIBRARY})
  set(Mkl_INCLUDE_DIRS ${Mkl_INCLUDE_DIR})

  if(NOT TARGET Mkl::Mkl)
    add_library(Mkl::Mkl UNKNOWN IMPORTED)
    set_target_properties(Mkl::Mkl PROPERTIES
      IMPORTED_LOCATION "${Mkl_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${Mkl_INCLUDE_DIR}"
    )
  endif()
endif()

