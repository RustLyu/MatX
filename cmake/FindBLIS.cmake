# Minimal BLIS finder for Windows/Linux.
#
# Variables:
# - BLIS_FOUND
# - BLIS_INCLUDE_DIRS
# - BLIS_LIBRARIES
#
# Imported target:
# - BLIS::BLIS

include(FindPackageHandleStandardArgs)

set(_BLIS_ROOT_HINTS "")
if(DEFINED BLIS_ROOT)
  list(APPEND _BLIS_ROOT_HINTS "${BLIS_ROOT}")
endif()

find_path(BLIS_INCLUDE_DIR
  NAMES blis.h
  HINTS ${_BLIS_ROOT_HINTS}
  PATH_SUFFIXES include
)

find_library(BLIS_LIBRARY
  NAMES blis libblis
  HINTS ${_BLIS_ROOT_HINTS}
  PATH_SUFFIXES lib lib64
)

find_package_handle_standard_args(BLIS
  REQUIRED_VARS BLIS_LIBRARY BLIS_INCLUDE_DIR
)

if(BLIS_FOUND)
  set(BLIS_LIBRARIES ${BLIS_LIBRARY})
  set(BLIS_INCLUDE_DIRS ${BLIS_INCLUDE_DIR})

  if(NOT TARGET BLIS::BLIS)
    add_library(BLIS::BLIS UNKNOWN IMPORTED)
    set_target_properties(BLIS::BLIS PROPERTIES
      IMPORTED_LOCATION "${BLIS_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${BLIS_INCLUDE_DIR}"
    )
  endif()
endif()

