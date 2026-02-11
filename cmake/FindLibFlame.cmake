# Minimal LibFlame finder for Windows/Linux.
#
# Variables:
# - LibFlame_FOUND
# - LibFlame_INCLUDE_DIRS
# - LibFlame_LIBRARIES
#
# Imported target:
# - LibFlame::LibFlame

include(FindPackageHandleStandardArgs)

set(_LibFlame_ROOT_HINTS "")
if(DEFINED LibFlame_ROOT)
  list(APPEND _LibFlame_ROOT_HINTS "${LibFlame_ROOT}")
endif()

find_path(LibFlame_INCLUDE_DIR
  NAMES FLAME.h
  HINTS ${_LibFlame_ROOT_HINTS}
  PATH_SUFFIXES include
)

find_library(LibFlame_LIBRARY
  NAMES LibFlame AOCL-LibFlame-Win-MT-dll.lib
  HINTS ${_BLIS_ROOT_HINTS}
  PATH_SUFFIXES lib lib64
)

find_package_handle_standard_args(LibFlame
  REQUIRED_VARS LibFlame_LIBRARY LibFlame_INCLUDE_DIR
)

if(LibFlame_FOUND)
  set(LibFlame_LIBRARIES ${LibFlame_LIBRARY})
  set(LibFlame_INCLUDE_DIRS ${LibFlame_INCLUDE_DIR})

  if(NOT TARGET LibFlame::LibFlame)
    add_library(LibFlame::LibFlame UNKNOWN IMPORTED)
    set_target_properties(LibFlame::LibFlame PROPERTIES
      IMPORTED_LOCATION "${LibFlame_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${LibFlame_INCLUDE_DIR}"
    )
  endif()
endif()

