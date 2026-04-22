set(aoclutils_ROOT "${aoclutils_DIR}/..")

add_library(aoclutils::aoclutils IMPORTED SHARED GLOBAL)

set_target_properties(aoclutils::aoclutils PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${aoclutils_ROOT}/include"
)

set_target_properties(aoclutils::aoclutils PROPERTIES
    IMPORTED_IMPLIB_DEBUG "${aoclutils_ROOT}/lib/libaoclutils.lib"
    IMPORTED_IMPLIB_RELEASE "${aoclutils_ROOT}/lib/libaoclutils.lib"
    IMPORTED_IMPLIB_MINSIZEREL "${aoclutils_ROOT}/lib/libaoclutils.lib"
    IMPORTED_IMPLIB_RELWITHDEBINFO "${aoclutils_ROOT}/lib/libaoclutils.lib"
    IMPORTED_LOCATION_DEBUG "${aoclutils_ROOT}/lib/libaoclutils.lib"
    IMPORTED_LOCATION_RELEASE "${aoclutils_ROOT}/lib/libaoclutils.lib"
    IMPORTED_LOCATION_MINSIZEREL "${aoclutils_ROOT}/lib/libaoclutils.lib"
    IMPORTED_LOCATION_RELWITHDEBINFO "${aoclutils_ROOT}/lib/libaoclutils.lib"
)