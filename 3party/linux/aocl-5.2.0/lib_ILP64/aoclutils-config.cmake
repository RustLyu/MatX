set(aoclutils_ROOT "${aoclutils_DIR}/..")

add_library(aoclutils::aoclutils IMPORTED SHARED GLOBAL)

set_target_properties(aoclutils::aoclutils PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${aoclutils_ROOT}/include_ILP64"
)

#set_target_properties(aoclutils::aoclutils PROPERTIES
#    IMPORTED_IMPLIB_DEBUG "${aoclutils_ROOT}/lib_ILP64/libaoclutils.so"
#    IMPORTED_IMPLIB_RELEASE "${aoclutils_ROOT}/lib_ILP64/libaoclutils.so"
#)

set_target_properties(aoclutils::aoclutils PROPERTIES
    IMPORTED_LOCATION_DEBUG "${aoclutils_ROOT}/lib_ILP64/libaoclutils.so"
    IMPORTED_LOCATION_RELEASE "${aoclutils_ROOT}/lib_ILP64/libaoclutils.so"
)
