set(aoclutils_ROOT "${aoclutils_DIR}/..")

add_library(aoclutils::aoclutils IMPORTED SHARED GLOBAL)

set_target_properties(aoclutils::aoclutils PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${AOCL_ROOT}/include"
)

set_target_properties(aoclutils::aoclutils PROPERTIES
    IMPORTED_LOCATION_DEBUG "${AOCL_ROOT}/lib/libaoclutils.so"
    IMPORTED_LOCATION_RELEASE "${AOCL_ROOT}/lib/libaoclutils.so"
)
