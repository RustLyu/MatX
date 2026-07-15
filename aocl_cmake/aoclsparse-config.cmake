set(aoclsparse_ROOT "${aoclsparse_DIR}/..")

add_library(aoclsparse::aoclsparse IMPORTED SHARED GLOBAL)

set_target_properties(aoclsparse::aoclsparse PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${AOCL_ROOT}/include"
)

set_target_properties(aoclsparse::aoclsparse PROPERTIES
    IMPORTED_LOCATION_DEBUG "${AOCL_ROOT}/lib/libaoclsparse.so"
    IMPORTED_LOCATION_RELEASE "${AOCL_ROOT}/lib/libaoclsparse.so"
)
