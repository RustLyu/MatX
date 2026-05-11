set(aoclsparse_ROOT "${aoclsparse_DIR}/..")

add_library(aoclsparse::aoclsparse IMPORTED SHARED GLOBAL)

set_target_properties(aoclsparse::aoclsparse PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${aoclsparse_ROOT}/include_ILP64"
)

#set_target_properties(aoclsparse::aoclsparse PROPERTIES
#    IMPORTED_IMPLIB_DEBUG "${aoclsparse_ROOT}/lib_IILP64/libaoclsparse.so.5.2.0"
#    IMPORTED_IMPLIB_RELEASE "${aoclsparse_ROOT}/lib_IILP64/libaoclsparse.so.5.2.0"
#)
set_target_properties(aoclsparse::aoclsparse PROPERTIES
    IMPORTED_LOCATION_DEBUG "${aoclsparse_ROOT}/lib_ILP64/libaoclsparse.so.5.2.0"
    IMPORTED_LOCATION_RELEASE "${aoclsparse_ROOT}/lib_ILP64/libaoclsparse.so.5.2.0"
)
