set(aoclsparse_ROOT "${aoclsparse_DIR}/..")

add_library(aoclsparse::aoclsparse IMPORTED SHARED GLOBAL)

set_target_properties(aoclsparse::aoclsparse PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${aoclsparse_ROOT}/include/"
)

set_target_properties(aoclsparse::aoclsparse PROPERTIES
    IMPORTED_IMPLIB_DEBUG "${aoclsparse_ROOT}/lib/ILP64/shared/aoclsparse.lib"
    IMPORTED_IMPLIB_RELEASE "${aoclsparse_ROOT}/lib/ILP64/shared/aoclsparse.lib"
    IMPORTED_IMPLIB_MINSIZEREL "${aoclsparse_ROOT}/lib/ILP64/shared/aoclsparse.lib"
    IMPORTED_IMPLIB_RELWITHDEBINFO "${aoclsparse_ROOT}/lib/ILP64/shared/aoclsparse.lib"
    IMPORTED_LOCATION_DEBUG "${aoclsparse_ROOT}/bin/ILP64/aoclsparse.dll"
    IMPORTED_LOCATION_RELEASE "${aoclsparse_ROOT}/bin/ILP64/aoclsparse.dll"
    IMPORTED_LOCATION_MINSIZEREL "${aoclsparse_ROOT}/bin/ILP64/aoclsparse.dll"
    IMPORTED_LOCATION_RELWITHDEBINFO "${aoclsparse_ROOT}/bin/ILP64/aoclsparse.dll"
)