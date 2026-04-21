set(amdblis_ROOT "${amdblis_DIR}/..")

add_library(amdblis::amdblis IMPORTED SHARED GLOBAL)

set_target_properties(amdblis::amdblis PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${amdblis_ROOT}/include_ILP64"
)

set_target_properties(amdblis::amdblis PROPERTIES
    IMPORTED_IMPLIB_DEBUG "${amdblis_ROOT}/lib_ILP64/libblis-mt.so.5.2.0"
    IMPORTED_IMPLIB_RELEASE "${amdblis_ROOT}/lib_ILP64/libblis-mt.so.5.2.0"
)