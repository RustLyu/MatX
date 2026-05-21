set(amdblis_ROOT "${amdblis_DIR}/..")

add_library(amdblis::amdblis IMPORTED SHARED GLOBAL)

set_target_properties(amdblis::amdblis PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${AOCL_ROOT}/include"
)

set_target_properties(amdblis::amdblis PROPERTIES
    IMPORTED_LOCATION_DEBUG "${AOCL_ROOT}/lib/libblis-mt.so.5.2.2"
    IMPORTED_LOCATION_RELEASE "${AOCL_ROOT}/lib/libblis-mt.so.5.2.2"
)
