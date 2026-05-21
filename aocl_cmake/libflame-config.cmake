set(libflame_ROOT "${libflame_DIR}/..")

add_library(libflame::libflame IMPORTED SHARED GLOBAL)

set_target_properties(libflame::libflame PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${AOCL_ROOT}/include"
)

set_target_properties(libflame::libflame PROPERTIES
    IMPORTED_LOCATION_DEBUG "${AOCL_ROOT}/lib/libflame.so"
    IMPORTED_LOCATION_RELEASE "${AOCL_ROOT}/lib/libflame.so"
)
