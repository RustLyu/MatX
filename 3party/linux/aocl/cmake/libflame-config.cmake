set(libflame_ROOT "${libflame_DIR}/..")

add_library(libflame::libflame IMPORTED SHARED GLOBAL)

set_target_properties(libflame::libflame PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${libflame_ROOT}/include_ILP64"
)

#set_target_properties(libflame::libflame PROPERTIES
#    IMPORTED_IMPLIB_DEBUG "${libflame_ROOT}/lib_ILP64/libflame.so"
#    IMPORTED_IMPLIB_RELEASE "${libflame_ROOT}/lib_LP64/libflame.so"
#)
set_target_properties(libflame::libflame PROPERTIES
    IMPORTED_LOCATION_DEBUG "${libflame_ROOT}/lib_ILP64/libflame.so"
    IMPORTED_LOCATION_RELEASE "${libflame_ROOT}/lib_ILP64/libflame.so"
)
