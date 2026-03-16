set(libflame_ROOT "${libflame_DIR}/..")

add_library(libflame::libflame IMPORTED SHARED GLOBAL)

set_target_properties(libflame::libflame PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${libflame_ROOT}/include/ILP64"
)

set_target_properties(libflame::libflame PROPERTIES
    IMPORTED_IMPLIB_DEBUG "${libflame_ROOT}/lib/ILP64/AOCL-LibFlame-Win-MT-dll.lib"
    IMPORTED_IMPLIB_RELEASE "${libflame_ROOT}/lib/ILP64/AOCL-LibFlame-Win-MT-dll.lib"
    IMPORTED_IMPLIB_MINSIZEREL "${libflame_ROOT}/lib/ILP64/AOCL-LibFlame-Win-MT-dll.lib"
    IMPORTED_IMPLIB_RELWITHDEBINFO "${libflame_ROOT}/lib/ILP64/AOCL-LibFlame-Win-MT-dll.lib"
    IMPORTED_LOCATION_DEBUG "${libflame_ROOT}/bin/ILP64/AOCL-LibFlame-Win-MT-dll.dll"
    IMPORTED_LOCATION_RELEASE "${libflame_ROOT}/bin/ILP64/AOCL-LibFlame-Win-MT-dll.dll"
    IMPORTED_LOCATION_MINSIZEREL "${libflame_ROOT}/bin/ILP64/AOCL-LibFlame-Win-MT-dll.dll"
    IMPORTED_LOCATION_RELWITHDEBINFO "${libflame_ROOT}/bin/ILP64/AOCL-LibFlame-Win-MT-dll.dll"
)