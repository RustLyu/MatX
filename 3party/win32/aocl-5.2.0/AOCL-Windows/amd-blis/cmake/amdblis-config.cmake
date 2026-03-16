set(amdblis_ROOT "${amdblis_DIR}/..")

add_library(amdblis::amdblis IMPORTED SHARED GLOBAL)

set_target_properties(amdblis::amdblis PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${amdblis_ROOT}/include/ILP64"
)

set_target_properties(amdblis::amdblis PROPERTIES
    IMPORTED_IMPLIB_DEBUG "${amdblis_ROOT}/lib/ILP64/AOCL-LibBlis-Win-MT-dll.lib"
    IMPORTED_IMPLIB_RELEASE "${amdblis_ROOT}/lib/ILP64/AOCL-LibBlis-Win-MT-dll.lib"
    IMPORTED_IMPLIB_MINSIZEREL "${amdblis_ROOT}/lib/ILP64/AOCL-LibBlis-Win-MT-dll.lib"
    IMPORTED_IMPLIB_RELWITHDEBINFO "${amdblis_ROOT}/lib/ILP64/AOCL-LibBlis-Win-MT-dll.lib"
    IMPORTED_LOCATION_DEBUG "${amdblis_ROOT}/bin/ILP64/AOCL-LibBlis-Win-MT-dll.dll"
    IMPORTED_LOCATION_RELEASE "${amdblis_ROOT}/bin/ILP64/AOCL-LibBlis-Win-MT-dll.dll"
    IMPORTED_LOCATION_MINSIZEREL "${amdblis_ROOT}/bin/ILP64/AOCL-LibBlis-Win-MT-dll.dll"
    IMPORTED_LOCATION_RELWITHDEBINFO "${amdblis_ROOT}/bin/ILP64/AOCL-LibBlis-Win-MT-dll.dll"
)