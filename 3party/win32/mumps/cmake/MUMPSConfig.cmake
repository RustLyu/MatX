set(mumps_ROOT "${mumps_DIR}/..")

add_library(mumps::mumps IMPORTED SHARED GLOBAL)

set_target_properties(mumps::mumps PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${mumps_ROOT}/include"
)


set_target_properties(mumps::mumps PROPERTIES
    IMPORTED_LOCATION_DEBUG "${mumps_ROOT}/lib/dmumps.lib"
    IMPORTED_LOCATION_RELEASE "${mumps_ROOT}/lib/dmumps.lib"
)
