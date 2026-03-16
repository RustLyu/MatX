set(openblas_ROOT "${openblas_DIR}/..")

add_library(openblas::openblas IMPORTED SHARED GLOBAL)

set_target_properties(openblas::openblas PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${openblas_ROOT}/include"
)

set_target_properties(openblas::openblas PROPERTIES
    IMPORTED_IMPLIB "${openblas_ROOT}/lib/openblas.lib"
    IMPORTED_LOCATION "${openblas_ROOT}/bin/openblas.dll"
)