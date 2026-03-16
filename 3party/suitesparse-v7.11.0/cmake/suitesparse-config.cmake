set(suitesparse_ROOT "${suitesparse_DIR}/..")

add_library(SuiteSparse::SuiteSparse IMPORTED SHARED GLOBAL)

set_target_properties(SuiteSparse::SuiteSparse PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${suitesparse_ROOT}/include"
)

set_target_properties(SuiteSparse::SuiteSparse PROPERTIES
    IMPORTED_IMPLIB "${suitesparse_ROOT}/lib/klu.lib"
    IMPORTED_LOCATION "${suitesparse_ROOT}/bin/klu.dll"
    INTERFACE_LINK_LIBRARIES "\
${suitesparse_ROOT}/lib/klu.lib;\
${suitesparse_ROOT}/lib/btf.lib;\
${suitesparse_ROOT}/lib/amd.lib;\
${suitesparse_ROOT}/lib/colamd.lib;\
${suitesparse_ROOT}/lib/suitesparseconfig.lib;\
${suitesparse_ROOT}/lib/ldl.lib"
)