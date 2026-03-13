set(suitesparse_INCLUDE_DIR ${suitesparse_DIR}/../include)
include_directories(${suitesparse_INCLUDE_DIR})

set(suitesparse_lib
    -Wl,--start-group
    ${suitesparse_DIR}/../lib/klu.lib
	${suitesparse_DIR}/../lib/btf.lib
	${suitesparse_DIR}/../lib/amd.lib
	${suitesparse_DIR}/../lib/colamd.lib
	${suitesparse_DIR}/../lib/suitesparseconfig.lib
	${suitesparse_DIR}/../lib/ldl.lib
    -Wl,--end-group
)
