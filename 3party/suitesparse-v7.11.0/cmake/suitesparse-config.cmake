set(suitesparse_INCLUDE_DIR ${suitesparse_path}/include)
include_directories(${suitesparse_INCLUDE_DIR})

set(suitesparse_lib
    -Wl,--start-group
    ${suitesparse_path}/lib/klu.lib
	${suitesparse_path}/lib/btf.lib
	${suitesparse_path}/lib/amd.lib
	${suitesparse_path}/lib/colamd.lib
	${suitesparse_path}/lib/suitesparseconfig.lib
	${suitesparse_path}/lib/ldl.lib
    -Wl,--end-group
)
