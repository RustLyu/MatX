set(graphblas_INCLUDE_DIR ${graphblas_path}/include)
include_directories(${graphblas_INCLUDE_DIR})

set(graphblas_lib
    -Wl,--start-group
    ${graphblas_path}/lib/graphblas.lib
    -Wl,--end-group
)
