set(graphblas_INCLUDE_DIR ${graphblas_DIR}/../include)
#include_directories(${graphblas_INCLUDE_DIR})

set(graphblas_lib
    -Wl,--start-group
    ${graphblas_DIR}/../lib/graphblas.lib
    -Wl,--end-group
)
