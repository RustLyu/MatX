set(mkl_INCLUDE_DIR ${mkl_DIR}/../include)
#include_directories(${mkl_INCLUDE_DIR})

set(mkl_lib
    -Wl,--start-group
    ${mkl_DIR}/lib/mkl_intel_ilp64_dll.lib
    ${mkl_DIR}/lib/mkl_intel_thread_dll.lib
    ${mkl_DIR}/lib/mkl_core_dll.lib
    -Wl,--end-group
)
