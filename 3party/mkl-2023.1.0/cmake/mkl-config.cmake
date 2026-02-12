set(mkl_INCLUDE_DIR ${mkl_path}/include)
include_directories(${mkl_INCLUDE_DIR})

set(mkl_lib
    -Wl,--start-group
    ${mkl_path}/lib/mkl_intel_ilp64_dll.lib
    ${mkl_path}/lib/mkl_intel_thread_dll.lib
    ${mkl_path}/lib/mkl_core_dll.lib
    -Wl,--end-group
)
