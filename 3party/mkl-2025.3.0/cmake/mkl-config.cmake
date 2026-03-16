set(mkl_ROOT "${mkl_DIR}/..")

add_library(mkl::mkl IMPORTED SHARED GLOBAL)

set_target_properties(mkl::mkl PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${mkl_ROOT}/include"
)

set_target_properties(mkl::mkl PROPERTIES
    IMPORTED_IMPLIB "${mkl_DIR}/lib/mkl_intel_ilp64_dll.lib"
    IMPORTED_LOCATION "${mkl_ROOT}/bin/mkl_intel_ilp64.dll"
    INTERFACE_LINK_LIBRARIES "\
${mkl_DIR}/lib/mkl_intel_ilp64_dll.lib;\
${mkl_DIR}/lib/mkl_intel_thread_dll.lib;\
${mkl_DIR}/lib/mkl_core_dll.lib"
)