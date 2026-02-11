set(amd-blis_INCLUDE_DIR ${amd-blis_path}/include/ILP64)
include_directories(${amd-blis_INCLUDE_DIR})

set(amdblis_lib
  $<$<CONFIG:Debug>:
    ${amd-blis_path}/ILP64/lib/AOCL-LibBlis-Win-MT-dll.lib
  >
  $<$<NOT:$<CONFIG:Debug>>:
    ${amd-blis_path}/lib/ILP64/AOCL-LibBlis-Win-MT-dll.lib
  >
)
