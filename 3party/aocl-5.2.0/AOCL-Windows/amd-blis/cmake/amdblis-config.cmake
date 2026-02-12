set(amdblis_INCLUDE_DIR ${amdblis_path}/include/ILP64)
include_directories(${amdblis_INCLUDE_DIR})

set(amdblis_lib
  $<$<CONFIG:Debug>:
    ${amdblis_path}/lib/ILP64/AOCL-LibBlis-Win-MT-dll.lib
  >
  $<$<NOT:$<CONFIG:Debug>>:
    ${amdblis_path}/lib/ILP64/AOCL-LibBlis-Win-MT-dll.lib
  >
)
