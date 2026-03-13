set(amdblis_INCLUDE_DIR ${amdblis_DIR}/../include/ILP64)
include_directories(${amdblis_INCLUDE_DIR})

set(amdblis_lib
  $<$<CONFIG:Debug>:
    ${amdblis_DIR}/../lib/ILP64/AOCL-LibBlis-Win-MT-dll.lib
  >
  $<$<NOT:$<CONFIG:Debug>>:
    ${amdblis_DIR}/../lib/ILP64/AOCL-LibBlis-Win-MT-dll.lib
  >
)
