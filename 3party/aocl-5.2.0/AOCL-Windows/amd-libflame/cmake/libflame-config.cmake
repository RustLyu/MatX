set(libflame_INCLUDE_DIR ${libflame_DIR}/../include/ILP64)
include_directories(${libflame_INCLUDE_DIR})

set(libflame_lib
  $<$<CONFIG:Debug>:
    ${libflame_DIR}/../lib/ILP64/AOCL-LibFlame-Win-MT-dll.lib
  >
  $<$<NOT:$<CONFIG:Debug>>:
    ${libflame_DIR}/../lib/ILP64/AOCL-LibFlame-Win-MT-dll.lib
  >
)

