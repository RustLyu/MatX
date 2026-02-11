set(libflame_INCLUDE_DIR ${libflame_path}/include/ILP64)
include_directories(${libflame_INCLUDE_DIR})

set(libflame_lib
  $<$<CONFIG:Debug>:
    ${libflame_path}/ILP64/lib/AOCL-LibFlame-Win-MT-dll.lib
  >
  $<$<NOT:$<CONFIG:Debug>>:
    ${libflame_path}/lib/ILP64/AOCL-LibFlame-Win-MT-dll.lib
  >
)

