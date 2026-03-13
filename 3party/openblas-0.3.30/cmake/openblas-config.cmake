set(openblas_INCLUDE_DIR ${openblas_DIR}/../include)
#include_directories(${openblas_INCLUDE_DIR})

set(openblas_lib
  $<$<CONFIG:Debug>:
    ${openblas_DIR}/../lib/openblas.lib
  >
  $<$<NOT:$<CONFIG:Debug>>:
    ${openblas_DIR}/../lib/openblas.lib
  >
)
