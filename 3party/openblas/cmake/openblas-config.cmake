set(openblas_INCLUDE_DIR ${openblas_path}/include)
include_directories(${openblas_INCLUDE_DIR})

set(openblas_lib
  $<$<CONFIG:Debug>:
    ${openblas_path}/lib/openblas.lib
  >
  $<$<NOT:$<CONFIG:Debug>>:
    ${openblas_path}/lib/openblas.lib
  >
)

