set(mkl_INCLUDE_DIR ${mkl_path}/include)
include_directories(${mkl_INCLUDE_DIR})

file(GLOB mkl_bin ${mkl_path}/bin/*)

add_library(mkl STATIC IMPORTED)
set_target_properties(
  mkl
  PROPERTIES "IMPORTED_LOCATION_DEBUG" ${mkl_path}/lib/*.lib
             "IMPORTED_LOCATION_RELEASE" ${mkl_path}/lib/*.lib
             "IMPORTED_LOCATION_MINSIZEREL" ${mkl_path}/lib/*.lib
             "IMPORTED_LOCATION_RELWITHDEBINFO" ${mkl_path}/lib/*.lib)
