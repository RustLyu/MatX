#!/bin/bash

# always enable
# suitesparse, graphblas
#
# enable test
# -DMATX_BUILD_TESTS=ON 
#
#if need dynamic lib
# -DBUILD_SHARED_LIBS=ON
#

# amd
#cmake ../ -DMATX_ENABLE_BLIS=ON -DMATX_ENABLE_LIBFLAME=ON -DMATX_ENABLE_AOCL_SPARSE=ON -DMATX_ENABLE_OPENBLAS=OFF

# intel
cmake ../ -DMATX_ENABLE_BLIS=OFF -DMATX_ENABLE_LIBFLAME=OFF -DMATX_ENABLE_AOCL_SPARSE=OFF -DMATX_ENABLE_OPENBLAS=ON

make -j{nproc}
