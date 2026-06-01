#!/bin/bash
matx_backend=OPENBLAS
if [ $1 -eq 0 ]; then
    matx_backend=OPENBLAS
elif [ $1 -eq 1 ]; then
    matx_backend=AMD_AOCL
else
    echo "./build.sh {0 | 1}"
fi
cmake ../../ -DCMAKE_BUILD_TYPE=DEBUG -DMATX_ENABLE_MUMPS=ON -DMATX_ENABLE_SUPERLU=ON -DMATX_BUILD_TESTS=ON -DMATX_BACKEND=$matx_backend
