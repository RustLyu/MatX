**************************************************************************
README for AOCL-RNG Users
=========================
**************************************************************************
***Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.***
**************************************************************************


This directory contains example and performance programs which uses
AOCL-RNG library. AOCL-RNG has dependency on AOCL-LibM and AOCL-BLIS
libraries. Hence both of these libraries must also be installed.
Cmake script is provided to compile and run the programs on both
linux and windows platforms.

By default, programs are build with dynamic AOCL-RNG library.
Using proper cmake configs same programs could also be build with static library.


Build on Linux
--------------

1. Move to installed examples directory,

   `$ cd amd-rng/examples`

2. Use one of the cmake config command,

   - Compiler: ***gcc, gfortran***, Integer Size: ***LP64***

   `$ CC=gcc FC=gfortran cmake -H. -B build -DBUILD_RNGEXAMPLE_OUTOFSRCTREE=ON`

   - Compiler: ***AOCC's clang, flang***, Integer Size: ***LP64***

   `$ CC=clang FC=flang cmake -H. -B build -DBUILD_RNGEXAMPLE_OUTOFSRCTREE=ON`

3. Cmake build/run command,

   `$ cmake --build build --target rngexample`


>
>
>
>  ***Notes***
>
>  Under cmake script, environment variable **LD_LIBRARY_PATH** is updated
>  with AOCL library installation path on linux.
>
>  - Use below command to set LD_LIBRARY_PATH if needed,
>
>    `$ export LD_LIBRARY_PATH=$AOCL_ROOT/lib:$LD_LIBRARY_PATH`
>
>
>


Build on Windows
----------------

1. Move to installed examples directory,

   `$ cd amd-rng/examples`

2. Use cmake config command,

   - Compiler: ***LLVM's clang-cl, Intel's ifort***, Interger Size: ***LP64***

   ```
    $ cmake -S. -B build -G "Visual Studio 17 2022" -A x64 -T "ClangCL"
            -DCMAKE_C_COMPILER=clang-cl -DCMAKE_Fortran_COMPILER=ifort
            -DBUILD_RNGEXAMPLE_OUTOFSRCTREE=ON
   ```


3. Cmake build/run command,

   `$ cmake --build build --target rngexample --config Release`


CMake config options
--------------------


  |       Config Parameters       |      Description      | CMake Config Commandline Parameter |
  |-------------------------------|-----------------------|------------------------------------|
  | BUILD_RNGEXAMPLE_OUTOFSRCTREE | Always Enabled        | -DBUILD_RNGEXAMPLE_OUTOFSRCTREE=ON |
  | BUILD_ILP64                   | Select 64-bit Integer          | -DBUILD_ILP64=ON          |
  | BUILD_WITH_DLL                | Disable build with Dynamic lib | -DBUILD_WITH_DLL=OFF      |
  | BUILD_WITH_STATICLIB          | Enable build with Static lib   | -DBUILD_WITH_STATICLIB=ON |


**************************************************************************
**************************************************************************
