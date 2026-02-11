#
# Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All rights reserved
#

## Common config
option (BUILD_RNGEXAMPLE_OUTOFSRCTREE
        "Allow out-of-source tree build for rngexample" OFF)


if (BUILD_RNGEXAMPLE_OUTOFSRCTREE)

  cmake_minimum_required (VERSION 3.22)

  project (RNGEXPERF LANGUAGES Fortran C)

  option (BUILD_ILP64 "ILP64 Support" OFF)
  option (BUILD_WITH_DLL      "Build example src with Dynamic AOCL-RNG"  ON)
  option (BUILD_WITH_STATICLIB "Build example src with Static AOCL-RNG" OFF)

  set (AOCL_ROOT  "${CMAKE_CURRENT_SOURCE_DIR}/../.."
       CACHE PATH "AOCL_ROOT path where AOCL libraries are installed")

  if ((NOT BUILD_WITH_DLL) AND (NOT BUILD_WITH_STATICLIB))
    message (FATAL_ERROR "BUILD_WITH_DLL & BUILD_WITH_STATICLIB both cannot be OFF !!!!!\n")
  endif ()

endif (BUILD_RNGEXAMPLE_OUTOFSRCTREE)


############################################################################
#####  Source listing  #####################################################
############################################################################

set (EXAMPLE_F_SOURCES
     dranduniform_example.f
     drandinitializeuser_example.f
     drandskipleap_example.f
)

set (EXAMPLE_C_SOURCES
     dranduniform_c_example.c
     drandmtuniform_c_example.c
     drandsfmtuniform_c_example.c
     srandsfmtuniform_c_example.c
     drandsbuniform_c_example.c
     drandinitializeuser_c_example.c
     drandsfmtgauss_c_example.c
     srandsfmtgauss_c_example.c
     drandskipleap_c_example.c
)
if (UNIX)
set (EXAMPLE_C_SOURCES ${EXAMPLE_C_SOURCES} dranduniformsecrng_c_example.c)
endif (UNIX)

set (EXAMPLE_PERF_F_SOURCES
     performance/time_dranduniform.f90
     performance/time_drandbeta.f90
)

## Final source var
set (RNGEXP_SOURCES ${EXAMPLE_C_SOURCES} ${EXAMPLE_F_SOURCES})
set (RNGPERF_SOURCES ${EXAMPLE_PERF_F_SOURCES})


############################################################################
#####  macros to build with dynamic lib ####################################
############################################################################

###########################################
##### macros to create rngexp targets #####
###########################################

macro (rngexp_dynamic)

  foreach (SOURCE_PATH IN ITEMS ${RNGEXP_SOURCES})

    get_filename_component (RNGEXP_EXE_NAME ${SOURCE_PATH} NAME_WE)
    set (RNGEXP_EXE_NAME_DYN "${RNGEXP_EXE_NAME}_dyn")
    add_executable (${RNGEXP_EXE_NAME_DYN} ${SOURCE_PATH})
    set (RNGEXPDYNTEST ${RNGEXPDYNTEST} ${RNGEXP_EXE_NAME_DYN})

    if ("${RNGEXP_EXE_NAME_DYN}" MATCHES "dranduniformsecrng_c_example_dyn")
      list (APPEND RNGEXP_DYNAMIC_COMMANDS
            COMMAND $<TARGET_FILE:${RNGEXP_EXE_NAME_DYN}> > ${RNGEXP_EXE_NAME_DYN}.res
            COMMAND ${CMAKE_COMMAND} -E cat ${RNGEXP_EXE_NAME_DYN}.res
      )
    else ()
      list (APPEND RNGEXP_DYNAMIC_COMMANDS
            COMMAND $<TARGET_FILE:${RNGEXP_EXE_NAME_DYN}> > ${RNGEXP_EXE_NAME_DYN}.res
            COMMAND ${CMAKE_COMMAND} -E cat ${RNGEXP_EXE_NAME_DYN}.res
            COMMAND ${CMAKE_COMMAND} -E copy
                            ${CMAKE_CURRENT_SOURCE_DIR}/${RNGEXP_EXE_NAME}.expected
                            ${EXPECTED_DATA_COPY_PATH}
            COMMAND ${CMAKE_COMMAND} -E compare_files --ignore-eol
                            ${RNGEXP_EXE_NAME_DYN}.res ${RNGEXP_EXE_NAME}.expected ||
                            echo "===== ${RNGEXP_EXE_NAME_DYN}----- FAILED ====="
      )
    endif ()

    target_link_libraries (${RNGEXP_EXE_NAME_DYN}
                           PRIVATE ${AOCLRNG_DLL_LIB} ${AOCLSECRNG_DLL_LIB})
    target_include_directories (${RNGEXP_EXE_NAME_DYN} PRIVATE ${AOCLRNGnSECRNG_INC})
    target_compile_options (${RNGEXP_EXE_NAME_DYN} PRIVATE
      "$<$<COMPILE_LANGUAGE:C>:$<$<BOOL:${BUILD_ILP64}>:${RNGEXP_CFLAGS}>>")
    target_compile_options (${RNGEXP_EXE_NAME_DYN} PRIVATE
      "$<$<COMPILE_LANGUAGE:Fortran>:$<$<BOOL:${BUILD_ILP64}>:${RNGEXP_FFLAGS}>>")

  endforeach ()

endmacro (rngexp_dynamic)

#############################################
##### macros to create rngperf targets ######
#############################################

macro (rngperf_dynamic)

  foreach (SOURCE_PATH IN ITEMS ${RNGPERF_SOURCES})

    get_filename_component (RNGPERF_EXE_NAME ${SOURCE_PATH} NAME_WE)
    set (RNGPERF_EXE_NAME_DYN "${RNGPERF_EXE_NAME}_dyn")
    add_executable (${RNGPERF_EXE_NAME_DYN} ${SOURCE_PATH})
    set (RNGPERFDYNTEST ${RNGPERFDYNTEST} ${RNGPERF_EXE_NAME_DYN})

    list (APPEND RNGPERF_DYNAMIC_COMMANDS
                 COMMAND echo "========================================="
                 COMMAND echo "========== ${RNGPERF_EXE_NAME_DYN} =========="
                 COMMAND $<TARGET_FILE:${RNGPERF_EXE_NAME_DYN}>
                 COMMAND echo "========================================="
    )

    target_link_libraries (${RNGPERF_EXE_NAME_DYN}
                           PRIVATE ${AOCLRNG_DLL_LIB} ${AOCLSECRNG_DLL_LIB})
    target_include_directories (${RNGPERF_EXE_NAME_DYN} PRIVATE ${AOCLRNGnSECRNG_INC})
    target_compile_options (${RNGPERF_EXE_NAME_DYN} PRIVATE
      "$<$<COMPILE_LANGUAGE:Fortran>:$<$<BOOL:${BUILD_ILP64}>:${RNGEXP_FFLAGS}>>")

  endforeach ()

endmacro (rngperf_dynamic)


############################################################################
#####  macros to build with static lib #####################################
############################################################################

############################################
#####  macros to create rngexp targets #####
############################################

macro (rngexp_static)

  foreach (SOURCE_PATH IN ITEMS ${RNGEXP_SOURCES})

    get_filename_component (RNGEXP_EXE_NAME ${SOURCE_PATH} NAME_WE)
    set (RNGEXP_EXE_NAME_STA "${RNGEXP_EXE_NAME}_sta")
    add_executable (${RNGEXP_EXE_NAME_STA} ${SOURCE_PATH})
    set (RNGEXPSTATTEST ${RNGEXPSTATTEST} ${RNGEXP_EXE_NAME_STA})

    if ("${RNGEXP_EXE_NAME_STA}" MATCHES "dranduniformsecrng_c_example_sta")
      list (APPEND RNGEXP_STATIC_COMMANDS
            COMMAND $<TARGET_FILE:${RNGEXP_EXE_NAME_STA}> > ${RNGEXP_EXE_NAME_STA}.res
            COMMAND ${CMAKE_COMMAND} -E cat ${RNGEXP_EXE_NAME_STA}.res
      )
    else ()
      list (APPEND RNGEXP_STATIC_COMMANDS
            COMMAND $<TARGET_FILE:${RNGEXP_EXE_NAME_STA}> > ${RNGEXP_EXE_NAME_STA}.res
            COMMAND ${CMAKE_COMMAND} -E cat ${RNGEXP_EXE_NAME_STA}.res
            COMMAND ${CMAKE_COMMAND} -E copy
                            ${CMAKE_CURRENT_SOURCE_DIR}/${RNGEXP_EXE_NAME}.expected
                            ${EXPECTED_DATA_COPY_PATH}
            COMMAND ${CMAKE_COMMAND} -E compare_files --ignore-eol
                            ${RNGEXP_EXE_NAME_STA}.res ${RNGEXP_EXE_NAME}.expected ||
                            echo "===== ${RNGEXP_EXE_NAME_STA}----- FAILED ====="
      )
    endif ()

    target_link_libraries (${RNGEXP_EXE_NAME_STA}
                           PRIVATE ${AOCLRNG_AR_LIB} ${AOCLSECRNG_AR_LIB})
    target_link_libraries (${RNGEXP_EXE_NAME_STA} PRIVATE ${RNGEXP_DEP_AR_LIBS})
    target_include_directories (${RNGEXP_EXE_NAME_STA} PRIVATE ${AOCLRNGnSECRNG_INC})
    target_compile_options (${RNGEXP_EXE_NAME_STA} PRIVATE
      "$<$<COMPILE_LANGUAGE:C>:$<$<BOOL:${BUILD_ILP64}>:${RNGEXP_CFLAGS}>>")
    target_compile_options (${RNGEXP_EXE_NAME_STA} PRIVATE
      "$<$<COMPILE_LANGUAGE:Fortran>:$<$<BOOL:${BUILD_ILP64}>:${RNGEXP_FFLAGS}>>")
    if (MSVC)
      target_compile_options (${RNGEXP_EXE_NAME_STA} PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:$<$<BOOL:${BUILD_ILP64}>:${RNGEXP_CFLAGS}>>")
    endif (MSVC)

  endforeach ()

endmacro (rngexp_static)

############################################
#####  macros to create rngperf targets ####
############################################

macro (rngperf_static)

  foreach (SOURCE_PATH IN ITEMS ${RNGPERF_SOURCES})

    get_filename_component (RNGPERF_EXE_NAME ${SOURCE_PATH} NAME_WE)
    set (RNGPERF_EXE_NAME_STA "${RNGPERF_EXE_NAME}_sta")
    add_executable (${RNGPERF_EXE_NAME_STA} ${SOURCE_PATH})
    set (RNGPERFSTATTEST ${RNGPERFSTATTEST} ${RNGPERF_EXE_NAME_STA})

    list (APPEND RNGPERF_STATIC_COMMANDS
                 COMMAND echo "========================================="
                 COMMAND echo "========== ${RNGPERF_EXE_NAME_STA} =========="
                 COMMAND $<TARGET_FILE:${RNGPERF_EXE_NAME_STA}>
                 COMMAND echo "========================================="
    )

    target_link_libraries (${RNGPERF_EXE_NAME_STA}
                           PRIVATE ${AOCLRNG_AR_LIB} ${AOCLSECRNG_AR_LIB})
    target_link_libraries (${RNGPERF_EXE_NAME_STA} PRIVATE ${RNGEXP_DEP_AR_LIBS})
    target_include_directories (${RNGPERF_EXE_NAME_STA} PRIVATE ${AOCLRNGnSECRNG_INC})
    target_compile_options (${RNGPERF_EXE_NAME_STA} PRIVATE
      "$<$<COMPILE_LANGUAGE:Fortran>:$<$<BOOL:${BUILD_ILP64}>:${RNGEXP_FFLAGS}>>")

  endforeach ()

endmacro (rngperf_static)


############################################################################
#####  Linux-specific out-of-source tree build  ############################
############################################################################

if (UNIX AND BUILD_RNGEXAMPLE_OUTOFSRCTREE)

    ## compiler options used for ILP64
  set (RNGEXP_CFLAGS -DINTEGER64)
  set (RNGEXP_FFLAGS -DINTEGER64 -fdefault-integer-8)


  set (EXPECTED_DATA_COPY_PATH ${CMAKE_CURRENT_BINARY_DIR}
       CACHE PATH "expected data copy path to compare with result")


    ## set header and library path
  if (BUILD_ILP64)
      ## set include path
    set (AOCL_INC_PATH ${AOCL_ROOT}/include_ILP64)
      ## set library path
    set (AOCL_LIB_PATH ${AOCL_ROOT}/lib_ILP64)

  else (NOT BUILD_ILP64)

      ## set include path
    set (AOCL_INC_PATH ${AOCL_ROOT}/include_LP64)
      ## set library path
    set (AOCL_LIB_PATH ${AOCL_ROOT}/lib_LP64)
  endif (BUILD_ILP64)


    ## set environment variable LD_LIBRARY_PATH
  if (BUILD_WITH_DLL)
    set (ENV{LD_LIBRARY_PATH} "${AOCL_LIB_PATH}:$ENV{LD_LIBRARY_PATH}")
  endif (BUILD_WITH_DLL)


  ###################################
  ##### using dynamic libraries #####
  ###################################

  if (BUILD_WITH_DLL)

    ## RNG header file path
  set (AOCLRNGnSECRNG_INC ${AOCL_INC_PATH})
    ## RNG DLL library
  set (AOCLRNG_DLL_LIB ${AOCL_LIB_PATH}/librng_amd.so)
    ## SecureRNG DLL library
  set (AOCLSECRNG_DLL_LIB ${AOCL_LIB_PATH}/libamdsecrng.so)

    ## LibM DLL library, rng depends on libm
  set (RNGEXTLIBMDYN ${AOCL_LIB_PATH}/libamdlibm.so)
    ## BLIS DLL library, rng depends on blis
  set (RNGEXTBLISDYN ${AOCL_LIB_PATH}/libblis.so)


    ## Check if all libries are installled
  if (NOT IS_DIRECTORY ${AOCLRNGnSECRNG_INC})
    message (FATAL_ERROR "Missing include directory: ${AOCLRNGnSECRNG_INC}")
  endif ()
  if (NOT EXISTS ${AOCLRNG_DLL_LIB})
    message (FATAL_ERROR "Missing AOCL-RNG lib: ${AOCLRNG_DLL_LIB}")
  endif ()
  if (NOT EXISTS ${AOCLSECRNG_DLL_LIB})
    message (FATAL_ERROR "Missing AOCL-SecureRNG lib: ${AOCLSECRNG_DLL_LIB}")
  endif ()
  if (NOT EXISTS ${RNGEXTLIBMDYN})
    message (FATAL_ERROR "Missing AOCL-LIBM lib: ${RNGEXTLIBMDYN}")
  endif ()
  if (NOT EXISTS ${RNGEXTBLISDYN})
    message (FATAL_ERROR "Missing AOCL-BLIS lib: ${RNGEXTBLISDYN}")
  endif ()


    ## Build and Run Sanity Tests with dynamic RNG on Lin
  rngexp_dynamic()
  rngperf_dynamic()

    ##
  list (PREPEND RNGEXP_DYNAMIC_COMMANDS
                COMMAND echo "========================================================"
                COMMAND echo "======= Run programs with Dynamic RNG Lib on Lin ======="
                COMMAND echo "========================================================"
  )

  endif (BUILD_WITH_DLL)


  ##################################
  ##### using static libraries #####
  ##################################

  if (BUILD_WITH_STATICLIB)

    ## RNG header file path
  set (AOCLRNGnSECRNG_INC ${AOCL_INC_PATH})
    ## RNG Static library
  set (AOCLRNG_AR_LIB ${AOCL_LIB_PATH}/librng_amd.a)
    ## SecureRNG Static library
  set (AOCLSECRNG_AR_LIB ${AOCL_LIB_PATH}/libamdsecrng.a)

    ## LibM Static library, rng depends on libm
  set (RNGEXTLIBMAR ${AOCL_LIB_PATH}/libamdlibm.a)
    ## UTILS Static library, libm depends on utils
  set (RNGEXTLIBUTILSAR ${AOCL_LIB_PATH}/libaoclutils.a)
    ## BLIS Static library, rng depends on blis
  set (RNGEXTBLISAR ${AOCL_LIB_PATH}/libblis.a)


    ## Check if all libries are installled
  if (NOT IS_DIRECTORY ${AOCLRNGnSECRNG_INC})
    message (FATAL_ERROR "Missing include directory: ${AOCLRNGnSECRNG_INC}")
  endif ()
  if (NOT EXISTS ${AOCLRNG_AR_LIB})
    message (FATAL_ERROR "Missing AOCL-RNG lib: ${AOCLRNG_AR_LIB}")
  endif ()
  if (NOT EXISTS ${AOCLSECRNG_AR_LIB})
    message (FATAL_ERROR "Missing AOCL-SecureRNG lib: ${AOCLSECRNG_AR_LIB}")
  endif ()
  if (NOT EXISTS ${RNGEXTLIBMAR})
    message (FATAL_ERROR "Missing AOCL-LIBM lib: ${RNGEXTLIBMAR}")
  endif ()
  if (NOT EXISTS ${RNGEXTLIBUTILSAR})
    message (FATAL_ERROR "Missing AOCL-UTILS lib: ${RNGEXTLIBUTILSAR}")
  endif ()
  if (NOT EXISTS ${RNGEXTBLISAR})
    message (FATAL_ERROR "Missing AOCL-BLIS lib: ${RNGEXTBLISAR}")
  endif ()


    ## list dependency static libraries
  set (RNGEXP_DEP_AR_LIBS
       ${RNGEXTBLISAR} ${RNGEXTLIBMAR} ${RNGEXTLIBUTILSAR}
       pthread m stdc++)

    ## Build and Run Sanity Tests with static RNG on Lin
  rngexp_static()
  rngperf_static()

    ##
  list (PREPEND RNGEXP_STATIC_COMMANDS
                COMMAND echo "======================================================="
                COMMAND echo "======= Run programs with Static RNG Lib on Lin ======="
                COMMAND echo "======================================================="
  )

  endif (BUILD_WITH_STATICLIB)


  ###################################################
  ##### common for dynamic and static libraries #####
  ###################################################

    ## clear out dynamic-lib based command list
  if (NOT BUILD_WITH_DLL)
    set (RNGEXP_DYNAMIC_COMMANDS "")
    set (RNGPERF_DYNAMIC_COMMANDS "")
    set (RNGEXPDYNTEST "")
    set (RNGPERFDYNTEST "")
  endif ()

    ## clear out static-lib based command list
  if (NOT BUILD_WITH_STATICLIB)
    set (RNGEXP_STATIC_COMMANDS "")
    set (RNGPERF_STATIC_COMMANDS "")
    set (RNGEXPSTATTEST "")
    set (RNGPERFSTATTEST "")
  endif ()


    ## Custom cmake target
  add_custom_target (rngexample
                     ${RNGEXP_STATIC_COMMANDS}
                     ${RNGPERF_STATIC_COMMANDS}
                     ${RNGEXP_DYNAMIC_COMMANDS}
                     ${RNGPERF_DYNAMIC_COMMANDS}
                     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                     COMMENT "Build & Run test for AOCL-RNG Library"
                     VERBATIM
  )
  add_dependencies (rngexample ${RNGEXPSTATTEST} ${RNGPERFSTATTEST}
                               ${RNGEXPDYNTEST}  ${RNGPERFDYNTEST})

endif (UNIX AND BUILD_RNGEXAMPLE_OUTOFSRCTREE)


############################################################################
#####  Windows-specific based on Visual Studio out-of-source tree build  ###
############################################################################

if (MSVC AND BUILD_RNGEXAMPLE_OUTOFSRCTREE)

    ## Change default installation path
  if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install"
         CACHE PATH "New defualt Intallation Path" FORCE)
  endif ()


    ## compiler options used for ILP64
  set (WIN_COMPFLAGS -DWIN64 -DWIN_IFORT)
  add_compile_options (${WIN_COMPFLAGS})

  set (RNGEXP_CFLAGS -DINTEGER64)
  set (RNGEXP_FFLAGS -DINTEGER64 -integer-size:64)


  set (EXPECTED_DATA_COPY_PATH "${CMAKE_INSTALL_PREFIX}/bin"
       CACHE PATH "expected data copy path to compare with result")


    ## set header and library path
  if (BUILD_ILP64)
      ## set include path
    set (AOCL_RNGINC_PATH ${AOCL_ROOT}/amd-rng/include)
      ## set library path
    set (AOCL_RNGLIB_PATH ${AOCL_ROOT}/amd-rng/lib/ILP64)
    set (AOCL_LIBMLIB_PATH ${AOCL_ROOT}/amd-libm/lib)
    set (AOCL_UTILSLIB_PATH ${AOCL_ROOT}/amd-utils/lib)
    set (AOCL_BLISLIB_PATH ${AOCL_ROOT}/amd-blis/lib/ILP64)

  else (NOT BUILD_ILP64)

      ## set include path
    set (AOCL_RNGINC_PATH ${AOCL_ROOT}/amd-rng/include)
      ## set library path
    set (AOCL_RNGLIB_PATH ${AOCL_ROOT}/amd-rng/lib/LP64)
    set (AOCL_LIBMLIB_PATH ${AOCL_ROOT}/amd-libm/lib)
    set (AOCL_UTILSLIB_PATH ${AOCL_ROOT}/amd-utils/lib)
    set (AOCL_BLISLIB_PATH ${AOCL_ROOT}/amd-blis/lib/LP64)
  endif (BUILD_ILP64)


  ##################################
  ##### using dynamic libraries ####
  ##################################

  if (BUILD_WITH_DLL)

    ## RNG header file path
  set (AOCLRNGnSECRNG_INC ${AOCL_RNGINC_PATH})
    ## RNG dll and import library
  set (AOCLRNG_DLL ${AOCL_RNGLIB_PATH}/rng_amd.dll)
  set (AOCLRNG_DLL_LIB ${AOCL_RNGLIB_PATH}/rng_amd.lib)
    ## SecureRNG dll and import library
  set (AOCLSECRNG_DLL ${AOCL_RNGLIB_PATH}/amdsecrng.dll)
  set (AOCLSECRNG_DLL_LIB ${AOCL_RNGLIB_PATH}/amdsecrng.lib)

    ## LibM dll and import library, rng depends on libm
  set (RNGEXTLIBMDYN  ${AOCL_LIBMLIB_PATH}/libalm.dll)
  set (RNGEXTLIBMDLIB ${AOCL_LIBMLIB_PATH}/libalm.lib)
    ## BLIS dll and import library, rng depends on blis
  set (RNGEXTBLISDYN  ${AOCL_BLISLIB_PATH}/AOCL-LibBlis-Win-dll.dll)
  set (RNGEXTBLISDLIB ${AOCL_BLISLIB_PATH}/AOCL-LibBlis-Win-dll.lib)


    ## Check if all libries are installled
  if (NOT IS_DIRECTORY ${AOCLRNGnSECRNG_INC})
    message (FATAL_ERROR "Missing include directory: ${AOCLRNGnSECRNG_INC}")
  endif ()
  if (NOT EXISTS ${AOCLRNG_DLL})
    message (FATAL_ERROR "Missing AOCL-RNG lib: ${AOCLRNG_DLL}")
  endif ()
  if (NOT EXISTS ${AOCLRNG_DLL_LIB})
    message (FATAL_ERROR "Missing AOCL-RNG import lib: ${AOCLRNG_DLL_LIB}")
  endif ()
  if (NOT EXISTS ${AOCLSECRNG_DLL})
    message (FATAL_ERROR "Missing AOCL-SecureRNG lib: ${AOCLSECRNG_DLL}")
  endif ()
  if (NOT EXISTS ${AOCLSECRNG_DLL_LIB})
    message (FATAL_ERROR "Missing AOCL-SecureRNG import lib: ${AOCLSECRNG_DLL_LIB}")
  endif ()
  if (NOT EXISTS ${RNGEXTLIBMDYN})
    message (FATAL_ERROR "Missing AOCL-LIBM lib: ${RNGEXTLIBMDYN}")
  endif ()
  if (NOT EXISTS ${RNGEXTLIBMDLIB})
    message (FATAL_ERROR "Missing AOCL-LIBM import lib: ${RNGEXTLIBMDLIB}")
  endif ()
  if (NOT EXISTS ${RNGEXTBLISDYN})
    message (FATAL_ERROR "Missing AOCL-BLIS lib: ${RNGEXTBLISDYN}")
  endif ()
  if (NOT EXISTS ${RNGEXTBLISDLIB})
    message (FATAL_ERROR "Missing AOCL-BLIS import lib: ${RNGEXTBLISDLIB}")
  endif ()


    ## Build and Run Sanity Tests with dynamic RNG on Lin
  rngexp_dynamic()
  rngperf_dynamic()

    ## add print
  list (PREPEND RNGEXP_DYNAMIC_COMMANDS
                COMMAND echo "========================================================"
                COMMAND echo "======= Run programs with Dynamic RNG Lib on Win ======="
                COMMAND echo "========================================================"
  )

    ## install exe, lib, dll to given path
  install (TARGETS ${RNGEXPDYNTEST}    DESTINATION bin)
  install (TARGETS ${RNGPERFDYNTEST}   DESTINATION bin)

  install (FILES
           ${AOCLRNG_DLL}    ${AOCLRNG_DLL_LIB}
           ${AOCLSECRNG_DLL} ${AOCLSECRNG_DLL_LIB}
           ${RNGEXTLIBMDYN}  ${RNGEXTLIBMDLIB}
           ${RNGEXTBLISDYN}  ${RNGEXTBLISDLIB}
           DESTINATION bin)

  endif (BUILD_WITH_DLL)


  ##################################
  ##### using static libraries #####
  ##################################

  if (BUILD_WITH_STATICLIB)

    ## RNG header file path
  set (AOCLRNGnSECRNG_INC ${AOCL_RNGINC_PATH})
    ## RNG Static library
  set (AOCLRNG_AR_LIB ${AOCL_RNGLIB_PATH}/rng_amd-static.lib)
    ## SecureRNG Static library
  set (AOCLSECRNG_AR_LIB ${AOCL_RNGLIB_PATH}/amdsecrng-static.lib)

    ## LibM Static library, rng depends on libm
  set (RNGEXTLIBMAR   ${AOCL_LIBMLIB_PATH}/libalm-static.lib)
    ## UTILS Static library, libm depends on utils
  set (RNGEXTLIBUTILSAR ${AOCL_UTILSLIB_PATH}/libaoclutils_static.lib)
    ## BLIS Static library, rng depends on blis
  set (RNGEXTBLISAR   ${AOCL_BLISLIB_PATH}/AOCL-LibBlis-Win.lib)


    ## Check if all libries are installled
  if (NOT IS_DIRECTORY ${AOCLRNGnSECRNG_INC})
    message (FATAL_ERROR "Missing include directory: ${AOCLRNGnSECRNG_INC}")
  endif ()
  if (NOT EXISTS ${AOCLRNG_AR_LIB})
    message (FATAL_ERROR "Missing AOCL-RNG lib: ${AOCLRNG_AR_LIB}")
  endif ()
  if (NOT EXISTS ${AOCLSECRNG_AR_LIB})
    message (FATAL_ERROR "Missing AOCL-SecureRNG lib: ${AOCLSECRNG_AR_LIB}")
  endif ()
  if (NOT EXISTS ${RNGEXTLIBMAR})
    message (FATAL_ERROR "Missing AOCL-LIBM lib: ${RNGEXTLIBMAR}")
  endif ()
  if (NOT EXISTS ${RNGEXTLIBUTILSAR})
    message (FATAL_ERROR "Missing AOCL-UTILS lib: ${RNGEXTLIBUTILSAR}")
  endif ()
  if (NOT EXISTS ${RNGEXTBLISAR})
    message (FATAL_ERROR "Missing AOCL-BLIS lib: ${RNGEXTBLISAR}")
  endif ()


    ## list dependency static libraries
  set (RNGEXP_DEP_AR_LIBS ${RNGEXTBLISAR} ${RNGEXTLIBMAR} ${RNGEXTLIBUTILSAR})

    ## include Intel oneAPI ifort library path
  find_library (IFORT_LIB_PATH ifconsol.lib)
  if (IFORT_LIB_PATH-NOTFOUND)
    message (FATAL_ERROR "Intel oneAPI installation path is not found !!!!!\n")
  endif ()

  cmake_path (REMOVE_FILENAME IFORT_LIB_PATH OUTPUT_VARIABLE IFORT_LIB_PATH)
  link_directories (${IFORT_LIB_PATH})

    ## Build and Run Sanity Tests with static RNG on Lin
  rngexp_static()
  rngperf_static()

    ## add print
  list (PREPEND RNGEXP_STATIC_COMMANDS
                COMMAND echo "======================================================="
                COMMAND echo "======= Run programs with Static RNG Lib on Win ======="
                COMMAND echo "======================================================="
  )


    ## install exe to given path
  install (TARGETS ${RNGEXPSTATTEST}   DESTINATION bin)
  install (TARGETS ${RNGPERFSTATTEST}  DESTINATION bin)

  endif (BUILD_WITH_STATICLIB)


  ###################################################
  ##### common for dynamic and static libraries #####
  ###################################################

    ## clear out dynamic-lib based command list
  if (NOT BUILD_WITH_DLL)
    set (RNGEXP_DYNAMIC_COMMANDS "")
    set (RNGPERF_DYNAMIC_COMMANDS "")
    set (RNGEXPDYNTEST "")
    set (RNGPERFDYNTEST "")
  endif ()

    ## clear out static-lib based command list
  if (NOT BUILD_WITH_STATICLIB)
    set (RNGEXP_STATIC_COMMANDS "")
    set (RNGPERF_STATIC_COMMANDS "")
    set (RNGEXPSTATTEST "")
    set (RNGPERFSTATTEST "")
  endif ()


    ## Custom cmake target
  add_custom_target (rngexample
                     COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
                                              --target install --config Release
                     COMMAND cd ${CMAKE_INSTALL_PREFIX}/bin
                     ${RNGEXP_STATIC_COMMANDS}
                     ${RNGPERF_STATIC_COMMANDS}
                     ${RNGEXP_DYNAMIC_COMMANDS}
                     ${RNGPERF_DYNAMIC_COMMANDS}
                     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                     COMMENT "Build & Run test for AOCL-RNG Library"
                     VERBATIM
  )
  add_dependencies (rngexample ${RNGEXPSTATTEST} ${RNGPERFSTATTEST}
                               ${RNGEXPDYNTEST}  ${RNGPERFDYNTEST})

endif (MSVC AND BUILD_RNGEXAMPLE_OUTOFSRCTREE)


############################################################################
#####  In-source tree build for example  ###################################
############################################################################
if (NOT BUILD_RNGEXAMPLE_OUTOFSRCTREE)

  set (RNGEXPCONFIGFILE
       "${PROJECT_SOURCE_DIR}/scripts/cmake/example_build_config.cmake"
       CACHE PATH "cmake script file path of in-src build")

  if (EXISTS ${RNGEXPCONFIGFILE})
    include (${RNGEXPCONFIGFILE})
  else ()
    message (FATAL_ERROR
             "\n"
             "$$$$$  ================================================ $$$$$\n"
	     "$$$$$  ============= IF (In-Source Build) ============= $$$$$\n"
             "$$$$$  CMake script file is missing for in-src build    $$$$$\n"
             "$$$$$  File path: ${RNGEXPCONFIGFILE}  $$$$$\n"
	     "$$$$$  ============= IF (Out-of-Source Build) ========= $$$$$\n"
             "$$$$$  Use cmake option -DBUILD_RNGEXAMPLE_OUTOFSRCTREE=ON $$\n"
             "$$$$$  ================================================ $$$$$\n"
    )
  endif ()

endif (NOT BUILD_RNGEXAMPLE_OUTOFSRCTREE)




## Configured Info
message (STATUS "\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
message (STATUS "@@ Filename: ${CMAKE_CURRENT_LIST_FILE} @@@\n")

message (STATUS "@@ Build Configuration for Examples Sources @@@@@@@@@@@@@@@@@@")
message (STATUS "   Current Generator <CMAKE_GENERATOR>..............${CMAKE_GENERATOR}")
message (STATUS "   Out-of-source tree build <BUILD_RNGEXAMPLE_OUTOFSRCTREE>..."
                                                  "${BUILD_RNGEXAMPLE_OUTOFSRCTREE}")
message (STATUS "   ILP64 Build Type <BUILD_ILP64>...................${BUILD_ILP64}")

if (MSVC AND BUILD_RNGEXAMPLE_OUTOFSRCTREE)
message (STATUS "   Compiler Options <WIN_COMPFLAGS>.................${WIN_COMPFLAGS}")
endif ()

if (BUILD_ILP64)
message (STATUS "   C- compiler Options <RNGEXP_CFLAGS>..............${RNGEXP_CFLAGS}")
message (STATUS "   Fortran compiler Options <RNGEXP_FFLAGS>.........${RNGEXP_FFLAGS}")
endif (BUILD_ILP64)

if (UNIX)
message (STATUS "   LD_LIBRARY_PATH<ENV{LD_LIBRARY_PATH}>............$ENV{LD_LIBRARY_PATH}")
endif (UNIX)

message (STATUS "   AOCLRNGnSECRNG inc path <AOCLRNGnSECRNG_INC>.....${AOCLRNGnSECRNG_INC}")

if ((BUILD_RNGEXAMPLE_OUTOFSRCTREE AND UNIX AND BUILD_WITH_DLL)
    OR
    (NOT BUILD_RNGEXAMPLE_OUTOFSRCTREE AND UNIX)
)
message (STATUS "   AOCL-RNG dyn lib path <AOCLRNG_DLL_LIB>..........${AOCLRNG_DLL_LIB}")
message (STATUS "   AOCL-SECRNG dyn lib path <AOCLSECRNG_DLL_LIB>....${AOCLSECRNG_DLL_LIB}")
message (STATUS "   AOCL-LIBM dyn lib path <RNGEXTLIBMDYN>...........${RNGEXTLIBMDYN}")
message (STATUS "   AOCL-BLIS dyn lib path <RNGEXTBLISDYN>...........${RNGEXTBLISDYN}")
endif ()

if ((BUILD_RNGEXAMPLE_OUTOFSRCTREE AND MSVC AND BUILD_WITH_DLL)
    OR
    (NOT BUILD_RNGEXAMPLE_OUTOFSRCTREE AND MSVC)
)
message (STATUS "   AOCL-RNG     dll path <AOCLRNG_DLL_LIB>..........${AOCLRNG_DLL_LIB}")
message (STATUS "   AOCL-RNG dll lib path <AOCLRNG_DLL>..............${AOCLRNG_DLL}")
message (STATUS "   AOCL-SECRNG     dll path <AOCLSECRNG_DLL_LIB>....${AOCLSECRNG_DLL_LIB}")
message (STATUS "   AOCL-SECRNG dll lib path <AOCLSECRNG_DLL>........${AOCLSECRNG_DLL}")
message (STATUS "   AOCL-LIBM     dll path <RNGEXTLIBMDYN>...........${RNGEXTLIBMDYN}")
message (STATUS "   AOCL-LIBM dll lib path <RNGEXTLIBMDLIB>..........${RNGEXTLIBMDLIB}")
message (STATUS "   AOCL-BLIS     dll path <RNGEXTBLISDYN>...........${RNGEXTBLISDYN}")
message (STATUS "   AOCL-BLIS dll lib path <RNGEXTBLISDLIB>..........${RNGEXTBLISDLIB}")
endif ()

if ((BUILD_RNGEXAMPLE_OUTOFSRCTREE AND BUILD_WITH_STATICLIB)
    OR
    (NOT BUILD_RNGEXAMPLE_OUTOFSRCTREE)
)
message (STATUS "   AOCL-RNG Static lib path <AOCLRNG_AR_LIB>........${AOCLRNG_AR_LIB}")
message (STATUS "   AOCL-SECRNG Static lib path <AOCLSECRNG_AR_LIB>..${AOCLSECRNG_AR_LIB}")
message (STATUS "   AOCL-LIBM Static lib path <RNGEXTLIBMAR>.........${RNGEXTLIBMAR}")
message (STATUS "   AOCL-UTILS Static lib path <RNGEXTLIBUTILSAR>....${RNGEXTLIBUTILSAR}")
message (STATUS "   AOCL-BLIS Static lib path <RNGEXTBLISAR>.........${RNGEXTBLISAR}")
message (STATUS "   Static lib dependency list <RNGEXP_DEP_AR_LIBS>..${RNGEXP_DEP_AR_LIBS}")
if (MSVC AND BUILD_RNGEXAMPLE_OUTOFSRCTREE)
message (STATUS "   Intel oneAPI lib path on Windows<IFORT_LIB_PATH>.${IFORT_LIB_PATH}")
endif (MSVC AND BUILD_RNGEXAMPLE_OUTOFSRCTREE)
endif ()

message (STATUS "\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
