# =========================================================
# DepMUMPS.cmake
# Fetch + build MUMPS (when MATX_ENABLE_MUMPS=ON)
# =========================================================

if(MATX_ENABLE_MUMPS)

    FetchContent_Declare(
        mumps
        GIT_REPOSITORY https://github.com/scivision/mumps-superbuild.git
        GIT_TAG v5.9.0.0
    )

FetchContent_GetProperties(mumps)

if(NOT mumps_POPULATED)
    FetchContent_Populate(mumps)
endif()

set(MUMPS_BUILD_DIR
    ${CMAKE_BINARY_DIR}/mumps-build)

file(MAKE_DIRECTORY
    ${MUMPS_BUILD_DIR})

execute_process(

    COMMAND
    ${CMAKE_COMMAND}

    -S ${mumps_SOURCE_DIR}
    -B ${MUMPS_BUILD_DIR}

    -DBUILD_SHARED_LIBS=OFF
    -DMUMPS_parallel=OFF
    -DBLAS_LIBRARIES=${BLAS_LIBRARIES}
    -DMUMPS_intsize64=ON
    -DBUILD_COMPLEX16=ON
    -DBUILD_COMPLEX=ON
    -DLAPACK_LIBRARY=${LAPACK_LIBRARIES}\;${BLAS_LIBRARIES}
    -DBUILD_SHARED_LIBS=ON
    -DCMAKE_INSTALL_PREFIX=${DEPEND_LIB_OUTPUT}/mumps

    RESULT_VARIABLE MUMPS_CONFIG_RESULT
)

if(NOT MUMPS_CONFIG_RESULT EQUAL 0)

    message(FATAL_ERROR
        "MUMPS configure failed")

endif()

execute_process(

    COMMAND
    ${CMAKE_COMMAND}
    --build ${MUMPS_BUILD_DIR}
    --parallel ${BUILD_JOBS}

    RESULT_VARIABLE MUMPS_BUILD_RESULT
)
execute_process(
    COMMAND ${CMAKE_COMMAND}
    --install ${MUMPS_BUILD_DIR}
    RESULT_VARIABLE MUMPS_INSTALL_RESULT
)

if(NOT MUMPS_BUILD_RESULT EQUAL 0)

    message(FATAL_ERROR
        "MUMPS build failed")

endif()

set(_mumps_libs
    dmumps
    zmumps
    mumps_common
    mpiseq_fortran
    mpiseq_c
    pord
)

unset(mumps_lib)
set (MUMPS_INSTALL_DIR ${DEPEND_LIB_OUTPUT}/mumps)
foreach(lib IN LISTS _mumps_libs)
    find_library(MUMPS_${lib}_LIBRARY
        NAMES ${lib}
        HINTS
        "${MUMPS_INSTALL_DIR}/lib"
        NO_DEFAULT_PATH
    )

if(NOT MUMPS_${lib}_LIBRARY)
    message(FATAL_ERROR
        "Cannot find MUMPS library: ${lib}")
endif()

list(APPEND mumps_lib
    "${MUMPS_${lib}_LIBRARY}"
)

endforeach()

add_library(MUMPS::MUMPS UNKNOWN IMPORTED GLOBAL)

set_target_properties(MUMPS::MUMPS PROPERTIES
    IMPORTED_LOCATION
    "${MUMPS_dmumps_LIBRARY}"

    INTERFACE_INCLUDE_DIRECTORIES
    "${MUMPS_INSTALL_DIR}/include"

    INTERFACE_LINK_LIBRARIES
    "${mumps_lib}"
)

endif()
