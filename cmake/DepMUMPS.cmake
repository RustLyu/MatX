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
        --parallel 1

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
    add_library(MUMPS::MUMPS STATIC IMPORTED GLOBAL)
if(WIN32)
    set_target_properties(MUMPS::MUMPS PROPERTIES
        IMPORTED_LOCATION
        "${DEPEND_LIB_OUTPUT}/mumps/lib/libdmumps.dll.a"

        INTERFACE_INCLUDE_DIRECTORIES
        "${DEPEND_LIB_OUTPUT}/mumps/include"
    )
else()
    set_target_properties(MUMPS::MUMPS PROPERTIES
        IMPORTED_LOCATION
        "${DEPEND_LIB_OUTPUT}/mumps/lib/libdmumps.so"

        INTERFACE_INCLUDE_DIRECTORIES
        "${DEPEND_LIB_OUTPUT}/mumps/include"
    )
endif()
if(WIN32)
    set(mumps_lib
        "${DEPEND_LIB_OUTPUT}/mumps/lib/libdmumps.dll.a"
        "${DEPEND_LIB_OUTPUT}/mumps/lib/libzmumps.dll.a"
        "${DEPEND_LIB_OUTPUT}/mumps/lib/libmumps_common.dll.a"
        "${DEPEND_LIB_OUTPUT}/mumps/lib/libmpiseq_fortran.dll.a"
        "${DEPEND_LIB_OUTPUT}/mumps/lib/libmpiseq_c.dll.a"
        "${DEPEND_LIB_OUTPUT}/mumps/lib/libpord.dll.a"
    )
else()
    set(mumps_lib
        "${DEPEND_LIB_OUTPUT}/mumps/lib/libdmumps.so"
        "${DEPEND_LIB_OUTPUT}/mumps/lib/libzmumps.so"
        "${DEPEND_LIB_OUTPUT}/mumps/lib/libmumps_common.so"
        "${DEPEND_LIB_OUTPUT}/mumps/lib/libmpiseq_fortran.so"
        "${DEPEND_LIB_OUTPUT}/mumps/lib/libmpiseq_c.so"
        "${DEPEND_LIB_OUTPUT}/mumps/lib/libpord.so"
    )
endif()

endif()
