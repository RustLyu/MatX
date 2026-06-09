# =========================================================
# DepOpenBLAS.cmake
# Fetch + build OpenBLAS (when MATX_ENABLE_OPENBLAS=ON)
# =========================================================

function(matx_add_openblas)

    include(FetchContent)

    FetchContent_Declare(
        OpenBLAS
        GIT_REPOSITORY https://github.com/OpenMathLib/OpenBLAS.git
        GIT_TAG v0.3.33
    )

    FetchContent_GetProperties(OpenBLAS)

    if(NOT OpenBLAS_POPULATED)
        FetchContent_Populate(OpenBLAS)
    endif()
    set(OPENBLAS_BUILD_DIR
        ${CMAKE_BINARY_DIR}/openblas-build)

    file(MAKE_DIRECTORY ${OPENBLAS_BUILD_DIR})

    execute_process(
        COMMAND
        ${CMAKE_COMMAND}
        -S ${openblas_SOURCE_DIR}
        -B ${OPENBLAS_BUILD_DIR}
        -DBUILD_SHARED_LIBS=ON
		-DCMAKE_BUILD_TYPE=Release
        -DCMAKE_INSTALL_PREFIX=${DEPEND_LIB_OUTPUT}/openblas

        RESULT_VARIABLE OPENBLAS_CONFIG_RESULT
    )

    if(NOT OPENBLAS_CONFIG_RESULT EQUAL 0)
        message(FATAL_ERROR "OpenBLAS configure failed")
    endif()

    execute_process(
        COMMAND
        ${CMAKE_COMMAND}
        --build ${OPENBLAS_BUILD_DIR}
		--config Release
		--parallel ${BUILD_JOBS}

        RESULT_VARIABLE OPENBLAS_BUILD_RESULT
    )
    execute_process(
        COMMAND ${CMAKE_COMMAND}
        --install ${OPENBLAS_BUILD_DIR}
        RESULT_VARIABLE OPENBLAS_INSTALL_RESULT
    )

    if(NOT OPENBLAS_BUILD_RESULT EQUAL 0)
        message(FATAL_ERROR "OpenBLAS build failed")
    endif()
    if(WIN32)

        if(EXISTS ${OPENBLAS_BUILD_DIR}/lib/libopenblas.dll.a)
            set(OPENBLAS_LIB
                ${OPENBLAS_BUILD_DIR}/lib/libopenblas.dll.a)

        elseif(EXISTS ${OPENBLAS_BUILD_DIR}/lib/libopenblas.a)
            set(OPENBLAS_LIB
                ${OPENBLAS_BUILD_DIR}/lib/libopenblas.a)

        else()
            message(FATAL_ERROR
                "Cannot find built OpenBLAS library")
        endif()

    else()

        set(OPENBLAS_LIB
            ${DEPEND_LIB_OUTPUT}/openblas/lib/libopenblas.so)

    endif()

    set(BLAS_LIBRARIES
        ${OPENBLAS_LIB}
        PARENT_SCOPE)

    set(LAPACK_LIBRARIES
        ${OPENBLAS_LIB}
        PARENT_SCOPE)

    message(STATUS "OpenBLAS lib: ${OPENBLAS_LIB}")

endfunction()

# ---- Conditionally build OpenBLAS and make it available ----
if(MATX_ENABLE_OPENBLAS)
    matx_add_openblas()
    set(OpenBLAS_DIR ${DEPEND_LIB_OUTPUT}/openblas/lib/cmake/OpenBLAS)
    find_package(OpenBLAS REQUIRED)
endif()
