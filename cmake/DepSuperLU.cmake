# =========================================================
# DepSuperLU.cmake
# Fetch + build SuperLU (when MATX_ENABLE_SUPERLU=ON)
# =========================================================

# SuperLU 7.0.1 declares its BLAS integer arguments as 32-bit `int`, while
# MatX's OPENBLAS backend is built with INTERFACE64=1. Linking these interfaces
# makes SuperLU pass invalid dimensions to OpenBLAS64 (for example DTRSV's LDA).
if(MATX_ENABLE_SUPERLU AND MATX_BACKEND STREQUAL "OPENBLAS")
    message(WARNING
        "Disabling SuperLU: SuperLU 7.0.1 uses LP64 BLAS arguments, but the MatX OPENBLAS backend uses ILP64.")
    set(MATX_ENABLE_SUPERLU OFF)
endif()

if(MATX_ENABLE_SUPERLU)

    FetchContent_Declare(
        superlu
        GIT_REPOSITORY https://github.com/xiaoyeli/superlu.git
        GIT_TAG v7.0.1
    )

    FetchContent_GetProperties(superlu)

    if(NOT superlu_POPULATED)
        FetchContent_Populate(superlu)
    endif()

    set(SUPERLU_BUILD_DIR
        ${CMAKE_BINARY_DIR}/superlu-build)

    file(MAKE_DIRECTORY
        ${SUPERLU_BUILD_DIR})

    if(WIN32 AND OPENBLAS_IMPLIB)
        set(_superlu_blas_lib ${OPENBLAS_IMPLIB})
    else()
        set(_superlu_blas_lib ${OPENBLAS_LIB})
    endif()

    execute_process(
        COMMAND
        ${CMAKE_COMMAND}
        -G ${CMAKE_GENERATOR}
        -S ${superlu_SOURCE_DIR}
        -B ${SUPERLU_BUILD_DIR}
        -DCMAKE_BUILD_TYPE=Release
        -DTPL_BLAS_LIBRARIES=${_superlu_blas_lib}
        -Denable_tests=OFF
        -Denable_examples=OFF
        -DBUILD_SHARED_LIBS=ON
        -DCMAKE_INSTALL_PREFIX=${DEPEND_LIB_OUTPUT}/superlu_lib

        RESULT_VARIABLE SUPERLU_CONFIG_RESULT
    )

    if(NOT SUPERLU_CONFIG_RESULT EQUAL 0)

        message(FATAL_ERROR
            "SuperLU configure failed")

    endif()

    execute_process(

        COMMAND
        ${CMAKE_COMMAND}
        --build ${SUPERLU_BUILD_DIR}
        --parallel ${BUILD_JOBS}

        RESULT_VARIABLE SUPERLU_BUILD_RESULT
    )

    if(NOT SUPERLU_BUILD_RESULT EQUAL 0)

        message(FATAL_ERROR
            "SuperLU build failed")

    endif()

    execute_process(
        COMMAND ${CMAKE_COMMAND}
        --install ${SUPERLU_BUILD_DIR}
        RESULT_VARIABLE MUMPS_INSTALL_RESULT
    )

    find_library(SUPERLU_LIBRARY
        NAMES superlu
        PATHS "${DEPEND_LIB_OUTPUT}/superlu_lib/lib"
        NO_DEFAULT_PATH
    )

    add_library(superlu UNKNOWN IMPORTED)

    set_target_properties(superlu PROPERTIES
        IMPORTED_LOCATION "${SUPERLU_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES
        "${DEPEND_LIB_OUTPUT}/superlu_lib/include"
    )

endif()
