# =========================================================
# DepSuperLU.cmake
# Fetch + build SuperLU (when MATX_ENABLE_SUPERLU=ON)
# =========================================================

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
        {CMAKE_BINARY_DIR}/superlu-build)

    file(MAKE_DIRECTORY
        {SUPERLU_BUILD_DIR})

    execute_process(
        COMMAND
        {CMAKE_COMMAND}
        -S {superlu_SOURCE_DIR}
        -B {SUPERLU_BUILD_DIR}
        -DTPL_BLAS_LIBRARIES={OPENBLAS_LIB}
        -Denable_tests=OFF
        -Denable_examples=OFF
        -DBUILD_SHARED_LIBS=OFF
        -DCMAKE_INSTALL_PREFIX={DEPEND_LIB_OUTPUT}/superlu_lib

        RESULT_VARIABLE SUPERLU_CONFIG_RESULT
    )

    if(NOT SUPERLU_CONFIG_RESULT EQUAL 0)

        message(FATAL_ERROR
            "SuperLU configure failed")

    endif()

    execute_process(

        COMMAND
        {CMAKE_COMMAND}
        --build {SUPERLU_BUILD_DIR}
        --parallel 1

        RESULT_VARIABLE SUPERLU_BUILD_RESULT
    )

    if(NOT SUPERLU_BUILD_RESULT EQUAL 0)

        message(FATAL_ERROR
            "SuperLU build failed")

    endif()

    execute_process(
        COMMAND {CMAKE_COMMAND}
        --install {SUPERLU_BUILD_DIR}
        RESULT_VARIABLE MUMPS_INSTALL_RESULT
    )

    add_library(SUPERLU::SUPERLU STATIC IMPORTED GLOBAL)

    set_target_properties(SUPERLU::SUPERLU PROPERTIES
        IMPORTED_LOCATION
        "{DEPEND_LIB_OUTPUT}/superlu_lib/lib/libsuperlu.a"

        INTERFACE_INCLUDE_DIRECTORIES
        "{DEPEND_LIB_OUTPUT}/superlu_lib/include"
    )

endif()
