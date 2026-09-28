# =========================================================
# DepSpdlog.cmake
# Fetch + build spdlog logging library
# =========================================================

function(matx_spdlog)
    include(FetchContent)
    FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.13.0
    )

    FetchContent_GetProperties(spdlog)

    if(NOT spdlog_POPULATED)
        FetchContent_Populate(spdlog)
    endif()

    execute_process(
        COMMAND ${CMAKE_COMMAND}
        -G ${CMAKE_GENERATOR}
        -S ${spdlog_SOURCE_DIR}
        -B ${spdlog_BINARY_DIR}
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_INSTALL_PREFIX=${DEPEND_LIB_OUTPUT}/spdlog
    )

    execute_process(
        COMMAND ${CMAKE_COMMAND}
        --build ${spdlog_BINARY_DIR}
        --parallel ${BUILD_JOBS}
        RESULT_VARIABLE BLIS_BUILD_RESULT
    )

    execute_process(
        COMMAND ${CMAKE_COMMAND}
        --install ${spdlog_BINARY_DIR}
        RESULT_VARIABLE spdlog_INSTALL_RESULT
    )
endfunction()

matx_spdlog()

set(spdlog_DIR ${DEPEND_LIB_OUTPUT}/spdlog/lib/cmake/spdlog)
find_package(spdlog REQUIRED)
