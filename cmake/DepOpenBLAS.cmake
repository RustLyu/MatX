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
		-DINTERFACE64=1
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

	if(NOT OPENBLAS_INSTALL_RESULT EQUAL 0)
		message(FATAL_ERROR "OpenBLAS install failed")
	endif()


endfunction()

# ---- Conditionally build OpenBLAS and make it available ----
if(MATX_ENABLE_OPENBLAS)
    matx_add_openblas()
    set(OPENBLAS_CMAKE_DIR "${DEPEND_LIB_OUTPUT}/openblas/lib/cmake/OpenBLAS64")
	set(OpenBLAS64_DIR "${OPENBLAS_CMAKE_DIR}")
	find_package(OpenBLAS64 REQUIRED)

	get_target_property(OPENBLAS_LIB OpenBLAS64::OpenBLAS IMPORTED_LOCATION_RELEASE)
	message(STATUS "1111111111 OpenBLAS library: ${OPENBLAS_LIB}")
	if(NOT OPENBLAS_LIB)
		get_target_property(OPENBLAS_LIB OpenBLAS64::OpenBLAS IMPORTED_LOCATION)
	endif()

	message(STATUS "OpenBLAS library: ${OPENBLAS_LIB}")

	set(BLAS_LIBRARIES
		${OPENBLAS_LIB})

	set(LAPACK_LIBRARIES
		${OPENBLAS_LIB})
	message(STATUS "BLAS_LIBRARIES library: ${BLAS_LIBRARIES}")
	message(STATUS "LAPACK_LIBRARIES library: ${LAPACK_LIBRARIES}")
endif()
