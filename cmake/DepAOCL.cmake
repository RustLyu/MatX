# =========================================================
# DepAOCL.cmake
# AMD AOCL stack: AOCL-utils, BLIS, libflame, aocl-sparse
# (when MATX_BACKEND=AMD_AOCL)
# =========================================================

# ---- AOCL Utils ----
if(MATX_BACKEND STREQUAL "AMD_AOCL")
    include(FetchContent)

    FetchContent_Declare(
        aocl-utils
        GIT_REPOSITORY https://github.com/amd/aocl-utils.git
        GIT_TAG 5.2.2
    )

    FetchContent_GetProperties(aocl-utils)

    if(NOT aocl-utils_POPULATED)
        FetchContent_Populate(aocl-utils)

        execute_process(
            COMMAND ${CMAKE_COMMAND}
            -S ${aocl-utils_SOURCE_DIR}
            -B ${aocl-utils_BINARY_DIR}
            -DENABLE_ILP64=ON
            -DCMAKE_BUILD_TYPE=RELEASE
            -DCMAKE_INSTALL_PREFIX=${AOCL_ROOT}
        )

        message(STATUS "Building AOCL Utils...")

        execute_process(
            COMMAND ${CMAKE_COMMAND}
            --build ${aocl-utils_BINARY_DIR}
            --parallel 1
            RESULT_VARIABLE AOCL_UTILS_BUILD_RESULT
        )

        if(NOT AOCL_UTILS_BUILD_RESULT EQUAL 0)
            message(FATAL_ERROR "AOCL Utils build failed")
        endif()

        message(STATUS "Installing AOCL Utils...")

        execute_process(
            COMMAND ${CMAKE_COMMAND}
            --install ${aocl-utils_BINARY_DIR}
            RESULT_VARIABLE AOCL_UTILS_INSTALL_RESULT
        )

        if(NOT AOCL_UTILS_INSTALL_RESULT EQUAL 0)
            message(FATAL_ERROR "AOCL Utils install failed")
        endif()

        set(AOCL_UTILS_ROOT
            ${aocl-utils_BINARY_DIR}/install
            CACHE PATH "AOCL Utils install path")

        message(STATUS "AOCL Utils installed to: ${AOCL_UTILS_ROOT}")
        set(AOCL_UTILS_LIBRARY ${AOCL_ROOT}/lib/libaoclutils.so)
        set(AOCL_UTILS_INCLUDE_DIR ${AOCL_ROOT}/include)
    endif()
endif()

# ---- BLIS (AMD BLAS) ----
if(MATX_ENABLE_BLIS)

    include(FetchContent)

    FetchContent_Declare(
        blis
        GIT_REPOSITORY https://github.com/amd/blis.git
        GIT_TAG 5.2.2
    )

    FetchContent_GetProperties(blis)

    if(NOT blis_POPULATED)
        FetchContent_Populate(blis)
    endif()
	if(CPU_PLATFORM STREQUAL "AMD")
			execute_process(
				COMMAND ${CMAKE_COMMAND}
				-S ${blis_SOURCE_DIR}
				-B ${blis_BINARY_DIR}
				-DBLAS_INT_SIZE=64
				-DBLIS_CONFIG_FAMILY=auto
				-DBLIS_DISABLE_F77_TYPES=ON
				-DBLIS_ENABLE_NOFORTRAN=ON
				-DBLIS_ENABLE_OPENMP=ON
				-DENABLE_THREADING=openmp
				-DENABLE_CBLAS=ON
				-DCMAKE_INSTALL_PREFIX=${AOCL_ROOT}
			)
	else()
		execute_process(
				COMMAND ${CMAKE_COMMAND}
				-S ${blis_SOURCE_DIR}
				-B ${blis_BINARY_DIR}
				-DBLAS_INT_SIZE=64
				-DBLIS_CONFIG_FAMILY=generic
				-DBLIS_DISABLE_F77_TYPES=ON
				-DBLIS_ENABLE_NOFORTRAN=ON
				-DBLIS_ENABLE_OPENMP=ON
				-DENABLE_THREADING=openmp
				-DENABLE_CBLAS=ON
				-DCMAKE_INSTALL_PREFIX=${AOCL_ROOT}
			)
	endif()

    execute_process(
        COMMAND ${CMAKE_COMMAND}
        --build ${blis_BINARY_DIR}
        --parallel
        RESULT_VARIABLE BLIS_BUILD_RESULT
    )

    execute_process(
        COMMAND ${CMAKE_COMMAND}
        --install ${blis_BINARY_DIR}
        RESULT_VARIABLE BLIS_INSTALL_RESULT
    )

    set(BLAS_LIBRARY ${AOCL_ROOT}/lib/libblis-mt.so)
    set(BLAS_LIBRARIES ${BLAS_LIBRARY})
    set(AOCL_BLAS_INCLUDE_DIR ${AOCL_ROOT}/include)
    set(amdblis_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aocl_cmake/")
    find_package(amdblis REQUIRED)

endif()

# ---- libflame (AMD LAPACK) ----
if(MATX_ENABLE_LIBFLAME)
    set(LIBFLAME_LIBRARY_PATH "${AOCL_ROOT}/lib/libflame.so")
    set(LIBFLAME_HEADER_PATH "${AOCL_ROOT}/include/FLAME.h")

    if(EXISTS ${LIBFLAME_LIBRARY_PATH} AND EXISTS ${LIBFLAME_HEADER_PATH})
        message(STATUS "libflame exists, skip compile: ${LIBFLAME_LIBRARY_PATH}")

        set(LAPACK_LIBRARIES ${LIBFLAME_LIBRARY_PATH})
    else()
        message(STATUS "libflame not found, compiling...")

        include(FetchContent)
        FetchContent_Declare(
            libflame
            GIT_REPOSITORY https://github.com/amd/libflame.git
            GIT_TAG 5.2.2
            BINARY_DIR ${CMAKE_BINARY_DIR}/libflame-build
        )

        FetchContent_GetProperties(libflame)
        if(NOT libflame_POPULATED)
            FetchContent_Populate(libflame)

            file(MAKE_DIRECTORY ${libflame_BINARY_DIR})
            execute_process(
                COMMAND ${CMAKE_COMMAND}
                -S ${libflame_SOURCE_DIR}
                -B ${libflame_BINARY_DIR}
                -DCMAKE_BUILD_TYPE=RELEASE
                -DBLIS_CONFIG_FAMILY=auto
                -DAOCL_ROOT=${AOCL_ROOT}
                -DENABLE_ILP64=ON
                -DCMAKE_C_STANDARD_LIBRARIES="-lm"
                -DCMAKE_SHARED_LINKER_FLAGS="-lm"
                -DBLAS_LIBRARY=${BLAS_LIBRARY}
                -DENABLE_AMD_FLAGS=ON
                -DENABLE_AOCL_BLAS=ON
                -DCMAKE_INSTALL_PREFIX=${AOCL_ROOT}
                -DBUILD_SHARED_LIBS=ON
            )

            execute_process(
                COMMAND ${CMAKE_COMMAND}
                --build ${libflame_BINARY_DIR}
                --parallel
                RESULT_VARIABLE libflame_BUILD_RESULT
            )

            execute_process(
                COMMAND ${CMAKE_COMMAND}
                --install ${libflame_BINARY_DIR}
                RESULT_VARIABLE LIBFLAME_INSTALL_RESULT
            )

            set(LAPACK_LIBRARIES ${AOCL_ROOT}/lib/libflame.so)
        endif()
    endif()
    set(libflame_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aocl_cmake/")
    find_package(libflame REQUIRED)

    set(aoclutils_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aocl_cmake/")
    find_package(aoclutils REQUIRED)
endif()

# ---- AOCL Sparse ----
if(MATX_ENABLE_AOCL_SPARSE)

    FetchContent_Declare(
        aocl-sparse
        GIT_REPOSITORY https://github.com/amd/aocl-sparse.git
        GIT_TAG 5.2.2
    )
    FetchContent_GetProperties(aocl-sparse)

    if(NOT aocl-sparse_POPULATED)
        FetchContent_Populate(aocl-sparse)
    endif()

    execute_process(
        COMMAND ${CMAKE_COMMAND}
        -S ${aocl-sparse_SOURCE_DIR}
        -B ${aocl-sparse_BINARY_DIR}
        -DCMAKE_BUILD_TYPE=RELEASE
        -DBLIS_CONFIG_FAMILY=auto
        -DCMAKE_AOCL_ROOT=${AOCL_ROOT}
        -DSUPPORT_OMP=ON
        -DBUILD_ILP64=ON
        -DCMAKE_INSTALL_PREFIX=${AOCL_ROOT}
        -DBUILD_SHARED_LIBS=ON
    )

    execute_process(
        COMMAND ${CMAKE_COMMAND}
        --build ${aocl-sparse_BINARY_DIR}
        --parallel
        RESULT_VARIABLE libflame_BUILD_RESULT
    )

    execute_process(
        COMMAND ${CMAKE_COMMAND}
        --install ${aocl-sparse_BINARY_DIR}
        RESULT VARIABLE LIBFLAME_INSTALL_RESULT
    )

    set(AOCLSPARSE_INCLUDE_DIR ${AOCL_ROOT}/include)
    set(AOCLSPARSE_LIBRARY ${AOCL_ROOT}/lib/libflame.so)

    add_library(AOCLSPARSE::AOCLSPARSE STATIC IMPORTED GLOBAL)
    set_target_properties(AOCLSPARSE::AOCLSPARSE PROPERTIES
        IMPORTED_LOCATION ${AOCLSPARSE_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES ${AOCLSPARSE_INCLUDE_DIR}
    )
    set(aoclsparse_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aocl_cmake/")
    find_package(aoclsparse REQUIRED)
endif()
