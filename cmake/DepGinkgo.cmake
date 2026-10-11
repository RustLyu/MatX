# Ginkgo is a CMake project. Build it as a regular subproject so its upstream
# target carries the correct include paths, compiler options, and OpenMP link.
if(MATX_ENABLE_GINKGO)
  if(MATX_BUILD_GINKGO_FROM_SOURCE)
    include(FetchContent)

    set(GINKGO_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GINKGO_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(GINKGO_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GINKGO_BUILD_REFERENCE ON CACHE BOOL "" FORCE)
    set(GINKGO_BUILD_OMP ON CACHE BOOL "" FORCE)
    set(GINKGO_BUILD_CUDA OFF CACHE BOOL "" FORCE)
    set(GINKGO_BUILD_HIP OFF CACHE BOOL "" FORCE)
    set(GINKGO_BUILD_SYCL OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
      matx_ginkgo_source
      URL https://github.com/ginkgo-project/ginkgo/archive/refs/tags/v1.11.0.tar.gz
    )
    # Ginkgo's aggregate static archive and LTO link temporaries are very
    # large. Build its upstream libraries shared without LTO to keep the
    # installed package and downstream link step practical. Restore MatX's
    # own build settings immediately after adding the upstream subproject.
    set(_matx_build_shared_libs "${BUILD_SHARED_LIBS}")
    set(_matx_cxx_flags_release "${CMAKE_CXX_FLAGS_RELEASE}")
    set(_matx_c_flags_release "${CMAKE_C_FLAGS_RELEASE}")
    set(_matx_exe_linker_flags_release "${CMAKE_EXE_LINKER_FLAGS_RELEASE}")
    set(_matx_shared_linker_flags_release "${CMAKE_SHARED_LINKER_FLAGS_RELEASE}")
    string(REPLACE "-flto" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    string(REPLACE "-flto" "" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
    string(REPLACE "-flto" "" CMAKE_EXE_LINKER_FLAGS_RELEASE
      "${CMAKE_EXE_LINKER_FLAGS_RELEASE}")
    string(REPLACE "-flto" "" CMAKE_SHARED_LINKER_FLAGS_RELEASE
      "${CMAKE_SHARED_LINKER_FLAGS_RELEASE}")
    set(BUILD_SHARED_LIBS ON)
    FetchContent_MakeAvailable(matx_ginkgo_source)
    set(BUILD_SHARED_LIBS "${_matx_build_shared_libs}")
    set(CMAKE_CXX_FLAGS_RELEASE "${_matx_cxx_flags_release}")
    set(CMAKE_C_FLAGS_RELEASE "${_matx_c_flags_release}")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${_matx_exe_linker_flags_release}")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${_matx_shared_linker_flags_release}")
    unset(_matx_build_shared_libs)
    unset(_matx_cxx_flags_release)
    unset(_matx_c_flags_release)
    unset(_matx_exe_linker_flags_release)
    unset(_matx_shared_linker_flags_release)

    if(NOT TARGET Ginkgo::ginkgo)
      message(FATAL_ERROR
        "Ginkgo source was built, but the upstream Ginkgo::ginkgo target is missing")
    endif()
    message(STATUS "MatX iterative dependency: Ginkgo v1.11.0 (source build)")
  else()
    find_package(Ginkgo CONFIG REQUIRED)
    if(NOT TARGET Ginkgo::ginkgo)
      message(FATAL_ERROR "Ginkgo was found but did not export Ginkgo::ginkgo")
    endif()
    message(STATUS "MatX iterative dependency: Ginkgo (installed package)")
  endif()
endif()
