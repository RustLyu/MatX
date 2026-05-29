# =========================================================
# MatXPlatform.cmake
# OS detection, compiler flags, C/C++ standard
# =========================================================

if (WIN32)
    add_definitions(-DOS_WINDOWS)
    message(STATUS "current os: Windows")
elseif (LINUX)
    message(STATUS "current os: Linux")
elseif (APPLE)
    message(STATUS "current os: macOS")
else ()
    message(WARNING "UNKNOWN OS!")
endif ()

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
    set(CMAKE_C_FLAGS_DEBUG "-O0 -g")
endif()

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
