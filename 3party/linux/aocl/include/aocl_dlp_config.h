#ifndef AOCL_DLP_CONFIG_H
#define AOCL_DLP_CONFIG_H

// Version information - numeric constants
// clang-format off
#define AOCL_DLP_VERSION_MAJOR 5
#define AOCL_DLP_VERSION_MINOR 2
#define AOCL_DLP_VERSION_PATCH 0
// clang-format on

// Version information - string constants
#define AOCL_DLP_VERSION_MAJOR_STR "5"
#define AOCL_DLP_VERSION_MINOR_STR "2"
#define AOCL_DLP_VERSION_PATCH_STR "0"
#define AOCL_DLP_VERSION_STRING    "5.2.0"

// Build configuration
#define DLP_ENABLE_OPENMP
/* #undef DLP_ENABLE_PTHREAD */

// Operating System
#define DLP_OS_UNIX 1
#define DLP_OS_LINUX 1
#define DLP_OS_MACOS 0
#define DLP_OS_WINDOWS 0

#if DLP_OS_LINUX
#define DLP_LINUX_DISTRIBUTION         "RedHatEnterprise"
#define DLP_LINUX_DISTRIBUTION_VERSION "8.6"
#endif

#if DLP_OS_WINDOWS
#define DLP_WINDOWS_VERSION ""
#define DLP_WINDOWS_NAME    ""
#endif

// Compiler information
#define DLP_COMPILER_GCC 1
#define DLP_COMPILER_CLANG 0
#define DLP_COMPILER_INTEL 0
#define DLP_COMPILER_MSVC 0

#define DLP_COMPILER_NAME    "GCC"
#define DLP_COMPILER_VERSION "14.2.1"

#if DLP_COMPILER_MSVC
#define DLP_COMPILER_MSVC_YEAR ""
#endif

// Endianness
#define DLP_IS_BIG_ENDIAN 0
#define DLP_ENDIAN "little"

// Testing Config
#define DLP_TESTING_ENABLE_HIGH_PRECISION_FLOAT 0
#define DLP_TESTING_ENABLE_DETAILED_DEBUG 0

#endif // AOCL_DLP_CONFIG_H
