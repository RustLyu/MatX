#ifndef MATX_TYPES_H
#define MATX_TYPES_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef MATX_BUILD_SHARED
#ifdef MATX_PLATFORM_WINDOWS
#ifdef matx_types_EXPORTS
#define MATX_API __declspec(dllexport)
#else
#define MATX_API __declspec(dllimport)
#endif
#else
#define MATX_API
#endif
#else
#define MATX_API
#endif

// ---- Version ----
#define MATX_VERSION_MAJOR 0
#define MATX_VERSION_MINOR 1
#define MATX_VERSION_PATCH 0
//#define MKL_ILP64
//#define OPENBLAS_USE64BITINT
#define aoclsparse_ILP64
#define HAVE_LAPACK_CONFIG_H
//#define __EMSCRIPTEN__
#define LAPACK_ILP64
//#define FLA_ENABLE_ILP64 1
#define LAPACK_COMPLEX_STRUCTURE

#ifndef INT_MAX
#define INT_MAX (2147483647)
#endif
typedef enum matx_status_t {
    MATX_OK = 0,
    MATX_ERR_INVALID_ARG = 1,
    MATX_ERR_OUT_OF_MEMORY = 2,
    MATX_ERR_NOT_SUPPORTED = 3,
    MATX_ERR_INTERNAL = 4
} matx_status_t;
#ifdef _WIN32
typedef long long matx_int64_t;
typedef double matx_double;
typedef bool matx_bool;
#elif __linux__
typedef int64_t matx_int64_t;
typedef double matx_double;
typedef bool matx_bool;
#endif

#if defined(_MSC_VER)
#define MATX_ALIGNED(x) __declspec(align(x))
#elif defined(__GNUC__) || defined(__clang__)
#define MATX_ALIGNED(x) __attribute__((aligned(x)))
#else
#define MATX_ALIGNED(x)
#endif

// ---- Alloc ----
typedef void* (*matx_malloc_fn)(size_t size, void* user);
typedef void (*matx_free_fn)(void* ptr, void* user);

typedef struct matx_alloc_t
{
    matx_malloc_fn malloc_fn;
    matx_free_fn free_fn;
    void* user;
} matx_alloc_t;

// ---- Layout ----
typedef enum matx_layout_t { MATX_ROW_MAJOR = 101, MATX_COL_MAJOR = 102 } matx_layout_t;

typedef enum matx_handle_type_t {
    MATX_HANDLE_TYPE_GRB_MATRIX = 1,
    MATX_HANDLE_TYPE_GRB_VECTOR = 2,
    MATX_HANDLE_TYPE_MKL_MATRIX = 3,
    MATX_HANDLE_TYPE_AOCL_MATRIX = 4
} matx_handle_type_t;

typedef void (*free_ptr_func)(void* ptr);

typedef struct matx_handle_t
{
    void* impl;
    matx_handle_type_t type;
    int8_t valid;
    free_ptr_func custom_free_func;

} matx_handle_t;

// ---- Real/complex scalar ----
#define MATX_COMPLEX_ALIGNMENT 16
typedef struct MATX_ALIGNED(MATX_COMPLEX_ALIGNMENT) matx_complex_d_t
{
    matx_double real;
    matx_double imag;
} matx_complex_d_t;

// ---- Dense vector (double) ----
typedef struct matx_vec_d_i8_opaque_t* matx_vec_d_i8_t;

// ---- Dense vector (complex) ----
typedef struct matx_vec_z_i8_opaque_t* matx_vec_z_i8_t;

// ---- Dense matrix (double) ----
typedef struct matx_dense_d_i8_opaque_t* matx_dense_d_i8_t;
// ---- Dense matrix (complex) ----
typedef struct matx_dense_z_i8_opaque_t* matx_dense_z_i8_t;

// ---- Sparse CSC/COO (real/complex) ----
typedef struct matx_csc_d_i8_opaque_t* matx_csc_d_i8_t;
typedef struct matx_coo_d_i8_opaque_t* matx_coo_d_i8_t;
typedef struct matx_csc_z_i8_opaque_t* matx_csc_z_i8_t;
typedef struct matx_coo_z_i8_opaque_t* matx_coo_z_i8_t;

MATX_API const char* matx_version_string(void);
#ifdef __cplusplus
}
#endif

#endif // MATX_TYPES_H