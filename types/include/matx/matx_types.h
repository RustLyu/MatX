#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef MATX_BUILD_SHARED
	#ifdef MATX_PLATFORM_WINDOWS
		#ifdef MATX_AOCL_EXPORTS
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
#define MKL_ILP64
#define OPENBLAS_USE64BITINT
#define aoclsparse_ILP64
	typedef enum matx_status_t {
		MATX_OK = 0,
		MATX_ERR_INVALID_ARG = 1,
		MATX_ERR_OUT_OF_MEMORY = 2,
		MATX_ERR_NOT_SUPPORTED = 3,
		MATX_ERR_INTERNAL = 4
	} matx_status_t;

	typedef long long matx_int64_t;
	typedef double matx_double;

	// ---- Alloc ----
	typedef void* (*matx_malloc_fn)(size_t size, void* user);
	typedef void (*matx_free_fn)(void* ptr, void* user);

	typedef struct matx_alloc_t {
		matx_malloc_fn malloc_fn;
		matx_free_fn free_fn;
		void* user;
	} matx_alloc_t;

	// ---- Layout ----
	typedef enum matx_layout_t {
		MATX_COL_MAJOR = 0,
		MATX_ROW_MAJOR = 1
	} matx_layout_t;

	typedef enum matx_handle_type_t {
		MATX_HANDLE_TYPE_GRB_MATRIX = 1,
		MATX_HANDLE_TYPE_GRB_VECTOR = 2,
		MATX_HANDLE_TYPE_MKL_MATRIX = 3,
		MATX_HANDLE_TYPE_AOCL_MATRIX = 4
	} matx_handle_type_t;

	typedef void(*free_ptr_func)(void* ptr);

	typedef struct matx_handle_t {
		void* impl;
		matx_handle_type_t type;
		int8_t valid;
		free_ptr_func custom_free_func;

	}matx_handle_t;

	// ---- Real/complex scalar ----
	typedef struct matx_complex_f64_t {
		matx_double real;
		matx_double imag;
	} matx_complex_f64;

	// ---- Dense vector (double) ----
	typedef struct matx_vec_f64_t {
		matx_int64_t n;
		matx_int64_t stride;
		matx_double* data;
		matx_int64_t flags;
		matx_handle_t handle_grb;
	} matx_vec_f64_t;

	// ---- Dense matrix (double) ----
	typedef struct matx_dense_f64_t {
		matx_int64_t  rows;
		matx_int64_t  cols;
		matx_int64_t  stride;     // leading dimension: if col-major => ld = stride (>= rows); if row-major => ld = stride (>= cols)
		matx_layout_t layout;
		matx_double* data;
		matx_int64_t flags;    // reserved for future (ownership, alignment, etc.)
		matx_handle_t handle_grb;
	} matx_dense_f64_t;

	// ---- Dense vector (complex) ----
	typedef struct matx_vec_c64_t {
		matx_int64_t n;
		matx_int64_t stride;
		matx_complex_f64* data;
		matx_int64_t flags;
		matx_handle_t handle_grb;
	} matx_vec_c64_t;

	// ---- Dense matrix (complex) ----
	typedef struct matx_dense_c64_t {
		matx_int64_t rows;
		matx_int64_t cols;
		matx_int64_t stride;
		matx_layout_t layout;
		matx_complex_f64* data;
		matx_int64_t flags;
		matx_handle_t handle_grb;
	} matx_dense_c64_t;

	// ---- Sparse CSC (real/complex) ----
	typedef struct matx_csc_f64_t {
		matx_int64_t nrows;
		matx_int64_t ncols;
		matx_int64_t nnz;
		const matx_int64_t* col_ptr;
		const matx_int64_t* row_ind;
		const matx_double* values;
		const matx_int64_t* coo_csc_index_map;
		matx_int64_t struct_update; // coo to csc conversion may involve sorting and duplicate summation, these flags can be used to track whether the structure/values are up to date with the original COO data
		matx_int64_t only_value_update; // if the structure is up to date, but values have been updated, this flag can be set to indicate that only values need to be updated in the CSC representation without redoing the entire COO to CSC conversion
		uint32_t flags;
	} matx_csc_f64_t;


	typedef struct matx_coo_f64_t {
		matx_int64_t nrows;
		matx_int64_t ncols;
		matx_int64_t nnz;
		const matx_int64_t* rows;
		const matx_int64_t* columns;
		const matx_double* values;
		matx_int64_t flags;
		matx_handle_t handle_grb;
		matx_handle_t handle_mkl;
		matx_handle_t handle_aocl;
		matx_csc_f64_t handle_csc; // for backends that require CSC format, we can lazily convert COO to CSC and store here to avoid repeated conversions
	} matx_coo_f64_t;

	typedef struct matx_csc_c64_t {
		matx_int64_t nrows;
		matx_int64_t ncols;
		matx_int64_t nnz;
		const matx_int64_t* col_ptr;
		const matx_int64_t* row_ind;
		const matx_complex_f64* values;
		const matx_int64_t* coo_csc_index_map;
		matx_int64_t struct_update; // coo to csc conversion may involve sorting and duplicate summation, these flags can be used to track whether the structure/values are up to date with the original COO data
		matx_int64_t only_value_update; // if the structure is up to date, but values have been updated, this flag can be set to indicate that only values need to be updated in the CSC representation without redoing the entire COO to CSC conversion
		matx_int64_t flags;
	} matx_csc_c64_t;

	typedef struct matx_coo_c64_t {
		matx_int64_t nrows;
		matx_int64_t ncols;
		matx_int64_t nnz;
		const matx_int64_t* rows;
		const matx_int64_t* columns;
		const matx_complex_f64* values;
		matx_int64_t flags;
		matx_handle_t handle_grb;
		matx_handle_t handle_mkl;
		matx_handle_t handle_aocl;
		matx_csc_c64_t handle_csc; // for backends that require CSC format, we can lazily convert COO to CSC and store here to avoid repeated conversions
	} matx_coo_c64_t;
	MATX_API const char* matx_version_string(void);
#ifdef __cplusplus
}
#endif

