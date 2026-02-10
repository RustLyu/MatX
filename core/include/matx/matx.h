#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

	// ---- Version ----
#define MATX_VERSION_MAJOR 0
#define MATX_VERSION_MINOR 1
#define MATX_VERSION_PATCH 0

	typedef enum matx_status_t {
		MATX_OK = 0,
		MATX_ERR_INVALID_ARG = 1,
		MATX_ERR_OUT_OF_MEMORY = 2,
		MATX_ERR_NOT_SUPPORTED = 3,
		MATX_ERR_INTERNAL = 4
	} matx_status_t;

	typedef uint64_t matx_uint64_t;
	typedef double matx_double;

	const char* matx_status_string(matx_status_t st);
	const char* matx_version_string(void);

	// ---- Alloc ----
	typedef void* (*matx_malloc_fn)(size_t size, void* user);
	typedef void (*matx_free_fn)(void* ptr, void* user);

	typedef struct matx_alloc_t {
		matx_malloc_fn malloc_fn;
		matx_free_fn free_fn;
		void* user;
	} matx_alloc_t;

	matx_alloc_t matx_alloc_default(void);
	void* matx_malloc(const matx_alloc_t* a, size_t size);
	void matx_free(const matx_alloc_t* a, void* ptr);

	// ---- Layout ----
	typedef enum matx_layout_t {
		MATX_COL_MAJOR = 0,
		MATX_ROW_MAJOR = 1
	} matx_layout_t;

	typedef enum matx_handle_type_t {
		MATX_HANDLE_TYPE_GRB_MATRIX = 1,
		MATX_HANDLE_TYPE_GRB_VECTOR = 2
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
		matx_uint64_t n;
		matx_uint64_t stride;
		matx_double* data;
		matx_uint64_t flags;
		matx_handle_t handle_grb;
	} matx_vec_f64_t;

	matx_status_t matx_vec_f64_create(matx_vec_f64_t* out,
		size_t n,
		const matx_alloc_t* alloc);

	matx_status_t matx_vec_f64_wrap(matx_vec_f64_t* out,
		size_t n,
		size_t stride,
		double* data);

	void matx_vec_f64_destroy(matx_vec_f64_t* v,
		const matx_alloc_t* alloc);

	// ---- Dense matrix (double) ----
	typedef struct matx_dense_f64_t {
		matx_uint64_t  rows;
		matx_uint64_t  cols;
		matx_uint64_t  stride;     // leading dimension: if col-major => ld = stride (>= rows); if row-major => ld = stride (>= cols)
		matx_layout_t layout;
		matx_double* data;
		matx_uint64_t flags;    // reserved for future (ownership, alignment, etc.)
		matx_handle_t handle_grb;
	} matx_dense_f64_t;

	matx_status_t matx_dense_f64_create(matx_dense_f64_t* out,
		matx_uint64_t rows,
		matx_uint64_t cols,
		matx_layout_t layout,
		const matx_alloc_t* alloc);

	matx_status_t matx_dense_f64_wrap(matx_dense_f64_t* out,
		matx_uint64_t rows,
		matx_uint64_t cols,
		matx_uint64_t stride,
		matx_layout_t layout,
		double* data);

	void matx_dense_f64_destroy(matx_dense_f64_t* m, const matx_alloc_t* alloc);

	// ---- Dense vector (complex) ----
	typedef struct matx_vec_c64_t {
		matx_uint64_t n;
		matx_uint64_t stride;
		matx_complex_f64* data;
		matx_uint64_t flags;
		matx_handle_t handle_grb;
	} matx_vec_c64_t;

	matx_status_t matx_vec_c64_create(matx_vec_c64_t* out,
		matx_uint64_t n,
		const matx_alloc_t* alloc);

	matx_status_t matx_vec_c64_wrap(matx_vec_c64_t* out,
		matx_uint64_t n,
		matx_uint64_t stride,
		matx_complex_f64* data);

	void matx_vec_c64_destroy(matx_vec_c64_t* v,
		const matx_alloc_t* alloc);

	// ---- Dense matrix (complex) ----
	typedef struct matx_dense_c64_t {
		matx_uint64_t rows;
		matx_uint64_t cols;
		matx_uint64_t stride;
		matx_layout_t layout;
		matx_complex_f64* data;
		matx_uint64_t flags;
		matx_handle_t handle_grb;
	} matx_dense_c64_t;

	matx_status_t matx_dense_c64_create(matx_dense_c64_t* out,
		matx_uint64_t rows,
		matx_uint64_t cols,
		matx_layout_t layout,
		const matx_alloc_t* alloc);

	matx_status_t matx_dense_c64_wrap(matx_dense_c64_t* out,
		matx_uint64_t rows,
		matx_uint64_t cols,
		matx_uint64_t stride,
		matx_layout_t layout,
		matx_complex_f64* data);

	void matx_dense_c64_destroy(matx_dense_c64_t* m,
		const matx_alloc_t* alloc);

	// ---- Sparse CSC (real/complex) ----
	typedef struct matx_csc_f64_t {
		matx_uint64_t nrows;
		matx_uint64_t ncols;
		matx_uint64_t nnz;
		const matx_uint64_t* col_ptr;
		const matx_uint64_t* row_ind;
		const matx_double* values;
		uint32_t flags;
	} matx_csc_f64_t;


	typedef struct matx_coo_f64_t {
		matx_uint64_t nrows;
		matx_uint64_t ncols;
		matx_uint64_t nnz;
		const matx_uint64_t* rows;
		const matx_uint64_t* columns;
		const matx_double* values;
		matx_uint64_t flags;
		matx_handle_t handle_grb;
	} matx_coo_f64_t;

	matx_status_t matx_sparse_f64_create(matx_csc_f64_t* out,
		matx_uint64_t nrows,
		matx_uint64_t ncols,
		matx_uint64_t nnz,
		const matx_alloc_t* alloc);

	matx_status_t matx_sparse_f64_wrap(matx_csc_f64_t* out,
		matx_uint64_t nrows,
		matx_uint64_t ncols,
		matx_uint64_t nnz,
		const matx_uint64_t* col_ptr,
		const matx_uint64_t* row_ind,
		const matx_double* values);

	void matx_sparse_f64_destroy(matx_coo_f64_t* m, const matx_alloc_t* alloc);

	typedef struct matx_csc_c64_t {
		matx_uint64_t nrows;
		matx_uint64_t ncols;
		matx_uint64_t nnz;
		const matx_uint64_t* col_ptr;
		const matx_uint64_t* row_ind;
		const matx_complex_f64* values;
		matx_uint64_t flags;
	} matx_csc_c64_t;

	typedef struct matx_coo_c64_t {
		matx_uint64_t nrows;
		matx_uint64_t ncols;
		matx_uint64_t nnz;
		const matx_uint64_t* rows;
		const matx_uint64_t* columns;
		const matx_complex_f64* values;
		matx_uint64_t flags;
		matx_handle_t handle_grb;
	} matx_coo_c64_t;

	matx_status_t matx_sparse_c64_create(matx_coo_c64_t* out,
		matx_uint64_t nrows,
		matx_uint64_t ncols,
		matx_uint64_t nnz,
		const matx_alloc_t* alloc);

	matx_status_t matx_sparse_c64_wrap(matx_coo_c64_t* out,
		matx_uint64_t nrows,
		matx_uint64_t ncols,
		matx_uint64_t nnz,
		const matx_uint64_t* rows,
		const matx_uint64_t* cols,
		const matx_complex_f64* values);

	void matx_sparse_c64_destroy(matx_coo_c64_t* m, const matx_alloc_t* alloc);

#ifdef __cplusplus
}
#endif

