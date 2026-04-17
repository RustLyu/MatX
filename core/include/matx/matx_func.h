#pragma once

#include <stddef.h>
#include <stdint.h>
#include "matx/matx_types.h"

#ifdef __cplusplus
extern "C" {
#endif

	MATX_API const char* matx_status_string(matx_status_t st);

	MATX_API matx_alloc_t matx_alloc_default(void);
	MATX_API void* matx_malloc(const matx_alloc_t* a, size_t size);
	MATX_API void matx_free(const matx_alloc_t* a, void* ptr);

	MATX_API matx_status_t matx_vec_f64_create(const matx_alloc_t* alloc, 
		matx_vec_f64_t* out, 
		matx_double* data,
		matx_int64_t n);

	MATX_API matx_status_t matx_vec_f64_wrap(matx_vec_f64_t* out,
		matx_int64_t n,
		matx_int64_t stride,
		matx_double* data);

	MATX_API void matx_vec_f64_destroy(const matx_alloc_t* alloc, matx_vec_f64_t* v);

	MATX_API matx_status_t matx_dense_f64_create(
		const matx_alloc_t* alloc,
		matx_dense_f64_t* out,
		matx_layout_t layout,
		matx_int64_t rows,
		matx_int64_t cols,
		matx_double* data);

	MATX_API matx_status_t matx_dense_f64_wrap(matx_dense_f64_t* out,
		matx_int64_t rows,
		matx_int64_t cols,
		matx_int64_t stride,
		matx_layout_t layout,
		matx_double* data);

	MATX_API void matx_dense_f64_destroy(const matx_alloc_t* alloc, matx_dense_f64_t* m);

	MATX_API matx_status_t matx_vec_c64_create(
		const matx_alloc_t* alloc,
		matx_vec_c64_t* out,
		matx_complex_f64* data,
		matx_int64_t n);

	MATX_API matx_status_t matx_vec_c64_wrap(matx_vec_c64_t* out,
		matx_int64_t n,
		matx_int64_t stride,
		matx_complex_f64* data);

	MATX_API void matx_vec_c64_destroy(const matx_alloc_t* alloc, matx_vec_c64_t* v);

	MATX_API matx_status_t matx_dense_c64_create(
		const matx_alloc_t* alloc,
		matx_dense_c64_t* out,
		matx_layout_t layout,
		matx_int64_t rows,
		matx_int64_t cols,
		matx_complex_f64* data);

	MATX_API matx_status_t matx_dense_c64_wrap(matx_dense_c64_t* out,
		matx_int64_t rows,
		matx_int64_t cols,
		matx_int64_t stride,
		matx_layout_t layout,
		matx_complex_f64* data);

	MATX_API void matx_dense_c64_destroy(const matx_alloc_t* alloc, matx_dense_c64_t* m);

	MATX_API matx_status_t matx_coo_sparse_f64_create(
		const matx_alloc_t* alloc,
		matx_coo_f64_t* out,
		matx_int64_t nrows,
		matx_int64_t ncols,
		matx_int64_t nnz,
		matx_int64_t* ap,
		matx_int64_t* ai,
		matx_double* ax);

	MATX_API matx_status_t matx_csc_sparse_f64_create(
		const matx_alloc_t* alloc,
		matx_csc_f64_t* out,
		matx_int64_t nrows,
		matx_int64_t ncols,
		matx_int64_t nnz);

	MATX_API matx_status_t matx_csc_sparse_f64_wrap(matx_csc_f64_t* out,
		matx_int64_t nrows,
		matx_int64_t ncols,
		matx_int64_t nnz,
		const matx_int64_t* col_ptr,
		const matx_int64_t* row_ind,
		const matx_double* values);

	MATX_API void matx_coo_sparse_f64_destroy(const matx_alloc_t* alloc, matx_coo_f64_t* m);

	MATX_API matx_status_t matx_coo_sparse_c64_create(
		const matx_alloc_t* alloc,
		matx_coo_c64_t* out,
		matx_int64_t nrows,
		matx_int64_t ncols,
		matx_int64_t nnz,
		matx_int64_t* ap,
		matx_int64_t* ai,
		matx_complex_f64* ax);

	MATX_API matx_status_t matx_coo_sparse_c64_wrap(matx_coo_c64_t* out,
		matx_int64_t nrows,
		matx_int64_t ncols,
		matx_int64_t nnz,
		const matx_int64_t* rows,
		const matx_int64_t* cols,
		const matx_complex_f64* values);

	MATX_API void matx_coo_sparse_c64_destroy(const matx_alloc_t* alloc, matx_coo_c64_t* m);

	MATX_API matx_status_t matx_csc_sparse_c64_create(
		const matx_alloc_t* alloc,
		matx_csc_c64_t* out,
		matx_int64_t nrows,
		matx_int64_t ncols,
		matx_int64_t nnz);
	MATX_API void matx_csc_sparse_c64_destroy(const matx_alloc_t* alloc, matx_csc_c64_t* m);
	MATX_API void matx_csc_sparse_f64_destroy(const matx_alloc_t* alloc, matx_csc_f64_t* m);

#ifdef __cplusplus
}
#endif

