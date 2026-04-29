#ifndef MATX_FUNC_H
#define MATX_FUNC_H

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

	MATX_API matx_status_t matx_vec_f64_create(
		const matx_alloc_t* alloc,
		matx_vec_f64_t* out,
		matx_double* data,
		matx_int64_t n);

	/**
	 * @brief Duplicate a real vector
	 * @formula out[i] := in[i],  i = 0, 1, ..., n-1
	 */
	MATX_API matx_status_t matx_vec_f64_dup(const matx_alloc_t* alloc,
		const matx_vec_f64_t in,
		matx_vec_f64_t* out);

	MATX_API matx_status_t matx_vec_f64_wrap(
		const matx_alloc_t* alloc,
		matx_vec_f64_t* out,
		matx_int64_t n,
		matx_int64_t stride,
		matx_double* data);

	MATX_API void matx_vec_f64_destroy(
		const matx_alloc_t* alloc,
		matx_vec_f64_t v);

	MATX_API matx_status_t matx_dense_f64_create(
		const matx_alloc_t* alloc,
		matx_dense_f64_t* out,
		matx_layout_t layout,
		matx_int64_t rows,
		matx_int64_t cols,
		matx_double* data);

	/**
	 * @brief Duplicate a real dense matrix
	 * @formula out[i][j] := in[i][j],  i = 0,...,rows-1; j = 0,...,cols-1
	 */
	MATX_API matx_status_t matx_dense_f64_dup(
		const matx_alloc_t* alloc,
		const matx_dense_f64_t in,
		matx_dense_f64_t* out
		);

	MATX_API matx_status_t matx_dense_f64_wrap(
		const matx_alloc_t* alloc,
		matx_dense_f64_t* out,
		matx_int64_t rows,
		matx_int64_t cols,
		matx_int64_t stride,
		matx_layout_t layout,
		matx_double* data);

	MATX_API void matx_dense_f64_destroy(const matx_alloc_t* alloc, matx_dense_f64_t m);

	MATX_API matx_status_t matx_vec_c64_create(
		const matx_alloc_t* alloc,
		matx_vec_c64_t* out,
		matx_complex_f64_t* data,
		matx_int64_t n);

	/**
	 * @brief Duplicate a complex vector
	 * @formula out[i] := in[i],  i = 0, 1, ..., n-1
	 */
	MATX_API matx_status_t matx_vec_c64_dup(
		const matx_alloc_t* alloc,
		const matx_vec_c64_t in,
		matx_vec_c64_t* out);

	MATX_API matx_status_t matx_vec_c64_wrap(
		const matx_alloc_t* alloc,
		matx_vec_c64_t* out,
		matx_int64_t n,
		matx_int64_t stride,
		matx_complex_f64_t* data);

	MATX_API void matx_vec_c64_destroy(const matx_alloc_t* alloc, matx_vec_c64_t v);

	MATX_API matx_status_t matx_dense_c64_create(
		const matx_alloc_t* alloc,
		matx_dense_c64_t* out,
		matx_layout_t layout,
		matx_int64_t rows,
		matx_int64_t cols,
		matx_complex_f64_t* data);

	/**
	 * @brief Duplicate a complex dense matrix
	 * @formula out[i][j] := in[i][j],  i = 0,...,rows-1; j = 0,...,cols-1
	 */
	MATX_API matx_status_t matx_dense_c64_dup(
		const matx_alloc_t* alloc,
		const matx_dense_c64_t in,
		matx_dense_c64_t* out
		);

	MATX_API matx_status_t matx_dense_c64_wrap(
		const matx_alloc_t* alloc,
		matx_dense_c64_t* out,
		matx_int64_t rows,
		matx_int64_t cols,
		matx_int64_t stride,
		matx_layout_t layout,
		matx_complex_f64_t* data);

	MATX_API void matx_dense_c64_destroy(const matx_alloc_t* alloc, matx_dense_c64_t m);

	MATX_API matx_status_t matx_coo_sparse_f64_create(
		const matx_alloc_t* alloc,
		matx_coo_f64_t* out,
		matx_int64_t nrows,
		matx_int64_t ncols,
		matx_int64_t nnz,
		matx_int64_t* ap,
		matx_int64_t* ai,
		matx_double* ax);

	/**
	 * @brief Duplicate a real COO sparse matrix
	 * @formula out.rows := in.rows; out.cols := in.cols; out.values := in.values  (copy all nnz entries)
	 */
	MATX_API matx_status_t matx_coo_f64_dup(
		const matx_alloc_t* alloc,
		const matx_coo_f64_t in,
		matx_coo_f64_t* out
		);

	MATX_API matx_status_t matx_csc_sparse_f64_create(
		const matx_alloc_t* alloc,
		matx_csc_f64_t* out,
		matx_int64_t nrows,
		matx_int64_t ncols,
		matx_int64_t nnz);

	MATX_API matx_status_t matx_csc_sparse_f64_wrap(const matx_alloc_t* alloc,
		matx_csc_f64_t* out,
		matx_int64_t nrows,
		matx_int64_t ncols,
		matx_int64_t nnz,
		const matx_int64_t* col_ptr,
		const matx_int64_t* row_ind,
		const matx_double* values);

	MATX_API void matx_coo_sparse_f64_destroy(const matx_alloc_t* alloc, matx_coo_f64_t m);

	MATX_API matx_status_t matx_coo_sparse_c64_create(
		const matx_alloc_t* alloc,
		matx_coo_c64_t* out,
		matx_int64_t nrows,
		matx_int64_t ncols,
		matx_int64_t nnz,
		matx_int64_t* ap,
		matx_int64_t* ai,
		matx_complex_f64_t* ax);

	/**
	 * @brief Duplicate a complex COO sparse matrix
	 * @formula out.rows := in.rows; out.cols := in.cols; out.values := in.values  (copy all nnz entries)
	 */
	MATX_API matx_status_t matx_coo_c64_dup(
		const matx_alloc_t* alloc,
		const matx_coo_c64_t in,
		matx_coo_c64_t* out
		);

	MATX_API matx_status_t matx_coo_sparse_c64_wrap(const matx_alloc_t* alloc,
		matx_coo_c64_t* out,
		matx_int64_t nrows,
		matx_int64_t ncols,
		matx_int64_t nnz,
		const matx_int64_t* rows,
		const matx_int64_t* cols,
		const matx_complex_f64_t* values);

	MATX_API void matx_coo_sparse_c64_destroy(const matx_alloc_t* alloc, matx_coo_c64_t m);

	MATX_API matx_status_t matx_csc_sparse_c64_create(
		const matx_alloc_t* alloc,
		matx_csc_c64_t* out,
		matx_int64_t nrows,
		matx_int64_t ncols,
		matx_int64_t nnz);
	MATX_API void matx_csc_sparse_c64_destroy(const matx_alloc_t* alloc, matx_csc_c64_t m);
	MATX_API void matx_csc_sparse_f64_destroy(const matx_alloc_t* alloc, matx_csc_f64_t m);

	// ---- Vector fill / init ----

	/**
	 * @brief Fill a real vector with a scalar value
	 * @formula v[i] := val,  i = 0, 1, ..., n-1
	 */
	MATX_API matx_status_t matx_vec_f64_fill(matx_vec_f64_t v, matx_double val);

	/**
	 * @brief Fill a complex vector with a scalar value
	 * @formula v[i] := val,  i = 0, 1, ..., n-1
	 */
	MATX_API matx_status_t matx_vec_c64_fill(matx_vec_c64_t v, matx_complex_f64_t val);

	/**
	 * @brief Set all elements of a real vector to zero
	 * @formula v[i] := 0,  i = 0, 1, ..., n-1
	 */
	MATX_API matx_status_t matx_vec_f64_zeros(matx_vec_f64_t v);

	/**
	 * @brief Set all elements of a complex vector to zero
	 * @formula v[i] := 0 + 0i,  i = 0, 1, ..., n-1
	 */
	MATX_API matx_status_t matx_vec_c64_zeros(matx_vec_c64_t v);

	/**
	 * @brief Set all elements of a real vector to one
	 * @formula v[i] := 1,  i = 0, 1, ..., n-1
	 */
	MATX_API matx_status_t matx_vec_f64_ones(matx_vec_f64_t v);

	// ---- Dense matrix fill / init ----

	/**
	 * @brief Fill a real dense matrix with a scalar value
	 * @formula m[i][j] := val,  i = 0,...,rows-1; j = 0,...,cols-1
	 */
	MATX_API matx_status_t matx_dense_f64_fill(matx_dense_f64_t m, matx_double val);

	/**
	 * @brief Fill a complex dense matrix with a scalar value
	 * @formula m[i][j] := val,  i = 0,...,rows-1; j = 0,...,cols-1
	 */
	MATX_API matx_status_t matx_dense_c64_fill(matx_dense_c64_t m, matx_complex_f64_t val);

	/**
	 * @brief Set all elements of a real dense matrix to zero
	 * @formula m[i][j] := 0,  i = 0,...,rows-1; j = 0,...,cols-1
	 */
	MATX_API matx_status_t matx_dense_f64_zeros(matx_dense_f64_t m);

	/**
	 * @brief Set all elements of a complex dense matrix to zero
	 * @formula m[i][j] := 0 + 0i,  i = 0,...,rows-1; j = 0,...,cols-1
	 */
	MATX_API matx_status_t matx_dense_c64_zeros(matx_dense_c64_t m);

	/**
	 * @brief Set all elements of a real dense matrix to one
	 * @formula m[i][j] := 1,  i = 0,...,rows-1; j = 0,...,cols-1
	 */
	MATX_API matx_status_t matx_dense_f64_ones(matx_dense_f64_t m);

	// ---- Dense matrix trace ----

	/**
	 * @brief Trace of a real dense matrix (sum of diagonal elements)
	 * @formula out := sum_{i=0}^{n-1} A[i][i]
	 *          where n = min(rows, cols)
	 */
	MATX_API matx_status_t matx_dense_f64_trace(const matx_dense_f64_t A, matx_double* out);

	/**
	 * @brief Trace of a complex dense matrix (sum of diagonal elements)
	 * @formula out := sum_{i=0}^{n-1} A[i][i]
	 *          where n = min(rows, cols)
	 */
	MATX_API matx_status_t matx_dense_c64_trace(const matx_dense_c64_t A, matx_complex_f64_t* out);

	// ---- Type conversion ----

	/**
	 * @brief Convert a real dense matrix to complex (zero imaginary part)
	 * @formula out[i][j] := A[i][j] + 0i,  i = 0,...,rows-1; j = 0,...,cols-1
	 */
	MATX_API matx_status_t matx_dense_f64_to_c64(const matx_alloc_t* alloc, const matx_dense_f64_t A, matx_dense_c64_t* out);

	/**
	 * @brief Convert a real vector to complex (zero imaginary part)
	 * @formula out[i] := v[i] + 0i,  i = 0, 1, ..., n-1
	 */
	MATX_API matx_status_t matx_vec_f64_to_c64(const matx_alloc_t* alloc, const matx_vec_f64_t v, matx_vec_c64_t* out);

#ifdef __cplusplus
}
#endif
#endif // MATX_FUNC_H