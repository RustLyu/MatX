#pragma once

#include "matx/matx_types.h"
#include "matx/matx_func.h"
#include "matx/matx_types_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

	typedef enum matx_sparse_backend_kind_t {
		MATX_SPARSE_BACKEND_REFERENCE = 0,
		MATX_SPARSE_BACKEND_GRAPHBLAS = 1,
		MATX_SPARSE_BACKEND_AOCL_CPARSE = 3,
	} matx_sparse_backend_kind_t;

	typedef struct matx_sparse_vtable_t {
		matx_status_t(*spmv_z_i8)(
			matx_complex_d_i8_t alpha,
			matx_coo_z_i8_t A,
			matx_vec_z_i8_t x,
			matx_complex_d_i8_t beta,
			matx_vec_z_i8_t y);
		matx_status_t(*spmm_z_i8)(
			matx_complex_d_i8_t alpha,
			matx_coo_z_i8_t A,
			matx_dense_z_i8_t B,
			matx_complex_d_i8_t beta,
			matx_dense_z_i8_t C);
		matx_status_t(*spmm_d_i8)(
			matx_double alpha,
			matx_coo_d_i8_t A,
			matx_dense_d_i8_t B,
			matx_double beta,
			matx_dense_d_i8_t C);
		matx_status_t(*dsp2md_d_i8)(
			matx_double alpha,
			matx_coo_d_i8_t A,
			matx_coo_d_i8_t B,
			matx_double beta,
			matx_dense_d_i8_t C);
		matx_status_t(*zsp2md_z_i8)(
			matx_complex_d_i8_t alpha,
			matx_coo_z_i8_t A,
			matx_coo_z_i8_t B,
			matx_complex_d_i8_t beta,
			matx_dense_z_i8_t C);
		matx_status_t(*spmv_d_i8)(
			matx_double alpha,
			matx_coo_d_i8_t A,
			matx_vec_d_i8_t x,
			matx_double beta,
			matx_vec_d_i8_t y);
		matx_status_t(*transpose_d_i8)(
			matx_coo_d_i8_t A,
			matx_coo_d_i8_t out);
		matx_status_t(*transpose_z_i8)(
			matx_coo_z_i8_t A,
			matx_coo_z_i8_t out);
		matx_status_t(*conj_trans_z_i8)(
			matx_coo_z_i8_t A,
			matx_coo_z_i8_t out);

		matx_status_t(*finalize)();

		// ---- Sparse matrix norms ----
		matx_status_t(*norm1_mat_d_i8)(matx_coo_d_i8_t A, matx_double* out);
		matx_status_t(*norminf_mat_d_i8)(matx_coo_d_i8_t A, matx_double* out);
		matx_status_t(*normfro_mat_d_i8)(matx_coo_d_i8_t A, matx_double* out);
		matx_status_t(*norm1_mat_z_i8)(matx_coo_z_i8_t A, matx_double* out);
		matx_status_t(*norminf_mat_z_i8)(matx_coo_z_i8_t A, matx_double* out);
		matx_status_t(*normfro_mat_z_i8)(matx_coo_z_i8_t A, matx_double* out);

		// ---- Sparse-sparse addition: C = alpha*A + beta*B ----
		matx_status_t(*spadd_d_i8)(matx_double alpha, matx_coo_d_i8_t A, matx_double beta, matx_coo_d_i8_t B, matx_coo_d_i8_t out);
		matx_status_t(*spadd_z_i8)(matx_complex_d_i8_t alpha, matx_coo_z_i8_t A, matx_complex_d_i8_t beta, matx_coo_z_i8_t B, matx_coo_z_i8_t out);

		// ---- Per-row / per-column non-zero counts ----
		matx_status_t(*spnnz_rows_d_i8)(matx_coo_d_i8_t A, matx_vec_d_i8_t out);
		matx_status_t(*spnnz_cols_d_i8)(matx_coo_d_i8_t A, matx_vec_d_i8_t out);
		matx_status_t(*spnnz_rows_z_i8)(matx_coo_z_i8_t A, matx_vec_z_i8_t out);
		matx_status_t(*spnnz_cols_z_i8)(matx_coo_z_i8_t A, matx_vec_z_i8_t out);

		// ---- Row / column sums (absolute values) ----
		matx_status_t(*sprowsums_d_i8)(matx_coo_d_i8_t A, matx_vec_d_i8_t out);
		matx_status_t(*spcolsums_d_i8)(matx_coo_d_i8_t A, matx_vec_d_i8_t out);
		matx_status_t(*sprowsums_z_i8)(matx_coo_z_i8_t A, matx_vec_z_i8_t out);
		matx_status_t(*spcolsums_z_i8)(matx_coo_z_i8_t A, matx_vec_z_i8_t out);

		// ---- Extract diagonal at offset ----
		matx_status_t(*spdiag_d_i8)(matx_coo_d_i8_t A, matx_int64_t offset, matx_vec_d_i8_t out);
		matx_status_t(*spdiag_z_i8)(matx_coo_z_i8_t A, matx_int64_t offset, matx_vec_z_i8_t out);

		// ---- In-place row/column scaling ----
		matx_status_t(*scale_rows_d_i8)(matx_coo_d_i8_t A, const matx_vec_d_i8_t s);
		matx_status_t(*scale_cols_d_i8)(matx_coo_d_i8_t A, const matx_vec_d_i8_t s);
		matx_status_t(*scale_rows_z_i8)(matx_coo_z_i8_t A, const matx_vec_z_i8_t s);
		matx_status_t(*scale_cols_z_i8)(matx_coo_z_i8_t A, const matx_vec_z_i8_t s);

	} matx_sparse_vtable_t;

	typedef struct matx_sparse_backend_t {
		matx_sparse_backend_kind_t kind;
		matx_sparse_vtable_t vt;
	} matx_sparse_backend_t;

	// Initialize default backend based on MATX_BLAS_BACKEND (AUTO picks a reasonable default at build time).
	MATX_API matx_sparse_backend_t matx_sparse_default(void);
	MATX_API matx_sparse_backend_t matx_sparse_by_type(matx_sparse_backend_kind_t t);
	MATX_API const char* matx_sparse_backend_name(matx_sparse_backend_kind_t k);

	/**
	 * @brief Sparse matrix-vector multiply for real COO matrix (SpMV)
	 * @formula y := alpha * A * x + beta * y
	 *          A is sparse (COO), x and y are dense vectors.
	 */
	MATX_API matx_status_t matx_spmv_coo_d_i8(const matx_sparse_backend_t* backend,
		matx_double alpha,
		matx_coo_d_i8_t A,
		matx_vec_d_i8_t x,
		matx_double beta,
		matx_vec_d_i8_t y);

	/**
	 * @brief Sparse matrix-vector multiply for complex COO matrix (SpMV)
	 * @formula y := alpha * A * x + beta * y
	 *          A is sparse (COO), x and y are dense vectors.
	 */
	MATX_API matx_status_t matx_spmv_coo_z_i8(const matx_sparse_backend_t* backend,
		matx_complex_d_i8_t alpha,
		matx_coo_z_i8_t A,
		matx_vec_z_i8_t x,
		matx_complex_d_i8_t beta,
		matx_vec_z_i8_t y);

	/**
	 * @brief Sparse-dense matrix multiply for real COO matrix (SpMM)
	 * @formula C := alpha * A * B + beta * C
	 *          A is sparse (COO), B and C are dense matrices.
	 */
	MATX_API matx_status_t matx_spmm_coo_d_i8(const matx_sparse_backend_t* backend,
		matx_double alpha,
		matx_coo_d_i8_t A,
		matx_dense_d_i8_t B,
		matx_double beta,
		matx_dense_d_i8_t C);

	/**
	 * @brief Sparse-dense matrix multiply for complex COO matrix (SpMM)
	 * @formula C := alpha * A * B + beta * C
	 *          A is sparse (COO), B and C are dense matrices.
	 */
	MATX_API matx_status_t matx_spmm_coo_z_i8(const matx_sparse_backend_t* backend,
		matx_complex_d_i8_t alpha,
		matx_coo_z_i8_t A,
		matx_dense_z_i8_t B,
		matx_complex_d_i8_t beta,
		matx_dense_z_i8_t C);

	/**
	 * @brief Sparse-sparse multiply producing dense result, real (SpGEMM -> dense)
	 * @formula C := alpha * A * B + beta * C
	 *          A and B are sparse (COO), C is dense matrix.
	 */
	MATX_API matx_status_t matx_dsp2md_coo_d_i8(const matx_sparse_backend_t* backend,
		matx_double alpha,
		matx_coo_d_i8_t A,
		matx_coo_d_i8_t B,
		matx_double beta,
		matx_dense_d_i8_t C);

	/**
	 * @brief Sparse-sparse multiply producing dense result, complex (SpGEMM -> dense)
	 * @formula C := alpha * A * B + beta * C
	 *          A and B are sparse (COO), C is dense matrix.
	 */
	MATX_API matx_status_t matx_zsp2md_coo_z_i8(const matx_sparse_backend_t* backend,
		matx_complex_d_i8_t alpha,
		matx_coo_z_i8_t A,
		matx_coo_z_i8_t B,
		matx_complex_d_i8_t beta,
		matx_dense_z_i8_t C);

	/**
	 * @brief Transpose a real sparse COO matrix
	 * @formula out[i][j] = A[j][i]  (swap row and column indices)
	 */
	MATX_API matx_status_t matx_transpose_coo_d_i8(const matx_sparse_backend_t* backend,
		matx_coo_d_i8_t A,
		matx_coo_d_i8_t out);

	/**
	 * @brief Transpose a complex sparse COO matrix (no conjugation)
	 * @formula out[i][j] = A[j][i]  (swap row and column indices)
	 */
	MATX_API matx_status_t matx_transpose_coo_z_i8(const matx_sparse_backend_t* backend,
		matx_coo_z_i8_t A,
		matx_coo_z_i8_t out);

	/**
	 * @brief Conjugate transpose of a complex sparse COO matrix
	 * @formula out[i][j] = conj(A[j][i])
	 */
	MATX_API matx_status_t matx_conj_coo_z_i8(const matx_sparse_backend_t* backend,
		matx_coo_z_i8_t A,
		matx_coo_z_i8_t out);


	MATX_API matx_status_t matx_finalize(const matx_sparse_backend_t* backend);

	// ---- Sparse matrix norms (f64) ----

	/**
	 * @brief 1-norm of a real sparse COO matrix (max column sum)
	 * @formula ||A||_1 = max_{j=0,...,n-1} sum_{i=0}^{m-1} |A[i][j]|
	 */
	MATX_API matx_status_t matx_norm1_mat_coo_d_i8(const matx_sparse_backend_t* backend, matx_coo_d_i8_t A, matx_double* out);

	/**
	 * @brief Infinity-norm of a real sparse COO matrix (max row sum)
	 * @formula ||A||_inf = max_{i=0,...,m-1} sum_{j=0}^{n-1} |A[i][j]|
	 */
	MATX_API matx_status_t matx_norminf_mat_coo_d_i8(const matx_sparse_backend_t* backend, matx_coo_d_i8_t A, matx_double* out);

	/**
	 * @brief Frobenius norm of a real sparse COO matrix
	 * @formula ||A||_F = sqrt( sum_{(i,j) in nnz} A[i][j]^2 )
	 */
	MATX_API matx_status_t matx_normfro_mat_coo_d_i8(const matx_sparse_backend_t* backend, matx_coo_d_i8_t A, matx_double* out);

	// ---- Sparse matrix norms (c64) ----

	/**
	 * @brief 1-norm of a complex sparse COO matrix (max column sum)
	 * @formula ||A||_1 = max_{j=0,...,n-1} sum_{i=0}^{m-1} |A[i][j]|
	 */
	MATX_API matx_status_t matx_norm1_mat_coo_z_i8(const matx_sparse_backend_t* backend, matx_coo_z_i8_t A, matx_double* out);

	/**
	 * @brief Infinity-norm of a complex sparse COO matrix (max row sum)
	 * @formula ||A||_inf = max_{i=0,...,m-1} sum_{j=0}^{n-1} |A[i][j]|
	 */
	MATX_API matx_status_t matx_norminf_mat_coo_z_i8(const matx_sparse_backend_t* backend, matx_coo_z_i8_t A, matx_double* out);

	/**
	 * @brief Frobenius norm of a complex sparse COO matrix
	 * @formula ||A||_F = sqrt( sum_{(i,j) in nnz} |A[i][j]|^2 )
	 */
	MATX_API matx_status_t matx_normfro_mat_coo_z_i8(const matx_sparse_backend_t* backend, matx_coo_z_i8_t A, matx_double* out);

	// ---- Sparse-sparse addition: out = alpha*A + beta*B (COO) ----

	/**
	 * @brief Sparse matrix addition for real COO matrices
	 * @formula out := alpha * A + beta * B
	 */
	MATX_API matx_status_t matx_spadd_coo_d_i8(const matx_sparse_backend_t* backend,
		matx_double alpha, matx_coo_d_i8_t A, matx_double beta, matx_coo_d_i8_t B, matx_coo_d_i8_t out);

	/**
	 * @brief Sparse matrix addition for complex COO matrices
	 * @formula out := alpha * A + beta * B
	 */
	MATX_API matx_status_t matx_spadd_coo_z_i8(const matx_sparse_backend_t* backend,
		matx_complex_d_i8_t alpha, matx_coo_z_i8_t A, matx_complex_d_i8_t beta, matx_coo_z_i8_t B, matx_coo_z_i8_t out);

	// ---- Non-zero count per row/column ----

	/**
	 * @brief Count non-zeros per row for real COO matrix
	 * @formula out[i] = number of non-zeros in row i
	 */
	MATX_API matx_status_t matx_spnnz_rows_coo_d_i8(const matx_sparse_backend_t* backend,
		matx_coo_d_i8_t A, matx_vec_d_i8_t out);

	/**
	 * @brief Count non-zeros per column for real COO matrix
	 * @formula out[j] = number of non-zeros in column j
	 */
	MATX_API matx_status_t matx_spnnz_cols_coo_d_i8(const matx_sparse_backend_t* backend,
		matx_coo_d_i8_t A, matx_vec_d_i8_t out);

	/**
	 * @brief Count non-zeros per row for complex COO matrix
	 */
	MATX_API matx_status_t matx_spnnz_rows_coo_z_i8(const matx_sparse_backend_t* backend,
		matx_coo_z_i8_t A, matx_vec_z_i8_t out);

	/**
	 * @brief Count non-zeros per column for complex COO matrix
	 */
	MATX_API matx_status_t matx_spnnz_cols_coo_z_i8(const matx_sparse_backend_t* backend,
		matx_coo_z_i8_t A, matx_vec_z_i8_t out);

	// ---- Row / column sums (absolute values) ----

	/**
	 * @brief Absolute row sums for real COO matrix
	 * @formula out[i] = sum_j |A[i,j]|
	 */
	MATX_API matx_status_t matx_sprowsums_coo_d_i8(const matx_sparse_backend_t* backend,
		matx_coo_d_i8_t A, matx_vec_d_i8_t out);

	/**
	 * @brief Absolute column sums for real COO matrix
	 * @formula out[j] = sum_i |A[i,j]|
	 */
	MATX_API matx_status_t matx_spcolsums_coo_d_i8(const matx_sparse_backend_t* backend,
		matx_coo_d_i8_t A, matx_vec_d_i8_t out);

	/**
	 * @brief Absolute row sums for complex COO matrix
	 */
	MATX_API matx_status_t matx_sprowsums_coo_z_i8(const matx_sparse_backend_t* backend,
		matx_coo_z_i8_t A, matx_vec_z_i8_t out);

	/**
	 * @brief Absolute column sums for complex COO matrix
	 */
	MATX_API matx_status_t matx_spcolsums_coo_z_i8(const matx_sparse_backend_t* backend,
		matx_coo_z_i8_t A, matx_vec_z_i8_t out);

	// ---- Extract diagonal at offset ----

	/**
	 * @brief Extract diagonal from real COO matrix at given offset
	 * @formula For entries where rows[i] - cols[i] == offset, collect values into out vector
	 */
	MATX_API matx_status_t matx_spdiag_coo_d_i8(const matx_sparse_backend_t* backend,
		matx_coo_d_i8_t A, matx_int64_t offset, matx_vec_d_i8_t out);

	/**
	 * @brief Extract diagonal from complex COO matrix at given offset
	 */
	MATX_API matx_status_t matx_spdiag_coo_z_i8(const matx_sparse_backend_t* backend,
		matx_coo_z_i8_t A, matx_int64_t offset, matx_vec_z_i8_t out);

	// ---- In-place row/column scaling ----

	/**
	 * @brief Scale rows of real COO matrix in-place
	 * @formula A[i,j] *= s[rows[i]]
	 */
	MATX_API matx_status_t matx_scale_rows_coo_d_i8(const matx_sparse_backend_t* backend,
		matx_coo_d_i8_t A, const matx_vec_d_i8_t s);

	/**
	 * @brief Scale columns of real COO matrix in-place
	 * @formula A[i,j] *= s[cols[i]]
	 */
	MATX_API matx_status_t matx_scale_cols_coo_d_i8(const matx_sparse_backend_t* backend,
		matx_coo_d_i8_t A, const matx_vec_d_i8_t s);

	/**
	 * @brief Scale rows of complex COO matrix in-place
	 */
	MATX_API matx_status_t matx_scale_rows_coo_z_i8(const matx_sparse_backend_t* backend,
		matx_coo_z_i8_t A, const matx_vec_z_i8_t s);

	/**
	 * @brief Scale columns of complex COO matrix in-place
	 */
	MATX_API matx_status_t matx_scale_cols_coo_z_i8(const matx_sparse_backend_t* backend,
		matx_coo_z_i8_t A, const matx_vec_z_i8_t s);

	MATX_API size_t coo_2_grb_d_i8(matx_coo_d_i8_t A);
	MATX_API size_t create_empty_grb_d_i8(matx_coo_d_i8_t A);
	MATX_API size_t create_empty_grb_z_i8(matx_coo_z_i8_t A);
	MATX_API size_t dense_2_grb_d_i8(matx_dense_d_i8_t A);
	MATX_API size_t grb_2_dense_d_i8(matx_dense_d_i8_t A);
	MATX_API size_t grb_2_coo_d_i8(matx_coo_d_i8_t A);
	MATX_API size_t vec_2_grb_d_i8(matx_vec_d_i8_t v);
	MATX_API size_t grb_2_vec_d_i8(matx_vec_d_i8_t v);
	MATX_API size_t coo_2_grb_z_i8(matx_coo_z_i8_t A);
	MATX_API size_t dense_2_grb_z_i8(matx_dense_z_i8_t A);
	MATX_API size_t grb_2_dense_z_i8(matx_dense_z_i8_t A);
	MATX_API size_t grb_2_coo_z_i8(matx_coo_z_i8_t A);
	MATX_API size_t vec_2_grb_z_i8(matx_vec_z_i8_t v);
	MATX_API size_t grb_2_vec_z_i8(matx_vec_z_i8_t v);

	MATX_API size_t coo_2_aocl_d_i8(matx_coo_d_i8_t A);
	MATX_API size_t coo_2_aocl_z_i8(matx_coo_z_i8_t A);
	MATX_API size_t aocl_2_coo_z_i8(matx_coo_z_i8_t A);
#ifdef __cplusplus
}
#endif
