#pragma once

#include "matx/matx_types.h"
#include "matx/matx_func.h"

#ifdef __cplusplus
extern "C" {
#endif

	typedef enum matx_sparse_backend_kind_t {
		MATX_SPARSE_BACKEND_REFERENCE = 0,
		MATX_SPARSE_BACKEND_GRAPHBLAS = 1,
		MATX_SPARSE_BACKEND_MKL = 2,
		MATX_SPARSE_BACKEND_AOCL_CPARSE = 3,
	} matx_sparse_backend_kind_t;

	typedef struct matx_sparse_vtable_t {
		matx_status_t(*spmv_c64)(
			matx_complex_f64_t alpha,
			matx_coo_c64_t A,
			matx_vec_c64_t x,
			matx_complex_f64_t beta,
			matx_vec_c64_t y);
		matx_status_t(*spmm_c64)(
			matx_complex_f64_t alpha,
			matx_coo_c64_t A,
			matx_dense_c64_t B,
			matx_complex_f64_t beta,
			matx_dense_c64_t C);
		matx_status_t(*spmm_f64)(
			matx_double alpha,
			matx_coo_f64_t A,
			matx_dense_f64_t B,
			matx_double beta,
			matx_dense_f64_t C);
		matx_status_t(*dsp2md_f64)(
			matx_double alpha,
			matx_coo_f64_t A,
			matx_coo_f64_t B,
			matx_double beta,
			matx_dense_f64_t C);
		matx_status_t(*zsp2md_c64)(
			matx_complex_f64_t alpha,
			matx_coo_c64_t A,
			matx_coo_c64_t B,
			matx_complex_f64_t beta,
			matx_dense_c64_t C);
		matx_status_t(*spmv_f64)(
			matx_double alpha,
			matx_coo_f64_t A,
			matx_vec_f64_t x,
			matx_double beta,
			matx_vec_f64_t y);
		matx_status_t(*transpose_f64)(
			matx_coo_f64_t A, 
			matx_coo_f64_t out);
		matx_status_t(*transpose_c64)(
			matx_coo_c64_t A,
			matx_coo_c64_t out);
		matx_status_t(*conj_trans_c64)(
			matx_coo_c64_t A,
			matx_coo_c64_t out);
		matx_status_t(*norm1_f64)(
			matx_vec_f64_t A,
			matx_double* out);
		matx_status_t(*norm2_f64)(
			matx_vec_f64_t A,
			matx_double* out);
		matx_status_t(*norminf_f64)(
			matx_vec_f64_t A,
			matx_double* out);

	} matx_sparse_vtable_t;

	typedef struct matx_sparse_backend_t {
		matx_sparse_backend_kind_t kind;
		matx_sparse_vtable_t vt;
	} matx_sparse_backend_t;

	// Initialize default backend based on MATX_BLAS_BACKEND (AUTO picks a reasonable default at build time).
	MATX_API matx_sparse_backend_t matx_sparse_default(void);
	MATX_API const char* matx_sparse_backend_name(matx_sparse_backend_kind_t k);

	// C = alpha * A * x + beta * y (A: sparse matrix, x: dense vector, C: dense vector) double version
	MATX_API matx_status_t matx_spmv_coo_f64(const matx_sparse_backend_t* backend,
		matx_double alpha,
		matx_coo_f64_t A,
		matx_vec_f64_t x,
		matx_double beta,
		matx_vec_f64_t y);

	// C = alpha * A * x + beta * y (A: sparse matrix, x: dense vector, C: dense vector) complex version
	MATX_API matx_status_t matx_spmv_coo_c64(const matx_sparse_backend_t* backend,
		matx_complex_f64_t alpha,
		matx_coo_c64_t A,
		matx_vec_c64_t x,
		matx_complex_f64_t beta,
		matx_vec_c64_t y);

	// C := alpha * A * B + beta * C  (sparse CSC * dense, op() = I for now) double version
	MATX_API matx_status_t matx_spmm_coo_f64(const matx_sparse_backend_t* backend,
		matx_double alpha,
		matx_coo_f64_t A,
		matx_dense_f64_t B,
		matx_double beta,
		matx_dense_f64_t C);

	// C = alpha * A * B + beta * C (A: sparse matrix, B: dense matrix, C: dense matrix) complex version
	MATX_API matx_status_t matx_spmm_coo_c64(const matx_sparse_backend_t* backend,
		matx_complex_f64_t alpha,
		matx_coo_c64_t A,
		matx_dense_c64_t B,
		matx_complex_f64_t beta,
		matx_dense_c64_t C);

	// C = alpha * A * B + beta * C (A: sparse matrix, B: sparse matrix, C: dense matrix) double version
	MATX_API matx_status_t matx_dsp2md_coo_f64(const matx_sparse_backend_t* backend,
		matx_double alpha,
		matx_coo_f64_t A,
		matx_coo_f64_t B,
		matx_double beta,
		matx_dense_f64_t C);

	// C = alpha * A * B + beta * C (A: sparse matrix, B: sparse matrix, C: dense matrix) comlex version
	MATX_API matx_status_t matx_zsp2md_coo_c64(const matx_sparse_backend_t* backend,
		matx_complex_f64_t alpha,
		matx_coo_c64_t A,
		matx_coo_c64_t B,
		matx_complex_f64_t beta,
		matx_dense_c64_t C);

	MATX_API matx_status_t matx_transpose_coo_f64(const matx_sparse_backend_t* backend,
		matx_coo_f64_t A,
		matx_coo_f64_t out);

	MATX_API matx_status_t matx_transpose_coo_c64(const matx_sparse_backend_t* backend,
		matx_coo_c64_t A,
		matx_coo_c64_t out);

	MATX_API matx_status_t matx_conj_coo_c64(const matx_sparse_backend_t* backend,
		matx_coo_c64_t A,
		matx_coo_c64_t out);

	MATX_API matx_status_t matx_norm1_f64(const matx_sparse_backend_t* backend,
		matx_vec_f64_t A,
		matx_double* out);

	MATX_API matx_status_t matx_norm2_f64(const matx_sparse_backend_t* backend,
		matx_vec_f64_t A,
		matx_double* out);

	MATX_API matx_status_t matx_norminf_f64(const matx_sparse_backend_t* backend,
		matx_vec_f64_t A,
		matx_double* out);

	MATX_API void free_grb_matrix(void* impl);
	MATX_API void free_grb_vector(void* impl);
	MATX_API size_t coo_2_grb_f64(matx_coo_f64_t A);
	MATX_API size_t create_empty_grb_f64(matx_coo_f64_t A);
	MATX_API size_t create_empty_grb_c64(matx_coo_c64_t A);
	MATX_API size_t dense_2_grb_f64(matx_dense_f64_t A);
	MATX_API size_t grb_2_dense_f64(matx_dense_f64_t A);
	MATX_API size_t grb_2_coo_f64(matx_coo_f64_t A);
	MATX_API size_t vec_2_grb_f64(matx_vec_f64_t v);
	MATX_API size_t grb_2_vec_f64(matx_vec_f64_t v);
	MATX_API size_t coo_2_grb_c64(matx_coo_c64_t A);
	MATX_API size_t dense_2_grb_c64(matx_dense_c64_t A);
	MATX_API size_t grb_2_dense_c64(matx_dense_c64_t A);
	MATX_API size_t grb_2_coo_c64(matx_coo_c64_t A);
	MATX_API size_t vec_2_grb_c64(matx_vec_c64_t v);
	MATX_API size_t grb_2_vec_c64(matx_vec_c64_t v);

	MATX_API void free_mkl_matrix(void* impl);
	MATX_API size_t coo_2_mkl_f64(matx_coo_f64_t A);
	MATX_API size_t coo_2_mkl_c64(matx_coo_c64_t A);

	MATX_API size_t coo_2_aocl_f64(matx_coo_f64_t A);
	MATX_API size_t coo_2_aocl_c64(matx_coo_c64_t A);
	MATX_API size_t aocl_2_coo_c64(matx_coo_c64_t A);
#ifdef __cplusplus
}
#endif

