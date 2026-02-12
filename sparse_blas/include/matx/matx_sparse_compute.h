#pragma once

#include "matx/matx.h"

#ifdef __cplusplus
extern "C" {
#endif

	typedef enum matx_sparse_backend_kind_t {
		MATX_SPARSE_BACKEND_REFERENCE = 0,
		MATX_SPARSE_BACKEND_GRAPHBLAS = 1,
		MATX_SPARSE_BACKEND_MKL = 2,
	} matx_sparse_backend_kind_t;

	typedef struct matx_sparse_vtable_t {
		matx_status_t(*spmv_c64)(
			matx_complex_f64 alpha,
			matx_coo_c64_t* A,
			matx_vec_c64_t* x,
			matx_complex_f64 beta,
			matx_vec_c64_t* y);
		matx_status_t(*spmm_c64)(
			matx_complex_f64 alpha,
			const matx_coo_c64_t* A,
			const matx_dense_c64_t* B,
			matx_complex_f64 beta,
			matx_dense_c64_t* C);
		matx_status_t(*spmm_f64)(
			matx_double alpha,
			matx_coo_f64_t* A,
			matx_dense_f64_t* B,
			matx_double beta,
			matx_dense_f64_t* C);
		matx_status_t(*spmv_f64)(
			matx_double alpha,
			matx_coo_f64_t* A,
			matx_vec_f64_t* x,
			matx_double beta,
			matx_vec_f64_t* y);

	} matx_sparse_vtable_t;

	typedef struct matx_sparse_backend_t {
		matx_sparse_backend_kind_t kind;
		matx_sparse_vtable_t vt;
	} matx_sparse_backend_t;

	matx_sparse_backend_t matx_sparse_make_reference_grb(void);

	// Initialize default backend based on MATX_BLAS_BACKEND (AUTO picks a reasonable default at build time).
	matx_sparse_backend_t matx_sparse_default(void);
	const char* matx_sparse_backend_name(matx_sparse_backend_kind_t k);

	// y := alpha * A * x + beta * y  (sparse CSC, op(A)=A for now)
	matx_status_t matx_spmv_coo_f64(const matx_sparse_backend_t* backend,
		matx_double alpha,
		const matx_coo_f64_t* A,
		const matx_vec_f64_t* x,
		matx_double beta,
		matx_vec_f64_t* y);

	matx_status_t matx_spmv_coo_c64(const matx_sparse_backend_t* backend,
		matx_complex_f64 alpha,
		const matx_coo_c64_t* A,
		const matx_vec_c64_t* x,
		matx_complex_f64 beta,
		matx_vec_c64_t* y);


	// C := alpha * A * B + beta * C  (sparse CSC * dense, op() = I for now)
	matx_status_t matx_spmm_coo_f64(const matx_sparse_backend_t* backend,
		matx_double alpha,
		matx_coo_f64_t* A,
		matx_dense_f64_t* B,
		matx_double beta,
		matx_dense_f64_t* C);

	matx_status_t matx_spmm_coo_c64(const matx_sparse_backend_t* backend,
		matx_complex_f64 alpha,
		const matx_coo_c64_t* A,
		const matx_dense_c64_t* B,
		matx_complex_f64 beta,
		matx_dense_c64_t* C);

	void free_grb_matrix(void* impl);
	void free_grb_vector(void* impl);
	size_t coo_2_grb_f64(matx_coo_f64_t* A);
	size_t dense_2_grb_f64(matx_dense_f64_t* A);
	size_t grb_2_dense_f64(matx_dense_f64_t* A);
	size_t grb_2_coo_f64(matx_coo_f64_t* A);
	size_t vec_2_grb_f64(matx_vec_f64_t* v);
	size_t grb_2_vec_f64(matx_vec_f64_t* v);
	size_t coo_2_grb_c64(matx_coo_c64_t* A);
	size_t dense_2_grb_c64(matx_dense_c64_t* A);
	size_t grb_2_dense_c64(matx_dense_c64_t* A);
	size_t grb_2_coo_c64(matx_coo_c64_t* A);
	size_t vec_2_grb_c64(matx_vec_c64_t* v);
	size_t grb_2_vec_c64(matx_vec_c64_t* v);

	void free_mkl_matrix(void* impl);
	size_t coo_2_mkl_f64(matx_coo_f64_t* A);
	size_t coo_2_mkl_c64(matx_coo_c64_t* A);
#ifdef __cplusplus
}
#endif

