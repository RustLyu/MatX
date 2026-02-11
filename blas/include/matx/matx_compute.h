#pragma once

#include "matx/matx.h"

#ifdef __cplusplus
extern "C" {
#endif

	typedef enum matx_dense_backend_kind_t {
		MATX_BLAS_BACKEND_REFERENCE = 0,
		MATX_BLAS_BACKEND_OPENBLAS = 1,
		MATX_BLAS_BACKEND_BLIS = 2,
	} matx_dense_backend_kind_t;

	typedef enum matx_sparse_backend_kind_t {
		MATX_SPARSE_BACKEND_REFERENCE = 0,
		MATX_SPARSE_BACKEND_GRAPHBLAS = 1,
		MATX_SPARSE_BACKEND_MKL = 2,
	} matx_sparse_backend_kind_t;

	typedef struct matx_blas_vtable_t {
		matx_status_t(*dgemm)(matx_layout_t layout,
			int trans_a,
			int trans_b,
			size_t m,
			size_t n,
			size_t k,
			matx_double alpha,
			const matx_double* a,
			size_t lda,
			const matx_double* b,
			size_t ldb,
			matx_double beta,
			matx_double* c,
			size_t ldc);
		matx_status_t(*zgemm)(matx_layout_t layout,
			int trans_a,
			int trans_b,
			size_t m,
			size_t n,
			size_t k,
			const void* alpha,
			const void* A,
			size_t lda,
			const void* B,
			size_t ldb,
			const void* beta,
			void* C,
			size_t ldc);
		matx_status_t(*zgemv)(matx_layout_t layout,
			int trans_a,
			size_t m,
			size_t n,
			const void* alpha,
			const void* A,
			size_t lda,
			const void* B,
			size_t ldb,
			const void* beta,
			void* C,
			size_t ldc);

		matx_status_t(*dgemv)(matx_layout_t layout,
			int trans_a,
			size_t m,
			size_t n,
			matx_double alpha,
			const matx_double* A,
			size_t lda,
			matx_double* B,
			size_t ldb,
			matx_double beta,
			matx_double* C,
			size_t ldc);

		matx_status_t(*daxpy)(size_t n,
			matx_double alpha,
			const matx_double* x,
			size_t lda,
			const void* y,
			size_t ldy);
		matx_status_t(*zaxpy)(size_t n,
			const void* alpha,
			const void* x,
			size_t lda,
			const void* y,
			size_t ldy);

		matx_status_t(*dgeadd)(matx_layout_t layout,
			size_t rows,
			size_t cols,
			matx_double alpha,
			const matx_double* A,
			size_t lda,
			matx_double beta,
			matx_double* B,
			size_t ldb);

		matx_status_t(*zgeadd)(matx_layout_t layout,
			size_t rows,
			size_t cols,
			const void* alpha,
			const void* A,
			size_t lda,
			const void* beta,
			void* B,
			size_t ldb);
	} matx_dense_vtable_t;

	typedef struct matx_dense_backend_t {
		matx_dense_backend_kind_t kind;
		matx_dense_vtable_t vt;
	} matx_dense_backend_t;

	matx_dense_backend_t matx_blas_make_reference(void);

	// Initialize default backend based on MATX_BLAS_BACKEND (AUTO picks a reasonable default at build time).
	matx_dense_backend_t matx_blas_default(void);
	const char* matx_blas_backend_name(matx_dense_backend_kind_t k);



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

	// Convenience API operating on MatX dense types.
	matx_status_t matx_gemm_f64(const matx_dense_backend_t* blas,
		matx_uint64_t trans_a,
		matx_uint64_t trans_b,
		matx_double alpha,
		const matx_dense_f64_t* A,
		const matx_dense_f64_t* B,
		matx_double beta,
		matx_dense_f64_t* C);

	matx_status_t matx_gemm_c64(const matx_dense_backend_t* blas,
		matx_uint64_t trans_a,
		matx_uint64_t trans_b,
		matx_complex_f64 alpha,
		const matx_dense_c64_t* A,
		const matx_dense_c64_t* B,
		matx_complex_f64 beta,
		matx_dense_c64_t* C);

	// ---- Level 1: vector ops ----
	// y := alpha * x + y
	matx_status_t matx_axpy_f64(const matx_dense_backend_t* blas,
		matx_double alpha,
		const matx_vec_f64_t* x,
		matx_vec_f64_t* y);

	matx_status_t matx_axpy_c64(
		const matx_dense_backend_t* blas,
		matx_complex_f64 alpha,
		const matx_vec_c64_t* x,
		matx_vec_c64_t* y);

	// ---- Level 2: matrix-vector ----
	// y := alpha * op(A) * x + beta * y  (dense)
	matx_status_t matx_gemv_f64(const matx_dense_backend_t* blas,
		matx_uint64_t trans_a,
		matx_double alpha,
		const matx_dense_f64_t* A,
		const matx_vec_f64_t* x,
		matx_double beta,
		matx_vec_f64_t* y);

	matx_status_t matx_gemv_c64(const matx_dense_backend_t* blas,
		matx_uint64_t trans_a,
		matx_complex_f64 alpha,
		const matx_dense_c64_t* A,
		const matx_vec_c64_t* x,
		matx_complex_f64 beta,
		matx_vec_c64_t* y);

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

	// ---- Level 3: matrix-matrix ----
	// C := alpha * op(A) * op(B) + beta * C  (dense)
	matx_status_t matx_gemm_f64(const matx_dense_backend_t* blas,
		matx_uint64_t trans_a,
		matx_uint64_t trans_b,
		matx_double alpha,
		const matx_dense_f64_t* A,
		const matx_dense_f64_t* B,
		matx_double beta,
		matx_dense_f64_t* C);

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

	// B := alpha * A + beta * B  (dense complex)
	matx_status_t matx_geadd_c64(const matx_dense_backend_t* blas,
		matx_complex_f64 alpha,
		const matx_dense_c64_t* A,
		matx_complex_f64 beta,
		matx_dense_c64_t* B);

	// B := alpha * A + beta * B  (dense real)
	matx_status_t matx_geadd_f64(const matx_dense_backend_t* blas,
		matx_double alpha,
		const matx_dense_f64_t* A,
		matx_double beta,
		matx_dense_f64_t* B);

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

