#pragma once

#include "matx/matx_types.h"
#include "matx/matx_func.h"

#ifdef __cplusplus
extern "C" {
#endif

	typedef enum matx_dense_backend_kind_t {
		MATX_BLAS_BACKEND_REFERENCE = 0,
		MATX_BLAS_BACKEND_OPENBLAS = 1,
		MATX_BLAS_BACKEND_BLIS = 2,
	} matx_dense_backend_kind_t;

	typedef struct matx_blas_vtable_t {
		matx_status_t(*dgemm)(matx_layout_t layout,
            matx_int64_t trans_a,
            matx_int64_t trans_b,
			matx_int64_t m,
			matx_int64_t n,
			matx_int64_t k,
			matx_double alpha,
			const matx_double* a,
			matx_int64_t lda,
			const matx_double* b,
			matx_int64_t ldb,
			matx_double beta,
			matx_double* c,
			matx_int64_t ldc);
		matx_status_t(*zgemm)(matx_layout_t layout,
            matx_int64_t trans_a,
            matx_int64_t trans_b,
			matx_int64_t m,
			matx_int64_t n,
			matx_int64_t k,
			const void* alpha,
			const void* A,
			matx_int64_t lda,
			const void* B,
			matx_int64_t ldb,
			const void* beta,
			void* C,
			matx_int64_t ldc);
		matx_status_t(*zgemv)(matx_layout_t layout,
            matx_int64_t trans_a,
			matx_int64_t m,
			matx_int64_t n,
			const void* alpha,
			const void* A,
			matx_int64_t lda,
			const void* B,
			matx_int64_t ldb,
			const void* beta,
			void* C,
			matx_int64_t ldc);

		matx_status_t(*dgemv)(matx_layout_t layout,
            matx_int64_t trans_a,
			matx_int64_t m,
			matx_int64_t n,
			matx_double alpha,
			const matx_double* A,
			matx_int64_t lda,
			matx_double* B,
			matx_int64_t ldb,
			matx_double beta,
			matx_double* C,
			matx_int64_t ldc);

		matx_status_t(*daxpy)(matx_int64_t n,
			matx_double alpha,
			const matx_double* x,
			matx_int64_t lda,
			const void* y,
			matx_int64_t ldy);
		matx_status_t(*zaxpy)(matx_int64_t n,
			const void* alpha,
			const void* x,
			matx_int64_t lda,
			const void* y,
			matx_int64_t ldy);

		matx_status_t(*dgeadd)(matx_layout_t layout,
			matx_int64_t rows,
			matx_int64_t cols,
			matx_double alpha,
			const matx_double* A,
			matx_int64_t lda,
			matx_double beta,
			matx_double* B,
			matx_int64_t ldb);

		matx_status_t(*zgeadd)(matx_layout_t layout,
			matx_int64_t rows,
			matx_int64_t cols,
			const void* alpha,
			const void* A,
			matx_int64_t lda,
			const void* beta,
			void* B,
			matx_int64_t ldb);

		matx_status_t (*inv_dense_f64)(
			matx_layout_t layout,
			matx_int64_t rows,
			matx_int64_t cols,
			const matx_double* A,
			matx_double* out_Ainv);

		matx_status_t(*inv_dense_c64)(matx_layout_t layout,
			matx_int64_t rows,
			matx_int64_t cols,
			const void* A,
			void* out_Ainv);

	} matx_dense_vtable_t;

	typedef struct matx_dense_backend_t {
		matx_dense_backend_kind_t kind;
		matx_dense_vtable_t vt;
	} matx_dense_backend_t;

	MATX_API matx_dense_backend_t matx_blas_make_reference(void);

	// Initialize default backend based on MATX_BLAS_BACKEND (AUTO picks a reasonable default at build time).
	MATX_API  matx_dense_backend_t matx_blas_default(void);
	MATX_API  const char* matx_blas_backend_name(matx_dense_backend_kind_t k);

	// Convenience API operating on MatX dense types.
	MATX_API  matx_status_t matx_gemm_f64(const matx_dense_backend_t* blas,
		matx_int64_t trans_a,
		matx_int64_t trans_b,
		matx_double alpha,
		const matx_dense_f64_t* A,
		const matx_dense_f64_t* B,
		matx_double beta,
		matx_dense_f64_t* C);

	MATX_API matx_status_t matx_gemm_c64(const matx_dense_backend_t* blas,
		matx_int64_t trans_a,
		matx_int64_t trans_b,
		matx_complex_f64 alpha,
		const matx_dense_c64_t* A,
		const matx_dense_c64_t* B,
		matx_complex_f64 beta,
		matx_dense_c64_t* C);

	// ---- Level 1: vector ops ----
	// y := alpha * x + y
	MATX_API matx_status_t matx_axpy_f64(const matx_dense_backend_t* blas,
		matx_double alpha,
		const matx_vec_f64_t* x,
		matx_vec_f64_t* y);

	MATX_API matx_status_t matx_axpy_c64(
		const matx_dense_backend_t* blas,
		matx_complex_f64 alpha,
		const matx_vec_c64_t* x,
		matx_vec_c64_t* y);

	// ---- Level 2: matrix-vector ----
	// y := alpha * op(A) * x + beta * y  (dense)
	MATX_API matx_status_t matx_gemv_f64(const matx_dense_backend_t* blas,
		matx_int64_t trans_a,
		matx_double alpha,
		const matx_dense_f64_t* A,
		const matx_vec_f64_t* x,
		matx_double beta,
		matx_vec_f64_t* y);

	MATX_API matx_status_t matx_gemv_c64(const matx_dense_backend_t* blas,
		matx_int64_t trans_a,
		matx_complex_f64 alpha,
		const matx_dense_c64_t* A,
		const matx_vec_c64_t* x,
		matx_complex_f64 beta,
		matx_vec_c64_t* y);

	// ---- Level 3: matrix-matrix ----
	// C := alpha * op(A) * op(B) + beta * C  (dense)
	MATX_API matx_status_t matx_gemm_f64(const matx_dense_backend_t* blas,
		matx_int64_t trans_a,
		matx_int64_t trans_b,
		matx_double alpha,
		const matx_dense_f64_t* A,
		const matx_dense_f64_t* B,
		matx_double beta,
		matx_dense_f64_t* C);

	// B := alpha * A + beta * B  (dense complex)
	MATX_API matx_status_t matx_geadd_c64(const matx_dense_backend_t* blas,
		matx_complex_f64 alpha,
		const matx_dense_c64_t* A,
		matx_complex_f64 beta,
		matx_dense_c64_t* B);

	// B := alpha * A + beta * B  (dense real)
	MATX_API matx_status_t matx_geadd_f64(const matx_dense_backend_t* blas,
		matx_double alpha,
		const matx_dense_f64_t* A,
		matx_double beta,
		matx_dense_f64_t* B);
#ifdef __cplusplus
}
#endif

