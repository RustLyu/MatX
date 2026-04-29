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
			void* y,
			matx_int64_t ldy);
		matx_status_t(*zaxpy)(matx_int64_t n,
			const void* alpha,
			const void* x,
			matx_int64_t lda,
			void* y,
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

		// ---- Level 1 additions ----
		matx_status_t(*dscal)(matx_int64_t n, matx_double alpha, matx_double* x, matx_int64_t incx);
		matx_status_t(*zscal)(matx_int64_t n, const void* alpha, void* x, matx_int64_t incx);
		matx_status_t(*dcopy)(matx_int64_t n, const matx_double* x, matx_int64_t incx, matx_double* y, matx_int64_t incy);
		matx_status_t(*zcopy)(matx_int64_t n, const void* x, matx_int64_t incx, void* y, matx_int64_t incy);
		matx_status_t(*dswap)(matx_int64_t n, matx_double* x, matx_int64_t incx, matx_double* y, matx_int64_t incy);
		matx_status_t(*zswap)(matx_int64_t n, void* x, matx_int64_t incx, void* y, matx_int64_t incy);
		matx_status_t(*ddot)(matx_int64_t n, const matx_double* x, matx_int64_t incx, const matx_double* y, matx_int64_t incy, matx_double* result);
		matx_status_t(*zdotu)(matx_int64_t n, const void* x, matx_int64_t incx, const void* y, matx_int64_t incy, void* result);
		matx_status_t(*zdotc)(matx_int64_t n, const void* x, matx_int64_t incx, const void* y, matx_int64_t incy, void* result);
		matx_status_t(*dnrm2)(matx_int64_t n, const matx_double* x, matx_int64_t incx, matx_double* result);
		matx_status_t(*dznrm2)(matx_int64_t n, const void* x, matx_int64_t incx, matx_double* result);
		matx_status_t(*dasum)(matx_int64_t n, const matx_double* x, matx_int64_t incx, matx_double* result);
		matx_status_t(*dzasum)(matx_int64_t n, const void* x, matx_int64_t incx, matx_double* result);
		matx_status_t(*idamax)(matx_int64_t n, const matx_double* x, matx_int64_t incx, matx_int64_t* result);
		matx_status_t(*izamax)(matx_int64_t n, const void* x, matx_int64_t incx, matx_int64_t* result);

		// ---- Level 2 additions ----
		matx_status_t(*dger)(matx_layout_t layout, matx_int64_t m, matx_int64_t n, matx_double alpha,
			const matx_double* x, matx_int64_t incx, const matx_double* y, matx_int64_t incy,
			matx_double* A, matx_int64_t lda);
		matx_status_t(*zgeru)(matx_layout_t layout, matx_int64_t m, matx_int64_t n, const void* alpha,
			const void* x, matx_int64_t incx, const void* y, matx_int64_t incy,
			void* A, matx_int64_t lda);
		matx_status_t(*dtrsv)(matx_layout_t layout, int uplo, int trans, int diag,
			matx_int64_t n, const matx_double* A, matx_int64_t lda, matx_double* x, matx_int64_t incx);
		matx_status_t(*ztrsv)(matx_layout_t layout, int uplo, int trans, int diag,
			matx_int64_t n, const void* A, matx_int64_t lda, void* x, matx_int64_t incx);

		// ---- Level 3 additions ----
		matx_status_t(*dtrsm)(matx_layout_t layout, int side, int uplo, int trans, int diag,
			matx_int64_t m, matx_int64_t n, matx_double alpha,
			const matx_double* A, matx_int64_t lda, matx_double* B, matx_int64_t ldb);
		matx_status_t(*ztrsm)(matx_layout_t layout, int side, int uplo, int trans, int diag,
			matx_int64_t m, matx_int64_t n, const void* alpha,
			const void* A, matx_int64_t lda, void* B, matx_int64_t ldb);
		matx_status_t(*dsyrk)(matx_layout_t layout, int uplo, int trans,
			matx_int64_t n, matx_int64_t k, matx_double alpha,
			const matx_double* A, matx_int64_t lda, matx_double beta, matx_double* C, matx_int64_t ldc);
		matx_status_t(*zherk)(matx_layout_t layout, int uplo, int trans,
			matx_int64_t n, matx_int64_t k, matx_double alpha,
			const void* A, matx_int64_t lda, matx_double beta, void* C, matx_int64_t ldc);

			// ---- Transpose ----
			matx_status_t(*transpose_f64)(matx_layout_t layout, matx_int64_t rows, matx_int64_t cols,
				const matx_double* A, matx_int64_t lda, matx_double* out, matx_int64_t ldc);
			matx_status_t(*transpose_c64)(matx_layout_t layout, matx_int64_t rows, matx_int64_t cols,
				const void* A, matx_int64_t lda, void* out, matx_int64_t ldc);
			matx_status_t(*conj_transpose_c64)(matx_layout_t layout, matx_int64_t rows, matx_int64_t cols,
				const void* A, matx_int64_t lda, void* out, matx_int64_t ldc);

			// ---- Norms ----
			matx_status_t(*norm1_f64)(matx_layout_t layout, matx_int64_t rows, matx_int64_t cols,
				const matx_double* A, matx_int64_t lda, matx_double* out);
			matx_status_t(*norminf_f64)(matx_layout_t layout, matx_int64_t rows, matx_int64_t cols,
				const matx_double* A, matx_int64_t lda, matx_double* out);
			matx_status_t(*normfro_f64)(matx_layout_t layout, matx_int64_t rows, matx_int64_t cols,
				const matx_double* A, matx_int64_t lda, matx_double* out);
			matx_status_t(*norm1_c64)(matx_layout_t layout, matx_int64_t rows, matx_int64_t cols,
				const void* A, matx_int64_t lda, matx_double* out);
			matx_status_t(*norminf_c64)(matx_layout_t layout, matx_int64_t rows, matx_int64_t cols,
				const void* A, matx_int64_t lda, matx_double* out);
			matx_status_t(*normfro_c64)(matx_layout_t layout, matx_int64_t rows, matx_int64_t cols,
				const void* A, matx_int64_t lda, matx_double* out);

	} matx_dense_vtable_t;

	typedef struct matx_dense_backend_t {
		matx_dense_backend_kind_t kind;
		matx_dense_vtable_t vt;
	} matx_dense_backend_t;

	// Initialize default backend based on MATX_BLAS_BACKEND (AUTO picks a reasonable default at build time).
	MATX_API  matx_dense_backend_t matx_blas_default(void);
	MATX_API  const char* matx_blas_backend_name(matx_dense_backend_kind_t k);

	// Convenience API operating on MatX dense types.
	MATX_API  matx_status_t matx_gemm_f64(const matx_dense_backend_t* blas,
		matx_int64_t trans_a,
		matx_int64_t trans_b,
		matx_double alpha,
		const matx_dense_f64_t A,
		const matx_dense_f64_t B,
		matx_double beta,
		matx_dense_f64_t C);

	MATX_API matx_status_t matx_gemm_c64(const matx_dense_backend_t* blas,
		matx_int64_t trans_a,
		matx_int64_t trans_b,
		matx_complex_f64_t alpha,
		const matx_dense_c64_t A,
		const matx_dense_c64_t B,
		matx_complex_f64_t beta,
		matx_dense_c64_t C);

	// ---- Level 1: vector ops ----
	// y := alpha * x + y
	MATX_API matx_status_t matx_axpy_f64(const matx_dense_backend_t* blas,
		matx_double alpha,
		const matx_vec_f64_t x,
		matx_vec_f64_t y);

	MATX_API matx_status_t matx_axpy_c64(
		const matx_dense_backend_t* blas,
		matx_complex_f64_t alpha,
		const matx_vec_c64_t x,
		matx_vec_c64_t y);

	// ---- Level 2: matrix-vector ----
	// y := alpha * op(A) * x + beta * y  (dense)
	MATX_API matx_status_t matx_gemv_f64(const matx_dense_backend_t* blas,
		matx_int64_t trans_a,
		matx_double alpha,
		const matx_dense_f64_t A,
		const matx_vec_f64_t x,
		matx_double beta,
		matx_vec_f64_t y);

	MATX_API matx_status_t matx_gemv_c64(const matx_dense_backend_t* blas,
		matx_int64_t trans_a,
		matx_complex_f64_t alpha,
		const matx_dense_c64_t A,
		const matx_vec_c64_t x,
		matx_complex_f64_t beta,
		matx_vec_c64_t y);

	// ---- Level 3: matrix-matrix ----
	// C := alpha * op(A) * op(B) + beta * C  (dense)
	MATX_API matx_status_t matx_gemm_f64(const matx_dense_backend_t* blas,
		matx_int64_t trans_a,
		matx_int64_t trans_b,
		matx_double alpha,
		const matx_dense_f64_t A,
		const matx_dense_f64_t B,
		matx_double beta,
		matx_dense_f64_t C);

	// B := alpha * A + beta * B  (dense complex)
	MATX_API matx_status_t matx_geadd_c64(const matx_dense_backend_t* blas,
		matx_complex_f64_t alpha,
		const matx_dense_c64_t A,
		matx_complex_f64_t beta,
		matx_dense_c64_t B);

	// B := alpha * A + beta * B  (dense real)
	MATX_API matx_status_t matx_geadd_f64(const matx_dense_backend_t* blas,
		matx_double alpha,
		const matx_dense_f64_t A,
		matx_double beta,
		matx_dense_f64_t B);

	// ---- Level 1: scale / copy / swap / dot / nrm2 / asum / iamax ----
	MATX_API matx_status_t matx_scal_f64(const matx_dense_backend_t* blas, matx_double alpha, matx_vec_f64_t x);
	MATX_API matx_status_t matx_scal_c64(const matx_dense_backend_t* blas, matx_complex_f64_t alpha, matx_vec_c64_t x);
	MATX_API matx_status_t matx_copy_f64(const matx_dense_backend_t* blas, const matx_vec_f64_t x, matx_vec_f64_t y);
	MATX_API matx_status_t matx_copy_c64(const matx_dense_backend_t* blas, const matx_vec_c64_t x, matx_vec_c64_t y);
	MATX_API matx_status_t matx_swap_f64(const matx_dense_backend_t* blas, matx_vec_f64_t x, matx_vec_f64_t y);
	MATX_API matx_status_t matx_swap_c64(const matx_dense_backend_t* blas, matx_vec_c64_t x, matx_vec_c64_t y);
	MATX_API matx_status_t matx_dot_f64(const matx_dense_backend_t* blas, const matx_vec_f64_t x, const matx_vec_f64_t y, matx_double* result);
	MATX_API matx_status_t matx_dotu_c64(const matx_dense_backend_t* blas, const matx_vec_c64_t x, const matx_vec_c64_t y, matx_complex_f64_t* result);
	MATX_API matx_status_t matx_dotc_c64(const matx_dense_backend_t* blas, const matx_vec_c64_t x, const matx_vec_c64_t y, matx_complex_f64_t* result);
	MATX_API matx_status_t matx_nrm2_f64(const matx_dense_backend_t* blas, const matx_vec_f64_t x, matx_double* result);
	MATX_API matx_status_t matx_nrm2_c64(const matx_dense_backend_t* blas, const matx_vec_c64_t x, matx_double* result);
	MATX_API matx_status_t matx_asum_f64(const matx_dense_backend_t* blas, const matx_vec_f64_t x, matx_double* result);
	MATX_API matx_status_t matx_asum_c64(const matx_dense_backend_t* blas, const matx_vec_c64_t x, matx_double* result);
	MATX_API matx_status_t matx_iamax_f64(const matx_dense_backend_t* blas, const matx_vec_f64_t x, matx_int64_t* result);
	MATX_API matx_status_t matx_iamax_c64(const matx_dense_backend_t* blas, const matx_vec_c64_t x, matx_int64_t* result);

	// ---- Level 2: rank-1 update / triangular solve ----
	MATX_API matx_status_t matx_ger_f64(const matx_dense_backend_t* blas, matx_double alpha,
		const matx_vec_f64_t x, const matx_vec_f64_t y, matx_dense_f64_t A);
	MATX_API matx_status_t matx_geru_c64(const matx_dense_backend_t* blas, matx_complex_f64_t alpha,
		const matx_vec_c64_t x, const matx_vec_c64_t y, matx_dense_c64_t A);
	MATX_API matx_status_t matx_trsv_f64(const matx_dense_backend_t* blas, int uplo, int trans, int diag,
		const matx_dense_f64_t A, matx_vec_f64_t x);
	MATX_API matx_status_t matx_trsv_c64(const matx_dense_backend_t* blas, int uplo, int trans, int diag,
		const matx_dense_c64_t A, matx_vec_c64_t x);

	// ---- Level 3: triangular solve / symmetric rank-k update ----
	MATX_API matx_status_t matx_trsm_f64(const matx_dense_backend_t* blas, int side, int uplo, int trans, int diag,
		matx_double alpha, const matx_dense_f64_t A, matx_dense_f64_t B);
	MATX_API matx_status_t matx_trsm_c64(const matx_dense_backend_t* blas, int side, int uplo, int trans, int diag,
		matx_complex_f64_t alpha, const matx_dense_c64_t A, matx_dense_c64_t B);
	MATX_API matx_status_t matx_syrk_f64(const matx_dense_backend_t* blas, int uplo, int trans,
		matx_double alpha, const matx_dense_f64_t A, matx_double beta, matx_dense_f64_t C);
	MATX_API matx_status_t matx_herk_c64(const matx_dense_backend_t* blas, int uplo, int trans,
		matx_double alpha, const matx_dense_c64_t A, matx_double beta, matx_dense_c64_t C);

	// ---- Transpose ----
	MATX_API matx_status_t matx_transpose_f64(const matx_dense_backend_t* blas,
		const matx_dense_f64_t A, matx_dense_f64_t out);
	MATX_API matx_status_t matx_transpose_c64(const matx_dense_backend_t* blas,
		const matx_dense_c64_t A, matx_dense_c64_t out);
	MATX_API matx_status_t matx_conj_transpose_c64(const matx_dense_backend_t* blas,
		const matx_dense_c64_t A, matx_dense_c64_t out);

	// ---- Matrix norms ----
	MATX_API matx_status_t matx_mat_norm1_f64(const matx_dense_backend_t* blas,
		const matx_dense_f64_t A, matx_double* out);
	MATX_API matx_status_t matx_mat_norminf_f64(const matx_dense_backend_t* blas,
		const matx_dense_f64_t A, matx_double* out);
	MATX_API matx_status_t matx_mat_normfro_f64(const matx_dense_backend_t* blas,
		const matx_dense_f64_t A, matx_double* out);
	MATX_API matx_status_t matx_mat_norm1_c64(const matx_dense_backend_t* blas,
		const matx_dense_c64_t A, matx_double* out);
	MATX_API matx_status_t matx_mat_norminf_c64(const matx_dense_backend_t* blas,
		const matx_dense_c64_t A, matx_double* out);
	MATX_API matx_status_t matx_mat_normfro_c64(const matx_dense_backend_t* blas,
		const matx_dense_c64_t A, matx_double* out);

#ifdef __cplusplus
}
#endif

