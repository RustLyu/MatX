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

		matx_status_t(*inv_dense_d_i8)(
			matx_layout_t layout,
			matx_int64_t rows,
			matx_int64_t cols,
			const matx_double* A,
			matx_double* out_Ainv);

		matx_status_t(*inv_dense_z_i8)(matx_layout_t layout,
			matx_int64_t rows,
			matx_int64_t cols,
			const void* A,
			void* out_Ainv);

		// ---- Level 2 additions ----
		matx_status_t(*dger)(matx_layout_t layout, matx_int64_t m, matx_int64_t n, matx_double alpha,
			const matx_double* x, matx_int64_t incx, const matx_double* y, matx_int64_t incy,
			matx_double* A, matx_int64_t lda);
		matx_status_t(*zgeru)(matx_layout_t layout, matx_int64_t m, matx_int64_t n, const void* alpha,
			const void* x, matx_int64_t incx, const void* y, matx_int64_t incy,
			void* A, matx_int64_t lda);
		matx_status_t(*zgerc)(matx_layout_t layout, matx_int64_t m, matx_int64_t n, const void* alpha,
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
		matx_status_t(*dsyr2k)(matx_layout_t layout, int uplo, int trans,
			matx_int64_t n, matx_int64_t k, matx_double alpha,
			const matx_double* A, matx_int64_t lda, const matx_double* B, matx_int64_t ldb,
			matx_double beta, matx_double* C, matx_int64_t ldc);
		matx_status_t(*zher2k)(matx_layout_t layout, int uplo, int trans,
			matx_int64_t n, matx_int64_t k, const void* alpha,
			const void* A, matx_int64_t lda, const void* B, matx_int64_t ldb,
			matx_double beta, void* C, matx_int64_t ldc);

		// ---- Transpose ----
		matx_status_t(*transpose_d_i8)(matx_layout_t layout, matx_int64_t rows, matx_int64_t cols,
			const matx_double* A, matx_int64_t lda, matx_double* out, matx_int64_t ldc);
		matx_status_t(*transpose_z_i8)(matx_layout_t layout, matx_int64_t rows, matx_int64_t cols,
			const void* A, matx_int64_t lda, void* out, matx_int64_t ldc);
		matx_status_t(*conj_transpose_z_i8)(matx_layout_t layout, matx_int64_t rows, matx_int64_t cols,
			const void* A, matx_int64_t lda, void* out, matx_int64_t ldc);

		// ---- Element-wise (Hadamard) ----
		matx_status_t(*hadamard_d_i8)(matx_layout_t layout, matx_int64_t rows, matx_int64_t cols,
			const matx_double* A, matx_int64_t lda, const matx_double* B, matx_int64_t ldb,
			matx_double* C, matx_int64_t ldc);
		matx_status_t(*hadamard_z_i8)(matx_layout_t layout, matx_int64_t rows, matx_int64_t cols,
			const void* A, matx_int64_t lda, const void* B, matx_int64_t ldb,
			void* C, matx_int64_t ldc);

		// ---- Norms ----
		matx_status_t(*norm1_d_i8)(matx_layout_t layout, matx_int64_t rows, matx_int64_t cols,
			const matx_double* A, matx_int64_t lda, matx_double* out);
		matx_status_t(*norminf_d_i8)(matx_layout_t layout, matx_int64_t rows, matx_int64_t cols,
			const matx_double* A, matx_int64_t lda, matx_double* out);
		matx_status_t(*normfro_d_i8)(matx_layout_t layout, matx_int64_t rows, matx_int64_t cols,
			const matx_double* A, matx_int64_t lda, matx_double* out);
		matx_status_t(*norm1_z_i8)(matx_layout_t layout, matx_int64_t rows, matx_int64_t cols,
			const void* A, matx_int64_t lda, matx_double* out);
		matx_status_t(*norminf_z_i8)(matx_layout_t layout, matx_int64_t rows, matx_int64_t cols,
			const void* A, matx_int64_t lda, matx_double* out);
		matx_status_t(*normfro_z_i8)(matx_layout_t layout, matx_int64_t rows, matx_int64_t cols,
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

	/**
	 * @brief General matrix-matrix multiply for real matrices (DGEMM)
	 * @formula C := alpha * op(A) * op(B) + beta * C
	 *          where op(X) = X, X^T, or X^H depending on trans_a/trans_b
	 */
	MATX_API  matx_status_t matx_gemm_d_i8(const matx_dense_backend_t* blas,
		matx_int64_t trans_a,
		matx_int64_t trans_b,
		matx_double alpha,
		const matx_dense_d_i8_t A,
		const matx_dense_d_i8_t B,
		matx_double beta,
		matx_dense_d_i8_t C);

	/**
	 * @brief General matrix-matrix multiply for complex matrices (ZGEMM)
	 * @formula C := alpha * op(A) * op(B) + beta * C
	 *          where op(X) = X, X^T, or X^H depending on trans_a/trans_b
	 */
	MATX_API matx_status_t matx_gemm_z_i8(const matx_dense_backend_t* blas,
		matx_int64_t trans_a,
		matx_int64_t trans_b,
		matx_complex_d_i8_t alpha,
		const matx_dense_z_i8_t A,
		const matx_dense_z_i8_t B,
		matx_complex_d_i8_t beta,
		matx_dense_z_i8_t C);

	// ---- Level 2: matrix-vector ----

	/**
	 * @brief General matrix-vector multiply for real matrices (DGEMV)
	 * @formula y := alpha * op(A) * x + beta * y
	 *          where op(A) = A or A^T depending on trans_a
	 */
	MATX_API matx_status_t matx_gemv_d_i8(const matx_dense_backend_t* blas,
		matx_int64_t trans_a,
		matx_double alpha,
		const matx_dense_d_i8_t A,
		matx_vec_d_i8_t x,
		matx_double beta,
		matx_vec_d_i8_t y);

	/**
	 * @brief General matrix-vector multiply for complex matrices (ZGEMV)
	 * @formula y := alpha * op(A) * x + beta * y
	 *          where op(A) = A, A^T, or A^H depending on trans_a
	 */
	MATX_API matx_status_t matx_gemv_z_i8(const matx_dense_backend_t* blas,
		matx_int64_t trans_a,
		matx_complex_d_i8_t alpha,
		const matx_dense_z_i8_t A,
		matx_vec_z_i8_t x,
		matx_complex_d_i8_t beta,
		matx_vec_z_i8_t y);

	// ---- Level 3: matrix-matrix ----

	/**
	 * @brief General matrix-matrix multiply for real matrices (DGEMM)
	 * @formula C := alpha * op(A) * op(B) + beta * C
	 *          where op(X) = X, X^T depending on trans_a/trans_b
	 */
	MATX_API matx_status_t matx_gemm_d_i8(const matx_dense_backend_t* blas,
		matx_int64_t trans_a,
		matx_int64_t trans_b,
		matx_double alpha,
		const matx_dense_d_i8_t A,
		const matx_dense_d_i8_t B,
		matx_double beta,
		matx_dense_d_i8_t C);

	/**
	 * @brief General matrix addition for complex matrices (ZGEADD)
	 * @formula B := alpha * A + beta * B
	 */
	MATX_API matx_status_t matx_geadd_z_i8(const matx_dense_backend_t* blas,
		matx_complex_d_i8_t alpha,
		const matx_dense_z_i8_t A,
		matx_complex_d_i8_t beta,
		matx_dense_z_i8_t B);

	/**
	 * @brief General matrix addition for real matrices (DGEADD)
	 * @formula B := alpha * A + beta * B
	 */
	MATX_API matx_status_t matx_geadd_d_i8(const matx_dense_backend_t* blas,
		matx_double alpha,
		const matx_dense_d_i8_t A,
		matx_double beta,
		matx_dense_d_i8_t B);

	// ---- Level 2: rank-1 update / triangular solve ----

	/**
	 * @brief Real rank-1 update (DGER)
	 * @formula A := alpha * x * y^T + A
	 *          A is m-by-n, x is m-by-1, y is n-by-1
	 */
	MATX_API matx_status_t matx_ger_d_i8(const matx_dense_backend_t* blas, matx_double alpha,
		const matx_vec_d_i8_t x, const matx_vec_d_i8_t y, matx_dense_d_i8_t A);

	/**
	 * @brief Complex unconjugated rank-1 update (ZGERU)
	 * @formula A := alpha * x * y^T + A
	 *          A is m-by-n, x is m-by-1, y is n-by-1
	 */
	MATX_API matx_status_t matx_geru_z_i8(const matx_dense_backend_t* blas, matx_complex_d_i8_t alpha,
		const matx_vec_z_i8_t x, const matx_vec_z_i8_t y, matx_dense_z_i8_t A);

	/**
	 * @brief Complex conjugated rank-1 update (ZGERC)
	 * @formula A := alpha * x * y^H + A
	 *          A is m-by-n, x is m-by-1, y is n-by-1
	 */
	MATX_API matx_status_t matx_gerc_z_i8(const matx_dense_backend_t* blas, matx_complex_d_i8_t alpha,
		const matx_vec_z_i8_t x, const matx_vec_z_i8_t y, matx_dense_z_i8_t A);

	/**
	 * @brief Real triangular solve (DTRSV)
	 * @formula x := op(A)^{-1} * x
	 *          where op(A) = A, A^T, or A^H; A is n-by-n triangular
	 */
	MATX_API matx_status_t matx_trsv_d_i8(const matx_dense_backend_t* blas, int uplo, int trans, int diag,
		const matx_dense_d_i8_t A, matx_vec_d_i8_t x);

	/**
	 * @brief Complex triangular solve (ZTRSV)
	 * @formula x := op(A)^{-1} * x
	 *          where op(A) = A, A^T, or A^H; A is n-by-n triangular
	 */
	MATX_API matx_status_t matx_trsv_z_i8(const matx_dense_backend_t* blas, int uplo, int trans, int diag,
		const matx_dense_z_i8_t A, matx_vec_z_i8_t x);

	// ---- Level 3: triangular solve / symmetric rank-k update ----

	/**
	 * @brief Real triangular solve with multiple right-hand sides (DTRSM)
	 * @formula B := alpha * op(A)^{-1} * B  (side=L)
	 *          or  B := alpha * B * op(A)^{-1}  (side=R)
	 *          where op(A) = A, A^T, or A^H; A is triangular
	 */
	MATX_API matx_status_t matx_trsm_d_i8(const matx_dense_backend_t* blas, int side, int uplo, int trans, int diag,
		matx_double alpha, const matx_dense_d_i8_t A, matx_dense_d_i8_t B);

	/**
	 * @brief Complex triangular solve with multiple right-hand sides (ZTRSM)
	 * @formula B := alpha * op(A)^{-1} * B  (side=L)
	 *          or  B := alpha * B * op(A)^{-1}  (side=R)
	 *          where op(A) = A, A^T, or A^H; A is triangular
	 */
	MATX_API matx_status_t matx_trsm_z_i8(const matx_dense_backend_t* blas, int side, int uplo, int trans, int diag,
		matx_complex_d_i8_t alpha, const matx_dense_z_i8_t A, matx_dense_z_i8_t B);

	/**
	 * @brief Real symmetric rank-k update (DSYRK)
	 * @formula C := alpha * op(A) * A^T + beta * C  (trans=N)
	 *          or  C := alpha * A^T * A + beta * C  (trans=T)
	 *          C is n-by-n symmetric, A is n-by-k or k-by-n
	 */
	MATX_API matx_status_t matx_syrk_d_i8(const matx_dense_backend_t* blas, int uplo, int trans,
		matx_double alpha, const matx_dense_d_i8_t A, matx_double beta, matx_dense_d_i8_t C);

	/**
	 * @brief Complex Hermitian rank-k update (ZHERK)
	 * @formula C := alpha * op(A) * A^H + beta * C  (trans=N)
	 *          or  C := alpha * A^H * A + beta * C  (trans=T)
	 *          C is n-by-n Hermitian, A is n-by-k or k-by-n, alpha and beta are real
	 */
	MATX_API matx_status_t matx_herk_z_i8(const matx_dense_backend_t* blas, int uplo, int trans,
		matx_double alpha, const matx_dense_z_i8_t A, matx_double beta, matx_dense_z_i8_t C);

	/**
	 * @brief Real symmetric rank-2k update (DSYR2K)
	 * @formula C := alpha * A * B^T + alpha * B * A^T + beta * C  (trans=N)
	 *          or  C := alpha * A^T * B + alpha * B^T * A + beta * C  (trans=T)
	 *          C is n-by-n symmetric, A and B are n-by-k or k-by-n
	 */
	MATX_API matx_status_t matx_syr2k_d_i8(const matx_dense_backend_t* blas, int uplo, int trans,
		matx_double alpha, const matx_dense_d_i8_t A, const matx_dense_d_i8_t B,
		matx_double beta, matx_dense_d_i8_t C);

	/**
	 * @brief Complex Hermitian rank-2k update (ZHER2K)
	 * @formula C := alpha * A * B^H + conj(alpha) * B * A^H + beta * C  (trans=N)
	 *          or  C := alpha * A^H * B + conj(alpha) * B^H * A + beta * C  (trans=T)
	 *          C is n-by-n Hermitian, A and B are n-by-k or k-by-n, beta is real
	 */
	MATX_API matx_status_t matx_her2k_z_i8(const matx_dense_backend_t* blas, int uplo, int trans,
		matx_complex_d_i8_t alpha, const matx_dense_z_i8_t A, const matx_dense_z_i8_t B,
		matx_double beta, matx_dense_z_i8_t C);

	// ---- Element-wise (Hadamard product) ----

	/**
	 * @brief Real element-wise matrix multiply (Hadamard product)
	 * @formula C = A .⊙ B  (element-wise)
	 */
	MATX_API matx_status_t matx_hadamard_d_i8(const matx_dense_backend_t* blas,
		const matx_dense_d_i8_t A, const matx_dense_d_i8_t B, matx_dense_d_i8_t C);

	/**
	 * @brief Complex element-wise matrix multiply (Hadamard product)
	 * @formula C = A .⊙ B  (element-wise)
	 */
	MATX_API matx_status_t matx_hadamard_z_i8(const matx_dense_backend_t* blas,
		const matx_dense_z_i8_t A, const matx_dense_z_i8_t B, matx_dense_z_i8_t C);

	// ---- Transpose ----

	/**
	 * @brief Transpose a real dense matrix
	 * @formula out[i][j] = A[j][i]
	 */
	MATX_API matx_status_t matx_transpose_d_i8(const matx_dense_backend_t* blas,
		const matx_dense_d_i8_t A, matx_dense_d_i8_t out);

	/**
	 * @brief Transpose a complex dense matrix (no conjugation)
	 * @formula out[i][j] = A[j][i]
	 */
	MATX_API matx_status_t matx_transpose_z_i8(const matx_dense_backend_t* blas,
		const matx_dense_z_i8_t A, matx_dense_z_i8_t out);

	/**
	 * @brief Conjugate transpose of a complex dense matrix (Hermitian transpose)
	 * @formula out[i][j] = conj(A[j][i])
	 */
	MATX_API matx_status_t matx_conj_transpose_z_i8(const matx_dense_backend_t* blas,
		const matx_dense_z_i8_t A, matx_dense_z_i8_t out);

	// ---- Matrix norms ----

	/**
	 * @brief 1-norm of a real dense matrix (max column sum)
	 * @formula ||A||_1 = max_{j=0,...,n-1} sum_{i=0}^{m-1} |A[i][j]|
	 */
	MATX_API matx_status_t matx_mat_norm1_d_i8(const matx_dense_backend_t* blas,
		const matx_dense_d_i8_t A, matx_double* out);

	/**
	 * @brief Infinity-norm of a real dense matrix (max row sum)
	 * @formula ||A||_inf = max_{i=0,...,m-1} sum_{j=0}^{n-1} |A[i][j]|
	 */
	MATX_API matx_status_t matx_mat_norminf_d_i8(const matx_dense_backend_t* blas,
		const matx_dense_d_i8_t A, matx_double* out);

	/**
	 * @brief Frobenius norm of a real dense matrix
	 * @formula ||A||_F = sqrt( sum_{i=0}^{m-1} sum_{j=0}^{n-1} A[i][j]^2 )
	 */
	MATX_API matx_status_t matx_mat_normfro_d_i8(const matx_dense_backend_t* blas,
		const matx_dense_d_i8_t A, matx_double* out);

	/**
	 * @brief 1-norm of a complex dense matrix (max column sum)
	 * @formula ||A||_1 = max_{j=0,...,n-1} sum_{i=0}^{m-1} |A[i][j]|
	 */
	MATX_API matx_status_t matx_mat_norm1_z_i8(const matx_dense_backend_t* blas,
		const matx_dense_z_i8_t A, matx_double* out);

	/**
	 * @brief Infinity-norm of a complex dense matrix (max row sum)
	 * @formula ||A||_inf = max_{i=0,...,m-1} sum_{j=0}^{n-1} |A[i][j]|
	 */
	MATX_API matx_status_t matx_mat_norminf_z_i8(const matx_dense_backend_t* blas,
		const matx_dense_z_i8_t A, matx_double* out);

	/**
	 * @brief Frobenius norm of a complex dense matrix
	 * @formula ||A||_F = sqrt( sum_{i=0}^{m-1} sum_{j=0}^{n-1} |A[i][j]|^2 )
	 */
	MATX_API matx_status_t matx_mat_normfro_z_i8(const matx_dense_backend_t* blas,
		const matx_dense_z_i8_t A, matx_double* out);

#ifdef __cplusplus
}
#endif
