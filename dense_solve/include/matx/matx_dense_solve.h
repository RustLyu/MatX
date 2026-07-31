#ifndef MATX_DENSE_SOLVE_H
#define MATX_DENSE_SOLVE_H

#include "matx/matx_types.h"
#include "matx/matx_func.h"

#ifdef __cplusplus
extern "C" {
#endif

	typedef enum matx_dense_linsolve_backend_kind_t {
		MATX_LINSOLVE_BACKEND_CBLAS = 0
	} matx_dense_linsolve_backend_kind_t;

	// Opaque factorization handles
	typedef struct matx_factor_dense_d_i8_t matx_factor_dense_d_i8_t;
	typedef struct matx_factor_dense_z_i8_t matx_factor_dense_z_i8_t;

	typedef struct matx_dense_linsolve_vtable_t {
		// Dense real LU
		matx_status_t(*factor_dense_d_i8)(const matx_dense_d_i8_t A,
			matx_factor_dense_d_i8_t** out_F);
		matx_status_t(*solve_dense_d_i8)(const matx_factor_dense_d_i8_t* F,
			const double* b,
			double* x);
		void (*factor_dense_d_i8_destroy)(matx_factor_dense_d_i8_t* F);

		// Dense complex LU
		matx_status_t(*factor_dense_z_i8)(const matx_dense_z_i8_t A,
			matx_factor_dense_z_i8_t** out_F);
		matx_status_t(*solve_dense_z_i8)(const matx_factor_dense_z_i8_t* F,
			const matx_vec_z_i8_t b,
			matx_vec_z_i8_t x);
		void (*factor_dense_z_i8_destroy)(matx_factor_dense_z_i8_t* F);

		// Cholesky
		matx_status_t(*potrf_d_i8)(const matx_dense_d_i8_t A, matx_uplo_t uplo, matx_factor_dense_d_i8_t** out_F);
		matx_status_t(*potrs_d_i8)(const matx_factor_dense_d_i8_t* F, const matx_double* b, matx_double* x);
		matx_status_t(*potrf_z_i8)(const matx_dense_z_i8_t A, matx_uplo_t uplo, matx_factor_dense_z_i8_t** out_F);
		matx_status_t(*potrs_z_i8)(const matx_factor_dense_z_i8_t* F, const matx_vec_z_i8_t b, matx_vec_z_i8_t x);

		// Least squares (GELS)
		matx_status_t(*gels_d_i8)(const matx_dense_d_i8_t A, const matx_double* b, matx_double* x);
		matx_status_t(*gels_z_i8)(const matx_dense_z_i8_t A, const matx_vec_z_i8_t b, matx_vec_z_i8_t x);

		// Symmetric eigenvalue (real only)
		matx_status_t(*syev_d_i8)(const matx_dense_d_i8_t A, matx_vec_d_i8_t eigenvalues, matx_dense_d_i8_t* eigenvectors);

		// SVD
		matx_status_t(*gesvd_d_i8)(const matx_dense_d_i8_t A, matx_vec_d_i8_t S, matx_dense_d_i8_t* U, matx_dense_d_i8_t* Vt);
		matx_status_t(*gesvd_z_i8)(const matx_dense_z_i8_t A, matx_vec_d_i8_t S, matx_dense_z_i8_t* U, matx_dense_z_i8_t* Vt);
	} matx_dense_linsolve_vtable_t;

	typedef struct matx_dense_linsolve_t {
		matx_dense_linsolve_backend_kind_t kind;
		matx_dense_linsolve_vtable_t vt;
	} matx_dense_linsolve_t;

	MATX_API matx_dense_linsolve_t matx_dense_linsolve_default(void);
	MATX_API const char* matx_dense_linsolve_backend_name(matx_dense_linsolve_backend_kind_t k);

	// High-level API (thin wrappers over vtable) -------------------------------

	// ---- Dense real LU ----

	/**
	 * @brief LU factorization of a real dense matrix with partial pivoting (DGETRF)
	 * @formula P * A = L * U
	 *          where P is a permutation matrix, L is lower triangular with unit diagonal,
	 *          U is upper triangular.
	 */
	MATX_API matx_status_t matx_factor_dense_d_i8(const matx_dense_linsolve_t* ls,
		const matx_dense_d_i8_t A,
		matx_factor_dense_d_i8_t** out_F);

	/**
	 * @brief Solve a real linear system using pre-computed LU factorization (DGETRS)
	 * @formula A * x = b  =>  x = A^{-1} * b
	 */
	MATX_API matx_status_t matx_solve_dense_d_i8_factor(const matx_dense_linsolve_t* ls,
		const matx_factor_dense_d_i8_t* F,
		const matx_double* b,
		matx_double* x);

	MATX_API void matx_factor_dense_d_i8_destroy(const matx_dense_linsolve_t* ls,
		matx_factor_dense_d_i8_t* F);

	/**
	 * @brief Solve a real linear system directly via LU factorization (DGETRF + DGETRS)
	 * @formula A * x = b  =>  x = A^{-1} * b
	 *          Factorizes A internally, solves for x, and discards the factorization.
	 */
	MATX_API matx_status_t matx_solve_dense_d_i8(const matx_dense_linsolve_t* ls,
		const matx_dense_d_i8_t A,
		const matx_double* b,
		matx_double* x);

	// ---- Dense complex LU ----

	/**
	 * @brief LU factorization of a complex dense matrix with partial pivoting (ZGETRF)
	 * @formula P * A = L * U
	 *          where P is a permutation matrix, L is lower triangular with unit diagonal,
	 *          U is upper triangular.
	 */
	MATX_API matx_status_t matx_factor_dense_z_i8(const matx_dense_linsolve_t* ls,
		const matx_dense_z_i8_t A,
		matx_factor_dense_z_i8_t** out_F);

	/**
	 * @brief Solve a complex linear system using pre-computed LU factorization (ZGETRS)
	 * @formula A * x = b  =>  x = A^{-1} * b
	 */
	MATX_API matx_status_t matx_solve_dense_z_i8_factor(const matx_dense_linsolve_t* ls,
		const matx_factor_dense_z_i8_t* F,
		const matx_vec_z_i8_t b,
		matx_vec_z_i8_t x);

	MATX_API void matx_factor_dense_z_i8_destroy(const matx_dense_linsolve_t* ls,
		matx_factor_dense_z_i8_t* F);

	/**
	 * @brief Solve a complex linear system directly via LU factorization (ZGETRF + ZGETRS)
	 * @formula A * x = b  =>  x = A^{-1} * b
	 *          Factorizes A internally, solves for x, and discards the factorization.
	 */
	MATX_API matx_status_t matx_solve_dense_z_i8(const matx_dense_linsolve_t* ls,
		const matx_dense_z_i8_t A,
		const matx_vec_z_i8_t b,
		matx_vec_z_i8_t x);

	// ---- Cholesky ----

	/**
	 * @brief Cholesky factorization of a real SPD matrix (DPOTRF)
	 * @formula A = L * L^T  (uplo=L)  or  A = U^T * U  (uplo=U)
	 *          where L is lower triangular, U is upper triangular.
	 *          A must be symmetric positive-definite.
	 */
	MATX_API matx_status_t matx_factor_chol_d_i8(const matx_dense_linsolve_t* ls,
		const matx_dense_d_i8_t A, matx_uplo_t uplo, matx_factor_dense_d_i8_t** out_F);

	/**
	 * @brief Solve a real SPD system using Cholesky factorization (DPOTRS)
	 * @formula A * x = b  =>  x = A^{-1} * b
	 */
	MATX_API matx_status_t matx_solve_chol_d_i8(const matx_dense_linsolve_t* ls,
		const matx_factor_dense_d_i8_t* F, const matx_double* b, matx_double* x);

	/**
	 * @brief Solve a real SPD system in one shot (DPOTRF + DPOTRS)
	 * @formula A * x = b  =>  x = A^{-1} * b
	 */
	MATX_API matx_status_t matx_solve_chol_d_i8_oneshot(const matx_dense_linsolve_t* ls,
		const matx_dense_d_i8_t A, matx_uplo_t uplo, const matx_double* b, matx_double* x);

	/**
	 * @brief Cholesky factorization of a complex HPD matrix (ZPOTRF)
	 * @formula A = L * L^H  (uplo=L)  or  A = U^H * U  (uplo=U)
	 *          where L is lower triangular, U is upper triangular.
	 *          A must be Hermitian positive-definite.
	 */
	MATX_API matx_status_t matx_factor_chol_z_i8(const matx_dense_linsolve_t* ls,
		const matx_dense_z_i8_t A, matx_uplo_t uplo, matx_factor_dense_z_i8_t** out_F);

	/**
	 * @brief Solve a complex HPD system using Cholesky factorization (ZPOTRS)
	 * @formula A * x = b  =>  x = A^{-1} * b
	 */
	MATX_API matx_status_t matx_solve_chol_z_i8(const matx_dense_linsolve_t* ls,
		const matx_factor_dense_z_i8_t* F, const matx_vec_z_i8_t b, matx_vec_z_i8_t x);

	// ---- Least squares (GELS) ----

	/**
	 * @brief Solve a real linear least-squares problem via QR/LQ (DGELS)
	 * @formula min ||b - A * x||_2  (overdetermined, m >= n)
	 *          or  min ||x||_2 s.t. A * x = b  (underdetermined, m < n)
	 */
	MATX_API matx_status_t matx_gels_d_i8(const matx_dense_linsolve_t* ls,
		const matx_dense_d_i8_t A, const matx_double* b, matx_double* x);

	/**
	 * @brief Solve a complex linear least-squares problem via QR/LQ (ZGELS)
	 * @formula min ||b - A * x||_2  (overdetermined, m >= n)
	 *          or  min ||x||_2 s.t. A * x = b  (underdetermined, m < n)
	 */
	MATX_API matx_status_t matx_gels_z_i8(const matx_dense_linsolve_t* ls,
		const matx_dense_z_i8_t A, const matx_vec_z_i8_t b, matx_vec_z_i8_t x);

	// ---- Symmetric eigenvalue (SYEV) ----

	/**
	 * @brief Eigenvalue decomposition of a real symmetric matrix (DSYEV)
	 * @formula A = Q * diag(W) * Q^T
	 *          where Q is orthogonal, W is the vector of eigenvalues in ascending order.
	 *          eigenvalues must be pre-allocated (size n); eigenvectors allocated by callee (or NULL to skip)
	 */
	MATX_API matx_status_t matx_syev_d_i8(const matx_dense_linsolve_t* ls,
		const matx_dense_d_i8_t A, matx_vec_d_i8_t eigenvalues, matx_dense_d_i8_t* eigenvectors);

	// ---- SVD (GESVD) ----

	/**
	 * @brief Singular value decomposition of a real matrix (DGESVD)
	 * @formula A = U * diag(S) * V^T
	 *          where U is m-by-min(m,n), S contains singular values in descending order,
	 *          V^T is min(m,n)-by-n. A is m-by-n, overwritten.
	 *          S must be pre-allocated (size min(m,n)); U and Vt allocated by callee (or NULL to skip)
	 */
	MATX_API matx_status_t matx_gesvd_d_i8(const matx_dense_linsolve_t* ls,
		const matx_dense_d_i8_t A, matx_vec_d_i8_t S, matx_dense_d_i8_t* U, matx_dense_d_i8_t* Vt);

	/**
	 * @brief Singular value decomposition of a complex matrix (ZGESVD)
	 * @formula A = U * diag(S) * V^H
	 *          where U is m-by-min(m,n), S contains singular values in descending order,
	 *          V^H is min(m,n)-by-n. A is m-by-n, overwritten.
	 *          S must be pre-allocated (size min(m,n)); U and Vt allocated by callee (or NULL to skip)
	 */
	MATX_API matx_status_t matx_gesvd_z_i8(const matx_dense_linsolve_t* ls,
		const matx_dense_z_i8_t A, matx_vec_d_i8_t S, matx_dense_z_i8_t* U, matx_dense_z_i8_t* Vt);

#ifdef __cplusplus
}
#endif

#endif // MATX_DENSE_SOLVE_H