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
	typedef struct matx_factor_dense_f64_t matx_factor_dense_f64_t;
	typedef struct matx_factor_dense_c64_t matx_factor_dense_c64_t;

	typedef struct matx_dense_linsolve_vtable_t {
		// Dense real LU
		matx_status_t(*factor_dense_f64)(const matx_dense_f64_t A,
			matx_factor_dense_f64_t** out_F);
		matx_status_t(*solve_dense_f64)(const matx_factor_dense_f64_t* F,
			const double* b,
			double* x);
		void (*factor_dense_f64_destroy)(matx_factor_dense_f64_t* F);

		// Dense complex LU
		matx_status_t(*factor_dense_c64)(const matx_dense_c64_t A,
			matx_factor_dense_c64_t** out_F);
		matx_status_t(*solve_dense_c64)(const matx_factor_dense_c64_t* F,
			const matx_vec_c64_t b,
			matx_vec_c64_t x);
		void (*factor_dense_c64_destroy)(matx_factor_dense_c64_t* F);

		// Cholesky
		matx_status_t(*potrf_f64)(const matx_dense_f64_t A, int uplo, matx_factor_dense_f64_t** out_F);
		matx_status_t(*potrs_f64)(const matx_factor_dense_f64_t* F, const matx_double* b, matx_double* x);
		matx_status_t(*potrf_c64)(const matx_dense_c64_t A, int uplo, matx_factor_dense_c64_t** out_F);
		matx_status_t(*potrs_c64)(const matx_factor_dense_c64_t* F, const matx_vec_c64_t b, matx_vec_c64_t x);

		// Least squares (GELS)
		matx_status_t(*gels_f64)(const matx_dense_f64_t A, const matx_double* b, matx_double* x);
		matx_status_t(*gels_c64)(const matx_dense_c64_t A, const matx_vec_c64_t b, matx_vec_c64_t x);

		// Symmetric eigenvalue (real only)
		matx_status_t(*syev_f64)(const matx_dense_f64_t A, matx_vec_f64_t eigenvalues, matx_dense_f64_t* eigenvectors);

		// SVD
		matx_status_t(*gesvd_f64)(const matx_dense_f64_t A, matx_vec_f64_t S, matx_dense_f64_t* U, matx_dense_f64_t* Vt);
		matx_status_t(*gesvd_c64)(const matx_dense_c64_t A, matx_vec_f64_t S, matx_dense_c64_t* U, matx_dense_c64_t* Vt);
	} matx_dense_linsolve_vtable_t;

	typedef struct matx_dense_linsolve_t {
		matx_dense_linsolve_backend_kind_t kind;
		matx_dense_linsolve_vtable_t vt;
	} matx_dense_linsolve_t;

	MATX_API matx_dense_linsolve_t matx_dense_linsolve_default(void);
	MATX_API const char* matx_dense_linsolve_backend_name(matx_dense_linsolve_backend_kind_t k);

	// High-level API (thin wrappers over vtable) -------------------------------

	// Dense real
	MATX_API matx_status_t matx_factor_dense_f64(const matx_dense_linsolve_t* ls,
		const matx_dense_f64_t A,
		matx_factor_dense_f64_t** out_F);


	MATX_API matx_status_t matx_solve_dense_f64_factor(const matx_dense_linsolve_t* ls,
		const matx_factor_dense_f64_t* F,
		const matx_double* b,
		matx_double* x);

	MATX_API void matx_factor_dense_f64_destroy(const matx_dense_linsolve_t* ls,
		matx_factor_dense_f64_t* F);

	MATX_API matx_status_t matx_solve_dense_f64(const matx_dense_linsolve_t* ls,
		const matx_dense_f64_t A,
		const matx_double* b,
		matx_double* x);


	MATX_API matx_status_t matx_factor_dense_c64(const matx_dense_linsolve_t* ls,
		const matx_dense_c64_t A,
		matx_factor_dense_c64_t** out_F);

	MATX_API matx_status_t matx_solve_dense_c64_factor(const matx_dense_linsolve_t* ls,
		const matx_factor_dense_c64_t* F,
		const matx_vec_c64_t b,
		matx_vec_c64_t x);

	MATX_API void matx_factor_dense_c64_destroy(const matx_dense_linsolve_t* ls,
		matx_factor_dense_c64_t* F);

	MATX_API matx_status_t matx_solve_dense_c64(const matx_dense_linsolve_t* ls,
		const matx_dense_c64_t A,
		const matx_vec_c64_t b,
		matx_vec_c64_t x);

	// ---- Cholesky ----
	MATX_API matx_status_t matx_factor_chol_f64(const matx_dense_linsolve_t* ls,
		const matx_dense_f64_t A, int uplo, matx_factor_dense_f64_t** out_F);
	MATX_API matx_status_t matx_solve_chol_f64(const matx_dense_linsolve_t* ls,
		const matx_factor_dense_f64_t* F, const matx_double* b, matx_double* x);
	MATX_API matx_status_t matx_solve_chol_f64_oneshot(const matx_dense_linsolve_t* ls,
		const matx_dense_f64_t A, int uplo, const matx_double* b, matx_double* x);
	MATX_API matx_status_t matx_factor_chol_c64(const matx_dense_linsolve_t* ls,
		const matx_dense_c64_t A, int uplo, matx_factor_dense_c64_t** out_F);
	MATX_API matx_status_t matx_solve_chol_c64(const matx_dense_linsolve_t* ls,
		const matx_factor_dense_c64_t* F, const matx_vec_c64_t b, matx_vec_c64_t x);

	// ---- Least squares (GELS) ----
	MATX_API matx_status_t matx_gels_f64(const matx_dense_linsolve_t* ls,
		const matx_dense_f64_t A, const matx_double* b, matx_double* x);
	MATX_API matx_status_t matx_gels_c64(const matx_dense_linsolve_t* ls,
		const matx_dense_c64_t A, const matx_vec_c64_t b, matx_vec_c64_t x);

	// ---- Symmetric eigenvalue (SYEV) ----
	// eigenvalues must be pre-allocated (size n); eigenvectors allocated by callee (or NULL to skip)
	MATX_API matx_status_t matx_syev_f64(const matx_dense_linsolve_t* ls,
		const matx_dense_f64_t A, matx_vec_f64_t eigenvalues, matx_dense_f64_t* eigenvectors);

	// ---- SVD (GESVD) ----
	// S must be pre-allocated (size min(m,n)); U and Vt allocated by callee (or NULL to skip)
	MATX_API matx_status_t matx_gesvd_f64(const matx_dense_linsolve_t* ls,
		const matx_dense_f64_t A, matx_vec_f64_t S, matx_dense_f64_t* U, matx_dense_f64_t* Vt);
	MATX_API matx_status_t matx_gesvd_c64(const matx_dense_linsolve_t* ls,
		const matx_dense_c64_t A, matx_vec_f64_t S, matx_dense_c64_t* U, matx_dense_c64_t* Vt);

#ifdef __cplusplus
}
#endif

#endif // MATX_DENSE_SOLVE_H