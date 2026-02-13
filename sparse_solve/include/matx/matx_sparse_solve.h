#pragma once

#include "matx/matx.h"

#ifdef __cplusplus
extern "C" {
#endif

	typedef enum matx_linsolve_backend_kind_t {
		MATX_LINSOLVE_BACKEND_SUITESPARSE = 0
	} matx_sparse_linsolve_backend_kind_t;

	// Opaque factorization handles
	typedef struct matx_factor_sparse_f64_t matx_factor_sparse_f64_t;
	typedef struct matx_factor_sparse_f64_t matx_factor_sparse_c64_t;

	typedef struct matx_sparse_linsolve_vtable_t {
		// Sparse real
		matx_status_t(*factor_csc_f64)(const matx_csc_f64_t* A,
			matx_factor_sparse_f64_t** out_F);
		matx_status_t(*solve_csc_f64)(const matx_factor_sparse_f64_t* F,
			const double* b,
			double* x);
		void (*factor_csc_f64_destroy)(matx_factor_sparse_f64_t* F);

		// Sparse complex
		matx_status_t(*factor_csc_c64)(const matx_csc_c64_t* A,
			matx_factor_sparse_c64_t** out_F);
		matx_status_t(*solve_csc_c64)(const matx_factor_sparse_c64_t* F,
			const matx_vec_c64_t* b,
			matx_vec_c64_t* x);
		void (*factor_csc_c64_destroy)(matx_factor_sparse_c64_t* F);

	} matx_sparse_linsolve_vtable_t;

	typedef struct matx_sparse_linsolve_t {
		matx_sparse_linsolve_backend_kind_t kind;
		matx_sparse_linsolve_vtable_t  vt;
	} matx_sparse_linsolve_t;

	matx_sparse_linsolve_t matx_sparse_linsolve_default(void);
	const char* matx_sparse_linsolve_backend_name(matx_sparse_linsolve_backend_kind_t k);

	// High-level API (thin wrappers over vtable) -------------------------------

	// Sparse real
	matx_status_t matx_factor_csc_f64(const matx_sparse_linsolve_t* ls,
		const matx_csc_f64_t* A,
		matx_factor_sparse_f64_t** out_F);

	matx_status_t matx_solve_csc_f64_factor(const matx_sparse_linsolve_t* ls,
		const matx_factor_sparse_f64_t* F,
		const matx_double* b,
		matx_double* x);

	matx_status_t matx_solve_csc_f64(const matx_sparse_linsolve_t* ls,
		const matx_csc_f64_t* A,
		const matx_double* b,
		matx_double* x);

	void matx_factor_csc_f64_destroy(const matx_sparse_linsolve_t* ls,
		matx_factor_sparse_f64_t* F);

	// Complex sparse/dense: API placeholders (impl may return NOT_SUPPORTED)
	matx_status_t matx_factor_csc_c64(const matx_sparse_linsolve_t* ls,
		const matx_csc_c64_t* A,
		matx_factor_sparse_c64_t** out_F);

	matx_status_t matx_solve_csc_c64_factor(const matx_sparse_linsolve_t* ls,
		const matx_factor_sparse_c64_t* F,
		const matx_vec_c64_t* b,
		matx_vec_c64_t* x);

	matx_status_t matx_solve_csc_c64(const matx_sparse_linsolve_t* ls,
		const matx_csc_c64_t* A,
		const matx_vec_c64_t* b,
		matx_vec_c64_t* x);

	void matx_factor_csc_c64_destroy(const matx_sparse_linsolve_t* ls,
		matx_factor_sparse_c64_t* F);

#ifdef __cplusplus
}
#endif

