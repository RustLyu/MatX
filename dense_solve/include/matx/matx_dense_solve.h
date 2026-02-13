#pragma once

#include "matx/matx.h"

#ifdef __cplusplus
extern "C" {
#endif

	typedef enum matx_dense_linsolve_backend_kind_t {
		MATX_LINSOLVE_BACKEND_CBLAS = 0
	} matx_dense_linsolve_backend_kind_t;

	// Opaque factorization handles
	typedef struct matx_factor_dense_f64_t matx_factor_dense_f64_t;
	typedef struct matx_factor_dense_f64_t matx_factor_dense_c64_t;

	typedef struct matx_dense_linsolve_vtable_t {
		// Dense real
		matx_status_t(*factor_dense_f64)(const matx_dense_f64_t* A,
			matx_factor_dense_f64_t** out_F);
		matx_status_t(*solve_dense_f64)(const matx_factor_dense_f64_t* F,
			const double* b,
			double* x);
		void (*factor_dense_f64_destroy)(matx_factor_dense_f64_t* F);


		// Dense complex
		matx_status_t(*factor_dense_c64)(const matx_dense_c64_t* A,
			matx_factor_dense_c64_t** out_F);
		matx_status_t(*solve_dense_c64)(const matx_factor_dense_c64_t* F,
			const matx_vec_c64_t* b,
			matx_vec_c64_t* x);
		void (*factor_dense_c64_destroy)(matx_factor_dense_c64_t* F);
	} matx_dense_linsolve_vtable_t;

	typedef struct matx_dense_linsolve_t {
		matx_dense_linsolve_backend_kind_t kind;
		matx_dense_linsolve_vtable_t vt;
	} matx_dense_linsolve_t;

	matx_dense_linsolve_t matx_dense_linsolve_default(void);
	const char* matx_dense_linsolve_backend_name(matx_dense_linsolve_backend_kind_t k);

	// High-level API (thin wrappers over vtable) -------------------------------

	// Dense real
	matx_status_t matx_factor_dense_f64(const matx_dense_linsolve_t* ls,
		const matx_dense_f64_t* A,
		matx_factor_dense_f64_t** out_F);


	matx_status_t matx_solve_dense_f64_factor(const matx_dense_linsolve_t* ls,
		const matx_factor_dense_f64_t* F,
		const matx_double* b,
		matx_double* x);

	void matx_factor_dense_f64_destroy(const matx_dense_linsolve_t* ls,
		matx_factor_dense_f64_t* F);

	matx_status_t matx_solve_dense_f64(const matx_dense_linsolve_t* ls,
		const matx_dense_f64_t* A,
		const matx_double* b,
		matx_double* x);


	matx_status_t matx_factor_dense_c64(const matx_dense_linsolve_t* ls,
		const matx_dense_c64_t* A,
		matx_factor_dense_c64_t** out_F);

	matx_status_t matx_solve_dense_c64_factor(const matx_dense_linsolve_t* ls,
		const matx_factor_dense_c64_t* F,
		const matx_vec_c64_t* b,
		matx_vec_c64_t* x);

	void matx_factor_dense_c64_destroy(const matx_dense_linsolve_t* ls,
		matx_factor_dense_c64_t* F);

	matx_status_t matx_solve_dense_c64(const matx_dense_linsolve_t* ls,
		const matx_dense_c64_t* A,
		const matx_vec_c64_t* b,
		matx_vec_c64_t* x);

#ifdef __cplusplus
}
#endif

