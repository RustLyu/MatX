#ifndef MATX_SPARSE_SOLVE_H
#define MATX_SPARSE_SOLVE_H

#include "matx/matx_types.h"
#include "matx/matx_func.h"

#ifdef __cplusplus
extern "C" {
#endif

	typedef enum matx_linsolve_backend_kind_t {
		MATX_LINSOLVE_BACKEND_SUITESPARSE = 0
	} matx_sparse_linsolve_backend_kind_t;

	// Opaque factorization handles
	typedef struct matx_factor_sparse_f64_t matx_factor_sparse_f64_t;
	typedef struct matx_factor_sparse_c64_t matx_factor_sparse_c64_t;

	typedef struct matx_sparse_linsolve_vtable_t {
		// Sparse real
		matx_status_t(*factor_csc_f64)(matx_coo_f64_t A,
			matx_factor_sparse_f64_t** out_F);
		matx_status_t(*solve_csc_f64)(matx_factor_sparse_f64_t* F,
			const matx_double* b,
			matx_double* x);
		void (*factor_csc_f64_destroy)(matx_factor_sparse_f64_t* F);

		// Sparse complex
		matx_status_t(*factor_csc_c64)(matx_coo_c64_t A,
			matx_factor_sparse_c64_t** out_F);
		matx_status_t(*solve_csc_c64)(matx_factor_sparse_c64_t* F,
			const matx_vec_c64_t b,
			matx_vec_c64_t x);
		void (*factor_csc_c64_destroy)(matx_factor_sparse_c64_t* F);

	} matx_sparse_linsolve_vtable_t;

	typedef struct matx_sparse_linsolve_t {
		matx_sparse_linsolve_backend_kind_t kind;
		matx_sparse_linsolve_vtable_t  vt;
	} matx_sparse_linsolve_t;

	MATX_API matx_sparse_linsolve_t matx_sparse_linsolve_default(void);
	MATX_API const char* matx_sparse_linsolve_backend_name(matx_sparse_linsolve_backend_kind_t k);

	// High-level API (thin wrappers over vtable) -------------------------------

	// ---- Sparse real LU ----

	/**
	 * @brief LU factorization of a real sparse matrix in COO format (KLU)
	 * @formula P * A * Q = L * U
	 *          where P and Q are permutation matrices, L is lower triangular,
	 *          U is upper triangular. A is sparse (COO), converted to CSC internally.
	 */
	MATX_API matx_status_t matx_factor_csc_f64(const matx_sparse_linsolve_t* ls,
		matx_coo_f64_t A,
		matx_factor_sparse_f64_t** out_F);

	/**
	 * @brief Solve a real sparse linear system using pre-computed LU factorization
	 * @formula A * x = b  =>  x = A^{-1} * b
	 *          Uses the LU factorization from matx_factor_csc_f64.
	 */
	MATX_API matx_status_t matx_solve_csc_f64_factor(const matx_sparse_linsolve_t* ls,
		matx_factor_sparse_f64_t* F,
		const matx_double* b,
		matx_double* x);

	/**
	 * @brief Solve a real sparse linear system directly (factor + solve)
	 * @formula A * x = b  =>  x = A^{-1} * b
	 *          Factorizes A internally, solves for x, and discards the factorization.
	 */
	MATX_API matx_status_t matx_solve_csc_f64(const matx_sparse_linsolve_t* ls,
		matx_coo_f64_t A,
		const matx_double* b,
		matx_double* x);

	MATX_API void matx_factor_csc_f64_destroy(const matx_sparse_linsolve_t* ls,
		matx_factor_sparse_f64_t* F);

	// ---- Sparse complex LU ----

	/**
	 * @brief LU factorization of a complex sparse matrix in COO format (KLU)
	 * @formula P * A * Q = L * U
	 *          where P and Q are permutation matrices, L is lower triangular,
	 *          U is upper triangular. A is sparse (COO), converted to CSC internally.
	 */
	MATX_API matx_status_t matx_factor_csc_c64(const matx_sparse_linsolve_t* ls,
		matx_coo_c64_t A,
		matx_factor_sparse_c64_t** out_F);

	/**
	 * @brief Solve a complex sparse linear system using pre-computed LU factorization
	 * @formula A * x = b  =>  x = A^{-1} * b
	 */
	MATX_API matx_status_t matx_solve_csc_c64_factor(const matx_sparse_linsolve_t* ls,
		matx_factor_sparse_c64_t* F,
		const matx_vec_c64_t b,
		matx_vec_c64_t x);

	/**
	 * @brief Solve a complex sparse linear system directly (factor + solve)
	 * @formula A * x = b  =>  x = A^{-1} * b
	 */
	MATX_API matx_status_t matx_solve_csc_c64(const matx_sparse_linsolve_t* ls,
		matx_coo_c64_t A,
		const matx_vec_c64_t b,
		matx_vec_c64_t x);

	MATX_API void matx_factor_csc_c64_destroy(const matx_sparse_linsolve_t* ls,
		matx_factor_sparse_c64_t* F);

	// ---- COO-to-CSC conversion ----

	/**
	 * @brief Convert a real COO sparse matrix to CSC format
	 * @formula CSC(col_ptr, row_ind, values) <- COO(rows, cols, values)
	 *          Compresses duplicate entries by summing their values.
	 */
	MATX_API matx_status_t coo_to_csc_f64(matx_coo_f64_t coo);

	/**
	 * @brief Convert a complex COO sparse matrix to CSC format
	 * @formula CSC(col_ptr, row_ind, values) <- COO(rows, cols, values)
	 *          Compresses duplicate entries by summing their values.
	 */
	MATX_API matx_status_t coo_to_csc_c64(matx_coo_c64_t coo);

	/**
	 * @brief Convert a complex COO to CSC with value remapping
	 * @formula CSC(col_ptr, row_ind, remapped_values) <- COO(rows, cols, values)
	 *          Applies a value remapping function during conversion.
	 */
	MATX_API matx_status_t coo_to_csc_c64_value_remap(matx_coo_c64_t coo);

	/**
	 * @brief Convert a real COO to CSC with value remapping
	 * @formula CSC(col_ptr, row_ind, remapped_values) <- COO(rows, cols, values)
	 *          Applies a value remapping function during conversion.
	 */
	MATX_API matx_status_t coo_to_csc_f64_value_remap(matx_coo_f64_t coo);

#ifdef __cplusplus
}
#endif
#endif