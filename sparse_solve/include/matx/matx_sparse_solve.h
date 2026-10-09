#ifndef MATX_SPARSE_SOLVE_H
#define MATX_SPARSE_SOLVE_H

#include "matx/matx_func.h"
#include "matx/matx_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum matx_linsolve_backend_kind_t {
    MATX_LINSOLVE_BACKEND_SUITESPARSE_KLU = 0,
    MATX_LINSOLVE_BACKEND_UMFPACK = 1,
    MATX_LINSOLVE_BACKEND_CXSPARSE = 2,
    MATX_LINSOLVE_BACKEND_SUPERLU = 3,
    MATX_LINSOLVE_BACKEND_MUMPS = 4
} matx_sparse_linsolve_backend_kind_t;

// Opaque factorization handles
typedef struct matx_factor_sparse_d_i8_t
{
    void* reserved;
    matx_alloc_t alloc;
} matx_factor_sparse_d_i8_t;

typedef struct matx_factor_sparse_z_i8_t
{
    void* reserved;
    matx_alloc_t alloc;
} matx_factor_sparse_z_i8_t;

typedef struct matx_sparse_linsolve_vtable_t
{
    // Sparse real
    matx_status_t (*factor_csc_d_i8)(const matx_alloc_t* alloc, matx_coo_d_i8_t A, matx_factor_sparse_d_i8_t* out_F);
    matx_status_t (*solve_csc_d_i8)(const matx_alloc_t* alloc, matx_factor_sparse_d_i8_t* F,
                                    const matx_double* b,
                                    matx_double* x);
    void (*factor_csc_d_i8_destroy)(const matx_alloc_t* alloc, matx_factor_sparse_d_i8_t* F);

    // Sparse complex
    matx_status_t (*factor_csc_z_i8)(const matx_alloc_t* alloc, matx_coo_z_i8_t A, matx_factor_sparse_z_i8_t* out_F);
    matx_status_t (*solve_csc_z_i8)(const matx_alloc_t* alloc, matx_factor_sparse_z_i8_t* F,
                                    const matx_vec_z_i8_t b,
                                    matx_vec_z_i8_t x);
    void (*factor_csc_z_i8_destroy)(const matx_alloc_t* alloc, matx_factor_sparse_z_i8_t* F);

    // Sparse Cholesky (real SPD)
    matx_status_t (*factor_chol_csc_d_i8)(const matx_alloc_t* alloc, matx_coo_d_i8_t A, matx_factor_sparse_d_i8_t* out_F);
    matx_status_t (*solve_chol_csc_d_i8)(const matx_alloc_t* alloc, matx_factor_sparse_d_i8_t* F,
                                         const matx_double* b,
                                         matx_double* x);
    void (*factor_chol_csc_d_i8_destroy)(const matx_alloc_t* alloc, matx_factor_sparse_d_i8_t* F);

    // Sparse Cholesky (complex HPD)
    matx_status_t (*factor_chol_csc_z_i8)(const matx_alloc_t* alloc, matx_coo_z_i8_t A, matx_factor_sparse_z_i8_t* out_F);
    matx_status_t (*solve_chol_csc_z_i8)(const matx_alloc_t* alloc, matx_factor_sparse_z_i8_t* F,
                                         const matx_vec_z_i8_t b,
                                         matx_vec_z_i8_t x);
    void (*factor_chol_csc_z_i8_destroy)(const matx_alloc_t* alloc, matx_factor_sparse_z_i8_t* F);

    // Numeric-only refactorization (reuse symbolic, new values)
    matx_status_t (*refactor_csc_d_i8)(const matx_alloc_t* alloc, matx_coo_d_i8_t A, matx_factor_sparse_d_i8_t* F);

} matx_sparse_linsolve_vtable_t;

typedef struct matx_sparse_linsolve_t
{
    matx_sparse_linsolve_backend_kind_t kind;
    matx_sparse_linsolve_vtable_t vt;
    matx_alloc_t alloc;
} matx_sparse_linsolve_t;

MATX_SPARSE_SOLVE_API matx_sparse_linsolve_t matx_sparse_linsolve_default(matx_alloc_t alloc);
MATX_SPARSE_SOLVE_API matx_sparse_linsolve_t matx_sparse_linsolve_by_type(matx_sparse_linsolve_backend_kind_t k, matx_alloc_t alloc);
MATX_SPARSE_SOLVE_API const char* matx_sparse_linsolve_backend_name(matx_sparse_linsolve_backend_kind_t k);

// High-level API (thin wrappers over vtable) -------------------------------

// ---- Sparse real LU ----

/**
	 * @brief LU factorization of a real sparse matrix in COO format (KLU)
	 * @formula P * A * Q = L * U
	 *          where P and Q are permutation matrices, L is lower triangular,
	 *          U is upper triangular. A is sparse (COO), converted to CSC internally.
	 */
MATX_SPARSE_SOLVE_API matx_status_t matx_factor_csc_d_i8(const matx_sparse_linsolve_t* ls,
                                            matx_coo_d_i8_t A,
                                            matx_factor_sparse_d_i8_t* out_F);

/**
	 * @brief Solve a real sparse linear system using pre-computed LU factorization
	 * @formula A * x = b  =>  x = A^{-1} * b
	 *          Uses the LU factorization from matx_factor_csc_d_i8.
	 */
MATX_SPARSE_SOLVE_API matx_status_t matx_solve_csc_d_i8_factor(const matx_sparse_linsolve_t* ls,
                                                  matx_factor_sparse_d_i8_t* F,
                                                  const matx_double* b,
                                                  matx_double* x);

/**
	 * @brief Solve a real sparse linear system directly (factor + solve)
	 * @formula A * x = b  =>  x = A^{-1} * b
	 *          Factorizes A internally, solves for x, and discards the factorization.
	 */
MATX_SPARSE_SOLVE_API matx_status_t matx_solve_csc_d_i8(const matx_sparse_linsolve_t* ls,
                                           matx_coo_d_i8_t A,
                                           const matx_double* b,
                                           matx_double* x);

MATX_SPARSE_SOLVE_API void matx_factor_csc_d_i8_destroy(const matx_sparse_linsolve_t* ls,
                                           matx_factor_sparse_d_i8_t* F);

// ---- Sparse complex LU ----

/**
	 * @brief LU factorization of a complex sparse matrix in COO format (KLU)
	 * @formula P * A * Q = L * U
	 *          where P and Q are permutation matrices, L is lower triangular,
	 *          U is upper triangular. A is sparse (COO), converted to CSC internally.
	 */
MATX_SPARSE_SOLVE_API matx_status_t matx_factor_csc_z_i8(const matx_sparse_linsolve_t* ls,
                                            matx_coo_z_i8_t A,
                                            matx_factor_sparse_z_i8_t* out_F);

/**
	 * @brief Solve a complex sparse linear system using pre-computed LU factorization
	 * @formula A * x = b  =>  x = A^{-1} * b
	 */
MATX_SPARSE_SOLVE_API matx_status_t matx_solve_csc_z_i8_factor(const matx_sparse_linsolve_t* ls,
                                                  matx_factor_sparse_z_i8_t* F,
                                                  const matx_vec_z_i8_t b,
                                                  matx_vec_z_i8_t x);

/**
	 * @brief Solve a complex sparse linear system directly (factor + solve)
	 * @formula A * x = b  =>  x = A^{-1} * b
	 */
MATX_SPARSE_SOLVE_API matx_status_t matx_solve_csc_z_i8(const matx_sparse_linsolve_t* ls,
                                           matx_coo_z_i8_t A,
                                           const matx_vec_z_i8_t b,
                                           matx_vec_z_i8_t x);

MATX_SPARSE_SOLVE_API void matx_factor_csc_z_i8_destroy(const matx_sparse_linsolve_t* ls,
                                           matx_factor_sparse_z_i8_t* F);

// ---- Sparse Cholesky (real SPD) ----

/**
	 * @brief Cholesky factorization of a real sparse SPD matrix in COO format
	 * @formula A = L * L^T
	 *          where L is lower triangular. A must be symmetric positive-definite.
	 *          Internally converts COO to CSC and uses CHOLMOD.
	 */
MATX_SPARSE_SOLVE_API matx_status_t matx_factor_chol_coo_d_i8(const matx_sparse_linsolve_t* ls,
                                                 matx_coo_d_i8_t A,
                                                 matx_factor_sparse_d_i8_t* out_F);

/**
	 * @brief Solve a real sparse SPD system using pre-computed Cholesky factorization
	 * @formula A * x = b  =>  x = A^{-1} * b
	 */
MATX_SPARSE_SOLVE_API matx_status_t matx_solve_chol_coo_d_i8_factor(const matx_sparse_linsolve_t* ls,
                                                       matx_factor_sparse_d_i8_t* F,
                                                       const matx_double* b,
                                                       matx_double* x);

MATX_SPARSE_SOLVE_API void matx_factor_chol_coo_d_i8_destroy(const matx_sparse_linsolve_t* ls,
                                                matx_factor_sparse_d_i8_t* F);

/**
	 * @brief Solve a real sparse SPD system directly via Cholesky (factor + solve)
	 * @formula A * x = b  =>  x = A^{-1} * b
	 */
MATX_SPARSE_SOLVE_API matx_status_t matx_solve_chol_coo_d_i8(const matx_sparse_linsolve_t* ls,
                                                matx_coo_d_i8_t A,
                                                const matx_double* b,
                                                matx_double* x);

// ---- Sparse Cholesky (complex HPD) ----

/**
 * @brief Cholesky factorization of a complex sparse HPD matrix in COO format
 * @formula A = L * L^H
 *          where L is lower triangular. A must be Hermitian positive-definite.
 */
MATX_SPARSE_SOLVE_API matx_status_t matx_factor_chol_coo_z_i8(const matx_sparse_linsolve_t* ls,
                                                 matx_coo_z_i8_t A,
                                                 matx_factor_sparse_z_i8_t* out_F);

/**
 * @brief Solve a complex sparse HPD system using pre-computed Cholesky factorization
 */
MATX_SPARSE_SOLVE_API matx_status_t matx_solve_chol_coo_z_i8_factor(const matx_sparse_linsolve_t* ls,
                                                       matx_factor_sparse_z_i8_t* F,
                                                       const matx_vec_z_i8_t b,
                                                       matx_vec_z_i8_t x);

MATX_SPARSE_SOLVE_API void matx_factor_chol_coo_z_i8_destroy(const matx_sparse_linsolve_t* ls,
                                                matx_factor_sparse_z_i8_t* F);

// ---- Numeric-only refactorization ----

/**
 * @brief Refactorize a real sparse matrix with new numeric values but same sparsity pattern
 * @formula Reuses the symbolic factorization from a previous call to matx_factor_csc_d_i8,
 *          performing only numeric factorization with updated matrix values.
 */
MATX_SPARSE_SOLVE_API matx_status_t matx_refactor_csc_d_i8(const matx_sparse_linsolve_t* ls,
                                              matx_coo_d_i8_t A,
                                              matx_factor_sparse_d_i8_t* F);

// ---- COO-to-CSC conversion ----

/**
	 * @brief Convert a real COO sparse matrix to CSC format
	 * @formula CSC(col_ptr, row_ind, values) <- COO(rows, cols, values)
	 *          Compresses duplicate entries by summing their values.
	 */
MATX_SPARSE_SOLVE_API matx_status_t coo_to_csc_d_i8(matx_coo_d_i8_t coo);

/**
	 * @brief Convert a complex COO sparse matrix to CSC format
	 * @formula CSC(col_ptr, row_ind, values) <- COO(rows, cols, values)
	 *          Compresses duplicate entries by summing their values.
	 */
MATX_SPARSE_SOLVE_API matx_status_t coo_to_csc_z_i8(matx_coo_z_i8_t coo);

/**
	 * @brief Convert a complex COO to CSC with value remapping
	 * @formula CSC(col_ptr, row_ind, remapped_values) <- COO(rows, cols, values)
	 *          Applies a value remapping function during conversion.
	 */
MATX_SPARSE_SOLVE_API matx_status_t coo_to_csc_z_i8_value_remap(matx_coo_z_i8_t coo);

/**
	 * @brief Convert a real COO to CSC with value remapping
	 * @formula CSC(col_ptr, row_ind, remapped_values) <- COO(rows, cols, values)
	 *          Applies a value remapping function during conversion.
	 */
MATX_SPARSE_SOLVE_API matx_status_t coo_to_csc_d_i8_value_remap(matx_coo_d_i8_t coo);

#ifdef __cplusplus
}
#endif
#endif
