#ifndef MATX_DENSE_SOLVE_H
#define MATX_DENSE_SOLVE_H

#include "matx/matx_func.h"
#include "matx/matx_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum matx_dense_linsolve_backend_kind_t {
    MATX_LINSOLVE_BACKEND_CBLAS = 0
} matx_dense_linsolve_backend_kind_t;

// Opaque factorization handles
typedef struct matx_factor_dense_d_i8_t matx_factor_dense_d_i8_t;
typedef struct matx_factor_dense_z_i8_t matx_factor_dense_z_i8_t;

typedef struct matx_dense_linsolve_vtable_t
{
    // Dense real LU
    matx_status_t (*factor_dense_d_i8)(const matx_alloc_t* alloc, const matx_dense_d_i8_t A, matx_factor_dense_d_i8_t** out_F);
    matx_status_t (*solve_dense_d_i8)(const matx_alloc_t* alloc, const matx_factor_dense_d_i8_t* F, const double* b, double* x);
    void (*factor_dense_d_i8_destroy)(const matx_alloc_t* alloc, matx_factor_dense_d_i8_t* F);

    // Dense complex LU
    matx_status_t (*factor_dense_z_i8)(const matx_alloc_t* alloc, const matx_dense_z_i8_t A, matx_factor_dense_z_i8_t** out_F);
    matx_status_t (*solve_dense_z_i8)(const matx_alloc_t* alloc, const matx_factor_dense_z_i8_t* F,
                                      const matx_vec_z_i8_t b,
                                      matx_vec_z_i8_t x);
    void (*factor_dense_z_i8_destroy)(const matx_alloc_t* alloc, matx_factor_dense_z_i8_t* F);

    // Cholesky
    matx_status_t (*potrf_d_i8)(const matx_alloc_t* alloc, const matx_dense_d_i8_t A,
                                matx_uplo_t uplo,
                                matx_factor_dense_d_i8_t** out_F);
    matx_status_t (*potrs_d_i8)(const matx_alloc_t* alloc, const matx_factor_dense_d_i8_t* F,
                                const matx_double* b,
                                matx_double* x);
    matx_status_t (*potrf_z_i8)(const matx_alloc_t* alloc, const matx_dense_z_i8_t A,
                                matx_uplo_t uplo,
                                matx_factor_dense_z_i8_t** out_F);
    matx_status_t (*potrs_z_i8)(const matx_alloc_t* alloc, const matx_factor_dense_z_i8_t* F,
                                const matx_vec_z_i8_t b,
                                matx_vec_z_i8_t x);

    // Least squares (GELS)
    matx_status_t (*gels_d_i8)(const matx_alloc_t* alloc, const matx_dense_d_i8_t A, const matx_double* b, matx_double* x);
    matx_status_t (*gels_z_i8)(const matx_alloc_t* alloc, const matx_dense_z_i8_t A,
                               const matx_vec_z_i8_t b,
                               matx_vec_z_i8_t x);

    // Symmetric eigenvalue (real only)
    matx_status_t (*syev_d_i8)(const matx_alloc_t* alloc, const matx_dense_d_i8_t A,
                               matx_vec_d_i8_t eigenvalues,
                               matx_dense_d_i8_t* eigenvectors);

    // Hermitian eigenvalue (complex)
    matx_status_t (*syev_z_i8)(const matx_alloc_t* alloc, const matx_dense_z_i8_t A,
                               matx_vec_d_i8_t eigenvalues,
                               matx_dense_z_i8_t* eigenvectors);

    // General non-symmetric eigenvalues
    matx_status_t (*geev_d_i8)(const matx_alloc_t* alloc, const matx_dense_d_i8_t A,
                               matx_vec_z_i8_t eigenvalues,
                               matx_dense_d_i8_t* vr,
                               matx_dense_d_i8_t* vl);
    matx_status_t (*geev_z_i8)(const matx_alloc_t* alloc, const matx_dense_z_i8_t A,
                               matx_vec_z_i8_t eigenvalues,
                               matx_dense_z_i8_t* vr,
                               matx_dense_z_i8_t* vl);

    // QR factorization
    matx_status_t (*qr_d_i8)(const matx_alloc_t* alloc, const matx_dense_d_i8_t A, matx_dense_d_i8_t* Q, matx_dense_d_i8_t* R);
    matx_status_t (*qr_z_i8)(const matx_alloc_t* alloc, const matx_dense_z_i8_t A, matx_dense_z_i8_t* Q, matx_dense_z_i8_t* R);

    // Determinant
    matx_status_t (*det_dense_d_i8)(const matx_alloc_t* alloc, const matx_dense_d_i8_t A, matx_double* det);
    matx_status_t (*det_dense_z_i8)(const matx_alloc_t* alloc, const matx_dense_z_i8_t A, matx_complex_d_t* det);

    // Condition number (1-norm)
    matx_status_t (*cond_dense_d_i8)(const matx_alloc_t* alloc, const matx_dense_d_i8_t A, matx_double* cond);
    matx_status_t (*cond_dense_z_i8)(const matx_alloc_t* alloc, const matx_dense_z_i8_t A, matx_double* cond);

    // SVD
    matx_status_t (*gesvd_d_i8)(const matx_alloc_t* alloc, const matx_dense_d_i8_t A,
                                matx_vec_d_i8_t S,
                                matx_dense_d_i8_t* U,
                                matx_dense_d_i8_t* Vt);
    matx_status_t (*gesvd_z_i8)(const matx_alloc_t* alloc, const matx_dense_z_i8_t A,
                                matx_vec_d_i8_t S,
                                matx_dense_z_i8_t* U,
                                matx_dense_z_i8_t* Vt);

    // Multi-RHS dense solve
    matx_status_t (*solve_dense_mrhs_d_i8)(const matx_alloc_t* alloc, const matx_factor_dense_d_i8_t* F,
                                          const matx_dense_d_i8_t B, matx_dense_d_i8_t* X);
    matx_status_t (*solve_dense_mrhs_z_i8)(const matx_alloc_t* alloc, const matx_factor_dense_z_i8_t* F,
                                          const matx_dense_z_i8_t B, matx_dense_z_i8_t* X);
    matx_status_t (*potrs_mrhs_d_i8)(const matx_alloc_t* alloc, const matx_factor_dense_d_i8_t* F,
                                    const matx_dense_d_i8_t B, matx_dense_d_i8_t* X);
    matx_status_t (*potrs_mrhs_z_i8)(const matx_alloc_t* alloc, const matx_factor_dense_z_i8_t* F,
                                    const matx_dense_z_i8_t B, matx_dense_z_i8_t* X);

    // LDL^T (symmetric indefinite) factorization
    matx_status_t (*sytrf_d_i8)(const matx_alloc_t* alloc, const matx_dense_d_i8_t A, matx_uplo_t uplo,
                               matx_factor_dense_d_i8_t** out_F);
    matx_status_t (*sytrs_d_i8)(const matx_alloc_t* alloc, const matx_factor_dense_d_i8_t* F,
                               const matx_double* b, matx_double* x);
    void (*sytrf_destroy_d)(const matx_alloc_t* alloc, matx_factor_dense_d_i8_t* F);
    matx_status_t (*sytrf_z_i8)(const matx_alloc_t* alloc, const matx_dense_z_i8_t A, matx_uplo_t uplo,
                               matx_factor_dense_z_i8_t** out_F);
    matx_status_t (*sytrs_z_i8)(const matx_alloc_t* alloc, const matx_factor_dense_z_i8_t* F,
                               const matx_vec_z_i8_t b, matx_vec_z_i8_t x);
    void (*sytrf_destroy_z)(const matx_alloc_t* alloc, matx_factor_dense_z_i8_t* F);

    // QR with column pivoting (rank-revealing)
    matx_status_t (*qrp_d_i8)(const matx_alloc_t* alloc, const matx_dense_d_i8_t A,
                             matx_dense_d_i8_t* Q, matx_dense_d_i8_t* R, matx_vec_d_i8_t* jpvt);
    matx_status_t (*qrp_z_i8)(const matx_alloc_t* alloc, const matx_dense_z_i8_t A,
                             matx_dense_z_i8_t* Q, matx_dense_z_i8_t* R, matx_vec_d_i8_t* jpvt);

    // Pseudo-inverse (via SVD)
    matx_status_t (*pinv_d_i8)(const matx_alloc_t* alloc, const matx_dense_d_i8_t A, matx_double rcond,
                              matx_dense_d_i8_t* out);
    matx_status_t (*pinv_z_i8)(const matx_alloc_t* alloc, const matx_dense_z_i8_t A, matx_double rcond,
                              matx_dense_z_i8_t* out);

    // Matrix rank (via SVD)
    matx_status_t (*rank_d_i8)(const matx_alloc_t* alloc, const matx_dense_d_i8_t A, matx_double tol,
                              matx_int64_t* rank);
    matx_status_t (*rank_z_i8)(const matx_alloc_t* alloc, const matx_dense_z_i8_t A, matx_double tol,
                              matx_int64_t* rank);

    // Generalized symmetric eigenvalue (SYGV/HEGV)
    matx_status_t (*sygv_d_i8)(const matx_alloc_t* alloc, const matx_dense_d_i8_t A, const matx_dense_d_i8_t B,
                              matx_vec_d_i8_t eigenvalues, matx_dense_d_i8_t* eigenvectors);
    matx_status_t (*sygv_z_i8)(const matx_alloc_t* alloc, const matx_dense_z_i8_t A, const matx_dense_z_i8_t B,
                              matx_vec_d_i8_t eigenvalues, matx_dense_z_i8_t* eigenvectors);

    // LQ factorization
    matx_status_t (*lq_d_i8)(const matx_alloc_t* alloc, const matx_dense_d_i8_t A,
                            matx_dense_d_i8_t* L, matx_dense_d_i8_t* Q);
    matx_status_t (*lq_z_i8)(const matx_alloc_t* alloc, const matx_dense_z_i8_t A,
                            matx_dense_z_i8_t* L, matx_dense_z_i8_t* Q);
} matx_dense_linsolve_vtable_t;

typedef struct matx_dense_linsolve_t
{
    matx_dense_linsolve_backend_kind_t kind;
    matx_dense_linsolve_vtable_t vt;
    matx_alloc_t alloc;
} matx_dense_linsolve_t;

MATX_DENSE_SOLVE_API matx_dense_linsolve_t matx_dense_linsolve_default(matx_alloc_t alloc);
MATX_DENSE_SOLVE_API const char* matx_dense_linsolve_backend_name(matx_dense_linsolve_backend_kind_t k);

// High-level API (thin wrappers over vtable) -------------------------------

// ---- Dense real LU ----

/**
	 * @brief LU factorization of a real dense matrix with partial pivoting (DGETRF)
	 * @formula P * A = L * U
	 *          where P is a permutation matrix, L is lower triangular with unit diagonal,
	 *          U is upper triangular.
	 */
MATX_DENSE_SOLVE_API matx_status_t matx_factor_dense_d_i8(const matx_dense_linsolve_t* ls,
                                              const matx_dense_d_i8_t A,
                                              matx_factor_dense_d_i8_t** out_F);

/**
	 * @brief Solve a real linear system using pre-computed LU factorization (DGETRS)
	 * @formula A * x = b  =>  x = A^{-1} * b
	 */
MATX_DENSE_SOLVE_API matx_status_t matx_solve_dense_d_i8_factor(const matx_dense_linsolve_t* ls,
                                                    const matx_factor_dense_d_i8_t* F,
                                                    const matx_double* b,
                                                    matx_double* x);

MATX_DENSE_SOLVE_API void matx_factor_dense_d_i8_destroy(const matx_dense_linsolve_t* ls,
                                             matx_factor_dense_d_i8_t* F);

/**
	 * @brief Solve a real linear system directly via LU factorization (DGETRF + DGETRS)
	 * @formula A * x = b  =>  x = A^{-1} * b
	 *          Factorizes A internally, solves for x, and discards the factorization.
	 */
MATX_DENSE_SOLVE_API matx_status_t matx_solve_dense_d_i8(const matx_dense_linsolve_t* ls,
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
MATX_DENSE_SOLVE_API matx_status_t matx_factor_dense_z_i8(const matx_dense_linsolve_t* ls,
                                              const matx_dense_z_i8_t A,
                                              matx_factor_dense_z_i8_t** out_F);

/**
	 * @brief Solve a complex linear system using pre-computed LU factorization (ZGETRS)
	 * @formula A * x = b  =>  x = A^{-1} * b
	 */
MATX_DENSE_SOLVE_API matx_status_t matx_solve_dense_z_i8_factor(const matx_dense_linsolve_t* ls,
                                                    const matx_factor_dense_z_i8_t* F,
                                                    const matx_vec_z_i8_t b,
                                                    matx_vec_z_i8_t x);

MATX_DENSE_SOLVE_API void matx_factor_dense_z_i8_destroy(const matx_dense_linsolve_t* ls,
                                             matx_factor_dense_z_i8_t* F);

/**
	 * @brief Solve a complex linear system directly via LU factorization (ZGETRF + ZGETRS)
	 * @formula A * x = b  =>  x = A^{-1} * b
	 *          Factorizes A internally, solves for x, and discards the factorization.
	 */
MATX_DENSE_SOLVE_API matx_status_t matx_solve_dense_z_i8(const matx_dense_linsolve_t* ls,
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
MATX_DENSE_SOLVE_API matx_status_t matx_factor_chol_d_i8(const matx_dense_linsolve_t* ls,
                                             const matx_dense_d_i8_t A,
                                             matx_uplo_t uplo,
                                             matx_factor_dense_d_i8_t** out_F);

/**
	 * @brief Solve a real SPD system using Cholesky factorization (DPOTRS)
	 * @formula A * x = b  =>  x = A^{-1} * b
	 */
MATX_DENSE_SOLVE_API matx_status_t matx_solve_chol_d_i8(const matx_dense_linsolve_t* ls,
                                            const matx_factor_dense_d_i8_t* F,
                                            const matx_double* b,
                                            matx_double* x);

/**
	 * @brief Solve a real SPD system in one shot (DPOTRF + DPOTRS)
	 * @formula A * x = b  =>  x = A^{-1} * b
	 */
MATX_DENSE_SOLVE_API matx_status_t matx_solve_chol_d_i8_oneshot(const matx_dense_linsolve_t* ls,
                                                    const matx_dense_d_i8_t A,
                                                    matx_uplo_t uplo,
                                                    const matx_double* b,
                                                    matx_double* x);

/**
	 * @brief Cholesky factorization of a complex HPD matrix (ZPOTRF)
	 * @formula A = L * L^H  (uplo=L)  or  A = U^H * U  (uplo=U)
	 *          where L is lower triangular, U is upper triangular.
	 *          A must be Hermitian positive-definite.
	 */
MATX_DENSE_SOLVE_API matx_status_t matx_factor_chol_z_i8(const matx_dense_linsolve_t* ls,
                                             const matx_dense_z_i8_t A,
                                             matx_uplo_t uplo,
                                             matx_factor_dense_z_i8_t** out_F);

/**
	 * @brief Solve a complex HPD system using Cholesky factorization (ZPOTRS)
	 * @formula A * x = b  =>  x = A^{-1} * b
	 */
MATX_DENSE_SOLVE_API matx_status_t matx_solve_chol_z_i8(const matx_dense_linsolve_t* ls,
                                            const matx_factor_dense_z_i8_t* F,
                                            const matx_vec_z_i8_t b,
                                            matx_vec_z_i8_t x);

// ---- Least squares (GELS) ----

/**
	 * @brief Solve a real linear least-squares problem via QR/LQ (DGELS)
	 * @formula min ||b - A * x||_2  (overdetermined, m >= n)
	 *          or  min ||x||_2 s.t. A * x = b  (underdetermined, m < n)
	 */
MATX_DENSE_SOLVE_API matx_status_t matx_gels_d_i8(const matx_dense_linsolve_t* ls,
                                      const matx_dense_d_i8_t A,
                                      const matx_double* b,
                                      matx_double* x);

/**
	 * @brief Solve a complex linear least-squares problem via QR/LQ (ZGELS)
	 * @formula min ||b - A * x||_2  (overdetermined, m >= n)
	 *          or  min ||x||_2 s.t. A * x = b  (underdetermined, m < n)
	 */
MATX_DENSE_SOLVE_API matx_status_t matx_gels_z_i8(const matx_dense_linsolve_t* ls,
                                      const matx_dense_z_i8_t A,
                                      const matx_vec_z_i8_t b,
                                      matx_vec_z_i8_t x);

// ---- Symmetric eigenvalue (SYEV) ----

/**
	 * @brief Eigenvalue decomposition of a real symmetric matrix (DSYEV)
	 * @formula A = Q * diag(W) * Q^T
	 *          where Q is orthogonal, W is the vector of eigenvalues in ascending order.
	 *          eigenvalues must be pre-allocated (size n); eigenvectors allocated by callee (or NULL to skip)
	 */
MATX_DENSE_SOLVE_API matx_status_t matx_syev_d_i8(const matx_dense_linsolve_t* ls,
                                      const matx_dense_d_i8_t A,
                                      matx_vec_d_i8_t eigenvalues,
                                      matx_dense_d_i8_t* eigenvectors);

// ---- SVD (GESVD) ----

/**
	 * @brief Singular value decomposition of a real matrix (DGESVD)
	 * @formula A = U * diag(S) * V^T
	 *          where U is m-by-min(m,n), S contains singular values in descending order,
	 *          V^T is min(m,n)-by-n. A is m-by-n, overwritten.
	 *          S must be pre-allocated (size min(m,n)); U and Vt allocated by callee (or NULL to skip)
	 */
MATX_DENSE_SOLVE_API matx_status_t matx_gesvd_d_i8(const matx_dense_linsolve_t* ls,
                                       const matx_dense_d_i8_t A,
                                       matx_vec_d_i8_t S,
                                       matx_dense_d_i8_t* U,
                                       matx_dense_d_i8_t* Vt);

/**
	 * @brief Singular value decomposition of a complex matrix (ZGESVD)
	 * @formula A = U * diag(S) * V^H
	 *          where U is m-by-min(m,n), S contains singular values in descending order,
	 *          V^H is min(m,n)-by-n. A is m-by-n, overwritten.
	 *          S must be pre-allocated (size min(m,n)); U and Vt allocated by callee (or NULL to skip)
	 */
MATX_DENSE_SOLVE_API matx_status_t matx_gesvd_z_i8(const matx_dense_linsolve_t* ls,
                                       const matx_dense_z_i8_t A,
                                       matx_vec_d_i8_t S,
                                       matx_dense_z_i8_t* U,
                                       matx_dense_z_i8_t* Vt);

// ---- Hermitian eigenvalue (complex) ----

MATX_DENSE_SOLVE_API matx_status_t matx_syev_z_i8(const matx_dense_linsolve_t* ls,
                                      const matx_dense_z_i8_t A,
                                      matx_vec_d_i8_t eigenvalues,
                                      matx_dense_z_i8_t* eigenvectors);

// ---- General eigenvalues ----

MATX_DENSE_SOLVE_API matx_status_t matx_geev_d_i8(const matx_dense_linsolve_t* ls,
                                      const matx_dense_d_i8_t A,
                                      matx_vec_z_i8_t eigenvalues,
                                      matx_dense_d_i8_t* eigenvectors_right,
                                      matx_dense_d_i8_t* eigenvectors_left);
MATX_DENSE_SOLVE_API matx_status_t matx_geev_z_i8(const matx_dense_linsolve_t* ls,
                                      const matx_dense_z_i8_t A,
                                      matx_vec_z_i8_t eigenvalues,
                                      matx_dense_z_i8_t* eigenvectors_right,
                                      matx_dense_z_i8_t* eigenvectors_left);

// ---- QR factorization ----

MATX_DENSE_SOLVE_API matx_status_t matx_qr_d_i8(const matx_dense_linsolve_t* ls,
                                    const matx_dense_d_i8_t A,
                                    matx_dense_d_i8_t* Q,
                                    matx_dense_d_i8_t* R);
MATX_DENSE_SOLVE_API matx_status_t matx_qr_z_i8(const matx_dense_linsolve_t* ls,
                                    const matx_dense_z_i8_t A,
                                    matx_dense_z_i8_t* Q,
                                    matx_dense_z_i8_t* R);

// ---- Determinant ----

MATX_DENSE_SOLVE_API matx_status_t matx_det_dense_d_i8(const matx_dense_linsolve_t* ls,
                                           const matx_dense_d_i8_t A,
                                           matx_double* det);
MATX_DENSE_SOLVE_API matx_status_t matx_det_dense_z_i8(const matx_dense_linsolve_t* ls,
                                           const matx_dense_z_i8_t A,
                                           matx_complex_d_t* det);

// ---- Condition number ----

MATX_DENSE_SOLVE_API matx_status_t matx_cond_dense_d_i8(const matx_dense_linsolve_t* ls,
                                            const matx_dense_d_i8_t A,
                                            matx_double* cond);
MATX_DENSE_SOLVE_API matx_status_t matx_cond_dense_z_i8(const matx_dense_linsolve_t* ls,
                                            const matx_dense_z_i8_t A,
                                            matx_double* cond);

// ---- Multi-RHS solve (dense) ----

/**
 * @brief Solve dense real system with multiple RHS using pre-computed LU (DGETRS)
 * @formula A * X = B  =>  X = A^{-1} * B  where B is m-by-nrhs
 */
MATX_DENSE_SOLVE_API matx_status_t matx_solve_dense_d_i8_factor_mrhs(const matx_dense_linsolve_t* ls,
                                                       const matx_factor_dense_d_i8_t* F,
                                                       const matx_dense_d_i8_t B,
                                                       matx_dense_d_i8_t* X);

/**
 * @brief Solve dense complex system with multiple RHS using pre-computed LU (ZGETRS)
 */
MATX_DENSE_SOLVE_API matx_status_t matx_solve_dense_z_i8_factor_mrhs(const matx_dense_linsolve_t* ls,
                                                       const matx_factor_dense_z_i8_t* F,
                                                       const matx_dense_z_i8_t B,
                                                       matx_dense_z_i8_t* X);

// ---- Multi-RHS Cholesky solve ----

/**
 * @brief Solve SPD system with multiple RHS using Cholesky (DPOTRS)
 */
MATX_DENSE_SOLVE_API matx_status_t matx_solve_chol_d_i8_factor_mrhs(const matx_dense_linsolve_t* ls,
                                                       const matx_factor_dense_d_i8_t* F,
                                                       const matx_dense_d_i8_t B,
                                                       matx_dense_d_i8_t* X);

/**
 * @brief Solve HPD system with multiple RHS using Cholesky (ZPOTRS)
 */
MATX_DENSE_SOLVE_API matx_status_t matx_solve_chol_z_i8_factor_mrhs(const matx_dense_linsolve_t* ls,
                                                       const matx_factor_dense_z_i8_t* F,
                                                       const matx_dense_z_i8_t B,
                                                       matx_dense_z_i8_t* X);

// ---- LDL^T factorization ----

/**
 * @brief LDL^T factorization of a real symmetric indefinite matrix (DSYTRF)
 */
MATX_DENSE_SOLVE_API matx_status_t matx_factor_ldl_d_i8(const matx_dense_linsolve_t* ls,
                                           const matx_dense_d_i8_t A,
                                           matx_uplo_t uplo,
                                           matx_factor_dense_d_i8_t** out_F);

/**
 * @brief Solve a real symmetric indefinite system using LDL^T factorization (DSYTRS)
 */
MATX_DENSE_SOLVE_API matx_status_t matx_solve_ldl_d_i8(const matx_dense_linsolve_t* ls,
                                          const matx_factor_dense_d_i8_t* F,
                                          const matx_double* b,
                                          matx_double* x);

MATX_DENSE_SOLVE_API void matx_factor_ldl_d_i8_destroy(const matx_dense_linsolve_t* ls,
                                           matx_factor_dense_d_i8_t* F);

/**
 * @brief LDL^H factorization of a complex Hermitian indefinite matrix (ZHETRF)
 */
MATX_DENSE_SOLVE_API matx_status_t matx_factor_ldl_z_i8(const matx_dense_linsolve_t* ls,
                                           const matx_dense_z_i8_t A,
                                           matx_uplo_t uplo,
                                           matx_factor_dense_z_i8_t** out_F);

/**
 * @brief Solve a complex Hermitian indefinite system using LDL^H factorization (ZHETRS)
 */
MATX_DENSE_SOLVE_API matx_status_t matx_solve_ldl_z_i8(const matx_dense_linsolve_t* ls,
                                          const matx_factor_dense_z_i8_t* F,
                                          const matx_vec_z_i8_t b,
                                          matx_vec_z_i8_t x);

MATX_DENSE_SOLVE_API void matx_factor_ldl_z_i8_destroy(const matx_dense_linsolve_t* ls,
                                           matx_factor_dense_z_i8_t* F);

// ---- QR with column pivoting ----

/**
 * @brief QR factorization with column pivoting of a real matrix (DGEQP3)
 * @formula A * P = Q * R
 *          jpvt must be pre-allocated (size ncols); returned with column permutations.
 */
MATX_DENSE_SOLVE_API matx_status_t matx_qrp_d_i8(const matx_dense_linsolve_t* ls,
                                    const matx_dense_d_i8_t A,
                                    matx_dense_d_i8_t* Q,
                                    matx_dense_d_i8_t* R,
                                    matx_vec_d_i8_t* jpvt);

/**
 * @brief QR factorization with column pivoting of a complex matrix (ZGEQP3)
 */
MATX_DENSE_SOLVE_API matx_status_t matx_qrp_z_i8(const matx_dense_linsolve_t* ls,
                                    const matx_dense_z_i8_t A,
                                    matx_dense_z_i8_t* Q,
                                    matx_dense_z_i8_t* R,
                                    matx_vec_d_i8_t* jpvt);

// ---- Pseudo-inverse ----

/**
 * @brief Moore-Penrose pseudo-inverse of a real matrix (via SVD)
 * @formula Ainv = V * diag(1/s_i) * U^T  for s_i > rcond * max(s)
 */
MATX_DENSE_SOLVE_API matx_status_t matx_pinv_dense_d_i8(const matx_dense_linsolve_t* ls,
                                          const matx_dense_d_i8_t A,
                                          matx_double rcond,
                                          matx_dense_d_i8_t* out);

/**
 * @brief Moore-Penrose pseudo-inverse of a complex matrix (via SVD)
 */
MATX_DENSE_SOLVE_API matx_status_t matx_pinv_dense_z_i8(const matx_dense_linsolve_t* ls,
                                          const matx_dense_z_i8_t A,
                                          matx_double rcond,
                                          matx_dense_z_i8_t* out);

// ---- Matrix rank ----

/**
 * @brief Numerical rank of a real matrix (via SVD)
 * @formula rank = count(s_i > tol * max(s))
 */
MATX_DENSE_SOLVE_API matx_status_t matx_rank_dense_d_i8(const matx_dense_linsolve_t* ls,
                                          const matx_dense_d_i8_t A,
                                          matx_double tol,
                                          matx_int64_t* rank);

/**
 * @brief Numerical rank of a complex matrix (via SVD)
 */
MATX_DENSE_SOLVE_API matx_status_t matx_rank_dense_z_i8(const matx_dense_linsolve_t* ls,
                                          const matx_dense_z_i8_t A,
                                          matx_double tol,
                                          matx_int64_t* rank);

// ---- Generalized symmetric eigenvalue ----

/**
 * @brief Generalized symmetric eigenvalue problem (DSYGV)
 * @formula A * v = lambda * B * v
 *          A is symmetric, B is symmetric positive-definite.
 */
MATX_DENSE_SOLVE_API matx_status_t matx_sygv_d_i8(const matx_dense_linsolve_t* ls,
                                     const matx_dense_d_i8_t A,
                                     const matx_dense_d_i8_t B,
                                     matx_vec_d_i8_t eigenvalues,
                                     matx_dense_d_i8_t* eigenvectors);

/**
 * @brief Generalized Hermitian eigenvalue problem (ZHEGV)
 * @formula A * v = lambda * B * v
 *          A is Hermitian, B is Hermitian positive-definite.
 */
MATX_DENSE_SOLVE_API matx_status_t matx_sygv_z_i8(const matx_dense_linsolve_t* ls,
                                     const matx_dense_z_i8_t A,
                                     const matx_dense_z_i8_t B,
                                     matx_vec_d_i8_t eigenvalues,
                                     matx_dense_z_i8_t* eigenvectors);

// ---- LQ factorization ----

/**
 * @brief LQ factorization of a real matrix (DGELQF)
 * @formula A = L * Q  where L is lower triangular, Q is orthogonal.
 */
MATX_DENSE_SOLVE_API matx_status_t matx_lq_d_i8(const matx_dense_linsolve_t* ls,
                                   const matx_dense_d_i8_t A,
                                   matx_dense_d_i8_t* L,
                                   matx_dense_d_i8_t* Q);

/**
 * @brief LQ factorization of a complex matrix (ZGELQF)
 * @formula A = L * Q  where L is lower triangular, Q is unitary.
 */
MATX_DENSE_SOLVE_API matx_status_t matx_lq_z_i8(const matx_dense_linsolve_t* ls,
                                   const matx_dense_z_i8_t A,
                                   matx_dense_z_i8_t* L,
                                   matx_dense_z_i8_t* Q);

#ifdef __cplusplus
}
#endif

#endif // MATX_DENSE_SOLVE_H