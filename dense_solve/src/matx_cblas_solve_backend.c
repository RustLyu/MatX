#include "matx/matx_dense_solve.h"
#include "matx/matx_log.h"

// Forward decls
matx_dense_linsolve_t matx_dense_linsolve_make_cblas(matx_alloc_t alloc);

const char* matx_dense_linsolve_backend_name(matx_dense_linsolve_backend_kind_t k)
{
    switch (k) {
    case MATX_LINSOLVE_BACKEND_CBLAS:
        return "CBLAS||BLIS";
    default:
        return "UNKNOWN";
    }
}

matx_dense_linsolve_t matx_dense_linsolve_default(matx_alloc_t alloc)
{
    return matx_dense_linsolve_make_cblas(alloc);
}

// High-level wrappers ------------------------------------------------------

// Dense real
matx_status_t matx_factor_dense_d_i8(const matx_dense_linsolve_t* ls,
                                     const matx_dense_d_i8_t A,
                                     matx_factor_dense_d_i8_t** out_F)
{
    if (!ls || !A || !out_F) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.factor_dense_d_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return ls->vt.factor_dense_d_i8(&ls->alloc, A, out_F);
}

matx_status_t matx_solve_dense_d_i8_factor(const matx_dense_linsolve_t* ls,
                                           const matx_factor_dense_d_i8_t* F,
                                           const matx_double* b,
                                           matx_double* x)
{
    if (!ls || !F || !b || !x) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.solve_dense_d_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return ls->vt.solve_dense_d_i8(&ls->alloc, F, b, x);
}

void matx_factor_dense_d_i8_destroy(const matx_dense_linsolve_t* ls, matx_factor_dense_d_i8_t* F)
{
    if (!ls || !F)
        return;
    if (ls->vt.factor_dense_d_i8_destroy) {
        ls->vt.factor_dense_d_i8_destroy(&ls->alloc, F);
    }
}

matx_status_t matx_solve_dense_d_i8(const matx_dense_linsolve_t* ls,
                                    const matx_dense_d_i8_t A,
                                    const matx_double* b,
                                    matx_double* x)
{
    if (!ls || !A || !b || !x) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_factor_dense_d_i8_t* F = NULL;
    matx_status_t st = matx_factor_dense_d_i8(ls, A, &F);
    if (st != MATX_OK)
        return st;
    st = matx_solve_dense_d_i8_factor(ls, F, b, x);
    matx_factor_dense_d_i8_destroy(ls, F);
    return st;
}

matx_status_t matx_factor_dense_z_i8(const matx_dense_linsolve_t* ls,
                                     const matx_dense_z_i8_t A,
                                     matx_factor_dense_z_i8_t** out_F)
{
    if (!ls || !A || !out_F) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.factor_dense_z_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return ls->vt.factor_dense_z_i8(&ls->alloc, A, out_F);
}

matx_status_t matx_solve_dense_z_i8_factor(const matx_dense_linsolve_t* ls,
                                           const matx_factor_dense_z_i8_t* F,
                                           const matx_vec_z_i8_t b,
                                           matx_vec_z_i8_t x)
{
    if (!ls || !F || !b || !x) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.solve_dense_z_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return ls->vt.solve_dense_z_i8(&ls->alloc, F, b, x);
}

void matx_factor_dense_z_i8_destroy(const matx_dense_linsolve_t* ls, matx_factor_dense_z_i8_t* F)
{
    if (!ls || !F)
        return;
    if (ls->vt.factor_dense_z_i8_destroy) {
        ls->vt.factor_dense_z_i8_destroy(&ls->alloc, F);
    }
}

matx_status_t matx_solve_dense_z_i8(const matx_dense_linsolve_t* ls,
                                    const matx_dense_z_i8_t A,
                                    const matx_vec_z_i8_t b,
                                    matx_vec_z_i8_t x)
{
    if (!ls || !A || !b || !x) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_factor_dense_z_i8_t* F = NULL;
    matx_status_t st = matx_factor_dense_z_i8(ls, A, &F);
    if (st != MATX_OK)
        return st;
    st = matx_solve_dense_z_i8_factor(ls, F, b, x);
    matx_factor_dense_z_i8_destroy(ls, F);
    return st;
}

// ---- Cholesky wrappers ----

matx_status_t matx_factor_chol_d_i8(const matx_dense_linsolve_t* ls,
                                    const matx_dense_d_i8_t A,
                                    matx_uplo_t uplo,
                                    matx_factor_dense_d_i8_t** out_F)
{
    if (!ls || !A || !out_F) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.potrf_d_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return ls->vt.potrf_d_i8(&ls->alloc, A, uplo, out_F);
}

matx_status_t matx_solve_chol_d_i8(const matx_dense_linsolve_t* ls,
                                   const matx_factor_dense_d_i8_t* F,
                                   const matx_double* b,
                                   matx_double* x)
{
    if (!ls || !F || !b || !x) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.potrs_d_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return ls->vt.potrs_d_i8(&ls->alloc, F, b, x);
}

matx_status_t matx_solve_chol_d_i8_oneshot(const matx_dense_linsolve_t* ls,
                                           const matx_dense_d_i8_t A,
                                           matx_uplo_t uplo,
                                           const matx_double* b,
                                           matx_double* x)
{
    if (!ls || !A || !b || !x) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_factor_dense_d_i8_t* F = NULL;
    matx_status_t st = matx_factor_chol_d_i8(ls, A, uplo, &F);
    if (st != MATX_OK)
        return st;
    st = matx_solve_chol_d_i8(ls, F, b, x);
    matx_factor_dense_d_i8_destroy(ls, F);
    return st;
}

matx_status_t matx_factor_chol_z_i8(const matx_dense_linsolve_t* ls,
                                    const matx_dense_z_i8_t A,
                                    matx_uplo_t uplo,
                                    matx_factor_dense_z_i8_t** out_F)
{
    if (!ls || !A || !out_F) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.potrf_z_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return ls->vt.potrf_z_i8(&ls->alloc, A, uplo, out_F);
}

matx_status_t matx_solve_chol_z_i8(const matx_dense_linsolve_t* ls,
                                   const matx_factor_dense_z_i8_t* F,
                                   const matx_vec_z_i8_t b,
                                   matx_vec_z_i8_t x)
{
    if (!ls || !F || !b || !x) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.potrs_z_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return ls->vt.potrs_z_i8(&ls->alloc, F, b, x);
}

// ---- GELS wrappers ----

matx_status_t matx_gels_d_i8(const matx_dense_linsolve_t* ls,
                             const matx_dense_d_i8_t A,
                             const matx_double* b,
                             matx_double* x)
{
    if (!ls || !A || !b || !x) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.gels_d_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return ls->vt.gels_d_i8(&ls->alloc, A, b, x);
}

matx_status_t matx_gels_z_i8(const matx_dense_linsolve_t* ls,
                             const matx_dense_z_i8_t A,
                             const matx_vec_z_i8_t b,
                             matx_vec_z_i8_t x)
{
    if (!ls || !A || !b || !x) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.gels_z_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return ls->vt.gels_z_i8(&ls->alloc, A, b, x);
}

// ---- SYEV wrapper ----

matx_status_t matx_syev_d_i8(const matx_dense_linsolve_t* ls,
                             const matx_dense_d_i8_t A,
                             matx_vec_d_i8_t eigenvalues,
                             matx_dense_d_i8_t* eigenvectors)
{
    if (!ls || !A || !eigenvalues) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.syev_d_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return ls->vt.syev_d_i8(&ls->alloc, A, eigenvalues, eigenvectors);
}

// ---- GESVD wrappers ----

matx_status_t matx_gesvd_d_i8(const matx_dense_linsolve_t* ls,
                              const matx_dense_d_i8_t A,
                              matx_vec_d_i8_t S,
                              matx_dense_d_i8_t* U,
                              matx_dense_d_i8_t* Vt)
{
    if (!ls || !A || !S) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.gesvd_d_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return ls->vt.gesvd_d_i8(&ls->alloc, A, S, U, Vt);
}

matx_status_t matx_gesvd_z_i8(const matx_dense_linsolve_t* ls,
                              const matx_dense_z_i8_t A,
                              matx_vec_d_i8_t S,
                              matx_dense_z_i8_t* U,
                              matx_dense_z_i8_t* Vt)
{
    if (!ls || !A || !S) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.gesvd_z_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return ls->vt.gesvd_z_i8(&ls->alloc, A, S, U, Vt);
}

// ---- Hermitian eigenvalue (complex) ----

matx_status_t matx_syev_z_i8(const matx_dense_linsolve_t* ls,
                             const matx_dense_z_i8_t A,
                             matx_vec_d_i8_t eigenvalues,
                             matx_dense_z_i8_t* eigenvectors)
{
    if (!ls || !A || !eigenvalues) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.syev_z_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return ls->vt.syev_z_i8(&ls->alloc, A, eigenvalues, eigenvectors);
}

// ---- General eigenvalues (real) ----

matx_status_t matx_geev_d_i8(const matx_dense_linsolve_t* ls,
                             const matx_dense_d_i8_t A,
                             matx_vec_z_i8_t eigenvalues,
                             matx_dense_d_i8_t* eigenvectors_right,
                             matx_dense_d_i8_t* eigenvectors_left)
{
    if (!ls || !A || !eigenvalues) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.geev_d_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return ls->vt.geev_d_i8(&ls->alloc, A, eigenvalues, eigenvectors_right, eigenvectors_left);
}

// ---- General eigenvalues (complex) ----

matx_status_t matx_geev_z_i8(const matx_dense_linsolve_t* ls,
                             const matx_dense_z_i8_t A,
                             matx_vec_z_i8_t eigenvalues,
                             matx_dense_z_i8_t* eigenvectors_right,
                             matx_dense_z_i8_t* eigenvectors_left)
{
    if (!ls || !A || !eigenvalues) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.geev_z_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return ls->vt.geev_z_i8(&ls->alloc, A, eigenvalues, eigenvectors_right, eigenvectors_left);
}

// ---- QR factorization ----

matx_status_t matx_qr_d_i8(const matx_dense_linsolve_t* ls,
                           const matx_dense_d_i8_t A,
                           matx_dense_d_i8_t* Q,
                           matx_dense_d_i8_t* R)
{
    if (!ls || !A || !Q || !R) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.qr_d_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return ls->vt.qr_d_i8(&ls->alloc, A, Q, R);
}

matx_status_t matx_qr_z_i8(const matx_dense_linsolve_t* ls,
                           const matx_dense_z_i8_t A,
                           matx_dense_z_i8_t* Q,
                           matx_dense_z_i8_t* R)
{
    if (!ls || !A || !Q || !R) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.qr_z_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return ls->vt.qr_z_i8(&ls->alloc, A, Q, R);
}

// ---- Determinant ----

matx_status_t matx_det_dense_d_i8(const matx_dense_linsolve_t* ls,
                                  const matx_dense_d_i8_t A,
                                  matx_double* det)
{
    if (!ls || !A || !det) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.det_dense_d_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return ls->vt.det_dense_d_i8(&ls->alloc, A, det);
}

matx_status_t matx_det_dense_z_i8(const matx_dense_linsolve_t* ls,
                                  const matx_dense_z_i8_t A,
                                  matx_complex_d_t* det)
{
    if (!ls || !A || !det) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.det_dense_z_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return ls->vt.det_dense_z_i8(&ls->alloc, A, det);
}

// ---- Condition number ----

matx_status_t matx_cond_dense_d_i8(const matx_dense_linsolve_t* ls,
                                   const matx_dense_d_i8_t A,
                                   matx_double* cond)
{
    if (!ls || !A || !cond) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.cond_dense_d_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return ls->vt.cond_dense_d_i8(&ls->alloc, A, cond);
}

matx_status_t matx_cond_dense_z_i8(const matx_dense_linsolve_t* ls,
                                   const matx_dense_z_i8_t A,
                                   matx_double* cond)
{
    if (!ls || !A || !cond) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.cond_dense_z_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return ls->vt.cond_dense_z_i8(&ls->alloc, A, cond);
}

// ---- Multi-RHS solve (dense) ----

matx_status_t matx_solve_dense_d_i8_factor_mrhs(const matx_dense_linsolve_t* ls,
                                                 const matx_factor_dense_d_i8_t* F,
                                                 const matx_dense_d_i8_t B,
                                                 matx_dense_d_i8_t* X)
{
    if (!ls || !F || !B || !X) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.solve_dense_mrhs_d_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return ls->vt.solve_dense_mrhs_d_i8(&ls->alloc, F, B, X);
}

matx_status_t matx_solve_dense_z_i8_factor_mrhs(const matx_dense_linsolve_t* ls,
                                                 const matx_factor_dense_z_i8_t* F,
                                                 const matx_dense_z_i8_t B,
                                                 matx_dense_z_i8_t* X)
{
    if (!ls || !F || !B || !X) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.solve_dense_mrhs_z_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return ls->vt.solve_dense_mrhs_z_i8(&ls->alloc, F, B, X);
}

// ---- Multi-RHS Cholesky solve ----

matx_status_t matx_solve_chol_d_i8_factor_mrhs(const matx_dense_linsolve_t* ls,
                                                const matx_factor_dense_d_i8_t* F,
                                                const matx_dense_d_i8_t B,
                                                matx_dense_d_i8_t* X)
{
    if (!ls || !F || !B || !X) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.potrs_mrhs_d_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return ls->vt.potrs_mrhs_d_i8(&ls->alloc, F, B, X);
}

matx_status_t matx_solve_chol_z_i8_factor_mrhs(const matx_dense_linsolve_t* ls,
                                                const matx_factor_dense_z_i8_t* F,
                                                const matx_dense_z_i8_t B,
                                                matx_dense_z_i8_t* X)
{
    if (!ls || !F || !B || !X) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.potrs_mrhs_z_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return ls->vt.potrs_mrhs_z_i8(&ls->alloc, F, B, X);
}

// ---- LDL^T factorization ----

matx_status_t matx_factor_ldl_d_i8(const matx_dense_linsolve_t* ls,
                                    const matx_dense_d_i8_t A,
                                    matx_uplo_t uplo,
                                    matx_factor_dense_d_i8_t** out_F)
{
    if (!ls || !A || !out_F) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.sytrf_d_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return ls->vt.sytrf_d_i8(&ls->alloc, A, uplo, out_F);
}

matx_status_t matx_solve_ldl_d_i8(const matx_dense_linsolve_t* ls,
                                   const matx_factor_dense_d_i8_t* F,
                                   const matx_double* b,
                                   matx_double* x)
{
    if (!ls || !F || !b || !x) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.sytrs_d_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return ls->vt.sytrs_d_i8(&ls->alloc, F, b, x);
}

void matx_factor_ldl_d_i8_destroy(const matx_dense_linsolve_t* ls,
                                   matx_factor_dense_d_i8_t* F)
{
    if (!ls || !F)
        return;
    if (ls->vt.sytrf_destroy_d)
        ls->vt.sytrf_destroy_d(&ls->alloc, F);
}

matx_status_t matx_factor_ldl_z_i8(const matx_dense_linsolve_t* ls,
                                    const matx_dense_z_i8_t A,
                                    matx_uplo_t uplo,
                                    matx_factor_dense_z_i8_t** out_F)
{
    if (!ls || !A || !out_F) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.sytrf_z_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return ls->vt.sytrf_z_i8(&ls->alloc, A, uplo, out_F);
}

matx_status_t matx_solve_ldl_z_i8(const matx_dense_linsolve_t* ls,
                                   const matx_factor_dense_z_i8_t* F,
                                   const matx_vec_z_i8_t b,
                                   matx_vec_z_i8_t x)
{
    if (!ls || !F || !b || !x) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.sytrs_z_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return ls->vt.sytrs_z_i8(&ls->alloc, F, b, x);
}

void matx_factor_ldl_z_i8_destroy(const matx_dense_linsolve_t* ls,
                                   matx_factor_dense_z_i8_t* F)
{
    if (!ls || !F)
        return;
    if (ls->vt.sytrf_destroy_z)
        ls->vt.sytrf_destroy_z(&ls->alloc, F);
}

// ---- QR with column pivoting ----

matx_status_t matx_qrp_d_i8(const matx_dense_linsolve_t* ls,
                             const matx_dense_d_i8_t A,
                             matx_dense_d_i8_t* Q,
                             matx_dense_d_i8_t* R,
                             matx_vec_d_i8_t* jpvt)
{
    if (!ls || !A || !Q || !R || !jpvt) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.qrp_d_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return ls->vt.qrp_d_i8(&ls->alloc, A, Q, R, jpvt);
}

matx_status_t matx_qrp_z_i8(const matx_dense_linsolve_t* ls,
                             const matx_dense_z_i8_t A,
                             matx_dense_z_i8_t* Q,
                             matx_dense_z_i8_t* R,
                             matx_vec_d_i8_t* jpvt)
{
    if (!ls || !A || !Q || !R || !jpvt) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.qrp_z_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return ls->vt.qrp_z_i8(&ls->alloc, A, Q, R, jpvt);
}

// ---- Pseudo-inverse ----

matx_status_t matx_pinv_dense_d_i8(const matx_dense_linsolve_t* ls,
                                    const matx_dense_d_i8_t A,
                                    matx_double rcond,
                                    matx_dense_d_i8_t* out)
{
    if (!ls || !A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.pinv_d_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return ls->vt.pinv_d_i8(&ls->alloc, A, rcond, out);
}

matx_status_t matx_pinv_dense_z_i8(const matx_dense_linsolve_t* ls,
                                    const matx_dense_z_i8_t A,
                                    matx_double rcond,
                                    matx_dense_z_i8_t* out)
{
    if (!ls || !A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.pinv_z_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return ls->vt.pinv_z_i8(&ls->alloc, A, rcond, out);
}

// ---- Matrix rank ----

matx_status_t matx_rank_dense_d_i8(const matx_dense_linsolve_t* ls,
                                    const matx_dense_d_i8_t A,
                                    matx_double tol,
                                    matx_int64_t* rank)
{
    if (!ls || !A || !rank) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.rank_d_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return ls->vt.rank_d_i8(&ls->alloc, A, tol, rank);
}

matx_status_t matx_rank_dense_z_i8(const matx_dense_linsolve_t* ls,
                                    const matx_dense_z_i8_t A,
                                    matx_double tol,
                                    matx_int64_t* rank)
{
    if (!ls || !A || !rank) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.rank_z_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return ls->vt.rank_z_i8(&ls->alloc, A, tol, rank);
}

// ---- Generalized symmetric eigenvalue ----

matx_status_t matx_sygv_d_i8(const matx_dense_linsolve_t* ls,
                              const matx_dense_d_i8_t A,
                              const matx_dense_d_i8_t B,
                              matx_vec_d_i8_t eigenvalues,
                              matx_dense_d_i8_t* eigenvectors)
{
    if (!ls || !A || !B || !eigenvalues) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.sygv_d_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return ls->vt.sygv_d_i8(&ls->alloc, A, B, eigenvalues, eigenvectors);
}

matx_status_t matx_sygv_z_i8(const matx_dense_linsolve_t* ls,
                              const matx_dense_z_i8_t A,
                              const matx_dense_z_i8_t B,
                              matx_vec_d_i8_t eigenvalues,
                              matx_dense_z_i8_t* eigenvectors)
{
    if (!ls || !A || !B || !eigenvalues) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.sygv_z_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return ls->vt.sygv_z_i8(&ls->alloc, A, B, eigenvalues, eigenvectors);
}

// ---- LQ factorization ----

matx_status_t matx_lq_d_i8(const matx_dense_linsolve_t* ls,
                            const matx_dense_d_i8_t A,
                            matx_dense_d_i8_t* L,
                            matx_dense_d_i8_t* Q)
{
    if (!ls || !A || !L || !Q) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.lq_d_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return ls->vt.lq_d_i8(&ls->alloc, A, L, Q);
}

matx_status_t matx_lq_z_i8(const matx_dense_linsolve_t* ls,
                            const matx_dense_z_i8_t A,
                            matx_dense_z_i8_t* L,
                            matx_dense_z_i8_t* Q)
{
    if (!ls || !A || !L || !Q) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.lq_z_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return ls->vt.lq_z_i8(&ls->alloc, A, L, Q);
}
