#include "matx/matx_log.h"
#include "matx/matx_sparse_solve.h"
#include "matx/matx_func.h"
#include <string.h>

/* ── small-n dense fallback: for tiny systems (n ≤ 64), COO→dense+solve ── */
#if MATX_HAVE_OPENBLAS
#define MATX_SMALL_DENSE_SOLVE_LIMIT 64

extern void dgesv_(const int* n, const int* nrhs, double* A, const int* lda,
                   int* ipiv, double* B, const int* ldb, int* info);
extern void zgesv_(const int* n, const int* nrhs, void* A, const int* lda,
                   int* ipiv, void* B, const int* ldb, int* info);

static matx_status_t solve_small_dense_d(const matx_alloc_t* alloc,
                                          matx_coo_d_i8_t A,
                                          const matx_double* b,
                                          matx_double* x)
{
    const int n = (int) A->nrows;
    const int nrhs = 1;
    const size_t mat_bytes = (size_t) n * (size_t) n * sizeof(matx_double);
    const size_t vec_bytes = (size_t) n * sizeof(matx_double);
    matx_double* dense_A = (matx_double*) matx_malloc(alloc, mat_bytes);
    matx_double* B       = (matx_double*) matx_malloc(alloc, vec_bytes);
    int*         ipiv    = (int*)         matx_malloc(alloc, (size_t) n * sizeof(int));
    if (!dense_A || !B || !ipiv) {
        matx_free(alloc, dense_A);
        matx_free(alloc, B);
        matx_free(alloc, ipiv);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(dense_A, 0, mat_bytes);
    for (matx_int64_t k = 0; k < A->nnz; ++k)
        dense_A[A->rows[k] + A->columns[k] * n] += A->values[k];
    memcpy(B, b, vec_bytes);

    int info = 0;
    dgesv_(&n, &nrhs, dense_A, &n, ipiv, B, &n, &info);
    if (info != 0) {
        matx_free(alloc, dense_A);
        matx_free(alloc, B);
        matx_free(alloc, ipiv);
        MATX_ERROR("%s: dgesv failed, info=%d", __func__, info);
        return MATX_ERR_INTERNAL;
    }
    memcpy(x, B, vec_bytes);
    matx_free(alloc, dense_A);
    matx_free(alloc, B);
    matx_free(alloc, ipiv);
    return MATX_OK;
}

static matx_status_t solve_small_dense_z(const matx_alloc_t* alloc,
                                          matx_coo_z_i8_t A,
                                          const matx_vec_z_i8_t b,
                                          matx_vec_z_i8_t x)
{
    const int n = (int) A->nrows;
    const int nrhs = 1;
    const size_t mat_bytes = (size_t) n * (size_t) n * sizeof(matx_complex_double);
    const size_t vec_bytes = (size_t) n * sizeof(matx_complex_double);
    matx_complex_double* dense_A = (matx_complex_double*) matx_malloc(alloc, mat_bytes);
    matx_complex_double* B       = (matx_complex_double*) matx_malloc(alloc, vec_bytes);
    int*                 ipiv    = (int*) matx_malloc(alloc, (size_t) n * sizeof(int));
    if (!dense_A || !B || !ipiv) {
        matx_free(alloc, dense_A);
        matx_free(alloc, B);
        matx_free(alloc, ipiv);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(dense_A, 0, mat_bytes);
    for (matx_int64_t k = 0; k < A->nnz; ++k) {
        dense_A[A->rows[k] + A->columns[k] * n].real += A->values[k].real;
        dense_A[A->rows[k] + A->columns[k] * n].imag += A->values[k].imag;
    }
    memcpy(B, b->data, vec_bytes);

    int info = 0;
    zgesv_(&n, &nrhs, dense_A, &n, ipiv, B, &n, &info);
    if (info != 0) {
        matx_free(alloc, dense_A);
        matx_free(alloc, B);
        matx_free(alloc, ipiv);
        MATX_ERROR("%s: zgesv failed, info=%d", __func__, info);
        return MATX_ERR_INTERNAL;
    }
    memcpy(x->data, B, vec_bytes);
    matx_free(alloc, dense_A);
    matx_free(alloc, B);
    matx_free(alloc, ipiv);
    return MATX_OK;
}
#endif /* MATX_HAVE_OPENBLAS */

// Forward decls
matx_sparse_linsolve_t matx_linsolve_make_suitesparse_klu(matx_alloc_t alloc);
matx_sparse_linsolve_t matx_linsolve_make_umfpack(matx_alloc_t alloc);
matx_sparse_linsolve_t matx_linsolve_make_cxsparse(matx_alloc_t alloc);
matx_sparse_linsolve_t matx_linsolve_make_superlu(matx_alloc_t alloc);
matx_sparse_linsolve_t matx_linsolve_make_mumps(matx_alloc_t alloc);

const char* matx_sparse_linsolve_backend_name(matx_sparse_linsolve_backend_kind_t k)
{
    switch (k) {
    case MATX_LINSOLVE_BACKEND_SUITESPARSE_KLU:
        return "SUITESPARSE";
    case MATX_LINSOLVE_BACKEND_UMFPACK:
        return "UMFPACK";
    case MATX_LINSOLVE_BACKEND_CXSPARSE:
        return "CXSPARSE";
    case MATX_LINSOLVE_BACKEND_SUPERLU:
        return "SUPERLU";
    case MATX_LINSOLVE_BACKEND_MUMPS:
        return "MUMPS";
    default:
        return "UNKNOWN";
    }
}

matx_sparse_linsolve_t matx_sparse_linsolve_default(matx_alloc_t alloc)
{
#if MATX_HAVE_UMFPACK
    //return matx_linsolve_make_mumps();
    return matx_linsolve_make_umfpack(alloc);
#elif MATX_HAVE_CXSPARSE
    return matx_linsolve_make_cxsparse(alloc);
#elif MATX_HAVE_SUPERLU
    return matx_linsolve_make_superlu(alloc);
#elif MATX_HAVE_MUMPS
    return matx_linsolve_make_mumps(alloc);
#else
    return matx_linsolve_make_suitesparse_klu(alloc);
#endif
}

matx_sparse_linsolve_t matx_sparse_linsolve_by_type(matx_sparse_linsolve_backend_kind_t k, matx_alloc_t alloc)
{
    switch (k) {
    case MATX_LINSOLVE_BACKEND_UMFPACK:
        return matx_linsolve_make_umfpack(alloc);
    case MATX_LINSOLVE_BACKEND_CXSPARSE:
        return matx_linsolve_make_cxsparse(alloc);
    case MATX_LINSOLVE_BACKEND_SUPERLU:
        return matx_linsolve_make_superlu(alloc);
    case MATX_LINSOLVE_BACKEND_MUMPS:
        return matx_linsolve_make_mumps(alloc);
    case MATX_LINSOLVE_BACKEND_SUITESPARSE_KLU:
    default:
        return matx_linsolve_make_suitesparse_klu(alloc);
    }
}

// High-level wrappers ------------------------------------------------------

// Sparse real
matx_status_t matx_factor_csc_d_i8(const matx_sparse_linsolve_t* ls,
                                   matx_coo_d_i8_t A,
                                   matx_factor_sparse_d_i8_t* out_F)
{
    if (!ls || !A || !out_F) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.factor_csc_d_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    out_F->alloc = ls->alloc;
    return ls->vt.factor_csc_d_i8(&ls->alloc, A, out_F);
}

matx_status_t matx_solve_csc_d_i8_factor(const matx_sparse_linsolve_t* ls,
                                         matx_factor_sparse_d_i8_t* F,
                                         const matx_double* b,
                                         matx_double* x)
{
    if (!ls || !F || !b || !x) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.solve_csc_d_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return ls->vt.solve_csc_d_i8(&ls->alloc, F, b, x);
}

void matx_factor_csc_d_i8_destroy(const matx_sparse_linsolve_t* ls, matx_factor_sparse_d_i8_t* F)
{
    if (!ls || !F)
        return;
    if (ls->vt.factor_csc_d_i8_destroy) {
        ls->vt.factor_csc_d_i8_destroy(&ls->alloc, F);
    }
}

matx_status_t matx_solve_csc_d_i8(const matx_sparse_linsolve_t* ls,
                                  matx_coo_d_i8_t A,
                                  const matx_double* b,
                                  matx_double* x)
{
    if (!ls || !A || !b || !x) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
#if MATX_HAVE_OPENBLAS
    if (A->nrows == A->ncols && A->nrows > 0
        && A->nrows <= MATX_SMALL_DENSE_SOLVE_LIMIT) {
        return solve_small_dense_d(&ls->alloc, A, b, x);
    }
#endif
    matx_factor_sparse_d_i8_t F;
    F.reserved = NULL;
    F.alloc = ls->alloc;
    matx_status_t st = matx_factor_csc_d_i8(ls, A, &F);
    if (st != MATX_OK)
        return st;
    st = matx_solve_csc_d_i8_factor(ls, &F, b, x);
    matx_factor_csc_d_i8_destroy(ls, &F);
    return st;
}

// Complex variants (default to NOT_SUPPORTED until backend provides them)
matx_status_t matx_factor_csc_z_i8(const matx_sparse_linsolve_t* ls,
                                   matx_coo_z_i8_t A,
                                   matx_factor_sparse_z_i8_t* out_F)
{
    if (!ls || !A || !out_F) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.factor_csc_z_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    out_F->alloc = ls->alloc;
    return ls->vt.factor_csc_z_i8(&ls->alloc, A, out_F);
}

matx_status_t matx_solve_csc_z_i8_factor(const matx_sparse_linsolve_t* ls,
                                         matx_factor_sparse_z_i8_t* F,
                                         const matx_vec_z_i8_t b,
                                         matx_vec_z_i8_t x)
{
    if (!ls || !F || !b || !x) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.solve_csc_z_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return ls->vt.solve_csc_z_i8(&ls->alloc, F, b, x);
}

void matx_factor_csc_z_i8_destroy(const matx_sparse_linsolve_t* ls, matx_factor_sparse_z_i8_t* F)
{
    if (!ls || !F)
        return;
    if (ls->vt.factor_csc_z_i8_destroy) {
        ls->vt.factor_csc_z_i8_destroy(&ls->alloc, F);
    }
}

matx_status_t matx_solve_csc_z_i8(const matx_sparse_linsolve_t* ls,
                                  matx_coo_z_i8_t A,
                                  const matx_vec_z_i8_t b,
                                  matx_vec_z_i8_t x)
{
    if (!ls || !A || !b || !x) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
#if MATX_HAVE_OPENBLAS
    if (A->nrows == A->ncols && A->nrows > 0
        && A->nrows <= MATX_SMALL_DENSE_SOLVE_LIMIT) {
        return solve_small_dense_z(&ls->alloc, A, b, x);
    }
#endif
    matx_factor_sparse_z_i8_t F;
    F.reserved = NULL;
    F.alloc = ls->alloc;
    matx_status_t st = matx_factor_csc_z_i8(ls, A, &F);
    if (st != MATX_OK)
        return st;
    st = matx_solve_csc_z_i8_factor(ls, &F, b, x);
    matx_factor_csc_z_i8_destroy(ls, &F);
    return st;
}

// ---- Sparse Cholesky wrappers ----

matx_status_t matx_factor_chol_coo_d_i8(const matx_sparse_linsolve_t* ls,
                                        matx_coo_d_i8_t A,
                                        matx_factor_sparse_d_i8_t* out_F)
{
    if (!ls || !A || !out_F) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.factor_chol_csc_d_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    out_F->alloc = ls->alloc;
    return ls->vt.factor_chol_csc_d_i8(&ls->alloc, A, out_F);
}

matx_status_t matx_solve_chol_coo_d_i8_factor(const matx_sparse_linsolve_t* ls,
                                              matx_factor_sparse_d_i8_t* F,
                                              const matx_double* b,
                                              matx_double* x)
{
    if (!ls || !F || !b || !x) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!ls->vt.solve_chol_csc_d_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return ls->vt.solve_chol_csc_d_i8(&ls->alloc, F, b, x);
}

void matx_factor_chol_coo_d_i8_destroy(const matx_sparse_linsolve_t* ls,
                                       matx_factor_sparse_d_i8_t* F)
{
    if (!ls || !F)
        return;
    if (ls->vt.factor_chol_csc_d_i8_destroy) {
        ls->vt.factor_chol_csc_d_i8_destroy(&ls->alloc, F);
    }
}

matx_status_t matx_solve_chol_coo_d_i8(const matx_sparse_linsolve_t* ls,
                                       matx_coo_d_i8_t A,
                                       const matx_double* b,
                                       matx_double* x)
{
    if (!ls || !A || !b || !x) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_factor_sparse_d_i8_t F;
    F.reserved = NULL;
    F.alloc = ls->alloc;
    matx_status_t st = matx_factor_chol_coo_d_i8(ls, A, &F);
    if (st != MATX_OK)
        return st;
    st = matx_solve_chol_coo_d_i8_factor(ls, &F, b, x);
    matx_factor_chol_coo_d_i8_destroy(ls, &F);
    return st;
}
