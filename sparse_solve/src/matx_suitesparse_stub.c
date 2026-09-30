#include "matx/matx_sparse_solve.h"
#include "matx/matx_types_internal.h"

#include <limits.h>
#include <string.h>

#if MATX_HAVE_SUITESPARSE
#include "cholmod.h"
#include "klu.h"
#endif
#include "matx/matx_log.h"

#if MATX_HAVE_SUITESPARSE

typedef struct matx_factor_sparse_d_i8_klu
{
    klu_l_symbolic* S;
    klu_l_numeric* N;
    klu_l_common common;
    matx_int64_t n;
} matx_factor_sparse_d_i8_klu_t;

typedef struct matx_factor_sparse_z_i8_klu
{
    klu_l_symbolic* S;
    klu_l_numeric* N;
    klu_l_common common;
    matx_int64_t n;
} matx_factor_sparse_z_i8_klu_t;

static void ss_factor_csc_d_i8_destroy(const matx_alloc_t* alloc, matx_factor_sparse_d_i8_t* F)
{
    if (!F || !F->reserved)
        return;
    matx_factor_sparse_d_i8_klu_t* ptr = (matx_factor_sparse_d_i8_klu_t*) F->reserved;
    klu_l_free_numeric(&ptr->N, &ptr->common);
    klu_l_free_symbolic(&ptr->S, &ptr->common);
    matx_free(alloc, ptr);
}

static void ss_factor_csc_z_i8_destroy(const matx_alloc_t* alloc, matx_factor_sparse_z_i8_t* F)
{
    if (!F || !F->reserved)
        return;
    matx_factor_sparse_z_i8_klu_t* ptr = (matx_factor_sparse_z_i8_klu_t*) F->reserved;
    klu_zl_free_numeric(&ptr->N, &ptr->common);
    klu_l_free_symbolic(&ptr->S, &ptr->common);
    matx_free(alloc, ptr);
}

// Sparse real: KLU-based ---------------------------------------------------
static matx_status_t ss_factor_csc_d_i8(const matx_alloc_t* alloc, matx_coo_d_i8_t A, matx_factor_sparse_d_i8_t* out_F)
{
    if (!A || !out_F) {
        MATX_ERROR("input pointer is null error");
        return MATX_ERR_INVALID_ARG;
    }
    if (out_F != NULL) {
        ss_factor_csc_d_i8_destroy(alloc, out_F);
    }

    matx_status_t st = coo_to_csc_d_i8(A);
    if (st != MATX_OK) {
        MATX_ERROR("coo_to_csc_d_i8 error:%d", st);
        return st;
    }
    if (!A->handle_csc->col_ptr || !A->handle_csc->row_ind || !A->handle_csc->values) {
        MATX_ERROR("csc pointer is null error");
        return MATX_ERR_INVALID_ARG;
    }
    if (A->nrows != A->ncols) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    const matx_int64_t n = A->nrows;
    if (n == 0) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (n > (matx_int64_t) INT_MAX) {
        MATX_ERROR("n > INT_MAX");
        return MATX_ERR_NOT_SUPPORTED;
    }

    matx_factor_sparse_d_i8_klu_t* F = (matx_factor_sparse_d_i8_klu_t*) matx_malloc(alloc, sizeof(*F));
    if (!F) {
        MATX_ERROR("malloc Factor handle error");
        return MATX_ERR_OUT_OF_MEMORY;
    }

    memset(F, 0, sizeof(*F));
    F->n = (matx_int64_t) n;
    klu_l_defaults(&F->common);

    F->S = klu_l_analyze(F->n, A->handle_csc->col_ptr, A->handle_csc->row_ind, &F->common);
    if (!F->S) {
        MATX_ERROR("call klu_l_analyze error:%d", F->common.status);
        matx_free(alloc, F);
        return MATX_ERR_INTERNAL;
    }

    st = coo_to_csc_d_i8_value_remap(A);

    if (st != MATX_OK) {
        MATX_ERROR("call coo_to_csc_d_i8_value_remap error:%d", F->common.status);
        klu_l_free_symbolic(&F->S, &F->common);
        matx_free(alloc, F);
        return st;
    }

    F->N = klu_l_factor(A->handle_csc->col_ptr,
                        A->handle_csc->row_ind,
                        A->handle_csc->values,
                        F->S,
                        &F->common);

    if (!F->N) {
        MATX_ERROR("call klu_l_factor error:%d", F->common.status);
        klu_l_free_symbolic(&F->S, &F->common);
        matx_free(alloc, F);
        return MATX_ERR_INTERNAL;
    }

    out_F->reserved = F;
    return MATX_OK;
}

static matx_status_t ss_solve_csc_d_i8(const matx_alloc_t* alloc,
                                       matx_factor_sparse_d_i8_t* F,
                                       const matx_double* b,
                                       matx_double* x)
{
    if (!F || !F->reserved || !b || !x) {
        MATX_ERROR("input pointer is null");
        return MATX_ERR_INVALID_ARG;
    }
    matx_factor_sparse_d_i8_klu_t* ptr = (matx_factor_sparse_d_i8_klu_t*) F->reserved;
    const matx_int64_t n = ptr->n;
    for (matx_int64_t i = 0; i < n; ++i) {
        x[i] = b[i];
    }

    const int status = klu_l_solve(ptr->S, ptr->N, n, 1, x, &ptr->common);
    if (!status) {
        MATX_ERROR("KLU solve failed with status %d", status);
        return MATX_ERR_INTERNAL;
    }
    return MATX_OK;
}

static matx_status_t ss_factor_csc_z_i8(const matx_alloc_t* alloc, matx_coo_z_i8_t A, matx_factor_sparse_z_i8_t* out_F)
{
    if (!A || !out_F) {
        MATX_ERROR("input pointer is null");
        return MATX_ERR_INVALID_ARG;
    }

    if (out_F != NULL) {
        ss_factor_csc_z_i8_destroy(alloc, out_F);
    }

    matx_status_t st = coo_to_csc_z_i8(A);
    if (st != MATX_OK) {
        MATX_ERROR("coo_to_csc_z_i8 error");
        return st;
    }

    if (!A->handle_csc->col_ptr || !A->handle_csc->row_ind || !A->handle_csc->values) {
        MATX_ERROR("csc pointer error");
        return MATX_ERR_INVALID_ARG;
    }
    if (A->nrows != A->ncols) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    const matx_int64_t n = A->nrows;
    if (n > INT_MAX) {
        MATX_ERROR("n > INT_MAX");
        return MATX_ERR_NOT_SUPPORTED;
    }

    matx_factor_sparse_z_i8_klu_t* F = (matx_factor_sparse_z_i8_klu_t*) matx_malloc(alloc, sizeof(*F));
    if (!F) {
        MATX_ERROR("malloc F failed");
        return MATX_ERR_OUT_OF_MEMORY;
    }

    memset(F, 0, sizeof(*F));
    F->n = (matx_int64_t) n;

    klu_l_defaults(&F->common);

    F->S = klu_l_analyze(F->n, A->handle_csc->col_ptr, A->handle_csc->row_ind, &F->common);

    if (!F->S) {
        MATX_ERROR("klu_l_analyze failed error:%d", F->common.status);
        goto fail;
    }

    st = coo_to_csc_z_i8_value_remap(A);
    if (st != MATX_OK) {
        MATX_ERROR("coo_to_csc_z_i8_value_remap failed");
        goto fail;
    }

    F->N = klu_zl_factor(A->handle_csc->col_ptr,
                         A->handle_csc->row_ind,
                         (double*) A->handle_csc->values,
                         F->S,
                         &F->common);

    if (!F->N) {
        MATX_ERROR("klu_zl_factor failed error:%d", F->common.status);
        goto fail;
    }

    out_F->reserved = F;
    return MATX_OK;

fail:
    klu_l_free_symbolic(&F->S, &F->common);
    matx_free(alloc, F);
    MATX_ERROR("%s: internal error", __func__);
    return MATX_ERR_INTERNAL;
}

static matx_status_t ss_solve_csc_z_i8(const matx_alloc_t* alloc,
                                       matx_factor_sparse_z_i8_t* F,
                                       const matx_vec_z_i8_t b,
                                       matx_vec_z_i8_t x)
{
    if (!F || !b || !x) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (b->stride != 1 || x->stride != 1) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    matx_factor_sparse_z_i8_klu_t* ptr = (matx_factor_sparse_z_i8_klu_t*) F->reserved;
    matx_int64_t n = ptr->n;

    memcpy(x->data, b->data, sizeof(matx_complex_d_t) * n);

    matx_int64_t status = klu_zl_solve(ptr->S, ptr->N, n, 1, (matx_double*) x->data, &ptr->common);

    if (!status) {
        MATX_ERROR("klu_zl_solve failed");
        return MATX_ERR_INTERNAL;
    }

    return MATX_OK;
}

// ---- Sparse Cholesky via CHOLMOD ----

typedef struct matx_factor_sparse_d_i8_cholmod
{
    cholmod_factor* L;
    cholmod_common common;
    matx_int64_t n;
} matx_factor_sparse_d_i8_cholmod_t;

static void ss_factor_chol_csc_d_i8_destroy(const matx_alloc_t* alloc, matx_factor_sparse_d_i8_t* F)
{
    if (!F || !F->reserved)
        return;
    matx_factor_sparse_d_i8_cholmod_t* ptr = (matx_factor_sparse_d_i8_cholmod_t*) F->reserved;
    if (ptr->L) {
        cholmod_l_free_factor(&ptr->L, &ptr->common);
    }
    cholmod_l_finish(&ptr->common);
    matx_free(alloc, ptr);
}

static matx_status_t ss_factor_chol_csc_d_i8(const matx_alloc_t* alloc, matx_coo_d_i8_t A, matx_factor_sparse_d_i8_t* out_F)
{
    if (!A || !out_F) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (A->nrows != A->ncols) {
        MATX_ERROR("%s: matrix must be square", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    matx_status_t st = coo_to_csc_d_i8(A);
    if (st != MATX_OK) {
        MATX_ERROR("coo_to_csc_d_i8 error:%d", st);
        return st;
    }

    const matx_int64_t n = A->nrows;
    const matx_int64_t nnz = A->handle_csc->nnz;

    matx_factor_sparse_d_i8_cholmod_t* F = (matx_factor_sparse_d_i8_cholmod_t*) matx_malloc(alloc, sizeof(*F));
    if (!F) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(F, 0, sizeof(*F));
    F->n = n;

    cholmod_l_start(&F->common);

    cholmod_sparse* A_chol = cholmod_l_allocate_sparse(n, n, nnz, 1, 1, 0, CHOLMOD_REAL, &F->common);
    if (!A_chol) {
        MATX_ERROR("%s: cholmod_l_allocate_sparse failed", __func__);
        cholmod_l_finish(&F->common);
        matx_free(alloc, F);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    matx_int64_t* Ap = (matx_int64_t*) A_chol->p;
    matx_int64_t* Ai = (matx_int64_t*) A_chol->i;
    matx_double* Ax = (matx_double*) A_chol->x;

    for (matx_int64_t j = 0; j <= n; ++j)
        Ap[j] = A->handle_csc->col_ptr[j];
    for (matx_int64_t k = 0; k < nnz; ++k) {
        Ai[k] = A->handle_csc->row_ind[k];
        Ax[k] = A->handle_csc->values[k];
    }
    A_chol->nzmax = nnz;

    F->L = cholmod_l_analyze(A_chol, &F->common);
    if (!F->L) {
        MATX_ERROR("%s: cholmod_l_analyze failed", __func__);
        cholmod_l_free_sparse(&A_chol, &F->common);
        cholmod_l_finish(&F->common);
        matx_free(alloc, F);
        return MATX_ERR_INTERNAL;
    }

    cholmod_l_factorize(A_chol, F->L, &F->common);
    if (F->common.status != CHOLMOD_OK) {
        MATX_ERROR("%s: cholmod_l_factorize failed, status=%d", __func__, F->common.status);
        cholmod_l_free_sparse(&A_chol, &F->common);
        ss_factor_chol_csc_d_i8_destroy(alloc, out_F);
        out_F->reserved = NULL;
        return MATX_ERR_INTERNAL;
    }

    cholmod_l_free_sparse(&A_chol, &F->common);
    out_F->reserved = F;
    return MATX_OK;
}

static matx_status_t ss_solve_chol_csc_d_i8(const matx_alloc_t* alloc,
                                            matx_factor_sparse_d_i8_t* F,
                                            const matx_double* b,
                                            matx_double* x)
{
    if (!F || !F->reserved || !b || !x) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_factor_sparse_d_i8_cholmod_t* ptr = (matx_factor_sparse_d_i8_cholmod_t*) F->reserved;
    const matx_int64_t n = ptr->n;

    cholmod_dense* b_chol = cholmod_l_allocate_dense(n, 1, n, CHOLMOD_REAL, &ptr->common);
    if (!b_chol) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memcpy(b_chol->x, b, n * sizeof(matx_double));

    cholmod_dense* x_chol = cholmod_l_solve(CHOLMOD_A, ptr->L, b_chol, &ptr->common);
    if (!x_chol) {
        MATX_ERROR("%s: cholmod_l_solve failed", __func__);
        cholmod_l_free_dense(&b_chol, &ptr->common);
        return MATX_ERR_INTERNAL;
    }
    memcpy(x, x_chol->x, n * sizeof(matx_double));

    cholmod_l_free_dense(&x_chol, &ptr->common);
    cholmod_l_free_dense(&b_chol, &ptr->common);
    return MATX_OK;
}

matx_sparse_linsolve_t matx_linsolve_make_suitesparse_klu(matx_alloc_t alloc)
{
    matx_sparse_linsolve_t ls
        = {.kind = MATX_LINSOLVE_BACKEND_SUITESPARSE_KLU,
           .alloc = alloc,
           .vt = {.factor_csc_d_i8 = &ss_factor_csc_d_i8,
                  .solve_csc_d_i8 = &ss_solve_csc_d_i8,
                  .factor_csc_d_i8_destroy = &ss_factor_csc_d_i8_destroy,
                  .factor_csc_z_i8 = &ss_factor_csc_z_i8,
                  .solve_csc_z_i8 = &ss_solve_csc_z_i8,
                  .factor_csc_z_i8_destroy = &ss_factor_csc_z_i8_destroy,
                  .factor_chol_csc_d_i8 = &ss_factor_chol_csc_d_i8,
                  .solve_chol_csc_d_i8 = &ss_solve_chol_csc_d_i8,
                  .factor_chol_csc_d_i8_destroy = &ss_factor_chol_csc_d_i8_destroy}};
    return ls;
}

#else

matx_sparse_linsolve_t matx_linsolve_make_suitesparse_klu(matx_alloc_t alloc)
{
    matx_sparse_linsolve_t ls = {.kind = MATX_LINSOLVE_BACKEND_SUITESPARSE_KLU, .alloc = alloc, .vt = {0}};
    return ls;
}

#endif /* MATX_HAVE_SUITESPARSE */