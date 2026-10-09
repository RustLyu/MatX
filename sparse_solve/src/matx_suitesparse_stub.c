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
    if (x != b)
        memmove(x, b, (size_t) n * sizeof(matx_double));

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
    if (!F || !F->reserved || !b || !x) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (b->stride != 1 || x->stride != 1) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    matx_factor_sparse_z_i8_klu_t* ptr = (matx_factor_sparse_z_i8_klu_t*) F->reserved;
    matx_int64_t n = ptr->n;

    if (x->data != b->data)
        memmove(x->data, b->data, (size_t) n * sizeof(matx_complex_d_t));

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
    cholmod_dense* rhs_work;
    cholmod_dense* solution_work;
    cholmod_dense* y_work;
    cholmod_dense* e_work;
    cholmod_common common;
    matx_int64_t n;
} matx_factor_sparse_d_i8_cholmod_t;

static void ss_factor_chol_csc_d_i8_destroy(const matx_alloc_t* alloc, matx_factor_sparse_d_i8_t* F)
{
    if (!F || !F->reserved)
        return;
    matx_factor_sparse_d_i8_cholmod_t* ptr = (matx_factor_sparse_d_i8_cholmod_t*) F->reserved;
    F->reserved = NULL;
    if (ptr->rhs_work)
        cholmod_l_free_dense(&ptr->rhs_work, &ptr->common);
    if (ptr->solution_work)
        cholmod_l_free_dense(&ptr->solution_work, &ptr->common);
    if (ptr->y_work)
        cholmod_l_free_dense(&ptr->y_work, &ptr->common);
    if (ptr->e_work)
        cholmod_l_free_dense(&ptr->e_work, &ptr->common);
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
    st = coo_to_csc_d_i8_value_remap(A);
    if (st != MATX_OK) {
        MATX_ERROR("coo_to_csc_d_i8_value_remap error:%d", st);
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

    cholmod_sparse A_chol_struct;
    memset(&A_chol_struct, 0, sizeof(A_chol_struct));
    A_chol_struct.nrow   = n;
    A_chol_struct.ncol   = n;
    A_chol_struct.nzmax  = nnz;
    A_chol_struct.p      = (void*) A->handle_csc->col_ptr;
    A_chol_struct.i      = (void*) A->handle_csc->row_ind;
    A_chol_struct.x      = (void*) A->handle_csc->values;
    A_chol_struct.stype  = -1;             /* lower triangular */
    A_chol_struct.itype  = CHOLMOD_LONG;
    A_chol_struct.xtype  = CHOLMOD_REAL;
    A_chol_struct.dtype  = CHOLMOD_DOUBLE;
    A_chol_struct.sorted = 1;
    A_chol_struct.packed = 1;

    F->L = cholmod_l_analyze(&A_chol_struct, &F->common);
    if (!F->L) {
        MATX_ERROR("%s: cholmod_l_analyze failed", __func__);
        cholmod_l_finish(&F->common);
        matx_free(alloc, F);
        return MATX_ERR_INTERNAL;
    }

    cholmod_l_factorize(&A_chol_struct, F->L, &F->common);
    if (F->common.status != CHOLMOD_OK) {
        MATX_ERROR("%s: cholmod_l_factorize failed, status=%d", __func__, F->common.status);
        ss_factor_chol_csc_d_i8_destroy(alloc, out_F);
        out_F->reserved = NULL;
        return MATX_ERR_INTERNAL;
    }

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

    if (!ptr->rhs_work) {
        ptr->rhs_work = cholmod_l_allocate_dense(n, 1, n, CHOLMOD_REAL, &ptr->common);
        if (!ptr->rhs_work) {
            MATX_ERROR("%s: out of memory", __func__);
            return MATX_ERR_OUT_OF_MEMORY;
        }
    }
    memcpy(ptr->rhs_work->x, b, (size_t) n * sizeof(matx_double));

    if (!cholmod_l_solve2(CHOLMOD_A,
                          ptr->L,
                          ptr->rhs_work,
                          NULL,
                          &ptr->solution_work,
                          NULL,
                          &ptr->y_work,
                          &ptr->e_work,
                          &ptr->common)
        || !ptr->solution_work || ptr->common.status < CHOLMOD_OK) {
        MATX_ERROR("%s: cholmod_l_solve2 failed, status=%d", __func__, ptr->common.status);
        return MATX_ERR_INTERNAL;
    }
    memcpy(x, ptr->solution_work->x, (size_t) n * sizeof(matx_double));
    return MATX_OK;
}

// ---- Sparse Cholesky via CHOLMOD (complex HPD) ----

typedef struct matx_factor_sparse_z_i8_cholmod
{
    cholmod_factor* L;
    cholmod_dense* rhs_work;
    cholmod_dense* solution_work;
    cholmod_dense* y_work;
    cholmod_dense* e_work;
    cholmod_common common;
    matx_int64_t n;
} matx_factor_sparse_z_i8_cholmod_t;

static void ss_factor_chol_csc_z_i8_destroy(const matx_alloc_t* alloc, matx_factor_sparse_z_i8_t* F)
{
    if (!F || !F->reserved)
        return;
    matx_factor_sparse_z_i8_cholmod_t* ptr = (matx_factor_sparse_z_i8_cholmod_t*) F->reserved;
    F->reserved = NULL;
    if (ptr->rhs_work)
        cholmod_l_free_dense(&ptr->rhs_work, &ptr->common);
    if (ptr->solution_work)
        cholmod_l_free_dense(&ptr->solution_work, &ptr->common);
    if (ptr->y_work)
        cholmod_l_free_dense(&ptr->y_work, &ptr->common);
    if (ptr->e_work)
        cholmod_l_free_dense(&ptr->e_work, &ptr->common);
    if (ptr->L)
        cholmod_l_free_factor(&ptr->L, &ptr->common);
    cholmod_l_finish(&ptr->common);
    matx_free(alloc, ptr);
}

static matx_status_t ss_factor_chol_csc_z_i8(const matx_alloc_t* alloc, matx_coo_z_i8_t A,
                                              matx_factor_sparse_z_i8_t* out_F)
{
    if (!A || !out_F) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (A->nrows != A->ncols) {
        MATX_ERROR("%s: matrix must be square", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    matx_status_t st = coo_to_csc_z_i8(A);
    if (st != MATX_OK) {
        MATX_ERROR("coo_to_csc_z_i8 error:%d", st);
        return st;
    }
    st = coo_to_csc_z_i8_value_remap(A);
    if (st != MATX_OK) {
        MATX_ERROR("coo_to_csc_z_i8_value_remap error:%d", st);
        return st;
    }

    const matx_int64_t n = A->nrows;
    const matx_int64_t nnz = A->handle_csc->nnz;

    matx_factor_sparse_z_i8_cholmod_t* F = (matx_factor_sparse_z_i8_cholmod_t*) matx_malloc(alloc, sizeof(*F));
    if (!F) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(F, 0, sizeof(*F));
    F->n = n;

    cholmod_l_start(&F->common);

    cholmod_sparse A_chol_struct;
    memset(&A_chol_struct, 0, sizeof(A_chol_struct));
    A_chol_struct.nrow   = n;
    A_chol_struct.ncol   = n;
    A_chol_struct.nzmax  = nnz;
    A_chol_struct.p      = (void*) A->handle_csc->col_ptr;
    A_chol_struct.i      = (void*) A->handle_csc->row_ind;
    A_chol_struct.x      = (void*) A->handle_csc->values;
    A_chol_struct.stype  = -1;             /* lower triangular */
    A_chol_struct.itype  = CHOLMOD_LONG;
    A_chol_struct.xtype  = CHOLMOD_COMPLEX;
    A_chol_struct.dtype  = CHOLMOD_DOUBLE;
    A_chol_struct.sorted = 1;
    A_chol_struct.packed = 1;

    F->L = cholmod_l_analyze(&A_chol_struct, &F->common);
    if (!F->L) {
        MATX_ERROR("%s: cholmod_l_analyze failed", __func__);
        cholmod_l_finish(&F->common);
        matx_free(alloc, F);
        return MATX_ERR_INTERNAL;
    }

    cholmod_l_factorize(&A_chol_struct, F->L, &F->common);
    if (F->common.status != CHOLMOD_OK) {
        MATX_ERROR("%s: cholmod_l_factorize failed, status=%d", __func__, F->common.status);
        ss_factor_chol_csc_z_i8_destroy(alloc, out_F);
        out_F->reserved = NULL;
        return MATX_ERR_INTERNAL;
    }

    out_F->reserved = F;
    return MATX_OK;
}

static matx_status_t ss_solve_chol_csc_z_i8(const matx_alloc_t* alloc,
                                             matx_factor_sparse_z_i8_t* F,
                                             const matx_vec_z_i8_t b,
                                             matx_vec_z_i8_t x)
{
    if (!F || !F->reserved || !b || !x) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_factor_sparse_z_i8_cholmod_t* ptr = (matx_factor_sparse_z_i8_cholmod_t*) F->reserved;
    const matx_int64_t n = ptr->n;

    if (!ptr->rhs_work) {
        ptr->rhs_work = cholmod_l_allocate_dense(n, 1, n, CHOLMOD_COMPLEX, &ptr->common);
        if (!ptr->rhs_work) {
            MATX_ERROR("%s: out of memory", __func__);
            return MATX_ERR_OUT_OF_MEMORY;
        }
    }
    /* Copy strided complex input into contiguous CHOLMOD dense buffer */
    {
        matx_complex_d_t* dst = (matx_complex_d_t*) ptr->rhs_work->x;
        for (matx_int64_t i = 0; i < n; ++i)
            dst[i] = b->data[i * b->stride];
    }

    if (!cholmod_l_solve2(CHOLMOD_A,
                          ptr->L,
                          ptr->rhs_work,
                          NULL,
                          &ptr->solution_work,
                          NULL,
                          &ptr->y_work,
                          &ptr->e_work,
                          &ptr->common)
        || !ptr->solution_work || ptr->common.status < CHOLMOD_OK) {
        MATX_ERROR("%s: cholmod_l_solve2 failed, status=%d", __func__, ptr->common.status);
        return MATX_ERR_INTERNAL;
    }
    {
        matx_complex_d_t* src = (matx_complex_d_t*) ptr->solution_work->x;
        for (matx_int64_t i = 0; i < n; ++i)
            x->data[i * x->stride] = src[i];
    }
    return MATX_OK;
}

// ---- KLU numeric-only refactorization ----

static matx_status_t ss_refactor_csc_d_i8(const matx_alloc_t* alloc,
                                           matx_coo_d_i8_t A,
                                           matx_factor_sparse_d_i8_t* F)
{
    if (!A || !F || !F->reserved) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (A->nrows != A->ncols) {
        MATX_ERROR("%s: matrix must be square", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    matx_factor_sparse_d_i8_klu_t* ptr = (matx_factor_sparse_d_i8_klu_t*) F->reserved;
    if (!ptr->S) {
        MATX_ERROR("%s: no symbolic factorization available", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    matx_status_t st = coo_to_csc_d_i8_value_remap(A);
    if (st != MATX_OK) {
        MATX_ERROR("coo_to_csc_d_i8_value_remap error:%d", st);
        return st;
    }

    /* Free old numeric factorization, recompute with new values */
    klu_l_free_numeric(&ptr->N, &ptr->common);
    ptr->N = klu_l_factor(A->handle_csc->col_ptr,
                          A->handle_csc->row_ind,
                          A->handle_csc->values,
                          ptr->S,
                          &ptr->common);
    if (!ptr->N) {
        MATX_ERROR("klu_l_factor (refactor) error:%d", ptr->common.status);
        return MATX_ERR_INTERNAL;
    }
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
                  .factor_chol_csc_d_i8_destroy = &ss_factor_chol_csc_d_i8_destroy,
                  .factor_chol_csc_z_i8 = &ss_factor_chol_csc_z_i8,
                  .solve_chol_csc_z_i8 = &ss_solve_chol_csc_z_i8,
                  .factor_chol_csc_z_i8_destroy = &ss_factor_chol_csc_z_i8_destroy,
                  .refactor_csc_d_i8 = &ss_refactor_csc_d_i8}};
    return ls;
}

#else

matx_sparse_linsolve_t matx_linsolve_make_suitesparse_klu(matx_alloc_t alloc)
{
    matx_sparse_linsolve_t ls = {.kind = MATX_LINSOLVE_BACKEND_SUITESPARSE_KLU, .alloc = alloc, .vt = {0}};
    return ls;
}

#endif /* MATX_HAVE_SUITESPARSE */
