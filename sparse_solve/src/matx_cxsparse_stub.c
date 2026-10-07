#include "matx/matx_log.h"
#include "matx/matx_sparse_solve.h"
#include "matx/matx_types_internal.h"

#include <stdlib.h>
#include <complex.h>
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

#if MATX_HAVE_CXSPARSE
#include "cs.h"
#endif

typedef struct matx_factor_sparse_d_i8_cxsparse
{
#if MATX_HAVE_CXSPARSE
    cs_dl A;
    cs_dls* S;
    cs_dln* N;
    matx_int64_t n;
    matx_double* solve_work;
    atomic_bool solve_busy;
#endif
    matx_int64_t unused;
} matx_factor_sparse_d_i8_cxsparse_t;

typedef struct matx_factor_sparse_z_i8_cxsparse
{
#if MATX_HAVE_CXSPARSE
    cs_cl A;
    cs_cls* S;
    cs_cln* N;
    matx_int64_t n;
    cs_complex_t* solve_work;
    atomic_bool solve_busy;
#endif
    matx_int64_t unused;
} matx_factor_sparse_z_i8_cxsparse_t;

static void cxs_factor_csc_d_i8_destroy(const matx_alloc_t* alloc, matx_factor_sparse_d_i8_t* F)
{
    if (!F || !F->reserved)
        return;
    matx_factor_sparse_d_i8_cxsparse_t* ptr = (matx_factor_sparse_d_i8_cxsparse_t*) F->reserved;
    F->reserved = NULL;
#if MATX_HAVE_CXSPARSE
    if (ptr->N)
        cs_dl_nfree(ptr->N);
    if (ptr->S)
        cs_dl_sfree(ptr->S);
    matx_free(alloc, ptr->solve_work);
#endif
    matx_free(alloc, ptr);
}

static void cxs_factor_csc_z_i8_destroy(const matx_alloc_t* alloc, matx_factor_sparse_z_i8_t* F)
{
    if (!F || !F->reserved)
        return;
    matx_factor_sparse_z_i8_cxsparse_t* ptr = (matx_factor_sparse_z_i8_cxsparse_t*) F->reserved;
    F->reserved = NULL;
#if MATX_HAVE_CXSPARSE
    if (ptr->N)
        cs_cl_nfree(ptr->N);
    if (ptr->S)
        cs_cl_sfree(ptr->S);
    matx_free(alloc, ptr->solve_work);
#endif
    matx_free(alloc, ptr);
}

static matx_status_t cxs_factor_csc_d_i8(const matx_alloc_t* alloc, matx_coo_d_i8_t A, matx_factor_sparse_d_i8_t* out_F)
{
    if (!A || !out_F) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
#if !MATX_HAVE_CXSPARSE
    (void) alloc;
    (void) A;
    (void) out_F;
    MATX_ERROR("%s: operation not supported", __func__);
    return MATX_ERR_NOT_SUPPORTED;
#else
    if (out_F && out_F->reserved)
        cxs_factor_csc_d_i8_destroy(alloc, out_F);
    if (A->nrows != A->ncols || A->nrows <= 0) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    matx_status_t st = coo_to_csc_d_i8(A);
    if (st != MATX_OK)
        return st;
    st = coo_to_csc_d_i8_value_remap(A);
    if (st != MATX_OK)
        return st;

    matx_factor_sparse_d_i8_cxsparse_t* F = (matx_factor_sparse_d_i8_cxsparse_t*) matx_malloc(alloc,
                                                                                               sizeof(*F));
    if (!F) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(F, 0, sizeof(*F));
    atomic_init(&F->solve_busy, 0);
    out_F->reserved = F;
    F->n = A->nrows;

    F->A.nzmax = A->handle_csc->nnz;
    F->A.m = A->nrows;
    F->A.n = A->ncols;
    F->A.p = A->handle_csc->col_ptr;
    F->A.i = A->handle_csc->row_ind;
    F->A.x = A->handle_csc->values;
    F->A.nz = -1;

    F->S = cs_dl_sqr(2, &F->A, 0);
    if (!F->S) {
        MATX_ERROR("cs_dl_sqr failed");
        cxs_factor_csc_d_i8_destroy(alloc, out_F);
        return MATX_ERR_INTERNAL;
    }

    F->N = cs_dl_lu(&F->A, F->S, 1e-12);
    if (!F->N) {
        MATX_ERROR("cs_dl_lu failed");
        cxs_factor_csc_d_i8_destroy(alloc, out_F);
        return MATX_ERR_INTERNAL;
    }
    if ((uint64_t) F->n <= SIZE_MAX / sizeof(matx_double))
        F->solve_work = (matx_double*) matx_malloc(alloc,
                                                  (size_t) F->n * sizeof(matx_double));
    return MATX_OK;
#endif
}

static matx_status_t cxs_solve_csc_d_i8(const matx_alloc_t* alloc,
                                        matx_factor_sparse_d_i8_t* F,
                                        const matx_double* b,
                                        matx_double* x)
{
    if (!F || !F->reserved || !b || !x) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
#if !MATX_HAVE_CXSPARSE
    (void) alloc;
    (void) F;
    (void) b;
    (void) x;
    MATX_ERROR("%s: operation not supported", __func__);
    return MATX_ERR_NOT_SUPPORTED;
#else
    matx_factor_sparse_d_i8_cxsparse_t* ptr = (matx_factor_sparse_d_i8_cxsparse_t*) F->reserved;
    const int use_workspace = ptr->solve_work
        && !atomic_exchange_explicit(&ptr->solve_busy, 1, memory_order_acquire);
    matx_double* y = use_workspace ? ptr->solve_work
        : (matx_double*) matx_malloc(alloc, sizeof(matx_double) * (size_t) ptr->n);
    if (!y) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    const int ok = cs_dl_ipvec(ptr->N->pinv, b, y, ptr->n)
        && cs_dl_lsolve(ptr->N->L, y)
        && cs_dl_usolve(ptr->N->U, y)
        && cs_dl_ipvec(ptr->S->q, y, x, ptr->n);

    if (use_workspace)
        atomic_store_explicit(&ptr->solve_busy, 0, memory_order_release);
    else
        matx_free(alloc, y);
    if (!ok) {
        MATX_ERROR("CXSparse solve failed");
        return MATX_ERR_INTERNAL;
    }
    return MATX_OK;
#endif
}

static matx_status_t cxs_factor_csc_z_i8(const matx_alloc_t* alloc, matx_coo_z_i8_t A, matx_factor_sparse_z_i8_t* out_F)
{
    if (!A || !out_F) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
#if !MATX_HAVE_CXSPARSE
    (void) alloc;
    (void) A;
    (void) out_F;
    MATX_ERROR("%s: operation not supported", __func__);
    return MATX_ERR_NOT_SUPPORTED;
#else
    if (out_F && out_F->reserved)
        cxs_factor_csc_z_i8_destroy(alloc, out_F);
    if (A->nrows != A->ncols || A->nrows <= 0) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    matx_status_t st = coo_to_csc_z_i8(A);
    if (st != MATX_OK)
        return st;
    st = coo_to_csc_z_i8_value_remap(A);
    if (st != MATX_OK)
        return st;

    matx_factor_sparse_z_i8_cxsparse_t* F = (matx_factor_sparse_z_i8_cxsparse_t*) matx_malloc(alloc,
                                                                                               sizeof(*F));
    if (!F) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(F, 0, sizeof(*F));
    atomic_init(&F->solve_busy, 0);
    out_F->reserved = F;
    F->n = A->nrows;

    F->A.nzmax = A->handle_csc->nnz;
    F->A.m = A->nrows;
    F->A.n = A->ncols;
    F->A.p = A->handle_csc->col_ptr;
    F->A.i = A->handle_csc->row_ind;
    F->A.x = (cs_complex_t*) A->handle_csc->values;
    F->A.nz = -1;

    F->S = cs_cl_sqr(2, &F->A, 0);
    if (!F->S) {
        MATX_ERROR("cs_cl_sqr failed");
        cxs_factor_csc_z_i8_destroy(alloc, out_F);
        return MATX_ERR_INTERNAL;
    }

    F->N = cs_cl_lu(&F->A, F->S, 1e-12);
    if (!F->N) {
        MATX_ERROR("cs_cl_lu failed");
        cxs_factor_csc_z_i8_destroy(alloc, out_F);
        return MATX_ERR_INTERNAL;
    }
    if ((uint64_t) F->n <= SIZE_MAX / (3 * sizeof(cs_complex_t)))
        F->solve_work = (cs_complex_t*) matx_malloc(
            alloc, (size_t) F->n * 3 * sizeof(cs_complex_t));
    return MATX_OK;
#endif
}

static matx_status_t cxs_solve_csc_z_i8(const matx_alloc_t* alloc,
                                        matx_factor_sparse_z_i8_t* F,
                                        const matx_vec_z_i8_t b,
                                        matx_vec_z_i8_t x)
{
    if (!F || !F->reserved || !b || !x || !b->data || !x->data
        || b->stride <= 0 || x->stride <= 0) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
#if !MATX_HAVE_CXSPARSE
    (void) alloc;
    (void) F;
    (void) b;
    (void) x;
    MATX_ERROR("%s: operation not supported", __func__);
    return MATX_ERR_NOT_SUPPORTED;
#else
    matx_factor_sparse_z_i8_cxsparse_t* ptr = (matx_factor_sparse_z_i8_cxsparse_t*) F->reserved;
    if (b->n < ptr->n || x->n < ptr->n
        || (uint64_t) ptr->n > SIZE_MAX / (3 * sizeof(cs_complex_t))) {
        MATX_ERROR("%s: vector length mismatch", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const int use_workspace = ptr->solve_work
        && !atomic_exchange_explicit(&ptr->solve_busy, 1, memory_order_acquire);
    cs_complex_t* work = use_workspace ? ptr->solve_work
        : (cs_complex_t*) matx_malloc(alloc,
                                      sizeof(cs_complex_t) * 3 * (size_t) ptr->n);
    if (!work) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    cs_complex_t* packed_rhs = work;
    cs_complex_t* y = work + ptr->n;
    cs_complex_t* packed_solution = work + 2 * ptr->n;
    const cs_complex_t* solve_rhs = (const cs_complex_t*) b->data;
    cs_complex_t* solve_output = (cs_complex_t*) x->data;
    if (b->stride != 1) {
        for (matx_int64_t i = 0; i < ptr->n; ++i)
            packed_rhs[i] = b->data[i * b->stride].real
                + b->data[i * b->stride].imag * I;
        solve_rhs = packed_rhs;
    }
    if (x->stride != 1)
        solve_output = packed_solution;

    const int ok = cs_cl_ipvec(ptr->N->pinv, (cs_complex_t*) solve_rhs, y, ptr->n)
        && cs_cl_lsolve(ptr->N->L, y)
        && cs_cl_usolve(ptr->N->U, y)
        && cs_cl_ipvec(ptr->S->q, y, solve_output, ptr->n);

    if (ok && x->stride != 1) {
        for (matx_int64_t i = 0; i < ptr->n; ++i) {
            x->data[i * x->stride].real = creal(solve_output[i]);
            x->data[i * x->stride].imag = cimag(solve_output[i]);
        }
    }
    if (use_workspace)
        atomic_store_explicit(&ptr->solve_busy, 0, memory_order_release);
    else
        matx_free(alloc, work);
    if (!ok) {
        MATX_ERROR("CXSparse complex solve failed");
        return MATX_ERR_INTERNAL;
    }
    return MATX_OK;
#endif
}

matx_sparse_linsolve_t matx_linsolve_make_cxsparse(matx_alloc_t alloc)
{
    matx_sparse_linsolve_t ls = {.kind = MATX_LINSOLVE_BACKEND_CXSPARSE,
                                 .alloc = alloc,
                                 .vt = {.factor_csc_d_i8 = &cxs_factor_csc_d_i8,
                                        .solve_csc_d_i8 = &cxs_solve_csc_d_i8,
                                        .factor_csc_d_i8_destroy = &cxs_factor_csc_d_i8_destroy,
                                        .factor_csc_z_i8 = &cxs_factor_csc_z_i8,
                                        .solve_csc_z_i8 = &cxs_solve_csc_z_i8,
                                        .factor_csc_z_i8_destroy = &cxs_factor_csc_z_i8_destroy}};
    return ls;
}
