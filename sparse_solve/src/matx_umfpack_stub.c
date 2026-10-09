#include "matx/matx_log.h"
#include "matx/matx_sparse_solve.h"
#include "matx/matx_types_internal.h"

#include <stdlib.h>
#include <stdatomic.h>
#include <string.h>

#if MATX_HAVE_UMFPACK
#include "umfpack.h"
#endif

/**
 * @brief Double-precision real sparse matrix factorization handle for UMFPACK
 */
typedef struct matx_factor_sparse_d_i8_umfpack
{
    void* symbolic;
    void* numeric;
    matx_int64_t n;
    matx_int64_t nnz;
    matx_int64_t* Ap;
    matx_int64_t* Ai;
    matx_double* Ax;
    matx_int64_t* solve_iwork;
    matx_double* solve_work;
    atomic_bool solve_busy;
} matx_factor_sparse_d_i8_umfpack_t;

/**
 * @brief Double-precision complex sparse matrix factorization handle for UMFPACK
 */
typedef struct matx_factor_sparse_z_i8_umfpack
{
#if MATX_HAVE_UMFPACK
    void* symbolic;
    void* numeric;
    matx_int64_t n;
    matx_int64_t nnz;
    matx_int64_t* Ap;
    matx_int64_t* Ai;
    matx_double* Ax;
    matx_double* Az;
    matx_int64_t* solve_iwork;
    matx_double* solve_work;
    matx_double* solve_vectors;
    atomic_bool solve_busy;
#endif
    int unused;
} matx_factor_sparse_z_i8_umfpack_t;

/**
 * @brief Destroy real f64 sparse factorization handle
 * @param F Factorization handle
 */
static void umf_factor_csc_d_i8_destroy(const matx_alloc_t* alloc, matx_factor_sparse_d_i8_t* F)
{
    if (!F || !F->reserved)
        return;
    matx_factor_sparse_d_i8_umfpack_t* ptr = (matx_factor_sparse_d_i8_umfpack_t*) F->reserved;
    F->reserved = NULL;
#if MATX_HAVE_UMFPACK
    if (ptr->numeric)
        umfpack_dl_free_numeric(&ptr->numeric);
    if (ptr->symbolic)
        umfpack_dl_free_symbolic(&ptr->symbolic);
#endif
    matx_free(alloc, ptr->Ap);
    matx_free(alloc, ptr->Ai);
    matx_free(alloc, ptr->Ax);
    matx_free(alloc, ptr->solve_iwork);
    matx_free(alloc, ptr->solve_work);
    matx_free(alloc, ptr);
}

/**
 * @brief Perform LU factorization for real CSC matrix using UMFPACK
 * @param A Input COO matrix
 * @param out_F Output factorization handle
 * @return matx_status_t
 */
static matx_status_t umf_factor_csc_d_i8(const matx_alloc_t* alloc, matx_coo_d_i8_t A, matx_factor_sparse_d_i8_t* out_F)
{
    if (!A || !out_F) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
#if !MATX_HAVE_UMFPACK
    (void) alloc;
    (void) A;
    (void) out_F;
    MATX_ERROR("%s: operation not supported", __func__);
    return MATX_ERR_NOT_SUPPORTED;
#else
    if (out_F->reserved)
        umf_factor_csc_d_i8_destroy(alloc, out_F);
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

    matx_factor_sparse_d_i8_umfpack_t* F = (matx_factor_sparse_d_i8_umfpack_t*) matx_malloc(alloc,
                                                                                         sizeof(*F));
    if (!F) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(F, 0, sizeof(*F));
    atomic_init(&F->solve_busy, 0);

    out_F->reserved = F;
    F->n = A->nrows;
    F->nnz = A->handle_csc->nnz;

    F->Ap = A->handle_csc->col_ptr;
    F->Ai = A->handle_csc->row_ind;
    F->Ax = A->handle_csc->values;
    A->handle_csc->col_ptr = NULL;
    A->handle_csc->row_ind = NULL;
    A->handle_csc->values = NULL;

    int status = umfpack_dl_symbolic((int64_t) F->n,
                                     (int64_t) F->n,
                                     (const int64_t*) F->Ap,
                                     (const int64_t*) F->Ai,
                                     (const double*) F->Ax,
                                     &F->symbolic,
                                     NULL,
                                     NULL);
    if (status != UMFPACK_OK) {
        MATX_ERROR("umfpack_dl_symbolic failed status=%d", status);
        umf_factor_csc_d_i8_destroy(alloc, out_F);
        return MATX_ERR_INTERNAL;
    }

    status = umfpack_dl_numeric((const int64_t*) F->Ap,
                                (const int64_t*) F->Ai,
                                (const double*) F->Ax,
                                F->symbolic,
                                &F->numeric,
                                NULL,
                                NULL);
    if (status != UMFPACK_OK) {
        MATX_ERROR("umfpack_dl_numeric failed status=%d", status);
        umf_factor_csc_d_i8_destroy(alloc, out_F);
        return MATX_ERR_INTERNAL;
    }

    /* Retain UMFPACK's documented wsolve workspace across repeated solves. */
    if ((uint64_t) F->n <= SIZE_MAX / (5 * sizeof(matx_double))
        && (uint64_t) F->n <= SIZE_MAX / sizeof(matx_int64_t)) {
        F->solve_iwork = (matx_int64_t*) matx_malloc(
            alloc, (size_t) F->n * sizeof(matx_int64_t));
        F->solve_work = (matx_double*) matx_malloc(
            alloc, (size_t) F->n * 5 * sizeof(matx_double));
        if (!F->solve_iwork || !F->solve_work) {
            matx_free(alloc, F->solve_iwork);
            matx_free(alloc, F->solve_work);
            F->solve_iwork = NULL;
            F->solve_work = NULL;
        }
    }

    return MATX_OK;
#endif
}

/**
 * @brief Solve real linear system Ax = b using pre-factored LU
 * @param F Factor handle
 * @param b RHS vector
 * @param x Solution vector
 * @return matx_status_t
 */
static matx_status_t umf_solve_csc_d_i8(const matx_alloc_t* alloc,
                                        matx_factor_sparse_d_i8_t* F,
                                        const matx_double* b,
                                        matx_double* x)
{
    if (!F || !F->reserved || !b || !x) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
#if !MATX_HAVE_UMFPACK
    (void) alloc;
    (void) F;
    (void) b;
    (void) x;
    MATX_ERROR("%s: operation not supported", __func__);
    return MATX_ERR_NOT_SUPPORTED;
#else
    matx_factor_sparse_d_i8_umfpack_t* ptr = (matx_factor_sparse_d_i8_umfpack_t*) F->reserved;
    int status;
    if (ptr->solve_iwork && ptr->solve_work && x != b
        && !atomic_exchange_explicit(&ptr->solve_busy, 1, memory_order_acquire)) {
        status = umfpack_dl_wsolve(UMFPACK_A,
                                   (const int64_t*) ptr->Ap,
                                   (const int64_t*) ptr->Ai,
                                   (const double*) ptr->Ax,
                                   (double*) x,
                                   (const double*) b,
                                   ptr->numeric,
                                   NULL,
                                   NULL,
                                   (int64_t*) ptr->solve_iwork,
                                   (double*) ptr->solve_work);
        atomic_store_explicit(&ptr->solve_busy, 0, memory_order_release);
    } else {
        status = umfpack_dl_solve(UMFPACK_A,
                                  (const int64_t*) ptr->Ap,
                                  (const int64_t*) ptr->Ai,
                                  (const double*) ptr->Ax,
                                  (double*) x,
                                  (const double*) b,
                                  ptr->numeric,
                                  NULL,
                                  NULL);
    }
    if (status != UMFPACK_OK) {
        MATX_ERROR("umfpack_dl_solve failed status=%d", status);
        return MATX_ERR_INTERNAL;
    }
    return MATX_OK;
#endif
}

// ==================== Complex c64 Implementation ====================

/**
 * @brief Destroy complex c64 sparse factorization handle
 * @param F Complex factorization handle
 */
static void umf_factor_csc_z_i8_destroy(const matx_alloc_t* alloc, matx_factor_sparse_z_i8_t* F)
{
    if (!F || !F->reserved)
        return;
    matx_factor_sparse_z_i8_umfpack_t* ptr = (matx_factor_sparse_z_i8_umfpack_t*) F->reserved;
    F->reserved = NULL;
#if MATX_HAVE_UMFPACK
    if (ptr->numeric)
        umfpack_zl_free_numeric(&ptr->numeric);
    if (ptr->symbolic)
        umfpack_zl_free_symbolic(&ptr->symbolic);
#endif
    #if MATX_HAVE_UMFPACK
    matx_free(alloc, ptr->Ap);
    matx_free(alloc, ptr->Ai);
    matx_free(alloc, ptr->Ax);
    matx_free(alloc, ptr->Az);
    matx_free(alloc, ptr->solve_iwork);
    matx_free(alloc, ptr->solve_work);
    matx_free(alloc, ptr->solve_vectors);
#endif
    matx_free(alloc, ptr);
}

/**
 * @brief Perform LU factorization for complex CSC matrix using UMFPACK
 * @param A Input complex COO matrix
 * @param out_F Output complex factorization handle
 * @return matx_status_t
 */
static matx_status_t umf_factor_csc_z_i8(const matx_alloc_t* alloc, matx_coo_z_i8_t A, matx_factor_sparse_z_i8_t* out_F)
{
    if (!A || !out_F) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
#if !MATX_HAVE_UMFPACK
    (void) alloc;
    (void) A;
    (void) out_F;
    MATX_ERROR("%s: operation not supported", __func__);
    return MATX_ERR_NOT_SUPPORTED;
#else
    if (out_F)
        umf_factor_csc_z_i8_destroy(alloc, out_F);
    if (A->nrows != A->ncols || A->nrows <= 0) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    // Convert complex COO to CSC format
    matx_status_t st = coo_to_csc_z_i8(A);
    if (st != MATX_OK)
        return st;
    st = coo_to_csc_z_i8_value_remap(A);
    if (st != MATX_OK)
        return st;

    // Allocate factorization handle
    matx_factor_sparse_z_i8_umfpack_t* F = (matx_factor_sparse_z_i8_umfpack_t*) matx_malloc(alloc,
                                                                                         sizeof(*F));
    if (!F) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(F, 0, sizeof(*F));
    atomic_init(&F->solve_busy, 0);
    out_F->reserved = F;
    F->n = A->nrows;
    F->nnz = A->handle_csc->nnz;
    // Transfer index arrays from CSC handle
    F->Ap = A->handle_csc->col_ptr;
    F->Ai = A->handle_csc->row_ind;
    A->handle_csc->col_ptr = NULL;
    A->handle_csc->row_ind = NULL;

    // Allocate and interleave complex values for UMFPACK (split real/imag)
    F->Ax = (matx_double*) matx_malloc(alloc, sizeof(matx_double) * (size_t) F->nnz);
    F->Az = (matx_double*) matx_malloc(alloc, sizeof(matx_double) * (size_t) F->nnz);
    if (!F->Az || !F->Ax) {
        umf_factor_csc_z_i8_destroy(alloc, out_F);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    for (matx_int64_t i = 0; i < F->nnz; i++) {
        F->Ax[i] = A->handle_csc->values[i].real;
        F->Az[i] = A->handle_csc->values[i].imag;
    }

    // Complex symbolic analysis
    int status = umfpack_zl_symbolic((int64_t) F->n,
                                     (int64_t) F->n,
                                     (const int64_t*) F->Ap,
                                     (const int64_t*) F->Ai,
                                     (const double*) F->Ax,
                                     (const double*) F->Az,
                                     &F->symbolic,
                                     NULL,
                                     NULL);
    if (status != UMFPACK_OK) {
        MATX_ERROR("umfpack_zl_symbolic failed status=%d", status);
        umf_factor_csc_z_i8_destroy(alloc, out_F);
        return MATX_ERR_INTERNAL;
    }

    // Complex numeric factorization
    status = umfpack_zl_numeric((const int64_t*) F->Ap,
                                (const int64_t*) F->Ai,
                                (const double*) F->Ax,
                                (const double*) F->Az,
                                F->symbolic,
                                &F->numeric,
                                NULL,
                                NULL);
    if (status != UMFPACK_OK) {
        MATX_ERROR("umfpack_zl_numeric failed status=%d", status);
        umf_factor_csc_z_i8_destroy(alloc, out_F);
        return MATX_ERR_INTERNAL;
    }

    /* Four split-complex vectors plus UMFPACK's reusable worst-case workspace. */
    if ((uint64_t) F->n <= SIZE_MAX / (10 * sizeof(matx_double))
        && (uint64_t) F->n <= SIZE_MAX / (4 * sizeof(matx_int64_t))) {
        F->solve_iwork = (matx_int64_t*) matx_malloc(
            alloc, (size_t) F->n * 4 * sizeof(matx_int64_t));
        F->solve_work = (matx_double*) matx_malloc(
            alloc, (size_t) F->n * 10 * sizeof(matx_double));
        F->solve_vectors = (matx_double*) matx_malloc(
            alloc, (size_t) F->n * 4 * sizeof(matx_double));
        if (!F->solve_iwork || !F->solve_work || !F->solve_vectors) {
            matx_free(alloc, F->solve_iwork);
            matx_free(alloc, F->solve_work);
            matx_free(alloc, F->solve_vectors);
            F->solve_iwork = NULL;
            F->solve_work = NULL;
            F->solve_vectors = NULL;
        }
    }

    return MATX_OK;
#endif
}

/**
 * @brief Solve complex linear system Ax = b using pre-factored LU
 * @param F Complex factor handle
 * @param b Complex RHS vector
 * @param x Complex solution vector
 * @return matx_status_t
 */
static matx_status_t umf_solve_csc_z_i8(const matx_alloc_t* alloc,
                                        matx_factor_sparse_z_i8_t* F,
                                        const matx_vec_z_i8_t b,
                                        matx_vec_z_i8_t x)
{
    if (!F || !F->reserved || !b || !x || !b->data || !x->data
        || b->stride <= 0 || x->stride <= 0) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
#if !MATX_HAVE_UMFPACK
    (void) alloc;
    (void) F;
    (void) b;
    (void) x;
    MATX_ERROR("%s: operation not supported", __func__);
    return MATX_ERR_NOT_SUPPORTED;
#else
    matx_factor_sparse_z_i8_umfpack_t* ptr = (matx_factor_sparse_z_i8_umfpack_t*) F->reserved;
    if (b->n < ptr->n || x->n < ptr->n
        || (uint64_t) ptr->n > SIZE_MAX / (4 * sizeof(matx_double))) {
        MATX_ERROR("%s: vector length mismatch", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    const int use_workspace = ptr->solve_iwork && ptr->solve_work && ptr->solve_vectors
        && !atomic_exchange_explicit(&ptr->solve_busy, 1, memory_order_acquire);
    const size_t vector_bytes = (size_t) ptr->n * sizeof(matx_double);
    matx_double* vectors = use_workspace
        ? ptr->solve_vectors
        : (matx_double*) matx_malloc(alloc, vector_bytes * 4);
    if (!vectors) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    matx_double* b_umf_x = vectors;
    matx_double* b_umf_z = vectors + ptr->n;
    matx_double* x_umf_x = vectors + 2 * ptr->n;
    matx_double* x_umf_z = vectors + 3 * ptr->n;

    for (matx_int64_t i = 0; i < ptr->n; ++i) {
        const matx_complex_d_t value = b->data[i * b->stride];
        b_umf_x[i] = value.real;
        b_umf_z[i] = value.imag;
    }

    int status;
    if (use_workspace) {
        status = umfpack_zl_wsolve(UMFPACK_A,
                                   (const int64_t*) ptr->Ap,
                                   (const int64_t*) ptr->Ai,
                                   (const double*) ptr->Ax,
                                   (const double*) ptr->Az,
                                   x_umf_x,
                                   x_umf_z,
                                   b_umf_x,
                                   b_umf_z,
                                   ptr->numeric,
                                   NULL,
                                   NULL,
                                   (int64_t*) ptr->solve_iwork,
                                   (double*) ptr->solve_work);
    } else {
        status = umfpack_zl_solve(UMFPACK_A,
                                  (const int64_t*) ptr->Ap,
                                  (const int64_t*) ptr->Ai,
                                  (const double*) ptr->Ax,
                                  (const double*) ptr->Az,
                                  x_umf_x,
                                  x_umf_z,
                                  b_umf_x,
                                  b_umf_z,
                                  ptr->numeric,
                                  NULL,
                                  NULL);
    }
    if (status == UMFPACK_OK) {
        for (matx_int64_t i = 0; i < ptr->n; ++i) {
            x->data[i * x->stride].real = x_umf_x[i];
            x->data[i * x->stride].imag = x_umf_z[i];
        }
    }

    if (use_workspace)
        atomic_store_explicit(&ptr->solve_busy, 0, memory_order_release);
    else
        matx_free(alloc, vectors);

    if (status != UMFPACK_OK) {
        MATX_ERROR("umfpack_zl_solve failed status=%d", status);
        return MATX_ERR_INTERNAL;
    }
    return MATX_OK;
#endif
}

// ---- UMFPACK numeric-only refactorization ----

static matx_status_t umf_refactor_csc_d_i8(const matx_alloc_t* alloc,
                                            matx_coo_d_i8_t A,
                                            matx_factor_sparse_d_i8_t* F)
{
    if (!A || !F || !F->reserved) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
#if !MATX_HAVE_UMFPACK
    (void) alloc;
    MATX_ERROR("%s: operation not supported", __func__);
    return MATX_ERR_NOT_SUPPORTED;
#else
    if (A->nrows != A->ncols || A->nrows <= 0) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    matx_factor_sparse_d_i8_umfpack_t* ptr = (matx_factor_sparse_d_i8_umfpack_t*) F->reserved;
    if (!ptr->symbolic) {
        MATX_ERROR("%s: no symbolic factorization available", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    matx_status_t st = coo_to_csc_d_i8_value_remap(A);
    if (st != MATX_OK) {
        MATX_ERROR("coo_to_csc_d_i8_value_remap error:%d", st);
        return st;
    }

    if (ptr->numeric) {
        umfpack_dl_free_numeric(&ptr->numeric);
        ptr->numeric = NULL;
    }
    ptr->Ax = A->handle_csc->values;

    int status = umfpack_dl_numeric((const int64_t*) ptr->Ap,
                                    (const int64_t*) ptr->Ai,
                                    (const double*) ptr->Ax,
                                    ptr->symbolic,
                                    &ptr->numeric,
                                    NULL,
                                    NULL);
    if (status != UMFPACK_OK) {
        MATX_ERROR("umfpack_dl_numeric (refactor) failed status=%d", status);
        return MATX_ERR_INTERNAL;
    }
    return MATX_OK;
#endif
}

/**
 * @brief Create UMFPACK linear solver backend (real + complex)
 * @return Fully initialized sparse linear solver interface
 */
matx_sparse_linsolve_t matx_linsolve_make_umfpack(matx_alloc_t alloc)
{
    matx_sparse_linsolve_t ls = {.kind = MATX_LINSOLVE_BACKEND_UMFPACK,
                                 .alloc = alloc,
                                 .vt = {.factor_csc_d_i8 = &umf_factor_csc_d_i8,
                                        .solve_csc_d_i8 = &umf_solve_csc_d_i8,
                                        .factor_csc_d_i8_destroy = &umf_factor_csc_d_i8_destroy,
                                        .factor_csc_z_i8 = &umf_factor_csc_z_i8,
                                        .solve_csc_z_i8 = &umf_solve_csc_z_i8,
                                        .factor_csc_z_i8_destroy = &umf_factor_csc_z_i8_destroy,
                                        .refactor_csc_d_i8 = &umf_refactor_csc_d_i8}};
    return ls;
}
