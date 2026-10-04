#include "matx/matx_log.h"
#include "matx/matx_sparse_solve.h"
#include "matx/matx_types_internal.h"

#include <stdlib.h>
#include <string.h>

#if MATX_HAVE_SUPERLU
#include "slu_ddefs.h"
#include "slu_zdefs.h"
#endif

/**
 * @brief Double precision sparse LU factorization handle
 */
typedef struct matx_factor_sparse_d_i8_slu
{
#if MATX_HAVE_SUPERLU
    SuperMatrix A;
    SuperMatrix L;
    SuperMatrix U;
    SuperMatrix B;
    SuperLUStat_t stat;
    superlu_options_t options;
    int* perm_c;
    int* perm_r;
    int* etree;
    int n;
    double* rhs;
    int stat_initialized;
#endif
    int place_holder;
} matx_factor_sparse_d_i8_slu_t;

/**
 * @brief Complex double precision sparse LU factorization handle
 */
typedef struct matx_factor_sparse_z_i8_slu
{
#if MATX_HAVE_SUPERLU
    SuperMatrix A;
    SuperMatrix L;
    SuperMatrix U;
    SuperMatrix B;
    SuperLUStat_t stat;
    superlu_options_t options;
    int* perm_c;
    int* perm_r;
    int* etree;
    int n;
    doublecomplex* rhs;
    int stat_initialized;
#endif
    int unused;
} matx_factor_sparse_z_i8_slu_t;

/**
 * @brief Destroy double precision sparse factorization handle and free resources
 * @param F Factor handle to destroy
 */
static void slu_factor_csc_d_i8_destroy(const matx_alloc_t* alloc, matx_factor_sparse_d_i8_t* F)
{
    if (!F || !F->reserved)
        return;
    matx_factor_sparse_d_i8_slu_t* ptr = (matx_factor_sparse_d_i8_slu_t*) F->reserved;
    F->reserved = NULL;
#if MATX_HAVE_SUPERLU
    if (ptr->L.Store) Destroy_SuperNode_Matrix(&ptr->L);
    if (ptr->U.Store) Destroy_CompCol_Matrix(&ptr->U);
    if (ptr->A.Store) Destroy_CompCol_Matrix(&ptr->A);
    if (ptr->B.Store) Destroy_Dense_Matrix(&ptr->B);
    else if (ptr->rhs) SUPERLU_FREE(ptr->rhs);
    if (ptr->stat_initialized) StatFree(&ptr->stat);
    matx_free(alloc, ptr->perm_c);
    matx_free(alloc, ptr->perm_r);
    matx_free(alloc, ptr->etree);
#endif
    matx_free(alloc, ptr);
}

#if MATX_HAVE_SUPERLU
static int superlu_index_fits(matx_int64_t value)
{
    if (value < 0) return 0;
    return sizeof(int_t) >= sizeof(matx_int64_t) || value <= INT_MAX;
}

static int superlu_csc_indices_fit(matx_csc_d_i8_t csc)
{
    if (!csc || !superlu_index_fits(csc->nnz)) return 0;
    for (matx_int64_t i = 0; i <= csc->ncols; ++i) {
        if (!superlu_index_fits(csc->col_ptr[i])) return 0;
    }
    for (matx_int64_t i = 0; i < csc->nnz; ++i) {
        if (!superlu_index_fits(csc->row_ind[i])) return 0;
    }
    return 1;
}

static int superlu_csc_indices_fit_z(matx_csc_z_i8_t csc)
{
    if (!csc || !superlu_index_fits(csc->nnz)) return 0;
    for (matx_int64_t i = 0; i <= csc->ncols; ++i) {
        if (!superlu_index_fits(csc->col_ptr[i])) return 0;
    }
    for (matx_int64_t i = 0; i < csc->nnz; ++i) {
        if (!superlu_index_fits(csc->row_ind[i])) return 0;
    }
    return 1;
}
#endif

/**
 * @brief Perform LU factorization for real double CSC matrix via SuperLU
 * @param A Input COO sparse matrix
 * @param out_F Output factorization handle
 * @return MATX_OK on success, error code otherwise
 */
static matx_status_t slu_factor_csc_d_i8(const matx_alloc_t* alloc, matx_coo_d_i8_t A, matx_factor_sparse_d_i8_t* out_F)
{
    if (!A || !out_F) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
#if !MATX_HAVE_SUPERLU
    (void) alloc;
    (void) A;
    (void) out_F;
    MATX_ERROR("%s: operation not supported", __func__);
    return MATX_ERR_NOT_SUPPORTED;
#else
    /* Free existing handle if allocated */
    if (out_F->reserved)
        slu_factor_csc_d_i8_destroy(alloc, out_F);

    /* Validate matrix dimension */
    if (A->nrows != A->ncols || A->nrows <= 0) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    /* Convert COO to CSC format */
    matx_status_t st = coo_to_csc_d_i8(A);
    if (st != MATX_OK)
        return st;
    st = coo_to_csc_d_i8_value_remap(A);
    if (st != MATX_OK)
        return st;
    if (A->nrows > INT_MAX
        || (sizeof(int_t) < sizeof(matx_int64_t) && A->nrows >= INT_MAX)
        || !superlu_csc_indices_fit(A->handle_csc)) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }

    /* Allocate factorization handle */
    matx_factor_sparse_d_i8_slu_t* F = (matx_factor_sparse_d_i8_slu_t*) matx_malloc(alloc, sizeof(*F));
    if (!F) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(F, 0, sizeof(*F));
    F->n = (int) A->nrows;
    out_F->reserved = F;

    /* Allocate permutation and working arrays */
    F->perm_c = (int*) matx_malloc(alloc, sizeof(int) * (size_t) F->n);
    F->perm_r = (int*) matx_malloc(alloc, sizeof(int) * (size_t) F->n);
    F->etree = (int*) matx_malloc(alloc, sizeof(int) * (size_t) F->n);
    F->rhs = doubleMalloc((size_t) F->n);
    if (F->rhs) memset(F->rhs, 0, (size_t) F->n * sizeof(double));

    if (!F->perm_c || !F->perm_r || !F->etree || !F->rhs) {
        slu_factor_csc_d_i8_destroy(alloc, out_F);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    int_t* row_ind = intMalloc((int_t) A->handle_csc->nnz);
    int_t* col_ptr = intMalloc((int_t) (F->n + 1));
    double* values = doubleMalloc((size_t) A->handle_csc->nnz);
    if (!row_ind || !col_ptr || !values) {
        if (row_ind) SUPERLU_FREE(row_ind);
        if (col_ptr) SUPERLU_FREE(col_ptr);
        if (values) SUPERLU_FREE(values);
        slu_factor_csc_d_i8_destroy(alloc, out_F);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    for (matx_int64_t i = 0; i < A->handle_csc->nnz; ++i) {
        row_ind[i] = (int_t) A->handle_csc->row_ind[i];
        values[i] = A->handle_csc->values[i];
    }
    for (matx_int64_t i = 0; i <= A->ncols; ++i) {
        col_ptr[i] = (int_t) A->handle_csc->col_ptr[i];
    }

    /* Create SuperLU CSC matrix and dense RHS matrix */
    dCreate_CompCol_Matrix(&F->A,
                           F->n,
                           F->n,
                           (int_t) A->handle_csc->nnz,
                           values,
                           row_ind,
                           col_ptr,
                           SLU_NC,
                           SLU_D,
                           SLU_GE);

    dCreate_Dense_Matrix(&F->B, F->n, 1, F->rhs, F->n, SLU_DN, SLU_D, SLU_GE);

    /* Initialize SuperLU solver options */
    set_default_options(&F->options);
    F->options.ColPerm = COLAMD;
    StatInit(&F->stat);
    F->stat_initialized = 1;

    /* Perform LU factorization with partial pivoting */
    int info = 0;
    dgssv(&F->options, &F->A, F->perm_c, F->perm_r, &F->L, &F->U, &F->B, &F->stat, &info);
    if (info != 0) {
        MATX_ERROR("dgssv factor failed info=%d", info);
        slu_factor_csc_d_i8_destroy(alloc, out_F);
        return MATX_ERR_INTERNAL;
    }

    /* Mark matrix as already factored */
    F->options.Fact = FACTORED;
    return MATX_OK;
#endif
}

/**
 * @brief Solve real sparse linear system Ax = b with pre-factored LU
 * @param F Factorization handle
 * @param b Right-hand side vector
 * @param x Solution output vector
 * @return MATX_OK on success, error code otherwise
 */
static matx_status_t slu_solve_csc_d_i8(const matx_alloc_t* alloc, matx_factor_sparse_d_i8_t* F,
                                        const matx_double* b,
                                        matx_double* x)
{
    if (!F || !F->reserved || !b || !x) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
#if !MATX_HAVE_SUPERLU
    (void) alloc;
    (void) F;
    (void) b;
    (void) x;
    MATX_ERROR("%s: operation not supported", __func__);
    return MATX_ERR_NOT_SUPPORTED;
#else
    matx_factor_sparse_d_i8_slu_t* ptr = (matx_factor_sparse_d_i8_slu_t*) F->reserved;
    /* Copy RHS to internal buffer */
    for (int i = 0; i < ptr->n; ++i)
        ptr->rhs[i] = b[i];

    /* Solve using existing LU factors */
    int info = 0;
    dgstrs(NOTRANS, &ptr->L, &ptr->U, ptr->perm_c, ptr->perm_r, &ptr->B, &ptr->stat, &info);
    if (info != 0) {
        MATX_ERROR("dgstrs solve failed info=%d", info);
        return MATX_ERR_INTERNAL;
    }

    /* Copy solution to output */
    for (int i = 0; i < ptr->n; ++i)
        x[i] = ptr->rhs[i];
    return MATX_OK;
#endif
}

/**
 * @brief Destroy complex sparse factorization handle and release all resources
 * @param F Complex factor handle to destroy
 */
static void slu_factor_csc_z_i8_destroy(const matx_alloc_t* alloc, matx_factor_sparse_z_i8_t* F)
{
    if (!F || !F->reserved)
        return;
    matx_factor_sparse_z_i8_slu_t* ptr = (matx_factor_sparse_z_i8_slu_t*) F->reserved;
    F->reserved = NULL;
#if MATX_HAVE_SUPERLU
    /* Release SuperLU internal matrices */
    if (ptr->L.Store) Destroy_SuperNode_Matrix(&ptr->L);
    if (ptr->U.Store) Destroy_CompCol_Matrix(&ptr->U);
    if (ptr->A.Store) Destroy_CompCol_Matrix(&ptr->A);
    if (ptr->B.Store) Destroy_Dense_Matrix(&ptr->B);
    else if (ptr->rhs) SUPERLU_FREE(ptr->rhs);

    /* Free solver stat and working arrays */
    if (ptr->stat_initialized) StatFree(&ptr->stat);
    matx_free(alloc, ptr->perm_c);
    matx_free(alloc, ptr->perm_r);
    matx_free(alloc, ptr->etree);
#endif
    matx_free(alloc, ptr);
}

/**
 * @brief Perform LU factorization for complex double CSC matrix via SuperLU
 * @param A Input complex COO sparse matrix
 * @param out_F Output complex factorization handle
 * @return MATX_OK on success, error code otherwise
 */
static matx_status_t slu_factor_csc_z_i8(const matx_alloc_t* alloc, matx_coo_z_i8_t A, matx_factor_sparse_z_i8_t* out_F)
{
    if (!A || !out_F) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
#if !MATX_HAVE_SUPERLU
    (void) alloc;
    (void) A;
    (void) out_F;
    MATX_ERROR("%s: operation not supported", __func__);
    return MATX_ERR_NOT_SUPPORTED;
#else
    /* Free existing handle if allocated */
    if (out_F->reserved)
        slu_factor_csc_z_i8_destroy(alloc, out_F);

    /* Validate square matrix and size limit */
    if (A->nrows != A->ncols || A->nrows <= 0) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    /* Convert complex COO to CSC format */
    matx_status_t st = coo_to_csc_z_i8(A);
    if (st != MATX_OK)
        return st;
    st = coo_to_csc_z_i8_value_remap(A);
    if (st != MATX_OK)
        return st;
    if (A->nrows > INT_MAX
        || (sizeof(int_t) < sizeof(matx_int64_t) && A->nrows >= INT_MAX)
        || !superlu_csc_indices_fit_z(A->handle_csc)) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }

    /* Allocate complex factorization handle */
    matx_factor_sparse_z_i8_slu_t* F = (matx_factor_sparse_z_i8_slu_t*) matx_malloc(alloc, sizeof(*F));
    if (!F) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(F, 0, sizeof(*F));
    F->n = (int) A->nrows;
    out_F->reserved = F;

    /* Allocate permutation and complex RHS buffer */
    F->perm_c = (int*) matx_malloc(alloc, sizeof(int) * (size_t) F->n);
    F->perm_r = (int*) matx_malloc(alloc, sizeof(int) * (size_t) F->n);
    F->etree = (int*) matx_malloc(alloc, sizeof(int) * (size_t) F->n);
    F->rhs = doublecomplexMalloc((size_t) F->n);
    if (F->rhs) memset(F->rhs, 0, (size_t) F->n * sizeof(doublecomplex));

    if (!F->perm_c || !F->perm_r || !F->etree || !F->rhs) {
        slu_factor_csc_z_i8_destroy(alloc, out_F);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    int_t* row_ind = intMalloc((int_t) A->handle_csc->nnz);
    int_t* col_ptr = intMalloc((int_t) (F->n + 1));
    doublecomplex* values = doublecomplexMalloc((size_t) A->handle_csc->nnz);
    if (!row_ind || !col_ptr || !values) {
        if (row_ind) SUPERLU_FREE(row_ind);
        if (col_ptr) SUPERLU_FREE(col_ptr);
        if (values) SUPERLU_FREE(values);
        slu_factor_csc_z_i8_destroy(alloc, out_F);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    for (matx_int64_t i = 0; i < A->handle_csc->nnz; ++i) {
        row_ind[i] = (int_t) A->handle_csc->row_ind[i];
        values[i].r = A->handle_csc->values[i].real;
        values[i].i = A->handle_csc->values[i].imag;
    }
    for (matx_int64_t i = 0; i <= A->ncols; ++i) {
        col_ptr[i] = (int_t) A->handle_csc->col_ptr[i];
    }

    /* Create SuperLU complex CSC matrix */
    zCreate_CompCol_Matrix(&F->A,
                           F->n,
                           F->n,
                           (int_t) A->handle_csc->nnz,
                           values,
                           row_ind,
                           col_ptr,
                           SLU_NC,
                           SLU_Z,
                           SLU_GE);

    /* Create complex dense RHS matrix */
    zCreate_Dense_Matrix(&F->B, F->n, 1, F->rhs, F->n, SLU_DN, SLU_Z, SLU_GE);

    /* Setup default SuperLU options */
    set_default_options(&F->options);
    F->options.ColPerm = COLAMD;
    StatInit(&F->stat);
    F->stat_initialized = 1;

    /* Complex LU factorization via zgssv */
    int info = 0;
    zgssv(&F->options, &F->A, F->perm_c, F->perm_r, &F->L, &F->U, &F->B, &F->stat, &info);
    if (info != 0) {
        MATX_ERROR("zgssv complex factor failed info=%d", info);
        slu_factor_csc_z_i8_destroy(alloc, out_F);
        return MATX_ERR_INTERNAL;
    }

    /* Mark as pre-factored */
    F->options.Fact = FACTORED;
    //*out_F = F;
    return MATX_OK;
#endif
}

/**
 * @brief Solve complex sparse linear system Ax = b with pre-factored LU
 * @param F Complex factorization handle
 * @param b Complex RHS vector
 * @param x Complex solution vector
 * @return MATX_OK on success, error code otherwise
 */
static matx_status_t slu_solve_csc_z_i8(const matx_alloc_t* alloc, matx_factor_sparse_z_i8_t* F,
                                        const matx_vec_z_i8_t b,
                                        matx_vec_z_i8_t x)
{
    if (!F || !F->reserved || !b || !x || !b->data || !x->data
        || b->stride <= 0 || x->stride <= 0) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
#if !MATX_HAVE_SUPERLU
    (void) alloc;
    (void) F;
    (void) b;
    (void) x;
    MATX_ERROR("%s: operation not supported", __func__);
    return MATX_ERR_NOT_SUPPORTED;
#else
    matx_factor_sparse_z_i8_slu_t* ptr = (matx_factor_sparse_z_i8_slu_t*) F->reserved;
    if (ptr->n <= 0 || b->n < ptr->n || x->n < ptr->n) {
        MATX_ERROR("%s: vector length mismatch", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    for (int i = 0; i < ptr->n; ++i) {
        ptr->rhs[i].r = b->data[(matx_int64_t) i * b->stride].real;
        ptr->rhs[i].i = b->data[(matx_int64_t) i * b->stride].imag;
    }
    /* Complex triangular solve with LU factors */
    int info = 0;
    zgstrs(NOTRANS, &ptr->L, &ptr->U, ptr->perm_c, ptr->perm_r, &ptr->B, &ptr->stat, &info);
    if (info != 0) {
        MATX_ERROR("zgstrs complex solve failed info=%d", info);
        return MATX_ERR_INTERNAL;
    }

    for (int i = 0; i < ptr->n; ++i) {
        x->data[(matx_int64_t) i * x->stride].real = ptr->rhs[i].r;
        x->data[(matx_int64_t) i * x->stride].imag = ptr->rhs[i].i;
    }
    return MATX_OK;
#endif
}

/**
 * @brief Create SuperLU linear solver backend interface
 * @return Initialized solver dispatch table
 */
matx_sparse_linsolve_t matx_linsolve_make_superlu(matx_alloc_t alloc)
{
    matx_sparse_linsolve_t ls = {.kind = MATX_LINSOLVE_BACKEND_SUPERLU,
                                 .alloc = alloc,
                                 .vt = {.factor_csc_d_i8 = &slu_factor_csc_d_i8,
                                        .solve_csc_d_i8 = &slu_solve_csc_d_i8,
                                        .factor_csc_d_i8_destroy = &slu_factor_csc_d_i8_destroy,
                                        .factor_csc_z_i8 = &slu_factor_csc_z_i8,
                                        .solve_csc_z_i8 = &slu_solve_csc_z_i8,
                                        .factor_csc_z_i8_destroy = &slu_factor_csc_z_i8_destroy}};
    return ls;
}
