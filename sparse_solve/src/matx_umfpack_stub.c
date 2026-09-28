#include "matx/matx_log.h"
#include "matx/matx_sparse_solve.h"
#include "matx/matx_types_internal.h"

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
#endif
    int unused;
} matx_factor_sparse_z_i8_umfpack_t;

/**
 * @brief Destroy real f64 sparse factorization handle
 * @param F Factorization handle
 */
static void umf_factor_csc_d_i8_destroy(matx_factor_sparse_d_i8_t* F)
{
    if (!F)
        return;
    matx_factor_sparse_d_i8_umfpack_t* ptr = (matx_factor_sparse_d_i8_umfpack_t*) F->reserved;
#if MATX_HAVE_UMFPACK
    if (ptr->numeric)
        umfpack_dl_free_numeric(&ptr->numeric);
    if (ptr->symbolic)
        umfpack_dl_free_symbolic(&ptr->symbolic);
#endif
    free(ptr->Ap);
    free(ptr->Ai);
    free(ptr->Ax);
    free(ptr);
}

/**
 * @brief Perform LU factorization for real CSC matrix using UMFPACK
 * @param A Input COO matrix
 * @param out_F Output factorization handle
 * @return matx_status_t
 */
static matx_status_t umf_factor_csc_d_i8(matx_coo_d_i8_t A, matx_factor_sparse_d_i8_t* out_F)
{
    if (!A || !out_F) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
#if !MATX_HAVE_UMFPACK
    (void) A;
    (void) out_F;
    MATX_ERROR("%s: operation not supported", __func__);
    return MATX_ERR_NOT_SUPPORTED;
#else
    if (out_F->reserved)
        umf_factor_csc_d_i8_destroy(out_F);
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

    matx_factor_sparse_d_i8_umfpack_t* F = (matx_factor_sparse_d_i8_umfpack_t*) calloc(1,
                                                                                       sizeof(*F));
    if (!F) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    out_F->reserved = F;
    F->n = A->nrows;
    F->nnz = A->nnz;

    F->Ap = (matx_int64_t*) malloc(sizeof(matx_int64_t) * (size_t) (F->n + 1));
    F->Ai = (matx_int64_t*) malloc(sizeof(matx_int64_t) * (size_t) F->nnz);
    F->Ax = (matx_double*) malloc(sizeof(matx_double) * (size_t) F->nnz);
    if (!F->Ap || !F->Ai || !F->Ax) {
        umf_factor_csc_d_i8_destroy(out_F);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    memcpy(F->Ap, A->handle_csc->col_ptr, sizeof(matx_int64_t) * (size_t) (F->n + 1));
    memcpy(F->Ai, A->handle_csc->row_ind, sizeof(matx_int64_t) * (size_t) F->nnz);
    memcpy(F->Ax, A->handle_csc->values, sizeof(matx_double) * (size_t) F->nnz);

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
        umf_factor_csc_d_i8_destroy(out_F);
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
        umf_factor_csc_d_i8_destroy(out_F);
        return MATX_ERR_INTERNAL;
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
static matx_status_t umf_solve_csc_d_i8(matx_factor_sparse_d_i8_t* F,
                                        const matx_double* b,
                                        matx_double* x)
{
    if (!F || !b || !x) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
#if !MATX_HAVE_UMFPACK
    (void) F;
    (void) b;
    (void) x;
    MATX_ERROR("%s: operation not supported", __func__);
    return MATX_ERR_NOT_SUPPORTED;
#else
    matx_factor_sparse_d_i8_umfpack_t* ptr = (matx_factor_sparse_d_i8_umfpack_t*) F->reserved;
    int status = umfpack_dl_solve(UMFPACK_A,
                                  (const int64_t*) ptr->Ap,
                                  (const int64_t*) ptr->Ai,
                                  (const double*) ptr->Ax,
                                  (double*) x,
                                  (double*) b,
                                  ptr->numeric,
                                  NULL,
                                  NULL);
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
static void umf_factor_csc_z_i8_destroy(matx_factor_sparse_z_i8_t* F)
{
    if (!F || !F->reserved)
        return;
    matx_factor_sparse_z_i8_umfpack_t* ptr = (matx_factor_sparse_z_i8_umfpack_t*) F->reserved;
#if MATX_HAVE_UMFPACK
    if (ptr->numeric)
        umfpack_zl_free_numeric(&ptr->numeric);
    if (ptr->symbolic)
        umfpack_zl_free_symbolic(&ptr->symbolic);
#endif
    #if MATX_HAVE_UMFPACK
    free(ptr->Ap);
    free(ptr->Ai);
    free(ptr->Az);
#endif
    free(ptr);
}

/**
 * @brief Perform LU factorization for complex CSC matrix using UMFPACK
 * @param A Input complex COO matrix
 * @param out_F Output complex factorization handle
 * @return matx_status_t
 */
static matx_status_t umf_factor_csc_z_i8(matx_coo_z_i8_t A, matx_factor_sparse_z_i8_t* out_F)
{
    if (!A || !out_F) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
#if !MATX_HAVE_UMFPACK
    (void) A;
    (void) out_F;
    MATX_ERROR("%s: operation not supported", __func__);
    return MATX_ERR_NOT_SUPPORTED;
#else
    if (out_F)
        umf_factor_csc_z_i8_destroy(out_F);
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
    matx_factor_sparse_z_i8_umfpack_t* F = (matx_factor_sparse_z_i8_umfpack_t*) calloc(1,
                                                                                       sizeof(*F));
    if (!F) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    out_F->reserved = F;
    F->n = A->nrows;
    F->nnz = A->nnz;
    // Allocate CSC arrays
    F->Ap = (matx_int64_t*) malloc(sizeof(matx_int64_t) * (size_t) (F->nnz + 1));
    F->Ai = (matx_int64_t*) malloc(sizeof(matx_int64_t) * (size_t) F->nnz);
    F->Ax = (matx_double*) malloc(sizeof(matx_double) * (size_t) F->nnz);
    F->Az = (matx_double*) malloc(sizeof(matx_double) * (size_t) F->nnz);
    if (!F->Ap || !F->Ai || !F->Az || !F->Ax) {
        umf_factor_csc_z_i8_destroy(out_F);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    // Copy indices
    memcpy(F->Ap, A->handle_csc->col_ptr, sizeof(matx_int64_t) * (size_t) (F->nnz + 1));
    memcpy(F->Ai, A->handle_csc->row_ind, sizeof(matx_int64_t) * (size_t) F->nnz);

    // Copy complex values (interleave real/imag for UMFPACK)
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
        umf_factor_csc_z_i8_destroy(out_F);
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
        umf_factor_csc_z_i8_destroy(out_F);
        return MATX_ERR_INTERNAL;
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
static matx_status_t umf_solve_csc_z_i8(matx_factor_sparse_z_i8_t* F,
                                        const matx_vec_z_i8_t b,
                                        matx_vec_z_i8_t x)
{
    if (!F || !b || !x) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
#if !MATX_HAVE_UMFPACK
    (void) F;
    (void) b;
    (void) x;
    MATX_ERROR("%s: operation not supported", __func__);
    return MATX_ERR_NOT_SUPPORTED;
#else
    matx_factor_sparse_z_i8_umfpack_t* ptr = (matx_factor_sparse_z_i8_umfpack_t*) F->reserved;
    // UMFPACK complex vectors: interleaved [real0, imag0, real1, imag1...]
    matx_double* b_umf_x = (matx_double*) malloc(sizeof(matx_double) * (size_t) ptr->nnz);
    matx_double* b_umf_z = (matx_double*) malloc(sizeof(matx_double) * (size_t) ptr->nnz);
    matx_double* x_umf_x = (matx_double*) malloc(sizeof(matx_double) * (size_t) ptr->nnz);
    matx_double* x_umf_z = (matx_double*) malloc(sizeof(matx_double) * (size_t) ptr->nnz);
    if (!b_umf_x || !b_umf_z || !x_umf_x || !x_umf_z) {
        free(b_umf_x);
        free(x_umf_x);
        free(b_umf_z);
        free(x_umf_z);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    // Pack input b
    for (matx_int64_t i = 0; i < ptr->nnz; i++) {
        b_umf_x[i] = b->data[i].real;
        b_umf_z[i] = b->data[i].imag;
    }

    // Solve
    int status = umfpack_zl_solve(UMFPACK_A,
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
    if (status != UMFPACK_OK) {
        MATX_ERROR("umfpack_zl_solve failed status=%d", status);
        free(b_umf_x);
        free(x_umf_x);
        free(b_umf_z);
        free(x_umf_z);
        MATX_ERROR("%s: internal error", __func__);
        return MATX_ERR_INTERNAL;
    }

    // Unpack solution x
    for (matx_int64_t i = 0; i < ptr->nnz; i++) {
        x->data[i].real = x_umf_x[i];
        x->data[i].imag = x_umf_z[i];
    }

    free(b_umf_x);
    free(x_umf_x);
    free(b_umf_z);
    free(x_umf_z);
    return MATX_OK;
#endif
}

/**
 * @brief Create UMFPACK linear solver backend (real + complex)
 * @return Fully initialized sparse linear solver interface
 */
matx_sparse_linsolve_t matx_linsolve_make_umfpack(void)
{
    matx_sparse_linsolve_t ls = {.kind = MATX_LINSOLVE_BACKEND_UMFPACK,
                                 .vt = {.factor_csc_d_i8 = &umf_factor_csc_d_i8,
                                        .solve_csc_d_i8 = &umf_solve_csc_d_i8,
                                        .factor_csc_d_i8_destroy = &umf_factor_csc_d_i8_destroy,
                                        .factor_csc_z_i8 = &umf_factor_csc_z_i8,
                                        .solve_csc_z_i8 = &umf_solve_csc_z_i8,
                                        .factor_csc_z_i8_destroy = &umf_factor_csc_z_i8_destroy}};
    return ls;
}