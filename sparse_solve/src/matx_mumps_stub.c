#include "matx/matx_sparse_solve.h"
#include "matx/matx_types_internal.h"
#include "matx/matx_log.h"

#include <stdlib.h>

#if MATX_HAVE_MUMPS
#include <dmumps_c.h>
#include <zmumps_c.h>
#endif

/**
 * @brief MUMPS context for real double (f64) sparse factorization
 */
typedef struct matx_factor_sparse_f64_mumps {
#if MATX_HAVE_MUMPS
    DMUMPS_STRUC_C mumps;
    matx_int64_t n;
    matx_int64_t nnz;
    matx_int64_t* irn;  // Row indices (1-based)
    matx_int64_t* jcn;  // Column indices (1-based)
    matx_double* a;     // Values
#endif
    int unused;
}matx_factor_sparse_f64_mumps_t;

/**
 * @brief MUMPS context for complex double (c64) sparse factorization
 */
typedef struct matx_factor_sparse_c64_mumps {
#if MATX_HAVE_MUMPS
    ZMUMPS_STRUC_C mumps;
    matx_int64_t n;
    matx_int64_t nnz;
    matx_int64_t* irn;   // Row indices (1-based)
    matx_int64_t* jcn;   // Column indices (1-based)
    matx_double* a;      // Complex values: [r0,i0,r1,i1,...]
#endif
    int unused;
} matx_factor_sparse_c64_mumps_t;

/**
 * @brief Destroy real f64 MUMPS factorization context
 * @param F Factor handle
 */
static void mumps_factor_csc_f64_destroy(matx_factor_sparse_f64_t* F)
{
    if (!F) return;
#if MATX_HAVE_MUMPS
    // Cleanup MUMPS internal data
    if (F->mumps.comm_fortran != -987654) {
        F->mumps.job = -2;
        dmumps_c(&F->mumps);
    }
    // Free allocated arrays
    free(F->irn);
    free(F->jcn);
    free(F->a);
#endif
    free(F);
}

/**
 * @brief Perform LU factorization for real CSC matrix using MUMPS
 * @param A Input COO sparse matrix
 * @param out_F Output factorization handle
 * @return matx_status_t
 */
static matx_status_t mumps_factor_csc_f64(matx_coo_f64_t A, matx_factor_sparse_f64_t* out_F)
{
    if (!A || !out_F) return MATX_ERR_INVALID_ARG;
#if !MATX_HAVE_MUMPS
    (void)A; (void)out_F;
    return MATX_ERR_NOT_SUPPORTED;
#else
    if (*out_F) mumps_factor_csc_f64_destroy(*out_F);
    if (A->nrows != A->ncols || A->nrows <= 0) return MATX_ERR_INVALID_ARG;

    // Convert to CSC format
    matx_status_t st = coo_to_csc_f64(A);
    if (st != MATX_OK) return st;
    st = coo_to_csc_f64_value_remap(A);
    if (st != MATX_OK) return st;

    // Allocate factor handle
    matx_factor_sparse_f64_t* F = (matx_factor_sparse_f64_t*)calloc(1, sizeof(*F));
    if (!F) return MATX_ERR_OUT_OF_MEMORY;
    F->n = A->nrows;
    F->nnz = A->nnz;

    // Allocate triplet format arrays for MUMPS
    F->irn = (matx_int64_t*)malloc(sizeof(matx_int64_t) * (size_t)F->nnz);
    F->jcn = (matx_int64_t*)malloc(sizeof(matx_int64_t) * (size_t)F->nnz);
    F->a = (matx_double*)malloc(sizeof(matx_double) * (size_t)F->nnz);
    if (!F->irn || !F->jcn || !F->a) {
        mumps_factor_csc_f64_destroy(F);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    // Convert CSC to 1-based triplet format (MUMPS requirement)
    matx_int64_t idx = 0;
    for (matx_int64_t col = 0; col < F->n; col++) {
        for (matx_int64_t p = A->handle_csc->col_ptr[col]; p < A->handle_csc->col_ptr[col + 1]; p++) {
            F->irn[idx] = A->handle_csc->row_ind[p] + 1;  // 1-based
            F->jcn[idx] = col + 1;
            F->a[idx] = A->handle_csc->values[p];
            idx++;
        }
    }

    // Initialize MUMPS for real double
    F->mumps.comm_fortran = -987654;  // Use default communicator
    F->mumps.par = 1;                 // Host does all work
    F->mumps.sym = 0;                 // Unsymmetric matrix
    dmumps_c(&F->mumps);

    // Set matrix dimensions and data
    F->mumps.n = F->n;
    F->mumps.nz = F->nnz;
    F->mumps.irn = F->irn;
    F->mumps.jcn = F->jcn;
    F->mumps.a = F->a;

    // Step 1: Analysis
    F->mumps.job = 1;
    dmumps_c(&F->mumps);
    if (F->mumps.info[0] != 0) {
        MATX_ERROR("MUMPS real analysis failed, info[0]=%d", F->mumps.info[0]);
        mumps_factor_csc_f64_destroy(F);
        return MATX_ERR_INTERNAL;
    }

    // Step 2: Factorization
    F->mumps.job = 2;
    dmumps_c(&F->mumps);
    if (F->mumps.info[0] != 0) {
        MATX_ERROR("MUMPS real factorization failed, info[0]=%d", F->mumps.info[0]);
        mumps_factor_csc_f64_destroy(F);
        return MATX_ERR_INTERNAL;
    }

    *out_F = F;
    return MATX_OK;
#endif
}

/**
 * @brief Solve real linear system Ax = b using pre-factored MUMPS
 * @param F Factor handle
 * @param b RHS vector
 * @param x Solution vector
 * @return matx_status_t
 */
static matx_status_t mumps_solve_csc_f64(matx_factor_sparse_f64_t* F, const matx_double* b, matx_double* x)
{
    if (!F || !b || !x) return MATX_ERR_INVALID_ARG;
#if !MATX_HAVE_MUMPS
    (void)F; (void)b; (void)x;
    return MATX_ERR_NOT_SUPPORTED;
#else
    // Copy RHS to MUMPS solution buffer
    for (matx_int64_t i = 0; i < F->n; i++) {
        x[i] = b[i];
    }
    F->mumps.rhs = x;

    // Solve
    F->mumps.job = 3;
    dmumps_c(&F->mumps);
    if (F->mumps.info[0] != 0) {
        MATX_ERROR("MUMPS real solve failed, info[0]=%d", F->mumps.info[0]);
        return MATX_ERR_INTERNAL;
    }

    return MATX_OK;
#endif
}

/**
 * @brief Destroy complex c64 MUMPS factorization context
 * @param F Complex factor handle
 */
static void mumps_factor_csc_c64_destroy(matx_factor_sparse_c64_t* F)
{
    if (!F) return;
#if MATX_HAVE_MUMPS
    // Cleanup MUMPS internal data
    if (F->mumps.comm_fortran != -987654) {
        F->mumps.job = -2;
        zmumps_c(&F->mumps);
    }
    // Free allocated arrays
    free(F->irn);
    free(F->jcn);
    free(F->a);
#endif
    free(F);
}

/**
 * @brief Perform LU factorization for complex CSC matrix using MUMPS
 * @param A Input complex COO sparse matrix
 * @param out_F Output complex factorization handle
 * @return matx_status_t
 */
static matx_status_t mumps_factor_csc_c64(matx_coo_c64_t A, matx_factor_sparse_c64_t* out_F)
{
    if (!A || !out_F) return MATX_ERR_INVALID_ARG;
#if !MATX_HAVE_MUMPS
    (void)A; (void)out_F;
    return MATX_ERR_NOT_SUPPORTED;
#else
    if (*out_F) mumps_factor_csc_c64_destroy(*out_F);
    if (A->nrows != A->ncols || A->nrows <= 0) return MATX_ERR_INVALID_ARG;

    // Convert complex COO to CSC
    matx_status_t st = coo_to_csc_c64(A);
    if (st != MATX_OK) return st;
    st = coo_to_csc_c64_value_remap(A);
    if (st != MATX_OK) return st;

    // Allocate complex factor handle
    matx_factor_sparse_c64_t* F = (matx_factor_sparse_c64_t*)calloc(1, sizeof(*F));
    if (!F) return MATX_ERR_OUT_OF_MEMORY;
    F->n = A->nrows;
    F->nnz = A->nnz;

    // Allocate triplet arrays (complex values interleaved)
    F->irn = (matx_int64_t*)malloc(sizeof(matx_int64_t) * (size_t)F->nnz);
    F->jcn = (matx_int64_t*)malloc(sizeof(matx_int64_t) * (size_t)F->nnz);
    F->a = (matx_double*)malloc(sizeof(matx_double) * 2 * (size_t)F->nnz);
    if (!F->irn || !F->jcn || !F->a) {
        mumps_factor_csc_c64_destroy(F);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    // Convert CSC to 1-based triplet + interleave complex values
    matx_int64_t idx = 0;
    for (matx_int64_t col = 0; col < F->n; col++) {
        for (matx_int64_t p = A->handle_csc->col_ptr[col]; p < A->handle_csc->col_ptr[col + 1]; p++) {
            F->irn[idx] = A->handle_csc->row_ind[p] + 1;
            F->jcn[idx] = col + 1;
            F->a[2 * idx] = A->handle_csc->values[p].r;
            F->a[2 * idx + 1] = A->handle_csc->values[p].i;
            idx++;
        }
    }

    // Initialize complex MUMPS
    F->mumps.comm_fortran = -987654;
    F->mumps.par = 1;
    F->mumps.sym = 0;
    zmumps_c(&F->mumps);

    // Set complex matrix data
    F->mumps.n = F->n;
    F->mumps.nz = F->nnz;
    F->mumps.irn = F->irn;
    F->mumps.jcn = F->jcn;
    F->mumps.a = F->a;

    // Analysis
    F->mumps.job = 1;
    zmumps_c(&F->mumps);
    if (F->mumps.info[0] != 0) {
        MATX_ERROR("MUMPS complex analysis failed, info[0]=%d", F->mumps.info[0]);
        mumps_factor_csc_c64_destroy(F);
        return MATX_ERR_INTERNAL;
    }

    // Factorization
    F->mumps.job = 2;
    zmumps_c(&F->mumps);
    if (F->mumps.info[0] != 0) {
        MATX_ERROR("MUMPS complex factorization failed, info[0]=%d", F->mumps.info[0]);
        mumps_factor_csc_c64_destroy(F);
        return MATX_ERR_INTERNAL;
    }

    *out_F = F;
    return MATX_OK;
#endif
}

/**
 * @brief Solve complex linear system Ax = b using pre-factored MUMPS
 * @param F Complex factor handle
 * @param b Complex RHS vector
 * @param x Complex solution vector
 * @return matx_status_t
 */
static matx_status_t mumps_solve_csc_c64(matx_factor_sparse_c64_t* F, const matx_vec_c64_t b, matx_vec_c64_t x)
{
    if (!F || !b || !x) return MATX_ERR_INVALID_ARG;
#if !MATX_HAVE_MUMPS
    (void)F; (void)b; (void)x;
    return MATX_ERR_NOT_SUPPORTED;
#else
    // Allocate interleaved complex RHS buffer for MUMPS
    matx_double* rhs_umf = (matx_double*)malloc(sizeof(matx_double) * 2 * (size_t)F->n);
    if (!rhs_umf) return MATX_ERR_OUT_OF_MEMORY;

    // Pack complex RHS
    for (matx_int64_t i = 0; i < F->n; i++) {
        rhs_umf[2 * i] = b[i].r;
        rhs_umf[2 * i + 1] = b[i].i;
    }
    F->mumps.rhs = rhs_umf;

    // Solve
    F->mumps.job = 3;
    zmumps_c(&F->mumps);
    if (F->mumps.info[0] != 0) {
        MATX_ERROR("MUMPS complex solve failed, info[0]=%d", F->mumps.info[0]);
        free(rhs_umf);
        return MATX_ERR_INTERNAL;
    }

    // Unpack solution
    for (matx_int64_t i = 0; i < F->n; i++) {
        x[i].r = rhs_umf[2 * i];
        x[i].i = rhs_umf[2 * i + 1];
    }

    free(rhs_umf);
    return MATX_OK;
#endif
}

/**
 * @brief Create MUMPS sparse linear solver backend (real + complex)
 * @return Initialized solver interface
 */
matx_sparse_linsolve_t matx_linsolve_make_mumps(void)
{
    matx_sparse_linsolve_t ls = {
        .kind = MATX_LINSOLVE_BACKEND_MUMPS,
        .vt = {
            .factor_csc_f64 = &mumps_factor_csc_f64,
            .solve_csc_f64 = &mumps_solve_csc_f64,
            .factor_csc_f64_destroy = &mumps_factor_csc_f64_destroy,
            .factor_csc_c64 = &mumps_factor_csc_c64,
            .solve_csc_c64 = &mumps_solve_csc_c64,
            .factor_csc_c64_destroy = &mumps_factor_csc_c64_destroy
        }
    };
    return ls;
}