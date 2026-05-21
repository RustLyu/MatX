#include "matx/matx_sparse_solve.h"
#include "matx/matx_types_internal.h"
#include "matx/matx_log.h"

#include <stdlib.h>

#if MATX_HAVE_SUPERLU
#include "slu_ddefs.h"
#include "slu_zdefs.h"
#endif

/**
 * @brief Double precision sparse LU factorization handle
 */
typedef struct matx_factor_sparse_f64_slu {
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
#endif
	int place_holder;
} matx_factor_sparse_f64_slu_t;

/**
 * @brief Complex double precision sparse LU factorization handle
 */
typedef struct matx_factor_sparse_c64_slu {
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
#endif
	int unused;
} matx_factor_sparse_c64_slu_t;

/**
 * @brief Destroy double precision sparse factorization handle and free resources
 * @param F Factor handle to destroy
 */
static void slu_factor_csc_f64_destroy(matx_factor_sparse_f64_t* F)
{
	if (!F) 
		return;
	matx_factor_sparse_f64_slu_t* ptr = (matx_factor_sparse_f64_slu_t*)F->reserved;
#if MATX_HAVE_SUPERLU
	Destroy_SuperNode_Matrix(&ptr->L);
	Destroy_CompCol_Matrix(&ptr->U);
	Destroy_CompCol_Matrix(&ptr->A);
	Destroy_Dense_Matrix(&ptr->B);
	StatFree(&ptr->stat);
	free(ptr->perm_c);
	free(ptr->perm_r);
	free(ptr->etree);
	free(ptr->rhs);
#endif
	free(ptr);
}

/**
 * @brief Perform LU factorization for real double CSC matrix via SuperLU
 * @param A Input COO sparse matrix
 * @param out_F Output factorization handle
 * @return MATX_OK on success, error code otherwise
 */
static matx_status_t slu_factor_csc_f64(matx_coo_f64_t A, matx_factor_sparse_f64_t* out_F)
{
	if (!A || !out_F) 
		return MATX_ERR_INVALID_ARG;
#if !MATX_HAVE_SUPERLU
	(void)A; (void)out_F;
	return MATX_ERR_NOT_SUPPORTED;
#else
	/* Free existing handle if allocated */
	if (out_F) slu_factor_csc_f64_destroy(out_F);

	/* Validate matrix dimension */
	if (A->nrows != A->ncols || A->nrows <= 0) return MATX_ERR_INVALID_ARG;
	if (A->nrows > INT_MAX || A->nnz > INT_MAX) return MATX_ERR_NOT_SUPPORTED;

	/* Convert COO to CSC format */
	matx_status_t st = coo_to_csc_f64(A);
	if (st != MATX_OK) return st;
	st = coo_to_csc_f64_value_remap(A);
	if (st != MATX_OK) return st;

	/* Allocate factorization handle */
        matx_factor_sparse_f64_slu_t* F = (matx_factor_sparse_f64_slu_t*)calloc(1, sizeof(*F));
        if (!F)
          return MATX_ERR_OUT_OF_MEMORY;
        out_F->reserved = F;
	F->n = (int)A->nrows;

	/* Allocate permutation and working arrays */
	F->perm_c = (int*)malloc(sizeof(int) * (size_t)F->n);
	F->perm_r = (int*)malloc(sizeof(int) * (size_t)F->n);
	F->etree = (int*)malloc(sizeof(int) * (size_t)F->n);
	F->rhs = (double*)malloc(sizeof(double) * (size_t)F->n);

	/* Check memory allocation */
	if (!F->perm_c || !F->perm_r || !F->etree || !F->rhs) {
                slu_factor_csc_f64_destroy(out_F);
		return MATX_ERR_OUT_OF_MEMORY;
	}

	/* Create SuperLU CSC matrix and dense RHS matrix */
	dCreate_CompCol_Matrix(&F->A, F->n, F->n, (int)A->nnz,
		A->handle_csc->values, (int*)A->handle_csc->row_ind, (int*)A->handle_csc->col_ptr,
		SLU_NC, SLU_D, SLU_GE);

	dCreate_Dense_Matrix(&F->B, F->n, 1, F->rhs, F->n, SLU_DN, SLU_D, SLU_GE);

	/* Initialize SuperLU solver options */
	set_default_options(&F->options);
	F->options.ColPerm = COLAMD;
	StatInit(&F->stat);

	/* Perform LU factorization with partial pivoting */
	int info = 0;
	dgssv(&F->options, &F->A, F->perm_c, F->perm_r, &F->L, &F->U, &F->B, &F->stat, &info);
	if (info != 0) {
		MATX_ERROR("dgssv factor failed info=%d", info);
                slu_factor_csc_f64_destroy(out_F);
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
static matx_status_t slu_solve_csc_f64(matx_factor_sparse_f64_t* F, const matx_double* b, matx_double* x)
{
	if (!F || !b || !x) return MATX_ERR_INVALID_ARG;
#if !MATX_HAVE_SUPERLU
	(void)F; (void)b; (void)x;
	return MATX_ERR_NOT_SUPPORTED;
#else
      matx_factor_sparse_f64_slu_t* ptr = (matx_factor_sparse_f64_slu_t*)F->reserved;
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
static void slu_factor_csc_c64_destroy(matx_factor_sparse_c64_t* F)
{
        if (!F) return;
#if MATX_HAVE_SUPERLU
        matx_factor_sparse_c64_slu_t* ptr = (matx_factor_sparse_c64_slu_t*)F->reserved;
        /* Release SuperLU internal matrices */
        Destroy_SuperNode_Matrix(&ptr->L);
        Destroy_CompCol_Matrix(&ptr->U);
        Destroy_CompCol_Matrix(&ptr->A);
        Destroy_Dense_Matrix(&ptr->B);

        /* Free solver stat and working arrays */
        StatFree(&ptr->stat);
        free(ptr->perm_c);
        free(ptr->perm_r);
        free(ptr->etree);
        free(ptr->rhs);
#endif
        free(ptr);
}

/**
 * @brief Perform LU factorization for complex double CSC matrix via SuperLU
 * @param A Input complex COO sparse matrix
 * @param out_F Output complex factorization handle
 * @return MATX_OK on success, error code otherwise
 */
static matx_status_t slu_factor_csc_c64(matx_coo_c64_t A, matx_factor_sparse_c64_t* out_F)
{
	if (!A || !out_F) return MATX_ERR_INVALID_ARG;
#if !MATX_HAVE_SUPERLU
	(void)A; (void)out_F;
	return MATX_ERR_NOT_SUPPORTED;
#else
	/* Free existing handle if allocated */
        if (out_F->reserved)
          slu_factor_csc_c64_destroy(out_F);

	/* Validate square matrix and size limit */
	if (A->nrows != A->ncols || A->nrows <= 0) return MATX_ERR_INVALID_ARG;
	if (A->nrows > INT_MAX || A->nnz > INT_MAX) return MATX_ERR_NOT_SUPPORTED;

	/* Convert complex COO to CSC format */
	matx_status_t st = coo_to_csc_c64(A);
	if (st != MATX_OK) return st;
	st = coo_to_csc_c64_value_remap(A);
	if (st != MATX_OK) return st;

	/* Allocate complex factorization handle */
        matx_factor_sparse_c64_slu_t* F = (matx_factor_sparse_c64_slu_t*)calloc(1, sizeof(*F));
        if (!F)
          return MATX_ERR_OUT_OF_MEMORY;
        out_F->reserved = F;
	F->n = (int)A->nrows;

	/* Allocate permutation and complex RHS buffer */
	F->perm_c = (int*)malloc(sizeof(int) * (size_t)F->n);
	F->perm_r = (int*)malloc(sizeof(int) * (size_t)F->n);
	F->etree = (int*)malloc(sizeof(int) * (size_t)F->n);
	F->rhs = (doublecomplex*)malloc(sizeof(doublecomplex) * (size_t)F->n);

	/* Check memory allocation status */
	if (!F->perm_c || !F->perm_r || !F->etree || !F->rhs) {
                slu_factor_csc_c64_destroy(out_F);
		return MATX_ERR_OUT_OF_MEMORY;
	}

	/* Create SuperLU complex CSC matrix */
	zCreate_CompCol_Matrix(&F->A, F->n, F->n, (int)A->nnz,
                (doublecomplex*)A->handle_csc->values, (int*)A->handle_csc->row_ind, (int*)A->handle_csc->col_ptr,
		SLU_NC, SLU_Z, SLU_GE);

	/* Create complex dense RHS matrix */
	zCreate_Dense_Matrix(&F->B, F->n, 1, F->rhs, F->n, SLU_DN, SLU_Z, SLU_GE);

	/* Setup default SuperLU options */
	set_default_options(&F->options);
	F->options.ColPerm = COLAMD;
	StatInit(&F->stat);

	/* Complex LU factorization via zgssv */
	int info = 0;
	zgssv(&F->options, &F->A, F->perm_c, F->perm_r, &F->L, &F->U, &F->B, &F->stat, &info);
	if (info != 0) {
		MATX_ERROR("zgssv complex factor failed info=%d", info);
                slu_factor_csc_c64_destroy(out_F);
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
static matx_status_t slu_solve_csc_c64(matx_factor_sparse_c64_t* F, const matx_vec_c64_t b, matx_vec_c64_t x)
{
	if (!F || !b || !x) return MATX_ERR_INVALID_ARG;
#if !MATX_HAVE_SUPERLU
	(void)F; (void)b; (void)x;
	return MATX_ERR_NOT_SUPPORTED;
#else
        matx_factor_sparse_c64_slu_t* ptr = (matx_factor_sparse_c64_slu_t*)F->reserved;
	/* Copy complex RHS (real + imaginary part) */
 //        for (int i = 0; i < ptr->n; ++i) {
 //                ptr->rhs[i].r = b[i].r;
 //                ptr->rhs[i].i = b[i].i;
        // }
        memcpy(ptr->rhs, b->data, sizeof(matx_double) * ptr->n * 2);
	/* Complex triangular solve with LU factors */
	int info = 0;
        zgstrs(NOTRANS, &ptr->L, &ptr->U, ptr->perm_c, ptr->perm_r, &ptr->B, &ptr->stat, &info);
	if (info != 0) {
		MATX_ERROR("zgstrs complex solve failed info=%d", info);
		return MATX_ERR_INTERNAL;
	}

	/* Copy complex solution back to output */
        // for (int i = 0; i < F->n; ++i) {
        // 	x[i].r = F->rhs[i].r;
        // 	x[i].i = F->rhs[i].i;
        // }
        memcpy(x->data, ptr->rhs, sizeof(matx_double) * ptr->n * 2);
	return MATX_OK;
#endif
}

/**
 * @brief Create SuperLU linear solver backend interface
 * @return Initialized solver dispatch table
 */
matx_sparse_linsolve_t matx_linsolve_make_superlu(void)
{
	matx_sparse_linsolve_t ls = {
		.kind = MATX_LINSOLVE_BACKEND_SUPERLU,
		.vt = {
			.factor_csc_f64 = &slu_factor_csc_f64,
			.solve_csc_f64 = &slu_solve_csc_f64,
			.factor_csc_f64_destroy = &slu_factor_csc_f64_destroy,
			.factor_csc_c64 = &slu_factor_csc_c64,
			.solve_csc_c64 = &slu_solve_csc_c64,
			.factor_csc_c64_destroy = &slu_factor_csc_c64_destroy
		}
	};
	return ls;
}
