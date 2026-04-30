#include "matx/matx_sparse_solve.h"
#include "matx/matx_types_internal.h"
#include "matx/matx_log.h"

#if MATX_HAVE_SUPERLU
#include "slu_ddefs.h"
#endif

struct matx_factor_sparse_f64_t {
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
};

struct matx_factor_sparse_c64_t {
	int unused;
};

static void slu_factor_csc_f64_destroy(matx_factor_sparse_f64_t* F)
{
	if (!F) return;
#if MATX_HAVE_SUPERLU
	Destroy_SuperNode_Matrix(&F->L);
	Destroy_CompCol_Matrix(&F->U);
	Destroy_CompCol_Matrix(&F->A);
	Destroy_Dense_Matrix(&F->B);
	StatFree(&F->stat);
	free(F->perm_c);
	free(F->perm_r);
	free(F->etree);
	free(F->rhs);
#endif
	free(F);
}

static matx_status_t slu_factor_csc_f64(matx_coo_f64_t A, matx_factor_sparse_f64_t** out_F)
{
	if (!A || !out_F) return MATX_ERR_INVALID_ARG;
#if !MATX_HAVE_SUPERLU
	(void)A; (void)out_F;
	return MATX_ERR_NOT_SUPPORTED;
#else
	if (*out_F) slu_factor_csc_f64_destroy(*out_F);
	if (A->nrows != A->ncols || A->nrows <= 0) return MATX_ERR_INVALID_ARG;
	if (A->nrows > INT_MAX || A->nnz > INT_MAX) return MATX_ERR_NOT_SUPPORTED;

	matx_status_t st = coo_to_csc_f64(A);
	if (st != MATX_OK) return st;
	st = coo_to_csc_f64_value_remap(A);
	if (st != MATX_OK) return st;

	matx_factor_sparse_f64_t* F = (matx_factor_sparse_f64_t*)calloc(1, sizeof(*F));
	if (!F) return MATX_ERR_OUT_OF_MEMORY;
	F->n = (int)A->nrows;

	F->perm_c = (int*)malloc(sizeof(int) * (size_t)F->n);
	F->perm_r = (int*)malloc(sizeof(int) * (size_t)F->n);
	F->etree = (int*)malloc(sizeof(int) * (size_t)F->n);
	F->rhs = (double*)malloc(sizeof(double) * (size_t)F->n);
	if (!F->perm_c || !F->perm_r || !F->etree || !F->rhs) {
		slu_factor_csc_f64_destroy(F);
		return MATX_ERR_OUT_OF_MEMORY;
	}

	dCreate_CompCol_Matrix(&F->A, F->n, F->n, (int)A->nnz,
		A->handle_csc->values, (int*)A->handle_csc->row_ind, (int*)A->handle_csc->col_ptr,
		SLU_NC, SLU_D, SLU_GE);
	dCreate_Dense_Matrix(&F->B, F->n, 1, F->rhs, F->n, SLU_DN, SLU_D, SLU_GE);

	set_default_options(&F->options);
	F->options.ColPerm = COLAMD;
	StatInit(&F->stat);

	int info = 0;
	dgssv(&F->options, &F->A, F->perm_c, F->perm_r, &F->L, &F->U, &F->B, &F->stat, &info);
	if (info != 0) {
		MATX_ERROR("dgssv factor failed info=%d", info);
		slu_factor_csc_f64_destroy(F);
		return MATX_ERR_INTERNAL;
	}

	F->options.Fact = FACTORED;
	*out_F = F;
	return MATX_OK;
#endif
}

static matx_status_t slu_solve_csc_f64(matx_factor_sparse_f64_t* F, const matx_double* b, matx_double* x)
{
	if (!F || !b || !x) return MATX_ERR_INVALID_ARG;
#if !MATX_HAVE_SUPERLU
	(void)F; (void)b; (void)x;
	return MATX_ERR_NOT_SUPPORTED;
#else
	for (int i = 0; i < F->n; ++i) F->rhs[i] = b[i];
	int info = 0;
	dgstrs(NOTRANS, &F->L, &F->U, F->perm_c, F->perm_r, &F->B, &F->stat, &info);
	if (info != 0) {
		MATX_ERROR("dgstrs solve failed info=%d", info);
		return MATX_ERR_INTERNAL;
	}
	for (int i = 0; i < F->n; ++i) x[i] = F->rhs[i];
	return MATX_OK;
#endif
}

static matx_status_t slu_factor_csc_c64(matx_coo_c64_t A, matx_factor_sparse_c64_t** out_F)
{
	(void)A; (void)out_F;
	return MATX_ERR_NOT_SUPPORTED;
}

static matx_status_t slu_solve_csc_c64(matx_factor_sparse_c64_t* F, const matx_vec_c64_t b, matx_vec_c64_t x)
{
	(void)F; (void)b; (void)x;
	return MATX_ERR_NOT_SUPPORTED;
}

static void slu_factor_csc_c64_destroy(matx_factor_sparse_c64_t* F)
{
	free(F);
}

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
