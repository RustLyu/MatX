#include "matx/matx_sparse_solve.h"
#include "matx/matx_types_internal.h"
#include "matx/matx_log.h"

#if MATX_HAVE_CXSPARSE
#include "cs.h"
#endif

typedef struct matx_factor_sparse_f64_cxsparse {
#if MATX_HAVE_CXSPARSE
	cs_dl A;
	cs_dls* S;
	cs_dln* N;
	matx_int64_t n;
#endif
} matx_factor_sparse_f64_cxsparse_t;

typedef struct matx_factor_sparse_c64_cxsparse {
	int unused;
} matx_factor_sparse_c64_cxsparse_t;

static void cxs_factor_csc_f64_destroy(matx_factor_sparse_f64_t* F)
{
	if (!F) 
		return;
	matx_factor_sparse_f64_cxsparse_t* ptr = (matx_factor_sparse_f64_cxsparse_t*)F->reserved;
#if MATX_HAVE_CXSPARSE
	if (ptr->N) 
		cs_dl_nfree(ptr->N);
	if (ptr->S)
		cs_dl_sfree(ptr->S);
#endif
	free(ptr);
}

static matx_status_t cxs_factor_csc_f64(matx_coo_f64_t A, matx_factor_sparse_f64_t* out_F)
{
	if (!A || !out_F) return MATX_ERR_INVALID_ARG;
#if !MATX_HAVE_CXSPARSE
	(void)A; (void)out_F;
	return MATX_ERR_NOT_SUPPORTED;
#else
	if (out_F) 
		cxs_factor_csc_f64_destroy(out_F);
	if (A->nrows != A->ncols || A->nrows <= 0) 
		return MATX_ERR_INVALID_ARG;

	matx_status_t st = coo_to_csc_f64(A);
	if (st != MATX_OK) return st;
	st = coo_to_csc_f64_value_remap(A);
	if (st != MATX_OK) return st;

	matx_factor_sparse_f64_cxsparse_t* F = (matx_factor_sparse_f64_cxsparse_t*)calloc(1, sizeof(*F));
	if (!F) return MATX_ERR_OUT_OF_MEMORY;
	F->n = A->nrows;

	F->A.nzmax = A->nnz;
	F->A.m = A->nrows;
	F->A.n = A->ncols;
	F->A.p = A->handle_csc->col_ptr;
	F->A.i = A->handle_csc->row_ind;
	F->A.x = A->handle_csc->values;
	F->A.nz = -1;

	F->S = cs_dl_sqr(2, &F->A, 0);
	if (!F->S) {
		MATX_ERROR("cs_dl_sqr failed");
		cxs_factor_csc_f64_destroy(F);
		return MATX_ERR_INTERNAL;
	}

	F->N = cs_dl_lu(&F->A, F->S, 1e-12);
	if (!F->N) {
		MATX_ERROR("cs_dl_lu failed");
		cxs_factor_csc_f64_destroy(F);
		return MATX_ERR_INTERNAL;
	}

	out_F->reserved = F;
	return MATX_OK;
#endif
}

static matx_status_t cxs_solve_csc_f64(matx_factor_sparse_f64_t* F, const matx_double* b, matx_double* x)
{
	if (!F || !b || !x) return MATX_ERR_INVALID_ARG;
#if !MATX_HAVE_CXSPARSE
	(void)F; (void)b; (void)x;
	return MATX_ERR_NOT_SUPPORTED;
#else
	matx_factor_sparse_f64_cxsparse_t* ptr = (matx_factor_sparse_f64_cxsparse_t*)F->reserved;
	matx_double* y = (matx_double*)malloc(sizeof(matx_double) * (size_t)ptr->n);
	if (!y) 
		return MATX_ERR_OUT_OF_MEMORY;

	cs_dl_ipvec(ptr->N->pinv, b, y, ptr->n);
	cs_dl_lsolve(ptr->N->L, y);
	cs_dl_usolve(ptr->N->U, y);
	cs_dl_ipvec(ptr->S->q, y, x, ptr->n);

	free(y);
	return MATX_OK;
#endif
}

static matx_status_t cxs_factor_csc_c64(matx_coo_c64_t A, matx_factor_sparse_c64_t** out_F)
{
	(void)A; (void)out_F;
	return MATX_ERR_NOT_SUPPORTED;
}

static matx_status_t cxs_solve_csc_c64(matx_factor_sparse_c64_t* F, const matx_vec_c64_t b, matx_vec_c64_t x)
{
	(void)F; (void)b; (void)x;
	return MATX_ERR_NOT_SUPPORTED;
}

static void cxs_factor_csc_c64_destroy(matx_factor_sparse_c64_t* F)
{
	free(F);
}

matx_sparse_linsolve_t matx_linsolve_make_cxsparse(void)
{
	matx_sparse_linsolve_t ls = {
		.kind = MATX_LINSOLVE_BACKEND_CXSPARSE,
		.vt = {
			.factor_csc_f64 = &cxs_factor_csc_f64,
			.solve_csc_f64 = &cxs_solve_csc_f64,
			.factor_csc_f64_destroy = &cxs_factor_csc_f64_destroy,
			.factor_csc_c64 = &cxs_factor_csc_c64,
			.solve_csc_c64 = &cxs_solve_csc_c64,
			.factor_csc_c64_destroy = &cxs_factor_csc_c64_destroy
		}
	};
	return ls;
}
