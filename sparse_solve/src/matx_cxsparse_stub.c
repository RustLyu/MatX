#include "matx/matx_sparse_solve.h"
#include "matx/matx_types_internal.h"
#include "matx/matx_log.h"

#if MATX_HAVE_CXSPARSE
#include "cs.h"
#endif

typedef struct matx_factor_sparse_d_i8_cxsparse {
#if MATX_HAVE_CXSPARSE
	cs_dl A;
	cs_dls* S;
	cs_dln* N;
	matx_int64_t n;
#endif
    matx_int64_t unused;
} matx_factor_sparse_d_i8_cxsparse_t;

typedef struct matx_factor_sparse_z_i8_cxsparse {
#if MATX_HAVE_CXSPARSE
    cs_cl A;
    cs_cls* S;
    cs_cln* N;
    matx_int64_t n;
#endif
    matx_int64_t unused;
} matx_factor_sparse_z_i8_cxsparse_t;

static void cxs_factor_csc_d_i8_destroy(matx_factor_sparse_d_i8_t* F)
{
	if (!F) 
		return;
	matx_factor_sparse_d_i8_cxsparse_t* ptr = (matx_factor_sparse_d_i8_cxsparse_t*)F->reserved;
#if MATX_HAVE_CXSPARSE
	if (ptr->N) 
		cs_dl_nfree(ptr->N);
	if (ptr->S)
		cs_dl_sfree(ptr->S);
#endif
	free(ptr);
}

static void cxs_factor_csc_z_i8_destroy(matx_factor_sparse_z_i8_t* F)
{
    if (!F)
        return;
    matx_factor_sparse_z_i8_cxsparse_t* ptr = (matx_factor_sparse_z_i8_cxsparse_t*)F->reserved;
#if MATX_HAVE_CXSPARSE
    if (ptr->N)
        cs_cl_nfree(ptr->N);
    if (ptr->S)
        cs_cl_sfree(ptr->S);
#endif
    free(ptr);
}

static matx_status_t cxs_factor_csc_d_i8(matx_coo_d_i8_t A, matx_factor_sparse_d_i8_t* out_F)
{
    if (!A || !out_F) {
    	MATX_ERROR("%s: invalid argument", __func__);
    	return MATX_ERR_INVALID_ARG;
    }
#if !MATX_HAVE_CXSPARSE
	(void)A; (void)out_F;
	MATX_ERROR("%s: operation not supported", __func__);
	return MATX_ERR_NOT_SUPPORTED;
#else
    if (out_F && out_F->reserved)
		cxs_factor_csc_d_i8_destroy(out_F);
	if (A->nrows != A->ncols || A->nrows <= 0) {
		MATX_ERROR("%s: invalid argument", __func__);
		return MATX_ERR_INVALID_ARG;
	}

	matx_status_t st = coo_to_csc_d_i8(A);
	if (st != MATX_OK) return st;
	st = coo_to_csc_d_i8_value_remap(A);
	if (st != MATX_OK) return st;

	matx_factor_sparse_d_i8_cxsparse_t* F = (matx_factor_sparse_d_i8_cxsparse_t*)calloc(1, sizeof(*F));
    if (!F) {
    	MATX_ERROR("%s: out of memory", __func__);
    	return MATX_ERR_OUT_OF_MEMORY;
    }
    out_F->reserved = F;
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
        cxs_factor_csc_d_i8_destroy(out_F);
		return MATX_ERR_INTERNAL;
	}

	F->N = cs_dl_lu(&F->A, F->S, 1e-12);
	if (!F->N) {
		MATX_ERROR("cs_dl_lu failed");
        cxs_factor_csc_d_i8_destroy(out_F);
		return MATX_ERR_INTERNAL;
	}
	return MATX_OK;
#endif
}

static matx_status_t cxs_solve_csc_d_i8(matx_factor_sparse_d_i8_t* F, const matx_double* b, matx_double* x)
{
    if (!F || !b || !x) {
    	MATX_ERROR("%s: invalid argument", __func__);
    	return MATX_ERR_INVALID_ARG;
    }
#if !MATX_HAVE_CXSPARSE
	(void)F; (void)b; (void)x;
	MATX_ERROR("%s: operation not supported", __func__);
	return MATX_ERR_NOT_SUPPORTED;
#else
	matx_factor_sparse_d_i8_cxsparse_t* ptr = (matx_factor_sparse_d_i8_cxsparse_t*)F->reserved;
	matx_double* y = (matx_double*)malloc(sizeof(matx_double) * (size_t)ptr->n);
	if (!y) {
		MATX_ERROR("%s: out of memory", __func__);
		return MATX_ERR_OUT_OF_MEMORY;
	}

	cs_dl_ipvec(ptr->N->pinv, b, y, ptr->n);
	cs_dl_lsolve(ptr->N->L, y);
	cs_dl_usolve(ptr->N->U, y);
	cs_dl_ipvec(ptr->S->q, y, x, ptr->n);

	free(y);
	return MATX_OK;
#endif
}

static matx_status_t cxs_factor_csc_z_i8(matx_coo_z_i8_t A, matx_factor_sparse_z_i8_t* out_F)
{
    if (!A || !out_F) {
    	MATX_ERROR("%s: invalid argument", __func__);
    	return MATX_ERR_INVALID_ARG;
    }
#if !MATX_HAVE_CXSPARSE
    (void)A; (void)out_F;
    MATX_ERROR("%s: operation not supported", __func__);
    return MATX_ERR_NOT_SUPPORTED;
#else
    if (out_F && out_F->reserved)
        cxs_factor_csc_z_i8_destroy(out_F);
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

    matx_factor_sparse_z_i8_cxsparse_t* F = (matx_factor_sparse_z_i8_cxsparse_t*)calloc(1, sizeof(*F));
    if (!F) {
    	MATX_ERROR("%s: out of memory", __func__);
    	return MATX_ERR_OUT_OF_MEMORY;
    }
    out_F->reserved = F;
    F->n = A->nrows;

    F->A.nzmax = A->nnz;
    F->A.m = A->nrows;
    F->A.n = A->ncols;
    F->A.p = A->handle_csc->col_ptr;
    F->A.i = A->handle_csc->row_ind;
    F->A.x = (cs_complex_t*)A->handle_csc->values;
    F->A.nz = -1;

    F->S = cs_cl_sqr(2, &F->A, 0);
    if (!F->S) {
        MATX_ERROR("cs_cl_sqr failed");
        cxs_factor_csc_z_i8_destroy(out_F);
        return MATX_ERR_INTERNAL;
    }

    F->N = cs_cl_lu(&F->A, F->S, 1e-12);
    if (!F->N) {
        MATX_ERROR("cs_cl_lu failed");
        cxs_factor_csc_z_i8_destroy(out_F);
        return MATX_ERR_INTERNAL;
    }
    return MATX_OK;
#endif
}

static matx_status_t cxs_solve_csc_z_i8(matx_factor_sparse_z_i8_t* F, const matx_vec_z_i8_t b, matx_vec_z_i8_t x)
{
    if (!F || !b || !x) {
    	MATX_ERROR("%s: invalid argument", __func__);
    	return MATX_ERR_INVALID_ARG;
    }
#if !MATX_HAVE_CXSPARSE
    (void)F; (void)b; (void)x;
    MATX_ERROR("%s: operation not supported", __func__);
    return MATX_ERR_NOT_SUPPORTED;
#else
    matx_factor_sparse_z_i8_cxsparse_t* ptr = (matx_factor_sparse_z_i8_cxsparse_t*)F->reserved;
    cs_complex_t* y = (cs_complex_t*)malloc(sizeof(cs_complex_t) * (size_t)ptr->n);
    if (!y) {
    	MATX_ERROR("%s: out of memory", __func__);
    	return MATX_ERR_OUT_OF_MEMORY;
    }

    cs_cl_ipvec(ptr->N->pinv, (cs_complex_t*)b->data, y, ptr->n);
    cs_cl_lsolve(ptr->N->L, y);
    cs_cl_usolve(ptr->N->U, y);
    cs_cl_ipvec(ptr->S->q, y, (cs_complex_t*)x->data, ptr->n);

    free(y);
    return MATX_OK;
#endif
}

matx_sparse_linsolve_t matx_linsolve_make_cxsparse(void)
{
	matx_sparse_linsolve_t ls = {
		.kind = MATX_LINSOLVE_BACKEND_CXSPARSE,
		.vt = {
			.factor_csc_d_i8 = &cxs_factor_csc_d_i8,
			.solve_csc_d_i8 = &cxs_solve_csc_d_i8,
			.factor_csc_d_i8_destroy = &cxs_factor_csc_d_i8_destroy,
			.factor_csc_z_i8 = &cxs_factor_csc_z_i8,
			.solve_csc_z_i8 = &cxs_solve_csc_z_i8,
			.factor_csc_z_i8_destroy = &cxs_factor_csc_z_i8_destroy
		}
	};
	return ls;
}
