#include "matx/matx_sparse_solve.h"

#include <limits.h>

#include "suitesparse/klu.h"

struct matx_factor_sparse_f64_t {
	klu_l_symbolic* S;
	klu_l_numeric* N;
	klu_l_common common;
	matx_int64_t n;
};

// Sparse real: KLU-based ---------------------------------------------------
static matx_status_t ss_factor_csc_f64(const matx_coo_f64_t* A,
	matx_factor_sparse_f64_t** out_F) {
	if (!A || !out_F) 
		return MATX_ERR_INVALID_ARG;
	
	matx_status_t st = coo_to_csc_f64(A);
	if(st != MATX_OK)
		return st;

	if (!A->handle_csc.col_ptr || !A->handle_csc.row_ind || !A->handle_csc.values) 
		return MATX_ERR_INVALID_ARG;
	if (A->nrows != A->ncols) 
		return MATX_ERR_INVALID_ARG;

	const matx_int64_t n = A->nrows;
	if (n == 0) 
		return MATX_ERR_INVALID_ARG;
	if (n > (matx_int64_t)INT_MAX) 
		return MATX_ERR_NOT_SUPPORTED;

	matx_factor_sparse_f64_t* F =
		(matx_factor_sparse_f64_t*)malloc(sizeof(*F));
	if (!F) 
		return MATX_ERR_OUT_OF_MEMORY;

	memset(F, 0, sizeof(*F));
	F->n = (matx_int64_t)n;
	klu_l_defaults(&F->common);

	F->S = klu_l_analyze(F->n,
		A->handle_csc.col_ptr,
		A->handle_csc.row_ind,
		&F->common);
	if (!F->S) {
		free(F);
		return MATX_ERR_INTERNAL;
	}

	st = coo_to_csc_f64_value_remap(A);

	if(st != MATX_OK) {
		klu_l_free_symbolic(&F->S, &F->common);
		free(F);
		return st;
	}

	F->N = klu_l_factor(A->handle_csc.col_ptr,
		A->handle_csc.row_ind,
		A->handle_csc.values,
		F->S,
		&F->common);

	if (!F->N) {
		klu_l_free_symbolic(&F->S, &F->common);
		free(F);
		return MATX_ERR_INTERNAL;
	}

	*out_F = F;
	return MATX_OK;
}

static matx_status_t ss_solve_csc_f64(const matx_factor_sparse_f64_t* F,
	const matx_double* b,
	matx_double* x) {
	if (!F || !b || !x) 
		return MATX_ERR_INVALID_ARG;

	const matx_int64_t n = F->n;
	for (matx_int64_t i = 0; i < n; ++i) {
		x[i] = b[i];
	}

	const int status = klu_l_solve(F->S, F->N, n, 1, x, &F->common);
	if (!status) 
		return MATX_ERR_INTERNAL;
	return MATX_OK;
}

static void ss_factor_csc_f64_destroy(matx_factor_sparse_f64_t* F) {
	if (!F) 
		return;
	klu_l_free_numeric(&F->N, &F->common);
	klu_l_free_symbolic(&F->S, &F->common);
	free(F);
}


static matx_status_t ss_factor_csc_c64(
	const matx_coo_c64_t* A,
	matx_factor_sparse_c64_t** out_F)
{
	if (!A || !out_F) 
		return MATX_ERR_INVALID_ARG;

	matx_status_t st = coo_to_csc_c64(A);
	if(st != MATX_OK)
		return st;

	if (!A->handle_csc.col_ptr || !A->handle_csc.row_ind || !A->values)
		return MATX_ERR_INVALID_ARG;
	if (A->nrows != A->ncols)
		return MATX_ERR_INVALID_ARG;

	const matx_int64_t n = A->nrows;
	if (n > INT_MAX) 
		return MATX_ERR_NOT_SUPPORTED;

	matx_factor_sparse_c64_t* F =
		(matx_factor_sparse_c64_t*)malloc(sizeof(*F));
	if (!F) 
		return MATX_ERR_OUT_OF_MEMORY;

	memset(F, 0, sizeof(*F));
	F->n = (matx_int64_t)n;

	klu_l_defaults(&F->common);

	F->S = klu_l_analyze(
		F->n,
		A->handle_csc.col_ptr,
		A->handle_csc.row_ind,
		&F->common);

	if (!F->S) 
		goto fail;

	st = coo_to_csc_c64_value_remap(A);
	if(st != MATX_OK)
		goto fail;

	F->N = klu_zl_factor(
		A->handle_csc.col_ptr,
		A->handle_csc.row_ind,
		A->handle_csc.values,
		F->S,
		&F->common);

	if (!F->N) 
		goto fail;

	*out_F = F;
	return MATX_OK;

fail:
	klu_l_free_symbolic(&F->S, &F->common);
	free(F);
	return MATX_ERR_INTERNAL;
}

static matx_status_t ss_solve_csc_c64(
	const matx_factor_sparse_c64_t* F,
	const matx_vec_c64_t* b,
	matx_vec_c64_t* x)
{
	if (!F || !b || !x) return MATX_ERR_INVALID_ARG;

	matx_int64_t n = F->n;

	memcpy(x->data, b->data, sizeof(matx_double) * 2 * n);

	matx_int64_t status = klu_zl_solve(
		F->S, F->N, n, 1, (double*)x->data, &F->common);

	if (!status) return MATX_ERR_INTERNAL;

	return MATX_OK;
}

static void ss_factor_csc_c64_destroy(
	matx_factor_sparse_c64_t* F)
{
	if (!F) return;

	klu_l_free_numeric(&F->N, &F->common);
	klu_l_free_symbolic(&F->S, &F->common);
	free(F);
}

matx_sparse_linsolve_t matx_linsolve_make_suitesparse(void) {
	matx_sparse_linsolve_t ls;
	ls.kind = MATX_LINSOLVE_BACKEND_SUITESPARSE;

	ls.vt.factor_csc_f64 = &ss_factor_csc_f64;
	ls.vt.solve_csc_f64 = &ss_solve_csc_f64;
	ls.vt.factor_csc_f64_destroy = &ss_factor_csc_f64_destroy;

	ls.vt.factor_csc_c64 = &ss_factor_csc_c64;
	ls.vt.solve_csc_c64 = &ss_solve_csc_c64;
	ls.vt.factor_csc_c64_destroy = &ss_factor_csc_c64_destroy;

	return ls;
}