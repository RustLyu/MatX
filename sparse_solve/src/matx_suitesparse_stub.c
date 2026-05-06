#include "matx/matx_sparse_solve.h"
#include "matx/matx_types_internal.h"

#include <limits.h>

#include "klu.h"
#include "matx/matx_log.h"

typedef struct matx_factor_sparse_f64_klu{
	klu_l_symbolic* S;
	klu_l_numeric* N;
	klu_l_common common;
	matx_int64_t n;
} matx_factor_sparse_f64_klu_t;

typedef struct matx_factor_sparse_c64_klu {
	klu_l_symbolic* S;
	klu_l_numeric* N;
	klu_l_common common;
	matx_int64_t n;
} matx_factor_sparse_c64_klu_t;

static void ss_factor_csc_f64_destroy(matx_factor_sparse_f64_t* F) {
	if (!F || !F->reserved)
		return;
	matx_factor_sparse_f64_klu_t* ptr = (matx_factor_sparse_f64_klu_t*)F->reserved;
	klu_l_free_numeric(&ptr->N, &ptr->common);
	klu_l_free_symbolic(&ptr->S, &ptr->common);
	free(ptr);
}

static void ss_factor_csc_c64_destroy(
	matx_factor_sparse_c64_t* F)
{
	if (!F || !F->reserved)
		return;
	matx_factor_sparse_c64_klu_t* ptr = (matx_factor_sparse_c64_klu_t*)F->reserved;
	klu_zl_free_numeric(&ptr->N, &ptr->common);
	klu_l_free_symbolic(&ptr->S, &ptr->common);
	free(ptr);
}

// Sparse real: KLU-based ---------------------------------------------------
static matx_status_t ss_factor_csc_f64(matx_coo_f64_t A,
	matx_factor_sparse_f64_t* out_F) {
	if (!A || !out_F)
	{
		MATX_ERROR("input pointer is null error");
		return MATX_ERR_INVALID_ARG;
	}
	if (out_F != NULL)
	{
		ss_factor_csc_f64_destroy(out_F);
	}

	matx_status_t st = coo_to_csc_f64(A);
	if (st != MATX_OK)
	{
		MATX_ERROR("coo_to_csc_f64 error:%d", st);
		return st;
	}
	if (!A->handle_csc->col_ptr || !A->handle_csc->row_ind || !A->handle_csc->values)
	{
		MATX_ERROR("csc pointer is null error");
		return MATX_ERR_INVALID_ARG;
	}
	if (A->nrows != A->ncols) 
		return MATX_ERR_INVALID_ARG;

	const matx_int64_t n = A->nrows;
	if (n == 0) 
		return MATX_ERR_INVALID_ARG;
	if (n > (matx_int64_t)INT_MAX)
	{
		MATX_ERROR("n > INT_MAX");
		return MATX_ERR_NOT_SUPPORTED;
	}

	matx_factor_sparse_f64_klu_t* F =
		(matx_factor_sparse_f64_klu_t*)malloc(sizeof(*F));
	if (!F)
	{
		MATX_ERROR("malloc Factor handle error");
		return MATX_ERR_OUT_OF_MEMORY;
	}

	memset(F, 0, sizeof(*F));
	F->n = (matx_int64_t)n;
	klu_l_defaults(&F->common);

	F->S = klu_l_analyze(F->n,
		A->handle_csc->col_ptr,
		A->handle_csc->row_ind,
		&F->common);
	if (!F->S) {
		MATX_ERROR("call klu_l_analyze error:%d", F->common.status);
		free(F);
		return MATX_ERR_INTERNAL;
	}

	st = coo_to_csc_f64_value_remap(A);

	if(st != MATX_OK) {
		MATX_ERROR("call coo_to_csc_f64_value_remap error:%d", F->common.status);
		klu_l_free_symbolic(&F->S, &F->common);
		free(F);
		return st;
	}

	F->N = klu_l_factor(A->handle_csc->col_ptr,
		A->handle_csc->row_ind,
		A->handle_csc->values,
		F->S,
		&F->common);

	if (!F->N) {
		MATX_ERROR("call klu_l_factor error:%d", F->common.status);
		klu_l_free_symbolic(&F->S, &F->common);
		free(F);
		return MATX_ERR_INTERNAL;
	}

	out_F->reserved = F;
	return MATX_OK;
}

static matx_status_t ss_solve_csc_f64(matx_factor_sparse_f64_t* F,
	const matx_double* b,
	matx_double* x) {
	if (!F || !F->reserved || !b || !x)
	{
		MATX_ERROR("input pointer is null");
		return MATX_ERR_INVALID_ARG;
	}
	matx_factor_sparse_f64_klu_t* ptr = (matx_factor_sparse_f64_klu_t*)F->reserved;
	const matx_int64_t n = ptr->n;
	for (matx_int64_t i = 0; i < n; ++i) {
		x[i] = b[i];
	}

	const int status = klu_l_solve(ptr->S, ptr->N, n, 1, x, &ptr->common);
	if (!status)
	{
		MATX_ERROR("KLU solve failed with status %d", status);
		return MATX_ERR_INTERNAL;
	}
	return MATX_OK;
}

static matx_status_t ss_factor_csc_c64(
	matx_coo_c64_t A,
	matx_factor_sparse_c64_t* out_F)
{
	if (!A || !out_F)
	{
		MATX_ERROR("input pointer is null");
		return MATX_ERR_INVALID_ARG;
	}

	if (out_F != NULL)
	{
		ss_factor_csc_c64_destroy(out_F);
	}

	matx_status_t st = coo_to_csc_c64(A);
	if (st != MATX_OK)
	{
		MATX_ERROR("coo_to_csc_c64 error");
		return st;
	}

	if (!A->handle_csc->col_ptr || !A->handle_csc->row_ind || !A->handle_csc->values)
	{
		MATX_ERROR("csc pointer error");
		return MATX_ERR_INVALID_ARG;
	}
	if (A->nrows != A->ncols)
		return MATX_ERR_INVALID_ARG;

	const matx_int64_t n = A->nrows;
	if (n > INT_MAX)
	{
		MATX_ERROR("n > INT_MAX");
		return MATX_ERR_NOT_SUPPORTED;
	}

	matx_factor_sparse_c64_klu_t* F =
		(matx_factor_sparse_c64_klu_t*)malloc(sizeof(*F));
	if (!F)
	{
		MATX_ERROR("malloc F failed");
		return MATX_ERR_OUT_OF_MEMORY;
	}

	memset(F, 0, sizeof(*F));
	F->n = (matx_int64_t)n;

	klu_l_defaults(&F->common);

	F->S = klu_l_analyze(
		F->n,
		A->handle_csc->col_ptr,
		A->handle_csc->row_ind,
		&F->common);

	if (!F->S)
	{
		MATX_ERROR("klu_l_analyze failed error:%d", F->common.status);
		goto fail;
	}

	st = coo_to_csc_c64_value_remap(A);
	if (st != MATX_OK)
	{
		MATX_ERROR("coo_to_csc_c64_value_remap failed");
		goto fail;
	}

	F->N = klu_zl_factor(
		A->handle_csc->col_ptr,
		A->handle_csc->row_ind,
		(double*)A->handle_csc->values,
		F->S,
		&F->common);

	if (!F->N)
	{
		MATX_ERROR("klu_zl_factor failed error:%d", F->common.status);
		goto fail;
	}

	out_F->reserved = F;
	return MATX_OK;

fail:
	klu_l_free_symbolic(&F->S, &F->common);
	free(F);
	return MATX_ERR_INTERNAL;
}

static matx_status_t ss_solve_csc_c64(
	matx_factor_sparse_c64_t* F,
	const matx_vec_c64_t b,
	matx_vec_c64_t x)
{
	if (!F || !b || !x)
		return MATX_ERR_INVALID_ARG;
	if (b->stride != 1 || x->stride != 1)
		return MATX_ERR_NOT_SUPPORTED;
	matx_factor_sparse_c64_klu_t* ptr = (matx_factor_sparse_c64_klu_t*)F->reserved;
	matx_int64_t n = ptr->n;

	memcpy(x->data, b->data, sizeof(matx_complex_f64_t) * n);

	matx_int64_t status = klu_zl_solve(
		ptr->S, ptr->N, n, 1, (matx_double*)x->data, &ptr->common);

	if (!status)
	{
		MATX_ERROR("klu_zl_solve failed");
		return MATX_ERR_INTERNAL;
	}

	return MATX_OK;
}

matx_sparse_linsolve_t matx_linsolve_make_suitesparse_klu(void) {
	matx_sparse_linsolve_t ls =
	{
		.kind = MATX_LINSOLVE_BACKEND_SUITESPARSE_KLU,
		.vt = {
			.factor_csc_f64 = &ss_factor_csc_f64,
			.solve_csc_f64 = &ss_solve_csc_f64,
			.factor_csc_f64_destroy = &ss_factor_csc_f64_destroy,
			.factor_csc_c64 = &ss_factor_csc_c64,
			.solve_csc_c64 = &ss_solve_csc_c64,
			.factor_csc_c64_destroy = &ss_factor_csc_c64_destroy
		}
	};
	return ls;
}