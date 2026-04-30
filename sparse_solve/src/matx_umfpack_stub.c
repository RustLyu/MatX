#include "matx/matx_sparse_solve.h"
#include "matx/matx_types_internal.h"

#include "matx/matx_log.h"

#if MATX_HAVE_UMFPACK
#include "umfpack.h"
#endif

struct matx_factor_sparse_f64_t {
	void* symbolic;
	void* numeric;
	matx_int64_t n;
	matx_int64_t nnz;
	matx_int64_t* Ap;
	matx_int64_t* Ai;
	matx_double* Ax;
};

struct matx_factor_sparse_c64_t {
	int unused;
};

static void umf_factor_csc_f64_destroy(matx_factor_sparse_f64_t* F)
{
	if (!F) return;
#if MATX_HAVE_UMFPACK
	if (F->numeric) umfpack_dl_free_numeric(&F->numeric);
	if (F->symbolic) umfpack_dl_free_symbolic(&F->symbolic);
#endif
	free(F->Ap);
	free(F->Ai);
	free(F->Ax);
	free(F);
}

static matx_status_t umf_factor_csc_f64(matx_coo_f64_t A, matx_factor_sparse_f64_t** out_F)
{
	if (!A || !out_F) return MATX_ERR_INVALID_ARG;
#if !MATX_HAVE_UMFPACK
	(void)A;
	(void)out_F;
	return MATX_ERR_NOT_SUPPORTED;
#else
	if (*out_F) umf_factor_csc_f64_destroy(*out_F);
	if (A->nrows != A->ncols || A->nrows <= 0) return MATX_ERR_INVALID_ARG;

	matx_status_t st = coo_to_csc_f64(A);
	if (st != MATX_OK) return st;
	st = coo_to_csc_f64_value_remap(A);
	if (st != MATX_OK) return st;

	matx_factor_sparse_f64_t* F = (matx_factor_sparse_f64_t*)calloc(1, sizeof(*F));
	if (!F) return MATX_ERR_OUT_OF_MEMORY;
	F->n = A->nrows;
	F->nnz = A->nnz;
	F->Ap = (matx_int64_t*)malloc(sizeof(matx_int64_t) * (size_t)(F->n + 1));
	F->Ai = (matx_int64_t*)malloc(sizeof(matx_int64_t) * (size_t)F->nnz);
	F->Ax = (matx_double*)malloc(sizeof(matx_double) * (size_t)F->nnz);
	if (!F->Ap || !F->Ai || !F->Ax) {
		umf_factor_csc_f64_destroy(F);
		return MATX_ERR_OUT_OF_MEMORY;
	}
	memcpy(F->Ap, A->handle_csc->col_ptr, sizeof(matx_int64_t) * (size_t)(F->n + 1));
	memcpy(F->Ai, A->handle_csc->row_ind, sizeof(matx_int64_t) * (size_t)F->nnz);
	memcpy(F->Ax, A->handle_csc->values, sizeof(matx_double) * (size_t)F->nnz);

	int status = umfpack_dl_symbolic(
		(int64_t)A->nrows, (int64_t)A->ncols,
		(const int64_t*)F->Ap, (const int64_t*)F->Ai, (const double*)F->Ax,
		&F->symbolic, NULL, NULL);
	if (status != UMFPACK_OK) {
		MATX_ERROR("umfpack_dl_symbolic failed status=%d", status);
		umf_factor_csc_f64_destroy(F);
		return MATX_ERR_INTERNAL;
	}

	status = umfpack_dl_numeric(
		(const int64_t*)F->Ap, (const int64_t*)F->Ai, (const double*)F->Ax,
		F->symbolic, &F->numeric, NULL, NULL);
	if (status != UMFPACK_OK) {
		MATX_ERROR("umfpack_dl_numeric failed status=%d", status);
		umf_factor_csc_f64_destroy(F);
		return MATX_ERR_INTERNAL;
	}

	*out_F = F;
	return MATX_OK;
#endif
}

static matx_status_t umf_solve_csc_f64(matx_factor_sparse_f64_t* F, const matx_double* b, matx_double* x)
{
	if (!F || !b || !x) return MATX_ERR_INVALID_ARG;
#if !MATX_HAVE_UMFPACK
	(void)F; (void)b; (void)x;
	return MATX_ERR_NOT_SUPPORTED;
#else
	int status = umfpack_dl_solve(UMFPACK_A,
		(const int64_t*)F->Ap, (const int64_t*)F->Ai, (const double*)F->Ax,
		(double*)x, (double*)b, F->numeric, NULL, NULL);
	if (status != UMFPACK_OK) {
		MATX_ERROR("umfpack_dl_solve failed status=%d", status);
		return MATX_ERR_INTERNAL;
	}
	return MATX_OK;
#endif
}

static matx_status_t umf_factor_csc_c64(matx_coo_c64_t A, matx_factor_sparse_c64_t** out_F)
{
	(void)A; (void)out_F;
	return MATX_ERR_NOT_SUPPORTED;
}

static matx_status_t umf_solve_csc_c64(matx_factor_sparse_c64_t* F, const matx_vec_c64_t b, matx_vec_c64_t x)
{
	(void)F; (void)b; (void)x;
	return MATX_ERR_NOT_SUPPORTED;
}

static void umf_factor_csc_c64_destroy(matx_factor_sparse_c64_t* F)
{
	free(F);
}

matx_sparse_linsolve_t matx_linsolve_make_umfpack(void)
{
	matx_sparse_linsolve_t ls = {
		.kind = MATX_LINSOLVE_BACKEND_UMFPACK,
		.vt = {
			.factor_csc_f64 = &umf_factor_csc_f64,
			.solve_csc_f64 = &umf_solve_csc_f64,
			.factor_csc_f64_destroy = &umf_factor_csc_f64_destroy,
			.factor_csc_c64 = &umf_factor_csc_c64,
			.solve_csc_c64 = &umf_solve_csc_c64,
			.factor_csc_c64_destroy = &umf_factor_csc_c64_destroy
		}
	};
	return ls;
}
