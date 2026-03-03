#include "matx/matx_dense_solve.h"

#include <limits.h>
#include <stdlib.h>

#if MATX_ENABLE_OPENBLAS
#include "openblas/cblas.h"
#elif MATX_ENABLE_LIBFLAME
#include "FLAME.h"
#endif

// Dense factorization (simple LU in C for now)
struct matx_factor_dense_f64_t {
	matx_int64_t n;
	matx_double* lu; // column-major, combined L+U
	matx_int64_t* piv;   // pivot indices, size n
};

// Dense real: LU + solve using LAPACK when available -----------------------
static matx_status_t ss_factor_dense_f64(const matx_dense_f64_t* A,
	matx_factor_dense_f64_t** out_F) {
	if (!A || !out_F) return MATX_ERR_INVALID_ARG;
	if (A->rows != A->cols) return MATX_ERR_INVALID_ARG;
	if (A->layout != MATX_COL_MAJOR) return MATX_ERR_NOT_SUPPORTED; // simplify: col-major only

	const size_t n = A->rows;
	matx_factor_dense_f64_t* F =
		(matx_factor_dense_f64_t*)malloc(sizeof(*F));
	if (!F) return MATX_ERR_OUT_OF_MEMORY;
	memset(F, 0, sizeof(*F));
	F->n = n;

	F->lu = (matx_double*)malloc(n * n * sizeof(matx_double));
	F->piv = (matx_int64_t*)malloc(n * sizeof(matx_int64_t));
	if (!F->lu || !F->piv) {
		free(F->lu);
		free(F->piv);
		free(F);
		return MATX_ERR_OUT_OF_MEMORY;
	}

	// Copy A into lu (column-major)
	for (size_t j = 0; j < n; ++j) {
		for (size_t i = 0; i < n; ++i) {
			F->lu[i + j * n] = A->data[i + j * A->stride];
		}
	}
	for (size_t i = 0; i < n; ++i) F->piv[i] = 0;

	matx_int64_t N = (matx_int64_t)n;
	matx_int64_t lda = (matx_int64_t)n;
	matx_int64_t info = 0;

	dgetrf_(&N, &N, F->lu, &lda, F->piv, &info);
	if (info != 0) {
		free(F->lu);
		free(F->piv);
		free(F);
		return MATX_ERR_INTERNAL;
	}

	*out_F = F;
	return MATX_OK;
}

static matx_status_t ss_solve_dense_f64(const matx_factor_dense_f64_t* F,
	const matx_double* b,
	matx_double* x) {
	if (!F || !b || !x) return MATX_ERR_INVALID_ARG;
	const matx_int64_t n = F->n;
	// Copy b into x
	for (matx_int64_t i = 0; i < n; ++i) {
		x[i] = b[i];
	}

	matx_int64_t N = (int)n;
	matx_int64_t nrhs = 1;
	matx_int64_t lda = (int)n;
	matx_int64_t ldb = (int)n;
	matx_int64_t info = 0;
	char trans = 'N';

	dgetrs_(&trans, &N, &nrhs, F->lu, &lda, F->piv, x, &ldb, &info);
	if (info != 0) {
		return MATX_ERR_INTERNAL;
	}

	return MATX_OK;
}

static void ss_factor_dense_f64_destroy(matx_factor_dense_f64_t* F) {
	if (!F) return;
	free(F->lu);
	free(F->piv);
	free(F);
}



static matx_status_t ss_factor_dense_c64(
	const matx_dense_c64_t* A,
	matx_factor_dense_c64_t** out_F)
{
#if !(defined(MATX_HAVE_OPENBLAS) || defined(MATX_HAVE_BLIS))
	return MATX_ERR_NOT_SUPPORTED;
#else
	if (!A || !out_F) return MATX_ERR_INVALID_ARG;
	if (A->rows != A->cols) return MATX_ERR_INVALID_ARG;
	if (A->layout != MATX_COL_MAJOR)
		return MATX_ERR_NOT_SUPPORTED;

	matx_int64_t n = A->rows;

	matx_factor_dense_c64_t* F =
		malloc(sizeof(*F));
	if (!F) return MATX_ERR_OUT_OF_MEMORY;

	F->n = n;

	F->lu = malloc(sizeof(matx_double) * 2 * n * n);
	F->piv = malloc(sizeof(matx_int64_t) * n);

	if (!F->lu || !F->piv) goto fail;

	/* copy matrix */
	for (size_t j = 0; j < n; ++j)
		for (size_t i = 0; i < n; ++i)
			memcpy(&F->lu[2 * (i + j * n)],
				&A->data[2 * (i + j * A->stride)],
				sizeof(matx_double) * 2);

	matx_int64_t N = (matx_int64_t)n;
	matx_int64_t lda = (matx_int64_t)n;
	matx_int64_t info = 0;

	zgetrf_(&N, &N, F->lu, &lda, F->piv, &info);

	if (info != 0) goto fail;

	*out_F = F;
	return MATX_OK;

fail:
	free(F->lu);
	free(F->piv);
	free(F);
	return MATX_ERR_INTERNAL;
#endif
}

static matx_status_t ss_solve_dense_c64(
	const matx_factor_dense_c64_t* F,
	const matx_vec_c64_t* b,
	matx_vec_c64_t* x)
{
	if (!F || !b || !x) return MATX_ERR_INVALID_ARG;

	size_t n = F->n;

	memcpy(x->data, b->data, sizeof(matx_double) * 2 * n);

	matx_int64_t N = (matx_int64_t)n;
	matx_int64_t nrhs = 1;
	matx_int64_t lda = (matx_int64_t)n;
	matx_int64_t ldb = (matx_int64_t)n;
	matx_int64_t info = 0;
	char trans = 'N';

	zgetrs_(&trans, &N, &nrhs,
		F->lu, &lda, F->piv,
		x->data, &ldb,
		&info);

	if (info != 0) return MATX_ERR_INTERNAL;

	return MATX_OK;
}

static void ss_factor_dense_c64_destroy(
	matx_factor_dense_c64_t* F)
{
	if (!F) return;
	free(F->lu);
	free(F->piv);
	free(F);
}


matx_dense_linsolve_t matx_dense_linsolve_make_cblas(void) {
	matx_dense_linsolve_t ls;
	ls.kind = MATX_LINSOLVE_BACKEND_CBLAS;

	ls.vt.factor_dense_f64 = &ss_factor_dense_f64;
	ls.vt.solve_dense_f64 = &ss_solve_dense_f64;
	ls.vt.factor_dense_f64_destroy = &ss_factor_dense_f64_destroy;

	ls.vt.factor_dense_c64 = &ss_factor_dense_c64;
	ls.vt.solve_dense_c64 = &ss_solve_dense_c64;
	ls.vt.factor_dense_c64_destroy = &ss_factor_dense_c64_destroy;

	return ls;
}