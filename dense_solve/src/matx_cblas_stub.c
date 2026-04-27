#include "matx/matx_dense_solve.h"
#include "matx/matx_log.h"
#include "matx/matx_types_internal.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#if MATX_ENABLE_OPENBLAS
	#include "cblas.h"
	#include "lapacke.h"
#elif MATX_ENABLE_LIBFLAME
	//#include "FLAME.h"
	#include "lapacke.h"
#endif

// Dense factorization (simple LU in C for now)
struct matx_factor_dense_f64_t {
	matx_int64_t n;
	matx_double* lu; // column-major, combined L+U
	matx_int64_t* piv;   // pivot indices, size n
	matx_layout_t layout;
};

struct matx_factor_dense_c64_t {
	matx_int64_t n;
	matx_double* lu;     // interleaved complex: [re,im,re,im,...] column-major
	matx_int64_t* piv;   // pivot indices
	matx_layout_t layout;
};

static void ss_factor_dense_f64_destroy(matx_factor_dense_f64_t* F) {
	if (!F) return;
	free(F->lu);
	free(F->piv);
	free(F);
}

static void ss_factor_dense_c64_destroy(
	matx_factor_dense_c64_t* F)
{
	if (!F) return;
	free(F->lu);
	free(F->piv);
	free(F);
}

// Dense real: LU + solve using LAPACK when available -----------------------
static matx_status_t ss_factor_dense_f64(const matx_dense_f64_t A,
	matx_factor_dense_f64_t** out_F) {
	if (!A || !out_F)
		return MATX_ERR_INVALID_ARG;

	if (A->nrows != A->ncols)
		return MATX_ERR_INVALID_ARG;

	const size_t n = A->nrows;

	if (*out_F != NULL)
	{
		ss_factor_dense_f64_destroy(*out_F);
	}

	matx_factor_dense_f64_t* F =
		(matx_factor_dense_f64_t*)malloc(sizeof(*F));
	if (!F) return MATX_ERR_OUT_OF_MEMORY;

	memset(F, 0, sizeof(*F));
	F->n = n;
	
	F->lu = (matx_double*)malloc(A->ncols * A->nrows * sizeof(matx_double));
	matx_int64_t piv_size = (A->nrows < A->ncols ? A->nrows : A->ncols);
	F->piv = (matx_int64_t*)malloc(piv_size * sizeof(matx_int64_t));
	matx_int64_t lda = A->layout == MATX_COL_MAJOR ? A->nrows : A->ncols;
	if (!F->lu || !F->piv) {
		free(F->lu);
		free(F->piv);
		free(F);
		return MATX_ERR_OUT_OF_MEMORY;
	}
	F->layout = A->layout;
	//cblas_dcopy(A->nrows * A->ncols, A->data, 1, F->lu, 1);

	memcpy(F->lu, A->data, A->nrows * A->ncols * sizeof(matx_double));

	matx_int64_t info = LAPACKE_dgetrf(
		A->layout == MATX_COL_MAJOR ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR,
		A->nrows,
		A->ncols,
		F->lu,
		lda,
		F->piv
	);

	if (info != 0)
	{
		MATX_ERROR("LAPACKE_dgetrf error:%d", info);
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
	if (!F || !b || !x) 
		return MATX_ERR_INVALID_ARG;
	const matx_int64_t n = F->n;
	
	//cblas_dcopy(n, b, 1, x, 1);
	memcpy(x, b, n * sizeof(matx_double));
	matx_int64_t N = (int)n;
	matx_int64_t nrhs = 1;
	matx_int64_t lda = (int)n;
	matx_int64_t ldb = (int)n;
	matx_int64_t info = 0;
	char trans = 'N';

	info = LAPACKE_dgetrs(F->layout == MATX_COL_MAJOR ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR, 
		trans, F->n, nrhs, F->lu, lda, F->piv, x, ldb);
	if (info != 0) 
	{
		MATX_ERROR("LAPACKE_dgetrs error:%d", info);
		return MATX_ERR_INTERNAL;
	}

	return MATX_OK;
}

static matx_status_t ss_factor_dense_c64(
	const matx_dense_c64_t A,
	matx_factor_dense_c64_t** out_F)
{
#if !(defined(MATX_HAVE_OPENBLAS) || !defined(MATX_HAVE_BLIS))
	return MATX_ERR_NOT_SUPPORTED;
#else
	if (!A || !out_F)
		return MATX_ERR_INVALID_ARG;

	if (A->nrows != A->ncols)
		return MATX_ERR_INVALID_ARG;

	if (*out_F != NULL)
	{
		ss_factor_dense_c64_destroy(*out_F);
	}

	const size_t n = A->nrows;

	matx_factor_dense_c64_t* F =
		(matx_factor_dense_c64_t*)malloc(sizeof(*F));
	if (!F) 
		return MATX_ERR_OUT_OF_MEMORY;

	memset(F, 0, sizeof(*F));
	F->n = n;

	F->lu = (matx_double*)malloc(A->ncols * A->nrows * sizeof(matx_complex_f64_t));
	matx_int64_t piv_size = (A->nrows < A->ncols ? A->nrows : A->ncols);
	F->piv = (matx_int64_t*)malloc(piv_size * sizeof(matx_int64_t));
	matx_int64_t lda = A->layout == MATX_COL_MAJOR ? A->nrows : A->ncols;
	if (!F->lu || !F->piv) {
		free(F->lu);
		free(F->piv);
		free(F);
		return MATX_ERR_OUT_OF_MEMORY;
	}
	F->layout = A->layout;
        //cblas_zcopy(A->nrows * A->ncols, A->data, 1, F->lu, 1);
        memcpy(F->lu, A->data, sizeof(matx_complex_f64_t) * A->nrows * A->ncols);
	matx_int64_t info = LAPACKE_zgetrf(
		A->layout == MATX_COL_MAJOR ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR,
		A->nrows,
		A->ncols,
		(lapack_complex_double*)F->lu,
		lda,
		F->piv
	);

	if (info != 0)
	{
		MATX_ERROR("LAPACKE_dgetrf error:%d", info);
		free(F->lu);
		free(F->piv);
		free(F);
		return MATX_ERR_INTERNAL;
	}

	*out_F = F;
	return MATX_OK;
#endif
}

static matx_status_t ss_solve_dense_c64(
	const matx_factor_dense_c64_t* F,
	const matx_vec_c64_t b,
	matx_vec_c64_t x)
{
	if (!F || !b || !x) 
		return MATX_ERR_INVALID_ARG;
	const matx_int64_t n = F->n;

        //cblas_zcopy(n, b->data, 1, x->data, 1);

        memcpy(x->data, b->data, sizeof(matx_complex_f64_t) * n);

	matx_int64_t N = (int)n;
	matx_int64_t nrhs = 1;
	matx_int64_t lda = (int)n;
	matx_int64_t ldb = (int)n;
	matx_int64_t info = 0;
	char trans = 'N';

	info = LAPACKE_zgetrs(F->layout == MATX_COL_MAJOR ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR, trans, F->n, 
		nrhs, (lapack_complex_double*)F->lu, lda, F->piv,
		(lapack_complex_double*)x->data, ldb);
	if (info != 0)
	{
		MATX_ERROR("LAPACKE_dgetrs error:%d", info);
		return MATX_ERR_INTERNAL;
	}

	return MATX_OK;
}

matx_dense_linsolve_t matx_dense_linsolve_make_cblas(void) {
	matx_dense_linsolve_t ls = {
		.kind = MATX_LINSOLVE_BACKEND_CBLAS,
		.vt = {
			.factor_dense_f64 = &ss_factor_dense_f64,
			.solve_dense_f64 = &ss_solve_dense_f64,
			.factor_dense_f64_destroy = &ss_factor_dense_f64_destroy,

			.factor_dense_c64 = &ss_factor_dense_c64,
			.solve_dense_c64 = &ss_solve_dense_c64,
			.factor_dense_c64_destroy = &ss_factor_dense_c64_destroy
		}
	};

	return ls;
}
