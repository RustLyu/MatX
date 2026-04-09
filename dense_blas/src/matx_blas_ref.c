#include "matx/matx_dense_compute.h"
#include <string.h>
#if MATX_ENABLE_OPENBLAS
	#include "cblas.h"
	#include "lapack.h"
#elif MATX_ENABLE_BLIS
	#include "blis.h"
	#include "FLAME.h"
#endif

static matx_status_t ref_dgemm(matx_layout_t layout,
	matx_int64_t trans_a,
	matx_int64_t trans_b,
	matx_int64_t m,
	matx_int64_t n,
	matx_int64_t k,
	matx_double alpha,
	const matx_double* a,
	matx_int64_t lda,
	const matx_double* b,
	matx_int64_t ldb,
	matx_double beta,
	matx_double* c,
	matx_int64_t ldc) {
	if (!a || !b || !c) 
		return MATX_ERR_INVALID_ARG;
	if (layout != MATX_COL_MAJOR && layout != MATX_ROW_MAJOR) 
		return MATX_ERR_INVALID_ARG;

	if (m > INT_MAX || n > INT_MAX || k > INT_MAX ||
		lda > INT_MAX || ldb > INT_MAX || ldc > INT_MAX) {
		return MATX_ERR_NOT_SUPPORTED;
	}

	const enum CBLAS_ORDER order =
		(layout == MATX_COL_MAJOR) ? CblasColMajor : CblasRowMajor;

	const enum CBLAS_TRANSPOSE ta =
		(trans_a != 0) ? CblasTrans : CblasNoTrans;
	const enum CBLAS_TRANSPOSE tb =
		(trans_b != 0) ? CblasTrans : CblasNoTrans;
	cblas_dgemm(order,
		ta,
		tb,
		m,
		n,
		k,
		alpha,
		a,
		lda,
		b,
		ldb,
		beta,
		c,
		ldc);
	return MATX_OK;
}

static matx_status_t ref_zgemm(matx_layout_t layout,
	matx_int64_t trans_a,
	matx_int64_t trans_b,
	matx_int64_t m,
	matx_int64_t n,
	matx_int64_t k,
	const void* alpha,
	const void* A,
	matx_int64_t lda,
	const void* B,
	matx_int64_t ldb,
	const void* beta,
	void* C,
	matx_int64_t ldc) {
	if (!A || !B || !C || !alpha || !beta)
		return MATX_ERR_INVALID_ARG;

	if (m > INT_MAX || n > INT_MAX || k > INT_MAX ||
		lda > INT_MAX || ldb > INT_MAX || ldc > INT_MAX)
		return MATX_ERR_NOT_SUPPORTED;

	const enum CBLAS_ORDER order =
		(layout == MATX_COL_MAJOR) ? CblasColMajor : CblasRowMajor;

	const enum CBLAS_TRANSPOSE ta =
		trans_a ? CblasTrans : CblasNoTrans;

	const enum CBLAS_TRANSPOSE tb =
		trans_b ? CblasTrans : CblasNoTrans;
	cblas_zgemm(order, ta, tb,
		m, n, k,
		alpha,
		A, lda,
		B, ldb,
		beta,
		C, ldc);

	return MATX_OK;
}

static matx_status_t ref_zgemv(matx_layout_t layout,
        matx_int64_t trans_a,
	matx_int64_t m,
	matx_int64_t n,
	const void* alpha,
	const void* A,
	matx_int64_t lda,
	const void* X,
	matx_int64_t ldx,
	const void* beta,
	void* C,
	matx_int64_t ldc) {
	if (!A || !X || !C || !alpha || !beta)
		return MATX_ERR_INVALID_ARG;

	if (m > INT_MAX || n > INT_MAX ||
		lda > INT_MAX || ldx > INT_MAX || ldc > INT_MAX)
		return MATX_ERR_NOT_SUPPORTED;

	const enum CBLAS_ORDER order =
		(layout == MATX_COL_MAJOR) ? CblasColMajor : CblasRowMajor;

	const enum CBLAS_TRANSPOSE ta =
		trans_a ? CblasTrans : CblasNoTrans;
	cblas_zgemv(order, ta,
		m, n,
		alpha,
		A, lda,
		X, ldx,
		beta,
		C, ldc);

	return MATX_OK;
}

static matx_status_t ref_daxpy(
	matx_int64_t n,
	matx_double alpha,
	const matx_double* x,
	matx_int64_t lda,
	void* y,
	matx_int64_t ldy) {
	if (!x || !y)
		return MATX_ERR_INVALID_ARG;
	if (lda != ldy)
		return MATX_ERR_INVALID_ARG;

	cblas_daxpy(n, alpha, x, lda, y,
		ldy);
	return MATX_OK;
}

static matx_status_t ref_zaxpy(
	matx_int64_t n,
	const void* alpha,
	const void* x,
	matx_int64_t lda,
	void* y,
	matx_int64_t ldy) {
	if (!x || !y)
		return MATX_ERR_INVALID_ARG;
	if (lda != ldy)
		return MATX_ERR_INVALID_ARG;

	cblas_zaxpy(n, alpha, x, lda, y,
		ldy);
	return MATX_OK;
}

static matx_status_t ref_dgemv(matx_layout_t layout,
	matx_int64_t trans_a,
	matx_int64_t m,
	matx_int64_t n,
	matx_double alpha,
	const matx_double* A,
	matx_int64_t lda,
	matx_double* B,
	matx_int64_t ldb,
	matx_double beta,
	matx_double* C,
	matx_int64_t ldc)
{
	if (!A || !C)
		return MATX_ERR_INVALID_ARG;

	if (m > INT_MAX || n > INT_MAX ||
		lda > INT_MAX || ldc > INT_MAX)
		return MATX_ERR_NOT_SUPPORTED;

	const enum CBLAS_ORDER order =
		(layout == MATX_COL_MAJOR) ? CblasColMajor : CblasRowMajor;

	const enum CBLAS_TRANSPOSE ta =
		trans_a ? CblasTrans : CblasNoTrans;

	cblas_dgemv(order, ta,
		m, n,
		alpha,
		A, lda,
		B, ldb,
		beta,
		C, ldc);

	return MATX_OK;
}

static matx_status_t ref_dgeadd(matx_layout_t trans_a,
	matx_int64_t rows,
	matx_int64_t cols,
	matx_double alpha,
	const matx_double* A,
	matx_int64_t lda,
	matx_double beta,
	matx_double* B,
	matx_int64_t ldb)
{
	if (!A || !B)
		return MATX_ERR_INVALID_ARG;

	if (rows > INT_MAX || cols > INT_MAX ||
		lda > INT_MAX || ldb > INT_MAX)
		return MATX_ERR_NOT_SUPPORTED;

#if MATX_ENABLE_OPENBLAS
	const enum CBLAS_ORDER order =
		(trans_a == MATX_COL_MAJOR) ? CblasColMajor : CblasRowMajor;

	cblas_dgeadd(
		order,
		rows, 
		cols,
		alpha,
		A, lda,
		beta,
		B, ldb
	);
#elif MATX_ENABLE_BLIS

	if (trans_a == MATX_ROW_MAJOR)
	{
		if (lda == cols && ldb == cols)
		{
			matx_int64_t len = rows * cols;

			if (beta != 1.0)
				cblas_dscal(len, beta, B, 1);

			if (alpha != 0.0)
				cblas_daxpy(len, alpha, A, 1, B, 1);

			return MATX_OK;
		}
	}
	else
	{
		if (lda == rows && ldb == rows)
		{
			matx_int64_t len = rows * cols;

			if (beta != 1.0)
				cblas_dscal(len, beta, B, 1);

			if (alpha != 0.0)
				cblas_daxpy(len, alpha, A, 1, B, 1);

			return MATX_OK;
		}
	}

	for (size_t j = 0; j < cols; ++j)
	{
		cblas_dscal(rows, beta, B + j * ldb, 1);
		cblas_daxpy(rows, alpha, A + j * lda, 1, B + j * ldb, 1);
	}
#endif
	return MATX_OK;
}

static matx_status_t ref_zgeadd(matx_layout_t trans_a,
	matx_int64_t rows,
	matx_int64_t cols,
	const void* alpha,
	const void* A,
	matx_int64_t lda,
	const void* beta,
	void* B,
	matx_int64_t ldb)
{
	if (!A || !B)
		return MATX_ERR_INVALID_ARG;

	if (rows > INT_MAX || cols > INT_MAX ||
		lda > INT_MAX || ldb > INT_MAX)
		return MATX_ERR_NOT_SUPPORTED;
#if MATX_ENABLE_OPENBLAS
	const enum CBLAS_ORDER order =
		(trans_a == MATX_COL_MAJOR) ? CblasColMajor : CblasRowMajor;

	cblas_zgeadd(
		order,
		rows, cols,
		alpha,
		A, lda,
		beta,
		B, ldb
	);
#elif MATX_ENABLE_BLIS

	const void* alpha_p = alpha;
	const void* beta_p = beta;

	size_t len;

	if (trans_a == MATX_ROW_MAJOR)
	{
		if (lda == cols && ldb == cols)
		{
			len = rows * cols;
			cblas_zscal(len, beta_p, B, 1);
			cblas_zaxpy(len, alpha_p, A, 1, B, 1);

			return MATX_OK;
		}
	}
	else
	{
		if (lda == rows && ldb == rows)
		{
			len = rows * cols;

			cblas_zscal((matx_int64_t)len, beta_p, B, 1);
			cblas_zaxpy((matx_int64_t)len, alpha_p, A, 1, B, 1);

			return MATX_OK;
		}
	}

	for (matx_int64_t j = 0; j < cols; ++j)
	{
		void* Bcol = (char*)B + j * ldb * sizeof(matx_double) * 2;
		const void* Acol = (const char*)A + j * lda * sizeof(matx_double) * 2;

		cblas_zscal(rows, beta_p, Bcol, 1);
		cblas_zaxpy(rows, alpha_p, Acol, 1, Bcol, 1);
	}
#endif
	return MATX_OK;
}

matx_status_t ref_inv_dense_f64(
	matx_layout_t layout,
	matx_int64_t rows,
	matx_int64_t cols,
	const matx_double* A,
	matx_double* out_Ainv)
{
	if (!A || !out_Ainv) 
		return MATX_ERR_INVALID_ARG;
	if (rows != cols) 
		return MATX_ERR_INVALID_ARG;
	if (layout != MATX_COL_MAJOR) 
		return MATX_ERR_NOT_SUPPORTED;
	memcpy(out_Ainv, A, sizeof(matx_double) * rows * cols);

	matx_int64_t N = rows;
	matx_int64_t lda = rows;
	matx_int64_t info = 0;

	matx_int64_t* piv = (matx_int64_t*)malloc(rows * sizeof(matx_int64_t));
	if (!piv) {
		free(out_Ainv);
		return MATX_ERR_OUT_OF_MEMORY;
	}

	dgetrf_(&N, &N, out_Ainv, &lda, piv, &info);
	if (info != 0) {
		free(piv);
		return MATX_ERR_INTERNAL;
	}

	matx_int64_t lwork = -1;
	matx_double work_query;

	dgetri_(&N, out_Ainv, &lda,
		piv,
		&work_query, &lwork, &info);

	if (info != 0) {
		free(piv);
		return MATX_ERR_INTERNAL;
	}

	matx_double* work = (matx_double*)malloc(lwork * sizeof(matx_double));
	if (!work) {
		free(piv);
		return MATX_ERR_OUT_OF_MEMORY;
	}

	dgetri_(&N, out_Ainv, &lda,
		piv,
		work, &lwork, &info);

	free(work);
	free(piv);

	if (info != 0) {
		return MATX_ERR_INTERNAL;
	}
	return MATX_OK;
}

matx_status_t ref_inv_dense_c64(
	matx_layout_t layout,
	matx_int64_t rows,
	matx_int64_t cols,
	const void* A,
	void* out_Ainv)
{
	if (!A || !out_Ainv)
		return MATX_ERR_INVALID_ARG;
	if (rows != cols)
		return MATX_ERR_INVALID_ARG;
	if (layout != MATX_COL_MAJOR)
		return MATX_ERR_NOT_SUPPORTED;
	memcpy(out_Ainv, A, sizeof(matx_complex_f64) * rows * cols);

	matx_int64_t N = rows;
	matx_int64_t lda = rows;
	matx_int64_t info = 0;

	matx_int64_t* piv = (matx_int64_t*)malloc(rows * sizeof(matx_int64_t));
	if (!piv) {
		return MATX_ERR_OUT_OF_MEMORY;
	}

	zgetrf_(&N, &N, out_Ainv, &lda, piv, &info);
	if (info != 0) {
		free(piv);
		return MATX_ERR_INTERNAL;
	}

	matx_int64_t lwork = -1;
	matx_complex_f64 work_query;

        LAPACK_zgetri(&N, out_Ainv, &lda,
		piv,
                (void*)&work_query, &lwork, &info);

	if (info != 0) {
		free(piv);
		return MATX_ERR_INTERNAL;
	}

	matx_complex_f64* work = (matx_complex_f64*)malloc(lwork * sizeof(matx_complex_f64));
	if (!work) {
		free(piv);
		return MATX_ERR_OUT_OF_MEMORY;
	}

	zgetri_(&N, out_Ainv, &lda,
		piv,
                (void*)work, &lwork, &info);

	free(work);
	free(piv);

	if (info != 0) {
		return MATX_ERR_INTERNAL;
	}
	return MATX_OK;
}

matx_dense_backend_t matx_blas_make_reference(void) {
	matx_dense_backend_t b = {
		.kind = MATX_BLAS_BACKEND_OPENBLAS,
		.vt = {
			.dgemm = &ref_dgemm,
			.zgemm = &ref_zgemm,
			.dgemv = &ref_dgemv,
			.zgemv = &ref_zgemv,
			.daxpy = &ref_daxpy,
			.zaxpy = &ref_zaxpy,
			.dgeadd = &ref_dgeadd,
			.zgeadd = &ref_zgeadd,
			.inv_dense_f64 = &ref_inv_dense_f64,
			.inv_dense_c64 = &ref_inv_dense_c64
		}
	};

	return b;
}

