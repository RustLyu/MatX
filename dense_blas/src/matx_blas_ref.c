#include "matx/matx_dense_compute.h"

#include <limits.h>

#if MATX_ENABLE_OPENBLAS
	#include "openblas/cblas.h"
#elif MATX_ENABLE_BLIS
	#include "blis.h"
#endif

static matx_status_t ref_dgemm(matx_layout_t layout,
	matx_int64_t trans_a,
	matx_int64_t trans_b,
	size_t m,
	size_t n,
	size_t k,
	matx_double alpha,
	const matx_double* a,
	size_t lda,
	const matx_double* b,
	size_t ldb,
	matx_double beta,
	matx_double* c,
	size_t ldc) {
	if (!a || !b || !c) return MATX_ERR_INVALID_ARG;
	if (layout != MATX_COL_MAJOR && layout != MATX_ROW_MAJOR) return MATX_ERR_INVALID_ARG;

	if (m > (size_t)INT_MAX || n > (size_t)INT_MAX || k > (size_t)INT_MAX ||
		lda > (size_t)INT_MAX || ldb > (size_t)INT_MAX || ldc > (size_t)INT_MAX) {
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
		(int)m,
		(int)n,
		(int)k,
		alpha,
		a,
		(int)lda,
		b,
		(int)ldb,
		beta,
		c,
		(int)ldc);
	return MATX_OK;
}

static matx_status_t ref_zgemm(matx_layout_t layout,
	matx_int64_t trans_a,
	matx_int64_t trans_b,
	size_t m,
	size_t n,
	size_t k,
	const void* alpha,
	const void* A,
	size_t lda,
	const void* B,
	size_t ldb,
	const void* beta,
	void* C,
	size_t ldc) {
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
		(int)m, (int)n, (int)k,
		alpha,
		A, (int)lda,
		B, (int)ldb,
		beta,
		C, (int)ldc);

	return MATX_OK;
}

static matx_status_t ref_zgemv(matx_layout_t layout,
	int trans_a,
	size_t m,
	size_t n,
	const void* alpha,
	const void* A,
	size_t lda,
	const void* X,
	size_t ldx,
	const void* beta,
	void* C,
	size_t ldc) {
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
		(int)m, (int)n,
		alpha,
		A, (int)lda,
		X, (int)ldx,
		beta,
		C, (int)ldc);

	return MATX_OK;
}

static matx_status_t ref_daxpy(
	size_t n,
	matx_double alpha,
	const matx_double* x,
	size_t lda,
	const void* y,
	size_t ldy) {
	if (!x || !y)
		return MATX_ERR_INVALID_ARG;
	if (lda != ldy)
		return MATX_ERR_INVALID_ARG;

	cblas_daxpy(n, alpha, x, lda, y,
		ldy);
	return MATX_OK;
}

static matx_status_t ref_zaxpy(
	size_t n,
	const void* alpha,
	const void* x,
	size_t lda,
	const void* y,
	size_t ldy) {
	if (!x || !y)
		return MATX_ERR_INVALID_ARG;
	if (lda != ldy)
		return MATX_ERR_INVALID_ARG;

	cblas_zaxpy((int)n, alpha, x, (int)lda, y,
		(int)ldy);
	return MATX_OK;
}

static matx_status_t ref_dgemv(matx_layout_t layout,
	matx_int64_t trans_a,
	size_t m,
	size_t n,
	matx_double alpha,
	const matx_double* A,
	size_t lda,
	matx_double* B,
	size_t ldb,
	matx_double beta,
	matx_double* C,
	size_t ldc)
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
		(int)m, (int)n,
		alpha,
		A, (int)lda,
		B, (int)ldb,
		beta,
		C, (int)ldc);

	return MATX_OK;
}

static matx_status_t ref_dgeadd(matx_layout_t trans_a,
	size_t rows,
	size_t cols,
	matx_double alpha,
	const matx_double* A,
	size_t lda,
	matx_double beta,
	matx_double* B,
	size_t ldb)
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
		(int)rows, (int)cols,
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
				cblas_dscal((int)len, beta, B, 1);

			if (alpha != 0.0)
				cblas_daxpy((int)len, alpha, A, 1, B, 1);

			return MATX_OK;
		}
	}
	else
	{
		if (lda == rows && ldb == rows)
		{
			matx_int64_t len = rows * cols;

			if (beta != 1.0)
				cblas_dscal((int)len, beta, B, 1);

			if (alpha != 0.0)
				cblas_daxpy((int)len, alpha, A, 1, B, 1);

			return MATX_OK;
		}
	}

	for (size_t j = 0; j < cols; ++j)
	{
		cblas_dscal((int)rows, beta, B + j * ldb, 1);
		cblas_daxpy((int)rows, alpha, A + j * lda, 1, B + j * ldb, 1);
	}
#endif
	return MATX_OK;
}

static matx_status_t ref_zgeadd(matx_layout_t trans_a,
	size_t rows,
	size_t cols,
	const void* alpha,
	const void* A,
	size_t lda,
	const void* beta,
	void* B,
	size_t ldb)
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
		(int)rows, (int)cols,
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
			cblas_zscal((int)len, beta_p, B, 1);
			cblas_zaxpy((int)len, alpha_p, A, 1, B, 1);

			return MATX_OK;
		}
	}
	else
	{
		if (lda == rows && ldb == rows)
		{
			len = rows * cols;

			cblas_zscal((int)len, beta_p, B, 1);
			cblas_zaxpy((int)len, alpha_p, A, 1, B, 1);

			return MATX_OK;
		}
	}

	for (int j = 0; j < (int)cols; ++j)
	{
		void* Bcol = (char*)B + j * ldb * sizeof(double) * 2;
		const void* Acol = (const char*)A + j * lda * sizeof(double) * 2;

		cblas_zscal((int)rows, beta_p, Bcol, 1);
		cblas_zaxpy((int)rows, alpha_p, Acol, 1, Bcol, 1);
	}
#endif
	return MATX_OK;
}

matx_dense_backend_t matx_blas_make_reference(void) {
	matx_dense_backend_t b;
	b.kind = MATX_BLAS_BACKEND_REFERENCE;
	b.vt.dgemm = &ref_dgemm;
	b.vt.zgemm = &ref_zgemm;
	b.vt.dgemv = &ref_dgemv;
	b.vt.zgemv = &ref_zgemv;
	b.vt.daxpy = &ref_daxpy;
	b.vt.zaxpy = &ref_zaxpy;
	b.vt.dgeadd = &ref_dgeadd;
	b.vt.zgeadd = &ref_zgeadd;
	return b;
}

