/* Sparse complex f64: spmv and spmm. Uses GraphBLAS complex when available. */
#include "matx/matx_compute.h"
#include "matx/matx_c64_utils.h"
#include "matx/matx.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include <suitesparse/GraphBLAS.h>

matx_status_t matx_spmv_csc_c64(
	matx_complex_f64 alpha,
	const matx_csc_c64_t* A,
	const matx_vec_c64_t* x,
	matx_complex_f64 beta,
	matx_vec_c64_t* y)
{
	GrB_Info info = GrB_init(GrB_NONBLOCKING);
	if (!A || !x || !y)
		return MATX_ERR_INVALID_ARG;

	if (A->ncols != x->n || A->nrows != y->n)
		return MATX_ERR_INVALID_ARG;

	/* ---------------- build GraphBLAS matrix ---------------- */

	GrB_Matrix gA;
	info = GrB_Matrix_new(&gA, GxB_FC64, A->nrows, A->ncols);
	if (info != GrB_SUCCESS) return MATX_ERR_INTERNAL;

	for (size_t j = 0; j < A->ncols; j++)
	{
		for (int k = A->col_ptr[j]; k < A->col_ptr[j + 1]; k++)
		{
			GxB_FC64_t val =
			{ A->values[k].real, A->values[k].imag };

			GxB_Matrix_setElement_FC64(
				gA, val, A->row_ind[k], j);
		}
	}

	/* ---------------- build vectors ---------------- */

	GrB_Vector gx, gy;

	GrB_Vector_new(&gx, GxB_FC64, x->n);
	GrB_Vector_new(&gy, GxB_FC64, y->n);

	for (size_t i = 0; i < x->n; i++)
	{
		GxB_FC64_t v =
		{ x->data[i * x->stride].real,
		 x->data[i * x->stride].imag };

		GxB_Vector_setElement_FC64(gx, v, i);
	}

	for (size_t i = 0; i < y->n; i++)
	{
		GxB_FC64_t v =
		{ y->data[i * y->stride].real,
		 y->data[i * y->stride].imag };

		GxB_Vector_setElement_FC64(gy, v, i);
	}

	/* ---------------- gy = alpha*A*x + beta*y ---------------- */

	GxB_FC64_t a = { alpha.real, alpha.imag };
	GxB_FC64_t b = { beta.real, beta.imag };

	GrB_Vector temp;
	GrB_Vector_new(&temp, GxB_FC64, y->n);
	// temp = A*x
	GrB_mxv(temp, NULL, NULL, GxB_PLUS_TIMES_FC64, gA, gx, NULL);
	// temp = alpha*temp
	GrB_apply(temp, NULL, NULL, GxB_TIMES_FC64, temp, &a, NULL);
	// gy = beta*gy
	GrB_apply(gy, NULL, NULL, GxB_TIMES_FC64, gy, &b, NULL);
	// gy = temp + gy
	GrB_eWiseAdd(gy, NULL, NULL, GxB_PLUS_FC64, temp, gy, NULL);
	GrB_Vector_free(&temp);

	/* ---------------- copy back ---------------- */

	for (size_t i = 0; i < y->n; i++)
	{
		GxB_FC64_t v;
		if (GxB_Vector_extractElement_FC64(&v, gy, i) == GrB_SUCCESS)
		{
			y->data[i * y->stride].real = v._Val[0];
			y->data[i * y->stride].imag = v._Val[1];
		}
	}

	GrB_free(&gx);
	GrB_free(&gy);
	GrB_free(&gA);

	return MATX_OK;
}

matx_status_t matx_spmm_csc_c64(
	matx_complex_f64 alpha,
	const matx_csc_c64_t* A,
	const matx_dense_c64_t* B,
	matx_complex_f64 beta,
	matx_dense_c64_t* C)
{
	GrB_Info info = GrB_init(GrB_NONBLOCKING);
	if (!A || !B || !C)
		return MATX_ERR_INVALID_ARG;

	if (A->ncols != B->rows ||
		A->nrows != C->rows ||
		B->cols != C->cols)
		return MATX_ERR_INVALID_ARG;

	GrB_Matrix gA, gB, gC;

	/* build A */
	GrB_Matrix_new(&gA, GxB_FC64, A->nrows, A->ncols);

	for (size_t j = 0; j < A->ncols; j++)
	{
		for (int k = A->col_ptr[j]; k < A->col_ptr[j + 1]; k++)
		{
			GxB_FC64_t v =
			{ A->values[k].real,A->values[k].imag };

			GxB_Matrix_setElement_FC64(
				gA, v, A->row_ind[k], j);
		}
	}

	/* build B */
	GrB_Matrix_new(&gB, GxB_FC64, B->rows, B->cols);

	for (size_t j = 0; j < B->cols; j++)
	{
		for (size_t i = 0; i < B->rows; i++)
		{
			matx_complex_f64 val = B->data[i + j * B->stride];

			GxB_FC64_t v = { val.real,val.imag };

			GxB_Matrix_setElement_FC64(gB, v, i, j);
		}
	}

	/* build C */
	GrB_Matrix_new(&gC, GxB_FC64, C->rows, C->cols);

	for (size_t j = 0; j < C->cols; j++)
	{
		for (size_t i = 0; i < C->rows; i++)
		{
			matx_complex_f64  val = C->data[i + j * C->stride];
			GxB_FC64_t v = { val.real,val.imag };

			GxB_Matrix_setElement_FC64(gC, v, i, j);
		}
	}

	/* C = alpha*A*B + beta*C */

	GxB_FC64_t a = { alpha.real,alpha.imag };
	GxB_FC64_t b = { beta.real,beta.imag };

	// 1. temp = alpha * A * B
	GrB_Matrix temp;
	GrB_Matrix_new(&temp, GxB_FC64, C->rows, C->cols);
	GrB_mxm(temp, NULL, NULL, GxB_PLUS_TIMES_FC64, gA, gB, NULL);
	GrB_apply(temp, NULL, NULL, GxB_TIMES_FC64, temp, &a, NULL);
	//2. gC = beta * gC
	GrB_apply(gC, NULL, NULL, GxB_TIMES_FC64, gC, &b, NULL);
	//3. gC = temp + gC
	GrB_eWiseAdd(gC, NULL, NULL, GxB_PLUS_FC64, temp, gC, NULL);
	GrB_Matrix_free(&temp);

	/* copy back */
	for (size_t j = 0; j < C->cols; j++)
	{
		for (size_t i = 0; i < C->rows; i++)
		{
			GxB_FC64_t v;
			if (GxB_Matrix_extractElement_FC64(&v, gC, i, j)
				== GrB_SUCCESS)
			{
				C->data[i + j * C->stride].real = v._Val[0];
				C->data[i + j * C->stride].imag = v._Val[1];
			}
		}
	}

	GrB_free(&gA);
	GrB_free(&gB);
	GrB_free(&gC);

	return MATX_OK;
}