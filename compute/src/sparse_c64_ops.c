/* Sparse complex f64: spmv and spmm. Uses GraphBLAS complex when available. */
#include "matx/matx_compute.h"
#include "matx/matx_c64_utils.h"
#include "matx/matx.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include <suitesparse/GraphBLAS.h>

size_t coo_2_grb(matx_coo_c64_t* A)
{
	GrB_Matrix_free(A->handle_grb.impl);
	GrB_Matrix_new(&A->handle_grb.impl, GxB_FC64, A->nrows, A->ncols);
	GrB_Info info = GxB_Matrix_import_FC64(A->handle_grb.impl, GxB_FC64, A->nrows, A->ncols, A->rows, A->columns, A->values,
	A->nnz, A->nnz, A->nnz, GrB_COO_FORMAT);
	A->handle_grb.type = MATX_HANDLE_TYPE_GRB_MATRIX;
	A->handle_grb.valid = 1;
	return 0;
}

size_t vec_2_grb(matx_vec_c64_t* v)
{
	GrB_Vector_free(v->handle_grb.impl);
	GrB_Info info = GrB_Vector_new(&v->handle_grb.impl, GxB_FC64, v->n);
	info = GxB_Vector_import_Full(v->handle_grb.impl, GxB_FC64, v->n, &v->data, v->n, false, NULL);
	v->handle_grb.type = MATX_HANDLE_TYPE_GRB_VECTOR;
	v->handle_grb.valid = 1;
	return 0;
}

size_t grb_2_vec(matx_vec_c64_t* v)
{
	GrB_Type t = GxB_FC64;
	bool iso = false;
	GrB_Info info = GxB_Vector_export_Full(v->handle_grb.impl, &t, &v->n, &v->data, &v->n, false, NULL);
	return 0;
}

matx_status_t matx_spmv_csc_c64(
	matx_complex_f64 alpha,
	matx_coo_c64_t* A,
	matx_vec_c64_t* x,
	matx_complex_f64 beta,
	matx_vec_c64_t* y)
{
	GrB_Info info = GrB_init(GrB_NONBLOCKING);
	if (!A || !x || !y)
		return MATX_ERR_INVALID_ARG;

	if (A->ncols != x->n || A->nrows != y->n)
		return MATX_ERR_INVALID_ARG;

	/* ---------------- build GraphBLAS matrix ---------------- */

	if (A->handle_grb.valid <= 0)
	{
		coo_2_grb(A);
	}
	/* ---------------- build vectors ---------------- */
	if (x->handle_grb.valid <= 0)
	{
		vec_2_grb(x);
	}

	if (y->handle_grb.valid <= 0)
	{
		vec_2_grb(y);
	}
	/* ---------------- gy = alpha*A*x + beta*y ---------------- */
	GxB_FC64_t a = { alpha.real, alpha.imag };
	GxB_FC64_t b = { beta.real, beta.imag };

	GrB_Vector temp;
	GrB_Vector_new(&temp, GxB_FC64, y->n);
	//// temp = A*x
	info = GrB_mxv(temp, NULL, NULL, GxB_PLUS_TIMES_FC64, *(GrB_Matrix*)A->handle_grb.impl, *(GrB_Vector*)x->handle_grb.impl, NULL);
	// temp = alpha*temp
	info = GrB_apply(temp, NULL, NULL, GxB_TIMES_FC64, temp, &a, NULL);
	// gy = beta*gy
	info = GrB_apply(*(GrB_Vector*)y->handle_grb.impl, NULL, NULL, GxB_TIMES_FC64, *(GrB_Vector*)y->handle_grb.impl, &b, NULL);
	// gy = temp + gy
	info = GrB_eWiseAdd(*(GrB_Vector*)y->handle_grb.impl, NULL, NULL, GxB_PLUS_FC64, temp, *(GrB_Vector*)y->handle_grb.impl, NULL);
	GrB_Vector_free(&temp);
	grb_2_vec(y);

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

	GrB_Matrix gA = GrB_NULL;
	GrB_Matrix gB, gC;

	/* build A */
	GrB_Index Ap_size = (A->ncols + 1) * sizeof(GrB_Index);
	GrB_Index Ai_size = A->nnz * sizeof(GrB_Index);
	GrB_Index Ax_size = A->nnz * sizeof(GxB_FC64_t);
	info = GxB_Matrix_import_CSC(
		&gA,
		GxB_FC64,
		A->nrows, 
		A->ncols,
		&A->col_ptr,
		&A->row_ind,
		&A->values,
		Ap_size,
		Ai_size,
		Ax_size,
		false,
		false, 
		GrB_NULL
	);

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

	GrB_free(&gB);
	GrB_free(&gC);

	return MATX_OK;
}