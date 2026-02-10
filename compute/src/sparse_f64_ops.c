/* Sparse real f64: spmv and spmm. Uses CXSparse when available. */
#include "matx/matx_compute.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include <suitesparse/GraphBLAS.h>

size_t coo_2_grb_f64(matx_coo_f64_t* A)
{
	GrB_Matrix_free(A->handle_grb.impl);
	GrB_Matrix_new(&A->handle_grb.impl, GrB_FP64, A->nrows, A->ncols);
	GrB_Info info = GrB_Matrix_import_FP64(A->handle_grb.impl, GrB_FP64, A->nrows, A->ncols, A->rows, A->columns, A->values,
		A->nnz, A->nnz, A->nnz, GrB_COO_FORMAT);
	A->handle_grb.type = MATX_HANDLE_TYPE_GRB_MATRIX;
	A->handle_grb.valid = 1;
	A->handle_grb.custom_free_func = &free_grb_matrix;
	return 0;
}

size_t dense_2_grb_f64(matx_dense_f64_t* A)
{
	GrB_Matrix_free(A->handle_grb.impl);
	GrB_Matrix_new(&A->handle_grb.impl, GrB_FP64, A->rows, A->cols);
	GrB_Info info = GxB_Matrix_import_FullC(A->handle_grb.impl, GrB_FP64, A->rows, A->cols, &A->data, A->rows * A->cols, false, NULL);
	A->handle_grb.type = MATX_HANDLE_TYPE_GRB_MATRIX;
	A->handle_grb.valid = 1;
	A->handle_grb.custom_free_func = &free_grb_matrix;
	return 0;
}

size_t grb_2_dense_f64(matx_dense_f64_t* A)
{
	GrB_Type t;
	matx_uint64_t s = 0;
	bool iso = false;
	GrB_Info info = GxB_Matrix_export_FullC(A->handle_grb.impl, &t, &A->rows, &A->cols, &A->data, &s, &iso, NULL);
	A->handle_grb.custom_free_func = &free_grb_matrix;
	return 0;
}

size_t grb_2_coo_f64(matx_coo_f64_t* A)
{
	GrB_Info info = GxB_Matrix_export_FC64(A->rows, A->columns, A->values, &A->nrows, &A->ncols, &A->nnz, GrB_COO_FORMAT, *(GrB_Matrix*)A->handle_grb.impl);
	A->handle_grb.custom_free_func = &free_grb_matrix;
	return 0;
}

size_t vec_2_grb_f64(matx_vec_f64_t* v)
{
	GrB_Vector_free(v->handle_grb.impl);
	GrB_Info info = GrB_Vector_new(&v->handle_grb.impl, GrB_FP64, v->n);
	info = GxB_Vector_import_Full(v->handle_grb.impl, GrB_FP64, v->n, &v->data, v->n, false, NULL);
	v->handle_grb.type = MATX_HANDLE_TYPE_GRB_VECTOR;
	v->handle_grb.valid = 1;
	v->handle_grb.custom_free_func = &free_grb_vector;
	return 0;
}

size_t grb_2_vec_f64(matx_vec_f64_t* v)
{
	GrB_Type t = GrB_FP64;
	bool iso = false;
	GrB_Info info = GxB_Vector_export_Full(v->handle_grb.impl, &t, &v->n, &v->data, &v->n, false, NULL);
	v->handle_grb.custom_free_func = &free_grb_vector;
	return 0;
}

matx_status_t matx_spmv_coo_f64(
	matx_double alpha,
	matx_coo_f64_t* A,
	matx_vec_f64_t* x,
	matx_double beta,
	matx_vec_f64_t* y)
{
	GrB_Info info = GrB_init(GrB_NONBLOCKING);
	if (!A || !x || !y)
		return MATX_ERR_INVALID_ARG;

	if (A->ncols != x->n || A->nrows != y->n)
		return MATX_ERR_INVALID_ARG;

	/* ---------------- build GraphBLAS matrix ---------------- */

	if (A->handle_grb.valid <= 0)
	{
		coo_2_grb_f64(A);
	}
	/* ---------------- build vectors ---------------- */
	if (x->handle_grb.valid <= 0)
	{
		vec_2_grb_f64(x);
	}

	if (y->handle_grb.valid <= 0)
	{
		vec_2_grb_f64(y);
	}

	/* ---------------- gy = alpha*A*x + beta*y ---------------- */

	GrB_Vector temp;
	info = GrB_Vector_new(&temp, GrB_FP64, y->n);
	// temp = A*x
	info = GrB_mxv(temp, NULL, NULL, GxB_PLUS_TIMES_FP64, *(GrB_Matrix*)A->handle_grb.impl, *(GrB_Vector*)x->handle_grb.impl, NULL);
	// temp = alpha*temp
	info = GrB_apply(temp, NULL, NULL, GrB_TIMES_FP64, temp, &alpha, NULL);
	// gy = beta*gy
	info = GrB_apply(*(GrB_Vector*)y->handle_grb.impl, NULL, NULL, GrB_TIMES_FP64, *(GrB_Vector*)y->handle_grb.impl, &beta, NULL);
	// gy = temp + gy
	info = GrB_eWiseAdd(*(GrB_Vector*)y->handle_grb.impl, NULL, NULL, GrB_PLUS_FP64, temp, *(GrB_Vector*)y->handle_grb.impl, NULL);

	GrB_Vector_free(&temp);
	grb_2_vec_f64(y);
	return MATX_OK;
}

matx_status_t matx_spmm_coo_f64(
	matx_double alpha,
	matx_coo_f64_t* A,
	matx_dense_f64_t* B,
	matx_double beta,
	matx_dense_f64_t* C)
{
	GrB_Info info = GrB_init(GrB_NONBLOCKING);
	if (!A || !B || !C)
		return MATX_ERR_INVALID_ARG;

	/* build A */
	if (A->handle_grb.valid <= 0)
	{
		coo_2_grb_f64(A);
	}
	if (B->handle_grb.valid <= 0)
	{
		dense_2_grb_f64(B);
	}
	if (C->handle_grb.valid <= 0)
	{
		dense_2_grb_f64(C);
	}

	/* C = alpha*A*B + beta*C */

	// 1. temp = alpha * A * B
	GrB_Matrix temp;
	info = GrB_Matrix_new(&temp, GrB_FP64, C->rows, C->cols);
	info = GrB_mxm(temp, NULL, NULL, GxB_PLUS_TIMES_FP64, *(GrB_Matrix*)A->handle_grb.impl, *(GrB_Matrix*)B->handle_grb.impl, NULL);
	info = GrB_apply(temp, NULL, NULL, GrB_TIMES_FP64, temp, &alpha, NULL);
	//2. gC = beta * gC
	info = GrB_apply(*(GrB_Matrix*)C->handle_grb.impl, NULL, NULL, GrB_TIMES_FP64, *(GrB_Matrix*)C->handle_grb.impl, &beta, NULL);
	//3. gC = temp + gC
	info = GrB_eWiseAdd(*(GrB_Matrix*)C->handle_grb.impl, NULL, NULL, GrB_PLUS_FP64, temp, *(GrB_Matrix*)C->handle_grb.impl, NULL);
	GrB_Matrix_free(&temp);
	grb_2_dense_f64(C);
	return MATX_OK;
}