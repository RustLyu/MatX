#include "matx/matx_dense_compute.h"

#include <suitesparse/GraphBLAS.h>

#if MATX_ENABLE_MKL
#include "mkl.h"
#endif

void free_grb_matrix(void* impl)
{
	GrB_Matrix_free(&impl);
}

void free_mkl_matrix(void* impl)
{
#if MATX_ENABLE_MKL
	if (!impl)
		return;

	sparse_matrix_t A = (sparse_matrix_t)impl;
	mkl_sparse_destroy(A);
#endif
}

void free_grb_vector(void* impl)
{
	GrB_Vector_free(&impl);
}

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

size_t coo_2_mkl_f64(matx_coo_f64_t* A)
{
#if MATX_ENABLE_MKL
	sparse_matrix_t coo;
	sparse_matrix_t csr;
	sparse_status_t st;
	int rows[100];
	int column[100];
	for (int i = 0; i < (int)A->nnz; ++i)
	{
		rows[i] = A->rows[i];
		column[i] = A->columns[i];
	}

	//st = mkl_sparse_d_create_coo(
	//	&coo,
	//	SPARSE_INDEX_BASE_ZERO,
	//	(MKL_INT)A->nrows,
	//	(MKL_INT)A->ncols,
	//	(MKL_INT)A->nnz,
	//	rows,
	//	column,
	//	A->values            
	//);
	st = mkl_sparse_d_create_coo(
		&coo,
		SPARSE_INDEX_BASE_ZERO,
		(MKL_INT)A->nrows,
		(MKL_INT)A->ncols,
		(MKL_INT)A->nnz,
		A->rows,
		A->columns,
		A->values
	);

	if (st != SPARSE_STATUS_SUCCESS) {
		return (size_t)-1;
	}

	st = mkl_sparse_convert_csr(coo, SPARSE_OPERATION_NON_TRANSPOSE, &csr);
	mkl_sparse_destroy(coo);

	if (st != SPARSE_STATUS_SUCCESS) {
		return (size_t)-1;
	}

	mkl_sparse_optimize(csr);

	A->handle_mkl.impl = csr;
	A->handle_mkl.type = MATX_HANDLE_TYPE_MKL_MATRIX;
	A->handle_mkl.valid = 1;
	A->handle_mkl.custom_free_func = &free_mkl_matrix;
#endif
	return 0;
}

size_t coo_2_mkl_c64(matx_coo_c64_t* A)
{
#if MATX_ENABLE_MKL
	sparse_matrix_t coo;
	sparse_matrix_t csr;

	sparse_status_t st;
	int rows[100];
	int column[100];
	for (int i = 0; i < (int)A->nnz; ++i)
	{
		rows[i] = A->rows[i];
		column[i] = A->columns[i];
	}
	st = mkl_sparse_z_create_coo(
		&coo,
		SPARSE_INDEX_BASE_ZERO,
		A->nrows,
		A->ncols,
		A->nnz,
		A->rows,
		A->columns,
		(MKL_Complex16*)A->values
	);
	//st = mkl_sparse_z_create_coo(
	//	&coo,
	//	SPARSE_INDEX_BASE_ZERO,
	//	A->nrows,
	//	A->ncols,
	//	A->nnz,
	//	rows,
	//	column,
	//	(MKL_Complex16*)A->values
	//);

	if (st != SPARSE_STATUS_SUCCESS)
		return -1;

	st = mkl_sparse_convert_csr(coo, SPARSE_OPERATION_NON_TRANSPOSE, &csr);
	mkl_sparse_destroy(coo);

	if (st != SPARSE_STATUS_SUCCESS)
		return -1;

	mkl_sparse_optimize(csr);

	A->handle_mkl.impl = csr;
	A->handle_mkl.type = MATX_HANDLE_TYPE_MKL_MATRIX;
	A->handle_mkl.valid = 1;
	A->handle_mkl.custom_free_func = &free_mkl_matrix;
#endif
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

size_t coo_2_grb_c64(matx_coo_c64_t* A)
{
	GrB_Matrix_free(A->handle_grb.impl);
	GrB_Matrix_new(&A->handle_grb.impl, GxB_FC64, A->nrows, A->ncols);
	GrB_Info info = GxB_Matrix_import_FC64(A->handle_grb.impl, GxB_FC64, A->nrows, A->ncols, A->rows, A->columns, A->values,
		A->nnz, A->nnz, A->nnz, GrB_COO_FORMAT);
	A->handle_grb.type = MATX_HANDLE_TYPE_GRB_MATRIX;
	A->handle_grb.valid = 1;
	A->handle_grb.custom_free_func = &free_grb_matrix;
	return 0;
}

size_t dense_2_grb_c64(matx_dense_c64_t* A)
{
	GrB_Matrix_free(A->handle_grb.impl);
	GrB_Matrix_new(&A->handle_grb.impl, GxB_FC64, A->rows, A->cols);
	GrB_Info info = GxB_Matrix_import_FullC(A->handle_grb.impl, GxB_FC64, A->rows, A->cols, &A->data, A->rows * A->cols, false, NULL);
	A->handle_grb.type = MATX_HANDLE_TYPE_GRB_MATRIX;
	A->handle_grb.valid = 1;
	A->handle_grb.custom_free_func = &free_grb_matrix;
	return 0;
}

size_t grb_2_dense_c64(matx_dense_c64_t* A)
{
	GrB_Type t;
	matx_uint64_t s = 0;
	bool iso = false;
	GrB_Info info = GxB_Matrix_export_FullC(A->handle_grb.impl, &t, &A->rows, &A->cols, &A->data, &s, &iso, NULL);
	A->handle_grb.custom_free_func = &free_grb_matrix;
	return 0;
}

size_t grb_2_coo_c64(matx_coo_c64_t* A)
{
	GrB_Info info = GxB_Matrix_export_FC64(A->rows, A->columns, A->values, &A->nrows, &A->ncols, &A->nnz, GrB_COO_FORMAT, *(GrB_Matrix*)A->handle_grb.impl);
	A->handle_grb.custom_free_func = &free_grb_matrix;
	return 0;
}

size_t vec_2_grb_c64(matx_vec_c64_t* v)
{
	GrB_Vector_free(v->handle_grb.impl);
	GrB_Info info = GrB_Vector_new(&v->handle_grb.impl, GxB_FC64, v->n);
	info = GxB_Vector_import_Full(v->handle_grb.impl, GxB_FC64, v->n, &v->data, v->n, false, NULL);
	v->handle_grb.type = MATX_HANDLE_TYPE_GRB_VECTOR;
	v->handle_grb.valid = 1;
	v->handle_grb.custom_free_func = &free_grb_vector;
	return 0;
}

size_t grb_2_vec_c64(matx_vec_c64_t* v)
{
	GrB_Type t = GxB_FC64;
	bool iso = false;
	GrB_Info info = GxB_Vector_export_Full(v->handle_grb.impl, &t, &v->n, &v->data, &v->n, false, NULL);
	v->handle_grb.custom_free_func = &free_grb_vector;
	return 0;
}