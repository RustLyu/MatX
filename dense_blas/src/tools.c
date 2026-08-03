#include "matx/matx_dense_compute.h"
#include "matx/matx_log.h"

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

size_t coo_2_grb_d_i8(matx_coo_d_i8_t* A)
{
	GrB_Matrix_free(MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl);
	GrB_Matrix_new(&MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl, GrB_FP64, A->nrows, A->ncols);
	GrB_Info info = GrB_Matrix_import_FP64(MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl, GrB_FP64, A->nrows, A->ncols, A->rows, A->columns, A->values,
		A->nnz, A->nnz, A->nnz, GrB_COO_FORMAT);
	MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->type = MATX_HANDLE_TYPE_GRB_MATRIX;
	MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->valid = 1;
	MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->custom_free_func = &free_grb_matrix;
	return 0;
}

size_t coo_2_mkl_d_i8(matx_coo_d_i8_t* A)
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
		MATX_ERROR("mkl_sparse_d_create_coo failed");
		return (size_t)-1;
	}

	st = mkl_sparse_convert_csr(coo, SPARSE_OPERATION_NON_TRANSPOSE, &csr);
	mkl_sparse_destroy(coo);

	if (st != SPARSE_STATUS_SUCCESS) {
		MATX_ERROR("mkl_sparse_convert_csr failed");
		return (size_t)-1;
	}

	mkl_sparse_optimize(csr);

	MATX_HANDLE(A, MATX_HANDLE_TYPE_MKL_MATRIX)->impl = csr;
	MATX_HANDLE(A, MATX_HANDLE_TYPE_MKL_MATRIX)->type = MATX_HANDLE_TYPE_MKL_MATRIX;
	MATX_HANDLE(A, MATX_HANDLE_TYPE_MKL_MATRIX)->valid = 1;
	MATX_HANDLE(A, MATX_HANDLE_TYPE_MKL_MATRIX)->custom_free_func = &free_mkl_matrix;
#endif
	return 0;
}

size_t coo_2_mkl_z_i8(matx_coo_z_i8_t* A)
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

	if (st != SPARSE_STATUS_SUCCESS) {
		MATX_ERROR("mkl_sparse_z_create_coo failed");
		return -1;
	}

	st = mkl_sparse_convert_csr(coo, SPARSE_OPERATION_NON_TRANSPOSE, &csr);
	mkl_sparse_destroy(coo);

	if (st != SPARSE_STATUS_SUCCESS) {
		MATX_ERROR("mkl_sparse_convert_csr (complex) failed");
		return -1;
	}

	mkl_sparse_optimize(csr);

	MATX_HANDLE(A, MATX_HANDLE_TYPE_MKL_MATRIX)->impl = csr;
	MATX_HANDLE(A, MATX_HANDLE_TYPE_MKL_MATRIX)->type = MATX_HANDLE_TYPE_MKL_MATRIX;
	MATX_HANDLE(A, MATX_HANDLE_TYPE_MKL_MATRIX)->valid = 1;
	MATX_HANDLE(A, MATX_HANDLE_TYPE_MKL_MATRIX)->custom_free_func = &free_mkl_matrix;
#endif
	return 0;
}

size_t dense_2_grb_d_i8(matx_dense_d_i8_t* A)
{
	GrB_Matrix_free(MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl);
	GrB_Matrix_new(&MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl, GrB_FP64, A->rows, A->cols);
	GrB_Info info = GxB_Matrix_import_FullC(MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl, GrB_FP64, A->rows, A->cols, &A->data, A->rows * A->cols, false, NULL);
	MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->type = MATX_HANDLE_TYPE_GRB_MATRIX;
	MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->valid = 1;
	MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->custom_free_func = &free_grb_matrix;
	return 0;
}

size_t grb_2_dense_d_i8(matx_dense_d_i8_t* A)
{
	GrB_Type t;
	matx_uint64_t s = 0;
	bool iso = false;
	GrB_Info info = GxB_Matrix_export_FullC(MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl, &t, &A->rows, &A->cols, &A->data, &s, &iso, NULL);
	MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->custom_free_func = &free_grb_matrix;
	return 0;
}

size_t grb_2_coo_d_i8(matx_coo_d_i8_t* A)
{
	GrB_Info info = GxB_Matrix_export_FC64(A->rows, A->columns, A->values, &A->nrows, &A->ncols, &A->nnz, GrB_COO_FORMAT, *(GrB_Matrix*)MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl);
	MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->custom_free_func = &free_grb_matrix;
	return 0;
}

size_t vec_2_grb_d_i8(matx_vec_d_i8_t* v)
{
	GrB_Vector_free(MATX_HANDLE(v, MATX_HANDLE_TYPE_GRB_VECTOR)->impl);
	GrB_Info info = GrB_Vector_new(&MATX_HANDLE(v, MATX_HANDLE_TYPE_GRB_VECTOR)->impl, GrB_FP64, v->n);
	info = GxB_Vector_import_Full(MATX_HANDLE(v, MATX_HANDLE_TYPE_GRB_VECTOR)->impl, GrB_FP64, v->n, &v->data, v->n, false, NULL);
	MATX_HANDLE(v, MATX_HANDLE_TYPE_GRB_VECTOR)->type = MATX_HANDLE_TYPE_GRB_VECTOR;
	MATX_HANDLE(v, MATX_HANDLE_TYPE_GRB_VECTOR)->valid = 1;
	MATX_HANDLE(v, MATX_HANDLE_TYPE_GRB_VECTOR)->custom_free_func = &free_grb_vector;
	return 0;
}

size_t grb_2_vec_d_i8(matx_vec_d_i8_t* v)
{
	GrB_Type t = GrB_FP64;
	bool iso = false;
	GrB_Info info = GxB_Vector_export_Full(MATX_HANDLE(v, MATX_HANDLE_TYPE_GRB_VECTOR)->impl, &t, &v->n, &v->data, &v->n, false, NULL);
	MATX_HANDLE(v, MATX_HANDLE_TYPE_GRB_VECTOR)->custom_free_func = &free_grb_vector;
	return 0;
}

size_t coo_2_grb_z_i8(matx_coo_z_i8_t* A)
{
	GrB_Matrix_free(MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl);
	GrB_Matrix_new(&MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl, GxB_FC64, A->nrows, A->ncols);
	GrB_Info info = GxB_Matrix_import_FC64(MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl, GxB_FC64, A->nrows, A->ncols, A->rows, A->columns, A->values,
		A->nnz, A->nnz, A->nnz, GrB_COO_FORMAT);
	MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->type = MATX_HANDLE_TYPE_GRB_MATRIX;
	MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->valid = 1;
	MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->custom_free_func = &free_grb_matrix;
	return 0;
}

size_t dense_2_grb_z_i8(matx_dense_z_i8_t* A)
{
	GrB_Matrix_free(MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl);
	GrB_Matrix_new(&MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl, GxB_FC64, A->rows, A->cols);
	GrB_Info info = GxB_Matrix_import_FullC(MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl, GxB_FC64, A->rows, A->cols, &A->data, A->rows * A->cols, false, NULL);
	MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->type = MATX_HANDLE_TYPE_GRB_MATRIX;
	MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->valid = 1;
	MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->custom_free_func = &free_grb_matrix;
	return 0;
}

size_t grb_2_dense_z_i8(matx_dense_z_i8_t* A)
{
	GrB_Type t;
	matx_uint64_t s = 0;
	bool iso = false;
	GrB_Info info = GxB_Matrix_export_FullC(MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl, &t, &A->rows, &A->cols, &A->data, &s, &iso, NULL);
	MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->custom_free_func = &free_grb_matrix;
	return 0;
}

size_t grb_2_coo_z_i8(matx_coo_z_i8_t* A)
{
	GrB_Info info = GxB_Matrix_export_FC64(A->rows, A->columns, A->values, &A->nrows, &A->ncols, &A->nnz, GrB_COO_FORMAT, *(GrB_Matrix*)MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl);
	MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->custom_free_func = &free_grb_matrix;
	return 0;
}

size_t vec_2_grb_z_i8(matx_vec_z_i8_t* v)
{
	GrB_Vector_free(MATX_HANDLE(v, MATX_HANDLE_TYPE_GRB_VECTOR)->impl);
	GrB_Info info = GrB_Vector_new(&MATX_HANDLE(v, MATX_HANDLE_TYPE_GRB_VECTOR)->impl, GxB_FC64, v->n);
	info = GxB_Vector_import_Full(MATX_HANDLE(v, MATX_HANDLE_TYPE_GRB_VECTOR)->impl, GxB_FC64, v->n, &v->data, v->n, false, NULL);
	MATX_HANDLE(v, MATX_HANDLE_TYPE_GRB_VECTOR)->type = MATX_HANDLE_TYPE_GRB_VECTOR;
	MATX_HANDLE(v, MATX_HANDLE_TYPE_GRB_VECTOR)->valid = 1;
	MATX_HANDLE(v, MATX_HANDLE_TYPE_GRB_VECTOR)->custom_free_func = &free_grb_vector;
	return 0;
}

size_t grb_2_vec_z_i8(matx_vec_z_i8_t* v)
{
	GrB_Type t = GxB_FC64;
	bool iso = false;
	GrB_Info info = GxB_Vector_export_Full(MATX_HANDLE(v, MATX_HANDLE_TYPE_GRB_VECTOR)->impl, &t, &v->n, &v->data, &v->n, false, NULL);
	MATX_HANDLE(v, MATX_HANDLE_TYPE_GRB_VECTOR)->custom_free_func = &free_grb_vector;
	return 0;
}