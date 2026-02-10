#include "matx/matx_compute.h"

#include <suitesparse/GraphBLAS.h>

void free_grb_matrix(void* impl)
{
	GrB_Matrix_free(&impl);
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