#include "matx/matx_sparse_compute.h"
#include "matx/matx_types.h"
#include "matx/matx_func.h"

#include <GraphBLAS.h>

#if MATX_ENABLE_MKL
#include "mkl.h"
#endif
#if MATX_HAVE_AOCL_SPARSE
#include <aoclsparse.h>
#endif

void free_grb_matrix(void* impl)
{
        GrB_Matrix_free((void*)&impl);
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
        GrB_Vector_free((void*)&impl);
}

void free_aocl_matrix(void* impl)
{
#if MATX_HAVE_AOCL_SPARSE
        aoclsparse_destroy((void*)&impl);
#endif
}

size_t coo_2_grb_f64(matx_coo_f64_t* A)
{
	GrB_Matrix_free(A->handle_grb.impl);
    GrB_Matrix_new((void*)&A->handle_grb.impl, GrB_FP64, A->nrows, A->ncols);
	GrB_Info info = GrB_Matrix_import_FP64(A->handle_grb.impl, GrB_FP64, A->nrows, A->ncols, A->rows, A->columns, A->values,
		A->nnz, A->nnz, A->nnz, GrB_COO_FORMAT);
	A->handle_grb.type = MATX_HANDLE_TYPE_GRB_MATRIX;
	A->handle_grb.valid = 1;
	A->handle_grb.custom_free_func = &free_grb_matrix;
	return 0;
}

size_t create_empty_grb_f64(matx_coo_f64_t* A)
{
	GrB_Matrix_free(A->handle_grb.impl);
	GrB_Info info = GrB_Matrix_new(&A->handle_grb.impl, GrB_FP64, A->nrows, A->ncols);
	A->handle_grb.type = MATX_HANDLE_TYPE_GRB_MATRIX;
	A->handle_grb.valid = 1;
	A->handle_grb.custom_free_func = &free_grb_matrix;
	return 0;
}

size_t create_empty_grb_c64(matx_coo_c64_t* A)
{
	GrB_Matrix_free(A->handle_grb.impl);
	GrB_Info info = GrB_Matrix_new(&A->handle_grb.impl, GxB_FC64, A->nrows, A->ncols);
	A->handle_grb.type = MATX_HANDLE_TYPE_GRB_MATRIX;
	A->handle_grb.valid = 1;
	A->handle_grb.custom_free_func = &free_grb_matrix;
	return 0;
}

typedef struct {
	matx_int64_t nrows;
	matx_int64_t ncols;
	matx_int64_t nnz;

	matx_int64_t* row_ptr;  // size nrows+1
	matx_int64_t* col_ind;  // size nnz
	matx_double* val;      // size nnz
} csr_matrix;

static inline void swap_int(matx_int64_t* a, matx_int64_t* b) {
	matx_int64_t t = *a; *a = *b; *b = t;
}

static inline void swap_double(matx_double* a, matx_double* b) {
	matx_double t = *a; *a = *b; *b = t;
}

static void insertion_sort(matx_int64_t* col, matx_double* val, matx_int64_t len) {
	for (matx_int64_t i = 1; i < len; ++i)
	{
		matx_int64_t    c = col[i];
		matx_double v = val[i];
		matx_int64_t j = i - 1;
		while (j >= 0 && col[j] > c) {
			col[j + 1] = col[j];
			val[j + 1] = val[j];
			j--;
		}
		col[j + 1] = c;
		val[j + 1] = v;
	}
}

static void quick_sort(matx_int64_t* col, matx_double* val, matx_int64_t left, matx_int64_t right) {
	if (left >= right) return;

	matx_int64_t i = left, j = right;
	matx_int64_t pivot = col[(left + right) >> 1];

	while (i <= j) {
		while (col[i] < pivot) i++;
		while (col[j] > pivot) j--;
		if (i <= j) {
			swap_int(&col[i], &col[j]);
			swap_double(&val[i], &val[j]);
			i++; j--;
		}
	}

	if (left < j)
		quick_sort(col, val, left, j);
	if (i < right)
		quick_sort(col, val, i, right);
}

static void sort_row(matx_int64_t* col, matx_double* val, matx_int64_t len) {
	if (len < 32)
		insertion_sort(col, val, len);
	else
		quick_sort(col, val, 0, len - 1);
}

int coo_to_csr_optimized(
	matx_int64_t nrows,
	matx_int64_t ncols,
	matx_int64_t nnz,
	const matx_int64_t* coo_row,
	const matx_int64_t* coo_col,
	const matx_double* coo_val,
	csr_matrix* csr)
{
	csr->nrows = nrows;
	csr->ncols = ncols;

	csr->row_ptr = (matx_int64_t*)calloc(nrows + 1, sizeof(matx_int64_t));
	csr->col_ind = (matx_int64_t*)malloc(nnz * sizeof(matx_int64_t));
	csr->val = (matx_double*)malloc(nnz * sizeof(matx_double));

	if (!csr->row_ptr || !csr->col_ind || !csr->val)
		return -1;
	for (matx_int64_t i = 0; i < nnz; ++i) {
		csr->row_ptr[coo_row[i] + 1]++;
	}

	for (matx_int64_t i = 0; i < nrows; ++i) {
		csr->row_ptr[i + 1] += csr->row_ptr[i];
	}

	matx_int64_t* offset = (matx_int64_t*)malloc(nrows * sizeof(matx_int64_t));
	if (offset == NULL || csr->row_ptr == NULL || nrows <= 0)
		return -1;
	memcpy(offset, csr->row_ptr, nrows * sizeof(matx_int64_t));

	for (int i = 0; i < nnz; ++i) {
		int r = coo_row[i];
		int dst = offset[r]++;

		csr->col_ind[dst] = coo_col[i];
		csr->val[dst] = coo_val[i];
	}

	free(offset);

	matx_int64_t new_nnz = 0;

	for (matx_int64_t i = 0; i < nrows; ++i) {
		matx_int64_t start = csr->row_ptr[i];
		matx_int64_t end = csr->row_ptr[i + 1];
		matx_int64_t len = end - start;

		if (len == 0) {
			csr->row_ptr[i] = new_nnz;
			continue;
		}

		matx_int64_t* col = csr->col_ind + start;
		matx_double* val = csr->val + start;

		sort_row(col, val, len);

		matx_int64_t write = 0;
		for (matx_int64_t j = 0; j < len; ++j) {
			if (j > 0 && col[j] == col[j - 1]) {
				val[write - 1] += val[j];
			}
			else {
				col[write] = col[j];
				val[write] = val[j];
				write++;
			}
		}

		for (matx_int64_t j = 0; j < write; ++j) {
			csr->col_ind[new_nnz + j] = col[j];
			csr->val[new_nnz + j] = val[j];
		}

		csr->row_ptr[i] = new_nnz;
		new_nnz += write;
	}

	csr->row_ptr[nrows] = new_nnz;
	csr->nnz = new_nnz;

	csr->col_ind = (matx_int64_t*)realloc(csr->col_ind, new_nnz * sizeof(matx_int64_t));
	csr->val = (matx_double*)realloc(csr->val, new_nnz * sizeof(matx_double));

	return 0;
}


size_t coo_2_aocl_f64(matx_coo_f64_t* A)
{
#if MATX_HAVE_AOCL_SPARSE

	//aoclsparse_matrix coo;
	aoclsparse_matrix csr;
	//aoclsparse_status st;

	//st = aoclsparse_create_dcoo(
	//	&coo,
	//	aoclsparse_index_base_zero,
	//	(aoclsparse_int)A->nrows,
	//	(aoclsparse_int)A->ncols,
	//	(aoclsparse_int)A->nnz,
	//	(aoclsparse_int*)A->rows,
	//	(aoclsparse_int*)A->columns,
	//	A->values
	//);

	//if (st != aoclsparse_status_success)
	//	return (size_t)-1;

	//st = aoclsparse_convert_csr(
	//	coo,
	//	aoclsparse_operation_none,
	//	&csr
	//);
	//aoclsparse_destroy(&coo);

	//if (st != aoclsparse_status_success)
	//	return (size_t)-1;
	csr_matrix csr_m;
	matx_int64_t ret = coo_to_csr_optimized(A->nrows, A->ncols, A->nnz, A->rows, A->columns, A->values, &csr_m);
	if (ret != 0)
		return -1;
	aoclsparse_index_base base = aoclsparse_index_base_zero;
	aoclsparse_status st = aoclsparse_create_dcsr(&csr, base, csr_m.nrows, csr_m.ncols, csr_m.nnz, csr_m.row_ptr, csr_m.col_ind, csr_m.val);
	if (st != aoclsparse_status_success)
		return -1;

	st = aoclsparse_optimize(csr);

	A->handle_aocl.impl = csr;
	A->handle_aocl.type = MATX_HANDLE_TYPE_AOCL_MATRIX;
	A->handle_aocl.valid = 1;
	A->handle_aocl.custom_free_func = &free_aocl_matrix;
#endif

	return 0;
}

size_t coo_2_aocl_c64(matx_coo_c64_t* A)
{
#if MATX_HAVE_AOCL_SPARSE

	aoclsparse_matrix coo;
	aoclsparse_matrix csr;
	aoclsparse_status st;

	st = aoclsparse_create_zcoo(
		&coo,
		aoclsparse_index_base_zero,
		(aoclsparse_int)A->nrows,
		(aoclsparse_int)A->ncols,
		(aoclsparse_int)A->nnz,
		(aoclsparse_int*)A->rows,
		(aoclsparse_int*)A->columns,
                (void*)A->values
	);

	if (st != aoclsparse_status_success)
		return (size_t)-1;

	st = aoclsparse_convert_csr(
		coo,
		aoclsparse_operation_none,
		&csr
	);

	aoclsparse_destroy(&coo);
	aoclsparse_optimize(csr);

	A->handle_aocl.impl = csr;
	A->handle_aocl.type = MATX_HANDLE_TYPE_AOCL_MATRIX;
	A->handle_aocl.valid = 1;
	A->handle_aocl.custom_free_func = &free_aocl_matrix;

#endif

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
        GrB_Matrix_new((void*)&A->handle_grb.impl, GrB_FP64, A->rows, A->cols);
        GrB_Info info = GxB_Matrix_import_FullC((void*)A->handle_grb.impl, GrB_FP64, A->rows, A->cols, (void*)&A->data, A->rows * A->cols, false, NULL);
	A->handle_grb.type = MATX_HANDLE_TYPE_GRB_MATRIX;
	A->handle_grb.valid = 1;
	A->handle_grb.custom_free_func = &free_grb_matrix;
	return 0;
}

size_t grb_2_dense_f64(matx_dense_f64_t* A)
{
	GrB_Type t;
	matx_int64_t s = 0;
	bool iso = false;
    GrB_Info info = GxB_Matrix_export_FullC((void*)A->handle_grb.impl, &t, &A->rows, &A->cols, (void*)&A->data, &s, &iso, NULL);
	A->handle_grb.custom_free_func = &free_grb_matrix;
	return 0;
}

size_t grb_2_coo_f64(matx_coo_f64_t* A)
{
	A->nrows = -1;
	A->ncols = -1;
	GrB_Info info = GrB_Matrix_export(A->rows, A->columns, A->values, &A->nrows, &A->ncols, &A->nnz, GrB_COO_FORMAT, A->handle_grb.impl);
	A->handle_grb.custom_free_func = &free_grb_matrix;
	A->handle_grb.type = MATX_HANDLE_TYPE_GRB_MATRIX;
	A->handle_grb.valid = 1;
	return 0;
}

size_t vec_2_grb_f64(matx_vec_f64_t* v)
{
	GrB_Vector_free(v->handle_grb.impl);
    GrB_Info info = GrB_Vector_new((void*)&v->handle_grb.impl, GrB_FP64, v->n);
    info = GxB_Vector_import_Full((void*)v->handle_grb.impl, GrB_FP64, v->n, (void*)&v->data, v->n, false, NULL);
	v->handle_grb.type = MATX_HANDLE_TYPE_GRB_VECTOR;
	v->handle_grb.valid = 1;
	v->handle_grb.custom_free_func = &free_grb_vector;
	return 0;
}

size_t grb_2_vec_f64(matx_vec_f64_t* v)
{
	GrB_Type t = GrB_FP64;
    GrB_Info info = GxB_Vector_export_Full((void*)v->handle_grb.impl, &t, &v->n, (void*)&v->data, &v->n, false, NULL);
	v->handle_grb.custom_free_func = &free_grb_vector;
	return 0;
}

size_t coo_2_grb_c64(matx_coo_c64_t* A)
{
	GrB_Matrix_free(A->handle_grb.impl);
        GrB_Matrix_new((void*)&A->handle_grb.impl, GxB_FC64, A->nrows, A->ncols);
        GrB_Info info = GxB_Matrix_import_FC64((void*)A->handle_grb.impl, GxB_FC64, A->nrows, A->ncols, A->rows, A->columns, (void*)A->values,
		A->nnz, A->nnz, A->nnz, GrB_COO_FORMAT);
	A->handle_grb.type = MATX_HANDLE_TYPE_GRB_MATRIX;
	A->handle_grb.valid = 1;
	A->handle_grb.custom_free_func = &free_grb_matrix;
	return 0;
}

size_t dense_2_grb_c64(matx_dense_c64_t* A)
{
	GrB_Matrix_free(A->handle_grb.impl);
        GrB_Matrix_new((void*)&A->handle_grb.impl, GxB_FC64, A->rows, A->cols);
        GrB_Info info = GxB_Matrix_import_FullC((void*)A->handle_grb.impl, GxB_FC64, A->rows, A->cols, (void*)&A->data, A->rows * A->cols, false, NULL);
	A->handle_grb.type = MATX_HANDLE_TYPE_GRB_MATRIX;
	A->handle_grb.valid = 1;
	A->handle_grb.custom_free_func = &free_grb_matrix;
	return 0;
}

size_t grb_2_dense_c64(matx_dense_c64_t* A)
{
	GrB_Type t;
	matx_int64_t s = 0;
	bool iso = false;
        GrB_Info info = GxB_Matrix_export_FullC((void*)A->handle_grb.impl, &t, &A->rows, &A->cols, (void*)&A->data, &s, &iso, NULL);
	A->handle_grb.custom_free_func = &free_grb_matrix;
	return 0;
}

size_t grb_2_coo_c64(matx_coo_c64_t* A)
{
	A->ncols = -1;
	A->nrows = -1;
	A->nnz = -1;
    GrB_Info info = GxB_Matrix_export_FC64(A->rows, A->columns, (void*)A->values, &A->nrows, &A->ncols, &A->nnz, GrB_COO_FORMAT, A->handle_grb.impl);
	A->handle_grb.custom_free_func = &free_grb_matrix;
	return 0;
}

size_t vec_2_grb_c64(matx_vec_c64_t* v)
{
	GrB_Vector_free(v->handle_grb.impl);
        GrB_Info info = GrB_Vector_new((void*)&v->handle_grb.impl, GxB_FC64, v->n);
        info = GxB_Vector_import_Full((void*)v->handle_grb.impl, GxB_FC64, v->n, (void*)&v->data, v->n, false, NULL);
	v->handle_grb.type = MATX_HANDLE_TYPE_GRB_VECTOR;
	v->handle_grb.valid = 1;
	v->handle_grb.custom_free_func = &free_grb_vector;
	return 0;
}

size_t grb_2_vec_c64(matx_vec_c64_t* v)
{
	GrB_Type t = GxB_FC64;
	bool iso = false;
        GrB_Info info = GxB_Vector_export_Full((void*)v->handle_grb.impl, &t, &v->n, (void*)&v->data, &v->n, false, NULL);
	v->handle_grb.custom_free_func = &free_grb_vector;
	return 0;
}