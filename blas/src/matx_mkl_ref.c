#include "matx/matx_compute.h"

#include <limits.h>

#include <mkl.h>

matx_status_t ref_spmv_c64_mkl(
	matx_complex_f64 alpha,
	matx_coo_c64_t* A,
	matx_vec_c64_t* x,
	matx_complex_f64 beta,
	matx_vec_c64_t* y)
{
	if (!A || !x || !y)
		return MATX_ERR_INVALID_ARG;

	if (A->ncols != x->n || A->nrows != y->n)
		return MATX_ERR_INVALID_ARG;

	if (A->handle_mkl.valid <= 0)
	{
		if (coo_2_mkl_c64(A) != 0)
			return MATX_ERR_INTERNAL;
	}

	struct matrix_descr descr;
	descr.type = SPARSE_MATRIX_TYPE_GENERAL;
	descr.mode = SPARSE_FILL_MODE_FULL;
	descr.diag = SPARSE_DIAG_NON_UNIT;

	MKL_Complex16 a = { alpha.real, alpha.imag };
	MKL_Complex16 b = { beta.real, beta.imag };

	mkl_sparse_z_mv(
		SPARSE_OPERATION_NON_TRANSPOSE,
		a,
		(sparse_matrix_t)A->handle_mkl.impl,
		descr,
		(MKL_Complex16*)x->data,
		b,
		(MKL_Complex16*)y->data
	);

	return MATX_OK;
}

matx_status_t ref_spmm_c64_mkl(
	matx_complex_f64 alpha,
	const matx_coo_c64_t* A,
	const matx_dense_c64_t* B,
	matx_complex_f64 beta,
	matx_dense_c64_t* C)
{
	if (!A || !B || !C)
		return MATX_ERR_INVALID_ARG;

	if (A->handle_mkl.valid <= 0)
	{
		coo_2_mkl_c64(A);
	}

	struct matrix_descr descr;
	descr.type = SPARSE_MATRIX_TYPE_GENERAL;

	MKL_Complex16 a = { alpha.real, alpha.imag };
	MKL_Complex16 b = { beta.real, beta.imag };

	mkl_sparse_z_mm(
		SPARSE_OPERATION_NON_TRANSPOSE,
		a,
		(sparse_matrix_t)A->handle_mkl.impl,
		descr,
		SPARSE_LAYOUT_ROW_MAJOR,
		(MKL_Complex16*)B->data,
		B->cols,
		B->cols,
		b,
		(MKL_Complex16*)C->data,
		C->cols
	);

	return MATX_OK;
}

matx_status_t ref_spmv_f64_mkl(
	matx_double alpha,
	matx_coo_f64_t* A,
	matx_vec_f64_t* x,
	matx_double beta,
	matx_vec_f64_t* y)
{
	if (!A || !x || !y)
		return MATX_ERR_INVALID_ARG;
	if (A->handle_mkl.valid <= 0)
	{
		coo_2_mkl_f64(A);
	}

	struct matrix_descr descr;
	descr.type = SPARSE_MATRIX_TYPE_GENERAL;

	mkl_sparse_d_mv(
		SPARSE_OPERATION_NON_TRANSPOSE,
		alpha,
		(sparse_matrix_t)A->handle_mkl.impl,
		descr,
		x->data,
		beta,
		y->data
	);

	return MATX_OK;
}

matx_status_t ref_spmm_f64_mkl(
	matx_double alpha,
	matx_coo_f64_t* A,
	matx_dense_f64_t* B,
	matx_double beta,
	matx_dense_f64_t* C)
{
	if (A->handle_mkl.valid <= 0)
	{
		coo_2_mkl_f64(A);
	}

	struct matrix_descr descr;
	descr.type = SPARSE_MATRIX_TYPE_GENERAL;

	mkl_sparse_d_mm(
		SPARSE_OPERATION_NON_TRANSPOSE,
		alpha,
		(sparse_matrix_t)A->handle_mkl.impl,
		descr,
		SPARSE_LAYOUT_ROW_MAJOR,
		B->data,
		B->cols,
		B->cols,
		beta,
		C->data,
		C->cols
	);

	return MATX_OK;
}


matx_sparse_backend_t matx_sparse_make_reference_mkl(void) {
	matx_sparse_backend_t b;
	b.kind = MATX_SPARSE_BACKEND_GRAPHBLAS;
	b.vt.spmm_c64 = ref_spmm_c64_mkl;
	b.vt.spmv_c64 = ref_spmv_c64_mkl;
	b.vt.spmm_f64 = ref_spmm_f64_mkl;
	b.vt.spmv_f64 = ref_spmv_f64_mkl;
	return b;
}

