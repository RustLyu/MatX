#include "matx/matx_sparse_compute.h"

#include <limits.h>
#if MATX_ENABLE_MKL
	#include "mkl.h"
#endif

matx_status_t ref_spmv_c64_mkl(
	matx_complex_f64_t alpha,
	matx_coo_c64_t* A,
	matx_vec_c64_t* x,
	matx_complex_f64_t beta,
	matx_vec_c64_t* y)
{
#if MATX_ENABLE_MKL
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
#endif
	return MATX_OK;
}

matx_status_t ref_spmm_c64_mkl(
	matx_complex_f64_t alpha,
	const matx_coo_c64_t* A,
	const matx_dense_c64_t* B,
	matx_complex_f64_t beta,
	matx_dense_c64_t* C)
{
#if MATX_ENABLE_MKL
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
#endif
	return MATX_OK;
}

matx_status_t ref_spmv_f64_mkl(
	matx_double alpha,
	matx_coo_f64_t* A,
	matx_vec_f64_t* x,
	matx_double beta,
	matx_vec_f64_t* y)
{
#if MATX_ENABLE_MKL
	if (!A || !x || !y)
		return MATX_ERR_INVALID_ARG;
	if (A->handle_mkl.valid <= 0)
	{
		coo_2_mkl_f64(A);
	}
	int64_t t0 = get_time_us();
	struct matrix_descr descr;
	descr.type = SPARSE_MATRIX_TYPE_GENERAL;
	descr.diag = SPARSE_DIAG_NON_UNIT;

	sparse_status_t st = mkl_sparse_d_mv(
		SPARSE_OPERATION_NON_TRANSPOSE,
		alpha,
		(sparse_matrix_t)A->handle_mkl.impl,
		descr,
		x->data,
		beta,
		y->data
	);
	int64_t t1 = get_time_us();
	printf("mkl time: %ld us\n", t1 - t0);
#endif
	return MATX_OK;
}

matx_status_t ref_spmm_f64_mkl(
	matx_double alpha,
	matx_coo_f64_t* A,
	matx_dense_f64_t* B,
	matx_double beta,
	matx_dense_f64_t* C)
{
#if MATX_ENABLE_MKL
	if (!A || !B || !C)
		return MATX_ERR_INVALID_ARG;

	if (A->handle_mkl.valid <= 0)
	{
		coo_2_mkl_c64(A);
	}

	struct matrix_descr descr;
	descr.type = SPARSE_MATRIX_TYPE_GENERAL;
	descr.diag = SPARSE_DIAG_NON_UNIT;



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
#endif
	return MATX_OK;
}


matx_sparse_backend_t matx_sparse_make_reference_mkl(void) {
	matx_sparse_backend_t b =
	{
		.kind = MATX_SPARSE_BACKEND_MKL,
		.vt = {
			.spmm_c64 = ref_spmm_c64_mkl,
			.spmv_c64 = ref_spmv_c64_mkl,
			.spmm_f64 = ref_spmm_f64_mkl,
			.spmv_f64 = ref_spmv_f64_mkl
		}
	};
	return b;
}

