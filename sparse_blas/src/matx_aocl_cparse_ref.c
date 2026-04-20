#include "matx/matx_sparse_compute.h"
#include "matx/matx_log.h"
#include "matx/matx_types_internal.h"

#include <memory.h>
#include <limits.h>

#if MATX_HAVE_AOCL_SPARSE
	#include <aoclsparse.h>
#endif

// y = \alpha \, op(A) \, x + \beta \, y,
matx_status_t ref_spmv_c64_aocl(
	matx_complex_f64_t alpha,
	matx_coo_c64_t A,
	matx_vec_c64_t x,
	matx_complex_f64_t beta,
	matx_vec_c64_t y)
{
#if MATX_HAVE_AOCL_SPARSE
	if (!A || !x || !y)
		return MATX_ERR_INVALID_ARG;

	if (A->ncols != x->n || A->nrows != y->n)
		return MATX_ERR_INVALID_ARG;

	if (A->handle_aocl.valid <= 0)
	{
		if (coo_2_aocl_c64(A) != 0)
			return MATX_ERR_INTERNAL;
	}

	aoclsparse_mat_descr descr;
	aoclsparse_create_mat_descr(&descr);

	aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_general);
	aoclsparse_set_mat_diag_type(descr, aoclsparse_diag_type_non_unit);

	aoclsparse_double_complex a = { alpha.real, alpha.imag };
	aoclsparse_double_complex b = { beta.real, beta.imag };

	aoclsparse_status status =
		aoclsparse_zmv(
			aoclsparse_operation_none,
			&a,
			(aoclsparse_matrix)A->handle_aocl.impl,
			descr,
			(aoclsparse_double_complex*)x->data,
			&b,
			(aoclsparse_double_complex*)y->data);
	
	aoclsparse_destroy_mat_descr(descr);

	if (status != aoclsparse_status_success)
	{
		MATX_ERROR("aoclsparse_zmv error: %d", status);
		return MATX_ERR_INTERNAL;
	}

#endif
	return MATX_OK;
}

//    C = \alpha \, op(A) \, B + \beta \, C,
matx_status_t ref_spmm_c64_aocl(
	matx_complex_f64_t alpha,
	matx_coo_c64_t A,
	const matx_dense_c64_t B,
	matx_complex_f64_t beta,
	matx_dense_c64_t C)
{
#if MATX_HAVE_AOCL_SPARSE
	if (!A || !B || !C)
		return MATX_ERR_INVALID_ARG;

	if (A->handle_aocl.valid <= 0)
	{
		if (coo_2_aocl_c64(A) != 0)
			return MATX_ERR_INTERNAL;
	}

	aoclsparse_mat_descr descr;
	aoclsparse_create_mat_descr(&descr);

	aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_general);

	aoclsparse_double_complex a = { alpha.real, alpha.imag };
	aoclsparse_double_complex b = { beta.real, beta.imag };

	aoclsparse_status status =
		aoclsparse_zcsrmm(
			aoclsparse_operation_none,
			a,
			(aoclsparse_matrix)A->handle_aocl.impl,
			descr,
			aoclsparse_order_row,
			(aoclsparse_double_complex*)B->data,
			B->ncols,
			B->stride,
			b,
			(aoclsparse_double_complex*)C->data,
			C->ncols
		);
	aoclsparse_destroy_mat_descr(descr);
	if (status != aoclsparse_status_success)
	{
		MATX_ERROR("aoclsparse_zcsrmm error: %d", status);
		return MATX_ERR_INTERNAL;
	}
#endif
	return MATX_OK;
}

// y = \alpha \, op(A) \, x + \beta \, y
matx_status_t ref_spmv_f64_aocl(
	matx_double alpha,
	matx_coo_f64_t A,
	matx_vec_f64_t x,
	matx_double beta,
	matx_vec_f64_t y)
{
#if MATX_HAVE_AOCL_SPARSE

	if (!A || !x || !y)
		return MATX_ERR_INVALID_ARG;

	if (A->handle_aocl.valid <= 0)
	{
		if (coo_2_aocl_f64(A) != 0)
			return MATX_ERR_INTERNAL;
	}

	aoclsparse_mat_descr descr;
	aoclsparse_create_mat_descr(&descr);

	aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_general);
	aoclsparse_set_mat_diag_type(descr, aoclsparse_diag_type_non_unit);

	aoclsparse_status status = aoclsparse_set_mv_hint(A->handle_aocl.impl, aoclsparse_operation_none, descr, 1);
	if (status != aoclsparse_status_success)
	{
		MATX_ERROR("aoclsparse_set_mv_hint error: %d", status);
		return MATX_ERR_INTERNAL;
	}
	status =
		aoclsparse_dmv(
			aoclsparse_operation_none,
			&alpha,
			(aoclsparse_matrix)A->handle_aocl.impl,
			descr,
			x->data,
			&beta,
			y->data
		);
	aoclsparse_destroy_mat_descr(descr);
	if (status != aoclsparse_status_success)
	{
		MATX_ERROR("aoclsparse_dmv error: %d", status);
		return MATX_ERR_INTERNAL;
	}
#endif

	return MATX_OK;
}
//C = α * A * B + β * C
matx_status_t ref_spmm_f64_aocl(
	matx_double alpha,
	matx_coo_f64_t A,
	matx_dense_f64_t B,
	matx_double beta,
	matx_dense_f64_t C)
{
#if MATX_HAVE_AOCL_SPARSE
	if (!A || !B || !C)
		return MATX_ERR_INVALID_ARG;

	if (A->handle_aocl.valid <= 0)
	{
		if (coo_2_aocl_f64(A) != 0)
			return MATX_ERR_INTERNAL;
	}

	aoclsparse_mat_descr descr;
	aoclsparse_create_mat_descr(&descr);

	aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_general);
	aoclsparse_set_mat_diag_type(descr, aoclsparse_diag_type_non_unit);

	aoclsparse_status status =
		aoclsparse_dcsrmm(
			aoclsparse_operation_none, alpha,
			(aoclsparse_matrix)A->handle_aocl.impl,
			descr, aoclsparse_order_row,
			B->data,
			B->ncols, B->stride,
			beta,
			C->data,
			C->stride
		);

	aoclsparse_destroy_mat_descr(descr);
	if (status != aoclsparse_status_success)
	{
		MATX_ERROR("aoclsparse_dcsrmm error: %d", status);
		return MATX_ERR_INTERNAL;
	}

#endif
	return MATX_OK;
}

// C := α · op(A) · op(B) + β · C
matx_status_t ref_dsp2md_f64_aocl(
	matx_double alpha,
	matx_coo_f64_t A,
	matx_coo_f64_t B,
	matx_double beta,
	matx_dense_f64_t C)
{
#if MATX_HAVE_AOCL_SPARSE
	if (!A || !B || !C)
		return MATX_ERR_INVALID_ARG;

	if (A->handle_aocl.valid <= 0)
	{
		if (coo_2_aocl_f64(A) != 0)
			return MATX_ERR_INTERNAL;
	}

	if (B->handle_aocl.valid <= 0)
	{
		if (coo_2_aocl_f64(B) != 0)
			return MATX_ERR_INTERNAL;
	}

	aoclsparse_mat_descr descr;
	aoclsparse_create_mat_descr(&descr);

	aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_general);
	aoclsparse_set_mat_diag_type(descr, aoclsparse_diag_type_non_unit);

	aoclsparse_status status =
		aoclsparse_dsp2md(
			aoclsparse_operation_none, descr,
			(aoclsparse_matrix)A->handle_aocl.impl,
			aoclsparse_operation_none, descr,
			(aoclsparse_matrix)B->handle_aocl.impl,
			alpha, beta,
			C->data,
			aoclsparse_order_row,
			C->stride
		);

	aoclsparse_destroy_mat_descr(descr);

	if (status != aoclsparse_status_success)
	{
		MATX_ERROR("aoclsparse_dsp2md error: %d", status);
		return MATX_ERR_INTERNAL;
	}
#endif
	return MATX_OK;
}

// C := α · op(A) · op(B) + β · C
matx_status_t ref_zsp2md_c64_aocl(
	matx_complex_f64_t alpha,
	matx_coo_c64_t A,
	matx_coo_c64_t B,
	matx_complex_f64_t beta,
	matx_dense_c64_t C)
{
#if MATX_HAVE_AOCL_SPARSE
	if (!A || !B || !C)
		return MATX_ERR_INVALID_ARG;

	if (A->handle_aocl.valid <= 0)
	{
		if (coo_2_aocl_c64(A) != 0)
			return MATX_ERR_INTERNAL;
	}

	if (B->handle_aocl.valid <= 0)
	{
		if (coo_2_aocl_c64(B) != 0)
			return MATX_ERR_INTERNAL;
	}

	aoclsparse_mat_descr descr;
	aoclsparse_create_mat_descr(&descr);

	aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_general);
	aoclsparse_set_mat_diag_type(descr, aoclsparse_diag_type_non_unit);

	aoclsparse_double_complex a = { alpha.real, alpha.imag };
	aoclsparse_double_complex b = { beta.real, beta.imag };

	aoclsparse_status status =
		aoclsparse_zsp2md(
			aoclsparse_operation_none, descr,
			(aoclsparse_matrix)A->handle_aocl.impl,
			aoclsparse_operation_none, descr,
			(aoclsparse_matrix)B->handle_aocl.impl,
			a, 
			b,
                        (void*)C->data,
			aoclsparse_order_row,
			C->stride
		);

	aoclsparse_destroy_mat_descr(descr);

	if (status != aoclsparse_status_success)
	{
		MATX_ERROR("aoclsparse_zsp2md error: %d", status);
		return MATX_ERR_INTERNAL;
	}
#endif
	return MATX_OK;
}

matx_status_t ref_transpose_f64_aocl(
	matx_coo_f64_t A,
	matx_coo_f64_t out)
{
	if (!A)
		return MATX_ERR_INVALID_ARG;
	out->nrows = A->ncols;
	out->ncols = A->nrows;
	out->nnz = A->nnz;
	memcpy(out->rows, A->columns, sizeof(matx_int64_t) * A->nnz);
	memcpy(out->columns, A->rows, sizeof(matx_int64_t) * A->nnz);
	memcpy(out->values, A->values, sizeof(matx_double) * A->nnz);
	return MATX_OK;
}

matx_status_t ref_transpose_c64_aocl(
	matx_coo_f64_t A,
	matx_coo_f64_t out)
{
	if (!A)
		return MATX_ERR_INVALID_ARG;
	out->nrows = A->ncols;
	out->ncols = A->nrows;
	out->nnz = A->nnz;
	memcpy(out->rows, A->columns, sizeof(matx_int64_t) * A->nnz);
	memcpy(out->columns, A->rows, sizeof(matx_int64_t) * A->nnz);
	memcpy(out->values, A->values, sizeof(matx_complex_f64_t) * A->nnz);
	return MATX_OK;
}

matx_status_t ref_conj_trans_c64_aocl(matx_coo_c64_t A,
	matx_coo_c64_t out)
{
	if (!A || !out)
		return MATX_ERR_INVALID_ARG;

	if (A->handle_aocl.valid <= 0) {
		if (coo_2_aocl_c64(A) != 0)
			return MATX_ERR_INTERNAL;
	}

	aoclsparse_status status = aoclsparse_convert_csr(
		A->handle_aocl.impl,
		aoclsparse_operation_conjugate_transpose,
		out->handle_aocl.impl
	);

	if (status != aoclsparse_status_success)
	{
		MATX_ERROR("aoclsparse_convert_csr error: %d", status);
		return MATX_ERR_INTERNAL;
	}
	aocl_2_coo_c64(out);
	return MATX_OK;

}

matx_status_t ref_finalize_aocl()
{
}

matx_sparse_backend_t matx_sparse_make_reference_aocl(void) {
	matx_sparse_backend_t b =
	{
		.kind = MATX_SPARSE_BACKEND_AOCL_CPARSE,
		.vt = {
			.spmm_c64 = ref_spmm_c64_aocl,
			.spmv_c64 = ref_spmv_c64_aocl,
			.spmm_f64 = ref_spmm_f64_aocl,
			.spmv_f64 = ref_spmv_f64_aocl,
			.dsp2md_f64 = ref_dsp2md_f64_aocl,
			.zsp2md_c64 = ref_zsp2md_c64_aocl,
			.transpose_f64 = ref_transpose_f64_aocl,
			.transpose_c64 = ref_transpose_c64_aocl,
			.conj_trans_c64 = ref_conj_trans_c64_aocl,
			.finalize = ref_finalize_aocl
		}
	};
	MATX_TRACE("AOCL INIT");
	return b;
}

