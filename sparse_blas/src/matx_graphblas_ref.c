#include "matx/matx_types.h"
#include "matx/matx_func.h"
#include "matx/matx_types_internal.h"
#include "matx/matx_sparse_compute.h"
#include "matx/matx_log.h"
#include "matx/matx_tm.h"

#ifdef MATX_ENABLE_GRAPHBLAS
#include "GraphBLAS.h"
#endif

#include <limits.h>
#include <math.h>

//C(i,j)=k⨁​(A(i,k)⊗B(k,j))


matx_status_t ref_spmv_c64_grb(
	matx_complex_f64_t alpha,
	matx_coo_c64_t A,
	matx_vec_c64_t x,
	matx_complex_f64_t beta,
	matx_vec_c64_t y)
{
#ifdef MATX_ENABLE_GRAPHBLAS
	if (!A || !x || !y)
		return MATX_ERR_INVALID_ARG;

	if (A->ncols != x->n || A->nrows != y->n)
		return MATX_ERR_INVALID_ARG;

	if (A->handle_grb.valid <= 0)
		coo_2_grb_c64(A);
	if (x->handle_grb.valid <= 0)
		vec_2_grb_c64(x);
	if (y->handle_grb.valid <= 0)
		vec_2_grb_c64(y);

	GxB_FC64_t a = { alpha.real, alpha.imag };
	GxB_FC64_t b = { beta.real, beta.imag };

	// gy = beta * gy
	GrB_Info info = GrB_apply((GrB_Vector)y->handle_grb.impl, NULL, NULL,
		GxB_TIMES_FC64, (GrB_Vector)y->handle_grb.impl, b, NULL);
	if (info != GrB_SUCCESS) {
		MATX_ERROR("GrB_apply beta*gy error: %d", info);
		return MATX_ERR_INTERNAL;
	}
	// gy += alpha * A * x  (accumulate into gy)
	GrB_Vector temp;
	GrB_Vector_new(&temp, GxB_FC64, y->n);
	info = GrB_mxv(temp, NULL, NULL, GxB_PLUS_TIMES_FC64,
		(GrB_Matrix)A->handle_grb.impl, (GrB_Vector)x->handle_grb.impl, NULL);
	if (info != GrB_SUCCESS) {
		GrB_Vector_free(&temp);
		MATX_ERROR("GrB_mxv error: %d", info);
		return MATX_ERR_INTERNAL;
	}
	info = GrB_apply(temp, NULL, NULL, GxB_TIMES_FC64, temp, a, NULL);
	if (info != GrB_SUCCESS) {
		GrB_Vector_free(&temp);
		MATX_ERROR("GrB_apply alpha*temp error: %d", info);
		return MATX_ERR_INTERNAL;
	}
	info = GrB_eWiseAdd((GrB_Vector)y->handle_grb.impl, NULL, NULL,
		GxB_PLUS_FC64, (GrB_Vector)y->handle_grb.impl, temp, NULL);
	GrB_Vector_free(&temp);
	if (info != GrB_SUCCESS) {
		MATX_ERROR("GrB_eWiseAdd error: %d", info);
		return MATX_ERR_INTERNAL;
	}
	grb_2_vec_c64(y);
#endif
	return MATX_OK;
}

matx_status_t ref_spmm_c64_grb(
	matx_complex_f64_t alpha,
	matx_coo_c64_t A,
	matx_dense_c64_t B,
	matx_complex_f64_t beta,
	matx_dense_c64_t C)
{
#ifdef MATX_ENABLE_GRAPHBLAS
	if (!A || !B || !C)
		return MATX_ERR_INVALID_ARG;

	if (A->ncols != B->nrows ||
		A->nrows != C->nrows ||
		B->ncols != C->ncols)
		return MATX_ERR_INVALID_ARG;

	/* build A */
	if (A->handle_grb.valid <= 0)
	{
		coo_2_grb_c64(A);
	}
	if (B->handle_grb.valid <= 0)
	{
		dense_2_grb_c64(B);
	}
	if (C->handle_grb.valid <= 0)
	{
		dense_2_grb_c64(C);
	}

	/* C = alpha*A*B + beta*C */
	GxB_FC64_t a = { alpha.real,alpha.imag };
	GxB_FC64_t b = { beta.real,beta.imag };

	// 1. temp = alpha * A * B
	GrB_Matrix temp;
	GrB_Matrix_new(&temp, GxB_FC64, C->nrows, C->ncols);
	GrB_Info info = GrB_mxm(temp, NULL, NULL, GxB_PLUS_TIMES_FC64, (GrB_Matrix)A->handle_grb.impl, (GrB_Matrix)B->handle_grb.impl, NULL);
	if (info != GrB_SUCCESS)
	{
		MATX_ERROR("GrB_mxm error: %d", info);
		return MATX_ERR_INTERNAL;
	}
	info = GrB_apply(temp, NULL, NULL, GxB_TIMES_FC64, temp, a, NULL);
	if (info != GrB_SUCCESS)
	{
		MATX_ERROR("GrB_apply error: %d", info);
		return MATX_ERR_INTERNAL;
	}
	//2. gC = beta * gC
	info = GrB_apply((GrB_Matrix)C->handle_grb.impl, NULL, NULL, GxB_TIMES_FC64, (GrB_Matrix)C->handle_grb.impl, b, NULL);
	if (info != GrB_SUCCESS)
	{
		MATX_ERROR("GrB_apply beta * gC error: %d", info);
		return MATX_ERR_INTERNAL;
	}
	//3. gC = temp + gC
	info = GrB_eWiseAdd((GrB_Matrix)C->handle_grb.impl, NULL, NULL, GxB_PLUS_FC64, temp, (GrB_Matrix)C->handle_grb.impl, NULL);
	if (info != GrB_SUCCESS)
	{
		MATX_ERROR("GrB_eWiseAdd temp + gC error: %d", info);
		return MATX_ERR_INTERNAL;
	}
	GrB_Matrix_free(&temp);
	grb_2_dense_c64(C);
#endif
	return MATX_OK;
}

matx_status_t ref_spmv_f64_grb(
	matx_double alpha,
	matx_coo_f64_t A,
	matx_vec_f64_t x,
	matx_double beta,
	matx_vec_f64_t y)
{
#ifdef MATX_ENABLE_GRAPHBLAS
	if (!A || !x || !y)
		return MATX_ERR_INVALID_ARG;

	if (A->ncols != x->n || A->nrows != y->n)
		return MATX_ERR_INVALID_ARG;

	if (A->handle_grb.valid <= 0)
		coo_2_grb_f64(A);
	if (x->handle_grb.valid <= 0)
		vec_2_grb_f64(x);
	if (y->handle_grb.valid <= 0)
		vec_2_grb_f64(y);

	// gy = beta * gy
	GrB_Info info = GrB_apply((GrB_Vector)y->handle_grb.impl, NULL, NULL,
		GrB_TIMES_FP64, (GrB_Vector)y->handle_grb.impl, beta, NULL);
	if (info != GrB_SUCCESS) {
		MATX_ERROR("GrB_apply beta*gy error: %d", info);
		return MATX_ERR_INTERNAL;
	}
	// temp = A*x, then gy += alpha*temp
	matx_int64_t t0 = matx_tm_now(MATX_TM_MICROSECOND);
	GrB_Vector temp;
	GrB_Vector_new(&temp, GrB_FP64, y->n);
	info = GrB_mxv(temp, NULL, NULL, GxB_PLUS_TIMES_FP64,
		(GrB_Matrix)A->handle_grb.impl, (GrB_Vector)x->handle_grb.impl, NULL);
	if (info != GrB_SUCCESS) {
		GrB_Vector_free(&temp);
		MATX_ERROR("GrB_mxv A*x error: %d", info);
		return MATX_ERR_INTERNAL;
	}
	info = GrB_apply(temp, NULL, NULL, GrB_TIMES_FP64, temp, alpha, NULL);
	if (info != GrB_SUCCESS) {
		GrB_Vector_free(&temp);
		MATX_ERROR("GrB_apply alpha*temp error: %d", info);
		return MATX_ERR_INTERNAL;
	}
	info = GrB_eWiseAdd((GrB_Vector)y->handle_grb.impl, NULL, NULL,
		GrB_PLUS_FP64, (GrB_Vector)y->handle_grb.impl, temp, NULL);
	GrB_Vector_free(&temp);
	if (info != GrB_SUCCESS) {
		MATX_ERROR("GrB_eWiseAdd temp + gy error: %d", info);
		return MATX_ERR_INTERNAL;
	}
	matx_int64_t t1 = matx_tm_now(MATX_TM_MICROSECOND);
	MATX_TRACE("GraphBLAS SpMV time: %ld micro.s", t1 - t0);
	grb_2_vec_f64(y);
#endif
	return MATX_OK;
}

matx_status_t ref_spmm_f64_grb(
	matx_double alpha,
	matx_coo_f64_t A,
	matx_dense_f64_t B,
	matx_double beta,
	matx_dense_f64_t C)
{
#ifdef MATX_ENABLE_GRAPHBLAS
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
	GrB_Info info = GrB_Matrix_new(&temp, GrB_FP64, C->nrows, C->ncols);
	info = GrB_mxm(temp, NULL, NULL, GxB_PLUS_TIMES_FP64, (GrB_Matrix)A->handle_grb.impl, (GrB_Matrix)B->handle_grb.impl, NULL);
	if (info != GrB_SUCCESS)
	{
		MATX_ERROR("GrB_mxm alpha * A * B error: %d", info);
		return MATX_ERR_INTERNAL;
	}
	info = GrB_apply(temp, NULL, NULL, GrB_TIMES_FP64, temp, alpha, NULL);
	if (info != GrB_SUCCESS)
	{
		MATX_ERROR("GrB_apply  error: %d", info);
		return MATX_ERR_INTERNAL;
	}
	//2. gC = beta * gC
	info = GrB_apply((GrB_Matrix)C->handle_grb.impl, NULL, NULL, GrB_TIMES_FP64, (GrB_Matrix)C->handle_grb.impl, beta, NULL);
	if (info != GrB_SUCCESS)
	{
		MATX_ERROR("GrB_apply beta * gC error: %d", info);
		return MATX_ERR_INTERNAL;
	}
	//3. gC = temp + gC
	info = GrB_eWiseAdd((GrB_Matrix)C->handle_grb.impl, NULL, NULL, GrB_PLUS_FP64, temp, (GrB_Matrix)C->handle_grb.impl, NULL);
	if (info != GrB_SUCCESS)
	{
		MATX_ERROR("GrB_eWiseAdd temp + gC error: %d", info);
		return MATX_ERR_INTERNAL;
	}
	GrB_Matrix_free(&temp);
	grb_2_dense_f64(C);
#endif
	return MATX_OK;
}

matx_status_t ref_dsp2md_f64_grb(
	matx_double alpha,
	matx_coo_f64_t A,
	matx_coo_f64_t B,
	matx_double beta,
	matx_dense_f64_t C)
{
#ifdef MATX_ENABLE_GRAPHBLAS
	if (!A || !B || !C)
		return MATX_ERR_INVALID_ARG;

	/* build A */
	if (A->handle_grb.valid <= 0)
	{
		coo_2_grb_f64(A);
	}
	if (B->handle_grb.valid <= 0)
	{
		coo_2_grb_f64(B);
	}
	if (C->handle_grb.valid <= 0)
	{
		dense_2_grb_f64(C);
	}

	/* C = alpha*A*B + beta*C */

	// 1. temp = alpha * A * B
	GrB_Matrix temp;
	GrB_Info info = GrB_Matrix_new(&temp, GrB_FP64, C->nrows, C->ncols);
	info = GrB_mxm(temp, NULL, NULL, GxB_PLUS_TIMES_FP64, (GrB_Matrix)A->handle_grb.impl, (GrB_Matrix)B->handle_grb.impl, NULL);
	if (info != GrB_SUCCESS)
	{
		MATX_ERROR("GrB_mxm alpha * A * B error: %d", info);
		return MATX_ERR_INTERNAL;
	}
	info = GrB_apply(temp, NULL, NULL, GrB_TIMES_FP64, temp, alpha, NULL);
	if (info != GrB_SUCCESS)
	{
		MATX_ERROR("GrB_apply alpha * A * B error: %d", info);
		return MATX_ERR_INTERNAL;
	}
	//2. gC = beta * gC
	info = GrB_apply((GrB_Matrix)C->handle_grb.impl, NULL, NULL, GrB_TIMES_FP64, (GrB_Matrix)C->handle_grb.impl, beta, NULL);
	if (info != GrB_SUCCESS)
	{
		MATX_ERROR("GrB_apply beta * gC error: %d", info);
		return MATX_ERR_INTERNAL;
	}
	//3. gC = temp + gC
	info = GrB_eWiseAdd((GrB_Matrix)C->handle_grb.impl, NULL, NULL, GrB_PLUS_FP64, temp, (GrB_Matrix)C->handle_grb.impl, NULL);
	if (info != GrB_SUCCESS)
	{
		MATX_ERROR("GrB_eWiseAdd temp + gC error: %d", info);
		return MATX_ERR_INTERNAL;
	}
	GrB_Matrix_free(&temp);
	grb_2_dense_f64(C);
#endif
	return MATX_OK;
}

matx_status_t ref_zsp2md_c64_grb(
	matx_complex_f64_t alpha,
	matx_coo_c64_t A,
	matx_coo_c64_t B,
	matx_complex_f64_t beta,
	matx_dense_c64_t C)
{
#ifdef MATX_ENABLE_GRAPHBLAS
	if (!A || !B || !C)
		return MATX_ERR_INVALID_ARG;

	/* build A */
	if (A->handle_grb.valid <= 0)
	{
		coo_2_grb_c64(A);
	}
	if (B->handle_grb.valid <= 0)
	{
		coo_2_grb_c64(B);
	}
	if (C->handle_grb.valid <= 0)
	{
		dense_2_grb_c64(C);
	}

	/* C = alpha*A*B + beta*C */
	GxB_FC64_t a = { alpha.real, alpha.imag };
	GxB_FC64_t b = { beta.real, beta.imag };
	// 1. temp = alpha * A * B
	GrB_Matrix temp;
	GrB_Info info = GrB_Matrix_new(&temp, GxB_FC64, C->nrows, C->ncols);
	info = GrB_mxm(temp, NULL, NULL, GxB_PLUS_TIMES_FC64, (GrB_Matrix)A->handle_grb.impl, (GrB_Matrix)B->handle_grb.impl, NULL);
	if (info != GrB_SUCCESS)
	{
		MATX_ERROR("GrB_mxm alpha * A * B error: %d", info);
		return MATX_ERR_INTERNAL;
	}
	info = GrB_apply(temp, NULL, NULL, GxB_TIMES_FC64, temp, a, NULL);
	if (info != GrB_SUCCESS)
	{
		MATX_ERROR("GrB_apply error: %d", info);
		return MATX_ERR_INTERNAL;
	}
	//2. gC = beta * gC
	info = GrB_apply((GrB_Matrix)C->handle_grb.impl, NULL, NULL, GxB_TIMES_FC64, (GrB_Matrix)C->handle_grb.impl, b, NULL);
	if (info != GrB_SUCCESS)
	{
		MATX_ERROR("GrB_apply beta * gC error: %d", info);
		return MATX_ERR_INTERNAL;
	}
	//3. gC = temp + gC
	info = GrB_eWiseAdd((GrB_Matrix)C->handle_grb.impl, NULL, NULL, GxB_PLUS_FC64, temp, (GrB_Matrix)C->handle_grb.impl, NULL);
	if (info != GrB_SUCCESS)
	{
		MATX_ERROR("GrB_eWiseAdd temp + gC error: %d", info);
		return MATX_ERR_INTERNAL;
	}
	GrB_Matrix_free(&temp);
	grb_2_dense_c64(C);
#endif
	return MATX_OK;
}

matx_status_t ref_transpose_f64_grb(
	matx_coo_f64_t A,
	matx_coo_f64_t out)
{
#ifdef MATX_ENABLE_GRAPHBLAS
	if (!A)
		return MATX_ERR_INVALID_ARG;

	/* build A */
	if (A->handle_grb.valid <= 0)
	{
		coo_2_grb_f64(A);
	}
	create_empty_grb_f64(out);
	GrB_Info info = GrB_transpose((GrB_Matrix)out->handle_grb.impl, NULL, NULL, (GrB_Matrix)A->handle_grb.impl, NULL);
	if (info != GrB_SUCCESS)
	{
		MATX_ERROR("GrB_transpose error: %d", info);
		return MATX_ERR_INTERNAL;
	}
	grb_2_coo_f64(out);
#endif
	return MATX_OK;
}

matx_status_t ref_transpose_c64_grb(
	matx_coo_c64_t A,
	matx_coo_c64_t out)
{
#ifdef MATX_ENABLE_GRAPHBLAS
	if (!A || !out)
		return MATX_ERR_INVALID_ARG;

	/* build A */
	if (A->handle_grb.valid <= 0)
	{
		coo_2_grb_c64(A);
	}
	if (!out->handle_grb.impl)
		create_empty_grb_c64(out);
	GrB_Info info = GrB_transpose((GrB_Matrix)(out->handle_grb.impl), NULL, NULL, (GrB_Matrix)(A->handle_grb.impl), NULL);
	if (info != GrB_SUCCESS)
	{
		MATX_ERROR("GrB_transpose error: %d", info);
		return MATX_ERR_INTERNAL;
	}
	grb_2_coo_c64(out);
#endif
	return MATX_OK;
}

matx_status_t ref_conj_trans_c64_grb(matx_coo_c64_t A,
	matx_coo_c64_t out)
{
#ifdef MATX_ENABLE_GRAPHBLAS
	if (!A)
		return MATX_ERR_INVALID_ARG;

	/* build A */
	if (A->handle_grb.valid <= 0)
	{
		coo_2_grb_c64(A);
	}
	if (!out->handle_grb.impl)
		create_empty_grb_c64(out);
	//1. transpose
	matx_status_t trans_status = ref_transpose_c64_grb(A, out);
	if (trans_status != MATX_OK)
	{
		MATX_ERROR("GrB_transpose error: %d", trans_status);
		return MATX_ERR_INTERNAL;
	}
	//2.  conj
	GrB_Info info = GrB_apply((GrB_Matrix)out->handle_grb.impl, NULL, NULL,
		GxB_CONJ_FC64, (GrB_Matrix)out->handle_grb.impl, NULL);
	if (info != GrB_SUCCESS)
	{
		MATX_ERROR("GrB_CONJ error: %d", info);
		return MATX_ERR_INTERNAL;
	}
	grb_2_coo_c64(out);
#endif
	return MATX_OK;

}


matx_status_t ref_finalize_grb()
{
#ifdef MATX_ENABLE_GRAPHBLAS
	GrB_finalize();
#endif
	return MATX_OK;
}

// ---- Sparse matrix norms ----

matx_status_t ref_norm1_mat_grb(matx_coo_f64_t A, matx_double* out) {
#ifdef MATX_ENABLE_GRAPHBLAS
	if (!A || !out) return MATX_ERR_INVALID_ARG;
	if (A->handle_grb.valid <= 0) coo_2_grb_f64(A);

	GrB_Matrix tmp;
	GrB_Matrix_dup(&tmp, (GrB_Matrix)A->handle_grb.impl);
	GrB_apply(tmp, NULL, NULL, GrB_ABS_FP64, tmp, NULL);
	GrB_Vector col_sums;
	GrB_Vector_new(&col_sums, GrB_FP64, A->ncols);
	GrB_reduce(col_sums, NULL, NULL, GrB_PLUS_MONOID_FP64, tmp, NULL);
	GrB_Matrix_free(&tmp);
	GrB_reduce(out, NULL, GrB_MAX_MONOID_FP64, col_sums, NULL);
	GrB_Vector_free(&col_sums);
#endif
	return MATX_OK;
}

matx_status_t ref_norminf_mat_grb(matx_coo_f64_t A, matx_double* out) {
#ifdef MATX_ENABLE_GRAPHBLAS
	if (!A || !out) return MATX_ERR_INVALID_ARG;
	if (A->handle_grb.valid <= 0) coo_2_grb_f64(A);

	GrB_Matrix tmp;
	GrB_Matrix_dup(&tmp, (GrB_Matrix)A->handle_grb.impl);
	GrB_apply(tmp, NULL, NULL, GrB_ABS_FP64, tmp, NULL);
	GrB_Vector row_sums;
	GrB_Vector_new(&row_sums, GrB_FP64, A->nrows);
	GrB_reduce(row_sums, NULL, NULL, GrB_PLUS_MONOID_FP64, tmp, GrB_DESC_T0);
	GrB_Matrix_free(&tmp);
	GrB_reduce(out, NULL, GrB_MAX_MONOID_FP64, row_sums, NULL);
	GrB_Vector_free(&row_sums);
#endif
	return MATX_OK;
}

matx_status_t ref_normfro_mat_grb(matx_coo_f64_t A, matx_double* out) {
#ifdef MATX_ENABLE_GRAPHBLAS
	if (!A || !out) return MATX_ERR_INVALID_ARG;
	if (A->handle_grb.valid <= 0) coo_2_grb_f64(A);

	GrB_Matrix tmp;
	GrB_Matrix_dup(&tmp, (GrB_Matrix)A->handle_grb.impl);
	GrB_eWiseMult(tmp, NULL, NULL, GrB_TIMES_FP64, tmp, tmp, NULL);
	double sumsq = 0.0;
	GrB_reduce(&sumsq, NULL, GrB_PLUS_MONOID_FP64, tmp, NULL);
	GrB_Matrix_free(&tmp);
	*out = sqrt(sumsq);
#endif
	return MATX_OK;
}

// ---- Sparse matrix norms (c64) ----

matx_status_t ref_norm1_mat_c64_grb(matx_coo_c64_t A, matx_double* out) {
#ifdef MATX_ENABLE_GRAPHBLAS
	if (!A || !out) return MATX_ERR_INVALID_ARG;
	if (A->handle_grb.valid <= 0) coo_2_grb_c64(A);

	GrB_Matrix abs_mat;
	GrB_Matrix_new(&abs_mat, GrB_FP64, A->nrows, A->ncols);
	GrB_apply(abs_mat, NULL, NULL, GxB_ABS_FC64, (GrB_Matrix)A->handle_grb.impl, NULL);
	GrB_Vector col_sums;
	GrB_Vector_new(&col_sums, GrB_FP64, A->ncols);
	GrB_reduce(col_sums, NULL, NULL, GrB_PLUS_MONOID_FP64, abs_mat, NULL);
	GrB_Matrix_free(&abs_mat);
	GrB_reduce(out, NULL, GrB_MAX_MONOID_FP64, col_sums, NULL);
	GrB_Vector_free(&col_sums);
#endif
	return MATX_OK;
}

matx_status_t ref_norminf_mat_c64_grb(matx_coo_c64_t A, matx_double* out) {
#ifdef MATX_ENABLE_GRAPHBLAS
	if (!A || !out) return MATX_ERR_INVALID_ARG;
	if (A->handle_grb.valid <= 0) coo_2_grb_c64(A);

	GrB_Matrix abs_mat;
	GrB_Matrix_new(&abs_mat, GrB_FP64, A->nrows, A->ncols);
	GrB_apply(abs_mat, NULL, NULL, GxB_ABS_FC64, (GrB_Matrix)A->handle_grb.impl, NULL);
	GrB_Vector row_sums;
	GrB_Vector_new(&row_sums, GrB_FP64, A->nrows);
	GrB_reduce(row_sums, NULL, NULL, GrB_PLUS_MONOID_FP64, abs_mat, GrB_DESC_T0);
	GrB_Matrix_free(&abs_mat);
	GrB_reduce(out, NULL, GrB_MAX_MONOID_FP64, row_sums, NULL);
	GrB_Vector_free(&row_sums);
#endif
	return MATX_OK;
}

matx_status_t ref_normfro_mat_c64_grb(matx_coo_c64_t A, matx_double* out) {
#ifdef MATX_ENABLE_GRAPHBLAS
	if (!A || !out) return MATX_ERR_INVALID_ARG;
	if (A->handle_grb.valid <= 0) coo_2_grb_c64(A);

	GrB_Matrix abs_mat;
	GrB_Matrix_new(&abs_mat, GrB_FP64, A->nrows, A->ncols);
	GrB_apply(abs_mat, NULL, NULL, GxB_ABS_FC64, (GrB_Matrix)A->handle_grb.impl, NULL);
	GrB_eWiseMult(abs_mat, NULL, NULL, GrB_TIMES_FP64, abs_mat, abs_mat, NULL);
	double sumsq = 0.0;
	GrB_reduce(&sumsq, NULL, GrB_PLUS_MONOID_FP64, abs_mat, NULL);
	GrB_Matrix_free(&abs_mat);
	*out = sqrt(sumsq);
#endif
	return MATX_OK;
}

// ---- Sparse-sparse addition ----

matx_status_t ref_spadd_f64_grb(matx_double alpha, matx_coo_f64_t A,
	matx_double beta, matx_coo_f64_t B, matx_coo_f64_t out) {
#ifdef MATX_ENABLE_GRAPHBLAS
	if (!A || !B || !out) return MATX_ERR_INVALID_ARG;
	if (A->nrows != B->nrows || A->ncols != B->ncols) return MATX_ERR_INVALID_ARG;
	if (A->handle_grb.valid <= 0) coo_2_grb_f64(A);
	if (B->handle_grb.valid <= 0) coo_2_grb_f64(B);

	GrB_Matrix temp_a;
	GrB_Matrix_dup(&temp_a, (GrB_Matrix)A->handle_grb.impl);
	GrB_apply(temp_a, NULL, NULL, GrB_TIMES_FP64, temp_a, alpha, NULL);
	GrB_Matrix temp_b;
	GrB_Matrix_dup(&temp_b, (GrB_Matrix)B->handle_grb.impl);
	GrB_apply(temp_b, NULL, NULL, GrB_TIMES_FP64, temp_b, beta, NULL);

	create_empty_grb_f64(out);
	GrB_Info info = GrB_eWiseAdd((GrB_Matrix)out->handle_grb.impl, NULL, NULL,
		GrB_PLUS_FP64, temp_a, temp_b, NULL);
	GrB_Matrix_free(&temp_a);
	GrB_Matrix_free(&temp_b);
	if (info != GrB_SUCCESS) {
		MATX_ERROR("GrB_eWiseAdd spadd_f64 error: %d", info);
		return MATX_ERR_INTERNAL;
	}
	grb_2_coo_f64(out);
#endif
	return MATX_OK;
}

matx_status_t ref_spadd_c64_grb(matx_complex_f64_t alpha, matx_coo_c64_t A,
	matx_complex_f64_t beta, matx_coo_c64_t B, matx_coo_c64_t out) {
#ifdef MATX_ENABLE_GRAPHBLAS
	if (!A || !B || !out) return MATX_ERR_INVALID_ARG;
	if (A->nrows != B->nrows || A->ncols != B->ncols) return MATX_ERR_INVALID_ARG;
	if (A->handle_grb.valid <= 0) coo_2_grb_c64(A);
	if (B->handle_grb.valid <= 0) coo_2_grb_c64(B);

	GxB_FC64_t a = { alpha.real, alpha.imag };
	GxB_FC64_t b = { beta.real, beta.imag };
	GrB_Matrix temp_a;
	GrB_Matrix_dup(&temp_a, (GrB_Matrix)A->handle_grb.impl);
	GrB_apply(temp_a, NULL, NULL, GxB_TIMES_FC64, temp_a, a, NULL);
	GrB_Matrix temp_b;
	GrB_Matrix_dup(&temp_b, (GrB_Matrix)B->handle_grb.impl);
	GrB_apply(temp_b, NULL, NULL, GxB_TIMES_FC64, temp_b, b, NULL);

	create_empty_grb_c64(out);
	GrB_Info info = GrB_eWiseAdd((GrB_Matrix)out->handle_grb.impl, NULL, NULL,
		GxB_PLUS_FC64, temp_a, temp_b, NULL);
	GrB_Matrix_free(&temp_a);
	GrB_Matrix_free(&temp_b);
	if (info != GrB_SUCCESS) {
		MATX_ERROR("GrB_eWiseAdd spadd_c64 error: %d", info);
		return MATX_ERR_INTERNAL;
	}
	grb_2_coo_c64(out);
#endif
	return MATX_OK;
}

static matx_bool grb_init_ok = false;

static void do_grb_init(void) {
#ifdef MATX_ENABLE_GRAPHBLAS
    GrB_Info info = GrB_init(GrB_NONBLOCKING);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("GraphBLAS initialization failed with error code %d", info);
    }
    else {
        grb_init_ok = true;
    }
#endif
}

#ifdef _WIN32
#include <windows.h>
static INIT_ONCE grb_init_flag = INIT_ONCE_STATIC_INIT;
static BOOL CALLBACK do_grb_init_win(PINIT_ONCE InitOnce, PVOID Parameter, PVOID* Context) {
    (void)InitOnce; (void)Parameter; (void)Context;
    do_grb_init();
    return TRUE;
}
#define matx_call_once(flag, func) \
InitOnceExecuteOnce(flag, do_grb_init_win, NULL, NULL)
#else
#include <threads.h>
#define matx_call_once(flag, func) call_once(flag, func)
#endif

matx_sparse_backend_t matx_sparse_make_reference_grb(void) {
    matx_call_once(&grb_init_flag, do_grb_init);
	if (!grb_init_ok) {
		MATX_ERROR("GraphBLAS initialization failed");
	}
	matx_sparse_backend_t b =
	{
		.kind = MATX_SPARSE_BACKEND_GRAPHBLAS,
		.vt = {
			.spmm_c64 = ref_spmm_c64_grb,
			.spmv_c64 = ref_spmv_c64_grb,
			.spmm_f64 = ref_spmm_f64_grb,
			.spmv_f64 = ref_spmv_f64_grb,
			.dsp2md_f64 = ref_dsp2md_f64_grb,
			.zsp2md_c64 = ref_zsp2md_c64_grb,
			.transpose_f64 = ref_transpose_f64_grb,
			.transpose_c64 = ref_transpose_c64_grb,
			.conj_trans_c64 = ref_conj_trans_c64_grb,
			.finalize = ref_finalize_grb,
			.norm1_mat_f64 = ref_norm1_mat_grb,
			.norminf_mat_f64 = ref_norminf_mat_grb,
			.normfro_mat_f64 = ref_normfro_mat_grb,
			.norm1_mat_c64 = ref_norm1_mat_c64_grb,
			.norminf_mat_c64 = ref_norminf_mat_c64_grb,
			.normfro_mat_c64 = ref_normfro_mat_c64_grb,
			.spadd_f64 = ref_spadd_f64_grb,
			.spadd_c64 = ref_spadd_c64_grb,
		}
	};
	return b;
}