#include "matx/matx.h"
#include "matx/matx_sparse_compute.h"

#include <limits.h>

#include <GraphBLAS.h>
#include "matx/matx_log.h"
#include "matx/matx_tm.h"

matx_status_t ref_spmv_c64(
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
		coo_2_grb_c64(A);
	}
	/* ---------------- build vectors ---------------- */
	if (x->handle_grb.valid <= 0)
	{
		vec_2_grb_c64(x);
	}

	if (y->handle_grb.valid <= 0)
	{
		vec_2_grb_c64(y);
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
	grb_2_vec_c64(y);
	return MATX_OK;
}

matx_status_t ref_spmm_c64(
	matx_complex_f64 alpha,
	const matx_coo_c64_t* A,
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
	GrB_Matrix_new(&temp, GxB_FC64, C->rows, C->cols);
	info = GrB_mxm(temp, NULL, NULL, GxB_PLUS_TIMES_FC64, *(GrB_Matrix*)A->handle_grb.impl, *(GrB_Matrix*)B->handle_grb.impl, NULL);
	info = GrB_apply(temp, NULL, NULL, GxB_TIMES_FC64, temp, &a, NULL);
	//2. gC = beta * gC
	info = GrB_apply(*(GrB_Matrix*)C->handle_grb.impl, NULL, NULL, GxB_TIMES_FC64, *(GrB_Matrix*)C->handle_grb.impl, &b, NULL);
	//3. gC = temp + gC
	info = GrB_eWiseAdd(*(GrB_Matrix*)C->handle_grb.impl, NULL, NULL, GxB_PLUS_FC64, temp, *(GrB_Matrix*)C->handle_grb.impl, NULL);
	GrB_Matrix_free(&temp);
	grb_2_dense_c64(C);
	return MATX_OK;
}

matx_status_t ref_spmv_f64(
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
	matx_int64_t t0 = matx_tm_now(MATX_TM_MICROSECOND);
	info = GrB_Vector_new(&temp, GrB_FP64, y->n);
	// temp = A*x
	info = GrB_mxv(temp, NULL, NULL, GxB_PLUS_TIMES_FP64, *(GrB_Matrix*)A->handle_grb.impl, *(GrB_Vector*)x->handle_grb.impl, NULL);
	// temp = alpha*temp
	info = GrB_apply(temp, NULL, NULL, GrB_TIMES_FP64, temp, &alpha, NULL);
	// gy = beta*gy
	info = GrB_apply(*(GrB_Vector*)y->handle_grb.impl, NULL, NULL, GrB_TIMES_FP64, *(GrB_Vector*)y->handle_grb.impl, &beta, NULL);
	// gy = temp + gy
	info = GrB_eWiseAdd(*(GrB_Vector*)y->handle_grb.impl, NULL, NULL, GrB_PLUS_FP64, temp, *(GrB_Vector*)y->handle_grb.impl, NULL);
	matx_int64_t t1 = matx_tm_now(MATX_TM_MICROSECOND);
	//printf("GraphBLAS SpMV time: %ld us\n", t1 - t0);
	matx_log_init("./logs");
	MATX_TRACE("GraphBLAS SpMV time: %ld micro.s", t1 - t0);
	GrB_Vector_free(&temp);
	grb_2_vec_f64(y);
	return MATX_OK;
}

matx_status_t ref_spmm_f64(
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

matx_sparse_backend_t matx_sparse_make_reference_grb(void) {
	matx_sparse_backend_t b;
	b.kind = MATX_SPARSE_BACKEND_GRAPHBLAS;
	b.vt.spmm_c64 = ref_spmm_c64;
	b.vt.spmv_c64 = ref_spmv_c64;
	b.vt.spmm_f64 = ref_spmm_f64;
	b.vt.spmv_f64 = ref_spmv_f64;
	return b;
}

