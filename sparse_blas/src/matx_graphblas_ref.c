#include "matx/matx_types.h"
#include "matx/matx_func.h"
#include "matx/matx_types_internal.h"
#include "matx/matx_sparse_compute.h"

#include <limits.h>

#include <GraphBLAS.h>
#include "matx/matx_log.h"
#include "matx/matx_tm.h"

//C(i,j)=k⨁​(A(i,k)⊗B(k,j))
//C(i,j)=k⨁​(A(i,k)⊗B(k,j))
//C(i,j)=k⨁​(A(i,k)⊗B(k,j))
//C(i,j)=k⨁​(A(i,k)⊗B(k,j))
//C(i,j)=k⨁​(A(i,k)⊗B(k,j))


matx_status_t ref_spmv_c64_grb(
	matx_complex_f64 alpha,
	matx_coo_c64_t A,
	matx_vec_c64_t x,
	matx_complex_f64 beta,
	matx_vec_c64_t y)
{
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
	GrB_Info info = GrB_mxv(temp, NULL, NULL, GxB_PLUS_TIMES_FC64, (GrB_Matrix)A->handle_grb.impl, (GrB_Vector)x->handle_grb.impl, NULL);
	if (info != GrB_SUCCESS)
	{
		MATX_ERROR("GrB_mxv error: %d", info);
		return MATX_ERR_INTERNAL;
	}
	// temp = alpha*temp
	info = GrB_apply(temp, NULL, NULL, GxB_TIMES_FC64, temp, a, NULL);
	if (info != GrB_SUCCESS)
	{
		MATX_ERROR("GrB_apply alpha*temp error: %d", info);
		return MATX_ERR_INTERNAL;
	}
	// gy = beta*gy
	info = GrB_apply((GrB_Vector)y->handle_grb.impl, NULL, NULL, GxB_TIMES_FC64, (GrB_Vector)y->handle_grb.impl, b, NULL);
	if (info != GrB_SUCCESS)
	{
		MATX_ERROR("GrB_apply beta*gy error: %d", info);
		return MATX_ERR_INTERNAL;
	}
	// gy = temp + gy
	info = GrB_eWiseAdd((GrB_Vector)y->handle_grb.impl, NULL, NULL, GxB_PLUS_FC64, temp, (GrB_Vector)y->handle_grb.impl, NULL);
	if (info != GrB_SUCCESS)
	{
		MATX_ERROR("GrB_eWiseAdd temp + gy error: %d", info);
		return MATX_ERR_INTERNAL;
	}
	GrB_Vector_free(&temp);
	grb_2_vec_c64(y);
	return MATX_OK;
}

matx_status_t ref_spmm_c64_grb(
	matx_complex_f64 alpha,
	matx_coo_c64_t A,
	matx_dense_c64_t B,
	matx_complex_f64 beta,
	matx_dense_c64_t C)
{
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
	return MATX_OK;
}

matx_status_t ref_spmv_f64_grb(
	matx_double alpha,
	matx_coo_f64_t A,
	matx_vec_f64_t x,
	matx_double beta,
	matx_vec_f64_t y)
{
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
	GrB_Info info = GrB_Vector_new(&temp, GrB_FP64, y->n);
	// temp = A*x
	info = GrB_mxv(temp, NULL, NULL, GxB_PLUS_TIMES_FP64, (GrB_Matrix)A->handle_grb.impl, (GrB_Vector)x->handle_grb.impl, NULL);
	if (info != GrB_SUCCESS)
	{
		MATX_ERROR("GrB_mxv A*x error: %d", info);
		return MATX_ERR_INTERNAL;
	}
	// temp = alpha*temp
	info = GrB_apply(temp, NULL, NULL, GrB_TIMES_FP64, temp, alpha, NULL);
	if (info != GrB_SUCCESS)
	{
		MATX_ERROR("GrB_apply alpha*temp error: %d", info);
		return MATX_ERR_INTERNAL;
	}
	// gy = beta*gy
	info = GrB_apply((GrB_Vector)y->handle_grb.impl, NULL, NULL, GrB_TIMES_FP64, (GrB_Vector)y->handle_grb.impl, beta, NULL);
	if (info != GrB_SUCCESS)
	{
		MATX_ERROR("GrB_apply beta*gy error: %d", info);
		return MATX_ERR_INTERNAL;
	}
	// gy = temp + gy
	info = GrB_eWiseAdd((GrB_Vector)y->handle_grb.impl, NULL, NULL, GrB_PLUS_FP64, temp, (GrB_Vector)y->handle_grb.impl, NULL);
	if (info != GrB_SUCCESS)
	{
		MATX_ERROR("GrB_eWiseAdd temp + gC error: %d", info);
		return MATX_ERR_INTERNAL;
	}
	matx_int64_t t1 = matx_tm_now(MATX_TM_MICROSECOND);
	//printf("GraphBLAS SpMV time: %ld us\n", t1 - t0);
	MATX_TRACE("GraphBLAS SpMV time: %ld micro.s", t1 - t0);
	GrB_Vector_free(&temp);
	grb_2_vec_f64(y);
	return MATX_OK;
}

matx_status_t ref_spmm_f64_grb(
	matx_double alpha,
	matx_coo_f64_t A,
	matx_dense_f64_t B,
	matx_double beta,
	matx_dense_f64_t C)
{
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
	return MATX_OK;
}

matx_status_t ref_dsp2md_f64_grb(
	matx_double alpha,
	matx_coo_f64_t A,
	matx_coo_f64_t B,
	matx_double beta,
	matx_dense_f64_t C)
{
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
	return MATX_OK;
}

matx_status_t ref_zsp2md_c64_grb(
	matx_complex_f64 alpha,
	matx_coo_c64_t A,
	matx_coo_c64_t B,
	matx_complex_f64 beta,
	matx_dense_c64_t C)
{
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
	return MATX_OK;
}

matx_status_t ref_transpose_f64_grb(
	matx_coo_f64_t A, 
	matx_coo_f64_t out)
{
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
	return MATX_OK;
}

matx_status_t ref_transpose_c64_grb(
	matx_coo_f64_t A, 
	matx_coo_f64_t out)
{
	if (!A || !out)
		return MATX_ERR_INVALID_ARG;

	/* build A */
	if (A->handle_grb.valid <= 0)
	{
		coo_2_grb_c64(A);
	}
	if(!out->handle_grb.impl)
		return MATX_ERR_INVALID_ARG;
	if (!out->handle_grb.impl)
		create_empty_grb_c64(out);
	GrB_Info info = GrB_transpose((GrB_Matrix)(out->handle_grb.impl), NULL, NULL, (GrB_Matrix)(A->handle_grb.impl), NULL);
	if (info != GrB_SUCCESS)
	{
		MATX_ERROR("GrB_transpose error: %d", info);
		return MATX_ERR_INTERNAL;
	}
	grb_2_coo_c64(out);
	return MATX_OK;
}

matx_status_t ref_conj_trans_c64_grb(matx_coo_c64_t A,
	matx_coo_c64_t out)
{
	if (!A)
		return MATX_ERR_INVALID_ARG;

	/* build A */
	if (A->handle_grb.valid <= 0)
	{
		coo_2_grb_c64(A);
	}
	if(!out->handle_grb.impl)
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
	return MATX_OK;

}

matx_status_t ref_norm1_grb(
	matx_vec_f64_t A,
	matx_double out)
{
	if (A->handle_grb.valid <= 0)
	{
		vec_2_grb_c64(A);
	}

	// tmp = abs(x)
	GrB_apply((GrB_Vector)A->handle_grb.impl, NULL, NULL, GrB_ABS_FP64, (GrB_Vector)A->handle_grb.impl, NULL);
	// sum(tmp)
	GrB_reduce(&out, NULL, GrB_PLUS_MONOID_FP64, (GrB_Vector)A->handle_grb.impl, NULL);
	return MATX_OK;
}

matx_status_t ref_norm2_grb(
	matx_vec_f64_t A,
	matx_double* out)
{
	if (A->handle_grb.valid <= 0)
	{
		vec_2_grb_c64(A);
	}

	// tmp = x .* x
	GrB_eWiseMult((GrB_Vector)A->handle_grb.impl, NULL, NULL, GrB_TIMES_FP64, (GrB_Vector)A->handle_grb.impl, (GrB_Vector)A->handle_grb.impl, NULL);
	// sum
	double sumsq;
	GrB_reduce(&sumsq, NULL, GrB_PLUS_MONOID_FP64, (GrB_Vector)A->handle_grb.impl, NULL);

	*out = sqrt(sumsq);
	return MATX_OK;
}

matx_status_t ref_norminf_grb(
	matx_vec_f64_t A,
	matx_double* out)
{
	if (A->handle_grb.valid <= 0)
	{
		vec_2_grb_c64(A);
	}
	GrB_apply((GrB_Vector)A->handle_grb.impl, NULL, NULL, GrB_ABS_FP64, (GrB_Vector)A->handle_grb.impl, NULL);
	GrB_reduce(out, NULL, GrB_MAX_MONOID_FP64, (GrB_Vector)A->handle_grb.impl, NULL);
	return MATX_OK;
}

matx_sparse_backend_t matx_sparse_make_reference_grb(void) {
	GrB_Info info = GrB_init(GrB_NONBLOCKING);
	if (info != GrB_SUCCESS)
	{
		MATX_ERROR("GraphBLAS initialization failed with error code %d", info);
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
			.norm1_f64 = ref_norm1_grb,
			.norm2_f64 = ref_norm2_grb,
			.norminf_f64 = ref_norminf_grb
		}
	};
	return b;
}