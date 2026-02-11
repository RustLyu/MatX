#include "matx/matx_compute.h"

#include <string.h>

// Forward decls
matx_sparse_backend_t matx_sparse_make_reference_grb(void);
matx_sparse_backend_t matx_sparse_make_reference_mkl(void);

const char* matx_sparse_backend_name(matx_sparse_backend_kind_t k) {
	switch (k) {
	case MATX_BLAS_BACKEND_REFERENCE: return "REFERENCE";
	case MATX_BLAS_BACKEND_OPENBLAS: return "OPENBLAS";
	case MATX_BLAS_BACKEND_BLIS: return "BLIS";
	default: return "UNKNOWN";
	}
}

static matx_sparse_backend_t choose_default_backend(void) {
	// Build-time selection (simple & portable). Can be extended to runtime CPUID switching later.
	// If user wants strict control: set -DMATX_BLAS_BACKEND=OPENBLAS/BLIS/REFERENCE
#if defined(MATX_BLAS_BACKEND_REFERENCE_ONLY)
	return matx_blas_make_reference();
#else
  // If no external backend is wired in, fall back to reference.
	return matx_sparse_make_reference_mkl();
#endif
}

matx_sparse_backend_t matx_sparse_default(void) {
	return choose_default_backend();
}

matx_status_t matx_spmv_coo_c64(const matx_sparse_backend_t* backend,
	matx_complex_f64 alpha,
	matx_coo_c64_t* A,
	matx_vec_c64_t* x,
	matx_complex_f64 beta,
	matx_vec_c64_t* y)
{
	return backend->vt.spmv_c64(alpha, A, x, beta, y);
	return MATX_OK;
}

matx_status_t matx_spmm_coo_c64(const matx_sparse_backend_t* backend,
	matx_complex_f64 alpha,
	const matx_coo_c64_t* A,
	const matx_dense_c64_t* B,
	matx_complex_f64 beta,
	matx_dense_c64_t* C)
{
	return backend->vt.spmm_c64(alpha, A, B, beta, C);
}

matx_status_t matx_spmv_coo_f64(const matx_sparse_backend_t* backend,
	matx_double alpha,
	matx_coo_f64_t* A,
	matx_vec_f64_t* x,
	matx_double beta,
	matx_vec_f64_t* y)
{
	return backend->vt.spmv_f64(alpha, A, x, beta, y);
	//GrB_Info info = GrB_init(GrB_NONBLOCKING);
	//if (!A || !x || !y)
	//	return MATX_ERR_INVALID_ARG;

	//if (A->ncols != x->n || A->nrows != y->n)
	//	return MATX_ERR_INVALID_ARG;

	///* ---------------- build GraphBLAS matrix ---------------- */

	//if (A->handle_grb.valid <= 0)
	//{
	//	coo_2_grb_f64(A);
	//}
	///* ---------------- build vectors ---------------- */
	//if (x->handle_grb.valid <= 0)
	//{
	//	vec_2_grb_f64(x);
	//}

	//if (y->handle_grb.valid <= 0)
	//{
	//	vec_2_grb_f64(y);
	//}

	///* ---------------- gy = alpha*A*x + beta*y ---------------- */

	//GrB_Vector temp;
	//info = GrB_Vector_new(&temp, GrB_FP64, y->n);
	//// temp = A*x
	//info = GrB_mxv(temp, NULL, NULL, GxB_PLUS_TIMES_FP64, *(GrB_Matrix*)A->handle_grb.impl, *(GrB_Vector*)x->handle_grb.impl, NULL);
	//// temp = alpha*temp
	//info = GrB_apply(temp, NULL, NULL, GrB_TIMES_FP64, temp, &alpha, NULL);
	//// gy = beta*gy
	//info = GrB_apply(*(GrB_Vector*)y->handle_grb.impl, NULL, NULL, GrB_TIMES_FP64, *(GrB_Vector*)y->handle_grb.impl, &beta, NULL);
	//// gy = temp + gy
	//info = GrB_eWiseAdd(*(GrB_Vector*)y->handle_grb.impl, NULL, NULL, GrB_PLUS_FP64, temp, *(GrB_Vector*)y->handle_grb.impl, NULL);

	//GrB_Vector_free(&temp);
	//grb_2_vec_f64(y);
	//return MATX_OK;
}

matx_status_t matx_spmm_coo_f64(const matx_sparse_backend_t* backend,
	matx_double alpha,
	matx_coo_f64_t* A,
	matx_dense_f64_t* B,
	matx_double beta,
	matx_dense_f64_t* C)
{
	return backend->vt.spmm_f64(alpha, A, B, beta, C);
	//GrB_Info info = GrB_init(GrB_NONBLOCKING);
	//if (!A || !B || !C)
	//	return MATX_ERR_INVALID_ARG;

	///* build A */
	//if (A->handle_grb.valid <= 0)
	//{
	//	coo_2_grb_f64(A);
	//}
	//if (B->handle_grb.valid <= 0)
	//{
	//	dense_2_grb_f64(B);
	//}
	//if (C->handle_grb.valid <= 0)
	//{
	//	dense_2_grb_f64(C);
	//}

	///* C = alpha*A*B + beta*C */

	//// 1. temp = alpha * A * B
	//GrB_Matrix temp;
	//info = GrB_Matrix_new(&temp, GrB_FP64, C->rows, C->cols);
	//info = GrB_mxm(temp, NULL, NULL, GxB_PLUS_TIMES_FP64, *(GrB_Matrix*)A->handle_grb.impl, *(GrB_Matrix*)B->handle_grb.impl, NULL);
	//info = GrB_apply(temp, NULL, NULL, GrB_TIMES_FP64, temp, &alpha, NULL);
	////2. gC = beta * gC
	//info = GrB_apply(*(GrB_Matrix*)C->handle_grb.impl, NULL, NULL, GrB_TIMES_FP64, *(GrB_Matrix*)C->handle_grb.impl, &beta, NULL);
	////3. gC = temp + gC
	//info = GrB_eWiseAdd(*(GrB_Matrix*)C->handle_grb.impl, NULL, NULL, GrB_PLUS_FP64, temp, *(GrB_Matrix*)C->handle_grb.impl, NULL);
	//GrB_Matrix_free(&temp);
	//grb_2_dense_f64(C);
	//return MATX_OK;
}