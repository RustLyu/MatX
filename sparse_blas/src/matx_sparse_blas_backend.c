#include "matx/matx_sparse_compute.h"
#include "matx/matx_types_internal.h"

// Forward decls
matx_sparse_backend_t matx_sparse_make_reference_grb(void);
matx_sparse_backend_t matx_sparse_make_reference_mkl(void);
matx_sparse_backend_t matx_sparse_make_reference_aocl(void);

const char* matx_sparse_backend_name(matx_sparse_backend_kind_t k) {
	switch (k) {
	case MATX_SPARSE_BACKEND_REFERENCE: return "REFERENCE";
	case MATX_SPARSE_BACKEND_GRAPHBLAS: return "GRAPHBLAS";
	case MATX_SPARSE_BACKEND_MKL: return "MKL";
	default: return "UNKNOWN";
	}
}

static matx_sparse_backend_t choose_default_backend(void) {
	return matx_sparse_make_reference_grb();
}

matx_sparse_backend_t matx_sparse_default(void) {
	return choose_default_backend();
}

MATX_API matx_sparse_backend_t matx_sparse_by_type(matx_sparse_backend_kind_t k)
{
	switch (k) {
	case MATX_SPARSE_BACKEND_AOCL_CPARSE: return matx_sparse_make_reference_aocl();
	default: return matx_sparse_make_reference_grb();
	}
}

matx_status_t matx_spmv_coo_c64(const matx_sparse_backend_t* backend,
	matx_complex_f64_t alpha,
        matx_coo_c64_t A,
        matx_vec_c64_t x,
	matx_complex_f64_t beta,
	matx_vec_c64_t y)
{
	if (!backend || !A || !x || !y) return MATX_ERR_INVALID_ARG;
	if (!backend->vt.spmv_c64) return MATX_ERR_NOT_SUPPORTED;
	if (!A->values || !x->data || !y->data) return MATX_ERR_INVALID_ARG;
	return backend->vt.spmv_c64(alpha, A, x, beta, y);
}

matx_status_t matx_spmm_coo_c64(const matx_sparse_backend_t* backend,
	matx_complex_f64_t alpha,
	matx_coo_c64_t A,
	matx_dense_c64_t B,
	matx_complex_f64_t beta,
	matx_dense_c64_t C)
{
	if (!backend || !A || !B || !C) return MATX_ERR_INVALID_ARG;
	if (!backend->vt.spmm_c64) return MATX_ERR_NOT_SUPPORTED;
	if (!A->values || !B->data || !C->data) return MATX_ERR_INVALID_ARG;
	return backend->vt.spmm_c64(alpha, A, B, beta, C);
}

matx_status_t matx_spmv_coo_f64(const matx_sparse_backend_t* backend,
	matx_double alpha,
        matx_coo_f64_t A,
        matx_vec_f64_t x,
	matx_double beta,
	matx_vec_f64_t y)
{
	if (!backend || !A || !x || !y) return MATX_ERR_INVALID_ARG;
	if (!backend->vt.spmv_f64) return MATX_ERR_NOT_SUPPORTED;
	if (!A->values || !x->data || !y->data) return MATX_ERR_INVALID_ARG;
	return backend->vt.spmv_f64(alpha, A, x, beta, y);
}

matx_status_t matx_spmm_coo_f64(const matx_sparse_backend_t* backend,
	matx_double alpha,
	matx_coo_f64_t A,
	matx_dense_f64_t B,
	matx_double beta,
	matx_dense_f64_t C)
{
	if (!backend || !A || !B || !C) return MATX_ERR_INVALID_ARG;
	if (!backend->vt.spmm_f64) return MATX_ERR_NOT_SUPPORTED;
	if (!A->values || !B->data || !C->data) return MATX_ERR_INVALID_ARG;
	return backend->vt.spmm_f64(alpha, A, B, beta, C);
}

matx_status_t matx_dsp2md_coo_f64(const matx_sparse_backend_t* backend,
	matx_double alpha,
	matx_coo_f64_t A,
	matx_coo_f64_t B,
	matx_double beta,
	matx_dense_f64_t C)
{
	if (!backend || !A || !B || !C) return MATX_ERR_INVALID_ARG;
	if (!backend->vt.dsp2md_f64) return MATX_ERR_NOT_SUPPORTED;
	if (!A->values || !B->values || !C->data) return MATX_ERR_INVALID_ARG;
	return backend->vt.dsp2md_f64(alpha, A, B, beta, C);
}

matx_status_t matx_zsp2md_coo_c64(const matx_sparse_backend_t* backend,
	matx_complex_f64_t alpha,
	matx_coo_c64_t A,
	matx_coo_c64_t B,
	matx_complex_f64_t beta,
	matx_dense_c64_t C)
{
	if (!backend || !A || !B || !C) return MATX_ERR_INVALID_ARG;
	if (!backend->vt.zsp2md_c64) return MATX_ERR_NOT_SUPPORTED;
	if (!A->values || !B->values || !C->data) return MATX_ERR_INVALID_ARG;
	return backend->vt.zsp2md_c64(alpha, A, B, beta, C);
}

matx_status_t matx_transpose_coo_f64(const matx_sparse_backend_t* backend,
	matx_coo_f64_t A,
	matx_coo_f64_t out)
{
	if (!backend || !A || !out) return MATX_ERR_INVALID_ARG;
	if (!backend->vt.transpose_f64) return MATX_ERR_NOT_SUPPORTED;
	if (!A->values || !out->values) return MATX_ERR_INVALID_ARG;
	return backend->vt.transpose_f64(A, out);
}

matx_status_t matx_transpose_coo_c64(const matx_sparse_backend_t* backend,
	matx_coo_c64_t A,
	matx_coo_c64_t out)
{
	if (!backend || !A || !out) return MATX_ERR_INVALID_ARG;
	if (!backend->vt.transpose_c64) return MATX_ERR_NOT_SUPPORTED;
	if (!A->values || !out->values) return MATX_ERR_INVALID_ARG;
	return backend->vt.transpose_c64(A, out);
}

matx_status_t matx_conj_coo_c64(const matx_sparse_backend_t* backend, matx_coo_c64_t A, matx_coo_c64_t out)
{
	if (!backend || !A || !out) return MATX_ERR_INVALID_ARG;
	if (!backend->vt.conj_trans_c64) return MATX_ERR_NOT_SUPPORTED;
	if (!A->values || !out->values) return MATX_ERR_INVALID_ARG;
	return backend->vt.conj_trans_c64(A, out);
}

matx_status_t matx_finalize(const matx_sparse_backend_t* backend)
{
	if (!backend) return MATX_ERR_INVALID_ARG;
	if (!backend->vt.finalize) return MATX_ERR_NOT_SUPPORTED;
	return backend->vt.finalize();
}

matx_status_t matx_norm1_mat_coo_f64(const matx_sparse_backend_t* backend, matx_coo_f64_t A, matx_double* out) {
	if (!backend || !A || !out) return MATX_ERR_INVALID_ARG;
	if (!backend->vt.norm1_mat_f64) return MATX_ERR_NOT_SUPPORTED;
	return backend->vt.norm1_mat_f64(A, out);
}

matx_status_t matx_norminf_mat_coo_f64(const matx_sparse_backend_t* backend, matx_coo_f64_t A, matx_double* out) {
	if (!backend || !A || !out) return MATX_ERR_INVALID_ARG;
	if (!backend->vt.norminf_mat_f64) return MATX_ERR_NOT_SUPPORTED;
	return backend->vt.norminf_mat_f64(A, out);
}

matx_status_t matx_normfro_mat_coo_f64(const matx_sparse_backend_t* backend, matx_coo_f64_t A, matx_double* out) {
	if (!backend || !A || !out) return MATX_ERR_INVALID_ARG;
	if (!backend->vt.normfro_mat_f64) return MATX_ERR_NOT_SUPPORTED;
	return backend->vt.normfro_mat_f64(A, out);
}

matx_status_t matx_norm1_mat_coo_c64(const matx_sparse_backend_t* backend, matx_coo_c64_t A, matx_double* out) {
	if (!backend || !A || !out) return MATX_ERR_INVALID_ARG;
	if (!backend->vt.norm1_mat_c64) return MATX_ERR_NOT_SUPPORTED;
	return backend->vt.norm1_mat_c64(A, out);
}

matx_status_t matx_norminf_mat_coo_c64(const matx_sparse_backend_t* backend, matx_coo_c64_t A, matx_double* out) {
	if (!backend || !A || !out) return MATX_ERR_INVALID_ARG;
	if (!backend->vt.norminf_mat_c64) return MATX_ERR_NOT_SUPPORTED;
	return backend->vt.norminf_mat_c64(A, out);
}

matx_status_t matx_normfro_mat_coo_c64(const matx_sparse_backend_t* backend, matx_coo_c64_t A, matx_double* out) {
	if (!backend || !A || !out) return MATX_ERR_INVALID_ARG;
	if (!backend->vt.normfro_mat_c64) return MATX_ERR_NOT_SUPPORTED;
	return backend->vt.normfro_mat_c64(A, out);
}

matx_status_t matx_spadd_coo_f64(const matx_sparse_backend_t* backend,
	matx_double alpha, matx_coo_f64_t A, matx_double beta, matx_coo_f64_t B, matx_coo_f64_t out) {
	if (!backend || !A || !B || !out) return MATX_ERR_INVALID_ARG;
	if (!backend->vt.spadd_f64) return MATX_ERR_NOT_SUPPORTED;
	return backend->vt.spadd_f64(alpha, A, beta, B, out);
}

matx_status_t matx_spadd_coo_c64(const matx_sparse_backend_t* backend,
	matx_complex_f64_t alpha, matx_coo_c64_t A, matx_complex_f64_t beta, matx_coo_c64_t B, matx_coo_c64_t out) {
	if (!backend || !A || !B || !out) return MATX_ERR_INVALID_ARG;
	if (!backend->vt.spadd_c64) return MATX_ERR_NOT_SUPPORTED;
	return backend->vt.spadd_c64(alpha, A, beta, B, out);
}
