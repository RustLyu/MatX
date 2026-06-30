#include "matx/matx_vec_compute.h"
#include "matx/matx_types_internal.h"

// Forward decls
matx_vec_backend_t matx_vec_blas_make_reference(void);

const char* matx_vec_backend_name(matx_vec_backend_kind_t k) {
	switch (k) {
	case MATX_VEC_BACKEND_REFERENCE: return "REFERENCE";
	case MATX_VEC_BACKEND_OPENBLAS: return "OPENBLAS";
	case MATX_VEC_BACKEND_BLIS: return "BLIS";
	default: return "UNKNOWN";
	}
}

static matx_vec_backend_t choose_default_backend(void) {
	return matx_vec_blas_make_reference();
}

matx_vec_backend_t matx_vec_default(void) {
	return choose_default_backend();
}

// ---- Level 1 wrappers ----

matx_status_t matx_vec_scal_d_i8(const matx_vec_backend_t* blas, matx_double alpha, matx_vec_d_i8_t x) {
	if (!blas || !x || !x->data) return MATX_ERR_INVALID_ARG;
	if (!blas->vt.dscal) return MATX_ERR_NOT_SUPPORTED;
	return blas->vt.dscal(x->n, alpha, x->data, x->stride);
}

matx_status_t matx_vec_scal_z_i8(const matx_vec_backend_t* blas, matx_complex_d_i8_t alpha, matx_vec_z_i8_t x) {
	if (!blas || !x || !x->data) return MATX_ERR_INVALID_ARG;
	if (!blas->vt.zscal) return MATX_ERR_NOT_SUPPORTED;
	return blas->vt.zscal(x->n, &alpha, x->data, x->stride);
}

matx_status_t matx_vec_copy_d_i8(const matx_vec_backend_t* blas, const matx_vec_d_i8_t x, matx_vec_d_i8_t y) {
	if (!blas || !x || !y || !x->data || !y->data) return MATX_ERR_INVALID_ARG;
	if (x->n != y->n) return MATX_ERR_INVALID_ARG;
	if (!blas->vt.dcopy) return MATX_ERR_NOT_SUPPORTED;
	return blas->vt.dcopy(x->n, x->data, x->stride, y->data, y->stride);
}

matx_status_t matx_vec_copy_z_i8(const matx_vec_backend_t* blas, const matx_vec_z_i8_t x, matx_vec_z_i8_t y) {
	if (!blas || !x || !y || !x->data || !y->data) return MATX_ERR_INVALID_ARG;
	if (x->n != y->n) return MATX_ERR_INVALID_ARG;
	if (!blas->vt.zcopy) return MATX_ERR_NOT_SUPPORTED;
	return blas->vt.zcopy(x->n, x->data, x->stride, y->data, y->stride);
}

matx_status_t matx_vec_swap_d_i8(const matx_vec_backend_t* blas, matx_vec_d_i8_t x, matx_vec_d_i8_t y) {
	if (!blas || !x || !y || !x->data || !y->data) return MATX_ERR_INVALID_ARG;
	if (x->n != y->n) return MATX_ERR_INVALID_ARG;
	if (!blas->vt.dswap) return MATX_ERR_NOT_SUPPORTED;
	return blas->vt.dswap(x->n, x->data, x->stride, y->data, y->stride);
}

matx_status_t matx_vec_swap_z_i8(const matx_vec_backend_t* blas, matx_vec_z_i8_t x, matx_vec_z_i8_t y) {
	if (!blas || !x || !y || !x->data || !y->data) return MATX_ERR_INVALID_ARG;
	if (x->n != y->n) return MATX_ERR_INVALID_ARG;
	if (!blas->vt.zswap) return MATX_ERR_NOT_SUPPORTED;
	return blas->vt.zswap(x->n, x->data, x->stride, y->data, y->stride);
}

matx_status_t matx_vec_dot_d_i8(const matx_vec_backend_t* blas, const matx_vec_d_i8_t x, const matx_vec_d_i8_t y, matx_double* result) {
	if (!blas || !x || !y || !x->data || !y->data || !result) return MATX_ERR_INVALID_ARG;
	if (x->n != y->n) return MATX_ERR_INVALID_ARG;
	if (!blas->vt.ddot) return MATX_ERR_NOT_SUPPORTED;
	return blas->vt.ddot(x->n, x->data, x->stride, y->data, y->stride, result);
}

matx_status_t matx_vec_dotu_z_i8(const matx_vec_backend_t* blas, const matx_vec_z_i8_t x, const matx_vec_z_i8_t y, matx_complex_d_i8_t* result) {
	if (!blas || !x || !y || !x->data || !y->data || !result) return MATX_ERR_INVALID_ARG;
	if (x->n != y->n) return MATX_ERR_INVALID_ARG;
	if (!blas->vt.zdotu) return MATX_ERR_NOT_SUPPORTED;
	return blas->vt.zdotu(x->n, x->data, x->stride, y->data, y->stride, result);
}

matx_status_t matx_vec_dotc_z_i8(const matx_vec_backend_t* blas, const matx_vec_z_i8_t x, const matx_vec_z_i8_t y, matx_complex_d_i8_t* result) {
	if (!blas || !x || !y || !x->data || !y->data || !result) return MATX_ERR_INVALID_ARG;
	if (x->n != y->n) return MATX_ERR_INVALID_ARG;
	if (!blas->vt.zdotc) return MATX_ERR_NOT_SUPPORTED;
	return blas->vt.zdotc(x->n, x->data, x->stride, y->data, y->stride, result);
}

matx_status_t matx_vec_nrm2_d_i8(const matx_vec_backend_t* blas, const matx_vec_d_i8_t x, matx_double* result) {
	if (!blas || !x || !x->data || !result) return MATX_ERR_INVALID_ARG;
	if (!blas->vt.dnrm2) return MATX_ERR_NOT_SUPPORTED;
	return blas->vt.dnrm2(x->n, x->data, x->stride, result);
}

matx_status_t matx_vec_nrm2_z_i8(const matx_vec_backend_t* blas, const matx_vec_z_i8_t x, matx_double* result) {
	if (!blas || !x || !x->data || !result) return MATX_ERR_INVALID_ARG;
	if (!blas->vt.dznrm2) return MATX_ERR_NOT_SUPPORTED;
	return blas->vt.dznrm2(x->n, x->data, x->stride, result);
}

matx_status_t matx_vec_asum_d_i8(const matx_vec_backend_t* blas, const matx_vec_d_i8_t x, matx_double* result) {
	if (!blas || !x || !x->data || !result) return MATX_ERR_INVALID_ARG;
	if (!blas->vt.dasum) return MATX_ERR_NOT_SUPPORTED;
	return blas->vt.dasum(x->n, x->data, x->stride, result);
}

matx_status_t matx_vec_asum_z_i8(const matx_vec_backend_t* blas, const matx_vec_z_i8_t x, matx_double* result) {
	if (!blas || !x || !x->data || !result) return MATX_ERR_INVALID_ARG;
	if (!blas->vt.dzasum) return MATX_ERR_NOT_SUPPORTED;
	return blas->vt.dzasum(x->n, x->data, x->stride, result);
}

matx_status_t matx_vec_iamax_d_i8(const matx_vec_backend_t* blas, const matx_vec_d_i8_t x, matx_int64_t* result) {
	if (!blas || !x || !x->data || !result) return MATX_ERR_INVALID_ARG;
	if (!blas->vt.idamax) return MATX_ERR_NOT_SUPPORTED;
	return blas->vt.idamax(x->n, x->data, x->stride, result);
}

matx_status_t matx_vec_iamax_z_i8(const matx_vec_backend_t* blas, const matx_vec_z_i8_t x, matx_int64_t* result) {
	if (!blas || !x || !x->data || !result) return MATX_ERR_INVALID_ARG;
	if (!blas->vt.izamax) return MATX_ERR_NOT_SUPPORTED;
	return blas->vt.izamax(x->n, x->data, x->stride, result);
}

matx_status_t matx_vec_axpy_d_i8(const matx_vec_backend_t* blas,
	matx_double alpha,
	const matx_vec_d_i8_t x,
	matx_vec_d_i8_t y) {
	if (!x || !y || !x->data || !y->data)
		return MATX_ERR_INVALID_ARG;
	if (x->n != y->n)
		return MATX_ERR_INVALID_ARG;
	if (!blas || !blas->vt.daxpy) return MATX_ERR_NOT_SUPPORTED;
	return blas->vt.daxpy(x->n, alpha, x->data, x->stride, y->data, y->stride);
}

matx_status_t matx_vec_axpy_z_i8(const matx_vec_backend_t* blas,
	matx_complex_d_i8_t alpha,
	const matx_vec_z_i8_t x,
	matx_vec_z_i8_t y) {
	if (!x || !y || !x->data || !y->data)
		return MATX_ERR_INVALID_ARG;
	if (x->n != y->n)
		return MATX_ERR_INVALID_ARG;
	if (!blas || !blas->vt.zaxpy) return MATX_ERR_NOT_SUPPORTED;
	return blas->vt.zaxpy(x->n, &alpha, x->data, x->stride, y->data, y->stride);
}

// ---- Vector norm wrappers ----

matx_status_t matx_vec_norm1_d_i8(const matx_vec_backend_t* backend, matx_vec_d_i8_t A, matx_double* out) {
	if (!backend || !A || !out) return MATX_ERR_INVALID_ARG;
	if (!backend->vt.norm1_d_i8) return MATX_ERR_NOT_SUPPORTED;
	return backend->vt.norm1_d_i8(A, out);
}

matx_status_t matx_vec_norm1_z_i8(const matx_vec_backend_t* backend, matx_vec_z_i8_t A, matx_double* out) {
	if (!backend || !A || !out) return MATX_ERR_INVALID_ARG;
	if (!backend->vt.norm1_z_i8) return MATX_ERR_NOT_SUPPORTED;
	return backend->vt.norm1_z_i8(A, out);
}

matx_status_t matx_vec_norm2_d_i8(const matx_vec_backend_t* backend, matx_vec_d_i8_t A, matx_double* out) {
	if (!backend || !A || !out) return MATX_ERR_INVALID_ARG;
	if (!backend->vt.norm2_d_i8) return MATX_ERR_NOT_SUPPORTED;
	return backend->vt.norm2_d_i8(A, out);
}

matx_status_t matx_vec_norm2_z_i8(const matx_vec_backend_t* backend, matx_vec_z_i8_t A, matx_double* out) {
	if (!backend || !A || !out) return MATX_ERR_INVALID_ARG;
	if (!backend->vt.norm2_z_i8) return MATX_ERR_NOT_SUPPORTED;
	return backend->vt.norm2_z_i8(A, out);
}

matx_status_t matx_vec_norminf_d_i8(const matx_vec_backend_t* backend, matx_vec_d_i8_t A, matx_double* out) {
	if (!backend || !A || !out) return MATX_ERR_INVALID_ARG;
	if (!backend->vt.norminf_d_i8) return MATX_ERR_NOT_SUPPORTED;
	return backend->vt.norminf_d_i8(A, out);
}

matx_status_t matx_vec_norminf_z_i8(const matx_vec_backend_t* backend, matx_vec_z_i8_t A, matx_double* out) {
	if (!backend || !A || !out) return MATX_ERR_INVALID_ARG;
	if (!backend->vt.norminf_z_i8) return MATX_ERR_NOT_SUPPORTED;
	return backend->vt.norminf_z_i8(A, out);
}