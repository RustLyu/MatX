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

matx_status_t matx_vec_scal_f64(const matx_vec_backend_t* blas, matx_double alpha, matx_vec_f64_t x) {
	if (!blas || !x || !x->data) return MATX_ERR_INVALID_ARG;
	if (!blas->vt.dscal) return MATX_ERR_NOT_SUPPORTED;
	return blas->vt.dscal(x->n, alpha, x->data, x->stride);
}

matx_status_t matx_vec_scal_c64(const matx_vec_backend_t* blas, matx_complex_f64_t alpha, matx_vec_c64_t x) {
	if (!blas || !x || !x->data) return MATX_ERR_INVALID_ARG;
	if (!blas->vt.zscal) return MATX_ERR_NOT_SUPPORTED;
	return blas->vt.zscal(x->n, &alpha, x->data, x->stride);
}

matx_status_t matx_vec_copy_f64(const matx_vec_backend_t* blas, const matx_vec_f64_t x, matx_vec_f64_t y) {
	if (!blas || !x || !y || !x->data || !y->data) return MATX_ERR_INVALID_ARG;
	if (x->n != y->n) return MATX_ERR_INVALID_ARG;
	if (!blas->vt.dcopy) return MATX_ERR_NOT_SUPPORTED;
	return blas->vt.dcopy(x->n, x->data, x->stride, y->data, y->stride);
}

matx_status_t matx_vec_copy_c64(const matx_vec_backend_t* blas, const matx_vec_c64_t x, matx_vec_c64_t y) {
	if (!blas || !x || !y || !x->data || !y->data) return MATX_ERR_INVALID_ARG;
	if (x->n != y->n) return MATX_ERR_INVALID_ARG;
	if (!blas->vt.zcopy) return MATX_ERR_NOT_SUPPORTED;
	return blas->vt.zcopy(x->n, x->data, x->stride, y->data, y->stride);
}

matx_status_t matx_vec_swap_f64(const matx_vec_backend_t* blas, matx_vec_f64_t x, matx_vec_f64_t y) {
	if (!blas || !x || !y || !x->data || !y->data) return MATX_ERR_INVALID_ARG;
	if (x->n != y->n) return MATX_ERR_INVALID_ARG;
	if (!blas->vt.dswap) return MATX_ERR_NOT_SUPPORTED;
	return blas->vt.dswap(x->n, x->data, x->stride, y->data, y->stride);
}

matx_status_t matx_vec_swap_c64(const matx_vec_backend_t* blas, matx_vec_c64_t x, matx_vec_c64_t y) {
	if (!blas || !x || !y || !x->data || !y->data) return MATX_ERR_INVALID_ARG;
	if (x->n != y->n) return MATX_ERR_INVALID_ARG;
	if (!blas->vt.zswap) return MATX_ERR_NOT_SUPPORTED;
	return blas->vt.zswap(x->n, x->data, x->stride, y->data, y->stride);
}

matx_status_t matx_vec_dot_f64(const matx_vec_backend_t* blas, const matx_vec_f64_t x, const matx_vec_f64_t y, matx_double* result) {
	if (!blas || !x || !y || !x->data || !y->data || !result) return MATX_ERR_INVALID_ARG;
	if (x->n != y->n) return MATX_ERR_INVALID_ARG;
	if (!blas->vt.ddot) return MATX_ERR_NOT_SUPPORTED;
	return blas->vt.ddot(x->n, x->data, x->stride, y->data, y->stride, result);
}

matx_status_t matx_vec_dotu_c64(const matx_vec_backend_t* blas, const matx_vec_c64_t x, const matx_vec_c64_t y, matx_complex_f64_t* result) {
	if (!blas || !x || !y || !x->data || !y->data || !result) return MATX_ERR_INVALID_ARG;
	if (x->n != y->n) return MATX_ERR_INVALID_ARG;
	if (!blas->vt.zdotu) return MATX_ERR_NOT_SUPPORTED;
	return blas->vt.zdotu(x->n, x->data, x->stride, y->data, y->stride, result);
}

matx_status_t matx_vec_dotc_c64(const matx_vec_backend_t* blas, const matx_vec_c64_t x, const matx_vec_c64_t y, matx_complex_f64_t* result) {
	if (!blas || !x || !y || !x->data || !y->data || !result) return MATX_ERR_INVALID_ARG;
	if (x->n != y->n) return MATX_ERR_INVALID_ARG;
	if (!blas->vt.zdotc) return MATX_ERR_NOT_SUPPORTED;
	return blas->vt.zdotc(x->n, x->data, x->stride, y->data, y->stride, result);
}

matx_status_t matx_vec_nrm2_f64(const matx_vec_backend_t* blas, const matx_vec_f64_t x, matx_double* result) {
	if (!blas || !x || !x->data || !result) return MATX_ERR_INVALID_ARG;
	if (!blas->vt.dnrm2) return MATX_ERR_NOT_SUPPORTED;
	return blas->vt.dnrm2(x->n, x->data, x->stride, result);
}

matx_status_t matx_vec_nrm2_c64(const matx_vec_backend_t* blas, const matx_vec_c64_t x, matx_double* result) {
	if (!blas || !x || !x->data || !result) return MATX_ERR_INVALID_ARG;
	if (!blas->vt.dznrm2) return MATX_ERR_NOT_SUPPORTED;
	return blas->vt.dznrm2(x->n, x->data, x->stride, result);
}

matx_status_t matx_vec_asum_f64(const matx_vec_backend_t* blas, const matx_vec_f64_t x, matx_double* result) {
	if (!blas || !x || !x->data || !result) return MATX_ERR_INVALID_ARG;
	if (!blas->vt.dasum) return MATX_ERR_NOT_SUPPORTED;
	return blas->vt.dasum(x->n, x->data, x->stride, result);
}

matx_status_t matx_vec_asum_c64(const matx_vec_backend_t* blas, const matx_vec_c64_t x, matx_double* result) {
	if (!blas || !x || !x->data || !result) return MATX_ERR_INVALID_ARG;
	if (!blas->vt.dzasum) return MATX_ERR_NOT_SUPPORTED;
	return blas->vt.dzasum(x->n, x->data, x->stride, result);
}

matx_status_t matx_vec_iamax_f64(const matx_vec_backend_t* blas, const matx_vec_f64_t x, matx_int64_t* result) {
	if (!blas || !x || !x->data || !result) return MATX_ERR_INVALID_ARG;
	if (!blas->vt.idamax) return MATX_ERR_NOT_SUPPORTED;
	return blas->vt.idamax(x->n, x->data, x->stride, result);
}

matx_status_t matx_vec_iamax_c64(const matx_vec_backend_t* blas, const matx_vec_c64_t x, matx_int64_t* result) {
	if (!blas || !x || !x->data || !result) return MATX_ERR_INVALID_ARG;
	if (!blas->vt.izamax) return MATX_ERR_NOT_SUPPORTED;
	return blas->vt.izamax(x->n, x->data, x->stride, result);
}

matx_status_t matx_vec_axpy_f64(const matx_vec_backend_t* blas,
	matx_double alpha,
	const matx_vec_f64_t x,
	matx_vec_f64_t y) {
	if (!x || !y || !x->data || !y->data)
		return MATX_ERR_INVALID_ARG;
	if (x->n != y->n)
		return MATX_ERR_INVALID_ARG;
	if (!blas || !blas->vt.daxpy) return MATX_ERR_NOT_SUPPORTED;
	return blas->vt.daxpy(x->n, alpha, x->data, x->stride, y->data, y->stride);
}

matx_status_t matx_vec_axpy_c64(const matx_vec_backend_t* blas,
	matx_complex_f64_t alpha,
	const matx_vec_c64_t x,
	matx_vec_c64_t y) {
	if (!x || !y || !x->data || !y->data)
		return MATX_ERR_INVALID_ARG;
	if (x->n != y->n)
		return MATX_ERR_INVALID_ARG;
	if (!blas || !blas->vt.zaxpy) return MATX_ERR_NOT_SUPPORTED;
	return blas->vt.zaxpy(x->n, &alpha, x->data, x->stride, y->data, y->stride);
}

// ---- Vector norm wrappers ----

matx_status_t matx_vec_norm1_f64(const matx_vec_backend_t* backend, matx_vec_f64_t A, matx_double* out) {
	if (!backend || !A || !out) return MATX_ERR_INVALID_ARG;
	if (!backend->vt.norm1_f64) return MATX_ERR_NOT_SUPPORTED;
	return backend->vt.norm1_f64(A, out);
}

matx_status_t matx_vec_norm1_c64(const matx_vec_backend_t* backend, matx_vec_c64_t A, matx_double* out) {
	if (!backend || !A || !out) return MATX_ERR_INVALID_ARG;
	if (!backend->vt.norm1_c64) return MATX_ERR_NOT_SUPPORTED;
	return backend->vt.norm1_c64(A, out);
}

matx_status_t matx_vec_norm2_f64(const matx_vec_backend_t* backend, matx_vec_f64_t A, matx_double* out) {
	if (!backend || !A || !out) return MATX_ERR_INVALID_ARG;
	if (!backend->vt.norm2_f64) return MATX_ERR_NOT_SUPPORTED;
	return backend->vt.norm2_f64(A, out);
}

matx_status_t matx_vec_norm2_c64(const matx_vec_backend_t* backend, matx_vec_c64_t A, matx_double* out) {
	if (!backend || !A || !out) return MATX_ERR_INVALID_ARG;
	if (!backend->vt.norm2_c64) return MATX_ERR_NOT_SUPPORTED;
	return backend->vt.norm2_c64(A, out);
}

matx_status_t matx_vec_norminf_f64(const matx_vec_backend_t* backend, matx_vec_f64_t A, matx_double* out) {
	if (!backend || !A || !out) return MATX_ERR_INVALID_ARG;
	if (!backend->vt.norminf_f64) return MATX_ERR_NOT_SUPPORTED;
	return backend->vt.norminf_f64(A, out);
}

matx_status_t matx_vec_norminf_c64(const matx_vec_backend_t* backend, matx_vec_c64_t A, matx_double* out) {
	if (!backend || !A || !out) return MATX_ERR_INVALID_ARG;
	if (!backend->vt.norminf_c64) return MATX_ERR_NOT_SUPPORTED;
	return backend->vt.norminf_c64(A, out);
}