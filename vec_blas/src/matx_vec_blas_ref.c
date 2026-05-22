#if __linux__
	#define _XOPEN_SOURCE 600
#endif

#include "matx/matx_vec_compute.h"
#include "matx/matx_log.h"
#include "matx/matx_types_internal.h"

#include <string.h>
#include <math.h>

#if MATX_ENABLE_OPENBLAS
    #include "cblas.h"
#elif MATX_ENABLE_BLIS
	#include "blis.h"
#endif

// ---- Level 1 implementations ----

static matx_status_t ref_dscal(matx_int64_t n, matx_double alpha, matx_double* x, matx_int64_t incx) {
	if (!x) return MATX_ERR_INVALID_ARG;
#if MATX_ENABLE_OPENBLAS || MATX_ENABLE_BLIS
	cblas_dscal(n, alpha, x, incx);
#else
	for (matx_int64_t i = 0; i < n; ++i)
		x[i * incx] *= alpha;
#endif
	return MATX_OK;
}

static matx_status_t ref_zscal(matx_int64_t n, const void* alpha, void* x, matx_int64_t incx) {
	if (!x || !alpha) return MATX_ERR_INVALID_ARG;
#if MATX_ENABLE_OPENBLAS || MATX_ENABLE_BLIS
	cblas_zscal(n, alpha, x, incx);
#else
	const matx_complex_f64_t* a = (const matx_complex_f64_t*)alpha;
	matx_complex_f64_t* xd = (matx_complex_f64_t*)x;
	for (matx_int64_t i = 0; i < n; ++i) {
		matx_complex_f64_t v = xd[i * incx];
		xd[i * incx].real = a->real * v.real - a->imag * v.imag;
		xd[i * incx].imag = a->real * v.imag + a->imag * v.real;
	}
#endif
	return MATX_OK;
}

static matx_status_t ref_dcopy(matx_int64_t n, const matx_double* x, matx_int64_t incx, matx_double* y, matx_int64_t incy) {
	if (!x || !y) return MATX_ERR_INVALID_ARG;
#if MATX_ENABLE_OPENBLAS || MATX_ENABLE_BLIS
	cblas_dcopy(n, x, incx, y, incy);
#else
	for (matx_int64_t i = 0; i < n; ++i)
		y[i * incy] = x[i * incx];
#endif
	return MATX_OK;
}

static matx_status_t ref_zcopy(matx_int64_t n, const void* x, matx_int64_t incx, void* y, matx_int64_t incy) {
	if (!x || !y) return MATX_ERR_INVALID_ARG;
#if MATX_ENABLE_OPENBLAS || MATX_ENABLE_BLIS
	cblas_zcopy(n, x, incx, y, incy);
#else
	const matx_complex_f64_t* xd = (const matx_complex_f64_t*)x;
	matx_complex_f64_t* yd = (matx_complex_f64_t*)y;
	for (matx_int64_t i = 0; i < n; ++i)
		yd[i * incy] = xd[i * incx];
#endif
	return MATX_OK;
}

static matx_status_t ref_dswap(matx_int64_t n, matx_double* x, matx_int64_t incx, matx_double* y, matx_int64_t incy) {
	if (!x || !y) return MATX_ERR_INVALID_ARG;
#if MATX_ENABLE_OPENBLAS || MATX_ENABLE_BLIS
	cblas_dswap(n, x, incx, y, incy);
#else
	for (matx_int64_t i = 0; i < n; ++i) {
		matx_double t = x[i * incx];
		x[i * incx] = y[i * incy];
		y[i * incy] = t;
	}
#endif
	return MATX_OK;
}

static matx_status_t ref_zswap(matx_int64_t n, void* x, matx_int64_t incx, void* y, matx_int64_t incy) {
	if (!x || !y) return MATX_ERR_INVALID_ARG;
#if MATX_ENABLE_OPENBLAS || MATX_ENABLE_BLIS
	cblas_zswap(n, x, incx, y, incy);
#else
	matx_complex_f64_t* xd = (matx_complex_f64_t*)x;
	matx_complex_f64_t* yd = (matx_complex_f64_t*)y;
	for (matx_int64_t i = 0; i < n; ++i) {
		matx_complex_f64_t t = xd[i * incx];
		xd[i * incx] = yd[i * incy];
		yd[i * incy] = t;
	}
#endif
	return MATX_OK;
}

static matx_status_t ref_ddot(matx_int64_t n, const matx_double* x, matx_int64_t incx,
	const matx_double* y, matx_int64_t incy, matx_double* result) {
	if (!x || !y || !result) return MATX_ERR_INVALID_ARG;
#if MATX_ENABLE_OPENBLAS || MATX_ENABLE_BLIS
	*result = cblas_ddot(n, x, incx, y, incy);
#else
	matx_double sum = 0.0;
	for (matx_int64_t i = 0; i < n; ++i)
		sum += x[i * incx] * y[i * incy];
	*result = sum;
#endif
	return MATX_OK;
}

static matx_status_t ref_zdotu(matx_int64_t n, const void* x, matx_int64_t incx,
	const void* y, matx_int64_t incy, void* result) {
	if (!x || !y || !result) return MATX_ERR_INVALID_ARG;
#if MATX_ENABLE_OPENBLAS || MATX_ENABLE_BLIS
	cblas_zdotu_sub(n, x, incx, y, incy, result);
#else
	const matx_complex_f64_t* xd = (const matx_complex_f64_t*)x;
	const matx_complex_f64_t* yd = (const matx_complex_f64_t*)y;
	matx_complex_f64_t* r = (matx_complex_f64_t*)result;
	r->real = 0.0; r->imag = 0.0;
	for (matx_int64_t i = 0; i < n; ++i) {
		r->real += xd[i * incx].real * yd[i * incy].real - xd[i * incx].imag * yd[i * incy].imag;
		r->imag += xd[i * incx].real * yd[i * incy].imag + xd[i * incx].imag * yd[i * incy].real;
	}
#endif
	return MATX_OK;
}

static matx_status_t ref_zdotc(matx_int64_t n, const void* x, matx_int64_t incx,
	const void* y, matx_int64_t incy, void* result) {
	if (!x || !y || !result) return MATX_ERR_INVALID_ARG;
#if MATX_ENABLE_OPENBLAS || MATX_ENABLE_BLIS
	cblas_zdotc_sub(n, x, incx, y, incy, result);
#else
	const matx_complex_f64_t* xd = (const matx_complex_f64_t*)x;
	const matx_complex_f64_t* yd = (const matx_complex_f64_t*)y;
	matx_complex_f64_t* r = (matx_complex_f64_t*)result;
	r->real = 0.0; r->imag = 0.0;
	for (matx_int64_t i = 0; i < n; ++i) {
		r->real += xd[i * incx].real * yd[i * incy].real + xd[i * incx].imag * yd[i * incy].imag;
		r->imag += -xd[i * incx].imag * yd[i * incy].real + xd[i * incx].real * yd[i * incy].imag;
	}
#endif
	return MATX_OK;
}

static matx_status_t ref_dnrm2(matx_int64_t n, const matx_double* x, matx_int64_t incx, matx_double* result) {
	if (!x || !result) return MATX_ERR_INVALID_ARG;
#if MATX_ENABLE_OPENBLAS || MATX_ENABLE_BLIS
	*result = cblas_dnrm2(n, x, incx);
#else
	matx_double sum = 0.0;
	for (matx_int64_t i = 0; i < n; ++i)
		sum += x[i * incx] * x[i * incx];
	*result = sqrt(sum);
#endif
	return MATX_OK;
}

static matx_status_t ref_dznrm2(matx_int64_t n, const void* x, matx_int64_t incx, matx_double* result) {
	if (!x || !result) return MATX_ERR_INVALID_ARG;
#if MATX_ENABLE_OPENBLAS || MATX_ENABLE_BLIS
	*result = cblas_dznrm2(n, x, incx);
#else
	const matx_complex_f64_t* xd = (const matx_complex_f64_t*)x;
	matx_double sum = 0.0;
	for (matx_int64_t i = 0; i < n; ++i) {
		matx_double re = xd[i * incx].real;
		matx_double im = xd[i * incx].imag;
		sum += re * re + im * im;
	}
	*result = sqrt(sum);
#endif
	return MATX_OK;
}

static matx_status_t ref_dasum(matx_int64_t n, const matx_double* x, matx_int64_t incx, matx_double* result) {
	if (!x || !result) return MATX_ERR_INVALID_ARG;
#if MATX_ENABLE_OPENBLAS || MATX_ENABLE_BLIS
	*result = cblas_dasum(n, x, incx);
#else
	matx_double sum = 0.0;
	for (matx_int64_t i = 0; i < n; ++i)
		sum += fabs(x[i * incx]);
	*result = sum;
#endif
	return MATX_OK;
}

static matx_status_t ref_dzasum(matx_int64_t n, const void* x, matx_int64_t incx, matx_double* result) {
	if (!x || !result) return MATX_ERR_INVALID_ARG;
#if MATX_ENABLE_OPENBLAS || MATX_ENABLE_BLIS
	*result = cblas_dzasum(n, x, incx);
#else
	const matx_complex_f64_t* xd = (const matx_complex_f64_t*)x;
	matx_double sum = 0.0;
	for (matx_int64_t i = 0; i < n; ++i)
		sum += fabs(xd[i * incx].real) + fabs(xd[i * incx].imag);
	*result = sum;
#endif
	return MATX_OK;
}

static matx_status_t ref_idamax(matx_int64_t n, const matx_double* x, matx_int64_t incx, matx_int64_t* result) {
	if (!x || !result) return MATX_ERR_INVALID_ARG;
#if MATX_ENABLE_OPENBLAS || MATX_ENABLE_BLIS
	*result = (matx_int64_t)cblas_idamax(n, x, incx);
#else
	matx_double max_val = -1.0;
	matx_int64_t idx = 0;
	for (matx_int64_t i = 0; i < n; ++i) {
		matx_double v = fabs(x[i * incx]);
		if (v > max_val) { max_val = v; idx = i; }
	}
	*result = idx;
#endif
	return MATX_OK;
}

static matx_status_t ref_izamax(matx_int64_t n, const void* x, matx_int64_t incx, matx_int64_t* result) {
	if (!x || !result) return MATX_ERR_INVALID_ARG;
#if MATX_ENABLE_OPENBLAS || MATX_ENABLE_BLIS
	*result = (matx_int64_t)cblas_izamax(n, x, incx);
#else
	const matx_complex_f64_t* xd = (const matx_complex_f64_t*)x;
	matx_double max_val = -1.0;
	matx_int64_t idx = 0;
	for (matx_int64_t i = 0; i < n; ++i) {
		matx_double re = xd[i * incx].real;
		matx_double im = xd[i * incx].imag;
		matx_double v = fabs(re) + fabs(im);
		if (v > max_val) { max_val = v; idx = i; }
	}
	*result = idx;
#endif
	return MATX_OK;
}

static matx_status_t ref_daxpy(matx_int64_t n, matx_double alpha, const matx_double* x,
	matx_int64_t lda, void* y, matx_int64_t ldy) {
	if (!x || !y) return MATX_ERR_INVALID_ARG;
#if MATX_ENABLE_OPENBLAS || MATX_ENABLE_BLIS
	cblas_daxpy(n, alpha, x, lda, y, ldy);
#else
	matx_double* yd = (matx_double*)y;
	for (matx_int64_t i = 0; i < n; ++i)
		yd[i * ldy] += alpha * x[i * lda];
#endif
	return MATX_OK;
}

static matx_status_t ref_zaxpy(matx_int64_t n, const void* alpha, const void* x,
	matx_int64_t lda, void* y, matx_int64_t ldy) {
	if (!x || !y) return MATX_ERR_INVALID_ARG;
#if MATX_ENABLE_OPENBLAS || MATX_ENABLE_BLIS
	cblas_zaxpy(n, alpha, x, lda, y, ldy);
#else
	const matx_complex_f64_t* a = (const matx_complex_f64_t*)alpha;
	const matx_complex_f64_t* xd = (const matx_complex_f64_t*)x;
	matx_complex_f64_t* yd = (matx_complex_f64_t*)y;
	for (matx_int64_t i = 0; i < n; ++i) {
		yd[i * ldy].real += a->real * xd[i * lda].real - a->imag * xd[i * lda].imag;
		yd[i * ldy].imag += a->real * xd[i * lda].imag + a->imag * xd[i * lda].real;
	}
#endif
	return MATX_OK;
}

// ---- Vector norm implementations (pure C loops) ----

static matx_status_t ref_vec_norm1_f64(matx_vec_f64_t A, matx_double* out) {
	if (!A || !out) return MATX_ERR_INVALID_ARG;
	matx_double sum = 0.0;
	for (matx_int64_t i = 0; i < A->n; ++i)
		sum += fabs(A->data[i * A->stride]);
	*out = sum;
	return MATX_OK;
}

static matx_status_t ref_vec_norm2_f64(matx_vec_f64_t A, matx_double* out) {
	if (!A || !out) return MATX_ERR_INVALID_ARG;
	matx_double sum = 0.0;
	for (matx_int64_t i = 0; i < A->n; ++i) {
		matx_double v = A->data[i * A->stride];
		sum += v * v;
	}
	*out = sqrt(sum);
	return MATX_OK;
}

static matx_status_t ref_vec_norminf_f64(matx_vec_f64_t A, matx_double* out) {
	if (!A || !out) return MATX_ERR_INVALID_ARG;
	matx_double max_val = 0.0;
	for (matx_int64_t i = 0; i < A->n; ++i) {
		matx_double v = fabs(A->data[i * A->stride]);
		if (v > max_val) max_val = v;
	}
	*out = max_val;
	return MATX_OK;
}

static matx_status_t ref_vec_norm1_c64(matx_vec_c64_t A, matx_double* out) {
	if (!A || !out) return MATX_ERR_INVALID_ARG;
	matx_double sum = 0.0;
	for (matx_int64_t i = 0; i < A->n; ++i) {
		matx_double re = A->data[i * A->stride].real;
		matx_double im = A->data[i * A->stride].imag;
		sum += sqrt(re * re + im * im);
	}
	*out = sum;
	return MATX_OK;
}

static matx_status_t ref_vec_norm2_c64(matx_vec_c64_t A, matx_double* out) {
	if (!A || !out) return MATX_ERR_INVALID_ARG;
	matx_double sum = 0.0;
	for (matx_int64_t i = 0; i < A->n; ++i) {
		matx_double re = A->data[i * A->stride].real;
		matx_double im = A->data[i * A->stride].imag;
		sum += re * re + im * im;
	}
	*out = sqrt(sum);
	return MATX_OK;
}

static matx_status_t ref_vec_norminf_c64(matx_vec_c64_t A, matx_double* out) {
	if (!A || !out) return MATX_ERR_INVALID_ARG;
	matx_double max_val = 0.0;
	for (matx_int64_t i = 0; i < A->n; ++i) {
		matx_double re = A->data[i * A->stride].real;
		matx_double im = A->data[i * A->stride].imag;
		matx_double mag = sqrt(re * re + im * im);
		if (mag > max_val) max_val = mag;
	}
	*out = max_val;
	return MATX_OK;
}

// ---- Vtable constructor ----

matx_vec_backend_t matx_vec_blas_make_reference(void) {
	matx_vec_backend_t b = {
#if MATX_ENABLE_BLIS
		.kind = MATX_VEC_BACKEND_BLIS,
#elif MATX_ENABLE_OPENBLAS
		.kind = MATX_VEC_BACKEND_OPENBLAS,
#else
		.kind = MATX_VEC_BACKEND_REFERENCE,
#endif
		.vt = {
			.dscal = &ref_dscal,
			.zscal = &ref_zscal,
			.dcopy = &ref_dcopy,
			.zcopy = &ref_zcopy,
			.dswap = &ref_dswap,
			.zswap = &ref_zswap,
			.ddot  = &ref_ddot,
			.zdotu = &ref_zdotu,
			.zdotc = &ref_zdotc,
			.dnrm2 = &ref_dnrm2,
			.dznrm2 = &ref_dznrm2,
			.dasum = &ref_dasum,
			.dzasum = &ref_dzasum,
			.idamax = &ref_idamax,
			.izamax = &ref_izamax,
			.daxpy = &ref_daxpy,
			.zaxpy = &ref_zaxpy,
			.norm1_f64    = &ref_vec_norm1_f64,
			.norm1_c64    = &ref_vec_norm1_c64,
			.norm2_f64    = &ref_vec_norm2_f64,
			.norm2_c64    = &ref_vec_norm2_c64,
			.norminf_f64  = &ref_vec_norminf_f64,
			.norminf_c64  = &ref_vec_norminf_c64,
		}
	};
	return b;
}
