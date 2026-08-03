#if __linux__
#define _XOPEN_SOURCE 600
#endif

#include "matx/matx_log.h"
#include "matx/matx_types_internal.h"
#include "matx/matx_vec_compute.h"

#include <math.h>
#include <string.h>

#if MATX_ENABLE_OPENBLAS
#include "cblas.h"
#elif MATX_ENABLE_BLIS
#include "blis.h"
#endif

// ---- Level 1 implementations ----

static matx_status_t ref_dscal(matx_int64_t n, matx_double alpha, matx_double* x, matx_int64_t incx)
{
    if (!x) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
#if MATX_ENABLE_OPENBLAS || MATX_ENABLE_BLIS
    cblas_dscal(n, alpha, x, incx);
#else
    for (matx_int64_t i = 0; i < n; ++i)
        x[i * incx] *= alpha;
#endif
    return MATX_OK;
}

static matx_status_t ref_zscal(matx_int64_t n, const void* alpha, void* x, matx_int64_t incx)
{
    if (!x || !alpha) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
#if MATX_ENABLE_OPENBLAS || MATX_ENABLE_BLIS
    cblas_zscal(n, alpha, x, incx);
#else
    const matx_complex_d_t* a = (const matx_complex_d_t*) alpha;
    matx_complex_d_t* xd = (matx_complex_d_t*) x;
    for (matx_int64_t i = 0; i < n; ++i) {
        matx_complex_d_t v = xd[i * incx];
        xd[i * incx].real = a->real * v.real - a->imag * v.imag;
        xd[i * incx].imag = a->real * v.imag + a->imag * v.real;
    }
#endif
    return MATX_OK;
}

static matx_status_t ref_dcopy(
    matx_int64_t n, const matx_double* x, matx_int64_t incx, matx_double* y, matx_int64_t incy)
{
    if (!x || !y) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
#if MATX_ENABLE_OPENBLAS || MATX_ENABLE_BLIS
    cblas_dcopy(n, x, incx, y, incy);
#else
    for (matx_int64_t i = 0; i < n; ++i)
        y[i * incy] = x[i * incx];
#endif
    return MATX_OK;
}

static matx_status_t ref_zcopy(
    matx_int64_t n, const void* x, matx_int64_t incx, void* y, matx_int64_t incy)
{
    if (!x || !y) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
#if MATX_ENABLE_OPENBLAS || MATX_ENABLE_BLIS
    cblas_zcopy(n, x, incx, y, incy);
#else
    const matx_complex_d_t* xd = (const matx_complex_d_t*) x;
    matx_complex_d_t* yd = (matx_complex_d_t*) y;
    for (matx_int64_t i = 0; i < n; ++i)
        yd[i * incy] = xd[i * incx];
#endif
    return MATX_OK;
}

static matx_status_t ref_dswap(
    matx_int64_t n, matx_double* x, matx_int64_t incx, matx_double* y, matx_int64_t incy)
{
    if (!x || !y) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
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

static matx_status_t ref_zswap(matx_int64_t n, void* x, matx_int64_t incx, void* y, matx_int64_t incy)
{
    if (!x || !y) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
#if MATX_ENABLE_OPENBLAS || MATX_ENABLE_BLIS
    cblas_zswap(n, x, incx, y, incy);
#else
    matx_complex_d_t* xd = (matx_complex_d_t*) x;
    matx_complex_d_t* yd = (matx_complex_d_t*) y;
    for (matx_int64_t i = 0; i < n; ++i) {
        matx_complex_d_t t = xd[i * incx];
        xd[i * incx] = yd[i * incy];
        yd[i * incy] = t;
    }
#endif
    return MATX_OK;
}

static matx_status_t ref_ddot(matx_int64_t n,
                              const matx_double* x,
                              matx_int64_t incx,
                              const matx_double* y,
                              matx_int64_t incy,
                              matx_double* result)
{
    if (!x || !y || !result) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
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

static matx_status_t ref_zdotu(
    matx_int64_t n, const void* x, matx_int64_t incx, const void* y, matx_int64_t incy, void* result)
{
    if (!x || !y || !result) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
#if MATX_ENABLE_OPENBLAS || MATX_ENABLE_BLIS
    cblas_zdotu_sub(n, x, incx, y, incy, result);
#else
    const matx_complex_d_t* xd = (const matx_complex_d_t*) x;
    const matx_complex_d_t* yd = (const matx_complex_d_t*) y;
    matx_complex_d_t* r = (matx_complex_d_t*) result;
    r->real = 0.0;
    r->imag = 0.0;
    for (matx_int64_t i = 0; i < n; ++i) {
        r->real += xd[i * incx].real * yd[i * incy].real - xd[i * incx].imag * yd[i * incy].imag;
        r->imag += xd[i * incx].real * yd[i * incy].imag + xd[i * incx].imag * yd[i * incy].real;
    }
#endif
    return MATX_OK;
}

static matx_status_t ref_zdotc(
    matx_int64_t n, const void* x, matx_int64_t incx, const void* y, matx_int64_t incy, void* result)
{
    if (!x || !y || !result) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
#if MATX_ENABLE_OPENBLAS || MATX_ENABLE_BLIS
    cblas_zdotc_sub(n, x, incx, y, incy, result);
#else
    const matx_complex_d_t* xd = (const matx_complex_d_t*) x;
    const matx_complex_d_t* yd = (const matx_complex_d_t*) y;
    matx_complex_d_t* r = (matx_complex_d_t*) result;
    r->real = 0.0;
    r->imag = 0.0;
    for (matx_int64_t i = 0; i < n; ++i) {
        r->real += xd[i * incx].real * yd[i * incy].real + xd[i * incx].imag * yd[i * incy].imag;
        r->imag += -xd[i * incx].imag * yd[i * incy].real + xd[i * incx].real * yd[i * incy].imag;
    }
#endif
    return MATX_OK;
}

static matx_status_t ref_dnrm2(matx_int64_t n,
                               const matx_double* x,
                               matx_int64_t incx,
                               matx_double* result)
{
    if (!x || !result) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
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

static matx_status_t ref_dznrm2(matx_int64_t n,
                                const void* x,
                                matx_int64_t incx,
                                matx_double* result)
{
    if (!x || !result) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
#if MATX_ENABLE_OPENBLAS || MATX_ENABLE_BLIS
    *result = cblas_dznrm2(n, x, incx);
#else
    const matx_complex_d_t* xd = (const matx_complex_d_t*) x;
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

static matx_status_t ref_dasum(matx_int64_t n,
                               const matx_double* x,
                               matx_int64_t incx,
                               matx_double* result)
{
    if (!x || !result) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
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

static matx_status_t ref_dzasum(matx_int64_t n,
                                const void* x,
                                matx_int64_t incx,
                                matx_double* result)
{
    if (!x || !result) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
#if MATX_ENABLE_OPENBLAS || MATX_ENABLE_BLIS
    *result = cblas_dzasum(n, x, incx);
#else
    const matx_complex_d_t* xd = (const matx_complex_d_t*) x;
    matx_double sum = 0.0;
    for (matx_int64_t i = 0; i < n; ++i)
        sum += fabs(xd[i * incx].real) + fabs(xd[i * incx].imag);
    *result = sum;
#endif
    return MATX_OK;
}

static matx_status_t ref_idamax(matx_int64_t n,
                                const matx_double* x,
                                matx_int64_t incx,
                                matx_int64_t* result)
{
    if (!x || !result) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
#if MATX_ENABLE_OPENBLAS || MATX_ENABLE_BLIS
    *result = (matx_int64_t) cblas_idamax(n, x, incx);
#else
    matx_double max_val = -1.0;
    matx_int64_t idx = 0;
    for (matx_int64_t i = 0; i < n; ++i) {
        matx_double v = fabs(x[i * incx]);
        if (v > max_val) {
            max_val = v;
            idx = i;
        }
    }
    *result = idx;
#endif
    return MATX_OK;
}

static matx_status_t ref_izamax(matx_int64_t n,
                                const void* x,
                                matx_int64_t incx,
                                matx_int64_t* result)
{
    if (!x || !result) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
#if MATX_ENABLE_OPENBLAS || MATX_ENABLE_BLIS
    *result = (matx_int64_t) cblas_izamax(n, x, incx);
#else
    const matx_complex_d_t* xd = (const matx_complex_d_t*) x;
    matx_double max_val = -1.0;
    matx_int64_t idx = 0;
    for (matx_int64_t i = 0; i < n; ++i) {
        matx_double re = xd[i * incx].real;
        matx_double im = xd[i * incx].imag;
        matx_double v = fabs(re) + fabs(im);
        if (v > max_val) {
            max_val = v;
            idx = i;
        }
    }
    *result = idx;
#endif
    return MATX_OK;
}

static matx_status_t ref_daxpy(matx_int64_t n,
                               matx_double alpha,
                               const matx_double* x,
                               matx_int64_t lda,
                               void* y,
                               matx_int64_t ldy)
{
    if (!x || !y) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
#if MATX_ENABLE_OPENBLAS || MATX_ENABLE_BLIS
    cblas_daxpy(n, alpha, x, lda, y, ldy);
#else
    matx_double* yd = (matx_double*) y;
    for (matx_int64_t i = 0; i < n; ++i)
        yd[i * ldy] += alpha * x[i * lda];
#endif
    return MATX_OK;
}

static matx_status_t ref_zaxpy(
    matx_int64_t n, const void* alpha, const void* x, matx_int64_t lda, void* y, matx_int64_t ldy)
{
    if (!x || !y) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
#if MATX_ENABLE_OPENBLAS || MATX_ENABLE_BLIS
    cblas_zaxpy(n, alpha, x, lda, y, ldy);
#else
    const matx_complex_d_t* a = (const matx_complex_d_t*) alpha;
    const matx_complex_d_t* xd = (const matx_complex_d_t*) x;
    matx_complex_d_t* yd = (matx_complex_d_t*) y;
    for (matx_int64_t i = 0; i < n; ++i) {
        yd[i * ldy].real += a->real * xd[i * lda].real - a->imag * xd[i * lda].imag;
        yd[i * ldy].imag += a->real * xd[i * lda].imag + a->imag * xd[i * lda].real;
    }
#endif
    return MATX_OK;
}

// ---- Vector norm implementations (pure C loops) ----

static matx_status_t ref_vec_norm1_d_i8(matx_vec_d_i8_t A, matx_double* out)
{
    if (!A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_double sum = 0.0;
    for (matx_int64_t i = 0; i < A->n; ++i)
        sum += fabs(A->data[i * A->stride]);
    *out = sum;
    return MATX_OK;
}

static matx_status_t ref_vec_norm2_d_i8(matx_vec_d_i8_t A, matx_double* out)
{
    if (!A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_double sum = 0.0;
    for (matx_int64_t i = 0; i < A->n; ++i) {
        matx_double v = A->data[i * A->stride];
        sum += v * v;
    }
    *out = sqrt(sum);
    return MATX_OK;
}

static matx_status_t ref_vec_norminf_d_i8(matx_vec_d_i8_t A, matx_double* out)
{
    if (!A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_double max_val = 0.0;
    for (matx_int64_t i = 0; i < A->n; ++i) {
        matx_double v = fabs(A->data[i * A->stride]);
        if (v > max_val)
            max_val = v;
    }
    *out = max_val;
    return MATX_OK;
}

static matx_status_t ref_vec_norm1_z_i8(matx_vec_z_i8_t A, matx_double* out)
{
    if (!A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_double sum = 0.0;
    for (matx_int64_t i = 0; i < A->n; ++i) {
        matx_double re = A->data[i * A->stride].real;
        matx_double im = A->data[i * A->stride].imag;
        sum += sqrt(re * re + im * im);
    }
    *out = sum;
    return MATX_OK;
}

static matx_status_t ref_vec_norm2_z_i8(matx_vec_z_i8_t A, matx_double* out)
{
    if (!A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_double sum = 0.0;
    for (matx_int64_t i = 0; i < A->n; ++i) {
        matx_double re = A->data[i * A->stride].real;
        matx_double im = A->data[i * A->stride].imag;
        sum += re * re + im * im;
    }
    *out = sqrt(sum);
    return MATX_OK;
}

static matx_status_t ref_vec_norminf_z_i8(matx_vec_z_i8_t A, matx_double* out)
{
    if (!A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_double max_val = 0.0;
    for (matx_int64_t i = 0; i < A->n; ++i) {
        matx_double re = A->data[i * A->stride].real;
        matx_double im = A->data[i * A->stride].imag;
        matx_double mag = sqrt(re * re + im * im);
        if (mag > max_val)
            max_val = mag;
    }
    *out = max_val;
    return MATX_OK;
}

// ---- Cross product implementations ----

static matx_status_t ref_vec_cross_d_i8(matx_vec_d_i8_t x, matx_vec_d_i8_t y, matx_vec_d_i8_t out)
{
    if (!x || !y || !out || !x->data || !y->data || !out->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_double x0 = x->data[0 * x->stride];
    matx_double x1 = x->data[1 * x->stride];
    matx_double x2 = x->data[2 * x->stride];
    matx_double y0 = y->data[0 * y->stride];
    matx_double y1 = y->data[1 * y->stride];
    matx_double y2 = y->data[2 * y->stride];
    out->data[0 * out->stride] = x1 * y2 - x2 * y1;
    out->data[1 * out->stride] = x2 * y0 - x0 * y2;
    out->data[2 * out->stride] = x0 * y1 - x1 * y0;
    return MATX_OK;
}

static matx_status_t ref_vec_cross_z_i8(matx_vec_z_i8_t x, matx_vec_z_i8_t y, matx_vec_z_i8_t out)
{
    if (!x || !y || !out || !x->data || !y->data || !out->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_complex_d_t x0 = x->data[0 * x->stride];
    matx_complex_d_t x1 = x->data[1 * x->stride];
    matx_complex_d_t x2 = x->data[2 * x->stride];
    matx_complex_d_t y0 = y->data[0 * y->stride];
    matx_complex_d_t y1 = y->data[1 * y->stride];
    matx_complex_d_t y2 = y->data[2 * y->stride];
    // out[0] = x1*y2 - x2*y1
    out->data[0 * out->stride].real = x1.real * y2.real - x1.imag * y2.imag
                                      - (x2.real * y1.real - x2.imag * y1.imag);
    out->data[0 * out->stride].imag = x1.real * y2.imag + x1.imag * y2.real
                                      - (x2.real * y1.imag + x2.imag * y1.real);
    // out[1] = x2*y0 - x0*y2
    out->data[1 * out->stride].real = x2.real * y0.real - x2.imag * y0.imag
                                      - (x0.real * y2.real - x0.imag * y2.imag);
    out->data[1 * out->stride].imag = x2.real * y0.imag + x2.imag * y0.real
                                      - (x0.real * y2.imag + x0.imag * y2.real);
    // out[2] = x0*y1 - x1*y0
    out->data[2 * out->stride].real = x0.real * y1.real - x0.imag * y1.imag
                                      - (x1.real * y0.real - x1.imag * y0.imag);
    out->data[2 * out->stride].imag = x0.real * y1.imag + x0.imag * y1.real
                                      - (x1.real * y0.imag + x1.imag * y0.real);
    return MATX_OK;
}

// ---- Vtable constructor ----

matx_vec_backend_t matx_vec_blas_make_reference(void)
{
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
        .ddot = &ref_ddot,
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
        .norm1_d_i8 = &ref_vec_norm1_d_i8,
        .norm1_z_i8 = &ref_vec_norm1_z_i8,
        .norm2_d_i8 = &ref_vec_norm2_d_i8,
        .norm2_z_i8 = &ref_vec_norm2_z_i8,
        .norminf_d_i8 = &ref_vec_norminf_d_i8,
        .norminf_z_i8 = &ref_vec_norminf_z_i8,
        .cross_d_i8 = &ref_vec_cross_d_i8,
        .cross_z_i8 = &ref_vec_cross_z_i8,
        }};
    return b;
}
