#include "matx/matx_func.h"
#include "matx/matx_types.h"
#include "matx/matx_types_internal.h"

#include "matx/matx_log.h"

#include <string.h>
#include <math.h>

matx_status_t matx_vec_d_i8_create(const matx_alloc_t* alloc,
                                   matx_vec_d_i8_t* out,
                                   matx_double* data,
                                   matx_int64_t n)
{
    if (!out || !alloc || n == 0) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    matx_vec_d_i8_opaque_t* out_value = matx_malloc(alloc, sizeof(matx_vec_d_i8_opaque_t));
    memset(out_value, 0, sizeof(matx_vec_d_i8_opaque_t));
    out_value->n = n;
    out_value->stride = 1;
    out_value->flags = 1u; // owns data
    out_value->data = (matx_double*) matx_malloc(alloc, n * sizeof(matx_double));

    if (!out_value->data) {
        matx_free(alloc, out_value);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    if (data != NULL) {
        memcpy(out_value->data, data, sizeof(matx_double) * out_value->n);
    }
    if (*out != NULL) {
        matx_vec_d_i8_destroy(alloc, *out);
    }
    *out = out_value;
    return MATX_OK;
}

matx_status_t matx_vec_d_i8_dup(const matx_alloc_t* alloc,
                                const matx_vec_d_i8_t in,
                                matx_vec_d_i8_t* out)
{
    return matx_vec_d_i8_create(alloc, out, in->data, in->n);
}

matx_status_t matx_vec_d_i8_wrap(const matx_alloc_t* alloc,
                                 matx_vec_d_i8_t* out,
                                 matx_int64_t n,
                                 matx_int64_t stride,
                                 matx_double* data)
{
    if (!out || !data || n == 0 || stride == 0) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    matx_vec_d_i8_opaque_t* out_value = matx_malloc(alloc, sizeof(matx_vec_d_i8_opaque_t));
    memset(out_value, 0, sizeof(matx_vec_d_i8_opaque_t));

    out_value->n = n;
    out_value->stride = stride;
    out_value->data = data;
    out_value->flags = 0u;
    if (*out != NULL) {
        matx_vec_d_i8_destroy(alloc, *out);
    }
    *out = out_value;
    return MATX_OK;
}

void matx_vec_d_i8_destroy(const matx_alloc_t* alloc, matx_vec_d_i8_t v)
{
    if (!v)
        return;
    if ((v->flags & 1u) != 0u && v->data && alloc) {
        matx_free(alloc, v->data);
    }
    matx_handles_destroy(v->backend_handles, &v->num_backend_handles);
    matx_free(alloc, v);
}

matx_status_t matx_vec_z_i8_create(const matx_alloc_t* alloc,
                                   matx_vec_z_i8_t* out,
                                   matx_complex_d_t* data,
                                   matx_int64_t n)
{
    if (!out || !alloc || n == 0) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_vec_z_i8_opaque_t* out_value = matx_malloc(alloc, sizeof(matx_vec_z_i8_opaque_t));
    memset(out_value, 0, sizeof(matx_vec_z_i8_opaque_t));

    out_value->n = n;
    out_value->stride = 1;
    out_value->flags = 1u;
    out_value->data = (matx_complex_d_t*) matx_malloc(alloc, n * sizeof(matx_complex_d_t));
    if (!out_value->data) {
        matx_free(alloc, out_value);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    if (data != NULL) {
        memcpy(out_value->data, data, sizeof(matx_complex_d_t) * out_value->n);
    }
    if (*out != NULL) {
        matx_vec_z_i8_destroy(alloc, *out);
    }
    *out = out_value;
    return MATX_OK;
}

matx_status_t matx_vec_z_i8_dup(const matx_alloc_t* alloc, matx_vec_z_i8_t in, matx_vec_z_i8_t* out)
{
    return matx_vec_z_i8_create(alloc, out, in->data, in->n);
}

matx_status_t matx_vec_z_i8_wrap(const matx_alloc_t* alloc,
                                 matx_vec_z_i8_t* out,
                                 matx_int64_t n,
                                 matx_int64_t stride,
                                 matx_complex_d_t* data)
{
    if (!out || !data || n == 0 || stride == 0) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    matx_vec_z_i8_opaque_t* out_value = matx_malloc(alloc, sizeof(matx_vec_z_i8_opaque_t));
    memset(out_value, 0, sizeof(matx_vec_z_i8_opaque_t));

    out_value->n = n;
    out_value->stride = stride;
    out_value->data = data;
    out_value->flags = 0u;
    if (*out != NULL) {
        matx_vec_z_i8_destroy(alloc, *out);
    }
    *out = out_value;
    return MATX_OK;
}

void matx_vec_z_i8_destroy(const matx_alloc_t* alloc, matx_vec_z_i8_t v)
{
    if (!v)
        return;
    if ((v->flags & 1u) != 0u && v->data && alloc) {
        matx_free(alloc, v->data);
        v->data = NULL;
    }

    matx_handles_destroy(v->backend_handles, &v->num_backend_handles);
    matx_free(alloc, v);
    //memset(v, 0, sizeof(*v));
}

matx_status_t matx_vec_d_i8_fill(matx_vec_d_i8_t v, matx_double val)
{
    if (!v || !v->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    for (matx_int64_t i = 0; i < v->n; ++i)
        v->data[i * v->stride] = val;
    return MATX_OK;
}

matx_status_t matx_vec_z_i8_fill(matx_vec_z_i8_t v, matx_complex_d_t val)
{
    if (!v || !v->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    for (matx_int64_t i = 0; i < v->n; ++i)
        v->data[i * v->stride] = val;
    return MATX_OK;
}

matx_status_t matx_vec_d_i8_zeros(matx_vec_d_i8_t v)
{
    if (!v || !v->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (v->stride == 1)
        memset(v->data, 0, v->n * sizeof(matx_double));
    else
        for (matx_int64_t i = 0; i < v->n; ++i)
            v->data[i * v->stride] = 0.0;
    return MATX_OK;
}

matx_status_t matx_vec_z_i8_zeros(matx_vec_z_i8_t v)
{
    if (!v || !v->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (v->stride == 1)
        memset(v->data, 0, v->n * sizeof(matx_complex_d_t));
    else {
        matx_complex_d_t zero = {0.0, 0.0};
        for (matx_int64_t i = 0; i < v->n; ++i)
            v->data[i * v->stride] = zero;
    }
    return MATX_OK;
}

matx_status_t matx_vec_d_i8_ones(matx_vec_d_i8_t v)
{
    return matx_vec_d_i8_fill(v, 1.0);
}

// ---- Element-wise math: vectors ----

matx_status_t matx_vec_d_i8_exp(const matx_alloc_t* alloc,
                                const matx_vec_d_i8_t v,
                                matx_vec_d_i8_t* out)
{
    if (!alloc || !v || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_status_t st = matx_vec_d_i8_create(alloc, out, NULL, v->n);
    if (st != MATX_OK)
        return st;
    for (matx_int64_t i = 0; i < v->n; ++i)
        (*out)->data[i] = exp(v->data[i * v->stride]);
    return MATX_OK;
}

matx_status_t matx_vec_z_i8_exp(const matx_alloc_t* alloc,
                                const matx_vec_z_i8_t v,
                                matx_vec_z_i8_t* out)
{
    if (!alloc || !v || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_status_t st = matx_vec_z_i8_create(alloc, out, NULL, v->n);
    if (st != MATX_OK)
        return st;
    for (matx_int64_t i = 0; i < v->n; ++i) {
        matx_double a = v->data[i * v->stride].real;
        matx_double b = v->data[i * v->stride].imag;
        matx_double e = exp(a);
        (*out)->data[i].real = e * cos(b);
        (*out)->data[i].imag = e * sin(b);
    }
    return MATX_OK;
}

matx_status_t matx_vec_d_i8_log(const matx_alloc_t* alloc,
                                const matx_vec_d_i8_t v,
                                matx_vec_d_i8_t* out)
{
    if (!alloc || !v || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_status_t st = matx_vec_d_i8_create(alloc, out, NULL, v->n);
    if (st != MATX_OK)
        return st;
    for (matx_int64_t i = 0; i < v->n; ++i) {
        matx_double x = v->data[i * v->stride];
        if (x <= 0.0) {
            matx_vec_d_i8_destroy(alloc, *out);
            *out = NULL;
            return MATX_ERR_INVALID_ARG;
        }
        (*out)->data[i] = log(x);
    }
    return MATX_OK;
}

matx_status_t matx_vec_z_i8_log(const matx_alloc_t* alloc,
                                const matx_vec_z_i8_t v,
                                matx_vec_z_i8_t* out)
{
    if (!alloc || !v || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_status_t st = matx_vec_z_i8_create(alloc, out, NULL, v->n);
    if (st != MATX_OK)
        return st;
    for (matx_int64_t i = 0; i < v->n; ++i) {
        matx_double a = v->data[i * v->stride].real;
        matx_double b = v->data[i * v->stride].imag;
        (*out)->data[i].real = 0.5 * log(a * a + b * b);
        (*out)->data[i].imag = atan2(b, a);
    }
    return MATX_OK;
}

matx_status_t matx_vec_d_i8_sqrt(const matx_alloc_t* alloc,
                                 const matx_vec_d_i8_t v,
                                 matx_vec_d_i8_t* out)
{
    if (!alloc || !v || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_status_t st = matx_vec_d_i8_create(alloc, out, NULL, v->n);
    if (st != MATX_OK)
        return st;
    for (matx_int64_t i = 0; i < v->n; ++i) {
        matx_double x = v->data[i * v->stride];
        if (x < 0.0) {
            matx_vec_d_i8_destroy(alloc, *out);
            *out = NULL;
            return MATX_ERR_INVALID_ARG;
        }
        (*out)->data[i] = sqrt(x);
    }
    return MATX_OK;
}

matx_status_t matx_vec_z_i8_sqrt(const matx_alloc_t* alloc,
                                 const matx_vec_z_i8_t v,
                                 matx_vec_z_i8_t* out)
{
    if (!alloc || !v || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_status_t st = matx_vec_z_i8_create(alloc, out, NULL, v->n);
    if (st != MATX_OK)
        return st;
    for (matx_int64_t i = 0; i < v->n; ++i) {
        matx_double a = v->data[i * v->stride].real;
        matx_double b = v->data[i * v->stride].imag;
        matx_double mag = sqrt(a * a + b * b);
        matx_double re = sqrt((mag + a) * 0.5);
        matx_double im = sqrt((mag - a) * 0.5);
        if (b < 0.0)
            im = -im;
        (*out)->data[i].real = re;
        (*out)->data[i].imag = im;
    }
    return MATX_OK;
}

matx_status_t matx_vec_d_i8_sin(const matx_alloc_t* alloc,
                                const matx_vec_d_i8_t v,
                                matx_vec_d_i8_t* out)
{
    if (!alloc || !v || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_status_t st = matx_vec_d_i8_create(alloc, out, NULL, v->n);
    if (st != MATX_OK)
        return st;
    for (matx_int64_t i = 0; i < v->n; ++i)
        (*out)->data[i] = sin(v->data[i * v->stride]);
    return MATX_OK;
}

matx_status_t matx_vec_z_i8_sin(const matx_alloc_t* alloc,
                                const matx_vec_z_i8_t v,
                                matx_vec_z_i8_t* out)
{
    if (!alloc || !v || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_status_t st = matx_vec_z_i8_create(alloc, out, NULL, v->n);
    if (st != MATX_OK)
        return st;
    for (matx_int64_t i = 0; i < v->n; ++i) {
        matx_double a = v->data[i * v->stride].real;
        matx_double b = v->data[i * v->stride].imag;
        (*out)->data[i].real = sin(a) * cosh(b);
        (*out)->data[i].imag = cos(a) * sinh(b);
    }
    return MATX_OK;
}

matx_status_t matx_vec_d_i8_cos(const matx_alloc_t* alloc,
                                const matx_vec_d_i8_t v,
                                matx_vec_d_i8_t* out)
{
    if (!alloc || !v || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_status_t st = matx_vec_d_i8_create(alloc, out, NULL, v->n);
    if (st != MATX_OK)
        return st;
    for (matx_int64_t i = 0; i < v->n; ++i)
        (*out)->data[i] = cos(v->data[i * v->stride]);
    return MATX_OK;
}

matx_status_t matx_vec_z_i8_cos(const matx_alloc_t* alloc,
                                const matx_vec_z_i8_t v,
                                matx_vec_z_i8_t* out)
{
    if (!alloc || !v || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_status_t st = matx_vec_z_i8_create(alloc, out, NULL, v->n);
    if (st != MATX_OK)
        return st;
    for (matx_int64_t i = 0; i < v->n; ++i) {
        matx_double a = v->data[i * v->stride].real;
        matx_double b = v->data[i * v->stride].imag;
        (*out)->data[i].real = cos(a) * cosh(b);
        (*out)->data[i].imag = -sin(a) * sinh(b);
    }
    return MATX_OK;
}

matx_status_t matx_vec_d_i8_abs(const matx_alloc_t* alloc,
                                const matx_vec_d_i8_t v,
                                matx_vec_d_i8_t* out)
{
    if (!alloc || !v || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_status_t st = matx_vec_d_i8_create(alloc, out, NULL, v->n);
    if (st != MATX_OK)
        return st;
    for (matx_int64_t i = 0; i < v->n; ++i)
        (*out)->data[i] = fabs(v->data[i * v->stride]);
    return MATX_OK;
}

matx_status_t matx_vec_z_i8_abs(const matx_alloc_t* alloc,
                                const matx_vec_z_i8_t v,
                                matx_vec_d_i8_t* out)
{
    if (!alloc || !v || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_status_t st = matx_vec_d_i8_create(alloc, out, NULL, v->n);
    if (st != MATX_OK)
        return st;
    for (matx_int64_t i = 0; i < v->n; ++i) {
        matx_double a = v->data[i * v->stride].real;
        matx_double b = v->data[i * v->stride].imag;
        (*out)->data[i] = sqrt(a * a + b * b);
    }
    return MATX_OK;
}

matx_status_t matx_vec_d_i8_pow(const matx_alloc_t* alloc,
                                const matx_vec_d_i8_t v,
                                matx_double exp_val,
                                matx_vec_d_i8_t* out)
{
    if (!alloc || !v || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_status_t st = matx_vec_d_i8_create(alloc, out, NULL, v->n);
    if (st != MATX_OK)
        return st;
    for (matx_int64_t i = 0; i < v->n; ++i)
        (*out)->data[i] = pow(v->data[i * v->stride], exp_val);
    return MATX_OK;
}

matx_status_t matx_vec_z_i8_pow(const matx_alloc_t* alloc,
                                const matx_vec_z_i8_t v,
                                matx_complex_d_t exp_val,
                                matx_vec_z_i8_t* out)
{
    if (!alloc || !v || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_status_t st = matx_vec_z_i8_create(alloc, out, NULL, v->n);
    if (st != MATX_OK)
        return st;
    for (matx_int64_t i = 0; i < v->n; ++i) {
        matx_double a = v->data[i * v->stride].real;
        matx_double b = v->data[i * v->stride].imag;
        matx_double r = sqrt(a * a + b * b);
        matx_double theta = atan2(b, a);
        matx_double c = exp_val.real;
        matx_double d = exp_val.imag;
        matx_double new_r = pow(r, c) * exp(-d * theta);
        matx_double new_theta = c * theta + d * log(r);
        (*out)->data[i].real = new_r * cos(new_theta);
        (*out)->data[i].imag = new_r * sin(new_theta);
    }
    return MATX_OK;
}

// ---- Vector element-wise arithmetic ----

matx_status_t matx_vec_d_i8_add(const matx_alloc_t* alloc,
                                const matx_vec_d_i8_t a,
                                const matx_vec_d_i8_t b,
                                matx_vec_d_i8_t* out)
{
    if (!alloc || !a || !b || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (a->n != b->n) {
        MATX_ERROR("%s: dimension mismatch", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_status_t st = matx_vec_d_i8_create(alloc, out, NULL, a->n);
    if (st != MATX_OK)
        return st;
    for (matx_int64_t i = 0; i < a->n; ++i)
        (*out)->data[i] = a->data[i * a->stride] + b->data[i * b->stride];
    return MATX_OK;
}

matx_status_t matx_vec_z_i8_add(const matx_alloc_t* alloc,
                                const matx_vec_z_i8_t a,
                                const matx_vec_z_i8_t b,
                                matx_vec_z_i8_t* out)
{
    if (!alloc || !a || !b || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (a->n != b->n) {
        MATX_ERROR("%s: dimension mismatch", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_status_t st = matx_vec_z_i8_create(alloc, out, NULL, a->n);
    if (st != MATX_OK)
        return st;
    for (matx_int64_t i = 0; i < a->n; ++i) {
        (*out)->data[i].real = a->data[i * a->stride].real + b->data[i * b->stride].real;
        (*out)->data[i].imag = a->data[i * a->stride].imag + b->data[i * b->stride].imag;
    }
    return MATX_OK;
}

matx_status_t matx_vec_d_i8_sub(const matx_alloc_t* alloc,
                                const matx_vec_d_i8_t a,
                                const matx_vec_d_i8_t b,
                                matx_vec_d_i8_t* out)
{
    if (!alloc || !a || !b || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (a->n != b->n) {
        MATX_ERROR("%s: dimension mismatch", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_status_t st = matx_vec_d_i8_create(alloc, out, NULL, a->n);
    if (st != MATX_OK)
        return st;
    for (matx_int64_t i = 0; i < a->n; ++i)
        (*out)->data[i] = a->data[i * a->stride] - b->data[i * b->stride];
    return MATX_OK;
}

matx_status_t matx_vec_z_i8_sub(const matx_alloc_t* alloc,
                                const matx_vec_z_i8_t a,
                                const matx_vec_z_i8_t b,
                                matx_vec_z_i8_t* out)
{
    if (!alloc || !a || !b || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (a->n != b->n) {
        MATX_ERROR("%s: dimension mismatch", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_status_t st = matx_vec_z_i8_create(alloc, out, NULL, a->n);
    if (st != MATX_OK)
        return st;
    for (matx_int64_t i = 0; i < a->n; ++i) {
        (*out)->data[i].real = a->data[i * a->stride].real - b->data[i * b->stride].real;
        (*out)->data[i].imag = a->data[i * a->stride].imag - b->data[i * b->stride].imag;
    }
    return MATX_OK;
}

matx_status_t matx_vec_d_i8_mul(const matx_alloc_t* alloc,
                                const matx_vec_d_i8_t a,
                                const matx_vec_d_i8_t b,
                                matx_vec_d_i8_t* out)
{
    if (!alloc || !a || !b || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (a->n != b->n) {
        MATX_ERROR("%s: dimension mismatch", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_status_t st = matx_vec_d_i8_create(alloc, out, NULL, a->n);
    if (st != MATX_OK)
        return st;
    for (matx_int64_t i = 0; i < a->n; ++i)
        (*out)->data[i] = a->data[i * a->stride] * b->data[i * b->stride];
    return MATX_OK;
}

matx_status_t matx_vec_z_i8_mul(const matx_alloc_t* alloc,
                                const matx_vec_z_i8_t a,
                                const matx_vec_z_i8_t b,
                                matx_vec_z_i8_t* out)
{
    if (!alloc || !a || !b || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (a->n != b->n) {
        MATX_ERROR("%s: dimension mismatch", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_status_t st = matx_vec_z_i8_create(alloc, out, NULL, a->n);
    if (st != MATX_OK)
        return st;
    for (matx_int64_t i = 0; i < a->n; ++i) {
        matx_double ar = a->data[i * a->stride].real, ai = a->data[i * a->stride].imag;
        matx_double br = b->data[i * b->stride].real, bi = b->data[i * b->stride].imag;
        (*out)->data[i].real = ar * br - ai * bi;
        (*out)->data[i].imag = ar * bi + ai * br;
    }
    return MATX_OK;
}

matx_status_t matx_vec_d_i8_div(const matx_alloc_t* alloc,
                                const matx_vec_d_i8_t a,
                                const matx_vec_d_i8_t b,
                                matx_vec_d_i8_t* out)
{
    if (!alloc || !a || !b || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (a->n != b->n) {
        MATX_ERROR("%s: dimension mismatch", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_status_t st = matx_vec_d_i8_create(alloc, out, NULL, a->n);
    if (st != MATX_OK)
        return st;
    for (matx_int64_t i = 0; i < a->n; ++i) {
        if (b->data[i * b->stride] == 0.0) {
            matx_vec_d_i8_destroy(alloc, *out);
            *out = NULL;
            return MATX_ERR_INVALID_ARG;
        }
        (*out)->data[i] = a->data[i * a->stride] / b->data[i * b->stride];
    }
    return MATX_OK;
}

matx_status_t matx_vec_z_i8_div(const matx_alloc_t* alloc,
                                const matx_vec_z_i8_t a,
                                const matx_vec_z_i8_t b,
                                matx_vec_z_i8_t* out)
{
    if (!alloc || !a || !b || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (a->n != b->n) {
        MATX_ERROR("%s: dimension mismatch", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_status_t st = matx_vec_z_i8_create(alloc, out, NULL, a->n);
    if (st != MATX_OK)
        return st;
    for (matx_int64_t i = 0; i < a->n; ++i) {
        matx_double ar = a->data[i * a->stride].real, ai = a->data[i * a->stride].imag;
        matx_double br = b->data[i * b->stride].real, bi = b->data[i * b->stride].imag;
        matx_double den = br * br + bi * bi;
        if (den == 0.0) {
            matx_vec_z_i8_destroy(alloc, *out);
            *out = NULL;
            return MATX_ERR_INVALID_ARG;
        }
        (*out)->data[i].real = (ar * br + ai * bi) / den;
        (*out)->data[i].imag = (ai * br - ar * bi) / den;
    }
    return MATX_OK;
}

// ---- In-place scalar operations ----

matx_status_t matx_vec_d_i8_add_scalar(matx_vec_d_i8_t v, matx_double val)
{
    if (!v || !v->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    for (matx_int64_t i = 0; i < v->n; ++i)
        v->data[i * v->stride] += val;
    return MATX_OK;
}

matx_status_t matx_vec_z_i8_add_scalar(matx_vec_z_i8_t v, matx_complex_d_t val)
{
    if (!v || !v->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    for (matx_int64_t i = 0; i < v->n; ++i) {
        v->data[i * v->stride].real += val.real;
        v->data[i * v->stride].imag += val.imag;
    }
    return MATX_OK;
}

matx_status_t matx_vec_d_i8_mul_scalar(matx_vec_d_i8_t v, matx_double val)
{
    if (!v || !v->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    for (matx_int64_t i = 0; i < v->n; ++i)
        v->data[i * v->stride] *= val;
    return MATX_OK;
}

matx_status_t matx_vec_z_i8_mul_scalar(matx_vec_z_i8_t v, matx_complex_d_t val)
{
    if (!v || !v->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    for (matx_int64_t i = 0; i < v->n; ++i) {
        matx_double re = v->data[i * v->stride].real;
        matx_double im = v->data[i * v->stride].imag;
        v->data[i * v->stride].real = re * val.real - im * val.imag;
        v->data[i * v->stride].imag = re * val.imag + im * val.real;
    }
    return MATX_OK;
}

// ---- Cumulative sum ----

matx_status_t matx_vec_d_i8_cumsum(const matx_alloc_t* alloc,
                                   const matx_vec_d_i8_t v,
                                   matx_vec_d_i8_t* out)
{
    if (!alloc || !v || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_status_t st = matx_vec_d_i8_create(alloc, out, NULL, v->n);
    if (st != MATX_OK)
        return st;
    matx_double acc = 0.0;
    for (matx_int64_t i = 0; i < v->n; ++i) {
        acc += v->data[i * v->stride];
        (*out)->data[i] = acc;
    }
    return MATX_OK;
}
