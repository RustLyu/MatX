#include "matx/matx_func.h"
#include "matx/matx_types.h"
#include "matx/matx_types_internal.h"

#include "matx/matx_log.h"
#include <math.h>
#include <string.h>

matx_status_t matx_dense_z_i8_create(const matx_alloc_t* alloc,
                                     matx_dense_z_i8_t* out,
                                     matx_layout_t layout,
                                     matx_int64_t rows,
                                     matx_int64_t cols,
                                     matx_complex_d_t* data)
{
    if (!out || !alloc || rows == 0 || cols == 0) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (layout != MATX_COL_MAJOR && layout != MATX_ROW_MAJOR) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    matx_dense_z_i8_opaque_t* out_value = matx_malloc(alloc, sizeof(matx_dense_z_i8_opaque_t));
    if (!out_value) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(out_value, 0, sizeof(*out_value));
    out_value->nrows = rows;
    out_value->ncols = cols;
    out_value->layout = layout;
    out_value->stride = (layout == MATX_COL_MAJOR) ? rows : cols;
    out_value->flags = 1u;

    const size_t n = rows * cols;
    out_value->data = (matx_complex_d_t*) matx_malloc(alloc, n * sizeof(matx_complex_d_t));
    if (!out_value->data) {
        matx_free(alloc, out_value);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    if (data != NULL) {
        memcpy(out_value->data,
               data,
               sizeof(matx_complex_d_t) * out_value->nrows * out_value->ncols);
    }
    if (*out != NULL) {
        matx_dense_z_i8_destroy(alloc, *out);
    }
    *out = out_value;
    return MATX_OK;
}

matx_status_t matx_dense_z_i8_dup(const matx_alloc_t* alloc,
                                  const matx_dense_z_i8_t in,
                                  matx_dense_z_i8_t* out)
{
    return matx_dense_z_i8_create(alloc, out, in->layout, in->nrows, in->ncols, in->data);
}

matx_status_t matx_dense_z_i8_wrap(const matx_alloc_t* alloc,
                                   matx_dense_z_i8_t* out,
                                   matx_int64_t rows,
                                   matx_int64_t cols,
                                   matx_int64_t stride,
                                   matx_layout_t layout,
                                   matx_complex_d_t* data)
{
    if (!out || !data || rows == 0 || cols == 0) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (layout != MATX_COL_MAJOR && layout != MATX_ROW_MAJOR) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (layout == MATX_COL_MAJOR) {
        if (stride < rows) {
            MATX_ERROR("%s: invalid argument", __func__);
            return MATX_ERR_INVALID_ARG;
        }
    } else {
        if (stride < cols) {
            MATX_ERROR("%s: invalid argument", __func__);
            return MATX_ERR_INVALID_ARG;
        }
    }

    matx_dense_z_i8_opaque_t* out_value = matx_malloc(alloc, sizeof(matx_dense_z_i8_opaque_t));
    if (!out_value) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(out_value, 0, sizeof(*out_value));

    out_value->nrows = rows;
    out_value->ncols = cols;
    out_value->stride = stride;
    out_value->layout = layout;
    out_value->data = data;
    out_value->flags = 0u;
    if (*out != NULL) {
        matx_dense_z_i8_destroy(alloc, *out);
    }
    *out = out_value;
    return MATX_OK;
}

void matx_dense_z_i8_destroy(const matx_alloc_t* alloc, matx_dense_z_i8_t m)
{
    if (!m)
        return;
    if ((m->flags & 1u) != 0u && m->data && alloc) {
        matx_free(alloc, m->data);
        m->data = NULL;
    }
    matx_handles_destroy(&m->backend_handles, &m->num_backend_handles);
    matx_free(alloc, m);
}

matx_status_t matx_dense_z_i8_fill(matx_dense_z_i8_t m, matx_complex_d_t val)
{
    if (!m || !m->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const matx_int64_t total = m->nrows * m->ncols;
    for (matx_int64_t i = 0; i < total; ++i)
        m->data[i] = val;
    return MATX_OK;
}

matx_status_t matx_dense_z_i8_zeros(matx_dense_z_i8_t m)
{
    if (!m || !m->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    memset(m->data, 0, (size_t) (m->nrows * m->ncols) * sizeof(matx_complex_d_t));
    return MATX_OK;
}

matx_status_t matx_dense_z_i8_trace(const matx_dense_z_i8_t A, matx_complex_d_t* out)
{
    if (!A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (A->nrows != A->ncols) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_complex_d_t sum = {0.0, 0.0};
    for (matx_int64_t i = 0; i < A->nrows; ++i) {
        matx_int64_t idx = (A->layout == MATX_COL_MAJOR) ? i + i * A->stride : i * A->stride + i;
        sum.real += A->data[idx].real;
        sum.imag += A->data[idx].imag;
    }
    *out = sum;
    return MATX_OK;
}

matx_status_t matx_vec_d_i8_to_z_i8(const matx_alloc_t* alloc,
                                    const matx_vec_d_i8_t v,
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
        (*out)->data[i].real = v->data[i * v->stride];
        (*out)->data[i].imag = 0.0;
    }
    return MATX_OK;
}

// ---- Element-wise math: dense complex ----

matx_status_t matx_dense_z_i8_exp(const matx_alloc_t* alloc,
                                  const matx_dense_z_i8_t A,
                                  matx_dense_z_i8_t* out)
{
    if (!alloc || !A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_status_t st = matx_dense_z_i8_create(alloc, out, A->layout, A->nrows, A->ncols, NULL);
    if (st != MATX_OK)
        return st;
    const matx_int64_t total = A->nrows * A->ncols;
    for (matx_int64_t i = 0; i < total; ++i) {
        matx_double a = A->data[i].real;
        matx_double b = A->data[i].imag;
        matx_double e = exp(a);
        (*out)->data[i].real = e * cos(b);
        (*out)->data[i].imag = e * sin(b);
    }
    return MATX_OK;
}

matx_status_t matx_dense_z_i8_log(const matx_alloc_t* alloc,
                                  const matx_dense_z_i8_t A,
                                  matx_dense_z_i8_t* out)
{
    if (!alloc || !A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_status_t st = matx_dense_z_i8_create(alloc, out, A->layout, A->nrows, A->ncols, NULL);
    if (st != MATX_OK)
        return st;
    const matx_int64_t total = A->nrows * A->ncols;
    for (matx_int64_t i = 0; i < total; ++i) {
        matx_double a = A->data[i].real;
        matx_double b = A->data[i].imag;
        (*out)->data[i].real = 0.5 * log(a * a + b * b);
        (*out)->data[i].imag = atan2(b, a);
    }
    return MATX_OK;
}

matx_status_t matx_dense_z_i8_sqrt(const matx_alloc_t* alloc,
                                   const matx_dense_z_i8_t A,
                                   matx_dense_z_i8_t* out)
{
    if (!alloc || !A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_status_t st = matx_dense_z_i8_create(alloc, out, A->layout, A->nrows, A->ncols, NULL);
    if (st != MATX_OK)
        return st;
    const matx_int64_t total = A->nrows * A->ncols;
    for (matx_int64_t i = 0; i < total; ++i) {
        matx_double a = A->data[i].real;
        matx_double b = A->data[i].imag;
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

matx_status_t matx_dense_z_i8_sin(const matx_alloc_t* alloc,
                                  const matx_dense_z_i8_t A,
                                  matx_dense_z_i8_t* out)
{
    if (!alloc || !A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_status_t st = matx_dense_z_i8_create(alloc, out, A->layout, A->nrows, A->ncols, NULL);
    if (st != MATX_OK)
        return st;
    const matx_int64_t total = A->nrows * A->ncols;
    for (matx_int64_t i = 0; i < total; ++i) {
        matx_double a = A->data[i].real;
        matx_double b = A->data[i].imag;
        (*out)->data[i].real = sin(a) * cosh(b);
        (*out)->data[i].imag = cos(a) * sinh(b);
    }
    return MATX_OK;
}

matx_status_t matx_dense_z_i8_cos(const matx_alloc_t* alloc,
                                  const matx_dense_z_i8_t A,
                                  matx_dense_z_i8_t* out)
{
    if (!alloc || !A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_status_t st = matx_dense_z_i8_create(alloc, out, A->layout, A->nrows, A->ncols, NULL);
    if (st != MATX_OK)
        return st;
    const matx_int64_t total = A->nrows * A->ncols;
    for (matx_int64_t i = 0; i < total; ++i) {
        matx_double a = A->data[i].real;
        matx_double b = A->data[i].imag;
        (*out)->data[i].real = cos(a) * cosh(b);
        (*out)->data[i].imag = -sin(a) * sinh(b);
    }
    return MATX_OK;
}

matx_status_t matx_dense_z_i8_abs(const matx_alloc_t* alloc,
                                  const matx_dense_z_i8_t A,
                                  matx_dense_d_i8_t* out)
{
    if (!alloc || !A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_status_t st = matx_dense_d_i8_create(alloc, out, A->layout, A->nrows, A->ncols, NULL);
    if (st != MATX_OK)
        return st;
    const matx_int64_t total = A->nrows * A->ncols;
    for (matx_int64_t i = 0; i < total; ++i) {
        matx_double a = A->data[i].real;
        matx_double b = A->data[i].imag;
        (*out)->data[i] = sqrt(a * a + b * b);
    }
    return MATX_OK;
}

matx_status_t matx_dense_z_i8_pow(const matx_alloc_t* alloc,
                                  const matx_dense_z_i8_t A,
                                  matx_complex_d_t exp_val,
                                  matx_dense_z_i8_t* out)
{
    if (!alloc || !A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_status_t st = matx_dense_z_i8_create(alloc, out, A->layout, A->nrows, A->ncols, NULL);
    if (st != MATX_OK)
        return st;
    const matx_int64_t total = A->nrows * A->ncols;
    for (matx_int64_t i = 0; i < total; ++i) {
        matx_double a = A->data[i].real;
        matx_double b = A->data[i].imag;
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

// ---- Dense element-wise arithmetic ----

matx_status_t matx_dense_z_i8_add(const matx_alloc_t* alloc,
                                  const matx_dense_z_i8_t A,
                                  const matx_dense_z_i8_t B,
                                  matx_dense_z_i8_t* out)
{
    if (!alloc || !A || !B || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (A->nrows != B->nrows || A->ncols != B->ncols || A->layout != B->layout) {
        MATX_ERROR("%s: dimension/layout mismatch", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_status_t st = matx_dense_z_i8_create(alloc, out, A->layout, A->nrows, A->ncols, NULL);
    if (st != MATX_OK)
        return st;
    const matx_int64_t total = A->nrows * A->ncols;
    for (matx_int64_t i = 0; i < total; ++i) {
        (*out)->data[i].real = A->data[i].real + B->data[i].real;
        (*out)->data[i].imag = A->data[i].imag + B->data[i].imag;
    }
    return MATX_OK;
}

matx_status_t matx_dense_z_i8_sub(const matx_alloc_t* alloc,
                                  const matx_dense_z_i8_t A,
                                  const matx_dense_z_i8_t B,
                                  matx_dense_z_i8_t* out)
{
    if (!alloc || !A || !B || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (A->nrows != B->nrows || A->ncols != B->ncols || A->layout != B->layout) {
        MATX_ERROR("%s: dimension/layout mismatch", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_status_t st = matx_dense_z_i8_create(alloc, out, A->layout, A->nrows, A->ncols, NULL);
    if (st != MATX_OK)
        return st;
    const matx_int64_t total = A->nrows * A->ncols;
    for (matx_int64_t i = 0; i < total; ++i) {
        (*out)->data[i].real = A->data[i].real - B->data[i].real;
        (*out)->data[i].imag = A->data[i].imag - B->data[i].imag;
    }
    return MATX_OK;
}

matx_status_t matx_dense_z_i8_mul(const matx_alloc_t* alloc,
                                  const matx_dense_z_i8_t A,
                                  const matx_dense_z_i8_t B,
                                  matx_dense_z_i8_t* out)
{
    if (!alloc || !A || !B || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (A->nrows != B->nrows || A->ncols != B->ncols || A->layout != B->layout) {
        MATX_ERROR("%s: dimension/layout mismatch", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_status_t st = matx_dense_z_i8_create(alloc, out, A->layout, A->nrows, A->ncols, NULL);
    if (st != MATX_OK)
        return st;
    const matx_int64_t total = A->nrows * A->ncols;
    for (matx_int64_t i = 0; i < total; ++i) {
        matx_double ar = A->data[i].real, ai = A->data[i].imag;
        matx_double br = B->data[i].real, bi = B->data[i].imag;
        (*out)->data[i].real = ar * br - ai * bi;
        (*out)->data[i].imag = ar * bi + ai * br;
    }
    return MATX_OK;
}

matx_status_t matx_dense_z_i8_div(const matx_alloc_t* alloc,
                                  const matx_dense_z_i8_t A,
                                  const matx_dense_z_i8_t B,
                                  matx_dense_z_i8_t* out)
{
    if (!alloc || !A || !B || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (A->nrows != B->nrows || A->ncols != B->ncols || A->layout != B->layout) {
        MATX_ERROR("%s: dimension/layout mismatch", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_status_t st = matx_dense_z_i8_create(alloc, out, A->layout, A->nrows, A->ncols, NULL);
    if (st != MATX_OK)
        return st;
    const matx_int64_t total = A->nrows * A->ncols;
    for (matx_int64_t i = 0; i < total; ++i) {
        matx_double ar = A->data[i].real, ai = A->data[i].imag;
        matx_double br = B->data[i].real, bi = B->data[i].imag;
        matx_double den = br * br + bi * bi;
        if (den == 0.0) {
            matx_dense_z_i8_destroy(alloc, *out);
            *out = NULL;
            return MATX_ERR_INVALID_ARG;
        }
        (*out)->data[i].real = (ar * br + ai * bi) / den;
        (*out)->data[i].imag = (ai * br - ar * bi) / den;
    }
    return MATX_OK;
}

// ---- In-place scalar operations ----

matx_status_t matx_dense_z_i8_add_scalar(matx_dense_z_i8_t m, matx_complex_d_t val)
{
    if (!m || !m->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const matx_int64_t total = m->nrows * m->ncols;
    for (matx_int64_t i = 0; i < total; ++i) {
        m->data[i].real += val.real;
        m->data[i].imag += val.imag;
    }
    return MATX_OK;
}

matx_status_t matx_dense_z_i8_mul_scalar(matx_dense_z_i8_t m, matx_complex_d_t val)
{
    if (!m || !m->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const matx_int64_t total = m->nrows * m->ncols;
    for (matx_int64_t i = 0; i < total; ++i) {
        matx_double re = m->data[i].real;
        matx_double im = m->data[i].imag;
        m->data[i].real = re * val.real - im * val.imag;
        m->data[i].imag = re * val.imag + im * val.real;
    }
    return MATX_OK;
}

// ---- Diagonal matrix creation / extraction ----

matx_status_t matx_diag_z_i8_create(const matx_alloc_t* alloc,
                                    const matx_vec_z_i8_t diag,
                                    matx_dense_z_i8_t* out)
{
    if (!alloc || !diag || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_status_t st = matx_dense_z_i8_create(alloc, out, MATX_COL_MAJOR, diag->n, diag->n, NULL);
    if (st != MATX_OK)
        return st;
    matx_dense_z_i8_zeros(*out);
    for (matx_int64_t i = 0; i < diag->n; ++i)
        (*out)->data[i + i * (*out)->stride] = diag->data[i * diag->stride];
    return MATX_OK;
}

matx_status_t matx_dense_z_i8_get_diag(const matx_alloc_t* alloc,
                                       const matx_dense_z_i8_t A,
                                       matx_vec_z_i8_t* out)
{
    if (!alloc || !A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_int64_t n = A->nrows < A->ncols ? A->nrows : A->ncols;
    matx_status_t st = matx_vec_z_i8_create(alloc, out, NULL, n);
    if (st != MATX_OK)
        return st;
    for (matx_int64_t i = 0; i < n; ++i) {
        matx_int64_t idx = (A->layout == MATX_COL_MAJOR) ? i + i * A->stride : i * A->stride + i;
        (*out)->data[i] = A->data[idx];
    }
    return MATX_OK;
}