#include "matx/matx_func.h"
#include "matx/matx_types.h"
#include "matx/matx_types_internal.h"

#include "matx/matx_log.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

matx_status_t matx_dense_d_i8_create(const matx_alloc_t* alloc,
                                     matx_dense_d_i8_t* out,
                                     matx_layout_t layout,
                                     matx_int64_t rows,
                                     matx_int64_t cols,
                                     matx_double* data)
{
    if (!out || rows == 0 || cols == 0) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (layout != MATX_COL_MAJOR && layout != MATX_ROW_MAJOR) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!alloc) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    matx_dense_d_i8_opaque_t* out_value = matx_malloc(alloc, sizeof(matx_dense_d_i8_opaque_t));
    if (!out_value) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(out_value, 0, sizeof(matx_dense_d_i8_opaque_t));
    out_value->nrows = rows;
    out_value->ncols = cols;
    out_value->layout = layout;
    out_value->stride = (layout == MATX_COL_MAJOR) ? rows : cols;
    out_value->flags = 1u;

    const size_t n = rows * cols;
    out_value->data = (matx_double*) matx_malloc(alloc, n * sizeof(matx_double));
    if (!out_value->data) {
        matx_free(alloc, out_value);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    if (data != NULL) {
        memcpy(out_value->data, data, sizeof(matx_double) * n);
    }
    if (*out != NULL) {
        matx_dense_d_i8_destroy(alloc, *out);
    }
    *out = out_value;
    return MATX_OK;
}

matx_status_t matx_dense_d_i8_dup(const matx_alloc_t* alloc,
                                  const matx_dense_d_i8_t in,
                                  matx_dense_d_i8_t* out)
{
    return matx_dense_d_i8_create(alloc, out, in->layout, in->nrows, in->ncols, in->data);
}

matx_status_t matx_dense_d_i8_wrap(const matx_alloc_t* alloc,
                                   matx_dense_d_i8_t* out,
                                   matx_int64_t rows,
                                   matx_int64_t cols,
                                   matx_int64_t stride,
                                   matx_layout_t layout,
                                   matx_double* data)
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
    matx_dense_d_i8_opaque_t* out_value = matx_malloc(alloc, sizeof(matx_dense_d_i8_opaque_t));
    if (!out_value) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(out_value, 0, sizeof(matx_dense_d_i8_opaque_t));

    out_value->nrows = rows;
    out_value->ncols = cols;
    out_value->stride = stride;
    out_value->layout = layout;
    out_value->data = data;
    out_value->flags = 0u;

    if (*out != NULL) {
        matx_dense_d_i8_destroy(alloc, *out);
    }

    *out = out_value;
    return MATX_OK;
}

void matx_dense_d_i8_destroy(const matx_alloc_t* alloc, matx_dense_d_i8_t m)
{
    if (!m)
        return;
    if ((m->flags & 1u) != 0u && m->data) {
        if (alloc) {
            matx_free(alloc, m->data);
            m->data = NULL;
        }
    } else if (m->flags == 0u) {
        m->data = NULL;
    }

    matx_handles_destroy(&m->backend_handles, &m->num_backend_handles);
    matx_free(alloc, m);
}

matx_status_t matx_dense_d_i8_fill(matx_dense_d_i8_t m, matx_double val)
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

matx_status_t matx_dense_d_i8_zeros(matx_dense_d_i8_t m)
{
    if (!m || !m->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    memset(m->data, 0, (size_t) (m->nrows * m->ncols) * sizeof(matx_double));
    return MATX_OK;
}

matx_status_t matx_dense_d_i8_ones(matx_dense_d_i8_t m)
{
    return matx_dense_d_i8_fill(m, 1.0);
}

matx_status_t matx_dense_d_i8_trace(const matx_dense_d_i8_t A, matx_double* out)
{
    if (!A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (A->nrows != A->ncols) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_double sum = 0.0;
    for (matx_int64_t i = 0; i < A->nrows; ++i) {
        matx_int64_t idx = (A->layout == MATX_COL_MAJOR) ? i + i * A->stride : i * A->stride + i;
        sum += A->data[idx];
    }
    *out = sum;
    return MATX_OK;
}

matx_status_t matx_dense_d_i8_to_z_i8(const matx_alloc_t* alloc,
                                      const matx_dense_d_i8_t A,
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
        (*out)->data[i].real = A->data[i];
        (*out)->data[i].imag = 0.0;
    }
    return MATX_OK;
}

// ---- Element-wise math: dense real ----

matx_status_t matx_dense_d_i8_exp(const matx_alloc_t* alloc,
                                  const matx_dense_d_i8_t A,
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
    for (matx_int64_t i = 0; i < total; ++i)
        (*out)->data[i] = exp(A->data[i]);
    return MATX_OK;
}

matx_status_t matx_dense_d_i8_log(const matx_alloc_t* alloc,
                                  const matx_dense_d_i8_t A,
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
        if (A->data[i] <= 0.0) {
            matx_dense_d_i8_destroy(alloc, *out);
            *out = NULL;
            return MATX_ERR_INVALID_ARG;
        }
        (*out)->data[i] = log(A->data[i]);
    }
    return MATX_OK;
}

matx_status_t matx_dense_d_i8_sqrt(const matx_alloc_t* alloc,
                                   const matx_dense_d_i8_t A,
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
        if (A->data[i] < 0.0) {
            matx_dense_d_i8_destroy(alloc, *out);
            *out = NULL;
            return MATX_ERR_INVALID_ARG;
        }
        (*out)->data[i] = sqrt(A->data[i]);
    }
    return MATX_OK;
}

matx_status_t matx_dense_d_i8_sin(const matx_alloc_t* alloc,
                                  const matx_dense_d_i8_t A,
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
    for (matx_int64_t i = 0; i < total; ++i)
        (*out)->data[i] = sin(A->data[i]);
    return MATX_OK;
}

matx_status_t matx_dense_d_i8_cos(const matx_alloc_t* alloc,
                                  const matx_dense_d_i8_t A,
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
    for (matx_int64_t i = 0; i < total; ++i)
        (*out)->data[i] = cos(A->data[i]);
    return MATX_OK;
}

matx_status_t matx_dense_d_i8_abs(const matx_alloc_t* alloc,
                                  const matx_dense_d_i8_t A,
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
    for (matx_int64_t i = 0; i < total; ++i)
        (*out)->data[i] = fabs(A->data[i]);
    return MATX_OK;
}

matx_status_t matx_dense_d_i8_pow(const matx_alloc_t* alloc,
                                  const matx_dense_d_i8_t A,
                                  matx_double exp_val,
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
    for (matx_int64_t i = 0; i < total; ++i)
        (*out)->data[i] = pow(A->data[i], exp_val);
    return MATX_OK;
}

// ---- Dense element-wise arithmetic ----

matx_status_t matx_dense_d_i8_add(const matx_alloc_t* alloc,
                                  const matx_dense_d_i8_t A,
                                  const matx_dense_d_i8_t B,
                                  matx_dense_d_i8_t* out)
{
    if (!alloc || !A || !B || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (A->nrows != B->nrows || A->ncols != B->ncols || A->layout != B->layout) {
        MATX_ERROR("%s: dimension/layout mismatch", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_status_t st = matx_dense_d_i8_create(alloc, out, A->layout, A->nrows, A->ncols, NULL);
    if (st != MATX_OK)
        return st;
    const matx_int64_t total = A->nrows * A->ncols;
    for (matx_int64_t i = 0; i < total; ++i)
        (*out)->data[i] = A->data[i] + B->data[i];
    return MATX_OK;
}

matx_status_t matx_dense_d_i8_sub(const matx_alloc_t* alloc,
                                  const matx_dense_d_i8_t A,
                                  const matx_dense_d_i8_t B,
                                  matx_dense_d_i8_t* out)
{
    if (!alloc || !A || !B || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (A->nrows != B->nrows || A->ncols != B->ncols || A->layout != B->layout) {
        MATX_ERROR("%s: dimension/layout mismatch", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_status_t st = matx_dense_d_i8_create(alloc, out, A->layout, A->nrows, A->ncols, NULL);
    if (st != MATX_OK)
        return st;
    const matx_int64_t total = A->nrows * A->ncols;
    for (matx_int64_t i = 0; i < total; ++i)
        (*out)->data[i] = A->data[i] - B->data[i];
    return MATX_OK;
}

matx_status_t matx_dense_d_i8_mul(const matx_alloc_t* alloc,
                                  const matx_dense_d_i8_t A,
                                  const matx_dense_d_i8_t B,
                                  matx_dense_d_i8_t* out)
{
    if (!alloc || !A || !B || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (A->nrows != B->nrows || A->ncols != B->ncols || A->layout != B->layout) {
        MATX_ERROR("%s: dimension/layout mismatch", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_status_t st = matx_dense_d_i8_create(alloc, out, A->layout, A->nrows, A->ncols, NULL);
    if (st != MATX_OK)
        return st;
    const matx_int64_t total = A->nrows * A->ncols;
    for (matx_int64_t i = 0; i < total; ++i)
        (*out)->data[i] = A->data[i] * B->data[i];
    return MATX_OK;
}

matx_status_t matx_dense_d_i8_div(const matx_alloc_t* alloc,
                                  const matx_dense_d_i8_t A,
                                  const matx_dense_d_i8_t B,
                                  matx_dense_d_i8_t* out)
{
    if (!alloc || !A || !B || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (A->nrows != B->nrows || A->ncols != B->ncols || A->layout != B->layout) {
        MATX_ERROR("%s: dimension/layout mismatch", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_status_t st = matx_dense_d_i8_create(alloc, out, A->layout, A->nrows, A->ncols, NULL);
    if (st != MATX_OK)
        return st;
    const matx_int64_t total = A->nrows * A->ncols;
    for (matx_int64_t i = 0; i < total; ++i) {
        if (B->data[i] == 0.0) {
            matx_dense_d_i8_destroy(alloc, *out);
            *out = NULL;
            return MATX_ERR_INVALID_ARG;
        }
        (*out)->data[i] = A->data[i] / B->data[i];
    }
    return MATX_OK;
}

// ---- In-place scalar operations ----

matx_status_t matx_dense_d_i8_add_scalar(matx_dense_d_i8_t m, matx_double val)
{
    if (!m || !m->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const matx_int64_t total = m->nrows * m->ncols;
    for (matx_int64_t i = 0; i < total; ++i)
        m->data[i] += val;
    return MATX_OK;
}

matx_status_t matx_dense_d_i8_mul_scalar(matx_dense_d_i8_t m, matx_double val)
{
    if (!m || !m->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const matx_int64_t total = m->nrows * m->ncols;
    for (matx_int64_t i = 0; i < total; ++i)
        m->data[i] *= val;
    return MATX_OK;
}

// ---- Diagonal matrix creation / extraction ----

matx_status_t matx_diag_d_i8_create(const matx_alloc_t* alloc,
                                    const matx_vec_d_i8_t diag,
                                    matx_dense_d_i8_t* out)
{
    if (!alloc || !diag || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_status_t st = matx_dense_d_i8_create(alloc, out, MATX_COL_MAJOR, diag->n, diag->n, NULL);
    if (st != MATX_OK)
        return st;
    matx_dense_d_i8_zeros(*out);
    for (matx_int64_t i = 0; i < diag->n; ++i)
        (*out)->data[i + i * (*out)->stride] = diag->data[i * diag->stride];
    return MATX_OK;
}

matx_status_t matx_dense_d_i8_get_diag(const matx_alloc_t* alloc,
                                       const matx_dense_d_i8_t A,
                                       matx_vec_d_i8_t* out)
{
    if (!alloc || !A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_int64_t n = A->nrows < A->ncols ? A->nrows : A->ncols;
    matx_status_t st = matx_vec_d_i8_create(alloc, out, NULL, n);
    if (st != MATX_OK)
        return st;
    for (matx_int64_t i = 0; i < n; ++i) {
        matx_int64_t idx = (A->layout == MATX_COL_MAJOR) ? i + i * A->stride : i * A->stride + i;
        (*out)->data[i] = A->data[idx];
    }
    return MATX_OK;
}

// ---- Random number generation ----

static unsigned int g_matx_rand_seed = 0;
static int g_matx_rand_seeded = 0;

static void matx_rand_seed(unsigned int seed)
{
    g_matx_rand_seed = seed;
    g_matx_rand_seeded = 1;
    srand(seed);
}

static matx_double matx_rand_uniform_double(matx_double low, matx_double high)
{
    if (!g_matx_rand_seeded) {
        matx_rand_seed((unsigned int) time(NULL));
    }
    return low + (high - low) * ((matx_double) rand() / (matx_double) RAND_MAX);
}

matx_status_t matx_vec_d_i8_rand_uniform(matx_vec_d_i8_t v,
                                         matx_double low,
                                         matx_double high,
                                         unsigned int seed)
{
    if (!v || !v->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_rand_seed(seed);
    for (matx_int64_t i = 0; i < v->n; ++i)
        v->data[i * v->stride] = matx_rand_uniform_double(low, high);
    return MATX_OK;
}

matx_status_t matx_dense_d_i8_rand_uniform(matx_dense_d_i8_t m,
                                           matx_double low,
                                           matx_double high,
                                           unsigned int seed)
{
    if (!m || !m->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_rand_seed(seed);
    const matx_int64_t total = m->nrows * m->ncols;
    for (matx_int64_t i = 0; i < total; ++i)
        m->data[i] = matx_rand_uniform_double(low, high);
    return MATX_OK;
}

matx_status_t matx_vec_d_i8_rand_normal(matx_vec_d_i8_t v,
                                        matx_double mean,
                                        matx_double stddev,
                                        unsigned int seed)
{
    if (!v || !v->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_rand_seed(seed);
    for (matx_int64_t i = 0; i < v->n; ++i) {
        matx_double u1 = matx_rand_uniform_double(0.0, 1.0);
        matx_double u2 = matx_rand_uniform_double(0.0, 1.0);
        matx_double z = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
        v->data[i * v->stride] = mean + stddev * z;
    }
    return MATX_OK;
}