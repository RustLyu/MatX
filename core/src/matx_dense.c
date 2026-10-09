#include "matx/matx_func.h"
#include "matx/matx_types.h"
#include "matx/matx_types_internal.h"

#include "matx/matx_log.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Dense element-wise operations traverse the contiguous dimension first. */
#define MATX_DENSE_FOR_EACH_2(A, B, OUT, ...)                                \
    do {                                                                     \
        if ((A)->layout == MATX_COL_MAJOR) {                                 \
            if ((A)->stride == (A)->nrows && (B)->stride == (B)->nrows       \
                && (OUT)->stride == (OUT)->nrows) {                          \
                const matx_int64_t count = (A)->nrows * (A)->ncols;          \
                for (matx_int64_t index = 0; index < count; ++index) {       \
                    const matx_int64_t aidx = index;                         \
                    const matx_int64_t bidx = index;                         \
                    const matx_int64_t oidx = index;                         \
                    __VA_ARGS__;                                             \
                }                                                            \
            } else {                                                         \
                for (matx_int64_t j = 0; j < (A)->ncols; ++j)                \
                    for (matx_int64_t i = 0; i < (A)->nrows; ++i) {          \
                        const matx_int64_t aidx = i + j * (A)->stride;       \
                        const matx_int64_t bidx = i + j * (B)->stride;       \
                        const matx_int64_t oidx = i + j * (OUT)->stride;    \
                        __VA_ARGS__;                                         \
                    }                                                        \
            }                                                                \
        } else if ((A)->stride == (A)->ncols && (B)->stride == (B)->ncols   \
                   && (OUT)->stride == (OUT)->ncols) {                      \
            const matx_int64_t count = (A)->nrows * (A)->ncols;              \
            for (matx_int64_t index = 0; index < count; ++index) {           \
                const matx_int64_t aidx = index;                             \
                const matx_int64_t bidx = index;                             \
                const matx_int64_t oidx = index;                             \
                __VA_ARGS__;                                                 \
            }                                                                \
        } else {                                                             \
            for (matx_int64_t i = 0; i < (A)->nrows; ++i)                    \
                for (matx_int64_t j = 0; j < (A)->ncols; ++j) {              \
                    const matx_int64_t aidx = i * (A)->stride + j;           \
                    const matx_int64_t bidx = i * (B)->stride + j;           \
                    const matx_int64_t oidx = i * (OUT)->stride + j;        \
                    __VA_ARGS__;                                             \
                }                                                            \
        }                                                                    \
    } while (0)

#define MATX_DENSE_FOR_EACH_1(A, OUT, ...)                                    \
    do {                                                                     \
        if ((A)->layout == MATX_COL_MAJOR) {                                 \
            if ((A)->stride == (A)->nrows && (OUT)->stride == (OUT)->nrows)  \
            {                                                                \
                const matx_int64_t count = (A)->nrows * (A)->ncols;          \
                for (matx_int64_t index = 0; index < count; ++index) {       \
                    const matx_int64_t aidx = index;                         \
                    const matx_int64_t oidx = index;                         \
                    __VA_ARGS__;                                             \
                }                                                            \
            } else {                                                         \
                for (matx_int64_t j = 0; j < (A)->ncols; ++j)                \
                    for (matx_int64_t i = 0; i < (A)->nrows; ++i) {          \
                        const matx_int64_t aidx = i + j * (A)->stride;       \
                        const matx_int64_t oidx = i + j * (OUT)->stride;    \
                        __VA_ARGS__;                                         \
                    }                                                        \
            }                                                                \
        } else if ((A)->stride == (A)->ncols && (OUT)->stride == (OUT)->ncols) { \
            const matx_int64_t count = (A)->nrows * (A)->ncols;              \
            for (matx_int64_t index = 0; index < count; ++index) {           \
                const matx_int64_t aidx = index;                             \
                const matx_int64_t oidx = index;                             \
                __VA_ARGS__;                                                 \
            }                                                                \
        } else {                                                             \
            for (matx_int64_t i = 0; i < (A)->nrows; ++i)                    \
                for (matx_int64_t j = 0; j < (A)->ncols; ++j) {              \
                    const matx_int64_t aidx = i * (A)->stride + j;           \
                    const matx_int64_t oidx = i * (OUT)->stride + j;        \
                    __VA_ARGS__;                                             \
                }                                                            \
        }                                                                    \
    } while (0)

/* ---- Macro generators for type-agnostic dense matrix functions ---- */

#define MATX_DEF_DENSE_CREATE(PREFIX, OPAQUE, SCA_TYPE)            \
matx_status_t matx_dense_##PREFIX##_create(const matx_alloc_t* alloc,          \
                                           matx_dense_##PREFIX##_t* out,       \
                                           matx_layout_t layout,               \
                                           matx_int64_t rows,                  \
                                           matx_int64_t cols,                  \
                                           SCA_TYPE* data)                     \
{                                                                              \
    if (!out || !alloc || !alloc->malloc_fn || !alloc->free_fn                 \
        || rows <= 0 || cols <= 0) {                                           \
        MATX_ERROR("%s: invalid argument", __func__);                          \
        return MATX_ERR_INVALID_ARG;                                           \
    }                                                                          \
    if ((uint64_t) rows > SIZE_MAX / (uint64_t) cols) {                        \
        MATX_ERROR("%s: matrix size overflow", __func__);                     \
        return MATX_ERR_INVALID_ARG;                                           \
    }                                                                          \
    const size_t element_count = (size_t) rows * (size_t) cols;                \
    if (element_count > SIZE_MAX / sizeof(SCA_TYPE)) {                         \
        MATX_ERROR("%s: matrix byte size overflow", __func__);                \
        return MATX_ERR_INVALID_ARG;                                           \
    }                                                                          \
    const size_t data_bytes = element_count * sizeof(SCA_TYPE);                \
    if (layout != MATX_COL_MAJOR && layout != MATX_ROW_MAJOR) {                \
        MATX_ERROR("%s: invalid argument", __func__);                          \
        return MATX_ERR_INVALID_ARG;                                           \
    }                                                                          \
    OPAQUE* out_value = matx_malloc(alloc, sizeof(OPAQUE));                    \
    if (!out_value) {                                                          \
        MATX_ERROR("%s: out of memory", __func__);                             \
        return MATX_ERR_OUT_OF_MEMORY;                                         \
    }                                                                          \
    memset(out_value, 0, sizeof(*out_value));                                  \
    out_value->nrows = rows;                                                   \
    out_value->ncols = cols;                                                   \
    out_value->layout = layout;                                                \
    out_value->stride = (layout == MATX_COL_MAJOR) ? rows : cols;              \
    out_value->flags = 1u;                                                     \
    out_value->alloc = *alloc;                                                 \
    out_value->data = (SCA_TYPE*) matx_malloc(alloc, data_bytes);              \
    if (!out_value->data) {                                                    \
        matx_free(alloc, out_value);                                           \
        MATX_ERROR("%s: out of memory", __func__);                             \
        return MATX_ERR_OUT_OF_MEMORY;                                         \
    }                                                                          \
    if (data != NULL) {                                                        \
        memcpy(out_value->data, data, data_bytes);                             \
    }                                                                          \
    if (*out != NULL) {                                                        \
        matx_dense_##PREFIX##_destroy(alloc, *out);                             \
    }                                                                          \
    *out = out_value;                                                          \
    return MATX_OK;                                                            \
}

#define MATX_DEF_DENSE_DUP(PREFIX, OPAQUE, SCA_TYPE)                \
matx_status_t matx_dense_##PREFIX##_dup(const matx_alloc_t* alloc,             \
                                        const matx_dense_##PREFIX##_t in,       \
                                        matx_dense_##PREFIX##_t* out)           \
{                                                                              \
    if (!in || !in->data || !out || in->nrows <= 0 || in->ncols <= 0          \
        || in->stride <= 0) {                                                  \
        MATX_ERROR("%s: invalid argument", __func__);                         \
        return MATX_ERR_INVALID_ARG;                                           \
    }                                                                          \
    matx_dense_##PREFIX##_t copy = NULL;                                       \
    matx_status_t st = matx_dense_##PREFIX##_create(                           \
        alloc, &copy, in->layout, in->nrows, in->ncols, NULL);                  \
    if (st != MATX_OK) return st;                                              \
    const matx_int64_t minor = (in->layout == MATX_COL_MAJOR)                   \
        ? in->nrows : in->ncols;                                                \
    if (in->stride == minor) {                                                 \
        const size_t data_bytes = (size_t) in->nrows                           \
            * (size_t) in->ncols * sizeof(SCA_TYPE);                           \
        memcpy(copy->data, in->data, data_bytes);                              \
    } else {                                                                   \
        MATX_DENSE_FOR_EACH_1(in, copy,                                         \
            copy->data[oidx] = in->data[aidx]);                                \
    }                                                                          \
    if (*out != NULL) matx_dense_##PREFIX##_destroy(alloc, *out);              \
    *out = copy;                                                               \
    return MATX_OK;                                                            \
}

#define MATX_DEF_DENSE_WRAP(PREFIX, OPAQUE, SCA_TYPE)               \
matx_status_t matx_dense_##PREFIX##_wrap(const matx_alloc_t* alloc,            \
                                          matx_dense_##PREFIX##_t* out,         \
                                          matx_int64_t rows,                    \
                                          matx_int64_t cols,                    \
                                          matx_int64_t stride,                  \
                                          matx_layout_t layout,                 \
                                          SCA_TYPE* data)                       \
{                                                                              \
    if (!out || !alloc || !alloc->malloc_fn || !alloc->free_fn || !data        \
        || rows <= 0 || cols <= 0) {                                          \
        MATX_ERROR("%s: invalid argument", __func__);                          \
        return MATX_ERR_INVALID_ARG;                                           \
    }                                                                          \
    if (layout != MATX_COL_MAJOR && layout != MATX_ROW_MAJOR) {                \
        MATX_ERROR("%s: invalid argument", __func__);                          \
        return MATX_ERR_INVALID_ARG;                                           \
    }                                                                          \
    const matx_int64_t major = (layout == MATX_COL_MAJOR) ? cols : rows;        \
    const matx_int64_t minor = (layout == MATX_COL_MAJOR) ? rows : cols;        \
    if (layout == MATX_COL_MAJOR) {                                            \
        if (stride < rows) {                                                   \
            MATX_ERROR("%s: invalid argument", __func__);                      \
            return MATX_ERR_INVALID_ARG;                                       \
        }                                                                      \
    } else {                                                                   \
        if (stride < cols) {                                                   \
            MATX_ERROR("%s: invalid argument", __func__);                      \
            return MATX_ERR_INVALID_ARG;                                       \
        }                                                                      \
    }                                                                          \
    const size_t max_elements = SIZE_MAX / sizeof(SCA_TYPE);                    \
    if ((uint64_t) minor > max_elements || (uint64_t) stride > max_elements    \
        || (uint64_t) (major - 1) > (max_elements - (size_t) minor)            \
                                      / (size_t) stride) {                      \
        MATX_ERROR("%s: wrapped matrix size overflow", __func__);             \
        return MATX_ERR_INVALID_ARG;                                           \
    }                                                                          \
    OPAQUE* out_value = matx_malloc(alloc, sizeof(OPAQUE));                    \
    if (!out_value) {                                                          \
        MATX_ERROR("%s: out of memory", __func__);                             \
        return MATX_ERR_OUT_OF_MEMORY;                                         \
    }                                                                          \
    memset(out_value, 0, sizeof(*out_value));                                  \
    out_value->nrows = rows;                                                   \
    out_value->ncols = cols;                                                   \
    out_value->stride = stride;                                                \
    out_value->layout = layout;                                                \
    out_value->data = data;                                                    \
    out_value->flags = 0u;                                                     \
    out_value->alloc = *alloc;                                                 \
    if (*out != NULL) {                                                        \
        matx_dense_##PREFIX##_destroy(alloc, *out);                             \
    }                                                                          \
    *out = out_value;                                                          \
    return MATX_OK;                                                            \
}

#define MATX_DEF_DENSE_DESTROY(PREFIX, OPAQUE, SCA_TYPE)            \
void matx_dense_##PREFIX##_destroy(const matx_alloc_t* alloc,                   \
                                    matx_dense_##PREFIX##_t m)                  \
{                                                                              \
    if (!m)                                                                    \
        return;                                                                \
    const matx_alloc_t* object_alloc = m->alloc.free_fn ? &m->alloc : alloc;   \
    if ((m->flags & 1u) != 0u && m->data && object_alloc) {                   \
        matx_free(object_alloc, m->data);                                     \
        m->data = NULL;                                                        \
    }                                                                          \
    matx_handles_destroy(m->backend_handles, &m->num_backend_handles);         \
    matx_free(object_alloc, m);                                                \
}

#define MATX_DEF_DENSE_FILL(PREFIX, OPAQUE, SCA_TYPE)               \
matx_status_t matx_dense_##PREFIX##_fill(matx_dense_##PREFIX##_t m,             \
                                          SCA_TYPE val)                         \
{                                                                              \
    if (!m || !m->data) {                                                      \
        MATX_ERROR("%s: invalid argument", __func__);                          \
        return MATX_ERR_INVALID_ARG;                                           \
    }                                                                          \
    const matx_int64_t minor = (m->layout == MATX_COL_MAJOR)                    \
        ? m->nrows : m->ncols;                                                  \
    if (m->stride == minor) {                                                   \
        const size_t count = (size_t) m->nrows * (size_t) m->ncols;             \
        for (size_t i = 0; i < count; ++i)                                      \
            m->data[i] = val;                                                    \
        return MATX_OK;                                                         \
    }                                                                          \
    if (m->layout == MATX_COL_MAJOR) {                                         \
        for (matx_int64_t j = 0; j < m->ncols; ++j)                            \
            for (matx_int64_t i = 0; i < m->nrows; ++i)                        \
                m->data[i + j * m->stride] = val;                              \
    } else {                                                                   \
        for (matx_int64_t i = 0; i < m->nrows; ++i)                            \
            for (matx_int64_t j = 0; j < m->ncols; ++j)                        \
                m->data[i * m->stride + j] = val;                              \
    }                                                                          \
    return MATX_OK;                                                            \
}

#define MATX_DEF_DENSE_ZEROS(PREFIX, OPAQUE, SCA_TYPE)              \
matx_status_t matx_dense_##PREFIX##_zeros(matx_dense_##PREFIX##_t m)            \
{                                                                              \
    if (!m || !m->data) {                                                      \
        MATX_ERROR("%s: invalid argument", __func__);                          \
        return MATX_ERR_INVALID_ARG;                                           \
    }                                                                          \
    const matx_int64_t minor = (m->layout == MATX_COL_MAJOR)                    \
        ? m->nrows : m->ncols;                                                  \
    if (m->stride == minor) {                                                   \
        memset(m->data, 0, (size_t) m->nrows * (size_t) m->ncols               \
               * sizeof(SCA_TYPE));                                             \
        return MATX_OK;                                                         \
    }                                                                          \
    if (m->layout == MATX_COL_MAJOR) {                                         \
        for (matx_int64_t j = 0; j < m->ncols; ++j)                            \
            memset(m->data + j * m->stride, 0,                                  \
                   (size_t) m->nrows * sizeof(SCA_TYPE));                       \
    } else {                                                                   \
        for (matx_int64_t i = 0; i < m->nrows; ++i)                            \
            memset(m->data + i * m->stride, 0,                                  \
                   (size_t) m->ncols * sizeof(SCA_TYPE));                       \
    }                                                                          \
    return MATX_OK;                                                            \
}

#define MATX_DEF_DIAG_CREATE(PREFIX, OPAQUE, SCA_TYPE)              \
matx_status_t matx_diag_##PREFIX##_create(const matx_alloc_t* alloc,            \
                                           const matx_vec_##PREFIX##_t diag,    \
                                           matx_dense_##PREFIX##_t* out)        \
{                                                                              \
    if (!alloc || !diag || !out) {                                             \
        MATX_ERROR("%s: invalid argument", __func__);                          \
        return MATX_ERR_INVALID_ARG;                                           \
    }                                                                          \
    matx_status_t st = matx_dense_##PREFIX##_create(alloc, out,                 \
                                                     MATX_COL_MAJOR,            \
                                                     diag->n, diag->n, NULL);   \
    if (st != MATX_OK)                                                         \
        return st;                                                             \
    matx_dense_##PREFIX##_zeros(*out);                                          \
    for (matx_int64_t i = 0; i < diag->n; ++i)                                 \
        (*out)->data[i + i * (*out)->stride] = diag->data[i * diag->stride];   \
    return MATX_OK;                                                            \
}

#define MATX_DEF_GET_DIAG(PREFIX, OPAQUE, SCA_TYPE)                 \
matx_status_t matx_dense_##PREFIX##_get_diag(const matx_alloc_t* alloc,         \
                                              const matx_dense_##PREFIX##_t A,  \
                                              matx_vec_##PREFIX##_t* out)       \
{                                                                              \
    if (!alloc || !A || !out) {                                                \
        MATX_ERROR("%s: invalid argument", __func__);                          \
        return MATX_ERR_INVALID_ARG;                                           \
    }                                                                          \
    matx_int64_t n = A->nrows < A->ncols ? A->nrows : A->ncols;                \
    matx_status_t st = matx_vec_##PREFIX##_create(alloc, out, NULL, n);         \
    if (st != MATX_OK)                                                         \
        return st;                                                             \
    for (matx_int64_t i = 0; i < n; ++i) {                                     \
        matx_int64_t idx = (A->layout == MATX_COL_MAJOR)                        \
                            ? i + i * A->stride                                 \
                            : i * A->stride + i;                                \
        (*out)->data[i] = A->data[idx];                                        \
    }                                                                          \
    return MATX_OK;                                                            \
}

#define MATX_DEF_GET_BLOCK(PREFIX, OPAQUE, SCA_TYPE)                         \
matx_status_t matx_dense_##PREFIX##_get_block(const matx_alloc_t* alloc,       \
                                               const matx_dense_##PREFIX##_t A,\
                                               matx_int64_t rs,                \
                                               matx_int64_t re,                \
                                               matx_int64_t cs,                \
                                               matx_int64_t ce,                \
                                               matx_dense_##PREFIX##_t* out)   \
{                                                                              \
    if (!alloc || !A || !out || rs < 0 || re <= rs || cs < 0 || ce <= cs       \
        || re > A->nrows || ce > A->ncols) {                                   \
        MATX_ERROR("%s: invalid argument", __func__);                          \
        return MATX_ERR_INVALID_ARG;                                           \
    }                                                                          \
    const matx_int64_t nrows_out = re - rs;                                    \
    const matx_int64_t ncols_out = ce - cs;                                    \
    matx_status_t st = matx_dense_##PREFIX##_create(alloc, out,                 \
                                                     A->layout,                 \
                                                     nrows_out, ncols_out,      \
                                                     NULL);                     \
    if (st != MATX_OK)                                                         \
        return st;                                                             \
    if (A->layout == MATX_COL_MAJOR) {                                         \
        for (matx_int64_t j = 0; j < ncols_out; ++j)                           \
            for (matx_int64_t i = 0; i < nrows_out; ++i) {                     \
                const matx_int64_t src_idx = (rs + i) + (cs + j) * A->stride;  \
                const matx_int64_t dst_idx = i + j * (*out)->stride;           \
                (*out)->data[dst_idx] = A->data[src_idx];                      \
            }                                                                  \
    } else {                                                                   \
        for (matx_int64_t i = 0; i < nrows_out; ++i)                           \
            for (matx_int64_t j = 0; j < ncols_out; ++j) {                     \
                const matx_int64_t src_idx = (rs + i) * A->stride + cs + j;    \
                const matx_int64_t dst_idx = i * (*out)->stride + j;           \
                (*out)->data[dst_idx] = A->data[src_idx];                      \
            }                                                                  \
    }                                                                          \
    return MATX_OK;                                                            \
}

#define MATX_DEF_SET_BLOCK(PREFIX, OPAQUE, SCA_TYPE)                         \
matx_status_t matx_dense_##PREFIX##_set_block(                                 \
    const matx_dense_##PREFIX##_t A,                                           \
    matx_int64_t rs, matx_int64_t re,                                          \
    matx_int64_t cs, matx_int64_t ce,                                          \
    matx_dense_##PREFIX##_t B,                                                 \
    matx_int64_t dr, matx_int64_t dc)                                          \
{                                                                              \
    if (!A || !B || rs < 0 || re <= rs || cs < 0 || ce <= cs                   \
        || dr < 0 || dc < 0) {                                                 \
        MATX_ERROR("%s: invalid argument", __func__);                          \
        return MATX_ERR_INVALID_ARG;                                           \
    }                                                                          \
    const matx_int64_t nrows_blk = re - rs;                                    \
    const matx_int64_t ncols_blk = ce - cs;                                    \
    if (re > A->nrows || ce > A->ncols                                         \
        || nrows_blk > B->nrows || ncols_blk > B->ncols                        \
        || dr > B->nrows - nrows_blk || dc > B->ncols - ncols_blk) {           \
        MATX_ERROR("%s: block out of bounds", __func__);                       \
        return MATX_ERR_INVALID_ARG;                                           \
    }                                                                          \
    if (A->layout != B->layout) {                                              \
        MATX_ERROR("%s: layout mismatch", __func__);                           \
        return MATX_ERR_INVALID_ARG;                                           \
    }                                                                          \
    if (A->layout == MATX_COL_MAJOR) {                                         \
        for (matx_int64_t j = 0; j < ncols_blk; ++j)                           \
            for (matx_int64_t i = 0; i < nrows_blk; ++i) {                     \
                const matx_int64_t src_idx = (rs + i) + (cs + j) * A->stride;  \
                const matx_int64_t dst_idx = (dr + i) + (dc + j) * B->stride;  \
                B->data[dst_idx] = A->data[src_idx];                           \
            }                                                                  \
    } else {                                                                   \
        for (matx_int64_t i = 0; i < nrows_blk; ++i)                           \
            for (matx_int64_t j = 0; j < ncols_blk; ++j) {                     \
                const matx_int64_t src_idx = (rs + i) * A->stride + cs + j;    \
                const matx_int64_t dst_idx = (dr + i) * B->stride + dc + j;    \
                B->data[dst_idx] = A->data[src_idx];                           \
            }                                                                  \
    }                                                                          \
    return MATX_OK;                                                            \
}

/* ---- Type-agnostic expansions ---- */

MATX_DEF_DENSE_CREATE(d_i8, matx_dense_d_i8_opaque_t, matx_double)
MATX_DEF_DENSE_CREATE(z_i8, matx_dense_z_i8_opaque_t, matx_complex_d_t)
MATX_DEF_DENSE_DUP(d_i8, matx_dense_d_i8_opaque_t, matx_double)
MATX_DEF_DENSE_DUP(z_i8, matx_dense_z_i8_opaque_t, matx_complex_d_t)
MATX_DEF_DENSE_WRAP(d_i8, matx_dense_d_i8_opaque_t, matx_double)
MATX_DEF_DENSE_WRAP(z_i8, matx_dense_z_i8_opaque_t, matx_complex_d_t)
MATX_DEF_DENSE_DESTROY(d_i8, matx_dense_d_i8_opaque_t, matx_double)
MATX_DEF_DENSE_DESTROY(z_i8, matx_dense_z_i8_opaque_t, matx_complex_d_t)
MATX_DEF_DENSE_FILL(d_i8, matx_dense_d_i8_opaque_t, matx_double)
MATX_DEF_DENSE_FILL(z_i8, matx_dense_z_i8_opaque_t, matx_complex_d_t)
MATX_DEF_DENSE_ZEROS(d_i8, matx_dense_d_i8_opaque_t, matx_double)
MATX_DEF_DENSE_ZEROS(z_i8, matx_dense_z_i8_opaque_t, matx_complex_d_t)
MATX_DEF_DIAG_CREATE(d_i8, matx_dense_d_i8_opaque_t, matx_double)
MATX_DEF_DIAG_CREATE(z_i8, matx_dense_z_i8_opaque_t, matx_complex_d_t)
MATX_DEF_GET_DIAG(d_i8, matx_dense_d_i8_opaque_t, matx_double)
MATX_DEF_GET_DIAG(z_i8, matx_dense_z_i8_opaque_t, matx_complex_d_t)
MATX_DEF_GET_BLOCK(d_i8, matx_dense_d_i8_opaque_t, matx_double)
MATX_DEF_GET_BLOCK(z_i8, matx_dense_z_i8_opaque_t, matx_complex_d_t)
MATX_DEF_SET_BLOCK(d_i8, matx_dense_d_i8_opaque_t, matx_double)
MATX_DEF_SET_BLOCK(z_i8, matx_dense_z_i8_opaque_t, matx_complex_d_t)

#undef MATX_DEF_DENSE_CREATE
#undef MATX_DEF_DENSE_DUP
#undef MATX_DEF_DENSE_WRAP
#undef MATX_DEF_DENSE_DESTROY
#undef MATX_DEF_DENSE_FILL
#undef MATX_DEF_DENSE_ZEROS
#undef MATX_DEF_DIAG_CREATE
#undef MATX_DEF_GET_DIAG
#undef MATX_DEF_GET_BLOCK
#undef MATX_DEF_SET_BLOCK

/* ---- Single-type: ones (real only) ---- */

matx_status_t matx_dense_d_i8_ones(matx_dense_d_i8_t m)
{
    return matx_dense_d_i8_fill(m, 1.0);
}

/* ---- Trace (structurally different scalar vs struct) ---- */

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

/* ---- Type conversion ---- */

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
    MATX_DENSE_FOR_EACH_1(A, *out,
        (*out)->data[oidx].real = A->data[aidx];
        (*out)->data[oidx].imag = 0.0);
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

/* ---- Element-wise math: dense real ---- */

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
    MATX_DENSE_FOR_EACH_1(A, *out,
        (*out)->data[oidx] = exp(A->data[aidx]));
    return MATX_OK;
}

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
    MATX_DENSE_FOR_EACH_1(A, *out,
        matx_double a = A->data[aidx].real;
        matx_double b = A->data[aidx].imag;
        matx_double e = exp(a);
        (*out)->data[oidx].real = e * cos(b);
        (*out)->data[oidx].imag = e * sin(b));
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
    MATX_DENSE_FOR_EACH_1(A, *out,
        if (A->data[aidx] <= 0.0) {
            matx_dense_d_i8_destroy(alloc, *out);
            *out = NULL;
            return MATX_ERR_INVALID_ARG;
        }
        (*out)->data[oidx] = log(A->data[aidx]));
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
    MATX_DENSE_FOR_EACH_1(A, *out,
        matx_double a = A->data[aidx].real;
        matx_double b = A->data[aidx].imag;
        (*out)->data[oidx].real = 0.5 * log(a * a + b * b);
        (*out)->data[oidx].imag = atan2(b, a));
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
    MATX_DENSE_FOR_EACH_1(A, *out,
        if (A->data[aidx] < 0.0) {
            matx_dense_d_i8_destroy(alloc, *out);
            *out = NULL;
            return MATX_ERR_INVALID_ARG;
        }
        (*out)->data[oidx] = sqrt(A->data[aidx]));
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
    MATX_DENSE_FOR_EACH_1(A, *out,
        matx_double a = A->data[aidx].real;
        matx_double b = A->data[aidx].imag;
        matx_double mag = sqrt(a * a + b * b);
        matx_double re = sqrt((mag + a) * 0.5);
        matx_double im = sqrt((mag - a) * 0.5);
        if (b < 0.0)
            im = -im;
        (*out)->data[oidx].real = re;
        (*out)->data[oidx].imag = im);
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
    MATX_DENSE_FOR_EACH_1(A, *out,
        (*out)->data[oidx] = sin(A->data[aidx]));
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
    MATX_DENSE_FOR_EACH_1(A, *out,
        matx_double a = A->data[aidx].real;
        matx_double b = A->data[aidx].imag;
        (*out)->data[oidx].real = sin(a) * cosh(b);
        (*out)->data[oidx].imag = cos(a) * sinh(b));
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
    MATX_DENSE_FOR_EACH_1(A, *out,
        (*out)->data[oidx] = cos(A->data[aidx]));
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
    MATX_DENSE_FOR_EACH_1(A, *out,
        matx_double a = A->data[aidx].real;
        matx_double b = A->data[aidx].imag;
        (*out)->data[oidx].real = cos(a) * cosh(b);
        (*out)->data[oidx].imag = -sin(a) * sinh(b));
    return MATX_OK;
}

matx_status_t matx_dense_d_i8_tan(const matx_alloc_t* alloc,
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
    MATX_DENSE_FOR_EACH_1(A, *out,
        (*out)->data[oidx] = tan(A->data[aidx]));
    return MATX_OK;
}

matx_status_t matx_dense_z_i8_tan(const matx_alloc_t* alloc,
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
    MATX_DENSE_FOR_EACH_1(A, *out,
        matx_double a = A->data[aidx].real;
        matx_double b = A->data[aidx].imag;
        matx_double sA = sin(a) * cosh(b);
        matx_double sB = cos(a) * sinh(b);
        matx_double cC = cos(a) * cosh(b);
        matx_double cD = -sin(a) * sinh(b);
        matx_double den = cC * cC + cD * cD;
        (*out)->data[oidx].real = (sA * cC + sB * cD) / den;
        (*out)->data[oidx].imag = (sB * cC - sA * cD) / den);
    return MATX_OK;
}

matx_status_t matx_dense_d_i8_asin(const matx_alloc_t* alloc,
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
    MATX_DENSE_FOR_EACH_1(A, *out,
        matx_double x = A->data[aidx];
        if (x < -1.0 || x > 1.0) {
            matx_dense_d_i8_destroy(alloc, *out);
            *out = NULL;
            return MATX_ERR_INVALID_ARG;
        }
        (*out)->data[oidx] = asin(x));
    return MATX_OK;
}

matx_status_t matx_dense_z_i8_asin(const matx_alloc_t* alloc,
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
    MATX_DENSE_FOR_EACH_1(A, *out,
        matx_double a = A->data[aidx].real;
        matx_double b = A->data[aidx].imag;
        matx_double p = 1.0 - a * a + b * b;
        matx_double q = -2.0 * a * b;
        matx_double mag = sqrt(p * p + q * q);
        matx_double re_s = sqrt((mag + p) * 0.5);
        matx_double im_s = (q >= 0.0 ? 1.0 : -1.0) * sqrt((mag - p) * 0.5);
        matx_double X = re_s - b;
        matx_double Y = im_s + a;
        (*out)->data[oidx].real = atan2(Y, X);
        (*out)->data[oidx].imag = -0.5 * log(X * X + Y * Y));
    return MATX_OK;
}

matx_status_t matx_dense_d_i8_acos(const matx_alloc_t* alloc,
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
    MATX_DENSE_FOR_EACH_1(A, *out,
        matx_double x = A->data[aidx];
        if (x < -1.0 || x > 1.0) {
            matx_dense_d_i8_destroy(alloc, *out);
            *out = NULL;
            return MATX_ERR_INVALID_ARG;
        }
        (*out)->data[oidx] = acos(x));
    return MATX_OK;
}

matx_status_t matx_dense_z_i8_acos(const matx_alloc_t* alloc,
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
    MATX_DENSE_FOR_EACH_1(A, *out,
        matx_double a = A->data[aidx].real;
        matx_double b = A->data[aidx].imag;
        matx_double p = 1.0 - a * a + b * b;
        matx_double q = -2.0 * a * b;
        matx_double mag = sqrt(p * p + q * q);
        matx_double re_s = sqrt((mag + p) * 0.5);
        matx_double im_s = (q >= 0.0 ? 1.0 : -1.0) * sqrt((mag - p) * 0.5);
        matx_double X = re_s - b;
        matx_double Y = im_s + a;
        (*out)->data[oidx].real = 1.5707963267948966 - atan2(Y, X);
        (*out)->data[oidx].imag = 0.5 * log(X * X + Y * Y));
    return MATX_OK;
}

matx_status_t matx_dense_d_i8_atan(const matx_alloc_t* alloc,
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
    MATX_DENSE_FOR_EACH_1(A, *out,
        (*out)->data[oidx] = atan(A->data[aidx]));
    return MATX_OK;
}

matx_status_t matx_dense_z_i8_atan(const matx_alloc_t* alloc,
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
    MATX_DENSE_FOR_EACH_1(A, *out,
        matx_double a = A->data[aidx].real;
        matx_double b = A->data[aidx].imag;
        matx_double den = a * a + (1.0 - b) * (1.0 - b);
        matx_double re_div = (1.0 - a * a - b * b) / den;
        matx_double im_div = -2.0 * a / den;
        (*out)->data[oidx].real = -0.5 * atan2(im_div, re_div);
        (*out)->data[oidx].imag = 0.25 * log(re_div * re_div + im_div * im_div));
    return MATX_OK;
}

matx_status_t matx_dense_d_i8_sinh(const matx_alloc_t* alloc,
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
    MATX_DENSE_FOR_EACH_1(A, *out,
        (*out)->data[oidx] = sinh(A->data[aidx]));
    return MATX_OK;
}

matx_status_t matx_dense_z_i8_sinh(const matx_alloc_t* alloc,
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
    MATX_DENSE_FOR_EACH_1(A, *out,
        matx_double a = A->data[aidx].real;
        matx_double b = A->data[aidx].imag;
        (*out)->data[oidx].real = sinh(a) * cos(b);
        (*out)->data[oidx].imag = cosh(a) * sin(b));
    return MATX_OK;
}

matx_status_t matx_dense_d_i8_cosh(const matx_alloc_t* alloc,
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
    MATX_DENSE_FOR_EACH_1(A, *out,
        (*out)->data[oidx] = cosh(A->data[aidx]));
    return MATX_OK;
}

matx_status_t matx_dense_z_i8_cosh(const matx_alloc_t* alloc,
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
    MATX_DENSE_FOR_EACH_1(A, *out,
        matx_double a = A->data[aidx].real;
        matx_double b = A->data[aidx].imag;
        (*out)->data[oidx].real = cosh(a) * cos(b);
        (*out)->data[oidx].imag = sinh(a) * sin(b));
    return MATX_OK;
}

matx_status_t matx_dense_d_i8_tanh(const matx_alloc_t* alloc,
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
    MATX_DENSE_FOR_EACH_1(A, *out,
        (*out)->data[oidx] = tanh(A->data[aidx]));
    return MATX_OK;
}

matx_status_t matx_dense_z_i8_tanh(const matx_alloc_t* alloc,
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
    MATX_DENSE_FOR_EACH_1(A, *out,
        matx_double a = A->data[aidx].real;
        matx_double b = A->data[aidx].imag;
        matx_double sA = sinh(a) * cos(b);
        matx_double sB = cosh(a) * sin(b);
        matx_double cC = cosh(a) * cos(b);
        matx_double cD = sinh(a) * sin(b);
        matx_double den = cC * cC + cD * cD;
        (*out)->data[oidx].real = (sA * cC + sB * cD) / den;
        (*out)->data[oidx].imag = (sB * cC - sA * cD) / den);
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
    MATX_DENSE_FOR_EACH_1(A, *out,
        (*out)->data[oidx] = fabs(A->data[aidx]));
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
    MATX_DENSE_FOR_EACH_1(A, *out,
        matx_double a = A->data[aidx].real;
        matx_double b = A->data[aidx].imag;
        (*out)->data[oidx] = sqrt(a * a + b * b));
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
    MATX_DENSE_FOR_EACH_1(A, *out,
        (*out)->data[oidx] = pow(A->data[aidx], exp_val));
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
    MATX_DENSE_FOR_EACH_1(A, *out,
        matx_double a = A->data[aidx].real;
        matx_double b = A->data[aidx].imag;
        matx_double r = sqrt(a * a + b * b);
        matx_double theta = atan2(b, a);
        matx_double c = exp_val.real;
        matx_double d = exp_val.imag;
        matx_double new_r = pow(r, c) * exp(-d * theta);
        matx_double new_theta = c * theta + d * log(r);
        (*out)->data[oidx].real = new_r * cos(new_theta);
        (*out)->data[oidx].imag = new_r * sin(new_theta));
    return MATX_OK;
}

/* ---- Dense element-wise arithmetic ---- */

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
    MATX_DENSE_FOR_EACH_2(A, B, *out,
        (*out)->data[oidx] = A->data[aidx] + B->data[bidx]);
    return MATX_OK;
}

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
    MATX_DENSE_FOR_EACH_2(A, B, *out,
        (*out)->data[oidx].real = A->data[aidx].real + B->data[bidx].real;
        (*out)->data[oidx].imag = A->data[aidx].imag + B->data[bidx].imag);
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
    MATX_DENSE_FOR_EACH_2(A, B, *out,
        (*out)->data[oidx] = A->data[aidx] - B->data[bidx]);
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
    MATX_DENSE_FOR_EACH_2(A, B, *out,
        (*out)->data[oidx].real = A->data[aidx].real - B->data[bidx].real;
        (*out)->data[oidx].imag = A->data[aidx].imag - B->data[bidx].imag);
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
    MATX_DENSE_FOR_EACH_2(A, B, *out,
        (*out)->data[oidx] = A->data[aidx] * B->data[bidx]);
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
    MATX_DENSE_FOR_EACH_2(A, B, *out,
        const matx_double ar = A->data[aidx].real;
        const matx_double ai = A->data[aidx].imag;
        const matx_double br = B->data[bidx].real;
        const matx_double bi = B->data[bidx].imag;
        (*out)->data[oidx].real = ar * br - ai * bi;
        (*out)->data[oidx].imag = ar * bi + ai * br);
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
    MATX_DENSE_FOR_EACH_2(A, B, *out,
        if (B->data[bidx] == 0.0) {
            matx_dense_d_i8_destroy(alloc, *out);
            *out = NULL;
            return MATX_ERR_INVALID_ARG;
        }
        (*out)->data[oidx] = A->data[aidx] / B->data[bidx]);
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
    MATX_DENSE_FOR_EACH_2(A, B, *out,
        const matx_double ar = A->data[aidx].real;
        const matx_double ai = A->data[aidx].imag;
        const matx_double br = B->data[bidx].real;
        const matx_double bi = B->data[bidx].imag;
        const matx_double den = br * br + bi * bi;
        if (den == 0.0) {
            matx_dense_z_i8_destroy(alloc, *out);
            *out = NULL;
            return MATX_ERR_INVALID_ARG;
        }
        (*out)->data[oidx].real = (ar * br + ai * bi) / den;
        (*out)->data[oidx].imag = (ai * br - ar * bi) / den);
    return MATX_OK;
}

/* ---- In-place scalar operations ---- */

matx_status_t matx_dense_d_i8_add_scalar(matx_dense_d_i8_t m, matx_double val)
{
    if (!m || !m->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (m->layout == MATX_COL_MAJOR) {
        for (matx_int64_t j = 0; j < m->ncols; ++j) {
            matx_double* column = m->data + j * m->stride;
            for (matx_int64_t i = 0; i < m->nrows; ++i)
                column[i] += val;
        }
    } else {
        for (matx_int64_t i = 0; i < m->nrows; ++i) {
            matx_double* row = m->data + i * m->stride;
            for (matx_int64_t j = 0; j < m->ncols; ++j)
                row[j] += val;
        }
    }
    return MATX_OK;
}

matx_status_t matx_dense_z_i8_add_scalar(matx_dense_z_i8_t m, matx_complex_d_t val)
{
    if (!m || !m->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (m->layout == MATX_COL_MAJOR) {
        for (matx_int64_t j = 0; j < m->ncols; ++j) {
            matx_complex_d_t* column = m->data + j * m->stride;
            for (matx_int64_t i = 0; i < m->nrows; ++i) {
                column[i].real += val.real;
                column[i].imag += val.imag;
            }
        }
    } else {
        for (matx_int64_t i = 0; i < m->nrows; ++i) {
            matx_complex_d_t* row = m->data + i * m->stride;
            for (matx_int64_t j = 0; j < m->ncols; ++j) {
                row[j].real += val.real;
                row[j].imag += val.imag;
            }
        }
    }
    return MATX_OK;
}

matx_status_t matx_dense_d_i8_mul_scalar(matx_dense_d_i8_t m, matx_double val)
{
    if (!m || !m->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (m->layout == MATX_COL_MAJOR) {
        for (matx_int64_t j = 0; j < m->ncols; ++j) {
            matx_double* column = m->data + j * m->stride;
            for (matx_int64_t i = 0; i < m->nrows; ++i)
                column[i] *= val;
        }
    } else {
        for (matx_int64_t i = 0; i < m->nrows; ++i) {
            matx_double* row = m->data + i * m->stride;
            for (matx_int64_t j = 0; j < m->ncols; ++j)
                row[j] *= val;
        }
    }
    return MATX_OK;
}

matx_status_t matx_dense_z_i8_mul_scalar(matx_dense_z_i8_t m, matx_complex_d_t val)
{
    if (!m || !m->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (m->layout == MATX_COL_MAJOR) {
        for (matx_int64_t j = 0; j < m->ncols; ++j) {
            matx_complex_d_t* column = m->data + j * m->stride;
            for (matx_int64_t i = 0; i < m->nrows; ++i) {
                const matx_double re = column[i].real;
                const matx_double im = column[i].imag;
                column[i].real = re * val.real - im * val.imag;
                column[i].imag = re * val.imag + im * val.real;
            }
        }
    } else {
        for (matx_int64_t i = 0; i < m->nrows; ++i) {
            matx_complex_d_t* row = m->data + i * m->stride;
            for (matx_int64_t j = 0; j < m->ncols; ++j) {
                const matx_double re = row[j].real;
                const matx_double im = row[j].imag;
                row[j].real = re * val.real - im * val.imag;
                row[j].imag = re * val.imag + im * val.real;
            }
        }
    }
    return MATX_OK;
}

/* ---- Random number generation ---- */

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

/* SplitMix64 keeps seeded uniform fills local to the call and avoids rand()'s
 * shared global state and per-sample locking. */
static inline uint64_t matx_rand_next_u64(uint64_t* state)
{
    uint64_t value = (*state += UINT64_C(0x9e3779b97f4a7c15));
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

static inline matx_double matx_rand_uniform_fast(matx_double low,
                                                 matx_double high,
                                                 uint64_t* state)
{
    const matx_double unit
        = (matx_double) (matx_rand_next_u64(state) >> 11) * 0x1.0p-53;
    return low + (high - low) * unit;
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
    uint64_t state = (uint64_t) seed;
    for (matx_int64_t i = 0; i < v->n; ++i)
        v->data[i * v->stride] = matx_rand_uniform_fast(low, high, &state);
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
    uint64_t state = (uint64_t) seed;
    const matx_int64_t minor
        = (m->layout == MATX_COL_MAJOR) ? m->nrows : m->ncols;
    if (m->stride == minor) {
        const matx_int64_t count = m->nrows * m->ncols;
        for (matx_int64_t i = 0; i < count; ++i)
            m->data[i] = matx_rand_uniform_fast(low, high, &state);
    } else if (m->layout == MATX_COL_MAJOR) {
        for (matx_int64_t j = 0; j < m->ncols; ++j) {
            matx_double* column = m->data + j * m->stride;
            for (matx_int64_t i = 0; i < m->nrows; ++i)
                column[i] = matx_rand_uniform_fast(low, high, &state);
        }
    } else {
        for (matx_int64_t i = 0; i < m->nrows; ++i) {
            matx_double* row = m->data + i * m->stride;
            for (matx_int64_t j = 0; j < m->ncols; ++j)
                row[j] = matx_rand_uniform_fast(low, high, &state);
        }
    }
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

/* ---- Complex random number generation ---- */

matx_status_t matx_vec_z_i8_rand_uniform(matx_vec_z_i8_t v,
                                         matx_double low_re,
                                         matx_double high_re,
                                         matx_double low_im,
                                         matx_double high_im,
                                         unsigned int seed)
{
    if (!v || !v->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    uint64_t state = (uint64_t) seed;
    for (matx_int64_t i = 0; i < v->n; ++i) {
        v->data[i * v->stride].real = matx_rand_uniform_fast(low_re, high_re, &state);
        v->data[i * v->stride].imag = matx_rand_uniform_fast(low_im, high_im, &state);
    }
    return MATX_OK;
}

matx_status_t matx_dense_z_i8_rand_uniform(matx_dense_z_i8_t m,
                                           matx_double low_re,
                                           matx_double high_re,
                                           matx_double low_im,
                                           matx_double high_im,
                                           unsigned int seed)
{
    if (!m || !m->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    uint64_t state = (uint64_t) seed;
    const matx_int64_t minor
        = (m->layout == MATX_COL_MAJOR) ? m->nrows : m->ncols;
    if (m->stride == minor) {
        const matx_int64_t count = m->nrows * m->ncols;
        for (matx_int64_t i = 0; i < count; ++i) {
            m->data[i].real = matx_rand_uniform_fast(low_re, high_re, &state);
            m->data[i].imag = matx_rand_uniform_fast(low_im, high_im, &state);
        }
    } else if (m->layout == MATX_COL_MAJOR) {
        for (matx_int64_t j = 0; j < m->ncols; ++j) {
            matx_complex_d_t* column = m->data + j * m->stride;
            for (matx_int64_t i = 0; i < m->nrows; ++i) {
                column[i].real = matx_rand_uniform_fast(low_re, high_re, &state);
                column[i].imag = matx_rand_uniform_fast(low_im, high_im, &state);
            }
        }
    } else {
        for (matx_int64_t i = 0; i < m->nrows; ++i) {
            matx_complex_d_t* row = m->data + i * m->stride;
            for (matx_int64_t j = 0; j < m->ncols; ++j) {
                row[j].real = matx_rand_uniform_fast(low_re, high_re, &state);
                row[j].imag = matx_rand_uniform_fast(low_im, high_im, &state);
            }
        }
    }
    return MATX_OK;
}

matx_status_t matx_vec_z_i8_rand_normal(matx_vec_z_i8_t v,
                                        matx_double mean_re,
                                        matx_double mean_im,
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
        u1 = matx_rand_uniform_double(0.0, 1.0);
        u2 = matx_rand_uniform_double(0.0, 1.0);
        matx_double z2 = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
        v->data[i * v->stride].real = mean_re + stddev * z;
        v->data[i * v->stride].imag = mean_im + stddev * z2;
    }
    return MATX_OK;
}

/* ---- Dense element-wise min/max/clip ---- */

matx_status_t matx_dense_d_i8_min(const matx_alloc_t* alloc,
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
    MATX_DENSE_FOR_EACH_2(A, B, *out,
        (*out)->data[oidx] = A->data[aidx] < B->data[bidx] ? A->data[aidx] : B->data[bidx]);
    return MATX_OK;
}

matx_status_t matx_dense_z_i8_min(const matx_alloc_t* alloc,
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
    MATX_DENSE_FOR_EACH_2(A, B, *out,
        matx_double ar = A->data[aidx].real;
        matx_double ai = A->data[aidx].imag;
        matx_double br = B->data[bidx].real;
        matx_double bi = B->data[bidx].imag;
        matx_double mag_a = ar * ar + ai * ai;
        matx_double mag_b = br * br + bi * bi;
        if (mag_a < mag_b) {
            (*out)->data[oidx].real = ar;
            (*out)->data[oidx].imag = ai;
        } else {
            (*out)->data[oidx].real = br;
            (*out)->data[oidx].imag = bi;
        });
    return MATX_OK;
}

matx_status_t matx_dense_d_i8_max(const matx_alloc_t* alloc,
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
    MATX_DENSE_FOR_EACH_2(A, B, *out,
        (*out)->data[oidx] = A->data[aidx] > B->data[bidx] ? A->data[aidx] : B->data[bidx]);
    return MATX_OK;
}

matx_status_t matx_dense_z_i8_max(const matx_alloc_t* alloc,
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
    MATX_DENSE_FOR_EACH_2(A, B, *out,
        matx_double ar = A->data[aidx].real;
        matx_double ai = A->data[aidx].imag;
        matx_double br = B->data[bidx].real;
        matx_double bi = B->data[bidx].imag;
        matx_double mag_a = ar * ar + ai * ai;
        matx_double mag_b = br * br + bi * bi;
        if (mag_a > mag_b) {
            (*out)->data[oidx].real = ar;
            (*out)->data[oidx].imag = ai;
        } else {
            (*out)->data[oidx].real = br;
            (*out)->data[oidx].imag = bi;
        });
    return MATX_OK;
}

matx_status_t matx_dense_d_i8_clip(const matx_alloc_t* alloc,
                                   const matx_dense_d_i8_t A,
                                   matx_double lo,
                                   matx_double hi,
                                   matx_dense_d_i8_t* out)
{
    if (!alloc || !A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_status_t st = matx_dense_d_i8_create(alloc, out, A->layout, A->nrows, A->ncols, NULL);
    if (st != MATX_OK)
        return st;
    MATX_DENSE_FOR_EACH_1(A, *out,
        matx_double x = A->data[aidx];
        if (x < lo) x = lo;
        if (x > hi) x = hi;
        (*out)->data[oidx] = x);
    return MATX_OK;
}

matx_status_t matx_dense_z_i8_clip(const matx_alloc_t* alloc,
                                   const matx_dense_z_i8_t A,
                                   matx_complex_d_t lo,
                                   matx_complex_d_t hi,
                                   matx_dense_z_i8_t* out)
{
    if (!alloc || !A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_status_t st = matx_dense_z_i8_create(alloc, out, A->layout, A->nrows, A->ncols, NULL);
    if (st != MATX_OK)
        return st;
    matx_double lo_mag = lo.real * lo.real + lo.imag * lo.imag;
    matx_double hi_mag = hi.real * hi.real + hi.imag * hi.imag;
    MATX_DENSE_FOR_EACH_1(A, *out,
        matx_double re = A->data[aidx].real;
        matx_double im = A->data[aidx].imag;
        matx_double mag = re * re + im * im;
        if (mag < lo_mag) { re = lo.real; im = lo.imag; }
        if (mag > hi_mag) { re = hi.real; im = hi.imag; }
        (*out)->data[oidx].real = re;
        (*out)->data[oidx].imag = im);
    return MATX_OK;
}

/* ---- Dense complex conjugate ---- */

matx_status_t matx_dense_z_i8_conj(const matx_alloc_t* alloc,
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
    MATX_DENSE_FOR_EACH_1(A, *out,
        (*out)->data[oidx].real =  A->data[aidx].real;
        (*out)->data[oidx].imag = -A->data[aidx].imag);
    return MATX_OK;
}

matx_status_t matx_dense_z_i8_conj_inplace(matx_dense_z_i8_t A)
{
    if (!A || !A->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (A->layout == MATX_COL_MAJOR) {
        for (matx_int64_t j = 0; j < A->ncols; ++j) {
            matx_complex_d_t* column = A->data + j * A->stride;
            for (matx_int64_t i = 0; i < A->nrows; ++i)
                column[i].imag = -column[i].imag;
        }
    } else {
        for (matx_int64_t i = 0; i < A->nrows; ++i) {
            matx_complex_d_t* row = A->data + i * A->stride;
            for (matx_int64_t j = 0; j < A->ncols; ++j)
                row[j].imag = -row[j].imag;
        }
    }
    return MATX_OK;
}

/* ---- Dense real / imaginary extraction ---- */

matx_status_t matx_dense_z_i8_real(const matx_alloc_t* alloc,
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
    MATX_DENSE_FOR_EACH_1(A, *out,
        (*out)->data[oidx] = A->data[aidx].real);
    return MATX_OK;
}

matx_status_t matx_dense_z_i8_imag(const matx_alloc_t* alloc,
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
    MATX_DENSE_FOR_EACH_1(A, *out,
        (*out)->data[oidx] = A->data[aidx].imag);
    return MATX_OK;
}

/* ---- Dense cumulative sum ---- */

matx_status_t matx_dense_d_i8_cumsum(const matx_alloc_t* alloc,
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
    if (A->layout == MATX_COL_MAJOR) {
        for (matx_int64_t j = 0; j < A->ncols; ++j) {
            matx_double acc = 0.0;
            for (matx_int64_t i = 0; i < A->nrows; ++i) {
                acc += A->data[i + j * A->stride];
                (*out)->data[i + j * (*out)->stride] = acc;
            }
        }
    } else {
        for (matx_int64_t i = 0; i < A->nrows; ++i) {
            matx_double acc = 0.0;
            for (matx_int64_t j = 0; j < A->ncols; ++j) {
                acc += A->data[i * A->stride + j];
                (*out)->data[i * (*out)->stride + j] = acc;
            }
        }
    }
    return MATX_OK;
}

matx_status_t matx_dense_z_i8_cumsum(const matx_alloc_t* alloc,
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
    if (A->layout == MATX_COL_MAJOR) {
        for (matx_int64_t j = 0; j < A->ncols; ++j) {
            matx_double acc_re = 0.0;
            matx_double acc_im = 0.0;
            for (matx_int64_t i = 0; i < A->nrows; ++i) {
                acc_re += A->data[i + j * A->stride].real;
                acc_im += A->data[i + j * A->stride].imag;
                (*out)->data[i + j * (*out)->stride].real = acc_re;
                (*out)->data[i + j * (*out)->stride].imag = acc_im;
            }
        }
    } else {
        for (matx_int64_t i = 0; i < A->nrows; ++i) {
            matx_double acc_re = 0.0;
            matx_double acc_im = 0.0;
            for (matx_int64_t j = 0; j < A->ncols; ++j) {
                acc_re += A->data[i * A->stride + j].real;
                acc_im += A->data[i * A->stride + j].imag;
                (*out)->data[i * (*out)->stride + j].real = acc_re;
                (*out)->data[i * (*out)->stride + j].imag = acc_im;
            }
        }
    }
    return MATX_OK;
}

/* ---- Kronecker product ---- */

matx_status_t matx_kron_d_i8(const matx_alloc_t* alloc,
                             const matx_dense_d_i8_t A,
                             const matx_dense_d_i8_t B,
                             matx_dense_d_i8_t* out)
{
    if (!alloc || !A || !B || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_int64_t out_rows = A->nrows * B->nrows;
    matx_int64_t out_cols = A->ncols * B->ncols;
    matx_status_t st = matx_dense_d_i8_create(alloc, out, MATX_COL_MAJOR, out_rows, out_cols, NULL);
    if (st != MATX_OK)
        return st;
    for (matx_int64_t ai = 0; ai < A->nrows; ++ai) {
        for (matx_int64_t aj = 0; aj < A->ncols; ++aj) {
            matx_double a_val = A->data[(A->layout == MATX_COL_MAJOR) ? ai + aj * A->stride : ai * A->stride + aj];
            if (a_val == 0.0)
                continue;
            for (matx_int64_t bi = 0; bi < B->nrows; ++bi) {
                for (matx_int64_t bj = 0; bj < B->ncols; ++bj) {
                    matx_double b_val = B->data[(B->layout == MATX_COL_MAJOR) ? bi + bj * B->stride : bi * B->stride + bj];
                    matx_int64_t oi = ai * B->nrows + bi;
                    matx_int64_t oj = aj * B->ncols + bj;
                    (*out)->data[oi + oj * (*out)->stride] = a_val * b_val;
                }
            }
        }
    }
    return MATX_OK;
}

matx_status_t matx_kron_z_i8(const matx_alloc_t* alloc,
                             const matx_dense_z_i8_t A,
                             const matx_dense_z_i8_t B,
                             matx_dense_z_i8_t* out)
{
    if (!alloc || !A || !B || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_int64_t out_rows = A->nrows * B->nrows;
    matx_int64_t out_cols = A->ncols * B->ncols;
    matx_status_t st = matx_dense_z_i8_create(alloc, out, MATX_COL_MAJOR, out_rows, out_cols, NULL);
    if (st != MATX_OK)
        return st;
    for (matx_int64_t ai = 0; ai < A->nrows; ++ai) {
        for (matx_int64_t aj = 0; aj < A->ncols; ++aj) {
            matx_complex_d_t a_val = A->data[(A->layout == MATX_COL_MAJOR) ? ai + aj * A->stride : ai * A->stride + aj];
            if (a_val.real == 0.0 && a_val.imag == 0.0)
                continue;
            for (matx_int64_t bi = 0; bi < B->nrows; ++bi) {
                for (matx_int64_t bj = 0; bj < B->ncols; ++bj) {
                    matx_complex_d_t b_val = B->data[(B->layout == MATX_COL_MAJOR) ? bi + bj * B->stride : bi * B->stride + bj];
                    matx_int64_t oi = ai * B->nrows + bi;
                    matx_int64_t oj = aj * B->ncols + bj;
                    (*out)->data[oi + oj * (*out)->stride].real = a_val.real * b_val.real - a_val.imag * b_val.imag;
                    (*out)->data[oi + oj * (*out)->stride].imag = a_val.real * b_val.imag + a_val.imag * b_val.real;
                }
            }
        }
    }
    return MATX_OK;
}

/* ---- Matrix concatenation ---- */

matx_status_t matx_dense_d_i8_hstack(const matx_alloc_t* alloc,
                                     const matx_dense_d_i8_t A,
                                     const matx_dense_d_i8_t B,
                                     matx_dense_d_i8_t* out)
{
    if (!alloc || !A || !B || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (A->nrows != B->nrows) {
        MATX_ERROR("%s: row count mismatch for hstack", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_int64_t out_cols = A->ncols + B->ncols;
    matx_status_t st = matx_dense_d_i8_create(alloc, out, A->layout, A->nrows, out_cols, NULL);
    if (st != MATX_OK)
        return st;
    if (A->layout == MATX_COL_MAJOR) {
        for (matx_int64_t j = 0; j < A->ncols; ++j)
            for (matx_int64_t i = 0; i < A->nrows; ++i)
                (*out)->data[i + j * (*out)->stride] = A->data[i + j * A->stride];
        for (matx_int64_t j = 0; j < B->ncols; ++j)
            for (matx_int64_t i = 0; i < B->nrows; ++i)
                (*out)->data[i + (A->ncols + j) * (*out)->stride] = B->data[i + j * B->stride];
    } else {
        for (matx_int64_t i = 0; i < A->nrows; ++i) {
            for (matx_int64_t j = 0; j < A->ncols; ++j)
                (*out)->data[i * (*out)->stride + j] = A->data[i * A->stride + j];
            for (matx_int64_t j = 0; j < B->ncols; ++j)
                (*out)->data[i * (*out)->stride + A->ncols + j] = B->data[i * B->stride + j];
        }
    }
    return MATX_OK;
}

matx_status_t matx_dense_z_i8_hstack(const matx_alloc_t* alloc,
                                     const matx_dense_z_i8_t A,
                                     const matx_dense_z_i8_t B,
                                     matx_dense_z_i8_t* out)
{
    if (!alloc || !A || !B || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (A->nrows != B->nrows) {
        MATX_ERROR("%s: row count mismatch for hstack", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_int64_t out_cols = A->ncols + B->ncols;
    matx_status_t st = matx_dense_z_i8_create(alloc, out, A->layout, A->nrows, out_cols, NULL);
    if (st != MATX_OK)
        return st;
    if (A->layout == MATX_COL_MAJOR) {
        for (matx_int64_t j = 0; j < A->ncols; ++j)
            for (matx_int64_t i = 0; i < A->nrows; ++i)
                (*out)->data[i + j * (*out)->stride] = A->data[i + j * A->stride];
        for (matx_int64_t j = 0; j < B->ncols; ++j)
            for (matx_int64_t i = 0; i < B->nrows; ++i)
                (*out)->data[i + (A->ncols + j) * (*out)->stride] = B->data[i + j * B->stride];
    } else {
        for (matx_int64_t i = 0; i < A->nrows; ++i) {
            for (matx_int64_t j = 0; j < A->ncols; ++j)
                (*out)->data[i * (*out)->stride + j] = A->data[i * A->stride + j];
            for (matx_int64_t j = 0; j < B->ncols; ++j)
                (*out)->data[i * (*out)->stride + A->ncols + j] = B->data[i * B->stride + j];
        }
    }
    return MATX_OK;
}

matx_status_t matx_dense_d_i8_vstack(const matx_alloc_t* alloc,
                                     const matx_dense_d_i8_t A,
                                     const matx_dense_d_i8_t B,
                                     matx_dense_d_i8_t* out)
{
    if (!alloc || !A || !B || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (A->ncols != B->ncols) {
        MATX_ERROR("%s: column count mismatch for vstack", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_int64_t out_rows = A->nrows + B->nrows;
    matx_status_t st = matx_dense_d_i8_create(alloc, out, A->layout, out_rows, A->ncols, NULL);
    if (st != MATX_OK)
        return st;
    if (A->layout == MATX_COL_MAJOR) {
        for (matx_int64_t j = 0; j < A->ncols; ++j) {
            for (matx_int64_t i = 0; i < A->nrows; ++i)
                (*out)->data[i + j * (*out)->stride] = A->data[i + j * A->stride];
            for (matx_int64_t i = 0; i < B->nrows; ++i)
                (*out)->data[(A->nrows + i) + j * (*out)->stride] = B->data[i + j * B->stride];
        }
    } else {
        for (matx_int64_t i = 0; i < A->nrows; ++i)
            for (matx_int64_t j = 0; j < A->ncols; ++j)
                (*out)->data[i * (*out)->stride + j] = A->data[i * A->stride + j];
        for (matx_int64_t i = 0; i < B->nrows; ++i)
            for (matx_int64_t j = 0; j < B->ncols; ++j)
                (*out)->data[(A->nrows + i) * (*out)->stride + j] = B->data[i * B->stride + j];
    }
    return MATX_OK;
}

matx_status_t matx_dense_z_i8_vstack(const matx_alloc_t* alloc,
                                     const matx_dense_z_i8_t A,
                                     const matx_dense_z_i8_t B,
                                     matx_dense_z_i8_t* out)
{
    if (!alloc || !A || !B || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (A->ncols != B->ncols) {
        MATX_ERROR("%s: column count mismatch for vstack", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_int64_t out_rows = A->nrows + B->nrows;
    matx_status_t st = matx_dense_z_i8_create(alloc, out, A->layout, out_rows, A->ncols, NULL);
    if (st != MATX_OK)
        return st;
    if (A->layout == MATX_COL_MAJOR) {
        for (matx_int64_t j = 0; j < A->ncols; ++j) {
            for (matx_int64_t i = 0; i < A->nrows; ++i)
                (*out)->data[i + j * (*out)->stride] = A->data[i + j * A->stride];
            for (matx_int64_t i = 0; i < B->nrows; ++i)
                (*out)->data[(A->nrows + i) + j * (*out)->stride] = B->data[i + j * B->stride];
        }
    } else {
        for (matx_int64_t i = 0; i < A->nrows; ++i)
            for (matx_int64_t j = 0; j < A->ncols; ++j)
                (*out)->data[i * (*out)->stride + j] = A->data[i * A->stride + j];
        for (matx_int64_t i = 0; i < B->nrows; ++i)
            for (matx_int64_t j = 0; j < B->ncols; ++j)
                (*out)->data[(A->nrows + i) * (*out)->stride + j] = B->data[i * B->stride + j];
    }
    return MATX_OK;
}
