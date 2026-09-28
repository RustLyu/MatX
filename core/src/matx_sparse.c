#include "matx/matx_func.h"
#include "matx/matx_types.h"
#include "matx/matx_types_internal.h"

#include "matx/matx_log.h"
#include <string.h>

/* ---- Macro generators for type-agnostic sparse functions ---- */

#define MATX_DEF_COO_CREATE(PREFIX, OPAQUE, SCA_TYPE)               \
matx_status_t matx_coo_sparse_##PREFIX##_create(const matx_alloc_t* alloc,      \
                                                 matx_coo_##PREFIX##_t* out,     \
                                                 matx_int64_t nrows,             \
                                                 matx_int64_t ncols,             \
                                                 matx_int64_t nnz,               \
                                                 matx_int64_t* ap,               \
                                                 matx_int64_t* ai,               \
                                                 SCA_TYPE* ax)                   \
{                                                                               \
    if (!out || !alloc || nrows == 0 || ncols == 0 || nnz == 0) {               \
        MATX_ERROR("%s: invalid argument", __func__);                           \
        return MATX_ERR_INVALID_ARG;                                            \
    }                                                                           \
    matx_int64_t* rows = (matx_int64_t*) matx_malloc(alloc,                      \
                                                      nnz * sizeof(matx_int64_t)); \
    matx_int64_t* cols = (matx_int64_t*) matx_malloc(alloc,                      \
                                                      (nnz) * sizeof(matx_int64_t)); \
    SCA_TYPE* values_buf = (SCA_TYPE*) matx_malloc(alloc,                        \
                                                    nnz * sizeof(SCA_TYPE));     \
    if (!rows || !cols || !values_buf) {                                        \
        if (rows)                                                               \
            matx_free(alloc, rows);                                             \
        if (cols)                                                               \
            matx_free(alloc, cols);                                             \
        if (values_buf)                                                         \
            matx_free(alloc, values_buf);                                       \
        memset(out, 0, sizeof(*out));                                           \
        MATX_ERROR("%s: out of memory", __func__);                              \
        return MATX_ERR_OUT_OF_MEMORY;                                          \
    }                                                                           \
    if (ap != NULL)                                                             \
        memcpy(rows, ap, sizeof(matx_int64_t) * nnz);                           \
    if (ai != NULL)                                                             \
        memcpy(cols, ai, sizeof(matx_int64_t) * nnz);                           \
    if (ax != NULL)                                                             \
        memcpy(values_buf, ax, sizeof(SCA_TYPE) * nnz);                         \
    OPAQUE* out_value = matx_malloc(alloc, sizeof(OPAQUE));                     \
    memset(out_value, 0, sizeof(OPAQUE));                                       \
    out_value->nrows = nrows;                                                   \
    out_value->ncols = ncols;                                                   \
    out_value->nnz = nnz;                                                       \
    out_value->rows = rows;                                                     \
    out_value->columns = cols;                                                  \
    out_value->values = values_buf;                                             \
    out_value->flags = 1u;                                                      \
    if (*out != NULL) {                                                         \
        matx_coo_sparse_##PREFIX##_destroy(alloc, *out);                         \
    }                                                                           \
    *out = out_value;                                                           \
    return MATX_OK;                                                             \
}

#define MATX_DEF_COO_DUP(PREFIX, OPAQUE, SCA_TYPE)                   \
matx_status_t matx_coo_##PREFIX##_dup(const matx_alloc_t* alloc,                \
                                       const matx_coo_##PREFIX##_t in,           \
                                       matx_coo_##PREFIX##_t* out)               \
{                                                                               \
    return matx_coo_sparse_##PREFIX##_create(alloc, out, in->nrows,              \
                                              in->ncols, in->nnz,                \
                                              in->rows, in->columns,             \
                                              in->values);                      \
}

#define MATX_DEF_CSC_CREATE(PREFIX, OPAQUE, SCA_TYPE)               \
matx_status_t matx_csc_sparse_##PREFIX##_create(const matx_alloc_t* alloc,      \
                                                 matx_csc_##PREFIX##_t* out,     \
                                                 matx_int64_t nrows,             \
                                                 matx_int64_t ncols,             \
                                                 matx_int64_t nnz)               \
{                                                                               \
    if (!out || !alloc || nrows == 0 || ncols == 0 || nnz == 0) {               \
        MATX_ERROR("%s: invalid argument", __func__);                           \
        return MATX_ERR_INVALID_ARG;                                            \
    }                                                                           \
    matx_int64_t* col_ptr_buf = (matx_int64_t*) matx_malloc(alloc,               \
                                             (ncols + 1) * sizeof(matx_int64_t)); \
    matx_int64_t* row_ind_buf = (matx_int64_t*) matx_malloc(alloc,               \
                                                 nnz * sizeof(matx_int64_t));    \
    SCA_TYPE* values_buf = (SCA_TYPE*) matx_malloc(alloc,                        \
                                                    nnz * sizeof(SCA_TYPE));     \
    matx_int64_t* coo_2_csc_id_map = (matx_int64_t*) matx_malloc(alloc,          \
                                                  nnz * sizeof(matx_int64_t));    \
    if (!col_ptr_buf || !row_ind_buf || !values_buf || !coo_2_csc_id_map) {      \
        if (col_ptr_buf)                                                        \
            matx_free(alloc, col_ptr_buf);                                      \
        if (row_ind_buf)                                                        \
            matx_free(alloc, row_ind_buf);                                      \
        if (values_buf)                                                         \
            matx_free(alloc, values_buf);                                       \
        if (coo_2_csc_id_map)                                                   \
            matx_free(alloc, coo_2_csc_id_map);                                 \
        MATX_ERROR("%s: out of memory", __func__);                              \
        return MATX_ERR_OUT_OF_MEMORY;                                          \
    }                                                                           \
    OPAQUE* out_value = matx_malloc(alloc, sizeof(OPAQUE));                     \
    memset(out_value, 0, sizeof(OPAQUE));                                       \
    out_value->nrows = nrows;                                                   \
    out_value->ncols = ncols;                                                   \
    out_value->nnz = nnz;                                                       \
    out_value->col_ptr = col_ptr_buf;                                           \
    out_value->row_ind = row_ind_buf;                                           \
    out_value->values = values_buf;                                             \
    out_value->coo_csc_index_map = coo_2_csc_id_map;                            \
    out_value->struct_update = -1;                                              \
    out_value->only_value_update = -1;                                          \
    out_value->flags = 1u;                                                      \
    if (*out != NULL) {                                                         \
        matx_csc_sparse_##PREFIX##_destroy(alloc, *out);                         \
    }                                                                           \
    *out = out_value;                                                           \
    return MATX_OK;                                                             \
}

#define MATX_DEF_CSC_DESTROY(PREFIX, OPAQUE, SCA_TYPE)              \
void matx_csc_sparse_##PREFIX##_destroy(const matx_alloc_t* alloc,              \
                                         matx_csc_##PREFIX##_t m)                \
{                                                                               \
    if (!m || !alloc) {                                                         \
        return;                                                                 \
    }                                                                           \
    if ((m->flags & 1u) != 0u) {                                                \
        matx_free(alloc, m->col_ptr);                                           \
        matx_free(alloc, m->row_ind);                                           \
        matx_free(alloc, m->values);                                            \
        matx_free(alloc, m->coo_csc_index_map);                                 \
    }                                                                           \
    memset(m, 0, sizeof(*m));                                                   \
}

#define MATX_DEF_COO_DESTROY(PREFIX, OPAQUE, SCA_TYPE)              \
void matx_coo_sparse_##PREFIX##_destroy(const matx_alloc_t* alloc,              \
                                         matx_coo_##PREFIX##_t m)                \
{                                                                               \
    if (!m || !alloc) {                                                         \
        return;                                                                 \
    }                                                                           \
    if ((m->flags & 1u) != 0u) {                                                \
        matx_free(alloc, (matx_int64_t*) m->rows);                              \
        m->rows = NULL;                                                         \
        matx_free(alloc, (matx_int64_t*) m->columns);                           \
        m->columns = NULL;                                                      \
        matx_free(alloc, (SCA_TYPE*) m->values);                                \
        m->values = NULL;                                                       \
    }                                                                           \
    matx_handles_destroy(m->backend_handles, &m->num_backend_handles);          \
    if (m->handle_csc != NULL) {                                                \
        matx_csc_sparse_##PREFIX##_destroy(alloc, m->handle_csc);               \
        matx_free(alloc, m->handle_csc);                                        \
    }                                                                           \
    matx_free(alloc, m);                                                        \
}

/* ---- Type-agnostic expansions ---- */

MATX_DEF_COO_CREATE(d_i8, matx_coo_d_i8_opaque_t, matx_double)
MATX_DEF_COO_CREATE(z_i8, matx_coo_z_i8_opaque_t, matx_complex_d_t)
MATX_DEF_COO_DUP(d_i8, matx_coo_d_i8_opaque_t, matx_double)
MATX_DEF_COO_DUP(z_i8, matx_coo_z_i8_opaque_t, matx_complex_d_t)
MATX_DEF_CSC_CREATE(d_i8, matx_csc_d_i8_opaque_t, matx_double)
MATX_DEF_CSC_CREATE(z_i8, matx_csc_z_i8_opaque_t, matx_complex_d_t)
MATX_DEF_CSC_DESTROY(d_i8, matx_csc_d_i8_opaque_t, matx_double)
MATX_DEF_CSC_DESTROY(z_i8, matx_csc_z_i8_opaque_t, matx_complex_d_t)
MATX_DEF_COO_DESTROY(d_i8, matx_coo_d_i8_opaque_t, matx_double)
MATX_DEF_COO_DESTROY(z_i8, matx_coo_z_i8_opaque_t, matx_complex_d_t)

#undef MATX_DEF_COO_CREATE
#undef MATX_DEF_COO_DUP
#undef MATX_DEF_CSC_CREATE
#undef MATX_DEF_CSC_DESTROY
#undef MATX_DEF_COO_DESTROY

/* ---- Single-type: wrap functions (one per format) ---- */

matx_status_t matx_csc_sparse_d_i8_wrap(const matx_alloc_t* alloc,
                                        matx_csc_d_i8_t* out,
                                        matx_int64_t nrows,
                                        matx_int64_t ncols,
                                        matx_int64_t nnz,
                                        const matx_int64_t* col_ptr,
                                        const matx_int64_t* row_ind,
                                        const matx_double* values)
{
    if (!out || !alloc || !col_ptr || !row_ind || !values) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (nrows == 0 || ncols == 0 || nnz == 0 || nnz > nrows * ncols) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    matx_csc_d_i8_opaque_t* out_value = matx_malloc(alloc, sizeof(matx_csc_d_i8_opaque_t));
    memset(out_value, 0, sizeof(matx_csc_d_i8_opaque_t));

    out_value->nrows = nrows;
    out_value->ncols = ncols;
    out_value->nnz = nnz;
    out_value->col_ptr = col_ptr;
    out_value->row_ind = row_ind;
    out_value->values = values;
    out_value->flags = 0u;
    out_value->struct_update = -1;
    out_value->only_value_update = -1;

    if (*out != NULL) {
        matx_csc_sparse_d_i8_destroy(alloc, *out);
    }

    *out = out_value;
    return MATX_OK;
}

matx_status_t matx_coo_sparse_z_i8_wrap(const matx_alloc_t* alloc,
                                        matx_coo_z_i8_t* out,
                                        matx_int64_t nrows,
                                        matx_int64_t ncols,
                                        matx_int64_t nnz,
                                        const matx_int64_t* rows,
                                        const matx_int64_t* cols,
                                        const matx_complex_d_t* values)
{
    if (!out || !rows || !cols || !values) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (nrows == 0 || ncols == 0 || nnz == 0) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (nnz > nrows * ncols) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    matx_coo_z_i8_opaque_t* out_value = matx_malloc(alloc, sizeof(matx_coo_z_i8_opaque_t));
    memset(out_value, 0, sizeof(matx_coo_z_i8_opaque_t));

    out_value->nrows = nrows;
    out_value->ncols = ncols;
    out_value->nnz = nnz;
    out_value->rows = rows;
    out_value->columns = cols;
    out_value->values = values;
    out_value->flags = 0u;

    if (*out != NULL) {
        matx_coo_sparse_z_i8_destroy(alloc, *out);
    }

    *out = out_value;
    return MATX_OK;
}