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
    if (!out || !alloc || !alloc->malloc_fn || !alloc->free_fn                 \
        || nrows <= 0 || ncols <= 0 || nnz <= 0) {                            \
        MATX_ERROR("%s: invalid argument", __func__);                           \
        return MATX_ERR_INVALID_ARG;                                            \
    }                                                                           \
    if ((uint64_t) nnz > SIZE_MAX / sizeof(matx_int64_t)                       \
        || (uint64_t) nnz > SIZE_MAX / sizeof(SCA_TYPE)) {                     \
        MATX_ERROR("%s: sparse size overflow", __func__);                      \
        return MATX_ERR_INVALID_ARG;                                            \
    }                                                                           \
    const size_t index_bytes = (size_t) nnz * sizeof(matx_int64_t);             \
    const size_t value_bytes = (size_t) nnz * sizeof(SCA_TYPE);                 \
    matx_int64_t* rows = (matx_int64_t*) matx_malloc(alloc,                      \
                                                      index_bytes);             \
    matx_int64_t* cols = (matx_int64_t*) matx_malloc(alloc,                      \
                                                      index_bytes);             \
    SCA_TYPE* values_buf = (SCA_TYPE*) matx_malloc(alloc,                        \
                                                    value_bytes);               \
    if (!rows || !cols || !values_buf) {                                        \
        if (rows)                                                               \
            matx_free(alloc, rows);                                             \
        if (cols)                                                               \
            matx_free(alloc, cols);                                             \
        if (values_buf)                                                         \
            matx_free(alloc, values_buf);                                       \
        MATX_ERROR("%s: out of memory", __func__);                              \
        return MATX_ERR_OUT_OF_MEMORY;                                          \
    }                                                                           \
    if (ap != NULL) memcpy(rows, ap, index_bytes);                              \
    else memset(rows, 0, index_bytes);                                           \
    if (ai != NULL) memcpy(cols, ai, index_bytes);                              \
    else memset(cols, 0, index_bytes);                                           \
    if (ax != NULL) memcpy(values_buf, ax, value_bytes);                        \
    else memset(values_buf, 0, value_bytes);                                     \
    if (ap != NULL) {                                                           \
        for (matx_int64_t i = 0; i < nnz; ++i) {                                \
            if (rows[i] < 0 || rows[i] >= nrows) {                              \
                matx_free(alloc, rows);                                         \
                matx_free(alloc, cols);                                         \
                matx_free(alloc, values_buf);                                   \
                MATX_ERROR("%s: COO row index out of bounds", __func__);       \
                return MATX_ERR_INVALID_ARG;                                    \
            }                                                                   \
        }                                                                       \
    }                                                                           \
    if (ai != NULL) {                                                           \
        for (matx_int64_t i = 0; i < nnz; ++i) {                                \
            if (cols[i] < 0 || cols[i] >= ncols) {                              \
                matx_free(alloc, rows);                                         \
                matx_free(alloc, cols);                                         \
                matx_free(alloc, values_buf);                                   \
                MATX_ERROR("%s: COO col index out of bounds", __func__);       \
                return MATX_ERR_INVALID_ARG;                                    \
            }                                                                   \
        }                                                                       \
    }                                                                           \
    OPAQUE* out_value = matx_malloc(alloc, sizeof(OPAQUE));                     \
    if (!out_value) {                                                           \
        matx_free(alloc, rows);                                                 \
        matx_free(alloc, cols);                                                 \
        matx_free(alloc, values_buf);                                           \
        MATX_ERROR("%s: out of memory", __func__);                             \
        return MATX_ERR_OUT_OF_MEMORY;                                          \
    }                                                                           \
    memset(out_value, 0, sizeof(OPAQUE));                                       \
    out_value->nrows = nrows;                                                   \
    out_value->ncols = ncols;                                                   \
    out_value->nnz = nnz;                                                       \
    out_value->rows = rows;                                                     \
    out_value->columns = cols;                                                  \
    out_value->values = values_buf;                                             \
    out_value->flags = 1u;                                                      \
    out_value->alloc = *alloc;                                                  \
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
    if (!in) {                                                                  \
        MATX_ERROR("%s: invalid argument", __func__);                           \
        return MATX_ERR_INVALID_ARG;                                            \
    }                                                                           \
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
    if (!out || !alloc || !alloc->malloc_fn || !alloc->free_fn                 \
        || nrows <= 0 || ncols <= 0 || nnz <= 0                                \
        || ncols == INT64_MAX) {                                                \
        MATX_ERROR("%s: invalid argument", __func__);                           \
        return MATX_ERR_INVALID_ARG;                                            \
    }                                                                           \
    if ((uint64_t) ncols + 1 > SIZE_MAX / sizeof(matx_int64_t)                 \
        || (uint64_t) nnz > SIZE_MAX / sizeof(matx_int64_t)                    \
        || (uint64_t) nnz > SIZE_MAX / sizeof(SCA_TYPE)) {                     \
        MATX_ERROR("%s: sparse size overflow", __func__);                      \
        return MATX_ERR_INVALID_ARG;                                            \
    }                                                                           \
    const size_t col_ptr_bytes = ((size_t) ncols + 1) * sizeof(matx_int64_t);   \
    const size_t index_bytes = (size_t) nnz * sizeof(matx_int64_t);             \
    const size_t value_bytes = (size_t) nnz * sizeof(SCA_TYPE);                 \
    matx_int64_t* col_ptr_buf = (matx_int64_t*) matx_malloc(alloc,               \
                                                          col_ptr_bytes);       \
    matx_int64_t* row_ind_buf = (matx_int64_t*) matx_malloc(alloc,               \
                                                          index_bytes);         \
    SCA_TYPE* values_buf = (SCA_TYPE*) matx_malloc(alloc,                        \
                                                    value_bytes);               \
    matx_int64_t* coo_2_csc_id_map = (matx_int64_t*) matx_malloc(alloc,          \
                                                          index_bytes);         \
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
    memset(col_ptr_buf, 0, col_ptr_bytes);                                      \
    memset(row_ind_buf, 0, index_bytes);                                        \
    memset(values_buf, 0, value_bytes);                                         \
    memset(coo_2_csc_id_map, 0, index_bytes);                                   \
    OPAQUE* out_value = matx_malloc(alloc, sizeof(OPAQUE));                     \
    if (!out_value) {                                                           \
        matx_free(alloc, col_ptr_buf);                                          \
        matx_free(alloc, row_ind_buf);                                          \
        matx_free(alloc, values_buf);                                           \
        matx_free(alloc, coo_2_csc_id_map);                                     \
        MATX_ERROR("%s: out of memory", __func__);                             \
        return MATX_ERR_OUT_OF_MEMORY;                                          \
    }                                                                           \
    memset(out_value, 0, sizeof(OPAQUE));                                       \
    out_value->nrows = nrows;                                                   \
    out_value->ncols = ncols;                                                   \
    out_value->nnz = nnz;                                                       \
    out_value->nnz_capacity = nnz;                                              \
    out_value->col_ptr = col_ptr_buf;                                           \
    out_value->row_ind = row_ind_buf;                                           \
    out_value->values = values_buf;                                             \
    out_value->coo_csc_index_map = coo_2_csc_id_map;                            \
    out_value->struct_update = -1;                                              \
    out_value->only_value_update = -1;                                          \
    out_value->flags = 1u;                                                      \
    out_value->alloc = *alloc;                                                  \
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
    if (!m) {                                                                   \
        return;                                                                 \
    }                                                                           \
    const matx_alloc_t* object_alloc = m->alloc.free_fn ? &m->alloc : alloc;     \
    if ((m->flags & 1u) != 0u) {                                                \
        matx_free(object_alloc, m->col_ptr);                                    \
        matx_free(object_alloc, m->row_ind);                                    \
        matx_free(object_alloc, m->values);                                     \
        matx_free(object_alloc, m->coo_csc_index_map);                          \
    }                                                                           \
    matx_free(object_alloc, m);                                                 \
}

#define MATX_DEF_COO_DESTROY(PREFIX, OPAQUE, SCA_TYPE)              \
void matx_coo_sparse_##PREFIX##_destroy(const matx_alloc_t* alloc,              \
                                         matx_coo_##PREFIX##_t m)                \
{                                                                               \
    if (!m) {                                                                   \
        return;                                                                 \
    }                                                                           \
    const matx_alloc_t* object_alloc = m->alloc.free_fn ? &m->alloc : alloc;     \
    if ((m->flags & 1u) != 0u) {                                                \
        matx_free(object_alloc, (matx_int64_t*) m->rows);                      \
        m->rows = NULL;                                                         \
        matx_free(object_alloc, (matx_int64_t*) m->columns);                   \
        m->columns = NULL;                                                      \
        matx_free(object_alloc, (SCA_TYPE*) m->values);                        \
        m->values = NULL;                                                       \
    }                                                                           \
    matx_handles_destroy(m->backend_handles, &m->num_backend_handles);          \
    matx_free(object_alloc, m->aocl_csr_row_ptr);                                \
    matx_free(object_alloc, m->aocl_csr_col_ind);                                \
    matx_free(object_alloc, m->aocl_csr_values);                                 \
    if (m->handle_csc != NULL) {                                                \
        matx_csc_sparse_##PREFIX##_destroy(object_alloc, m->handle_csc);        \
    }                                                                           \
    matx_free(object_alloc, m);                                                 \
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
    if (!out || !alloc || !alloc->malloc_fn || !alloc->free_fn
        || !col_ptr || !row_ind || !values || nrows <= 0 || ncols <= 0
        || nnz <= 0 || ncols == INT64_MAX
        || (uint64_t) ncols + 1 > SIZE_MAX / sizeof(matx_int64_t)
        || (uint64_t) nnz > SIZE_MAX / sizeof(matx_int64_t)
        || (uint64_t) nnz > SIZE_MAX / sizeof(matx_double)) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (col_ptr[0] != 0 || col_ptr[ncols] != nnz) {
        MATX_ERROR("%s: invalid CSC column pointers", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    for (matx_int64_t col = 0; col < ncols; ++col) {
        if (col_ptr[col] < 0 || col_ptr[col] > col_ptr[col + 1]
            || col_ptr[col + 1] > nnz) {
            MATX_ERROR("%s: invalid CSC column pointers", __func__);
            return MATX_ERR_INVALID_ARG;
        }
    }
    for (matx_int64_t k = 0; k < nnz; ++k) {
        if (row_ind[k] < 0 || row_ind[k] >= nrows) {
            MATX_ERROR("%s: CSC row index out of bounds", __func__);
            return MATX_ERR_INVALID_ARG;
        }
    }

    matx_csc_d_i8_opaque_t* out_value = matx_malloc(alloc, sizeof(matx_csc_d_i8_opaque_t));
    if (!out_value) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(out_value, 0, sizeof(matx_csc_d_i8_opaque_t));

    out_value->nrows = nrows;
    out_value->ncols = ncols;
    out_value->nnz = nnz;
    out_value->nnz_capacity = nnz;
    out_value->col_ptr = (matx_int64_t*) col_ptr;
    out_value->row_ind = (matx_int64_t*) row_ind;
    out_value->values = (matx_double*) values;
    out_value->flags = 0u;
    out_value->alloc = *alloc;
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
    if (!out || !alloc || !alloc->malloc_fn || !alloc->free_fn
        || !rows || !cols || !values || nrows <= 0 || ncols <= 0 || nnz <= 0
        || (uint64_t) nnz > SIZE_MAX / sizeof(matx_int64_t)
        || (uint64_t) nnz > SIZE_MAX / sizeof(matx_complex_d_t)) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    for (matx_int64_t k = 0; k < nnz; ++k) {
        if (rows[k] < 0 || rows[k] >= nrows || cols[k] < 0 || cols[k] >= ncols) {
            MATX_ERROR("%s: COO index out of bounds", __func__);
            return MATX_ERR_INVALID_ARG;
        }
    }

    matx_coo_z_i8_opaque_t* out_value = matx_malloc(alloc, sizeof(matx_coo_z_i8_opaque_t));
    if (!out_value) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(out_value, 0, sizeof(matx_coo_z_i8_opaque_t));

    out_value->nrows = nrows;
    out_value->ncols = ncols;
    out_value->nnz = nnz;
    out_value->rows = (matx_int64_t*) rows;
    out_value->columns = (matx_int64_t*) cols;
    out_value->values = (matx_complex_d_t*) values;
    out_value->flags = 0u;
    out_value->alloc = *alloc;

    if (*out != NULL) {
        matx_coo_sparse_z_i8_destroy(alloc, *out);
    }

    *out = out_value;
    return MATX_OK;
}
