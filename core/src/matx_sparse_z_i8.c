#include "matx/matx_types.h"
#include "matx/matx_func.h"
#include "matx/matx_types_internal.h"

#include <string.h>

matx_status_t matx_coo_sparse_z_i8_create(
    const matx_alloc_t* alloc,
    matx_coo_z_i8_t* out,
    matx_int64_t nrows,
    matx_int64_t ncols,
    matx_int64_t nnz,
    matx_int64_t* ap,
    matx_int64_t* ai,
    matx_complex_d_t* ax) {
    if (!out || !alloc || nrows == 0 || ncols == 0 || nnz == 0) {
        return MATX_ERR_INVALID_ARG;
    }

    memset(out, 0, sizeof(*out));
    
    matx_int64_t* rows = (matx_int64_t*)matx_malloc(alloc, nnz * sizeof(matx_int64_t));
    matx_int64_t* cols = (matx_int64_t*)matx_malloc(alloc, (nnz) * sizeof(matx_int64_t));
    matx_complex_d_t* values_buf = (matx_complex_d_t*)matx_malloc(alloc, nnz * sizeof(matx_complex_d_t));

    if (!rows || !cols || !values_buf) {
        if (rows) 
            matx_free(alloc, rows);
        if (cols) 
            matx_free(alloc, cols);
        if (values_buf) 
            matx_free(alloc, values_buf);
        memset(out, 0, sizeof(*out));
        return MATX_ERR_OUT_OF_MEMORY;
    }
    if(ap != NULL)
        memcpy(rows, ap, sizeof(matx_int64_t) * nnz);
    if (ai != NULL)
        memcpy(cols, ai, sizeof(matx_int64_t) * nnz);
    if (ax != NULL)
        memcpy(values_buf, ax, sizeof(matx_complex_d_t) * nnz);

    matx_coo_z_i8_opaque_t* out_value = matx_malloc(alloc, sizeof(matx_coo_z_i8_opaque_t));
    memset(out_value, 0, sizeof(matx_coo_z_i8_opaque_t));

    out_value->nrows = nrows;
    out_value->ncols = ncols;
    out_value->nnz = nnz;
    out_value->rows = rows;
    out_value->columns = cols;
    out_value->values = values_buf;
    out_value->flags = 1u;

    if (*out != NULL)
    {
        matx_coo_sparse_z_i8_destroy(alloc, *out);
    }

    *out = out_value;

    return MATX_OK;
}

matx_status_t matx_coo_z_i8_dup(const matx_alloc_t* alloc, const matx_coo_z_i8_t in, matx_coo_z_i8_t* out)
{
    return matx_coo_sparse_z_i8_create(alloc, out, in->nrows, in->ncols, in->nnz, in->rows, in->columns, in->values);
}

matx_status_t matx_csc_sparse_z_i8_create(
    const matx_alloc_t* alloc,
    matx_csc_z_i8_t* out,
    matx_int64_t nrows,
    matx_int64_t ncols,
    matx_int64_t nnz) {
    if (!out || !alloc || nrows == 0 || ncols == 0 || nnz == 0) {
        return MATX_ERR_INVALID_ARG;
    }

    memset(out, 0, sizeof(*out));

    matx_int64_t* col_ptr_buf = (matx_int64_t*)matx_malloc(alloc, (ncols + 1) * sizeof(matx_int64_t));
    matx_int64_t* row_ind_buf = (matx_int64_t*)matx_malloc(alloc, nnz * sizeof(matx_int64_t));
    matx_complex_d_t* values_buf = (matx_complex_d_t*)matx_malloc(alloc, nnz * sizeof(matx_complex_d_t));
    matx_int64_t* coo_2_csc_id_map = (matx_int64_t*)matx_malloc(alloc, nnz * sizeof(matx_int64_t));
    memset(values_buf, 0, sizeof(matx_complex_d_t) * nnz);
    if (!col_ptr_buf || !row_ind_buf || !values_buf) {
        if (col_ptr_buf) matx_free(alloc, col_ptr_buf);
        if (row_ind_buf) matx_free(alloc, row_ind_buf);
        if (values_buf) matx_free(alloc, values_buf);
        memset(out, 0, sizeof(*out));
        return MATX_ERR_OUT_OF_MEMORY;
    }

    matx_csc_z_i8_opaque_t* out_value = matx_malloc(alloc, sizeof(matx_csc_z_i8_opaque_t));
    memset(out_value, 0, sizeof(matx_csc_z_i8_opaque_t));

    out_value->nrows = nrows;
    out_value->ncols = ncols;
    out_value->nnz = nnz;
    out_value->col_ptr = col_ptr_buf;
    out_value->row_ind = row_ind_buf;
    out_value->values = values_buf;
    out_value->coo_csc_index_map = coo_2_csc_id_map;
    out_value->struct_update = -1;
    out_value->only_value_update = -1;
    out_value->flags = 1u;

    if (*out != NULL)
    {
        matx_csc_sparse_z_i8_destroy(alloc, *out);
    }

    *out = out_value;
    return MATX_OK;
}

matx_status_t matx_coo_sparse_z_i8_wrap(const matx_alloc_t* alloc, matx_coo_z_i8_t* out,
    matx_int64_t nrows,
    matx_int64_t ncols,
    matx_int64_t nnz,
    const matx_int64_t* rows,
    const matx_int64_t* cols,
    const matx_complex_d_t* values) {
    if (!out || !rows || !cols || !values) {
        return MATX_ERR_INVALID_ARG;
    }
    if (nrows == 0 || ncols == 0 || nnz == 0) {
        return MATX_ERR_INVALID_ARG;
    }
    if (nnz > nrows * ncols) {
        return MATX_ERR_INVALID_ARG;
    }

    memset(out, 0, sizeof(*out));

    matx_coo_z_i8_opaque_t* out_value = matx_malloc(alloc, sizeof(matx_coo_z_i8_opaque_t));
    memset(out_value, 0, sizeof(matx_coo_z_i8_opaque_t));

    out_value->nrows = nrows;
    out_value->ncols = ncols;
    out_value->nnz = nnz;
    out_value->rows = rows;
    out_value->columns = cols;
    out_value->values = values;
    out_value->flags = 0u;

    *out = out_value;
    return MATX_OK;
}

void matx_csc_sparse_z_i8_destroy(const matx_alloc_t* alloc, matx_csc_z_i8_t m) {
    if (!m || !alloc) {
        return;
    }

    if ((m->flags & 1u) != 0u) {
        matx_free(alloc, m->col_ptr);
        matx_free(alloc, m->row_ind);
        matx_free(alloc, m->values);
        matx_free(alloc, m->coo_csc_index_map);
    }
    memset(m, 0, sizeof(*m));
}

void matx_coo_sparse_z_i8_destroy(const matx_alloc_t* alloc, matx_coo_z_i8_t m) {
    if (!m || !alloc) {
        return;
    }

    if ((m->flags & 1u) != 0u) {
        matx_free(alloc, (matx_int64_t*)m->rows);
m->rows = NULL;
        matx_free(alloc, (matx_int64_t*)m->columns);
m->columns = NULL;
        matx_free(alloc, (matx_complex_d_t*)m->values);
m->values = NULL;
    }

    if (m->handle_grb.valid > 0)
    {
        if (m->handle_grb.custom_free_func && m->handle_grb.impl)
        {
            m->handle_grb.custom_free_func(m->handle_grb.impl);
        }
        m->handle_grb.impl = NULL;
        m->handle_grb.valid = -1;
    }

    if (m->handle_mkl.valid > 0)
    {
        if (m->handle_mkl.custom_free_func && m->handle_mkl.impl)
        {
            m->handle_mkl.custom_free_func(m->handle_mkl.impl);
        }
        m->handle_mkl.impl = NULL;
        m->handle_mkl.valid = -1;
    }
    if (m->handle_csc != NULL)
    {
        matx_csc_sparse_z_i8_destroy(alloc, m->handle_csc);
        matx_free(alloc, m->handle_csc);
    }
    matx_free(alloc, m);
}