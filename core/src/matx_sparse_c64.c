#include "matx/matx.h"

#include <string.h>
#include <stdint.h>

matx_status_t matx_coo_sparse_c64_create(matx_coo_c64_t* out,
    matx_int64_t nrows,
    matx_int64_t ncols,
    matx_int64_t nnz,
    const matx_alloc_t* alloc) {
    if (!out || !alloc || nrows == 0 || ncols == 0 || nnz == 0) {
        return MATX_ERR_INVALID_ARG;
    }

    memset(out, 0, sizeof(*out));
    
    matx_int64_t* rows = (matx_int64_t*)matx_malloc(alloc, nnz * sizeof(matx_int64_t));
    matx_int64_t* cols = (matx_int64_t*)matx_malloc(alloc, (nnz) * sizeof(matx_int64_t));
    matx_complex_f64* values_buf = (matx_complex_f64*)matx_malloc(alloc, nnz * sizeof(matx_complex_f64));

    if (!rows || !cols || !values_buf) {
        if (rows) matx_free(alloc, rows);
        if (cols) matx_free(alloc, cols);
        if (values_buf) matx_free(alloc, values_buf);
        memset(out, 0, sizeof(*out));
        return MATX_ERR_OUT_OF_MEMORY;
    }

    out->nrows = nrows;
    out->ncols = ncols;
    out->nnz = nnz;
    out->rows = rows;
    out->columns = cols;
    out->values = values_buf;
    out->flags = 1u;

    return MATX_OK;
}

matx_status_t matx_csc_sparse_c64_create(matx_csc_c64_t* out,
    matx_int64_t nrows,
    matx_int64_t ncols,
    matx_int64_t nnz,
    const matx_alloc_t* alloc) {
    if (!out || !alloc || nrows == 0 || ncols == 0 || nnz == 0) {
        return MATX_ERR_INVALID_ARG;
    }

    memset(out, 0, sizeof(*out));

    matx_int64_t* col_ptr_buf = (matx_int64_t*)matx_malloc(alloc, (ncols + 1) * sizeof(matx_int64_t));
    matx_int64_t* row_ind_buf = (matx_int64_t*)matx_malloc(alloc, nnz * sizeof(matx_int64_t));
    matx_complex_f64* values_buf = (matx_complex_f64*)matx_malloc(alloc, nnz * sizeof(matx_complex_f64));
    matx_int64_t* coo_2_csc_id_map = (matx_int64_t*)matx_malloc(alloc, nnz * sizeof(matx_int64_t));

    if (!col_ptr_buf || !row_ind_buf || !values_buf) {
        if (col_ptr_buf) matx_free(alloc, col_ptr_buf);
        if (row_ind_buf) matx_free(alloc, row_ind_buf);
        if (values_buf) matx_free(alloc, values_buf);
        memset(out, 0, sizeof(*out));
        return MATX_ERR_OUT_OF_MEMORY;
    }

    out->nrows = nrows;
    out->ncols = ncols;
    out->nnz = nnz;
    out->col_ptr = col_ptr_buf;
    out->row_ind = row_ind_buf;
    out->values = values_buf;
    out->coo_csc_index_map = coo_2_csc_id_map;
    out->struct_update = -1;
    out->only_value_update = -1;
    out->flags = 1u;

    return MATX_OK;
}

matx_status_t matx_coo_sparse_c64_wrap(matx_coo_c64_t* out,
    matx_int64_t nrows,
    matx_int64_t ncols,
    matx_int64_t nnz,
    const matx_int64_t* rows,
    const matx_int64_t* cols,
    const matx_complex_f64* values) {
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

    out->nrows = nrows;
    out->ncols = ncols;
    out->nnz = nnz;
    out->columns = rows;
    out->rows = cols;
    out->values = values;
    out->flags = 0u;

    return MATX_OK;
}

void matx_csc_sparse_c64_destroy(matx_csc_c64_t* m, const matx_alloc_t* alloc) {
    if (!m || !alloc) {
        return;
    }

    if ((m->flags & 1u) != 0u) {
        matx_free(alloc, (matx_int64_t*)m->col_ptr);
        matx_free(alloc, (matx_int64_t*)m->row_ind);
        matx_free(alloc, (matx_complex_f64*)m->values);
        matx_free(alloc, (matx_complex_f64*)m->coo_csc_index_map);
    }
    memset(m, 0, sizeof(*m));
}

void matx_coo_sparse_c64_destroy(matx_coo_c64_t* m, const matx_alloc_t* alloc) {
    if (!m || !alloc) {
        return;
    }

    if ((m->flags & 1u) != 0u) {
        matx_free(alloc, (matx_int64_t*)m->rows);
        matx_free(alloc, (matx_int64_t*)m->columns);
        matx_free(alloc, (matx_complex_f64*)m->values);
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
	matx_csc_sparse_c64_destroy(&m->handle_csc, alloc);
    memset(m, 0, sizeof(*m));
}