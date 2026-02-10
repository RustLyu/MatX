#include "matx/matx.h"
#include <string.h>
#include <stdint.h>

matx_status_t matx_sparse_f64_create(matx_csc_f64_t* out,
    matx_uint64_t nrows,
    matx_uint64_t ncols,
    matx_uint64_t nnz,
    const matx_alloc_t* alloc) {
    if (!out || !alloc || nrows == 0 || ncols == 0 || nnz == 0) {
        return MATX_ERR_INVALID_ARG;
    }

    memset(out, 0, sizeof(*out));

    matx_uint64_t* col_ptr_buf = (matx_uint64_t*)matx_malloc(alloc, (ncols + 1) * sizeof(matx_uint64_t));
    matx_uint64_t* row_ind_buf = (matx_uint64_t*)matx_malloc(alloc, nnz * sizeof(matx_uint64_t));
    matx_double* values_buf = (matx_double*)matx_malloc(alloc, nnz * sizeof(matx_double));

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
    out->flags = 1u;

    return MATX_OK;
}

matx_status_t matx_sparse_f64_wrap(matx_csc_f64_t* out,
    matx_uint64_t nrows,
    matx_uint64_t ncols,
    matx_uint64_t nnz,
    const matx_uint64_t* col_ptr,
    const matx_uint64_t* row_ind,
    const matx_double* values) {
    if (!out || !col_ptr || !row_ind || !values) {
        return MATX_ERR_INVALID_ARG;
    }
    if (nrows == 0 || ncols == 0 || nnz == 0 || nnz > nrows * ncols) {
        return MATX_ERR_INVALID_ARG;
    }

    memset(out, 0, sizeof(*out));

    out->nrows = nrows;
    out->ncols = ncols;
    out->nnz = nnz;
    out->col_ptr = col_ptr;
    out->row_ind = row_ind;
    out->values = values;
    out->flags = 0u;

    return MATX_OK;
}

void matx_sparse_f64_destroy(matx_coo_f64_t* m, const matx_alloc_t* alloc) {
    if (!m || !alloc) {
        return;
    }

    if ((m->flags & 1u) != 0u) {
        matx_free(alloc, (matx_uint64_t*)m->rows);
        matx_free(alloc, (matx_uint64_t*)m->columns);
        matx_free(alloc, (matx_double*)m->values);
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

    memset(m, 0, sizeof(*m));
}