#include "matx/matx.h"

#include <string.h>
#include <stdint.h>

matx_status_t matx_sparse_c64_create(matx_csc_c64_t* out,
    size_t nrows,
    size_t ncols,
    size_t nnz,
    const matx_alloc_t* alloc) {
    if (!out || !alloc || nrows == 0 || ncols == 0 || nnz == 0) {
        return MATX_ERR_INVALID_ARG;
    }

    memset(out, 0, sizeof(*out));

    int* col_ptr_buf = (int*)matx_malloc(alloc, (ncols + 1) * sizeof(int));
    int* row_ind_buf = (int*)matx_malloc(alloc, nnz * sizeof(int));
    matx_complex_f64* values_buf = (matx_complex_f64*)matx_malloc(alloc, nnz * sizeof(matx_complex_f64));

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

matx_status_t matx_sparse_c64_wrap(matx_csc_c64_t* out,
    size_t nrows,
    size_t ncols,
    size_t nnz,
    const int* col_ptr,
    const int* row_ind,
    const matx_complex_f64* values) {
    if (!out || !col_ptr || !row_ind || !values) {
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
    out->col_ptr = col_ptr;
    out->row_ind = row_ind;
    out->values = values;
    out->flags = 0u;

    return MATX_OK;
}

void matx_sparse_c64_destroy(matx_csc_c64_t* m, const matx_alloc_t* alloc) {
    if (!m || !alloc) {
        return;
    }

    if ((m->flags & 1u) != 0u) {
        matx_free(alloc, (int*)m->col_ptr);
        matx_free(alloc, (int*)m->row_ind);
        matx_free(alloc, (matx_complex_f64*)m->values);
    }

    memset(m, 0, sizeof(*m));
}