#include "matx/matx_func.h"
#include "matx/matx_log.h"
#include "matx/matx_types.h"
#include "matx/matx_types_internal.h"

#include <stdlib.h>
#include <string.h>

#define COO2CSC_SUCCESS 0
#define COO2CSC_ERR_OUT_OF_RANGE 1
#define COO2CSC_ERR_MEMORY 2
#define COO2CSC_ERR_NULL_PTR 3

typedef struct
{
    matx_int64_t col;
    matx_int64_t row;
    matx_int64_t idx;
} Entry;

static int entry_compare(const void* a, const void* b)
{
    const Entry* entry_a = (const Entry*) a;
    const Entry* entry_b = (const Entry*) b;

    if (entry_a->col != entry_b->col) {
        return (entry_a->col < entry_b->col) ? -1 : 1;
    }
    if (entry_a->row != entry_b->row) {
        return (entry_a->row < entry_b->row) ? -1 : 1;
    }
    return 0;
}

static matx_status_t coo2csc_status(int status)
{
    if (status == COO2CSC_ERR_MEMORY) return MATX_ERR_OUT_OF_MEMORY;
    if (status == COO2CSC_ERR_OUT_OF_RANGE || status == COO2CSC_ERR_NULL_PTR) {
        return MATX_ERR_INVALID_ARG;
    }
    return MATX_ERR_INTERNAL;
}

int coo_2_csc(matx_int64_t* columns,
              matx_int64_t* rows,
              const matx_int64_t nrows,
              const matx_int64_t ncols,
              matx_int64_t nnz,
              matx_int64_t* Ap,
              matx_int64_t* Ai,
              matx_int64_t* coo2csc,
              matx_int64_t* out_nnz)
{
    if (!columns || !rows || !Ap || !Ai || !coo2csc || !out_nnz) {
        MATX_ERROR("%s: invalid argument", __func__);
        return COO2CSC_ERR_NULL_PTR;
    }
    if (nrows <= 0 || ncols <= 0 || nnz <= 0 || ncols == INT64_MAX
        || (uint64_t) nnz > SIZE_MAX / sizeof(Entry)
        || (uint64_t) ncols + 1 > SIZE_MAX / sizeof(matx_int64_t)) {
        MATX_ERROR("%s: invalid dimensions or size overflow", __func__);
        return COO2CSC_ERR_OUT_OF_RANGE;
    }

    Entry* entries = (Entry*) malloc((size_t) nnz * sizeof(Entry));
    if (entries == NULL) {
        MATX_ERROR("%s: out of memory", __func__);
        return COO2CSC_ERR_MEMORY;
    }

    for (matx_int64_t k = 0; k < nnz; ++k) {
        if (columns[k] < 0 || columns[k] >= ncols
            || rows[k] < 0 || rows[k] >= nrows) {
            MATX_ERROR("COO index out of range at position %lld", (long long) k);
            free(entries);
            return COO2CSC_ERR_OUT_OF_RANGE;
        }
        entries[k].col = columns[k];
        entries[k].row = rows[k];
        entries[k].idx = k;
    }

    qsort(entries, nnz, sizeof(Entry), entry_compare);

    memset(Ap, 0, ((size_t) ncols + 1) * sizeof(matx_int64_t));
    for (matx_int64_t k = 0; k < nnz; ++k) {
        if (k == 0 || entries[k].col != entries[k - 1].col
            || entries[k].row != entries[k - 1].row) {
            ++Ap[entries[k].col + 1];
        }
    }
    for (matx_int64_t col = 0; col < ncols; ++col) {
        Ap[col + 1] += Ap[col];
    }

    matx_int64_t unique_nnz = 0;
    matx_int64_t current_col = -1;
    matx_int64_t current_dst = -1;
    for (matx_int64_t k = 0; k < nnz; ++k) {
        const Entry* entry = &entries[k];
        if (k == 0 || entry->col != entries[k - 1].col
            || entry->row != entries[k - 1].row) {
            if (entry->col != current_col) {
                current_col = entry->col;
                current_dst = Ap[current_col];
            } else {
                ++current_dst;
            }
            Ai[current_dst] = entry->row;
            ++unique_nnz;
        }
        coo2csc[entry->idx] = current_dst;
    }

    *out_nnz = unique_nnz;
    free(entries);

    return COO2CSC_SUCCESS;
}

int build_Ax_from_coo_z_i8(const matx_int64_t* coo2csc,
                           matx_int64_t coo2csc_len,
                           matx_int64_t csc_nnz,
                           const matx_complex_d_t* values,
                           matx_complex_d_t* Ax)
{
    if (coo2csc == NULL || values == NULL || Ax == NULL) {
        MATX_ERROR("%s: invalid argument", __func__);
        return COO2CSC_ERR_NULL_PTR;
    }
    if (coo2csc_len <= 0 || csc_nnz <= 0
        || (uint64_t) csc_nnz > SIZE_MAX / sizeof(matx_complex_d_t)) {
        MATX_ERROR("%s: error", __func__);
        return COO2CSC_ERR_OUT_OF_RANGE;
    }

    memset(Ax, 0, (size_t) csc_nnz * sizeof(matx_complex_d_t));
    for (matx_int64_t i = 0; i < coo2csc_len; ++i) {
        const matx_int64_t idx = coo2csc[i];
        if (idx < 0 || idx >= csc_nnz) {
            MATX_ERROR("%s: invalid COO to CSC map", __func__);
            return COO2CSC_ERR_OUT_OF_RANGE;
        }
        Ax[idx].real += values[i].real;
        Ax[idx].imag += values[i].imag;
    }

    return COO2CSC_SUCCESS;
}

int build_Ax_from_coo_d_i8(const matx_int64_t* coo2csc,
                           matx_int64_t coo2csc_len,
                           matx_int64_t csc_nnz,
                           const matx_double* values,
                           matx_double* Ax)
{
    if (coo2csc == NULL || values == NULL || Ax == NULL) {
        MATX_ERROR("%s: invalid argument", __func__);
        return COO2CSC_ERR_NULL_PTR;
    }
    if (coo2csc_len <= 0 || csc_nnz <= 0
        || (uint64_t) csc_nnz > SIZE_MAX / sizeof(matx_double)) {
        MATX_ERROR("%s: error", __func__);
        return COO2CSC_ERR_OUT_OF_RANGE;
    }

    memset(Ax, 0, (size_t) csc_nnz * sizeof(matx_double));
    for (matx_int64_t k = 0; k < coo2csc_len; ++k) {
        const matx_int64_t idx = coo2csc[k];
        if (idx < 0 || idx >= csc_nnz) {
            MATX_ERROR("%s: invalid COO to CSC map", __func__);
            return COO2CSC_ERR_OUT_OF_RANGE;
        }
        Ax[idx] += values[k];
    }

    return COO2CSC_SUCCESS;
}

static matx_status_t ensure_csc_d_i8(matx_coo_d_i8_t coo)
{
    if (!coo || coo->nrows <= 0 || coo->ncols <= 0 || coo->nnz <= 0
        || !coo->rows || !coo->columns || !coo->values
        || !coo->alloc.malloc_fn || !coo->alloc.free_fn) {
        return MATX_ERR_INVALID_ARG;
    }
    matx_csc_d_i8_t csc = coo->handle_csc;
    if (!csc || csc->nrows != coo->nrows || csc->ncols != coo->ncols
        || csc->nnz_capacity < coo->nnz || !csc->col_ptr || !csc->row_ind
        || !csc->values || !csc->coo_csc_index_map) {
        matx_status_t st = matx_csc_sparse_d_i8_create(&coo->alloc,
                                                        &coo->handle_csc,
                                                        coo->nrows,
                                                        coo->ncols,
                                                        coo->nnz);
        if (st != MATX_OK) return st;
    }
    return MATX_OK;
}

static matx_status_t ensure_csc_z_i8(matx_coo_z_i8_t coo)
{
    if (!coo || coo->nrows <= 0 || coo->ncols <= 0 || coo->nnz <= 0
        || !coo->rows || !coo->columns || !coo->values
        || !coo->alloc.malloc_fn || !coo->alloc.free_fn) {
        return MATX_ERR_INVALID_ARG;
    }
    matx_csc_z_i8_t csc = coo->handle_csc;
    if (!csc || csc->nrows != coo->nrows || csc->ncols != coo->ncols
        || csc->nnz_capacity < coo->nnz || !csc->col_ptr || !csc->row_ind
        || !csc->values || !csc->coo_csc_index_map) {
        matx_status_t st = matx_csc_sparse_z_i8_create(&coo->alloc,
                                                        &coo->handle_csc,
                                                        coo->nrows,
                                                        coo->ncols,
                                                        coo->nnz);
        if (st != MATX_OK) return st;
    }
    return MATX_OK;
}

matx_status_t coo_to_csc_d_i8(matx_coo_d_i8_t coo)
{
    matx_status_t st = ensure_csc_d_i8(coo);
    if (st != MATX_OK) return st;
    matx_int64_t final_nnz = 0;
    int s = coo_2_csc(coo->columns, coo->rows, coo->nrows, coo->ncols,
                      coo->nnz, coo->handle_csc->col_ptr,
                      coo->handle_csc->row_ind,
                      coo->handle_csc->coo_csc_index_map, &final_nnz);
    if (s != COO2CSC_SUCCESS) {
        MATX_ERROR("%s: coo_2_csc failed with code %d", __func__, s);
        return coo2csc_status(s);
    }
    coo->handle_csc->nnz = final_nnz;
    coo->handle_csc->struct_update = 0;
    return MATX_OK;
}

matx_status_t coo_to_csc_z_i8(matx_coo_z_i8_t coo)
{
    matx_status_t st = ensure_csc_z_i8(coo);
    if (st != MATX_OK) return st;
    matx_int64_t final_nnz = 0;
    int s = coo_2_csc(coo->columns, coo->rows, coo->nrows, coo->ncols,
                      coo->nnz, coo->handle_csc->col_ptr,
                      coo->handle_csc->row_ind,
                      coo->handle_csc->coo_csc_index_map, &final_nnz);
    if (s != COO2CSC_SUCCESS) {
        MATX_ERROR("%s: coo_2_csc failed with code %d", __func__, s);
        return coo2csc_status(s);
    }
    coo->handle_csc->nnz = final_nnz;
    coo->handle_csc->struct_update = 0;
    return MATX_OK;
}

matx_status_t coo_to_csc_z_i8_value_remap(matx_coo_z_i8_t coo)
{
    if (!coo || !coo->handle_csc || !coo->values
        || coo->nnz > coo->handle_csc->nnz_capacity
        || coo->handle_csc->struct_update != 0) return MATX_ERR_INVALID_ARG;
    int s = build_Ax_from_coo_z_i8(coo->handle_csc->coo_csc_index_map,
                                   coo->nnz, coo->handle_csc->nnz,
                                   coo->values, coo->handle_csc->values);
    if (s != COO2CSC_SUCCESS) return coo2csc_status(s);
    coo->handle_csc->only_value_update = 0;
    coo->handle_csc->struct_update = 0;
    return MATX_OK;
}

matx_status_t coo_to_csc_d_i8_value_remap(matx_coo_d_i8_t coo)
{
    if (!coo || !coo->handle_csc || !coo->values
        || coo->nnz > coo->handle_csc->nnz_capacity
        || coo->handle_csc->struct_update != 0) return MATX_ERR_INVALID_ARG;
    int s = build_Ax_from_coo_d_i8(coo->handle_csc->coo_csc_index_map,
                                   coo->nnz, coo->handle_csc->nnz,
                                   coo->values, coo->handle_csc->values);
    if (s != COO2CSC_SUCCESS) return coo2csc_status(s);
    coo->handle_csc->only_value_update = 0;
    coo->handle_csc->struct_update = 0;
    return MATX_OK;
}
