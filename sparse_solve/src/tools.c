#include "matx/matx_types.h"
#include "matx/matx_func.h"
#include "matx/matx_log.h"
#include "matx/matx_types_internal.h"

#include <stdlib.h>
#include <string.h>

#define COO2CSC_SUCCESS 0
#define COO2CSC_ERR_OUT_OF_RANGE 1
#define COO2CSC_ERR_MEMORY 2
#define COO2CSC_ERR_NULL_PTR 3


typedef struct {
    matx_int64_t col;
    matx_int64_t row;
    matx_int64_t idx;
} Entry;

typedef struct {
    matx_int64_t col;
    matx_int64_t row;
    matx_int64_t* coo_indices;
    matx_int64_t coo_indices_len;
} MergedEntry;

static int entry_compare(const void* a, const void* b) {
    const Entry* entry_a = (const Entry*)a;
    const Entry* entry_b = (const Entry*)b;

    if (entry_a->col != entry_b->col) {
        return (entry_a->col < entry_b->col) ? -1 : 1;
    }
    return (entry_a->row < entry_b->row) ? -1 : 1;
}

static void free_merged_entries(MergedEntry* merged, matx_int64_t merged_len) {
    if (merged == NULL) 
        return;
    for (matx_int64_t i = 0; i < merged_len; ++i) {
        free(merged[i].coo_indices);
    }
    free(merged);
}

int coo_2_csc(matx_int64_t* columns, matx_int64_t* rows, const matx_int64_t n, matx_int64_t nnz,
    matx_int64_t* Ap,
    matx_int64_t* Ai,
    matx_int64_t* coo2csc) {
    if (columns == NULL || rows == NULL || Ap == NULL || Ai == NULL || coo2csc == NULL) {
        return COO2CSC_ERR_NULL_PTR;
    }
    if (nnz < 0 || n < 0) {
        return COO2CSC_ERR_OUT_OF_RANGE;
    }

    Entry* entries = (Entry*)malloc(nnz * sizeof(Entry));
    if (entries == NULL) {
        return COO2CSC_ERR_MEMORY;
    }

    for (matx_int64_t k = 0; k < nnz; ++k) {
        if (columns[k] < 0 || (columns[k] - 1) >= n) {
            MATX_ERROR("column index biger than row. index:%d", k);
            free(entries);
            return COO2CSC_ERR_OUT_OF_RANGE;
        }
        if (rows[k] < 0) {
            free(entries);
            MATX_ERROR("row index less than 0. index:%d", k);
            return COO2CSC_ERR_OUT_OF_RANGE;
        }
        entries[k].col = columns[k];
        entries[k].row = rows[k];
        entries[k].idx = k;
    }

    qsort(entries, nnz, sizeof(Entry), entry_compare);

    MergedEntry* merged = (MergedEntry*)malloc(nnz * sizeof(MergedEntry));
    if (merged == NULL) {
        free(entries);
        return COO2CSC_ERR_MEMORY;
    }
    matx_int64_t merged_len = 0;

    for (matx_int64_t k = 0; k < nnz; ++k) {
        const Entry* e = &entries[k];
        if (merged_len > 0 &&
            merged[merged_len - 1].col == e->col &&
            merged[merged_len - 1].row == e->row) {
            MergedEntry* last = &merged[merged_len - 1];
            matx_int64_t* new_coo_indices = (matx_int64_t*)realloc(last->coo_indices, (last->coo_indices_len + 1) * sizeof(matx_int64_t));
            if (new_coo_indices == NULL) {
                free_merged_entries(merged, merged_len);
                free(entries);
                return COO2CSC_ERR_MEMORY;
            }
            last->coo_indices = new_coo_indices;
            last->coo_indices[last->coo_indices_len] = e->idx;
            last->coo_indices_len++;
        }
        else {
            merged[merged_len].col = e->col;
            merged[merged_len].row = e->row;
            merged[merged_len].coo_indices = (matx_int64_t*)malloc(1 * sizeof(matx_int64_t));
            if (merged[merged_len].coo_indices == NULL) {
                free_merged_entries(merged, merged_len);
                free(entries);
                return COO2CSC_ERR_MEMORY;
            }
            merged[merged_len].coo_indices[0] = e->idx;
            merged[merged_len].coo_indices_len = 1;
            merged_len++;
        }
    }
    const matx_int64_t final_nnz = merged_len;

	memset(Ap, 0, (n + 1) * sizeof(matx_int64_t));
    for (matx_int64_t k = 0; k < merged_len; ++k) {
        Ap[merged[k].col + 1]++;
    }
    for (matx_int64_t j = 0; j < n; ++j) {
        Ap[j + 1] += Ap[j];
    }

    if (Ai == NULL || coo2csc == NULL) {
        free(Ap);
        free(Ai);
        free(coo2csc);
        free_merged_entries(merged, merged_len);
        free(entries);
        return COO2CSC_ERR_MEMORY;
    }

    matx_int64_t* next = (matx_int64_t*)malloc(final_nnz * sizeof(matx_int64_t));
    if (next == NULL) {
        free(Ap);
        free(Ai);
        free(coo2csc);
        free_merged_entries(merged, merged_len);
        free(entries);
        return COO2CSC_ERR_MEMORY;
    }
    memcpy(next, Ap, final_nnz * sizeof(matx_int64_t));

    for (matx_int64_t k = 0; k < merged_len; ++k) {
        const MergedEntry* e = &merged[k];
        matx_int64_t dst = next[e->col]++;
        Ai[dst] = e->row;

        for (matx_int64_t ci = 0; ci < e->coo_indices_len; ++ci) {
            matx_int64_t coo_idx = e->coo_indices[ci];
            coo2csc[coo_idx] = dst;
        }
    }

    free(next);
    free_merged_entries(merged, merged_len);
    free(entries);

    return COO2CSC_SUCCESS;
}

int build_Ax_from_coo_z_i8(const matx_int64_t* coo2csc, matx_int64_t coo2csc_len,
    const matx_complex_d_i8_t* values,
    matx_complex_d_i8_t* Ax) {
    if (coo2csc == NULL || values == NULL || Ax == NULL) {
        return COO2CSC_ERR_NULL_PTR;
    }
    if (coo2csc_len <= 0) {
        return COO2CSC_ERR_OUT_OF_RANGE;
    }

    matx_int64_t max_idx = -1;
    for (matx_int64_t i = 0; i < coo2csc_len; ++i) {
        if (coo2csc[i] > max_idx) {
            max_idx = coo2csc[i];
        }
    }

    for (matx_int64_t i = 0; i < coo2csc_len; ++i) {
        matx_int64_t idx = coo2csc[i];
        Ax[idx].real += values[i].real;
        Ax[idx].imag += values[i].imag;
    }

    return COO2CSC_SUCCESS;
}

int build_Ax_from_coo_d_i8(const matx_int64_t* coo2csc, matx_int64_t coo2csc_len,
    const matx_double* values,
    matx_double* Ax) {
    if (coo2csc == NULL || values == NULL || Ax == NULL) {
        return COO2CSC_ERR_NULL_PTR;
    }
    if (coo2csc_len <= 0) {
        return COO2CSC_ERR_OUT_OF_RANGE;
    }

    memset(Ax, 0, sizeof(matx_double) * coo2csc_len);
    for (matx_int64_t k = 0; k < coo2csc_len; ++k) {
        matx_int64_t idx = coo2csc[k];
        Ax[idx] += values[k];
    }

    return COO2CSC_SUCCESS;
}


matx_status_t coo_to_csc_d_i8(matx_coo_d_i8_t coo)
{
    int s = coo_2_csc(coo->columns, coo->rows, coo->ncols, coo->nnz, coo->handle_csc->col_ptr, coo->handle_csc->row_ind, (matx_int64_t*)coo->handle_csc->coo_csc_index_map);
    coo->handle_csc->struct_update = 0;
    return MATX_OK;
}

matx_status_t coo_to_csc_z_i8(matx_coo_z_i8_t coo)
{
    coo_2_csc(coo->columns, coo->rows, coo->ncols, coo->nnz, coo->handle_csc->col_ptr, coo->handle_csc->row_ind, (matx_int64_t*)coo->handle_csc->coo_csc_index_map);
    coo->handle_csc->struct_update = 0;
    return MATX_OK;
}

matx_status_t coo_to_csc_z_i8_value_remap(matx_coo_z_i8_t coo)
{
    build_Ax_from_coo_z_i8((matx_int64_t*)coo->handle_csc->coo_csc_index_map, coo->nnz, coo->values, coo->handle_csc->values);
    coo->handle_csc->only_value_update = 0;
    coo->handle_csc->struct_update = 0;
    return MATX_OK;
}

matx_status_t coo_to_csc_d_i8_value_remap(matx_coo_d_i8_t coo)
{
    build_Ax_from_coo_d_i8((matx_int64_t*)coo->handle_csc->coo_csc_index_map, coo->nnz, coo->values, coo->handle_csc->values);
    coo->handle_csc->only_value_update = 0;
    coo->handle_csc->struct_update = 0;
    return MATX_OK;
}
