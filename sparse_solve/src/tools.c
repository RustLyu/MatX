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
#define COO2CSC_RADIX_THRESHOLD 256

typedef struct
{
    matx_int64_t row;
    matx_int64_t idx;
} Entry;

static void swap_entries(Entry* a, Entry* b)
{
    const Entry tmp = *a;
    *a = *b;
    *b = tmp;
}

static void sift_entries_down(Entry* entries, matx_int64_t root, matx_int64_t end)
{
    for (;;) {
        matx_int64_t child = root * 2 + 1;
        if (child > end) return;
        if (child + 1 <= end && entries[child].row < entries[child + 1].row) {
            ++child;
        }
        if (entries[root].row >= entries[child].row) return;
        swap_entries(&entries[root], &entries[child]);
        root = child;
    }
}

static void heapsort_entries(Entry* entries, matx_int64_t len)
{
    for (matx_int64_t start = len / 2; start > 0;) {
        --start;
        sift_entries_down(entries, start, len - 1);
    }
    for (matx_int64_t end = len - 1; end > 0; --end) {
        swap_entries(&entries[0], &entries[end]);
        sift_entries_down(entries, 0, end - 1);
    }
}

static matx_int64_t median_row(matx_int64_t a, matx_int64_t b, matx_int64_t c)
{
    if (a < b) return (b < c) ? b : (a < c ? c : a);
    return (a < c) ? a : (b < c ? c : b);
}

static void introsort_entries(Entry* entries, matx_int64_t len, unsigned depth_limit)
{
    while (len > 32) {
        if (depth_limit == 0) {
            heapsort_entries(entries, len);
            return;
        }
        --depth_limit;

        const matx_int64_t pivot
            = median_row(entries[0].row, entries[len / 2].row, entries[len - 1].row);
        matx_int64_t left = 0;
        matx_int64_t right = len - 1;
        while (left <= right) {
            while (entries[left].row < pivot) ++left;
            while (entries[right].row > pivot) {
                if (right == 0) break;
                --right;
            }
            if (left <= right) {
                swap_entries(&entries[left], &entries[right]);
                ++left;
                if (right == 0) {
                    right = -1;
                    break;
                }
                --right;
            }
        }

        const matx_int64_t left_len = right + 1;
        const matx_int64_t right_len = len - left;
        if (left_len < right_len) {
            if (left_len > 1) introsort_entries(entries, left_len, depth_limit);
            entries += left;
            len = right_len;
        } else {
            if (right_len > 1)
                introsort_entries(entries + left, right_len, depth_limit);
            len = left_len;
        }
    }

    for (matx_int64_t i = 1; i < len; ++i) {
        const Entry value = entries[i];
        matx_int64_t j = i;
        while (j > 0 && entries[j - 1].row > value.row) {
            entries[j] = entries[j - 1];
            --j;
        }
        entries[j] = value;
    }
}

static void sort_column_entries(Entry* entries, matx_int64_t len)
{
    if (len < 2) return;
    unsigned depth_limit = 0;
    for (matx_int64_t size = len; size > 1; size >>= 1) {
        depth_limit += 2;
    }
    introsort_entries(entries, len, depth_limit);
}

static void radix_sort_column_indices(matx_int64_t* indices,
                                      matx_int64_t* scratch,
                                      const matx_int64_t* rows,
                                      matx_int64_t begin,
                                      matx_int64_t len,
                                      matx_int64_t nrows)
{
    if (len < 2 || nrows <= 1) return;

    uint64_t max_row = (uint64_t) (nrows - 1);
    unsigned passes = 0;
    do {
        ++passes;
        max_row >>= 8;
    } while (max_row != 0);

    matx_int64_t* src = indices + begin;
    matx_int64_t* dst = scratch + begin;
    for (unsigned pass = 0; pass < passes; ++pass) {
        uint64_t counts[256] = {0};
        const unsigned shift = pass * 8;
        for (matx_int64_t i = 0; i < len; ++i) {
            const uint64_t row = (uint64_t) rows[src[i]];
            ++counts[(row >> shift) & UINT64_C(0xff)];
        }

        uint64_t offset = (uint64_t) begin;
        for (size_t bucket = 0; bucket < 256; ++bucket) {
            const uint64_t count = counts[bucket];
            counts[bucket] = offset;
            offset += count;
        }
        for (matx_int64_t i = 0; i < len; ++i) {
            const matx_int64_t index = src[i];
            const uint64_t row = (uint64_t) rows[index];
            dst[counts[(row >> shift) & UINT64_C(0xff)]++ - (uint64_t) begin]
                = index;
        }

        matx_int64_t* tmp = src;
        src = dst;
        dst = tmp;
    }

    if (src != indices + begin) {
        memcpy(indices + begin, src, (size_t) len * sizeof(*indices));
    }
}

static matx_status_t coo2csc_status(int status)
{
    if (status == COO2CSC_ERR_MEMORY) return MATX_ERR_OUT_OF_MEMORY;
    if (status == COO2CSC_ERR_OUT_OF_RANGE || status == COO2CSC_ERR_NULL_PTR) {
        return MATX_ERR_INVALID_ARG;
    }
    return MATX_ERR_INTERNAL;
}

static int coo_2_csc_from_entries(const matx_int64_t* columns,
                                  const matx_int64_t* rows,
                                  const matx_int64_t ncols,
                                  const matx_int64_t nnz,
                                  matx_int64_t* Ap,
                                  matx_int64_t* Ai,
                                  matx_int64_t* coo2csc,
                                  matx_int64_t* out_nnz)
{
    if ((uint64_t) nnz > SIZE_MAX / sizeof(Entry)) {
        MATX_ERROR("%s: temporary entry array size overflow", __func__);
        memset(Ap, 0, ((size_t) ncols + 1) * sizeof(matx_int64_t));
        return COO2CSC_ERR_OUT_OF_RANGE;
    }
    Entry* entries = (Entry*) malloc((size_t) nnz * sizeof(*entries));
    if (entries == NULL) {
        MATX_ERROR("%s: out of memory", __func__);
        memset(Ap, 0, ((size_t) ncols + 1) * sizeof(matx_int64_t));
        return COO2CSC_ERR_MEMORY;
    }

    /* Reverse scatter preserves source order within each column bucket. */
    for (matx_int64_t k = nnz; k-- > 0;) {
        const matx_int64_t col = columns[k];
        const matx_int64_t dst = --Ap[col + 1];
        entries[dst].row = rows[k];
        entries[dst].idx = k;
    }
    for (matx_int64_t col = 0; col < ncols; ++col) {
        Ap[col] = Ap[col + 1];
    }
    Ap[ncols] = nnz;

    matx_int64_t unique_nnz = 0;
    matx_int64_t src_begin = 0;
    matx_int64_t src_end = Ap[1];
    for (matx_int64_t col = 0; col < ncols; ++col) {
        const matx_int64_t next_end = (col + 1 < ncols) ? Ap[col + 2] : nnz;
        const matx_int64_t len = src_end - src_begin;
        sort_column_entries(entries + src_begin, len);
        Ap[col] = unique_nnz;

        matx_int64_t previous_row = -1;
        for (matx_int64_t i = 0; i < len; ++i) {
            const Entry entry = entries[src_begin + i];
            if (i == 0 || entry.row != previous_row) {
                Ai[unique_nnz] = entry.row;
                ++unique_nnz;
            }
            coo2csc[entry.idx] = unique_nnz - 1;
            previous_row = entry.row;
        }

        src_begin = src_end;
        src_end = next_end;
    }
    Ap[ncols] = unique_nnz;
    *out_nnz = unique_nnz;
    free(entries);
    return COO2CSC_SUCCESS;
}

static int coo_2_csc_from_counts(const matx_int64_t* columns,
                                 const matx_int64_t* rows,
                                 const matx_int64_t nrows,
                                 const matx_int64_t ncols,
                                 const matx_int64_t nnz,
                                 matx_int64_t* Ap,
                                 matx_int64_t* Ai,
                                 matx_int64_t* coo2csc,
                                 matx_int64_t* out_nnz)
{
    int has_radix_columns = 0;
    for (matx_int64_t col = 0; col < ncols; ++col) {
        has_radix_columns |= (Ap[col + 1] >= COO2CSC_RADIX_THRESHOLD);
        Ap[col + 1] += Ap[col];
    }

    if (!has_radix_columns)
        return coo_2_csc_from_entries(columns, rows, ncols, nnz, Ap, Ai,
                                      coo2csc, out_nnz);

    matx_int64_t* indices
        = (matx_int64_t*) malloc((size_t) nnz * sizeof(*indices));
    if (indices == NULL) {
        MATX_ERROR("%s: out of memory", __func__);
        memset(Ap, 0, ((size_t) ncols + 1) * sizeof(matx_int64_t));
        return COO2CSC_ERR_MEMORY;
    }

    /* Use the column ends as reverse-scatter cursors, then turn them into starts. */
    for (matx_int64_t k = nnz; k-- > 0;) {
        const matx_int64_t col = columns[k];
        const matx_int64_t dst = --Ap[col + 1];
        indices[dst] = k;
    }
    for (matx_int64_t col = 0; col < ncols; ++col) {
        Ap[col] = Ap[col + 1];
    }
    Ap[ncols] = nnz;

    matx_int64_t radix_begin = 0;
    matx_int64_t radix_end = Ap[1];
    for (matx_int64_t col = 0; col < ncols; ++col) {
        const matx_int64_t next_end = (col + 1 < ncols) ? Ap[col + 2] : nnz;
        const matx_int64_t len = radix_end - radix_begin;
        if (len >= COO2CSC_RADIX_THRESHOLD) {
            radix_sort_column_indices(indices, coo2csc, rows, radix_begin,
                                      len, nrows);
        }
        radix_begin = radix_end;
        radix_end = next_end;
    }

    matx_int64_t unique_nnz = 0;
    matx_int64_t src_begin = 0;
    matx_int64_t src_end = Ap[1];
    Entry small_entries[COO2CSC_RADIX_THRESHOLD];
    for (matx_int64_t col = 0; col < ncols; ++col) {
        const matx_int64_t next_end = (col + 1 < ncols) ? Ap[col + 2] : nnz;
        const matx_int64_t len = src_end - src_begin;
        const int use_small_sort = len < COO2CSC_RADIX_THRESHOLD;
        if (use_small_sort) {
            for (matx_int64_t i = 0; i < len; ++i) {
                const matx_int64_t index = indices[src_begin + i];
                small_entries[i].row = rows[index];
                small_entries[i].idx = index;
            }
            if (nrows > 1) sort_column_entries(small_entries, len);
        }

        Ap[col] = unique_nnz;
        matx_int64_t previous_row = -1;
        for (matx_int64_t i = 0; i < len; ++i) {
            const matx_int64_t index
                = use_small_sort ? small_entries[i].idx : indices[src_begin + i];
            const matx_int64_t row = use_small_sort ? small_entries[i].row : rows[index];
            if (i == 0 || row != previous_row) {
                Ai[unique_nnz] = row;
                ++unique_nnz;
            }
            coo2csc[index] = unique_nnz - 1;
            previous_row = row;
        }

        src_begin = src_end;
        src_end = next_end;
    }
    Ap[ncols] = unique_nnz;

    *out_nnz = unique_nnz;
    free(indices);
    return COO2CSC_SUCCESS;
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
        || (uint64_t) nnz > SIZE_MAX / sizeof(matx_int64_t)
        || (uint64_t) ncols + 1 > SIZE_MAX / sizeof(matx_int64_t)) {
        MATX_ERROR("%s: invalid dimensions or size overflow", __func__);
        return COO2CSC_ERR_OUT_OF_RANGE;
    }

    int sampled_unsorted = 0;
    const matx_int64_t sample_end = nnz < 33 ? nnz : 33;
    for (matx_int64_t k = 1; k < sample_end; ++k) {
        if (columns[k] < columns[k - 1]
            || (columns[k] == columns[k - 1] && rows[k] < rows[k - 1])) {
            sampled_unsorted = 1;
            break;
        }
    }

    if (sampled_unsorted) {
        memset(Ap, 0, ((size_t) ncols + 1) * sizeof(matx_int64_t));
        for (matx_int64_t k = 0; k < nnz; ++k) {
            if (columns[k] < 0 || columns[k] >= ncols
                || rows[k] < 0 || rows[k] >= nrows) {
                MATX_ERROR("COO index out of range at position %lld", (long long) k);
                memset(Ap, 0, ((size_t) ncols + 1) * sizeof(matx_int64_t));
                return COO2CSC_ERR_OUT_OF_RANGE;
            }
            ++Ap[columns[k] + 1];
        }
        return coo_2_csc_from_counts(columns, rows, nrows, ncols, nnz, Ap, Ai,
                                    coo2csc, out_nnz);
    }

    int is_sorted = 1;
    for (matx_int64_t k = 0; k < nnz; ++k) {
        if (columns[k] < 0 || columns[k] >= ncols
            || rows[k] < 0 || rows[k] >= nrows) {
            MATX_ERROR("COO index out of range at position %lld", (long long) k);
            return COO2CSC_ERR_OUT_OF_RANGE;
        }
        if (k > 0
            && (columns[k] < columns[k - 1]
                || (columns[k] == columns[k - 1] && rows[k] < rows[k - 1]))) {
            is_sorted = 0;
        }
    }

    memset(Ap, 0, ((size_t) ncols + 1) * sizeof(matx_int64_t));
    if (is_sorted) {
        for (matx_int64_t k = 0; k < nnz; ++k) {
            if (k == 0 || columns[k] != columns[k - 1] || rows[k] != rows[k - 1]) {
                ++Ap[columns[k] + 1];
            }
        }
        for (matx_int64_t col = 0; col < ncols; ++col) {
            Ap[col + 1] += Ap[col];
        }

        matx_int64_t unique_nnz = 0;
        for (matx_int64_t k = 0; k < nnz; ++k) {
            if (k == 0 || columns[k] != columns[k - 1] || rows[k] != rows[k - 1]) {
                Ai[unique_nnz++] = rows[k];
            }
            coo2csc[k] = unique_nnz - 1;
        }

        *out_nnz = unique_nnz;
        return COO2CSC_SUCCESS;
    }

    for (matx_int64_t k = 0; k < nnz; ++k) {
        ++Ap[columns[k] + 1];
    }
    return coo_2_csc_from_counts(columns, rows, nrows, ncols, nnz, Ap, Ai,
                                coo2csc, out_nnz);
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
