#include "matx/matx_func.h"
#include "matx/matx_log.h"
#include "matx/matx_sparse_compute.h"
#include "matx/matx_types.h"
#include "matx/matx_types_internal.h"

#include <stdlib.h>
#include <string.h>

#if MATX_HAVE_GRAPHBLAS
#include <GraphBLAS.h>
#endif

#if MATX_HAVE_AOCL_SPARSE
#include <aoclsparse.h>
#endif

#if MATX_HAVE_GRAPHBLAS

void free_grb_matrix(void* impl)
{
    //GrB_Matrix_free(&impl);
    GrB_Matrix_free((GrB_Matrix*) &impl);
}

void free_grb_vector(void* impl)
{
    GrB_Vector_free((GrB_Vector*) &impl);
}

#endif /* MATX_HAVE_GRAPHBLAS */

void free_aocl_matrix(void* impl)
{
#if MATX_HAVE_AOCL_SPARSE
    aoclsparse_destroy((void*) &impl);
#endif
}

#if MATX_HAVE_GRAPHBLAS

size_t coo_2_grb_d_i8(matx_coo_d_i8_t A)
{
    GrB_Matrix_free(MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl);
    GrB_Info info
        = GrB_Matrix_import_FP64((GrB_Matrix*) &MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl,
                                 GrB_FP64,
                                 A->nrows,
                                 A->ncols,
                                 A->rows,
                                 A->columns,
                                 A->values,
                                 A->nnz,
                                 A->nnz,
                                 A->nnz,
                                 GrB_COO_FORMAT);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("coo to grb error:%d", info);
        return -1;
    }

    MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->type = MATX_HANDLE_TYPE_GRB_MATRIX;
    MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->valid = 1;
    MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->custom_free_func = &free_grb_matrix;
    return 0;
}

size_t create_empty_grb_d_i8(matx_coo_d_i8_t A)
{
    GrB_Matrix_free(MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl);
    GrB_Info info = GrB_Matrix_new((GrB_Matrix*) &MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl,
                                   GrB_FP64,
                                   A->nrows,
                                   A->ncols);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("create f grb handle error:%d", info);
        return -1;
    }
    MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->type = MATX_HANDLE_TYPE_GRB_MATRIX;
    MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->valid = 1;
    MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->custom_free_func = &free_grb_matrix;
    return 0;
}

size_t create_empty_grb_z_i8(matx_coo_z_i8_t A)
{
    GrB_Matrix_free(MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl);
    GrB_Info info = GrB_Matrix_new((GrB_Matrix*) &MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl,
                                   GxB_FC64,
                                   A->nrows,
                                   A->ncols);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("create c grb handle error:%d", info);
        return -1;
    }
    MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->type = MATX_HANDLE_TYPE_GRB_MATRIX;
    MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->valid = 1;
    MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->custom_free_func = &free_grb_matrix;
    return 0;
}

#endif /* MATX_HAVE_GRAPHBLAS */

typedef struct
{
    matx_int64_t nrows;
    matx_int64_t ncols;
    matx_int64_t nnz;

    matx_int64_t* row_ptr; // size nrows+1
    matx_int64_t* col_ind; // size nnz
    matx_double* val;      // size nnz
} csr_matrix;

static inline void swap_int(matx_int64_t* a, matx_int64_t* b)
{
    matx_int64_t t = *a;
    *a = *b;
    *b = t;
}

static inline void swap_double(matx_double* a, matx_double* b)
{
    matx_double t = *a;
    *a = *b;
    *b = t;
}

static void insertion_sort(matx_int64_t* col, matx_double* val, matx_int64_t len)
{
    for (matx_int64_t i = 1; i < len; ++i) {
        matx_int64_t c = col[i];
        matx_double v = val[i];
        matx_int64_t j = i - 1;
        while (j >= 0 && col[j] > c) {
            col[j + 1] = col[j];
            val[j + 1] = val[j];
            j--;
        }
        col[j + 1] = c;
        val[j + 1] = v;
    }
}

static void swap_csr_entry(matx_int64_t* col, matx_double* val,
                           matx_int64_t left, matx_int64_t right)
{
    swap_int(&col[left], &col[right]);
    swap_double(&val[left], &val[right]);
}

static void sift_csr_entries_down(matx_int64_t* col, matx_double* val,
                                  matx_int64_t root, matx_int64_t end)
{
    for (;;) {
        matx_int64_t child = root * 2 + 1;
        if (child > end) return;
        if (child + 1 <= end && col[child] < col[child + 1]) ++child;
        if (col[root] >= col[child]) return;
        swap_csr_entry(col, val, root, child);
        root = child;
    }
}

static void heapsort_csr_entries(matx_int64_t* col, matx_double* val, matx_int64_t len)
{
    for (matx_int64_t start = len / 2; start > 0;) {
        --start;
        sift_csr_entries_down(col, val, start, len - 1);
    }
    for (matx_int64_t end = len - 1; end > 0; --end) {
        swap_csr_entry(col, val, 0, end);
        sift_csr_entries_down(col, val, 0, end - 1);
    }
}

static matx_int64_t median_col(matx_int64_t a, matx_int64_t b, matx_int64_t c)
{
    if (a < b) return (b < c) ? b : (a < c ? c : a);
    return (a < c) ? a : (b < c ? c : b);
}

static void introsort_csr_entries(matx_int64_t* col, matx_double* val,
                                  matx_int64_t len, unsigned depth_limit)
{
    while (len > 32) {
        if (depth_limit == 0) {
            heapsort_csr_entries(col, val, len);
            return;
        }
        --depth_limit;

        const matx_int64_t pivot = median_col(col[0], col[len / 2], col[len - 1]);
        matx_int64_t left = 0;
        matx_int64_t right = len - 1;
        while (left <= right) {
            while (col[left] < pivot) ++left;
            while (col[right] > pivot) {
                if (right == 0) break;
                --right;
            }
            if (left <= right) {
                swap_csr_entry(col, val, left, right);
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
            if (left_len > 1)
                introsort_csr_entries(col, val, left_len, depth_limit);
            col += left;
            val += left;
            len = right_len;
        } else {
            if (right_len > 1)
                introsort_csr_entries(col + left, val + left, right_len, depth_limit);
            len = left_len;
        }
    }

    insertion_sort(col, val, len);
}

static void sort_row(matx_int64_t* col, matx_double* val, matx_int64_t len)
{
    if (len <= 32)
        insertion_sort(col, val, len);
    else {
        unsigned depth_limit = 0;
        for (matx_int64_t size = len; size > 1; size >>= 1) {
            depth_limit += 2;
        }
        introsort_csr_entries(col, val, len, depth_limit);
    }
}

static void free_csr_matrix(const matx_alloc_t* alloc, csr_matrix* csr)
{
    if (!csr) return;
    matx_free(alloc, csr->row_ptr);
    matx_free(alloc, csr->col_ind);
    matx_free(alloc, csr->val);
    memset(csr, 0, sizeof(*csr));
}

int coo_to_csr_optimized(const matx_alloc_t* alloc,
                         matx_int64_t nrows,
                         matx_int64_t ncols,
                         matx_int64_t nnz,
                         const matx_int64_t* coo_row,
                         const matx_int64_t* coo_col,
                         const matx_double* coo_val,
                         csr_matrix* csr)
{
    if (!alloc || !alloc->malloc_fn || !alloc->free_fn || !csr
        || !coo_row || !coo_col || !coo_val || nrows <= 0 || ncols <= 0
        || nnz <= 0 || nrows == INT64_MAX
        || (uint64_t) nrows + 1 > SIZE_MAX / sizeof(matx_int64_t)
        || (uint64_t) nrows > SIZE_MAX / sizeof(matx_int64_t)
        || (uint64_t) nnz > SIZE_MAX / sizeof(matx_int64_t)
        || (uint64_t) nnz > SIZE_MAX / sizeof(matx_double)) {
        MATX_ERROR("invalid COO matrix or size overflow");
        return -1;
    }
    memset(csr, 0, sizeof(*csr));
    csr->nrows = nrows;
    csr->ncols = ncols;

    csr->row_ptr = (matx_int64_t*) matx_malloc(alloc,
                                    ((size_t) nrows + 1) * sizeof(matx_int64_t));
    csr->col_ind = (matx_int64_t*) matx_malloc(alloc,
                                    (size_t) nnz * sizeof(matx_int64_t));
    csr->val = (matx_double*) matx_malloc(alloc,
                                    (size_t) nnz * sizeof(matx_double));

    if (!csr->row_ptr || !csr->col_ind || !csr->val) {
        free_csr_matrix(alloc, csr);
        MATX_ERROR("out of memory converting COO to CSR");
        return -1;
    }
    memset(csr->row_ptr, 0, ((size_t) nrows + 1) * sizeof(matx_int64_t));
    for (matx_int64_t i = 0; i < nnz; ++i) {
        const matx_int64_t row = coo_row[i];
        if (row < 0 || row >= nrows) {
            free_csr_matrix(alloc, csr);
            MATX_ERROR("COO index out of bounds at position %lld", (long long) i);
            return -1;
        }
        csr->row_ptr[row + 1]++;
    }

    for (matx_int64_t i = 0; i < nrows; ++i) {
        csr->row_ptr[i + 1] += csr->row_ptr[i];
    }

    for (matx_int64_t i = 0; i < nnz; ++i) {
        const matx_int64_t r = coo_row[i];
        const matx_int64_t c = coo_col[i];
        if (c < 0 || c >= ncols) {
            free_csr_matrix(alloc, csr);
            MATX_ERROR("COO index out of bounds at position %lld", (long long) i);
            return -1;
        }
        const matx_int64_t dst = csr->row_ptr[r]++;

        csr->col_ind[dst] = c;
        csr->val[dst] = coo_val[i];
    }

    /* Restore row starts by shifting the advanced per-row cursors right. */
    for (matx_int64_t row = nrows; row > 0; --row) {
        csr->row_ptr[row] = csr->row_ptr[row - 1];
    }
    csr->row_ptr[0] = 0;

    matx_int64_t new_nnz = 0;

    for (matx_int64_t i = 0; i < nrows; ++i) {
        matx_int64_t start = csr->row_ptr[i];
        matx_int64_t end = csr->row_ptr[i + 1];
        matx_int64_t len = end - start;

        if (len == 0) {
            csr->row_ptr[i] = new_nnz;
            continue;
        }

        matx_int64_t* col = csr->col_ind + start;
        matx_double* val = csr->val + start;

        sort_row(col, val, len);

        matx_int64_t write = 0;
        for (matx_int64_t j = 0; j < len; ++j) {
            if (j > 0 && col[j] == col[j - 1]) {
                val[write - 1] += val[j];
            } else {
                col[write] = col[j];
                val[write] = val[j];
                write++;
            }
        }

        for (matx_int64_t j = 0; j < write; ++j) {
            csr->col_ind[new_nnz + j] = col[j];
            csr->val[new_nnz + j] = val[j];
        }

        csr->row_ptr[i] = new_nnz;
        new_nnz += write;
    }

    csr->row_ptr[nrows] = new_nnz;
    csr->nnz = new_nnz;

    return 0;
}

size_t coo_2_aocl_d_i8(matx_coo_d_i8_t A)
{
#if MATX_HAVE_AOCL_SPARSE

    aoclsparse_matrix csr;
    csr_matrix csr_m;
    if (!A || !A->alloc.malloc_fn || !A->alloc.free_fn
        || !A->rows || !A->columns || !A->values) {
        MATX_ERROR("%s: invalid argument", __func__);
        return (size_t) -1;
    }

    matx_handle_t* handle = MATX_HANDLE(A, MATX_HANDLE_TYPE_AOCL_MATRIX);
    if (handle->valid && handle->custom_free_func) {
        handle->custom_free_func(handle->impl);
    }
    handle->impl = NULL;
    handle->valid = 0;
    handle->custom_free_func = NULL;
    matx_free(&A->alloc, A->aocl_csr_row_ptr);
    matx_free(&A->alloc, A->aocl_csr_col_ind);
    matx_free(&A->alloc, A->aocl_csr_values);
    A->aocl_csr_row_ptr = NULL;
    A->aocl_csr_col_ind = NULL;
    A->aocl_csr_values = NULL;

    matx_int64_t ret = coo_to_csr_optimized(&A->alloc, A->nrows, A->ncols,
                                            A->nnz, A->rows, A->columns,
                                            A->values, &csr_m);
    if (ret != 0) {
        MATX_ERROR("coo 2 csr op. error");
        return -1;
    }
    aoclsparse_index_base base = aoclsparse_index_base_zero;
    aoclsparse_status st = aoclsparse_create_dcsr(&csr,
                                                  base,
                                                  csr_m.nrows,
                                                  csr_m.ncols,
                                                  csr_m.nnz,
                                                  csr_m.row_ptr,
                                                  csr_m.col_ind,
                                                  csr_m.val);
    if (st != aoclsparse_status_success) {
        free_csr_matrix(&A->alloc, &csr_m);
        MATX_ERROR("aocl create dcsr. error: %d", st);
        return -1;
    }

    st = aoclsparse_optimize(csr);

    if (st != aoclsparse_status_success) {
        aoclsparse_destroy(&csr);
        free_csr_matrix(&A->alloc, &csr_m);
        MATX_ERROR("aocl op mtx error: %d", st);
        return -1;
    }
    handle->impl = csr;
    handle->type = MATX_HANDLE_TYPE_AOCL_MATRIX;
    handle->valid = 1;
    handle->custom_free_func = &free_aocl_matrix;
    A->aocl_csr_row_ptr = csr_m.row_ptr;
    A->aocl_csr_col_ind = csr_m.col_ind;
    A->aocl_csr_values = csr_m.val;
#endif

    return 0;
}

size_t coo_2_aocl_z_i8(matx_coo_z_i8_t A)
{
#if MATX_HAVE_AOCL_SPARSE

    aoclsparse_matrix coo;
    aoclsparse_matrix csr;
    aoclsparse_status st;

    st = aoclsparse_create_zcoo(&coo,
                                aoclsparse_index_base_zero,
                                (aoclsparse_int) A->nrows,
                                (aoclsparse_int) A->ncols,
                                (aoclsparse_int) A->nnz,
                                (aoclsparse_int*) A->rows,
                                (aoclsparse_int*) A->columns,
                                (void*) A->values);

    if (st != aoclsparse_status_success) {
        MATX_ERROR("aocl create zcoo error: %d", st);
        return (size_t) -1;
    }

    st = aoclsparse_convert_csr(coo, aoclsparse_operation_none, &csr);

    if (st != aoclsparse_status_success) {
        MATX_ERROR("aocl convert csr error: %d", st);
        return (size_t) -1;
    }

    aoclsparse_destroy(&coo);
    aoclsparse_optimize(csr);

    MATX_HANDLE(A, MATX_HANDLE_TYPE_AOCL_MATRIX)->impl = csr;
    MATX_HANDLE(A, MATX_HANDLE_TYPE_AOCL_MATRIX)->type = MATX_HANDLE_TYPE_AOCL_MATRIX;
    MATX_HANDLE(A, MATX_HANDLE_TYPE_AOCL_MATRIX)->valid = 1;
    MATX_HANDLE(A, MATX_HANDLE_TYPE_AOCL_MATRIX)->custom_free_func = &free_aocl_matrix;

#endif

    return 0;
}

size_t aocl_2_coo_z_i8(matx_coo_z_i8_t A)
{
#if MATX_HAVE_AOCL_SPARSE
    aoclsparse_index_base base;
    aoclsparse_status st = aoclsparse_export_zcoo(MATX_HANDLE(A, MATX_HANDLE_TYPE_AOCL_MATRIX)->impl,
                                                  &base,
                                                  &A->nrows,
                                                  &A->ncols,
                                                  &A->nnz,
                                                  &A->rows,
                                                  &A->columns,
                                                  (aoclsparse_double_complex**) (&A->values));
    if (st != aoclsparse_status_success) {
        MATX_ERROR("aocl expoert zcoo error: %d", st);
        return (size_t) -1;
    }
    MATX_HANDLE(A, MATX_HANDLE_TYPE_AOCL_MATRIX)->type = MATX_HANDLE_TYPE_AOCL_MATRIX;
    MATX_HANDLE(A, MATX_HANDLE_TYPE_AOCL_MATRIX)->valid = 1;
    MATX_HANDLE(A, MATX_HANDLE_TYPE_AOCL_MATRIX)->custom_free_func = &free_aocl_matrix;
#endif
    return 0;
}

#if MATX_HAVE_GRAPHBLAS

size_t dense_2_grb_d_i8(matx_dense_d_i8_t A)
{
    const uint64_t values_size
        = (uint64_t) A->nrows * (uint64_t) A->ncols * sizeof(*A->data);
    GrB_Matrix_free(MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl);
    GrB_Info info
        = GxB_Matrix_import_FullC((GrB_Matrix*) &MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl,
                                  GrB_FP64,
                                  A->nrows,
                                  A->ncols,
                                  (void*) &A->data,
                                  values_size,
                                  false,
                                  NULL);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("grb import fullc, %d", info);
        return -1;
    }
    MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->type = MATX_HANDLE_TYPE_GRB_MATRIX;
    MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->valid = 1;
    MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->custom_free_func = NULL;
    return 0;
}

size_t grb_2_dense_d_i8(matx_dense_d_i8_t A)
{
    GrB_Type t;
    uint64_t values_size = 0;
    bool iso = false;
    GrB_Info info
        = GxB_Matrix_export_FullC((GrB_Matrix*) &MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl,
                                  &t,
                                  &A->nrows,
                                  &A->ncols,
                                  (void*) &A->data,
                                  &values_size,
                                  &iso,
                                  NULL);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("grb export fullc, %d", info);
        return -1;
    }
    const uint64_t expected_size
        = (uint64_t) A->nrows * (uint64_t) A->ncols * sizeof(*A->data);
    MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->valid = 0;
    if (values_size != expected_size) {
        MATX_ERROR("grb export fullc returned an unexpected size");
        return -1;
    }
    MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->custom_free_func = NULL;
    return 0;
}

size_t grb_2_coo_d_i8(matx_coo_d_i8_t A)
{
    A->nrows = -1;
    A->ncols = -1;
    GrB_Info info = GrB_Matrix_export(A->rows,
                                      A->columns,
                                      A->values,
                                      &A->nrows,
                                      &A->ncols,
                                      &A->nnz,
                                      GrB_COO_FORMAT,
                                      MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("grb export error: %d", info);
        return -1;
    }
    MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->valid = 1;
    return 0;
}

size_t vec_2_grb_d_i8(matx_vec_d_i8_t v)
{
    const uint64_t values_size = (uint64_t) v->n * sizeof(*v->data);
    GrB_Info info
        = GxB_Vector_import_Full((GrB_Vector*) &MATX_HANDLE(v, MATX_HANDLE_TYPE_GRB_VECTOR)->impl,
                                 GrB_FP64,
                                 v->n,
                                 (void*) &v->data,
                                 values_size,
                                 false,
                                 NULL);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("grb vector import error: %d", info);
        return -1;
    }
    MATX_HANDLE(v, MATX_HANDLE_TYPE_GRB_VECTOR)->type = MATX_HANDLE_TYPE_GRB_VECTOR;
    MATX_HANDLE(v, MATX_HANDLE_TYPE_GRB_VECTOR)->valid = 1;
    MATX_HANDLE(v, MATX_HANDLE_TYPE_GRB_VECTOR)->custom_free_func = NULL;
    return 0;
}

size_t grb_2_vec_d_i8(matx_vec_d_i8_t v)
{
    GrB_Type t = GrB_FP64;
    uint64_t vector_size = 0;
    uint64_t values_size = 0;
    bool iso = false;
    GrB_Info info
        = GxB_Vector_export_Full((GrB_Vector*) &MATX_HANDLE(v, MATX_HANDLE_TYPE_GRB_VECTOR)->impl,
                                 &t,
                                 &vector_size,
                                 (void*) &v->data,
                                 &values_size,
                                 &iso,
                                 NULL);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("grb vector export error: %d", info);
        return -1;
    }
    if (vector_size != (uint64_t) v->n || values_size != vector_size * sizeof(*v->data)) {
        MATX_HANDLE(v, MATX_HANDLE_TYPE_GRB_VECTOR)->valid = 0;
        MATX_ERROR("grb vector export returned an unexpected size");
        return -1;
    }
    MATX_HANDLE(v, MATX_HANDLE_TYPE_GRB_VECTOR)->valid = 0;
    MATX_HANDLE(v, MATX_HANDLE_TYPE_GRB_VECTOR)->custom_free_func = NULL;
    return 0;
}

size_t coo_2_grb_z_i8(matx_coo_z_i8_t A)
{
    GrB_Info info
        = GxB_Matrix_import_FC64((GrB_Matrix*) &MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl,
                                 GxB_FC64,
                                 A->nrows,
                                 A->ncols,
                                 A->rows,
                                 A->columns,
                                 (void*) A->values,
                                 A->nnz,
                                 A->nnz,
                                 A->nnz,
                                 GrB_COO_FORMAT);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("grb mtx import error: %d", info);
        return -1;
    }
    MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->type = MATX_HANDLE_TYPE_GRB_MATRIX;
    MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->valid = 1;
    MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->custom_free_func = NULL;
    return 0;
}

size_t dense_2_grb_z_i8(matx_dense_z_i8_t A)
{
    const uint64_t values_size
        = (uint64_t) A->nrows * (uint64_t) A->ncols * sizeof(*A->data);
    GrB_Info info
        = GxB_Matrix_import_FullC((GrB_Matrix*) &MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl,
                                  GxB_FC64,
                                  A->nrows,
                                  A->ncols,
                                  (void*) &A->data,
                                  values_size,
                                  false,
                                  NULL);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("grb dense mtx import error: %d", info);
        return -1;
    }
    MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->type = MATX_HANDLE_TYPE_GRB_MATRIX;
    MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->valid = 1;
    MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->custom_free_func = NULL;
    return 0;
}

size_t grb_2_dense_z_i8(matx_dense_z_i8_t A)
{
    GrB_Type t;
    uint64_t values_size = 0;
    bool iso = false;
    GrB_Info info
        = GxB_Matrix_export_FullC((GrB_Matrix*) &MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl,
                                  &t,
                                  &A->nrows,
                                  &A->ncols,
                                  (void*) &A->data,
                                  &values_size,
                                  &iso,
                                  NULL);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("grb dense mtx export error: %d", info);
        return -1;
    }
    const uint64_t expected_size
        = (uint64_t) A->nrows * (uint64_t) A->ncols * sizeof(*A->data);
    MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->valid = 0;
    if (values_size != expected_size) {
        MATX_ERROR("grb dense export returned an unexpected size");
        return -1;
    }
    MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->custom_free_func = NULL;
    return 0;
}

size_t grb_2_coo_z_i8(matx_coo_z_i8_t A)
{
    A->ncols = -1;
    A->nrows = -1;
    GrB_Info info = GrB_Matrix_export(A->rows,
                                      A->columns,
                                      (GxB_FC64_t*) A->values,
                                      &A->nrows,
                                      &A->ncols,
                                      &A->nnz,
                                      GrB_COO_FORMAT,
                                      MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("grb 2 coo mtx export error: %d", info);
        return -1;
    }
    MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->custom_free_func = NULL;
    return 0;
}

size_t vec_2_grb_z_i8(matx_vec_z_i8_t v)
{
    const uint64_t values_size = (uint64_t) v->n * sizeof(*v->data);
    if (MATX_HANDLE(v, MATX_HANDLE_TYPE_GRB_VECTOR)->impl != NULL) {
        GrB_Vector_free((GrB_Vector*) &MATX_HANDLE(v, MATX_HANDLE_TYPE_GRB_VECTOR)->impl);
    }
    GrB_Info info
        = GxB_Vector_import_Full((GrB_Vector*) &MATX_HANDLE(v, MATX_HANDLE_TYPE_GRB_VECTOR)->impl,
                                 GxB_FC64,
                                 v->n,
                                 (void*) &v->data,
                                 values_size,
                                 false,
                                 NULL);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("vec 2 grb mtx export error: %d", info);
        return -1;
    }
    MATX_HANDLE(v, MATX_HANDLE_TYPE_GRB_VECTOR)->type = MATX_HANDLE_TYPE_GRB_VECTOR;
    MATX_HANDLE(v, MATX_HANDLE_TYPE_GRB_VECTOR)->valid = 1;
    MATX_HANDLE(v, MATX_HANDLE_TYPE_GRB_VECTOR)->custom_free_func = NULL;
    return 0;
}

size_t grb_2_vec_z_i8(matx_vec_z_i8_t v)
{
    GrB_Type t = GxB_FC64;
    uint64_t vector_size = 0;
    uint64_t values_size = 0;
    bool iso = false;
    GrB_Info info
        = GxB_Vector_export_Full((GrB_Vector*) &MATX_HANDLE(v, MATX_HANDLE_TYPE_GRB_VECTOR)->impl,
                                 &t,
                                 &vector_size,
                                 (void*) &v->data,
                                 &values_size,
                                 &iso,
                                 NULL);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("grb 2 vec export error: %d", info);
        return -1;
    }
    if (vector_size != (uint64_t) v->n || values_size != vector_size * sizeof(*v->data)) {
        MATX_HANDLE(v, MATX_HANDLE_TYPE_GRB_VECTOR)->valid = 0;
        MATX_ERROR("grb vector export returned an unexpected size");
        return -1;
    }
    MATX_HANDLE(v, MATX_HANDLE_TYPE_GRB_VECTOR)->valid = 0;
    MATX_HANDLE(v, MATX_HANDLE_TYPE_GRB_VECTOR)->custom_free_func = NULL;
    return 0;
}

#endif /* MATX_HAVE_GRAPHBLAS */

// ---- COO extraction helpers ----

matx_status_t matx_coo_get_row_d_i8(matx_coo_d_i8_t A, matx_int64_t row_idx, matx_vec_d_i8_t out)
{
    if (!A || !A->rows || !A->columns || !A->values || !out || !out->data
        || out->n != A->ncols || out->stride <= 0
        || row_idx < 0 || row_idx >= A->nrows) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    for (matx_int64_t j = 0; j < A->ncols; ++j)
        out->data[j * out->stride] = 0.0;
    for (matx_int64_t k = 0; k < A->nnz; ++k) {
        if (A->rows[k] == row_idx)
            out->data[A->columns[k] * out->stride] += A->values[k];
    }
    return MATX_OK;
}

matx_status_t matx_coo_get_row_z_i8(matx_coo_z_i8_t A, matx_int64_t row_idx, matx_vec_z_i8_t out)
{
    if (!A || !A->rows || !A->columns || !A->values || !out || !out->data
        || out->n != A->ncols || out->stride <= 0
        || row_idx < 0 || row_idx >= A->nrows) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    for (matx_int64_t j = 0; j < A->ncols; ++j) {
        out->data[j * out->stride].real = 0.0;
        out->data[j * out->stride].imag = 0.0;
    }
    for (matx_int64_t k = 0; k < A->nnz; ++k) {
        if (A->rows[k] == row_idx)
        {
            out->data[A->columns[k] * out->stride].real += A->values[k].real;
            out->data[A->columns[k] * out->stride].imag += A->values[k].imag;
        }
    }
    return MATX_OK;
}

matx_status_t matx_coo_get_col_d_i8(matx_coo_d_i8_t A, matx_int64_t col_idx, matx_vec_d_i8_t out)
{
    if (!A || !A->rows || !A->columns || !A->values || !out || !out->data
        || out->n != A->nrows || out->stride <= 0
        || col_idx < 0 || col_idx >= A->ncols) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    for (matx_int64_t i = 0; i < A->nrows; ++i)
        out->data[i * out->stride] = 0.0;
    for (matx_int64_t k = 0; k < A->nnz; ++k) {
        if (A->columns[k] == col_idx)
            out->data[A->rows[k] * out->stride] += A->values[k];
    }
    return MATX_OK;
}

matx_status_t matx_coo_get_col_z_i8(matx_coo_z_i8_t A, matx_int64_t col_idx, matx_vec_z_i8_t out)
{
    if (!A || !A->rows || !A->columns || !A->values || !out || !out->data
        || out->n != A->nrows || out->stride <= 0
        || col_idx < 0 || col_idx >= A->ncols) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    for (matx_int64_t i = 0; i < A->nrows; ++i) {
        out->data[i * out->stride].real = 0.0;
        out->data[i * out->stride].imag = 0.0;
    }
    for (matx_int64_t k = 0; k < A->nnz; ++k) {
        if (A->columns[k] == col_idx)
        {
            out->data[A->rows[k] * out->stride].real += A->values[k].real;
            out->data[A->rows[k] * out->stride].imag += A->values[k].imag;
        }
    }
    return MATX_OK;
}

static void matx_zero_dense_logical(void* data,
                                    matx_int64_t nrows,
                                    matx_int64_t ncols,
                                    matx_int64_t stride,
                                    matx_layout_t layout,
                                    size_t element_size)
{
    const matx_int64_t minor = (layout == MATX_COL_MAJOR) ? nrows : ncols;
    const size_t elements = (size_t) nrows * (size_t) ncols;
    const size_t single_clear_limit = (size_t) 2 * 1024 * 1024;
    if (stride == minor && elements <= single_clear_limit / element_size) {
        memset(data, 0, elements * element_size);
    } else if (layout == MATX_COL_MAJOR) {
        for (matx_int64_t c = 0; c < ncols; ++c) {
            memset((char*) data + (size_t) c * (size_t) stride * element_size,
                   0, (size_t) nrows * element_size);
        }
    } else {
        for (matx_int64_t r = 0; r < nrows; ++r) {
            memset((char*) data + (size_t) r * (size_t) stride * element_size,
                   0, (size_t) ncols * element_size);
        }
    }
}

matx_status_t matx_coo_to_dense_d_i8(matx_coo_d_i8_t A, matx_dense_d_i8_t out)
{
    if (!A || !A->rows || !A->columns || !A->values || !out || !out->data
        || A->nrows <= 0 || A->ncols <= 0 || A->nnz <= 0
        || out->stride <= 0 || (out->layout != MATX_ROW_MAJOR
                                && out->layout != MATX_COL_MAJOR)) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (out->nrows != A->nrows || out->ncols != A->ncols) {
        MATX_ERROR("%s: dimension mismatch", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_zero_dense_logical(out->data, out->nrows, out->ncols, out->stride,
                            out->layout, sizeof(*out->data));
    for (matx_int64_t k = 0; k < A->nnz; ++k) {
        matx_int64_t r = A->rows[k];
        matx_int64_t c = A->columns[k];
        if (r < 0 || r >= A->nrows || c < 0 || c >= A->ncols)
            return MATX_ERR_INVALID_ARG;
        const matx_int64_t idx = (out->layout == MATX_COL_MAJOR)
            ? r + c * out->stride : r * out->stride + c;
        out->data[idx] += A->values[k];
    }
    return MATX_OK;
}

matx_status_t matx_coo_to_dense_z_i8(matx_coo_z_i8_t A, matx_dense_z_i8_t out)
{
    if (!A || !A->rows || !A->columns || !A->values || !out || !out->data
        || A->nrows <= 0 || A->ncols <= 0 || A->nnz <= 0
        || out->stride <= 0 || (out->layout != MATX_ROW_MAJOR
                                && out->layout != MATX_COL_MAJOR)) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (out->nrows != A->nrows || out->ncols != A->ncols) {
        MATX_ERROR("%s: dimension mismatch", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_zero_dense_logical(out->data, out->nrows, out->ncols, out->stride,
                            out->layout, sizeof(*out->data));
    for (matx_int64_t k = 0; k < A->nnz; ++k) {
        matx_int64_t r = A->rows[k];
        matx_int64_t c = A->columns[k];
        if (r < 0 || r >= A->nrows || c < 0 || c >= A->ncols)
            return MATX_ERR_INVALID_ARG;
        const matx_int64_t idx = (out->layout == MATX_COL_MAJOR)
            ? r + c * out->stride : r * out->stride + c;
        out->data[idx].real += A->values[k].real;
        out->data[idx].imag += A->values[k].imag;
    }
    return MATX_OK;
}
