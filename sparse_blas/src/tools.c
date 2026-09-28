#include "matx/matx_func.h"
#include "matx/matx_log.h"
#include "matx/matx_sparse_compute.h"
#include "matx/matx_types.h"
#include "matx/matx_types_internal.h"

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

static void quick_sort(matx_int64_t* col, matx_double* val, matx_int64_t left, matx_int64_t right)
{
    if (left >= right)
        return;

    matx_int64_t i = left, j = right;
    matx_int64_t pivot = col[(left + right) >> 1];

    while (i <= j) {
        while (col[i] < pivot)
            i++;
        while (col[j] > pivot)
            j--;
        if (i <= j) {
            swap_int(&col[i], &col[j]);
            swap_double(&val[i], &val[j]);
            i++;
            j--;
        }
    }

    if (left < j)
        quick_sort(col, val, left, j);
    if (i < right)
        quick_sort(col, val, i, right);
}

static void sort_row(matx_int64_t* col, matx_double* val, matx_int64_t len)
{
    if (len < 32)
        insertion_sort(col, val, len);
    else
        quick_sort(col, val, 0, len - 1);
}

int coo_to_csr_optimized(matx_int64_t nrows,
                         matx_int64_t ncols,
                         matx_int64_t nnz,
                         const matx_int64_t* coo_row,
                         const matx_int64_t* coo_col,
                         const matx_double* coo_val,
                         csr_matrix* csr)
{
    csr->nrows = nrows;
    csr->ncols = ncols;

    csr->row_ptr = (matx_int64_t*) calloc(nrows + 1, sizeof(matx_int64_t));
    csr->col_ind = (matx_int64_t*) malloc(nnz * sizeof(matx_int64_t));
    csr->val = (matx_double*) malloc(nnz * sizeof(matx_double));

    if (!csr->row_ptr || !csr->col_ind || !csr->val) {
        MATX_ERROR("NULL POINTER");
        return -1;
    }
    for (matx_int64_t i = 0; i < nnz; ++i) {
        csr->row_ptr[coo_row[i] + 1]++;
    }

    for (matx_int64_t i = 0; i < nrows; ++i) {
        csr->row_ptr[i + 1] += csr->row_ptr[i];
    }

    matx_int64_t* offset = (matx_int64_t*) malloc(nrows * sizeof(matx_int64_t));
    if (offset == NULL || csr->row_ptr == NULL || nrows <= 0) {
        MATX_ERROR("NULL POINTER or offset is 0");
        return -1;
    }
    memcpy(offset, csr->row_ptr, nrows * sizeof(matx_int64_t));

    for (int i = 0; i < nnz; ++i) {
        int r = coo_row[i];
        int dst = offset[r]++;

        csr->col_ind[dst] = coo_col[i];
        csr->val[dst] = coo_val[i];
    }

    free(offset);

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

    if (csr->col_ind == NULL) {
        return -1;
    }
    csr->col_ind = (matx_int64_t*) realloc(csr->col_ind, new_nnz * sizeof(matx_int64_t));
    if (csr->val == NULL) {
        return -1;
    }
    csr->val = (matx_double*) realloc(csr->val, new_nnz * sizeof(matx_double));

    return 0;
}

size_t coo_2_aocl_d_i8(matx_coo_d_i8_t A)
{
#if MATX_HAVE_AOCL_SPARSE

    aoclsparse_matrix csr;
    csr_matrix csr_m;
    matx_int64_t ret
        = coo_to_csr_optimized(A->nrows, A->ncols, A->nnz, A->rows, A->columns, A->values, &csr_m);
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
        MATX_ERROR("aocl create dcsr. error: %d", st);
        return -1;
    }

    st = aoclsparse_optimize(csr);

    if (st != aoclsparse_status_success) {
        MATX_ERROR("aocl op mtx error: %d", st);
        return -1;
    }
    MATX_HANDLE(A, MATX_HANDLE_TYPE_AOCL_MATRIX)->impl = csr;
    MATX_HANDLE(A, MATX_HANDLE_TYPE_AOCL_MATRIX)->type = MATX_HANDLE_TYPE_AOCL_MATRIX;
    MATX_HANDLE(A, MATX_HANDLE_TYPE_AOCL_MATRIX)->valid = 1;
    MATX_HANDLE(A, MATX_HANDLE_TYPE_AOCL_MATRIX)->custom_free_func = &free_aocl_matrix;
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
    GrB_Matrix_free(MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl);
    GrB_Info info
        = GxB_Matrix_import_FullC((GrB_Matrix*) &MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl,
                                  GrB_FP64,
                                  A->nrows,
                                  A->ncols,
                                  (void*) &A->data,
                                  A->nrows * A->ncols,
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
    matx_int64_t s = 0;
    bool iso = false;
    GrB_Info info
        = GxB_Matrix_export_FullC((GrB_Matrix*) &MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl,
                                  &t,
                                  &A->nrows,
                                  &A->ncols,
                                  (void*) &A->data,
                                  &s,
                                  &iso,
                                  NULL);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("grb export fullc, %d", info);
        return -1;
    }
    MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->valid = 1;
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
    GrB_Info info
        = GxB_Vector_import_Full((GrB_Vector*) &MATX_HANDLE(v, MATX_HANDLE_TYPE_GRB_VECTOR)->impl,
                                 GrB_FP64,
                                 v->n,
                                 (void*) &v->data,
                                 v->n,
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
    GrB_Info info
        = GxB_Vector_export_Full((GrB_Vector*) &MATX_HANDLE(v, MATX_HANDLE_TYPE_GRB_VECTOR)->impl,
                                 &t,
                                 &v->n,
                                 (void*) &v->data,
                                 &v->n,
                                 false,
                                 NULL);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("grb vector export error: %d", info);
        return -1;
    }
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
    GrB_Info info
        = GxB_Matrix_import_FullC((GrB_Matrix*) &MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl,
                                  GxB_FC64,
                                  A->nrows,
                                  A->ncols,
                                  (void*) &A->data,
                                  A->nrows * A->ncols,
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
    matx_int64_t s = 0;
    bool iso = false;
    GrB_Info info
        = GxB_Matrix_export_FullC((GrB_Matrix*) &MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl,
                                  &t,
                                  &A->nrows,
                                  &A->ncols,
                                  (void*) &A->data,
                                  &s,
                                  &iso,
                                  NULL);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("grb dense mtx export error: %d", info);
        return -1;
    }
    MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->custom_free_func = NULL;
    return 0;
}

size_t grb_2_coo_z_i8(matx_coo_z_i8_t A)
{
    A->ncols = -1;
    A->nrows = -1;
    A->nnz = 16;
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
    GrB_Vector_free(MATX_HANDLE(v, MATX_HANDLE_TYPE_GRB_VECTOR)->impl);
    GrB_Info info
        = GxB_Vector_import_Full((GrB_Vector*) &MATX_HANDLE(v, MATX_HANDLE_TYPE_GRB_VECTOR)->impl,
                                 GxB_FC64,
                                 v->n,
                                 (void*) &v->data,
                                 v->n,
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
    bool iso = false;
    GrB_Info info
        = GxB_Vector_export_Full((GrB_Vector*) &MATX_HANDLE(v, MATX_HANDLE_TYPE_GRB_VECTOR)->impl,
                                 &t,
                                 &v->n,
                                 (void*) &v->data,
                                 &v->n,
                                 false,
                                 NULL);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("grb 2 vec export error: %d", info);
        return -1;
    }
    MATX_HANDLE(v, MATX_HANDLE_TYPE_GRB_VECTOR)->custom_free_func = NULL;
    return 0;
}

#endif /* MATX_HAVE_GRAPHBLAS */

// ---- COO extraction helpers ----

matx_status_t matx_coo_get_row_d_i8(matx_coo_d_i8_t A, matx_int64_t row_idx, matx_vec_d_i8_t out)
{
    if (!A || !out || !out->data || row_idx < 0 || row_idx >= A->nrows) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    for (matx_int64_t j = 0; j < A->ncols; ++j)
        out->data[j * out->stride] = 0.0;
    for (matx_int64_t k = 0; k < A->nnz; ++k) {
        if (A->rows[k] == row_idx)
            out->data[A->columns[k] * out->stride] = A->values[k];
    }
    return MATX_OK;
}

matx_status_t matx_coo_get_row_z_i8(matx_coo_z_i8_t A, matx_int64_t row_idx, matx_vec_z_i8_t out)
{
    if (!A || !out || !out->data || row_idx < 0 || row_idx >= A->nrows) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    for (matx_int64_t j = 0; j < A->ncols; ++j) {
        out->data[j * out->stride].real = 0.0;
        out->data[j * out->stride].imag = 0.0;
    }
    for (matx_int64_t k = 0; k < A->nnz; ++k) {
        if (A->rows[k] == row_idx)
            out->data[A->columns[k] * out->stride] = A->values[k];
    }
    return MATX_OK;
}

matx_status_t matx_coo_get_col_d_i8(matx_coo_d_i8_t A, matx_int64_t col_idx, matx_vec_d_i8_t out)
{
    if (!A || !out || !out->data || col_idx < 0 || col_idx >= A->ncols) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    for (matx_int64_t i = 0; i < A->nrows; ++i)
        out->data[i * out->stride] = 0.0;
    for (matx_int64_t k = 0; k < A->nnz; ++k) {
        if (A->columns[k] == col_idx)
            out->data[A->rows[k] * out->stride] = A->values[k];
    }
    return MATX_OK;
}

matx_status_t matx_coo_get_col_z_i8(matx_coo_z_i8_t A, matx_int64_t col_idx, matx_vec_z_i8_t out)
{
    if (!A || !out || !out->data || col_idx < 0 || col_idx >= A->ncols) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    for (matx_int64_t i = 0; i < A->nrows; ++i) {
        out->data[i * out->stride].real = 0.0;
        out->data[i * out->stride].imag = 0.0;
    }
    for (matx_int64_t k = 0; k < A->nnz; ++k) {
        if (A->columns[k] == col_idx)
            out->data[A->rows[k] * out->stride] = A->values[k];
    }
    return MATX_OK;
}

matx_status_t matx_coo_to_dense_d_i8(matx_coo_d_i8_t A, matx_dense_d_i8_t out)
{
    if (!A || !out || !out->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (out->nrows != A->nrows || out->ncols != A->ncols) {
        MATX_ERROR("%s: dimension mismatch", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_int64_t total = out->nrows * out->ncols;
    for (matx_int64_t i = 0; i < total; ++i)
        out->data[i] = 0.0;
    for (matx_int64_t k = 0; k < A->nnz; ++k) {
        matx_int64_t r = A->rows[k];
        matx_int64_t c = A->columns[k];
        out->data[r + c * out->nrows] = A->values[k];
    }
    return MATX_OK;
}

matx_status_t matx_coo_to_dense_z_i8(matx_coo_z_i8_t A, matx_dense_z_i8_t out)
{
    if (!A || !out || !out->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (out->nrows != A->nrows || out->ncols != A->ncols) {
        MATX_ERROR("%s: dimension mismatch", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_int64_t total = out->nrows * out->ncols;
    for (matx_int64_t i = 0; i < total; ++i) {
        out->data[i].real = 0.0;
        out->data[i].imag = 0.0;
    }
    for (matx_int64_t k = 0; k < A->nnz; ++k) {
        matx_int64_t r = A->rows[k];
        matx_int64_t c = A->columns[k];
        out->data[r + c * out->nrows] = A->values[k];
    }
    return MATX_OK;
}
