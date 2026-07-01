#include "matx/matx_log.h"
#include "matx/matx_sparse_compute.h"

#include <limits.h>
#include <math.h>
#include <memory.h>
#include <stdlib.h>

#if MATX_HAVE_AOCL_SPARSE
#include "aoclsparse.h"
#include "aoclsparse_convert.h"
#include "aoclsparse_types.h"
#endif

// y = \alpha \, op(A) \, x + \beta \, y,
static matx_status_t ref_spmv_z_i8_aocl(matx_complex_d_t alpha,
                                 matx_coo_z_i8_t A,
                                 matx_vec_z_i8_t x,
                                 matx_complex_d_t beta,
                                 matx_vec_z_i8_t y)
{
#if MATX_HAVE_AOCL_SPARSE
    if (!A || !x || !y)
        return MATX_ERR_INVALID_ARG;

    if (A->ncols != x->n || A->nrows != y->n)
        return MATX_ERR_INVALID_ARG;

    if (A->handle_aocl.valid <= 0) {
        if (coo_2_aocl_z_i8(A) != 0)
            return MATX_ERR_INTERNAL;
    }

    aoclsparse_mat_descr descr;
    aoclsparse_create_mat_descr(&descr);

    aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_general);
    aoclsparse_set_mat_diag_type(descr, aoclsparse_diag_type_non_unit);

    aoclsparse_double_complex a = {alpha.real, alpha.imag};
    aoclsparse_double_complex b = {beta.real, beta.imag};

    aoclsparse_status status = aoclsparse_zmv(aoclsparse_operation_none,
                                              &a,
                                              (aoclsparse_matrix) A->handle_aocl.impl,
                                              descr,
                                              (aoclsparse_double_complex*) x->data,
                                              &b,
                                              (aoclsparse_double_complex*) y->data);

    aoclsparse_destroy_mat_descr(descr);

    if (status != aoclsparse_status_success) {
        MATX_ERROR("aoclsparse_zmv error: %d", status);
        return MATX_ERR_INTERNAL;
    }

#endif
    return MATX_OK;
}

//    C = \alpha \, op(A) \, B + \beta \, C,
static matx_status_t ref_spmm_z_i8_aocl(matx_complex_d_t alpha,
                                 matx_coo_z_i8_t A,
                                 const matx_dense_z_i8_t B,
                                 matx_complex_d_t beta,
                                 matx_dense_z_i8_t C)
{
#if MATX_HAVE_AOCL_SPARSE
    if (!A || !B || !C)
        return MATX_ERR_INVALID_ARG;

    if (A->handle_aocl.valid <= 0) {
        if (coo_2_aocl_z_i8(A) != 0)
            return MATX_ERR_INTERNAL;
    }

    aoclsparse_mat_descr descr;
    aoclsparse_create_mat_descr(&descr);

    aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_general);

    aoclsparse_double_complex a = {alpha.real, alpha.imag};
    aoclsparse_double_complex b = {beta.real, beta.imag};

    aoclsparse_status status = aoclsparse_zcsrmm(aoclsparse_operation_none,
                                                 a,
                                                 (aoclsparse_matrix) A->handle_aocl.impl,
                                                 descr,
                                                 aoclsparse_order_row,
                                                 (aoclsparse_double_complex*) B->data,
                                                 B->ncols,
                                                 B->stride,
                                                 b,
                                                 (aoclsparse_double_complex*) C->data,
                                                 C->ncols);
    aoclsparse_destroy_mat_descr(descr);
    if (status != aoclsparse_status_success) {
        MATX_ERROR("aoclsparse_zcsrmm error: %d", status);
        return MATX_ERR_INTERNAL;
    }
#endif
    return MATX_OK;
}

// y = \alpha \, op(A) \, x + \beta \, y
static matx_status_t ref_spmv_d_i8_aocl(
    matx_double alpha, matx_coo_d_i8_t A, matx_vec_d_i8_t x, matx_double beta, matx_vec_d_i8_t y)
{
#if MATX_HAVE_AOCL_SPARSE

    if (!A || !x || !y)
        return MATX_ERR_INVALID_ARG;

    if (A->handle_aocl.valid <= 0) {
        if (coo_2_aocl_d_i8(A) != 0)
            return MATX_ERR_INTERNAL;
    }

    aoclsparse_mat_descr descr;
    aoclsparse_create_mat_descr(&descr);

    aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_general);
    aoclsparse_set_mat_diag_type(descr, aoclsparse_diag_type_non_unit);

    aoclsparse_status status = aoclsparse_set_mv_hint(A->handle_aocl.impl,
                                                      aoclsparse_operation_none,
                                                      descr,
                                                      1);
    if (status != aoclsparse_status_success) {
        MATX_ERROR("aoclsparse_set_mv_hint error: %d", status);
        return MATX_ERR_INTERNAL;
    }
    status = aoclsparse_dmv(aoclsparse_operation_none,
                            &alpha,
                            (aoclsparse_matrix) A->handle_aocl.impl,
                            descr,
                            x->data,
                            &beta,
                            y->data);
    aoclsparse_destroy_mat_descr(descr);
    if (status != aoclsparse_status_success) {
        MATX_ERROR("aoclsparse_dmv error: %d", status);
        return MATX_ERR_INTERNAL;
    }
#endif

    return MATX_OK;
}
//C = α * A * B + β * C
static matx_status_t ref_spmm_d_i8_aocl(matx_double alpha,
                                 matx_coo_d_i8_t A,
                                 matx_dense_d_i8_t B,
                                 matx_double beta,
                                 matx_dense_d_i8_t C)
{
#if MATX_HAVE_AOCL_SPARSE
    if (!A || !B || !C)
        return MATX_ERR_INVALID_ARG;

    if (A->handle_aocl.valid <= 0) {
        if (coo_2_aocl_d_i8(A) != 0)
            return MATX_ERR_INTERNAL;
    }

    aoclsparse_mat_descr descr;
    aoclsparse_create_mat_descr(&descr);

    aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_general);
    aoclsparse_set_mat_diag_type(descr, aoclsparse_diag_type_non_unit);

    aoclsparse_status status = aoclsparse_dcsrmm(aoclsparse_operation_none,
                                                 alpha,
                                                 (aoclsparse_matrix) A->handle_aocl.impl,
                                                 descr,
                                                 aoclsparse_order_row,
                                                 B->data,
                                                 B->ncols,
                                                 B->stride,
                                                 beta,
                                                 C->data,
                                                 C->stride);

    aoclsparse_destroy_mat_descr(descr);
    if (status != aoclsparse_status_success) {
        MATX_ERROR("aoclsparse_dcsrmm error: %d", status);
        return MATX_ERR_INTERNAL;
    }

#endif
    return MATX_OK;
}

// C := α · op(A) · op(B) + β · C
static matx_status_t ref_dsp2md_d_i8_aocl(
    matx_double alpha, matx_coo_d_i8_t A, matx_coo_d_i8_t B, matx_double beta, matx_dense_d_i8_t C)
{
#if MATX_HAVE_AOCL_SPARSE
    if (!A || !B || !C)
        return MATX_ERR_INVALID_ARG;

    if (A->handle_aocl.valid <= 0) {
        if (coo_2_aocl_d_i8(A) != 0)
            return MATX_ERR_INTERNAL;
    }

    if (B->handle_aocl.valid <= 0) {
        if (coo_2_aocl_d_i8(B) != 0)
            return MATX_ERR_INTERNAL;
    }

    aoclsparse_mat_descr descr;
    aoclsparse_create_mat_descr(&descr);

    aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_general);
    aoclsparse_set_mat_diag_type(descr, aoclsparse_diag_type_non_unit);

    aoclsparse_status status = aoclsparse_dsp2md(aoclsparse_operation_none,
                                                 descr,
                                                 (aoclsparse_matrix) A->handle_aocl.impl,
                                                 aoclsparse_operation_none,
                                                 descr,
                                                 (aoclsparse_matrix) B->handle_aocl.impl,
                                                 alpha,
                                                 beta,
                                                 C->data,
                                                 aoclsparse_order_row,
                                                 C->stride);

    aoclsparse_destroy_mat_descr(descr);

    if (status != aoclsparse_status_success) {
        MATX_ERROR("aoclsparse_dsp2md error: %d", status);
        return MATX_ERR_INTERNAL;
    }
#endif
    return MATX_OK;
}

// C := α · op(A) · op(B) + β · C
static matx_status_t ref_zsp2md_z_i8_aocl(matx_complex_d_t alpha,
                                   matx_coo_z_i8_t A,
                                   matx_coo_z_i8_t B,
                                   matx_complex_d_t beta,
                                   matx_dense_z_i8_t C)
{
#if MATX_HAVE_AOCL_SPARSE
    if (!A || !B || !C)
        return MATX_ERR_INVALID_ARG;

    if (A->handle_aocl.valid <= 0) {
        if (coo_2_aocl_z_i8(A) != 0)
            return MATX_ERR_INTERNAL;
    }

    if (B->handle_aocl.valid <= 0) {
        if (coo_2_aocl_z_i8(B) != 0)
            return MATX_ERR_INTERNAL;
    }

    aoclsparse_mat_descr descr;
    aoclsparse_create_mat_descr(&descr);

    aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_general);
    aoclsparse_set_mat_diag_type(descr, aoclsparse_diag_type_non_unit);

    aoclsparse_double_complex a = {alpha.real, alpha.imag};
    aoclsparse_double_complex b = {beta.real, beta.imag};

    aoclsparse_status status = aoclsparse_zsp2md(aoclsparse_operation_none,
                                                 descr,
                                                 (aoclsparse_matrix) A->handle_aocl.impl,
                                                 aoclsparse_operation_none,
                                                 descr,
                                                 (aoclsparse_matrix) B->handle_aocl.impl,
                                                 a,
                                                 b,
                                                 (void*) C->data,
                                                 aoclsparse_order_row,
                                                 C->stride);

    aoclsparse_destroy_mat_descr(descr);

    if (status != aoclsparse_status_success) {
        MATX_ERROR("aoclsparse_zsp2md error: %d", status);
        return MATX_ERR_INTERNAL;
    }
#endif
    return MATX_OK;
}

static matx_status_t ref_transpose_d_i8_aocl(matx_coo_d_i8_t A, matx_coo_d_i8_t out)
{
    if (!A)
        return MATX_ERR_INVALID_ARG;
    out->nrows = A->ncols;
    out->ncols = A->nrows;
    out->nnz = A->nnz;
    memcpy(out->rows, A->columns, sizeof(matx_int64_t) * A->nnz);
    memcpy(out->columns, A->rows, sizeof(matx_int64_t) * A->nnz);
    memcpy(out->values, A->values, sizeof(matx_double) * A->nnz);
    return MATX_OK;
}

static matx_status_t ref_transpose_z_i8_aocl(matx_coo_z_i8_t A, matx_coo_z_i8_t out)
{
    if (!A)
        return MATX_ERR_INVALID_ARG;
    out->nrows = A->ncols;
    out->ncols = A->nrows;
    out->nnz = A->nnz;
    memcpy(out->rows, A->columns, sizeof(matx_int64_t) * A->nnz);
    memcpy(out->columns, A->rows, sizeof(matx_int64_t) * A->nnz);
    memcpy(out->values, A->values, sizeof(matx_complex_d_t) * A->nnz);
    return MATX_OK;
}

static matx_status_t ref_conj_trans_z_i8_aocl(matx_coo_z_i8_t A, matx_coo_z_i8_t out)
{
#if MATX_HAVE_AOCL_SPARSE
    if (!A || !out)
        return MATX_ERR_INVALID_ARG;

    if (A->handle_aocl.valid <= 0) {
        if (coo_2_aocl_z_i8(A) != 0)
            return MATX_ERR_INTERNAL;
    }

    aoclsparse_status status = aoclsparse_convert_csr(A->handle_aocl.impl,
                                                      aoclsparse_operation_conjugate_transpose,
                                                      out->handle_aocl.impl);

    if (status != aoclsparse_status_success) {
        MATX_ERROR("aoclsparse_convert_csr error: %d", status);
        return MATX_ERR_INTERNAL;
    }
    aocl_2_coo_z_i8(out);
#endif
    return MATX_OK;
}

static matx_status_t ref_finalize_aocl()
{
    return MATX_OK;
}

// ---- Sparse matrix norms ----

static matx_status_t ref_norm1_mat_aocl(matx_coo_d_i8_t A, matx_double* out)
{
    if (!A || !out)
        return MATX_ERR_INVALID_ARG;
    matx_double* col_sums = (matx_double*) calloc((size_t) A->ncols, sizeof(matx_double));
    if (!col_sums)
        return MATX_ERR_OUT_OF_MEMORY;
    for (matx_int64_t i = 0; i < A->nnz; ++i)
        col_sums[A->columns[i]] += fabs(A->values[i]);
    matx_double max_col = 0.0;
    for (matx_int64_t j = 0; j < A->ncols; ++j)
        if (col_sums[j] > max_col)
            max_col = col_sums[j];
    free(col_sums);
    *out = max_col;
    return MATX_OK;
}

static matx_status_t ref_norminf_mat_aocl(matx_coo_d_i8_t A, matx_double* out)
{
    if (!A || !out)
        return MATX_ERR_INVALID_ARG;
    matx_double* row_sums = (matx_double*) calloc((size_t) A->nrows, sizeof(matx_double));
    if (!row_sums)
        return MATX_ERR_OUT_OF_MEMORY;
    for (matx_int64_t i = 0; i < A->nnz; ++i)
        row_sums[A->rows[i]] += fabs(A->values[i]);
    matx_double max_row = 0.0;
    for (matx_int64_t i = 0; i < A->nrows; ++i)
        if (row_sums[i] > max_row)
            max_row = row_sums[i];
    free(row_sums);
    *out = max_row;
    return MATX_OK;
}

static matx_status_t ref_normfro_mat_aocl(matx_coo_d_i8_t A, matx_double* out)
{
    if (!A || !out)
        return MATX_ERR_INVALID_ARG;
    matx_double sum = 0.0;
    for (matx_int64_t i = 0; i < A->nnz; ++i)
        sum += A->values[i] * A->values[i];
    *out = sqrt(sum);
    return MATX_OK;
}

// ---- Sparse matrix norms (c64) ----

static matx_status_t ref_norm1_mat_z_i8_aocl(matx_coo_z_i8_t A, matx_double* out)
{
    if (!A || !out)
        return MATX_ERR_INVALID_ARG;
    matx_double* col_sums = (matx_double*) calloc((size_t) A->ncols, sizeof(matx_double));
    if (!col_sums)
        return MATX_ERR_OUT_OF_MEMORY;
    for (matx_int64_t i = 0; i < A->nnz; ++i) {
        matx_double re = A->values[i].real;
        matx_double im = A->values[i].imag;
        col_sums[A->columns[i]] += sqrt(re * re + im * im);
    }
    matx_double max_col = 0.0;
    for (matx_int64_t j = 0; j < A->ncols; ++j)
        if (col_sums[j] > max_col)
            max_col = col_sums[j];
    free(col_sums);
    *out = max_col;
    return MATX_OK;
}

static matx_status_t ref_norminf_mat_z_i8_aocl(matx_coo_z_i8_t A, matx_double* out)
{
    if (!A || !out)
        return MATX_ERR_INVALID_ARG;
    matx_double* row_sums = (matx_double*) calloc((size_t) A->nrows, sizeof(matx_double));
    if (!row_sums)
        return MATX_ERR_OUT_OF_MEMORY;
    for (matx_int64_t i = 0; i < A->nnz; ++i) {
        matx_double re = A->values[i].real;
        matx_double im = A->values[i].imag;
        row_sums[A->rows[i]] += sqrt(re * re + im * im);
    }
    matx_double max_row = 0.0;
    for (matx_int64_t i = 0; i < A->nrows; ++i)
        if (row_sums[i] > max_row)
            max_row = row_sums[i];
    free(row_sums);
    *out = max_row;
    return MATX_OK;
}

static matx_status_t ref_normfro_mat_z_i8_aocl(matx_coo_z_i8_t A, matx_double* out)
{
    if (!A || !out)
        return MATX_ERR_INVALID_ARG;
    matx_double sum = 0.0;
    for (matx_int64_t i = 0; i < A->nnz; ++i) {
        matx_double re = A->values[i].real;
        matx_double im = A->values[i].imag;
        sum += re * re + im * im;
    }
    *out = sqrt(sum);
    return MATX_OK;
}
// ---- Sparse-sparse addition ----

static matx_status_t ref_spadd_d_i8_aocl(
    matx_double alpha, matx_coo_d_i8_t A, matx_double beta, matx_coo_d_i8_t B, matx_coo_d_i8_t out)
{
    if (!A || !B || !out)
        return MATX_ERR_INVALID_ARG;
    if (A->nrows != B->nrows || A->ncols != B->ncols)
        return MATX_ERR_INVALID_ARG;

    out->nrows = A->nrows;
    out->ncols = A->ncols;
    out->nnz = A->nnz + B->nnz;
    memcpy(out->rows, A->rows, sizeof(matx_int64_t) * A->nnz);
    memcpy(out->rows + A->nnz, B->rows, sizeof(matx_int64_t) * B->nnz);
    memcpy(out->columns, A->columns, sizeof(matx_int64_t) * A->nnz);
    memcpy(out->columns + A->nnz, B->columns, sizeof(matx_int64_t) * B->nnz);
    for (matx_int64_t i = 0; i < A->nnz; ++i)
        out->values[i] = alpha * A->values[i];
    for (matx_int64_t i = 0; i < B->nnz; ++i)
        out->values[A->nnz + i] = beta * B->values[i];
    return MATX_OK;
}

static matx_status_t ref_spadd_z_i8_aocl(matx_complex_d_t alpha,
                                  matx_coo_z_i8_t A,
                                  matx_complex_d_t beta,
                                  matx_coo_z_i8_t B,
                                  matx_coo_z_i8_t out)
{
    if (!A || !B || !out)
        return MATX_ERR_INVALID_ARG;
    if (A->nrows != B->nrows || A->ncols != B->ncols)
        return MATX_ERR_INVALID_ARG;

    out->nrows = A->nrows;
    out->ncols = A->ncols;
    out->nnz = A->nnz + B->nnz;
    memcpy(out->rows, A->rows, sizeof(matx_int64_t) * A->nnz);
    memcpy(out->rows + A->nnz, B->rows, sizeof(matx_int64_t) * B->nnz);
    memcpy(out->columns, A->columns, sizeof(matx_int64_t) * A->nnz);
    memcpy(out->columns + A->nnz, B->columns, sizeof(matx_int64_t) * B->nnz);
    for (matx_int64_t i = 0; i < A->nnz; ++i) {
        out->values[i].real = alpha.real * A->values[i].real - alpha.imag * A->values[i].imag;
        out->values[i].imag = alpha.real * A->values[i].imag + alpha.imag * A->values[i].real;
    }
    for (matx_int64_t i = 0; i < B->nnz; ++i) {
        out->values[A->nnz + i].real = beta.real * B->values[i].real
                                       - beta.imag * B->values[i].imag;
        out->values[A->nnz + i].imag = beta.real * B->values[i].imag
                                       + beta.imag * B->values[i].real;
    }
    return MATX_OK;
}

// ---- Non-zero count per row/column ----

static matx_status_t ref_spnnz_rows_d_i8_aocl(matx_coo_d_i8_t A, matx_vec_d_i8_t out)
{
    if (!A || !out || !A->values)
        return MATX_ERR_INVALID_ARG;
    memset(out->data, 0, sizeof(matx_double) * (size_t) A->nrows);
    for (matx_int64_t i = 0; i < A->nnz; ++i)
        out->data[A->rows[i]] += 1.0;
    return MATX_OK;
}

static matx_status_t ref_spnnz_cols_d_i8_aocl(matx_coo_d_i8_t A, matx_vec_d_i8_t out)
{
    if (!A || !out || !A->values)
        return MATX_ERR_INVALID_ARG;
    memset(out->data, 0, sizeof(matx_double) * (size_t) A->ncols);
    for (matx_int64_t i = 0; i < A->nnz; ++i)
        out->data[A->columns[i]] += 1.0;
    return MATX_OK;
}

static matx_status_t ref_spnnz_rows_z_i8_aocl(matx_coo_z_i8_t A, matx_vec_z_i8_t out)
{
    if (!A || !out || !A->values)
        return MATX_ERR_INVALID_ARG;
    for (matx_int64_t i = 0; i < A->nrows; ++i) {
        out->data[i].real = 0.0;
        out->data[i].imag = 0.0;
    }
    for (matx_int64_t i = 0; i < A->nnz; ++i)
        out->data[A->rows[i]].real += 1.0;
    return MATX_OK;
}

static matx_status_t ref_spnnz_cols_z_i8_aocl(matx_coo_z_i8_t A, matx_vec_z_i8_t out)
{
    if (!A || !out || !A->values)
        return MATX_ERR_INVALID_ARG;
    for (matx_int64_t j = 0; j < A->ncols; ++j) {
        out->data[j].real = 0.0;
        out->data[j].imag = 0.0;
    }
    for (matx_int64_t i = 0; i < A->nnz; ++i)
        out->data[A->columns[i]].real += 1.0;
    return MATX_OK;
}

// ---- Row / column sums ----

static matx_status_t ref_sprowsums_d_i8_aocl(matx_coo_d_i8_t A, matx_vec_d_i8_t out)
{
    if (!A || !out || !A->values)
        return MATX_ERR_INVALID_ARG;
    memset(out->data, 0, sizeof(matx_double) * (size_t) A->nrows);
    for (matx_int64_t i = 0; i < A->nnz; ++i)
        out->data[A->rows[i]] += fabs(A->values[i]);
    return MATX_OK;
}

static matx_status_t ref_spcolsums_d_i8_aocl(matx_coo_d_i8_t A, matx_vec_d_i8_t out)
{
    if (!A || !out || !A->values)
        return MATX_ERR_INVALID_ARG;
    memset(out->data, 0, sizeof(matx_double) * (size_t) A->ncols);
    for (matx_int64_t i = 0; i < A->nnz; ++i)
        out->data[A->columns[i]] += fabs(A->values[i]);
    return MATX_OK;
}

static matx_status_t ref_sprowsums_z_i8_aocl(matx_coo_z_i8_t A, matx_vec_z_i8_t out)
{
    if (!A || !out || !A->values)
        return MATX_ERR_INVALID_ARG;
    for (matx_int64_t i = 0; i < A->nrows; ++i) {
        out->data[i].real = 0.0;
        out->data[i].imag = 0.0;
    }
    for (matx_int64_t k = 0; k < A->nnz; ++k) {
        matx_double re = A->values[k].real;
        matx_double im = A->values[k].imag;
        out->data[A->rows[k]].real += sqrt(re * re + im * im);
    }
    return MATX_OK;
}

static matx_status_t ref_spcolsums_z_i8_aocl(matx_coo_z_i8_t A, matx_vec_z_i8_t out)
{
    if (!A || !out || !A->values)
        return MATX_ERR_INVALID_ARG;
    for (matx_int64_t j = 0; j < A->ncols; ++j) {
        out->data[j].real = 0.0;
        out->data[j].imag = 0.0;
    }
    for (matx_int64_t k = 0; k < A->nnz; ++k) {
        matx_double re = A->values[k].real;
        matx_double im = A->values[k].imag;
        out->data[A->columns[k]].real += sqrt(re * re + im * im);
    }
    return MATX_OK;
}

// ---- Diagonal extraction ----

static matx_status_t ref_spdiag_d_i8_aocl(matx_coo_d_i8_t A, matx_int64_t offset, matx_vec_d_i8_t out)
{
    if (!A || !out)
        return MATX_ERR_INVALID_ARG;
    matx_int64_t diag_len = (offset >= 0)
                                ? ((A->ncols - offset < A->nrows) ? A->ncols - offset : A->nrows)
                                : ((A->nrows + offset < A->ncols) ? A->nrows + offset : A->ncols);
    if (diag_len < 0)
        diag_len = 0;
    if (out->n < diag_len)
        return MATX_ERR_INVALID_ARG;
    memset(out->data, 0, sizeof(matx_double) * (size_t) diag_len);
    for (matx_int64_t i = 0; i < A->nnz; ++i) {
        matx_int64_t d = A->rows[i] - A->columns[i];
        if (d == offset) {
            matx_int64_t idx = (offset >= 0) ? A->rows[i] : A->columns[i];
            if (idx >= 0 && idx < diag_len)
                out->data[idx] = A->values[i];
        }
    }
    return MATX_OK;
}

static matx_status_t ref_spdiag_z_i8_aocl(matx_coo_z_i8_t A, matx_int64_t offset, matx_vec_z_i8_t out)
{
    if (!A || !out)
        return MATX_ERR_INVALID_ARG;
    matx_int64_t diag_len = (offset >= 0)
                                ? ((A->ncols - offset < A->nrows) ? A->ncols - offset : A->nrows)
                                : ((A->nrows + offset < A->ncols) ? A->nrows + offset : A->ncols);
    if (diag_len < 0)
        diag_len = 0;
    if (out->n < diag_len)
        return MATX_ERR_INVALID_ARG;
    for (matx_int64_t i = 0; i < diag_len; ++i) {
        out->data[i].real = 0.0;
        out->data[i].imag = 0.0;
    }
    for (matx_int64_t i = 0; i < A->nnz; ++i) {
        matx_int64_t d = A->rows[i] - A->columns[i];
        if (d == offset) {
            matx_int64_t idx = (offset >= 0) ? A->rows[i] : A->columns[i];
            if (idx >= 0 && idx < diag_len)
                out->data[idx] = A->values[i];
        }
    }
    return MATX_OK;
}

// ---- In-place scaling ----

static matx_status_t ref_scale_rows_d_i8_aocl(matx_coo_d_i8_t A, const matx_vec_d_i8_t s)
{
    if (!A || !s || !A->values || !s->data)
        return MATX_ERR_INVALID_ARG;
    if (s->n < A->nrows)
        return MATX_ERR_INVALID_ARG;
    for (matx_int64_t i = 0; i < A->nnz; ++i)
        A->values[i] *= s->data[A->rows[i]];
    return MATX_OK;
}

static matx_status_t ref_scale_cols_d_i8_aocl(matx_coo_d_i8_t A, const matx_vec_d_i8_t s)
{
    if (!A || !s || !A->values || !s->data)
        return MATX_ERR_INVALID_ARG;
    if (s->n < A->ncols)
        return MATX_ERR_INVALID_ARG;
    for (matx_int64_t i = 0; i < A->nnz; ++i)
        A->values[i] *= s->data[A->columns[i]];
    return MATX_OK;
}

static matx_status_t ref_scale_rows_z_i8_aocl(matx_coo_z_i8_t A, const matx_vec_z_i8_t s)
{
    if (!A || !s || !A->values || !s->data)
        return MATX_ERR_INVALID_ARG;
    if (s->n < A->nrows)
        return MATX_ERR_INVALID_ARG;
    for (matx_int64_t i = 0; i < A->nnz; ++i) {
        matx_double sr = s->data[A->rows[i]].real;
        matx_double si = s->data[A->rows[i]].imag;
        matx_double ar = A->values[i].real;
        matx_double ai = A->values[i].imag;
        A->values[i].real = ar * sr - ai * si;
        A->values[i].imag = ar * si + ai * sr;
    }
    return MATX_OK;
}

static matx_status_t ref_scale_cols_z_i8_aocl(matx_coo_z_i8_t A, const matx_vec_z_i8_t s)
{
    if (!A || !s || !A->values || !s->data)
        return MATX_ERR_INVALID_ARG;
    if (s->n < A->ncols)
        return MATX_ERR_INVALID_ARG;
    for (matx_int64_t i = 0; i < A->nnz; ++i) {
        matx_double sr = s->data[A->columns[i]].real;
        matx_double si = s->data[A->columns[i]].imag;
        matx_double ar = A->values[i].real;
        matx_double ai = A->values[i].imag;
        A->values[i].real = ar * sr - ai * si;
        A->values[i].imag = ar * si + ai * sr;
    }
    return MATX_OK;
}

matx_sparse_backend_t matx_sparse_make_reference_aocl(void)
{
    matx_sparse_backend_t b = {.kind = MATX_SPARSE_BACKEND_AOCL_CPARSE,
                               .vt = {
                               .spmm_z_i8 = ref_spmm_z_i8_aocl,
                               .spmv_z_i8 = ref_spmv_z_i8_aocl,
                               .spmm_d_i8 = ref_spmm_d_i8_aocl,
                               .spmv_d_i8 = ref_spmv_d_i8_aocl,
                               .dsp2md_d_i8 = ref_dsp2md_d_i8_aocl,
                               .zsp2md_z_i8 = ref_zsp2md_z_i8_aocl,
                               .transpose_d_i8 = ref_transpose_d_i8_aocl,
                               .transpose_z_i8 = ref_transpose_z_i8_aocl,
                               .conj_trans_z_i8 = ref_conj_trans_z_i8_aocl,
                               .finalize = ref_finalize_aocl,
                               .norm1_mat_d_i8 = ref_norm1_mat_aocl,
                               .norminf_mat_d_i8 = ref_norminf_mat_aocl,
                               .normfro_mat_d_i8 = ref_normfro_mat_aocl,
                               .norm1_mat_z_i8 = ref_norm1_mat_z_i8_aocl,
                               .norminf_mat_z_i8 = ref_norminf_mat_z_i8_aocl,
                               .normfro_mat_z_i8 = ref_normfro_mat_z_i8_aocl,
                               .spadd_d_i8 = ref_spadd_d_i8_aocl,
                               .spadd_z_i8 = ref_spadd_z_i8_aocl,
                               .spnnz_rows_d_i8 = ref_spnnz_rows_d_i8_aocl,
                               .spnnz_cols_d_i8 = ref_spnnz_cols_d_i8_aocl,
                               .spnnz_rows_z_i8 = ref_spnnz_rows_z_i8_aocl,
                               .spnnz_cols_z_i8 = ref_spnnz_cols_z_i8_aocl,
                               .sprowsums_d_i8 = ref_sprowsums_d_i8_aocl,
                               .spcolsums_d_i8 = ref_spcolsums_d_i8_aocl,
                               .sprowsums_z_i8 = ref_sprowsums_z_i8_aocl,
                               .spcolsums_z_i8 = ref_spcolsums_z_i8_aocl,
                               .spdiag_d_i8 = ref_spdiag_d_i8_aocl,
                               .spdiag_z_i8 = ref_spdiag_z_i8_aocl,
                               .scale_rows_d_i8 = ref_scale_rows_d_i8_aocl,
                               .scale_cols_d_i8 = ref_scale_cols_d_i8_aocl,
                               .scale_rows_z_i8 = ref_scale_rows_z_i8_aocl,
                               .scale_cols_z_i8 = ref_scale_cols_z_i8_aocl,
                               }};
    MATX_TRACE("AOCL INIT");
    return b;
}