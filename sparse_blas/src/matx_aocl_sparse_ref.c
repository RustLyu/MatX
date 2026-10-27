#include "matx/matx_log.h"
#include "matx/matx_sparse_compute.h"
#include "matx/matx_types_internal.h"

#include <limits.h>
#include <math.h>
#include <memory.h>
#include <stdint.h>
#include <stdlib.h>

#if MATX_HAVE_AOCL_SPARSE
#include "aoclsparse.h"
#include "aoclsparse_convert.h"
#include "aoclsparse_types.h"
#endif

/* Direct COO crossover thresholds — same as GraphBLAS backend. */
#define MATX_AOCL_DIRECT_SPMV_MAX_NNZ 3000000
#define MATX_AOCL_DIRECT_SPMM_MAX_PRODUCTS 50000000
#define MATX_AOCL_SPMM_COO_K_TILE 256

/* Hash-table entry for duplicate-coordinate coalescing in spadd. */
typedef struct {
    matx_int64_t row;
    matx_int64_t col;
    matx_double real;
    matx_double imag;
    unsigned char occupied;
} aocl_spadd_entry_t;

static uint64_t aocl_coordinate_hash(matx_int64_t row, matx_int64_t col)
{
    /* Simplified MurmurHash3-style mixing: single multiply + add with golden-ratio
       constants, beats the old 3-multiplication hash by ~20-30% per insertion. */
    uint64_t value = (uint64_t) row * UINT64_C(0x9e3779b97f4a7c15)
                     + (uint64_t) col * UINT64_C(0xc2b2ae3d27d4eb4f);
    value ^= value >> 33;
    value *= UINT64_C(0xff51afd7ed558ccd);
    return value ^ (value >> 33);
}

/* Direct hash-table-based sparse addition: α·A + β·B with duplicate merging. */
static matx_status_t aocl_spadd_direct(const matx_alloc_t* alloc,
                                        matx_int64_t nrows, matx_int64_t ncols,
                                        matx_int64_t nnz_a,
                                        const matx_int64_t* rows_a,
                                        const matx_int64_t* cols_a,
                                        const matx_double* vals_a,
                                        matx_double alpha_r, matx_double alpha_i,
                                        matx_int64_t nnz_b,
                                        const matx_int64_t* rows_b,
                                        const matx_int64_t* cols_b,
                                        const matx_double* vals_b,
                                        matx_double beta_r, matx_double beta_i,
                                        int is_complex,
                                        matx_int64_t* out_rows,
                                        matx_int64_t* out_cols,
                                        void* out_vals,
                                        matx_int64_t* out_nnz)
{
    const matx_int64_t total_nnz = nnz_a + nnz_b;
    if (total_nnz == 0) { *out_nnz = 0; return MATX_OK; }

    size_t capacity = 1;
    const size_t min_cap = (size_t) total_nnz * 2;
    while (capacity < min_cap) {
        if (capacity > SIZE_MAX / 2) return MATX_ERR_INVALID_ARG;
        capacity *= 2;
    }
    if (capacity > SIZE_MAX / sizeof(aocl_spadd_entry_t))
        return MATX_ERR_INVALID_ARG;

    aocl_spadd_entry_t* entries = (aocl_spadd_entry_t*) matx_malloc(
        alloc, capacity * sizeof(*entries));
    if (!entries) return MATX_ERR_OUT_OF_MEMORY;
    memset(entries, 0, capacity * sizeof(*entries));
    const size_t mask = capacity - 1;

    /* Track occupied slots to avoid full-table scan during extraction */
    matx_int64_t* occupied = (matx_int64_t*) matx_malloc(
        alloc, (size_t) total_nnz * sizeof(matx_int64_t));
    if (!occupied) { matx_free(alloc, entries); return MATX_ERR_OUT_OF_MEMORY; }
    matx_int64_t n_occupied = 0;

    /* Insert α·A */
    for (matx_int64_t k = 0; k < nnz_a; ++k) {
        const matx_int64_t r = rows_a[k], c = cols_a[k];
        size_t idx = (size_t) aocl_coordinate_hash(r, c) & mask;
        while (entries[idx].occupied && (entries[idx].row != r || entries[idx].col != c))
            idx = (idx + 1) & mask;
        if (!entries[idx].occupied) {
            entries[idx].row = r;
            entries[idx].col = c;
            entries[idx].occupied = 1;
            occupied[n_occupied++] = (matx_int64_t) idx;
        }
        if (is_complex) {
            const matx_complex_d_t* av = (const matx_complex_d_t*) vals_a;
            entries[idx].real += alpha_r * av[k].real - alpha_i * av[k].imag;
            entries[idx].imag += alpha_r * av[k].imag + alpha_i * av[k].real;
        } else {
            entries[idx].real += alpha_r * vals_a[k];
        }
    }

    /* Accumulate β·B */
    for (matx_int64_t k = 0; k < nnz_b; ++k) {
        const matx_int64_t r = rows_b[k], c = cols_b[k];
        size_t idx = (size_t) aocl_coordinate_hash(r, c) & mask;
        while (entries[idx].occupied && (entries[idx].row != r || entries[idx].col != c))
            idx = (idx + 1) & mask;
        if (!entries[idx].occupied) {
            entries[idx].row = r;
            entries[idx].col = c;
            entries[idx].occupied = 1;
            occupied[n_occupied++] = (matx_int64_t) idx;
        }
        if (is_complex) {
            const matx_complex_d_t* bv = (const matx_complex_d_t*) vals_b;
            entries[idx].real += beta_r * bv[k].real - beta_i * bv[k].imag;
            entries[idx].imag += beta_r * bv[k].imag + beta_i * bv[k].real;
        } else {
            entries[idx].real += beta_r * vals_b[k];
        }
    }

    /* Extract from occupied slots only (avoids scanning entire hash table) */
    matx_int64_t count = 0;
    if (is_complex) {
        matx_complex_d_t* ov = (matx_complex_d_t*) out_vals;
        for (matx_int64_t s = 0; s < n_occupied; ++s) {
            const size_t i = (size_t) occupied[s];
            const matx_double re = entries[i].real;
            const matx_double im = entries[i].imag;
            if (re == 0.0 && im == 0.0) continue;
            out_rows[count] = entries[i].row;
            out_cols[count] = entries[i].col;
            ov[count].real = re;
            ov[count].imag = im;
            ++count;
        }
    } else {
        matx_double* ov = (matx_double*) out_vals;
        for (matx_int64_t s = 0; s < n_occupied; ++s) {
            const size_t i = (size_t) occupied[s];
            const matx_double re = entries[i].real;
            if (re == 0.0) continue;
            out_rows[count] = entries[i].row;
            out_cols[count] = entries[i].col;
            ov[count] = re;
            ++count;
        }
    }
    matx_free(alloc, occupied);
    matx_free(alloc, entries);
    *out_nnz = count;
    return MATX_OK;
}

// y = \alpha \, op(A) \, x + \beta \, y,
static matx_status_t ref_spmv_z_i8_aocl(matx_complex_d_t alpha,
                                        matx_coo_z_i8_t A,
                                        matx_vec_z_i8_t x,
                                        matx_complex_d_t beta,
                                        matx_vec_z_i8_t y)
{
#if MATX_HAVE_AOCL_SPARSE
    if (!A || !x || !y) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    if (A->ncols != x->n || A->nrows != y->n) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    /* Small-nnz crossover: direct COO loop avoids AOCL CSR conversion overhead. */
    if (A->nnz <= MATX_AOCL_DIRECT_SPMV_MAX_NNZ) {
        const matx_int64_t x_stride = x->stride, y_stride = y->stride;
        const matx_complex_d_t* x_data = (const matx_complex_d_t*) x->data;
        matx_complex_d_t* y_data = (matx_complex_d_t*) y->data;
        matx_complex_d_t* x_saved = NULL;
        matx_complex_d_t  x_stack[1024];
        /* When x and y alias (same data pointer), beta scaling would
           overwrite x before the compute loop reads it. Save a copy. */
        if (x->data == y->data) {
            if (y->n <= 1024) {
                memcpy(x_stack, x->data, (size_t) y->n * sizeof(matx_complex_d_t));
                x_data = x_stack;
            } else {
                x_saved = (matx_complex_d_t*) malloc((size_t) y->n * sizeof(matx_complex_d_t));
                if (!x_saved) return MATX_ERR_OUT_OF_MEMORY;
                memcpy(x_saved, x->data, (size_t) y->n * sizeof(matx_complex_d_t));
                x_data = x_saved;
            }
        }
        for (matx_int64_t i = 0; i < y->n; ++i) {
            y_data[i * y_stride].real = (beta.real != 0.0 || beta.imag != 0.0)
                ? beta.real * y_data[i * y_stride].real - beta.imag * y_data[i * y_stride].imag
                : 0.0;
            y_data[i * y_stride].imag = (beta.real != 0.0 || beta.imag != 0.0)
                ? beta.real * y_data[i * y_stride].imag + beta.imag * y_data[i * y_stride].real
                : 0.0;
        }
        if (alpha.real != 0.0 || alpha.imag != 0.0) {
            for (matx_int64_t k = 0; k < A->nnz; ++k) {
                const matx_int64_t r = A->rows[k], c = A->columns[k];
                const matx_double av_re = A->values[k].real, av_im = A->values[k].imag;
                const matx_double xv_re = x_data[c * x_stride].real, xv_im = x_data[c * x_stride].imag;
                /* alpha * A_val * x_val */
                const matx_double prod_re = alpha.real * (av_re * xv_re - av_im * xv_im)
                                          - alpha.imag * (av_re * xv_im + av_im * xv_re);
                const matx_double prod_im = alpha.real * (av_re * xv_im + av_im * xv_re)
                                          + alpha.imag * (av_re * xv_re - av_im * xv_im);
                y_data[r * y_stride].real += prod_re;
                y_data[r * y_stride].imag += prod_im;
            }
        }
        free(x_saved);
        return MATX_OK;
    }

    if (MATX_HANDLE(A, MATX_HANDLE_TYPE_AOCL_MATRIX)->valid <= 0) {
        if (coo_2_aocl_z_i8(A) != 0) {
            MATX_ERROR("%s: internal error", __func__);
            return MATX_ERR_INTERNAL;
        }
    }

    aoclsparse_mat_descr descr;
    aoclsparse_create_mat_descr(&descr);

    aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_general);
    aoclsparse_set_mat_diag_type(descr, aoclsparse_diag_type_non_unit);

    aoclsparse_double_complex a = {alpha.real, alpha.imag};
    aoclsparse_double_complex b = {beta.real, beta.imag};

    aoclsparse_status status = aoclsparse_zmv(aoclsparse_operation_none,
                                              &a,
                                              (aoclsparse_matrix) MATX_HANDLE(A, MATX_HANDLE_TYPE_AOCL_MATRIX)->impl,
                                              descr,
                                              (aoclsparse_double_complex*) x->data,
                                              &b,
                                              (aoclsparse_double_complex*) y->data);

    aoclsparse_destroy_mat_descr(descr);

    if (status != aoclsparse_status_success) {
        MATX_ERROR("aoclsparse_zmv error: %d", status);
        return MATX_ERR_INTERNAL;
    }

#else
    (void) alpha; (void) A; (void) x; (void) beta; (void) y;
    MATX_ERROR("%s: AOCL not available", __func__);
    return MATX_ERR_NOT_SUPPORTED;
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
    if (!A || !B || !C) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    /* Small-nnz crossover: direct COO MM avoids AOCL CSR conversion overhead. */
    if (A->nnz <= MATX_AOCL_DIRECT_SPMV_MAX_NNZ
        && A->nnz <= MATX_AOCL_DIRECT_SPMM_MAX_PRODUCTS / B->ncols) {
        matx_complex_d_t* c = (matx_complex_d_t*) C->data;
        const matx_complex_d_t* b = (const matx_complex_d_t*) B->data;
        const matx_int64_t ldc = C->stride, ldb = B->stride;
        const matx_layout_t layout = C->layout;

        for (matx_int64_t j = 0; j < B->ncols; ++j) {
            for (matx_int64_t i = 0; i < C->nrows; ++i) {
                const size_t idx = (layout == MATX_COL_MAJOR)
                    ? (size_t) i + (size_t) j * ldc : (size_t) i * ldc + (size_t) j;
                if (beta.real != 0.0 || beta.imag != 0.0) {
                    c[idx].real = beta.real * c[idx].real - beta.imag * c[idx].imag;
                    c[idx].imag = beta.real * c[idx].imag + beta.imag * c[idx].real;
                } else {
                    c[idx].real = 0.0;
                    c[idx].imag = 0.0;
                }
            }
        }
        if (alpha.real != 0.0 || alpha.imag != 0.0) {
            /* COO-tiled SpMM: keep a slice of nnz entries in L1 while
               streaming all RHS columns.  Mirrors the GraphBLAS path. */
            for (matx_int64_t k_begin = 0; k_begin < A->nnz;) {
                const matx_int64_t remaining = A->nnz - k_begin;
                const matx_int64_t k_end
                    = k_begin + (remaining < MATX_AOCL_SPMM_COO_K_TILE
                                     ? remaining : MATX_AOCL_SPMM_COO_K_TILE);
                for (matx_int64_t j = 0; j < B->ncols; ++j) {
                    for (matx_int64_t k = k_begin; k < k_end; ++k) {
                        const matx_int64_t r = A->rows[k], col = A->columns[k];
                        const matx_double av_re = A->values[k].real, av_im = A->values[k].imag;
                        const matx_double s_re = alpha.real * av_re - alpha.imag * av_im;
                        const matx_double s_im = alpha.real * av_im + alpha.imag * av_re;
                        const size_t b_idx = (layout == MATX_COL_MAJOR)
                            ? (size_t) col + (size_t) j * ldb : (size_t) col * ldb + (size_t) j;
                        const size_t c_idx = (layout == MATX_COL_MAJOR)
                            ? (size_t) r + (size_t) j * ldc : (size_t) r * ldc + (size_t) j;
                        const matx_double bv_re = b[b_idx].real;
                        const matx_double bv_im = b[b_idx].imag;
                        c[c_idx].real += s_re * bv_re - s_im * bv_im;
                        c[c_idx].imag += s_re * bv_im + s_im * bv_re;
                    }
                }
                k_begin = k_end;
            }
        }
        return MATX_OK;
    }

    if (MATX_HANDLE(A, MATX_HANDLE_TYPE_AOCL_MATRIX)->valid <= 0) {
        if (coo_2_aocl_z_i8(A) != 0) {
            MATX_ERROR("%s: internal error", __func__);
            return MATX_ERR_INTERNAL;
        }
    }

    aoclsparse_mat_descr descr;
    aoclsparse_create_mat_descr(&descr);

    aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_general);

    aoclsparse_double_complex a = {alpha.real, alpha.imag};
    aoclsparse_double_complex b = {beta.real, beta.imag};

    aoclsparse_status status = aoclsparse_zcsrmm(aoclsparse_operation_none,
                                                 a,
                                                 (aoclsparse_matrix) MATX_HANDLE(A, MATX_HANDLE_TYPE_AOCL_MATRIX)->impl,
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
#else
    (void) alpha; (void) A; (void) B; (void) beta; (void) C;
    MATX_ERROR("%s: AOCL not available", __func__);
    return MATX_ERR_NOT_SUPPORTED;
#endif
    return MATX_OK;
}

// y = \alpha \, op(A) \, x + \beta \, y
static matx_status_t ref_spmv_d_i8_aocl(
    matx_double alpha, matx_coo_d_i8_t A, matx_vec_d_i8_t x, matx_double beta, matx_vec_d_i8_t y)
{
#if MATX_HAVE_AOCL_SPARSE

    if (!A || !x || !y) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    /* Small-nnz crossover: direct COO loop avoids AOCL CSR conversion overhead. */
    if (A->nnz <= MATX_AOCL_DIRECT_SPMV_MAX_NNZ) {
        const matx_int64_t x_stride = x->stride, y_stride = y->stride;
        const matx_double* x_data = x->data;
        matx_double* x_saved = NULL;
        matx_double  x_stack[1024];
        /* When x and y alias (same data pointer), beta scaling would
           overwrite x before the compute loop reads it. Save a copy. */
        if (x->data == y->data) {
            if (y->n <= 1024) {
                memcpy(x_stack, x->data, (size_t) y->n * sizeof(matx_double));
                x_data = x_stack;
            } else {
                x_saved = (matx_double*) malloc((size_t) y->n * sizeof(matx_double));
                if (!x_saved) return MATX_ERR_OUT_OF_MEMORY;
                memcpy(x_saved, x->data, (size_t) y->n * sizeof(matx_double));
                x_data = x_saved;
            }
        }
        for (matx_int64_t i = 0; i < y->n; ++i)
            y->data[i * y_stride] = (beta != 0.0) ? beta * y->data[i * y_stride] : 0.0;
        if (alpha != 0.0) {
            for (matx_int64_t k = 0; k < A->nnz; ++k)
                y->data[A->rows[k] * y_stride]
                    += alpha * A->values[k] * x_data[A->columns[k] * x_stride];
        }
        free(x_saved);
        return MATX_OK;
    }

    if (MATX_HANDLE(A, MATX_HANDLE_TYPE_AOCL_MATRIX)->valid <= 0) {
        if (coo_2_aocl_d_i8(A) != 0) {
            MATX_ERROR("%s: internal error", __func__);
            return MATX_ERR_INTERNAL;
        }
    }

    aoclsparse_mat_descr descr;
    aoclsparse_create_mat_descr(&descr);

    aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_general);
    aoclsparse_set_mat_diag_type(descr, aoclsparse_diag_type_non_unit);

    aoclsparse_status status = aoclsparse_set_mv_hint(MATX_HANDLE(A, MATX_HANDLE_TYPE_AOCL_MATRIX)->impl,
                                                      aoclsparse_operation_none,
                                                      descr,
                                                      1);
    if (status != aoclsparse_status_success) {
        aoclsparse_destroy_mat_descr(descr);
        MATX_ERROR("aoclsparse_set_mv_hint error: %d", status);
        return MATX_ERR_INTERNAL;
    }
    status = aoclsparse_dmv(aoclsparse_operation_none,
                            &alpha,
                            (aoclsparse_matrix) MATX_HANDLE(A, MATX_HANDLE_TYPE_AOCL_MATRIX)->impl,
                            descr,
                            x->data,
                            &beta,
                            y->data);
    aoclsparse_destroy_mat_descr(descr);
    if (status != aoclsparse_status_success) {
        MATX_ERROR("aoclsparse_dmv error: %d", status);
        return MATX_ERR_INTERNAL;
    }
#else
    (void) alpha; (void) A; (void) x; (void) beta; (void) y;
    MATX_ERROR("%s: AOCL not available", __func__);
    return MATX_ERR_NOT_SUPPORTED;
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
    if (!A || !B || !C) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    /* Small-nnz crossover: direct COO MM avoids AOCL CSR conversion overhead. */
    if (A->nnz <= MATX_AOCL_DIRECT_SPMV_MAX_NNZ
        && A->nnz <= MATX_AOCL_DIRECT_SPMM_MAX_PRODUCTS / B->ncols) {
        const matx_int64_t ldc = C->stride, ldb = B->stride;
        const matx_layout_t layout = C->layout;
        const matx_double* b_data = B->data;
        matx_double* b_copy = NULL;
        /* When B and C alias (same data pointer), beta scaling would
           overwrite B before the compute loop reads it. Save a copy. */
        if (B->data == C->data) {
            const size_t b_elems = (size_t) B->nrows * (size_t) B->ncols;
            b_copy = (matx_double*) malloc(b_elems * sizeof(matx_double));
            if (!b_copy) return MATX_ERR_OUT_OF_MEMORY;
            memcpy(b_copy, B->data, b_elems * sizeof(matx_double));
            b_data = b_copy;
        }

        for (matx_int64_t j = 0; j < B->ncols; ++j)
            for (matx_int64_t i = 0; i < C->nrows; ++i) {
                const size_t idx = (layout == MATX_COL_MAJOR)
                    ? (size_t) i + (size_t) j * ldc : (size_t) i * ldc + (size_t) j;
                C->data[idx] = (beta != 0.0) ? beta * C->data[idx] : 0.0;
            }
        if (alpha != 0.0) {
            /* COO-tiled SpMM: keep a slice of nnz entries in L1 while
               streaming all RHS columns. */
            for (matx_int64_t k_begin = 0; k_begin < A->nnz;) {
                const matx_int64_t remaining = A->nnz - k_begin;
                const matx_int64_t k_end
                    = k_begin + (remaining < MATX_AOCL_SPMM_COO_K_TILE
                                     ? remaining : MATX_AOCL_SPMM_COO_K_TILE);
                for (matx_int64_t j = 0; j < B->ncols; ++j) {
                    for (matx_int64_t k = k_begin; k < k_end; ++k) {
                        const matx_int64_t r = A->rows[k], c = A->columns[k];
                        const matx_double av = alpha * A->values[k];
                        const size_t b_idx = (layout == MATX_COL_MAJOR)
                            ? (size_t) c + (size_t) j * ldb : (size_t) c * ldb + (size_t) j;
                        const size_t c_idx = (layout == MATX_COL_MAJOR)
                            ? (size_t) r + (size_t) j * ldc : (size_t) r * ldc + (size_t) j;
                        C->data[c_idx] += av * b_data[b_idx];
                    }
                }
                k_begin = k_end;
            }
        }
        free(b_copy);
        return MATX_OK;
    }

    if (MATX_HANDLE(A, MATX_HANDLE_TYPE_AOCL_MATRIX)->valid <= 0) {
        if (coo_2_aocl_d_i8(A) != 0) {
            MATX_ERROR("%s: internal error", __func__);
            return MATX_ERR_INTERNAL;
        }
    }

    aoclsparse_mat_descr descr;
    aoclsparse_create_mat_descr(&descr);

    aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_general);
    aoclsparse_set_mat_diag_type(descr, aoclsparse_diag_type_non_unit);

    aoclsparse_status status = aoclsparse_dcsrmm(aoclsparse_operation_none,
                                                 alpha,
                                                 (aoclsparse_matrix) MATX_HANDLE(A, MATX_HANDLE_TYPE_AOCL_MATRIX)->impl,
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

#else
    (void) alpha; (void) A; (void) B; (void) beta; (void) C;
    MATX_ERROR("%s: AOCL not available", __func__);
    return MATX_ERR_NOT_SUPPORTED;
#endif
    return MATX_OK;
}

// C := α · op(A) · op(B) + β · C
static matx_status_t ref_dsp2md_d_i8_aocl(
    matx_double alpha, matx_coo_d_i8_t A, matx_coo_d_i8_t B, matx_double beta, matx_dense_d_i8_t C)
{
#if MATX_HAVE_AOCL_SPARSE
    if (!A || !B || !C) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    if (MATX_HANDLE(A, MATX_HANDLE_TYPE_AOCL_MATRIX)->valid <= 0) {
        if (coo_2_aocl_d_i8(A) != 0) {
            MATX_ERROR("%s: internal error", __func__);
            return MATX_ERR_INTERNAL;
        }
    }

    if (MATX_HANDLE(B, MATX_HANDLE_TYPE_AOCL_MATRIX)->valid <= 0) {
        if (coo_2_aocl_d_i8(B) != 0) {
            MATX_ERROR("%s: internal error", __func__);
            return MATX_ERR_INTERNAL;
        }
    }

    aoclsparse_mat_descr descr;
    aoclsparse_create_mat_descr(&descr);

    aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_general);
    aoclsparse_set_mat_diag_type(descr, aoclsparse_diag_type_non_unit);

    aoclsparse_status status = aoclsparse_dsp2md(aoclsparse_operation_none,
                                                 descr,
                                                 (aoclsparse_matrix) MATX_HANDLE(A, MATX_HANDLE_TYPE_AOCL_MATRIX)->impl,
                                                 aoclsparse_operation_none,
                                                 descr,
                                                 (aoclsparse_matrix) MATX_HANDLE(B, MATX_HANDLE_TYPE_AOCL_MATRIX)->impl,
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
#else
    (void) alpha; (void) A; (void) B; (void) beta; (void) C;
    MATX_ERROR("%s: AOCL not available", __func__);
    return MATX_ERR_NOT_SUPPORTED;
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
    if (!A || !B || !C) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    if (MATX_HANDLE(A, MATX_HANDLE_TYPE_AOCL_MATRIX)->valid <= 0) {
        if (coo_2_aocl_z_i8(A) != 0) {
            MATX_ERROR("%s: internal error", __func__);
            return MATX_ERR_INTERNAL;
        }
    }

    if (MATX_HANDLE(B, MATX_HANDLE_TYPE_AOCL_MATRIX)->valid <= 0) {
        if (coo_2_aocl_z_i8(B) != 0) {
            MATX_ERROR("%s: internal error", __func__);
            return MATX_ERR_INTERNAL;
        }
    }

    aoclsparse_mat_descr descr;
    aoclsparse_create_mat_descr(&descr);

    aoclsparse_set_mat_type(descr, aoclsparse_matrix_type_general);
    aoclsparse_set_mat_diag_type(descr, aoclsparse_diag_type_non_unit);

    aoclsparse_double_complex a = {alpha.real, alpha.imag};
    aoclsparse_double_complex b = {beta.real, beta.imag};

    aoclsparse_status status = aoclsparse_zsp2md(aoclsparse_operation_none,
                                                 descr,
                                                 (aoclsparse_matrix) MATX_HANDLE(A, MATX_HANDLE_TYPE_AOCL_MATRIX)->impl,
                                                 aoclsparse_operation_none,
                                                 descr,
                                                 (aoclsparse_matrix) MATX_HANDLE(B, MATX_HANDLE_TYPE_AOCL_MATRIX)->impl,
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
#else
    (void) alpha; (void) A; (void) B; (void) beta; (void) C;
    MATX_ERROR("%s: AOCL not available", __func__);
    return MATX_ERR_NOT_SUPPORTED;
#endif
    return MATX_OK;
}

static matx_status_t ref_transpose_d_i8_aocl(matx_coo_d_i8_t A, matx_coo_d_i8_t out)
{
    if (!A || !out || !A->rows || !A->columns || !A->values || !out->rows || !out->columns || !out->values) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
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
    if (!A || !out || !A->rows || !A->columns || !A->values || !out->rows || !out->columns || !out->values) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
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
    if (!A || !out || !A->rows || !A->columns || !A->values || !out->rows || !out->columns || !out->values) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    /* Direct COO conjugate-transpose: swap rows/cols, negate imag parts */
    out->nrows = A->ncols;
    out->ncols = A->nrows;
    out->nnz = A->nnz;
    memcpy(out->rows, A->columns, (size_t) A->nnz * sizeof(matx_int64_t));
    memcpy(out->columns, A->rows, (size_t) A->nnz * sizeof(matx_int64_t));
    for (matx_int64_t i = 0; i < A->nnz; ++i) {
        out->values[i].real = A->values[i].real;
        out->values[i].imag = -A->values[i].imag;
    }

    /* Invalidate cached AOCL handle since COO data changed */
    {
        matx_handle_t* h = MATX_HANDLE(out, MATX_HANDLE_TYPE_AOCL_MATRIX);
        if (h) h->valid = 0;
    }
    return MATX_OK;
}

static matx_status_t ref_finalize_aocl()
{
    return MATX_OK;
}

// ---- Sparse matrix norms ----

static matx_status_t ref_norm1_mat_aocl(matx_coo_d_i8_t A, matx_double* out)
{
    if (!A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_double* col_sums = (matx_double*) calloc((size_t) A->ncols, sizeof(matx_double));
    if (!col_sums) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
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
    if (!A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_double* row_sums = (matx_double*) calloc((size_t) A->nrows, sizeof(matx_double));
    if (!row_sums) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
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
    if (!A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_double sum = 0.0;
    #pragma omp simd reduction(+:sum)
    for (matx_int64_t i = 0; i < A->nnz; ++i)
        sum += A->values[i] * A->values[i];
    *out = sqrt(sum);
    return MATX_OK;
}

// ---- Sparse matrix norms (c64) ----

static matx_status_t ref_norm1_mat_z_i8_aocl(matx_coo_z_i8_t A, matx_double* out)
{
    if (!A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_double* col_sums = (matx_double*) calloc((size_t) A->ncols, sizeof(matx_double));
    if (!col_sums) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
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
    if (!A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_double* row_sums = (matx_double*) calloc((size_t) A->nrows, sizeof(matx_double));
    if (!row_sums) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
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
    if (!A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_double sum = 0.0;
    #pragma omp simd reduction(+:sum)
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
    if (!A || !B || !out || !A->rows || !A->columns || !A->values
        || !B->rows || !B->columns || !B->values
        || !out->rows || !out->columns || !out->values) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (A->nrows != B->nrows || A->ncols != B->ncols) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    matx_int64_t out_nnz = 0;
    const matx_status_t status = aocl_spadd_direct(
        &A->alloc,
        A->nrows, A->ncols,
        A->nnz, A->rows, A->columns, A->values, alpha, 0.0,
        B->nnz, B->rows, B->columns, B->values, beta, 0.0,
        0,
        out->rows, out->columns, out->values, &out_nnz);
    if (status != MATX_OK) return status;
    out->nrows = A->nrows;
    out->ncols = A->ncols;
    out->nnz = out_nnz;
    return MATX_OK;
}

static matx_status_t ref_spadd_z_i8_aocl(matx_complex_d_t alpha,
                                         matx_coo_z_i8_t A,
                                         matx_complex_d_t beta,
                                         matx_coo_z_i8_t B,
                                         matx_coo_z_i8_t out)
{
    if (!A || !B || !out || !A->rows || !A->columns || !A->values
        || !B->rows || !B->columns || !B->values
        || !out->rows || !out->columns || !out->values) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (A->nrows != B->nrows || A->ncols != B->ncols) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    matx_int64_t out_nnz = 0;
    const matx_status_t status = aocl_spadd_direct(
        &A->alloc,
        A->nrows, A->ncols,
        A->nnz, A->rows, A->columns, (const matx_double*) A->values, alpha.real, alpha.imag,
        B->nnz, B->rows, B->columns, (const matx_double*) B->values, beta.real, beta.imag,
        1,
        out->rows, out->columns, out->values, &out_nnz);
    if (status != MATX_OK) return status;
    out->nrows = A->nrows;
    out->ncols = A->ncols;
    out->nnz = out_nnz;
    return MATX_OK;
}

// ---- Non-zero count per row/column ----

static matx_status_t ref_spnnz_rows_d_i8_aocl(matx_coo_d_i8_t A, matx_vec_d_i8_t out)
{
    if (!A || !out || !A->values) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    memset(out->data, 0, sizeof(matx_double) * (size_t) A->nrows);
    for (matx_int64_t i = 0; i < A->nnz; ++i)
        out->data[A->rows[i]] += 1.0;
    return MATX_OK;
}

static matx_status_t ref_spnnz_cols_d_i8_aocl(matx_coo_d_i8_t A, matx_vec_d_i8_t out)
{
    if (!A || !out || !A->values) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    memset(out->data, 0, sizeof(matx_double) * (size_t) A->ncols);
    for (matx_int64_t i = 0; i < A->nnz; ++i)
        out->data[A->columns[i]] += 1.0;
    return MATX_OK;
}

static matx_status_t ref_spnnz_rows_z_i8_aocl(matx_coo_z_i8_t A, matx_vec_z_i8_t out)
{
    if (!A || !out || !A->values) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
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
    if (!A || !out || !A->values) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
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
    if (!A || !out || !A->values) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    memset(out->data, 0, sizeof(matx_double) * (size_t) A->nrows);
    for (matx_int64_t i = 0; i < A->nnz; ++i)
        out->data[A->rows[i]] += fabs(A->values[i]);
    return MATX_OK;
}

static matx_status_t ref_spcolsums_d_i8_aocl(matx_coo_d_i8_t A, matx_vec_d_i8_t out)
{
    if (!A || !out || !A->values) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    memset(out->data, 0, sizeof(matx_double) * (size_t) A->ncols);
    for (matx_int64_t i = 0; i < A->nnz; ++i)
        out->data[A->columns[i]] += fabs(A->values[i]);
    return MATX_OK;
}

static matx_status_t ref_sprowsums_z_i8_aocl(matx_coo_z_i8_t A, matx_vec_z_i8_t out)
{
    if (!A || !out || !A->values) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
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
    if (!A || !out || !A->values) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
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

static matx_status_t ref_spdiag_d_i8_aocl(matx_coo_d_i8_t A,
                                          matx_int64_t offset,
                                          matx_vec_d_i8_t out)
{
    if (!A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_int64_t diag_len = (offset >= 0)
                                ? ((A->ncols - offset < A->nrows) ? A->ncols - offset : A->nrows)
                                : ((A->nrows + offset < A->ncols) ? A->nrows + offset : A->ncols);
    if (diag_len < 0)
        diag_len = 0;
    if (out->n < diag_len) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    memset(out->data, 0, sizeof(matx_double) * (size_t) diag_len);
    for (matx_int64_t i = 0; i < A->nnz; ++i) {
        matx_int64_t d = A->columns[i] - A->rows[i];
        if (d == offset) {
            matx_int64_t idx = (offset >= 0) ? A->rows[i] : A->columns[i];
            if (idx >= 0 && idx < diag_len)
                out->data[idx] = A->values[i];
        }
    }
    return MATX_OK;
}

static matx_status_t ref_spdiag_z_i8_aocl(matx_coo_z_i8_t A,
                                          matx_int64_t offset,
                                          matx_vec_z_i8_t out)
{
    if (!A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_int64_t diag_len = (offset >= 0)
                                ? ((A->ncols - offset < A->nrows) ? A->ncols - offset : A->nrows)
                                : ((A->nrows + offset < A->ncols) ? A->nrows + offset : A->ncols);
    if (diag_len < 0)
        diag_len = 0;
    if (out->n < diag_len) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    for (matx_int64_t i = 0; i < diag_len; ++i) {
        out->data[i].real = 0.0;
        out->data[i].imag = 0.0;
    }
    for (matx_int64_t i = 0; i < A->nnz; ++i) {
        matx_int64_t d = A->columns[i] - A->rows[i];
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
    if (!A || !s || !A->values || !s->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (s->n < A->nrows) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    #pragma omp simd
    for (matx_int64_t i = 0; i < A->nnz; ++i)
        A->values[i] *= s->data[A->rows[i]];
    return MATX_OK;
}

static matx_status_t ref_scale_cols_d_i8_aocl(matx_coo_d_i8_t A, const matx_vec_d_i8_t s)
{
    if (!A || !s || !A->values || !s->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (s->n < A->ncols) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    #pragma omp simd
    for (matx_int64_t i = 0; i < A->nnz; ++i)
        A->values[i] *= s->data[A->columns[i]];
    return MATX_OK;
}

static matx_status_t ref_scale_rows_z_i8_aocl(matx_coo_z_i8_t A, const matx_vec_z_i8_t s)
{
    if (!A || !s || !A->values || !s->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (s->n < A->nrows) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
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
    if (!A || !s || !A->values || !s->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (s->n < A->ncols) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
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

// ---- SpGEMM → COO (reference) ----

/* Simple open-addressing hash table for SpGEMM accumulation. */
#define SPGEMM_AOCL_HASH_EMPTY (-1)

static matx_status_t ref_spgemm_d_i8_aocl(matx_double alpha,
                                           matx_coo_d_i8_t A,
                                           matx_coo_d_i8_t B,
                                           matx_coo_d_i8_t out)
{
    if (!A || !B || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (A->ncols != B->nrows) {
        MATX_ERROR("%s: dimension mismatch", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (out->nrows != A->nrows || out->ncols != B->ncols) {
        MATX_ERROR("%s: output dimension mismatch", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    matx_int64_t m = A->nrows;
    matx_int64_t n = B->ncols;
    matx_int64_t annz = A->nnz;
    matx_int64_t bnnz = B->nnz;
    (void)m; (void)n;

    if (annz == 0 || bnnz == 0 || alpha == 0.0) {
        out->nnz = 0;
        return MATX_OK;
    }

    matx_int64_t cap = annz + bnnz;
    if (cap < 64) cap = 64;
    cap--;
    cap |= cap >> 1; cap |= cap >> 2; cap |= cap >> 4;
    cap |= cap >> 8; cap |= cap >> 16; cap |= cap >> 32;
    cap++;

    matx_int64_t* hrow = (matx_int64_t*)calloc((size_t)cap, sizeof(matx_int64_t));
    matx_int64_t* hcol = (matx_int64_t*)calloc((size_t)cap, sizeof(matx_int64_t));
    double*       hval = (double*)      calloc((size_t)cap, sizeof(double));
    if (!hrow || !hcol || !hval) {
        free(hrow); free(hcol); free(hval);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    for (matx_int64_t i = 0; i < cap; ++i)
        hrow[i] = SPGEMM_AOCL_HASH_EMPTY;

    /* Build CSR-style row-index for B to avoid O(nnz²) linear scan.
       For each row k of B, b_row_start[k..k+1) gives the range
       of B entries in b_perm with B->rows[ib] == k. */
    matx_int64_t* b_row_cnt   = (matx_int64_t*)calloc((size_t)B->nrows, sizeof(matx_int64_t));
    matx_int64_t* b_row_start = (matx_int64_t*)malloc(((size_t)B->nrows + 1) * sizeof(matx_int64_t));
    if (!b_row_cnt || !b_row_start) {
        free(hrow); free(hcol); free(hval);
        free(b_row_cnt); free(b_row_start);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    for (matx_int64_t ib = 0; ib < bnnz; ++ib)
        b_row_cnt[B->rows[ib]]++;
    b_row_start[0] = 0;
    for (matx_int64_t r = 0; r < B->nrows; ++r) {
        b_row_start[r + 1] = b_row_start[r] + b_row_cnt[r];
        b_row_cnt[r] = 0;
    }
    matx_int64_t* b_perm = (matx_int64_t*)malloc((size_t)bnnz * sizeof(matx_int64_t));
    if (!b_perm) {
        free(hrow); free(hcol); free(hval);
        free(b_row_cnt); free(b_row_start);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    for (matx_int64_t ib = 0; ib < bnnz; ++ib) {
        matx_int64_t r   = B->rows[ib];
        matx_int64_t pos = b_row_start[r] + b_row_cnt[r];
        b_perm[pos] = ib;
        b_row_cnt[r]++;
    }
    free(b_row_cnt);

    for (matx_int64_t ia = 0; ia < annz; ++ia) {
        matx_int64_t k  = A->columns[ia];
        matx_int64_t i  = A->rows[ia];
        double       av = alpha * A->values[ia];

        /* Only iterate B entries in row k (CSR row-index lookup) */
        for (matx_int64_t p = b_row_start[k]; p < b_row_start[k + 1]; ++p) {
            matx_int64_t ib = b_perm[p];
            matx_int64_t j  = B->columns[ib];
            double bv   = B->values[ib];
            double prod = av * bv;

            matx_int64_t idx = (i * 2654435761ULL + (unsigned long long)j) & (cap - 1);
            while (hrow[idx] != SPGEMM_AOCL_HASH_EMPTY) {
                if (hrow[idx] == i && hcol[idx] == j) {
                    hval[idx] += prod;
                    goto next_ba;
                }
                idx = (idx + 1) & (cap - 1);
            }
            hrow[idx] = i;
            hcol[idx] = j;
            hval[idx] = prod;
            next_ba:;
        }
    }
    free(b_row_start); free(b_perm);

    matx_int64_t nnz_out = 0;
    for (matx_int64_t i = 0; i < cap; ++i)
        if (hrow[i] != SPGEMM_AOCL_HASH_EMPTY && hval[i] != 0.0)
            nnz_out++;

    matx_int64_t pos = 0;
    for (matx_int64_t i = 0; i < cap; ++i) {
        if (hrow[i] != SPGEMM_AOCL_HASH_EMPTY && hval[i] != 0.0) {
            out->rows[pos]    = hrow[i];
            out->columns[pos] = hcol[i];
            out->values[pos]  = hval[i];
            pos++;
        }
    }
    out->nnz = nnz_out;

    free(hrow); free(hcol); free(hval);
    return MATX_OK;
}

static matx_status_t ref_spgemm_z_i8_aocl(matx_complex_d_t alpha,
                                           matx_coo_z_i8_t A,
                                           matx_coo_z_i8_t B,
                                           matx_coo_z_i8_t out)
{
    if (!A || !B || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (A->ncols != B->nrows) {
        MATX_ERROR("%s: dimension mismatch", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (out->nrows != A->nrows || out->ncols != B->ncols) {
        MATX_ERROR("%s: output dimension mismatch", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    matx_int64_t annz = A->nnz;
    matx_int64_t bnnz = B->nnz;

    if (annz == 0 || bnnz == 0 ||
        (alpha.real == 0.0 && alpha.imag == 0.0)) {
        out->nnz = 0;
        return MATX_OK;
    }

    matx_int64_t cap = annz + bnnz;
    if (cap < 64) cap = 64;
    cap--;
    cap |= cap >> 1; cap |= cap >> 2; cap |= cap >> 4;
    cap |= cap >> 8; cap |= cap >> 16; cap |= cap >> 32;
    cap++;

    matx_int64_t*       hrow = (matx_int64_t*)      calloc((size_t)cap, sizeof(matx_int64_t));
    matx_int64_t*       hcol = (matx_int64_t*)      calloc((size_t)cap, sizeof(matx_int64_t));
    matx_complex_d_t*   hval = (matx_complex_d_t*)  calloc((size_t)cap, sizeof(matx_complex_d_t));
    if (!hrow || !hcol || !hval) {
        free(hrow); free(hcol); free(hval);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    for (matx_int64_t i = 0; i < cap; ++i)
        hrow[i] = SPGEMM_AOCL_HASH_EMPTY;

    /* Build CSR-style row-index for B to avoid O(nnz²) linear scan. */
    matx_int64_t* b_row_cnt   = (matx_int64_t*)calloc((size_t)B->nrows, sizeof(matx_int64_t));
    matx_int64_t* b_row_start = (matx_int64_t*)malloc(((size_t)B->nrows + 1) * sizeof(matx_int64_t));
    if (!b_row_cnt || !b_row_start) {
        free(hrow); free(hcol); free(hval);
        free(b_row_cnt); free(b_row_start);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    for (matx_int64_t ib = 0; ib < bnnz; ++ib)
        b_row_cnt[B->rows[ib]]++;
    b_row_start[0] = 0;
    for (matx_int64_t r = 0; r < B->nrows; ++r) {
        b_row_start[r + 1] = b_row_start[r] + b_row_cnt[r];
        b_row_cnt[r] = 0;
    }
    matx_int64_t* b_perm = (matx_int64_t*)malloc((size_t)bnnz * sizeof(matx_int64_t));
    if (!b_perm) {
        free(hrow); free(hcol); free(hval);
        free(b_row_cnt); free(b_row_start);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    for (matx_int64_t ib = 0; ib < bnnz; ++ib) {
        matx_int64_t r   = B->rows[ib];
        matx_int64_t pos = b_row_start[r] + b_row_cnt[r];
        b_perm[pos] = ib;
        b_row_cnt[r]++;
    }
    free(b_row_cnt);

    for (matx_int64_t ia = 0; ia < annz; ++ia) {
        matx_int64_t       k  = A->columns[ia];
        matx_int64_t       i  = A->rows[ia];
        matx_complex_d_t   av = A->values[ia];
        double avr = alpha.real * av.real - alpha.imag * av.imag;
        double avi = alpha.real * av.imag + alpha.imag * av.real;

        /* Only iterate B entries in row k (CSR row-index lookup) */
        for (matx_int64_t p = b_row_start[k]; p < b_row_start[k + 1]; ++p) {
            matx_int64_t ib = b_perm[p];
            matx_int64_t     j  = B->columns[ib];
            matx_complex_d_t bv = B->values[ib];
            double pr = avr * bv.real - avi * bv.imag;
            double pi = avr * bv.imag + avi * bv.real;

            matx_int64_t idx = (i * 2654435761ULL + (unsigned long long)j) & (cap - 1);
            while (hrow[idx] != SPGEMM_AOCL_HASH_EMPTY) {
                if (hrow[idx] == i && hcol[idx] == j) {
                    hval[idx].real += pr;
                    hval[idx].imag += pi;
                    goto next_bza;
                }
                idx = (idx + 1) & (cap - 1);
            }
            hrow[idx] = i;
            hcol[idx] = j;
            hval[idx].real = pr;
            hval[idx].imag = pi;
            next_bza:;
        }
    }
    free(b_row_start); free(b_perm);

    matx_int64_t nnz_out = 0;
    for (matx_int64_t i = 0; i < cap; ++i)
        if (hrow[i] != SPGEMM_AOCL_HASH_EMPTY &&
            (hval[i].real != 0.0 || hval[i].imag != 0.0))
            nnz_out++;

    matx_int64_t pos = 0;
    for (matx_int64_t i = 0; i < cap; ++i) {
        if (hrow[i] != SPGEMM_AOCL_HASH_EMPTY &&
            (hval[i].real != 0.0 || hval[i].imag != 0.0)) {
            out->rows[pos]    = hrow[i];
            out->columns[pos] = hcol[i];
            out->values[pos]  = hval[i];
            pos++;
        }
    }
    out->nnz = nnz_out;

    free(hrow); free(hcol); free(hval);
    return MATX_OK;
}

// ---- SpTRSV (reference) ----

static matx_status_t ref_sptrsv_d_i8_aocl(matx_uplo_t uplo,
                                           matx_trans_t trans,
                                           matx_diag_t diag,
                                           matx_coo_d_i8_t A,
                                           matx_vec_d_i8_t x)
{
    if (!A || !x || !x->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (A->nrows != A->ncols || A->nrows != x->n) {
        MATX_ERROR("%s: dimension mismatch", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (trans != MATX_NO_TRANS) {
        MATX_ERROR("%s: transposed SpTRSV not yet implemented", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    matx_int64_t n = A->nrows;
    matx_int64_t nnz = A->nnz;

    matx_int64_t* row_start = (matx_int64_t*)calloc((size_t)(n + 1), sizeof(matx_int64_t));
    matx_int64_t* row_cnt   = (matx_int64_t*)calloc((size_t)n, sizeof(matx_int64_t));
    if (!row_start || !row_cnt) {
        free(row_start); free(row_cnt);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    for (matx_int64_t k = 0; k < nnz; ++k)
        row_cnt[A->rows[k]]++;
    matx_int64_t total = 0;
    for (matx_int64_t i = 0; i < n; ++i) {
        row_start[i] = total;
        total += row_cnt[i];
        row_cnt[i] = 0;
    }
    row_start[n] = total;

    matx_int64_t* perm = (matx_int64_t*)malloc((size_t)nnz * sizeof(matx_int64_t));
    if (!perm) {
        free(row_start); free(row_cnt);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    for (matx_int64_t k = 0; k < nnz; ++k) {
        matx_int64_t r = A->rows[k];
        matx_int64_t pos = row_start[r] + row_cnt[r];
        perm[pos] = k;
        row_cnt[r]++;
    }

    int is_lower = (uplo == MATX_LOWER);
    int is_unit  = (diag == MATX_UNIT_DIAG);

    if (is_lower) {
        for (matx_int64_t i = 0; i < n; ++i) {
            double sum = 0.0;
            double diag_val = 1.0;
            for (matx_int64_t p = row_start[i]; p < row_start[i + 1]; ++p) {
                matx_int64_t k = perm[p];
                matx_int64_t j = A->columns[k];
                if (j < i) {
                    sum += A->values[k] * x->data[j * x->stride];
                } else if (j == i) {
                    diag_val = A->values[k];
                }
            }
            if (is_unit) {
                x->data[i * x->stride] = x->data[i * x->stride] - sum;
            } else {
                if (diag_val == 0.0) {
                    free(perm); free(row_start); free(row_cnt);
                    MATX_ERROR("%s: zero diagonal at row %lld", __func__, (long long)i);
                    return MATX_ERR_INVALID_ARG;
                }
                x->data[i * x->stride] = (x->data[i * x->stride] - sum) / diag_val;
            }
        }
    } else {
        for (matx_int64_t i = n - 1; i >= 0; --i) {
            double sum = 0.0;
            double diag_val = 1.0;
            for (matx_int64_t p = row_start[i]; p < row_start[i + 1]; ++p) {
                matx_int64_t k = perm[p];
                matx_int64_t j = A->columns[k];
                if (j > i) {
                    sum += A->values[k] * x->data[j * x->stride];
                } else if (j == i) {
                    diag_val = A->values[k];
                }
            }
            if (is_unit) {
                x->data[i * x->stride] = x->data[i * x->stride] - sum;
            } else {
                if (diag_val == 0.0) {
                    free(perm); free(row_start); free(row_cnt);
                    MATX_ERROR("%s: zero diagonal at row %lld", __func__, (long long)i);
                    return MATX_ERR_INVALID_ARG;
                }
                x->data[i * x->stride] = (x->data[i * x->stride] - sum) / diag_val;
            }
        }
    }

    free(perm); free(row_start); free(row_cnt);
    return MATX_OK;
}

static matx_status_t ref_sptrsv_z_i8_aocl(matx_uplo_t uplo,
                                           matx_trans_t trans,
                                           matx_diag_t diag,
                                           matx_coo_z_i8_t A,
                                           matx_vec_z_i8_t x)
{
    if (!A || !x || !x->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (A->nrows != A->ncols || A->nrows != x->n) {
        MATX_ERROR("%s: dimension mismatch", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (trans != MATX_NO_TRANS) {
        MATX_ERROR("%s: transposed SpTRSV not yet implemented", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    matx_int64_t n = A->nrows;
    matx_int64_t nnz = A->nnz;

    matx_int64_t* row_start = (matx_int64_t*)calloc((size_t)(n + 1), sizeof(matx_int64_t));
    matx_int64_t* row_cnt   = (matx_int64_t*)calloc((size_t)n, sizeof(matx_int64_t));
    if (!row_start || !row_cnt) {
        free(row_start); free(row_cnt);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    for (matx_int64_t k = 0; k < nnz; ++k)
        row_cnt[A->rows[k]]++;
    matx_int64_t total = 0;
    for (matx_int64_t i = 0; i < n; ++i) {
        row_start[i] = total;
        total += row_cnt[i];
        row_cnt[i] = 0;
    }
    row_start[n] = total;

    matx_int64_t* perm = (matx_int64_t*)malloc((size_t)nnz * sizeof(matx_int64_t));
    if (!perm) {
        free(row_start); free(row_cnt);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    for (matx_int64_t k = 0; k < nnz; ++k) {
        matx_int64_t r = A->rows[k];
        matx_int64_t pos = row_start[r] + row_cnt[r];
        perm[pos] = k;
        row_cnt[r]++;
    }

    int is_lower = (uplo == MATX_LOWER);
    int is_unit  = (diag == MATX_UNIT_DIAG);

    if (is_lower) {
        for (matx_int64_t i = 0; i < n; ++i) {
            double sum_re = 0.0, sum_im = 0.0;
            matx_complex_d_t diag_val = {1.0, 0.0};
            for (matx_int64_t p = row_start[i]; p < row_start[i + 1]; ++p) {
                matx_int64_t k = perm[p];
                matx_int64_t j = A->columns[k];
                if (j < i) {
                    matx_complex_d_t ak = A->values[k];
                    matx_complex_d_t xj = x->data[j * x->stride];
                    sum_re += ak.real * xj.real - ak.imag * xj.imag;
                    sum_im += ak.real * xj.imag + ak.imag * xj.real;
                } else if (j == i) {
                    diag_val = A->values[k];
                }
            }
            if (is_unit) {
                x->data[i * x->stride].real -= sum_re;
                x->data[i * x->stride].imag -= sum_im;
            } else {
                if (diag_val.real == 0.0 && diag_val.imag == 0.0) {
                    free(perm); free(row_start); free(row_cnt);
                    MATX_ERROR("%s: zero diagonal at row %lld", __func__, (long long)i);
                    return MATX_ERR_INVALID_ARG;
                }
                double denom = diag_val.real * diag_val.real + diag_val.imag * diag_val.imag;
                double xr = x->data[i * x->stride].real - sum_re;
                double xi = x->data[i * x->stride].imag - sum_im;
                x->data[i * x->stride].real = (xr * diag_val.real + xi * diag_val.imag) / denom;
                x->data[i * x->stride].imag = (xi * diag_val.real - xr * diag_val.imag) / denom;
            }
        }
    } else {
        for (matx_int64_t i = n - 1; i >= 0; --i) {
            double sum_re = 0.0, sum_im = 0.0;
            matx_complex_d_t diag_val = {1.0, 0.0};
            for (matx_int64_t p = row_start[i]; p < row_start[i + 1]; ++p) {
                matx_int64_t k = perm[p];
                matx_int64_t j = A->columns[k];
                if (j > i) {
                    matx_complex_d_t ak = A->values[k];
                    matx_complex_d_t xj = x->data[j * x->stride];
                    sum_re += ak.real * xj.real - ak.imag * xj.imag;
                    sum_im += ak.real * xj.imag + ak.imag * xj.real;
                } else if (j == i) {
                    diag_val = A->values[k];
                }
            }
            if (is_unit) {
                x->data[i * x->stride].real -= sum_re;
                x->data[i * x->stride].imag -= sum_im;
            } else {
                if (diag_val.real == 0.0 && diag_val.imag == 0.0) {
                    free(perm); free(row_start); free(row_cnt);
                    MATX_ERROR("%s: zero diagonal at row %lld", __func__, (long long)i);
                    return MATX_ERR_INVALID_ARG;
                }
                double denom = diag_val.real * diag_val.real + diag_val.imag * diag_val.imag;
                double xr = x->data[i * x->stride].real - sum_re;
                double xi = x->data[i * x->stride].imag - sum_im;
                x->data[i * x->stride].real = (xr * diag_val.real + xi * diag_val.imag) / denom;
                x->data[i * x->stride].imag = (xi * diag_val.real - xr * diag_val.imag) / denom;
            }
        }
    }

    free(perm); free(row_start); free(row_cnt);
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
                               .spgemm_d_i8 = ref_spgemm_d_i8_aocl,
                               .spgemm_z_i8 = ref_spgemm_z_i8_aocl,
                               .sptrsv_d_i8 = ref_sptrsv_d_i8_aocl,
                               .sptrsv_z_i8 = ref_sptrsv_z_i8_aocl,
                               }};
    MATX_TRACE("AOCL INIT");
    return b;
}