#include "matx/matx_func.h"
#include "matx/matx_log.h"
#include "matx/matx_sparse_compute.h"
#include "matx/matx_tm.h"
#include "matx/matx_types.h"
#include "matx/matx_types_internal.h"

#ifdef MATX_ENABLE_GRAPHBLAS
#include "GraphBLAS.h"
#endif

#include <limits.h>
#include <math.h>
#include <stdlib.h>

//C(i,j)=k⨁​(A(i,k)⊗B(k,j))

/* These crossover limits come from the bundled scale/density benchmark sweep. */
#define MATX_SPARSE_DIRECT_SPMV_MAX_NNZ 3000000
#define MATX_SPARSE_DIRECT_SPMM_MAX_PRODUCTS 50000000
#define MATX_SPARSE_DIRECT_NORM_MAX_NNZ 8192
#define MATX_SPARSE_DIRECT_NORM1_REAL_MAX_NNZ 2048
#define MATX_SPARSE_DIRECT_NORM_MAX_DIMENSION 65536

typedef struct
{
    matx_int64_t row;
    matx_int64_t col;
    matx_double real;
    matx_double imag;
    unsigned char occupied;
} matx_sparse_norm_entry_t;

typedef enum
{
    MATX_SPARSE_NORM_ONE,
    MATX_SPARSE_NORM_INFINITY,
    MATX_SPARSE_NORM_FROBENIUS
} matx_sparse_norm_kind_t;

static uint64_t sparse_coordinate_hash(matx_int64_t row, matx_int64_t col)
{
    uint64_t value = (uint64_t) row * UINT64_C(0x9e3779b185ebca87)
                     ^ (uint64_t) col * UINT64_C(0xc2b2ae3d27d4eb4f);
    value ^= value >> 33;
    value *= UINT64_C(0xff51afd7ed558ccd);
    value ^= value >> 33;
    value *= UINT64_C(0xc4ceb9fe1a85ec53);
    return value ^ (value >> 33);
}

/*
 * GraphBLAS reductions allocate temporary matrices and vectors for each norm.
 * For small COO inputs, coalesce duplicates in a temporary hash table and
 * reduce directly. This preserves COO-as-matrix semantics: duplicates are
 * summed before taking absolute values or squares.
 */
static matx_status_t ref_sparse_norm_coo(const matx_alloc_t* alloc,
                                         matx_int64_t nrows,
                                         matx_int64_t ncols,
                                         matx_int64_t nnz,
                                         const matx_int64_t* rows,
                                         const matx_int64_t* columns,
                                         const matx_double* real_values,
                                         const matx_complex_d_t* complex_values,
                                         matx_sparse_norm_kind_t kind,
                                         matx_double* out)
{
    if (!alloc || !out || nrows <= 0 || ncols <= 0 || nnz < 0
        || (nnz > 0 && (!rows || !columns || (!real_values && !complex_values)))) {
        return MATX_ERR_INVALID_ARG;
    }
    if (nnz == 0) {
        *out = 0.0;
        return MATX_OK;
    }
    if ((uint64_t) nnz > SIZE_MAX / 2) {
        return MATX_ERR_INVALID_ARG;
    }

    size_t capacity = 1;
    const size_t minimum_capacity = (size_t) nnz * 2;
    while (capacity < minimum_capacity) {
        if (capacity > SIZE_MAX / 2) {
            return MATX_ERR_INVALID_ARG;
        }
        capacity *= 2;
    }
    if (capacity > SIZE_MAX / sizeof(matx_sparse_norm_entry_t)) {
        return MATX_ERR_INVALID_ARG;
    }

    matx_sparse_norm_entry_t* entries
        = (matx_sparse_norm_entry_t*) matx_malloc(alloc,
                                                  capacity * sizeof(*entries));
    if (!entries) {
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(entries, 0, capacity * sizeof(*entries));
    const size_t mask = capacity - 1;
    for (matx_int64_t i = 0; i < nnz; ++i) {
        const matx_int64_t row = rows[i];
        const matx_int64_t col = columns[i];
        if (row < 0 || row >= nrows || col < 0 || col >= ncols) {
            matx_free(alloc, entries);
            return MATX_ERR_INVALID_ARG;
        }
        const matx_double real = complex_values ? complex_values[i].real : real_values[i];
        const matx_double imag = complex_values ? complex_values[i].imag : 0.0;
        if (isnan(real) || isnan(imag)) {
            matx_free(alloc, entries);
            return MATX_ERR_NOT_SUPPORTED;
        }

        size_t slot = (size_t) sparse_coordinate_hash(row, col) & mask;
        while (entries[slot].occupied
               && (entries[slot].row != row || entries[slot].col != col)) {
            slot = (slot + 1) & mask;
        }
        if (!entries[slot].occupied) {
            entries[slot].row = row;
            entries[slot].col = col;
            entries[slot].occupied = 1;
        }
        entries[slot].real += real;
        entries[slot].imag += imag;
        if (isnan(entries[slot].real) || isnan(entries[slot].imag)) {
            matx_free(alloc, entries);
            return MATX_ERR_NOT_SUPPORTED;
        }
    }

    matx_status_t status = MATX_OK;
    if (kind == MATX_SPARSE_NORM_FROBENIUS) {
        matx_double sum_squares = 0.0;
        for (size_t i = 0; i < capacity; ++i) {
            if (!entries[i].occupied) continue;
            const matx_double magnitude = complex_values
                                              ? hypot(entries[i].real, entries[i].imag)
                                              : fabs(entries[i].real);
            sum_squares += magnitude * magnitude;
        }
        *out = sqrt(sum_squares);
    } else {
        const matx_int64_t count = kind == MATX_SPARSE_NORM_ONE ? ncols : nrows;
        if ((uint64_t) count > SIZE_MAX / sizeof(matx_double)) {
            status = MATX_ERR_INVALID_ARG;
        } else {
            matx_double* sums = (matx_double*) matx_malloc(
                alloc, (size_t) count * sizeof(*sums));
            if (!sums) {
                status = MATX_ERR_OUT_OF_MEMORY;
            } else {
                memset(sums, 0, (size_t) count * sizeof(*sums));
                for (size_t i = 0; i < capacity; ++i) {
                    if (!entries[i].occupied) continue;
                    const matx_double magnitude = complex_values
                                                      ? hypot(entries[i].real, entries[i].imag)
                                                      : fabs(entries[i].real);
                    const matx_int64_t index = kind == MATX_SPARSE_NORM_ONE
                                                   ? entries[i].col
                                                   : entries[i].row;
                    sums[index] += magnitude;
                }
                matx_double maximum = 0.0;
                for (matx_int64_t i = 0; i < count; ++i) {
                    if (sums[i] > maximum) maximum = sums[i];
                }
                *out = maximum;
                matx_free(alloc, sums);
            }
        }
    }
    matx_free(alloc, entries);
    return status;
}

static matx_complex_d_t complex_multiply(matx_complex_d_t a, matx_complex_d_t b)
{
    matx_complex_d_t result = {
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real
    };
    return result;
}

static matx_status_t ref_spmv_d_i8_coo(matx_double alpha,
                                       matx_coo_d_i8_t A,
                                       matx_vec_d_i8_t x,
                                       matx_double beta,
                                       matx_vec_d_i8_t y)
{
    if (!A || !x || !y || !A->rows || !A->columns || !A->values || !x->data || !y->data
        || A->ncols != x->n || A->nrows != y->n || A->nnz < 0 || x->stride <= 0
        || y->stride <= 0) {
        return MATX_ERR_INVALID_ARG;
    }

    matx_double* x_copy = NULL;
    const matx_double* x_data = x->data;
    matx_int64_t x_stride = x->stride;
    if (x == y) {
        x_copy = (matx_double*) malloc((size_t) x->n * sizeof(*x_copy));
        if (!x_copy) return MATX_ERR_OUT_OF_MEMORY;
        for (matx_int64_t i = 0; i < x->n; ++i)
            x_copy[i] = x->data[i * x->stride];
        x_data = x_copy;
        x_stride = 1;
    }

    for (matx_int64_t i = 0; i < y->n; ++i) {
        matx_double* output = &y->data[i * y->stride];
        if (beta == 0.0)
            *output = 0.0;
        else if (beta != 1.0)
            *output *= beta;
    }
    if (alpha != 0.0) {
        for (matx_int64_t k = 0; k < A->nnz; ++k) {
            const matx_int64_t row = A->rows[k];
            const matx_int64_t col = A->columns[k];
            const matx_double product = A->values[k] * x_data[col * x_stride];
            y->data[row * y->stride] += alpha == 1.0 ? product : alpha * product;
        }
    }
    free(x_copy);
    return MATX_OK;
}

static matx_status_t ref_spmv_z_i8_coo(matx_complex_d_t alpha,
                                       matx_coo_z_i8_t A,
                                       matx_vec_z_i8_t x,
                                       matx_complex_d_t beta,
                                       matx_vec_z_i8_t y)
{
    if (!A || !x || !y || !A->rows || !A->columns || !A->values || !x->data || !y->data
        || A->ncols != x->n || A->nrows != y->n || A->nnz < 0 || x->stride <= 0
        || y->stride <= 0) {
        return MATX_ERR_INVALID_ARG;
    }

    matx_complex_d_t* x_copy = NULL;
    const matx_complex_d_t* x_data = x->data;
    matx_int64_t x_stride = x->stride;
    if (x == y) {
        x_copy = (matx_complex_d_t*) malloc((size_t) x->n * sizeof(*x_copy));
        if (!x_copy) return MATX_ERR_OUT_OF_MEMORY;
        for (matx_int64_t i = 0; i < x->n; ++i)
            x_copy[i] = x->data[i * x->stride];
        x_data = x_copy;
        x_stride = 1;
    }

    for (matx_int64_t i = 0; i < y->n; ++i) {
        matx_complex_d_t* output = &y->data[i * y->stride];
        if (beta.real == 0.0 && beta.imag == 0.0)
            *output = (matx_complex_d_t) {0.0, 0.0};
        else if (beta.real != 1.0 || beta.imag != 0.0)
            *output = complex_multiply(beta, *output);
    }
    if (alpha.real != 0.0 || alpha.imag != 0.0) {
        for (matx_int64_t k = 0; k < A->nnz; ++k) {
            const matx_int64_t row = A->rows[k];
            const matx_int64_t col = A->columns[k];
            const matx_complex_d_t product
                = complex_multiply(A->values[k], x_data[col * x_stride]);
            const matx_complex_d_t scaled
                = alpha.real == 1.0 && alpha.imag == 0.0
                      ? product
                      : complex_multiply(alpha, product);
            y->data[row * y->stride].real += scaled.real;
            y->data[row * y->stride].imag += scaled.imag;
        }
    }
    free(x_copy);
    return MATX_OK;
}

static matx_status_t ref_spmm_d_i8_coo(matx_double alpha,
                                       matx_coo_d_i8_t A,
                                       matx_dense_d_i8_t B,
                                       matx_double beta,
                                       matx_dense_d_i8_t C)
{
    if (!A || !B || !C || !A->rows || !A->columns || !A->values || !B->data || !C->data
        || A->ncols != B->nrows || A->nrows != C->nrows || B->ncols != C->ncols
        || A->nnz < 0 || B->stride <= 0 || C->stride <= 0) {
        return MATX_ERR_INVALID_ARG;
    }

    matx_double* b_copy = NULL;
    const matx_double* b_data = B->data;
    matx_int64_t b_stride = B->stride;
    matx_layout_t b_layout = B->layout;
    if (B == C) {
        b_copy = (matx_double*) malloc((size_t) B->nrows * (size_t) B->ncols * sizeof(*b_copy));
        if (!b_copy) return MATX_ERR_OUT_OF_MEMORY;
        for (matx_int64_t col = 0; col < B->ncols; ++col) {
            for (matx_int64_t row = 0; row < B->nrows; ++row) {
                const size_t source = B->layout == MATX_COL_MAJOR
                                          ? (size_t) row + (size_t) col * B->stride
                                          : (size_t) row * B->stride + (size_t) col;
                b_copy[(size_t) row + (size_t) col * B->nrows] = B->data[source];
            }
        }
        b_data = b_copy;
        b_stride = B->nrows;
        b_layout = MATX_COL_MAJOR;
    }

    for (matx_int64_t col = 0; col < C->ncols; ++col) {
        for (matx_int64_t row = 0; row < C->nrows; ++row) {
            const size_t index = C->layout == MATX_COL_MAJOR
                                     ? (size_t) row + (size_t) col * C->stride
                                     : (size_t) row * C->stride + (size_t) col;
            if (beta == 0.0)
                C->data[index] = 0.0;
            else if (beta != 1.0)
                C->data[index] *= beta;
        }
    }
    if (alpha != 0.0) {
        for (matx_int64_t k = 0; k < A->nnz; ++k) {
            const matx_int64_t row = A->rows[k];
            const matx_int64_t inner = A->columns[k];
            for (matx_int64_t col = 0; col < B->ncols; ++col) {
                const size_t b_index = b_layout == MATX_COL_MAJOR
                                           ? (size_t) inner + (size_t) col * b_stride
                                           : (size_t) inner * b_stride + (size_t) col;
                const size_t c_index = C->layout == MATX_COL_MAJOR
                                           ? (size_t) row + (size_t) col * C->stride
                                           : (size_t) row * C->stride + (size_t) col;
                const matx_double product = A->values[k] * b_data[b_index];
                C->data[c_index] += alpha == 1.0 ? product : alpha * product;
            }
        }
    }
    free(b_copy);
    return MATX_OK;
}

static matx_status_t ref_spmm_z_i8_coo(matx_complex_d_t alpha,
                                       matx_coo_z_i8_t A,
                                       matx_dense_z_i8_t B,
                                       matx_complex_d_t beta,
                                       matx_dense_z_i8_t C)
{
    if (!A || !B || !C || !A->rows || !A->columns || !A->values || !B->data || !C->data
        || A->ncols != B->nrows || A->nrows != C->nrows || B->ncols != C->ncols
        || A->nnz < 0 || B->stride <= 0 || C->stride <= 0) {
        return MATX_ERR_INVALID_ARG;
    }

    matx_complex_d_t* b_copy = NULL;
    const matx_complex_d_t* b_data = B->data;
    matx_int64_t b_stride = B->stride;
    matx_layout_t b_layout = B->layout;
    if (B == C) {
        b_copy = (matx_complex_d_t*) malloc((size_t) B->nrows * (size_t) B->ncols * sizeof(*b_copy));
        if (!b_copy) return MATX_ERR_OUT_OF_MEMORY;
        for (matx_int64_t col = 0; col < B->ncols; ++col) {
            for (matx_int64_t row = 0; row < B->nrows; ++row) {
                const size_t source = B->layout == MATX_COL_MAJOR
                                          ? (size_t) row + (size_t) col * B->stride
                                          : (size_t) row * B->stride + (size_t) col;
                b_copy[(size_t) row + (size_t) col * B->nrows] = B->data[source];
            }
        }
        b_data = b_copy;
        b_stride = B->nrows;
        b_layout = MATX_COL_MAJOR;
    }

    for (matx_int64_t col = 0; col < C->ncols; ++col) {
        for (matx_int64_t row = 0; row < C->nrows; ++row) {
            const size_t index = C->layout == MATX_COL_MAJOR
                                     ? (size_t) row + (size_t) col * C->stride
                                     : (size_t) row * C->stride + (size_t) col;
            if (beta.real == 0.0 && beta.imag == 0.0)
                C->data[index] = (matx_complex_d_t) {0.0, 0.0};
            else if (beta.real != 1.0 || beta.imag != 0.0)
                C->data[index] = complex_multiply(beta, C->data[index]);
        }
    }
    if (alpha.real != 0.0 || alpha.imag != 0.0) {
        for (matx_int64_t k = 0; k < A->nnz; ++k) {
            const matx_int64_t row = A->rows[k];
            const matx_int64_t inner = A->columns[k];
            for (matx_int64_t col = 0; col < B->ncols; ++col) {
                const size_t b_index = b_layout == MATX_COL_MAJOR
                                           ? (size_t) inner + (size_t) col * b_stride
                                           : (size_t) inner * b_stride + (size_t) col;
                const size_t c_index = C->layout == MATX_COL_MAJOR
                                           ? (size_t) row + (size_t) col * C->stride
                                           : (size_t) row * C->stride + (size_t) col;
                const matx_complex_d_t product
                    = complex_multiply(A->values[k], b_data[b_index]);
                const matx_complex_d_t scaled
                    = alpha.real == 1.0 && alpha.imag == 0.0
                          ? product
                          : complex_multiply(alpha, product);
                C->data[c_index].real += scaled.real;
                C->data[c_index].imag += scaled.imag;
            }
        }
    }
    free(b_copy);
    return MATX_OK;
}

static matx_status_t ref_spmv_z_i8_grb(matx_complex_d_t alpha,
                                       matx_coo_z_i8_t A,
                                       matx_vec_z_i8_t x,
                                       matx_complex_d_t beta,
                                       matx_vec_z_i8_t y)
{
#ifdef MATX_ENABLE_GRAPHBLAS
    if (A && A->nnz <= MATX_SPARSE_DIRECT_SPMV_MAX_NNZ)
        return ref_spmv_z_i8_coo(alpha, A, x, beta, y);

    if (!A || !x || !y) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }

    if (A->ncols != x->n || A->nrows != y->n) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }

    if (MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->valid <= 0
        && coo_2_grb_z_i8(A) != 0) {
        return MATX_ERR_INTERNAL;
    }
    if (MATX_HANDLE(x, MATX_HANDLE_TYPE_GRB_VECTOR)->valid <= 0
        && vec_2_grb_z_i8(x) != 0) {
        return MATX_ERR_INTERNAL;
    }
    if (MATX_HANDLE(y, MATX_HANDLE_TYPE_GRB_VECTOR)->valid <= 0
        && vec_2_grb_z_i8(y) != 0) {
        if (MATX_HANDLE(x, MATX_HANDLE_TYPE_GRB_VECTOR)->valid > 0)
            grb_2_vec_z_i8(x);
        return MATX_ERR_INTERNAL;
    }

    GxB_FC64_t a = GxB_CMPLX(alpha.real, alpha.imag);
    GxB_FC64_t b = GxB_CMPLX(beta.real, beta.imag);

    // gy = beta * gy
    GrB_Info info = GrB_apply((GrB_Vector) MATX_HANDLE(y, MATX_HANDLE_TYPE_GRB_VECTOR)->impl,
                              NULL,
                              NULL,
                              GxB_TIMES_FC64,
                              (GrB_Vector) MATX_HANDLE(y, MATX_HANDLE_TYPE_GRB_VECTOR)->impl,
                              b,
                              NULL);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("GrB_apply beta*gy error: %d", info);
        grb_2_vec_z_i8(x);
        grb_2_vec_z_i8(y);
        return MATX_ERR_INTERNAL;
    }
    // gy += alpha * A * x  (accumulate into gy)
    GrB_Vector temp;
    info = GrB_Vector_new(&temp, GxB_FC64, y->n);
    if (info != GrB_SUCCESS) {
        grb_2_vec_z_i8(x);
        grb_2_vec_z_i8(y);
        return MATX_ERR_INTERNAL;
    }
    info = GrB_mxv(temp,
                   NULL,
                   NULL,
                   GxB_PLUS_TIMES_FC64,
                   (GrB_Matrix) MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl,
                   (GrB_Vector) MATX_HANDLE(x, MATX_HANDLE_TYPE_GRB_VECTOR)->impl,
                   NULL);
    if (info != GrB_SUCCESS) {
        GrB_Vector_free(&temp);
        MATX_ERROR("GrB_mxv error: %d", info);
        grb_2_vec_z_i8(x);
        grb_2_vec_z_i8(y);
        return MATX_ERR_INTERNAL;
    }
    info = GrB_apply(temp, NULL, NULL, GxB_TIMES_FC64, temp, a, NULL);
    if (info != GrB_SUCCESS) {
        GrB_Vector_free(&temp);
        MATX_ERROR("GrB_apply alpha*temp error: %d", info);
        grb_2_vec_z_i8(x);
        grb_2_vec_z_i8(y);
        return MATX_ERR_INTERNAL;
    }
    info = GrB_eWiseAdd((GrB_Vector) MATX_HANDLE(y, MATX_HANDLE_TYPE_GRB_VECTOR)->impl,
                        NULL,
                        NULL,
                        GxB_PLUS_FC64,
                        (GrB_Vector) MATX_HANDLE(y, MATX_HANDLE_TYPE_GRB_VECTOR)->impl,
                        temp,
                        NULL);
    GrB_Vector_free(&temp);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("GrB_eWiseAdd error: %d", info);
        grb_2_vec_z_i8(x);
        grb_2_vec_z_i8(y);
        return MATX_ERR_INTERNAL;
    }
    const size_t x_export_status = grb_2_vec_z_i8(x);
    const size_t y_export_status = grb_2_vec_z_i8(y);
    if (x_export_status != 0 || y_export_status != 0)
        return MATX_ERR_INTERNAL;
#else
    (void) alpha; (void) A; (void) x; (void) beta; (void) y;
    MATX_ERROR("%s: GraphBLAS not available", __func__);
    return MATX_ERR_NOT_SUPPORTED;
#endif
    return MATX_OK;
}

static matx_status_t ref_spmm_z_i8_grb(matx_complex_d_t alpha,
                                       matx_coo_z_i8_t A,
                                       matx_dense_z_i8_t B,
                                       matx_complex_d_t beta,
                                       matx_dense_z_i8_t C)
{
#ifdef MATX_ENABLE_GRAPHBLAS
    if (A && B && B->ncols > 0
        && (uint64_t) A->nnz * (uint64_t) B->ncols <= MATX_SPARSE_DIRECT_SPMM_MAX_PRODUCTS) {
        return ref_spmm_z_i8_coo(alpha, A, B, beta, C);
    }

    if (!A || !B || !C) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }

    if (A->ncols != B->nrows || A->nrows != C->nrows || B->ncols != C->ncols) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }

    /* build A */
    if (MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->valid <= 0
        && coo_2_grb_z_i8(A) != 0) {
        return MATX_ERR_INTERNAL;
    }
    if (MATX_HANDLE(B, MATX_HANDLE_TYPE_GRB_MATRIX)->valid <= 0
        && dense_2_grb_z_i8(B) != 0) {
        return MATX_ERR_INTERNAL;
    }
    if (MATX_HANDLE(C, MATX_HANDLE_TYPE_GRB_MATRIX)->valid <= 0
        && dense_2_grb_z_i8(C) != 0) {
        if (MATX_HANDLE(B, MATX_HANDLE_TYPE_GRB_MATRIX)->valid > 0)
            grb_2_dense_z_i8(B);
        return MATX_ERR_INTERNAL;
    }

    /* C = alpha*A*B + beta*C */
    GxB_FC64_t a = GxB_CMPLX(alpha.real, alpha.imag);
    GxB_FC64_t b = GxB_CMPLX(beta.real, beta.imag);

    // 1. temp = alpha * A * B
    GrB_Matrix temp;
    GrB_Info info = GrB_Matrix_new(&temp, GxB_FC64, C->nrows, C->ncols);
    if (info != GrB_SUCCESS) {
        grb_2_dense_z_i8(B);
        grb_2_dense_z_i8(C);
        return MATX_ERR_INTERNAL;
    }
    info = GrB_mxm(temp,
                            NULL,
                            NULL,
                            GxB_PLUS_TIMES_FC64,
                            (GrB_Matrix) MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl,
                            (GrB_Matrix) MATX_HANDLE(B, MATX_HANDLE_TYPE_GRB_MATRIX)->impl,
                            NULL);
    if (info != GrB_SUCCESS) {
        GrB_Matrix_free(&temp);
        MATX_ERROR("GrB_mxm error: %d", info);
        grb_2_dense_z_i8(B);
        grb_2_dense_z_i8(C);
        return MATX_ERR_INTERNAL;
    }
    info = GrB_apply(temp, NULL, NULL, GxB_TIMES_FC64, temp, a, NULL);
    if (info != GrB_SUCCESS) {
        GrB_Matrix_free(&temp);
        MATX_ERROR("GrB_apply error: %d", info);
        grb_2_dense_z_i8(B);
        grb_2_dense_z_i8(C);
        return MATX_ERR_INTERNAL;
    }
    //2. gC = beta * gC
    info = GrB_apply((GrB_Matrix) MATX_HANDLE(C, MATX_HANDLE_TYPE_GRB_MATRIX)->impl,
                     NULL,
                     NULL,
                     GxB_TIMES_FC64,
                     (GrB_Matrix) MATX_HANDLE(C, MATX_HANDLE_TYPE_GRB_MATRIX)->impl,
                     b,
                     NULL);
    if (info != GrB_SUCCESS) {
        GrB_Matrix_free(&temp);
        MATX_ERROR("GrB_apply beta * gC error: %d", info);
        grb_2_dense_z_i8(B);
        grb_2_dense_z_i8(C);
        return MATX_ERR_INTERNAL;
    }
    //3. gC = temp + gC
    info = GrB_eWiseAdd((GrB_Matrix) MATX_HANDLE(C, MATX_HANDLE_TYPE_GRB_MATRIX)->impl,
                        NULL,
                        NULL,
                        GxB_PLUS_FC64,
                        temp,
                        (GrB_Matrix) MATX_HANDLE(C, MATX_HANDLE_TYPE_GRB_MATRIX)->impl,
                        NULL);
    if (info != GrB_SUCCESS) {
        GrB_Matrix_free(&temp);
        MATX_ERROR("GrB_eWiseAdd temp + gC error: %d", info);
        grb_2_dense_z_i8(B);
        grb_2_dense_z_i8(C);
        return MATX_ERR_INTERNAL;
    }
    GrB_Matrix_free(&temp);
    const size_t b_export_status = grb_2_dense_z_i8(B);
    const size_t c_export_status = grb_2_dense_z_i8(C);
    if (b_export_status != 0 || c_export_status != 0)
        return MATX_ERR_INTERNAL;
#else
    (void) alpha; (void) A; (void) B; (void) beta; (void) C;
    MATX_ERROR("%s: GraphBLAS not available", __func__);
    return MATX_ERR_NOT_SUPPORTED;
#endif
    return MATX_OK;
}

static matx_status_t ref_spmv_d_i8_grb(
    matx_double alpha, matx_coo_d_i8_t A, matx_vec_d_i8_t x, matx_double beta, matx_vec_d_i8_t y)
{
#ifdef MATX_ENABLE_GRAPHBLAS
    if (A && A->nnz <= MATX_SPARSE_DIRECT_SPMV_MAX_NNZ)
        return ref_spmv_d_i8_coo(alpha, A, x, beta, y);

    if (!A || !x || !y) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }

    if (A->ncols != x->n || A->nrows != y->n) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }

    if (MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->valid <= 0
        && coo_2_grb_d_i8(A) != 0) {
        return MATX_ERR_INTERNAL;
    }
    if (MATX_HANDLE(x, MATX_HANDLE_TYPE_GRB_VECTOR)->valid <= 0
        && vec_2_grb_d_i8(x) != 0) {
        return MATX_ERR_INTERNAL;
    }
    if (MATX_HANDLE(y, MATX_HANDLE_TYPE_GRB_VECTOR)->valid <= 0
        && vec_2_grb_d_i8(y) != 0) {
        if (MATX_HANDLE(x, MATX_HANDLE_TYPE_GRB_VECTOR)->valid > 0)
            grb_2_vec_d_i8(x);
        return MATX_ERR_INTERNAL;
    }

    // gy = beta * gy
    GrB_Info info = GrB_apply((GrB_Vector) MATX_HANDLE(y, MATX_HANDLE_TYPE_GRB_VECTOR)->impl,
                              NULL,
                              NULL,
                              GrB_TIMES_FP64,
                              (GrB_Vector) MATX_HANDLE(y, MATX_HANDLE_TYPE_GRB_VECTOR)->impl,
                              beta,
                              NULL);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("GrB_apply beta*gy error: %d", info);
        grb_2_vec_d_i8(x);
        grb_2_vec_d_i8(y);
        return MATX_ERR_INTERNAL;
    }
    // temp = A*x, then gy += alpha*temp
    matx_int64_t t0 = matx_tm_now(MATX_TM_MICROSECOND);
    GrB_Vector temp;
    info = GrB_Vector_new(&temp, GrB_FP64, y->n);
    if (info != GrB_SUCCESS) {
        grb_2_vec_d_i8(x);
        grb_2_vec_d_i8(y);
        return MATX_ERR_INTERNAL;
    }
    info = GrB_mxv(temp,
                   NULL,
                   NULL,
                   GxB_PLUS_TIMES_FP64,
                   (GrB_Matrix) MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl,
                   (GrB_Vector) MATX_HANDLE(x, MATX_HANDLE_TYPE_GRB_VECTOR)->impl,
                   NULL);
    if (info != GrB_SUCCESS) {
        GrB_Vector_free(&temp);
        MATX_ERROR("GrB_mxv A*x error: %d", info);
        grb_2_vec_d_i8(x);
        grb_2_vec_d_i8(y);
        return MATX_ERR_INTERNAL;
    }
    info = GrB_apply(temp, NULL, NULL, GrB_TIMES_FP64, temp, alpha, NULL);
    if (info != GrB_SUCCESS) {
        GrB_Vector_free(&temp);
        MATX_ERROR("GrB_apply alpha*temp error: %d", info);
        grb_2_vec_d_i8(x);
        grb_2_vec_d_i8(y);
        return MATX_ERR_INTERNAL;
    }
    info = GrB_eWiseAdd((GrB_Vector) MATX_HANDLE(y, MATX_HANDLE_TYPE_GRB_VECTOR)->impl,
                        NULL,
                        NULL,
                        GrB_PLUS_FP64,
                        (GrB_Vector) MATX_HANDLE(y, MATX_HANDLE_TYPE_GRB_VECTOR)->impl,
                        temp,
                        NULL);
    GrB_Vector_free(&temp);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("GrB_eWiseAdd temp + gy error: %d", info);
        grb_2_vec_d_i8(x);
        grb_2_vec_d_i8(y);
        return MATX_ERR_INTERNAL;
    }
    matx_int64_t t1 = matx_tm_now(MATX_TM_MICROSECOND);
    MATX_TRACE("GraphBLAS SpMV time: %ld micro.s", t1 - t0);
    const size_t x_export_status = grb_2_vec_d_i8(x);
    const size_t y_export_status = grb_2_vec_d_i8(y);
    if (x_export_status != 0 || y_export_status != 0)
        return MATX_ERR_INTERNAL;
#else
    (void) alpha; (void) A; (void) x; (void) beta; (void) y;
    MATX_ERROR("%s: GraphBLAS not available", __func__);
    return MATX_ERR_NOT_SUPPORTED;
#endif
    return MATX_OK;
}

static matx_status_t ref_spmm_d_i8_grb(matx_double alpha,
                                       matx_coo_d_i8_t A,
                                       matx_dense_d_i8_t B,
                                       matx_double beta,
                                       matx_dense_d_i8_t C)
{
#ifdef MATX_ENABLE_GRAPHBLAS
    if (A && B && B->ncols > 0
        && A->nnz <= MATX_SPARSE_DIRECT_SPMV_MAX_NNZ
        && A->nnz <= MATX_SPARSE_DIRECT_SPMM_MAX_PRODUCTS / B->ncols) {
        return ref_spmm_d_i8_coo(alpha, A, B, beta, C);
    }

    if (!A || !B || !C) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }

    /* build A */
    if (A->ncols != B->nrows || A->nrows != C->nrows || B->ncols != C->ncols) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    if (MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->valid <= 0
        && coo_2_grb_d_i8(A) != 0) {
        return MATX_ERR_INTERNAL;
    }
    if (MATX_HANDLE(B, MATX_HANDLE_TYPE_GRB_MATRIX)->valid <= 0
        && dense_2_grb_d_i8(B) != 0) {
        return MATX_ERR_INTERNAL;
    }
    if (MATX_HANDLE(C, MATX_HANDLE_TYPE_GRB_MATRIX)->valid <= 0
        && dense_2_grb_d_i8(C) != 0) {
        if (MATX_HANDLE(B, MATX_HANDLE_TYPE_GRB_MATRIX)->valid > 0)
            grb_2_dense_d_i8(B);
        return MATX_ERR_INTERNAL;
    }

    /* C = alpha*A*B + beta*C */

    // 1. temp = alpha * A * B
    GrB_Matrix temp;
    GrB_Info info = GrB_Matrix_new(&temp, GrB_FP64, C->nrows, C->ncols);
    if (info != GrB_SUCCESS) {
        grb_2_dense_d_i8(B);
        grb_2_dense_d_i8(C);
        return MATX_ERR_INTERNAL;
    }
    info = GrB_mxm(temp,
                   NULL,
                   NULL,
                   GxB_PLUS_TIMES_FP64,
                   (GrB_Matrix) MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl,
                   (GrB_Matrix) MATX_HANDLE(B, MATX_HANDLE_TYPE_GRB_MATRIX)->impl,
                   NULL);
    if (info != GrB_SUCCESS) {
        GrB_Matrix_free(&temp);
        MATX_ERROR("GrB_mxm alpha * A * B error: %d", info);
        grb_2_dense_d_i8(B);
        grb_2_dense_d_i8(C);
        return MATX_ERR_INTERNAL;
    }
    info = GrB_apply(temp, NULL, NULL, GrB_TIMES_FP64, temp, alpha, NULL);
    if (info != GrB_SUCCESS) {
        GrB_Matrix_free(&temp);
        MATX_ERROR("GrB_apply  error: %d", info);
        grb_2_dense_d_i8(B);
        grb_2_dense_d_i8(C);
        return MATX_ERR_INTERNAL;
    }
    //2. gC = beta * gC
    info = GrB_apply((GrB_Matrix) MATX_HANDLE(C, MATX_HANDLE_TYPE_GRB_MATRIX)->impl,
                     NULL,
                     NULL,
                     GrB_TIMES_FP64,
                     (GrB_Matrix) MATX_HANDLE(C, MATX_HANDLE_TYPE_GRB_MATRIX)->impl,
                     beta,
                     NULL);
    if (info != GrB_SUCCESS) {
        GrB_Matrix_free(&temp);
        MATX_ERROR("GrB_apply beta * gC error: %d", info);
        grb_2_dense_d_i8(B);
        grb_2_dense_d_i8(C);
        return MATX_ERR_INTERNAL;
    }
    //3. gC = temp + gC
    info = GrB_eWiseAdd((GrB_Matrix) MATX_HANDLE(C, MATX_HANDLE_TYPE_GRB_MATRIX)->impl,
                        NULL,
                        NULL,
                        GrB_PLUS_FP64,
                        temp,
                        (GrB_Matrix) MATX_HANDLE(C, MATX_HANDLE_TYPE_GRB_MATRIX)->impl,
                        NULL);
    if (info != GrB_SUCCESS) {
        GrB_Matrix_free(&temp);
        MATX_ERROR("GrB_eWiseAdd temp + gC error: %d", info);
        grb_2_dense_d_i8(B);
        grb_2_dense_d_i8(C);
        return MATX_ERR_INTERNAL;
    }
    GrB_Matrix_free(&temp);
    const size_t b_export_status = grb_2_dense_d_i8(B);
    const size_t c_export_status = grb_2_dense_d_i8(C);
    if (b_export_status != 0 || c_export_status != 0)
        return MATX_ERR_INTERNAL;
#else
    (void) alpha; (void) A; (void) B; (void) beta; (void) C;
    MATX_ERROR("%s: GraphBLAS not available", __func__);
    return MATX_ERR_NOT_SUPPORTED;
#endif
    return MATX_OK;
}

static matx_status_t ref_dsp2md_d_i8_grb(
    matx_double alpha, matx_coo_d_i8_t A, matx_coo_d_i8_t B, matx_double beta, matx_dense_d_i8_t C)
{
#ifdef MATX_ENABLE_GRAPHBLAS
    if (!A || !B || !C) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }

    /* build A */
    if (MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->valid <= 0) {
        coo_2_grb_d_i8(A);
    }
    if (MATX_HANDLE(B, MATX_HANDLE_TYPE_GRB_MATRIX)->valid <= 0) {
        coo_2_grb_d_i8(B);
    }
    if (MATX_HANDLE(C, MATX_HANDLE_TYPE_GRB_MATRIX)->valid <= 0) {
        dense_2_grb_d_i8(C);
    }

    /* C = alpha*A*B + beta*C */

    GrB_Matrix c_grb = (GrB_Matrix) MATX_HANDLE(C, MATX_HANDLE_TYPE_GRB_MATRIX)->impl;
    GrB_Matrix a_grb = (GrB_Matrix) MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl;
    GrB_Matrix b_grb = (GrB_Matrix) MATX_HANDLE(B, MATX_HANDLE_TYPE_GRB_MATRIX)->impl;

    if (beta == 0.0) {
        /* C = alpha * A * B — output directly into C, no temp needed. */
        GrB_Descriptor desc;
        GrB_Descriptor_new(&desc);
        GrB_Descriptor_set(desc, GrB_OUTP, GrB_REPLACE);
        GrB_Info info = GrB_mxm(c_grb, NULL, NULL, GxB_PLUS_TIMES_FP64, a_grb, b_grb, desc);
        GrB_Descriptor_free(&desc);
        if (info != GrB_SUCCESS) {
            MATX_ERROR("GrB_mxm error: %d", info);
            return MATX_ERR_INTERNAL;
        }
        if (alpha != 1.0) {
            info = GrB_apply(c_grb, NULL, NULL, GrB_TIMES_FP64, c_grb, alpha, NULL);
            if (info != GrB_SUCCESS) {
                MATX_ERROR("GrB_apply scale error: %d", info);
                return MATX_ERR_INTERNAL;
            }
        }
    } else {
        GrB_Matrix temp;
        GrB_Info info = GrB_Matrix_new(&temp, GrB_FP64, C->nrows, C->ncols);
        if (info != GrB_SUCCESS) {
            MATX_ERROR("GrB_Matrix_new error: %d", info);
            return MATX_ERR_INTERNAL;
        }
        info = GrB_mxm(temp, NULL, NULL, GxB_PLUS_TIMES_FP64, a_grb, b_grb, NULL);
        if (info != GrB_SUCCESS) {
            MATX_ERROR("GrB_mxm alpha * A * B error: %d", info);
            return MATX_ERR_INTERNAL;
        }
        info = GrB_apply(temp, NULL, NULL, GrB_TIMES_FP64, temp, alpha, NULL);
        if (info != GrB_SUCCESS) {
            MATX_ERROR("GrB_apply alpha * A * B error: %d", info);
            return MATX_ERR_INTERNAL;
        }
        info = GrB_apply(c_grb, NULL, NULL, GrB_TIMES_FP64, c_grb, beta, NULL);
        if (info != GrB_SUCCESS) {
            MATX_ERROR("GrB_apply beta * gC error: %d", info);
            return MATX_ERR_INTERNAL;
        }
        info = GrB_eWiseAdd(c_grb, NULL, NULL, GrB_PLUS_FP64, temp, c_grb, NULL);
        if (info != GrB_SUCCESS) {
            MATX_ERROR("GrB_eWiseAdd temp + gC error: %d", info);
            GrB_Matrix_free(&temp);
            return MATX_ERR_INTERNAL;
        }
        GrB_Matrix_free(&temp);
    }
    grb_2_dense_d_i8(C);
#else
    (void) alpha; (void) A; (void) B; (void) beta; (void) C;
    MATX_ERROR("%s: GraphBLAS not available", __func__);
    return MATX_ERR_NOT_SUPPORTED;
#endif
    return MATX_OK;
}

static matx_status_t ref_zsp2md_z_i8_grb(matx_complex_d_t alpha,
                                         matx_coo_z_i8_t A,
                                         matx_coo_z_i8_t B,
                                         matx_complex_d_t beta,
                                         matx_dense_z_i8_t C)
{
#ifdef MATX_ENABLE_GRAPHBLAS
    if (!A || !B || !C) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }

    /* build A */
    if (MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->valid <= 0) {
        coo_2_grb_z_i8(A);
    }
    if (MATX_HANDLE(B, MATX_HANDLE_TYPE_GRB_MATRIX)->valid <= 0) {
        coo_2_grb_z_i8(B);
    }
    if (MATX_HANDLE(C, MATX_HANDLE_TYPE_GRB_MATRIX)->valid <= 0) {
        dense_2_grb_z_i8(C);
    }

    /* C = alpha*A*B + beta*C */
    GrB_Matrix c_grb = (GrB_Matrix) MATX_HANDLE(C, MATX_HANDLE_TYPE_GRB_MATRIX)->impl;
    GrB_Matrix a_grb = (GrB_Matrix) MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl;
    GrB_Matrix b_grb = (GrB_Matrix) MATX_HANDLE(B, MATX_HANDLE_TYPE_GRB_MATRIX)->impl;
    GxB_FC64_t a = GxB_CMPLX(alpha.real, alpha.imag);
    GxB_FC64_t b = GxB_CMPLX(beta.real, beta.imag);

    if (beta.real == 0.0 && beta.imag == 0.0) {
        /* C = alpha * A * B — output directly into C, no temp needed. */
        GrB_Descriptor desc;
        GrB_Descriptor_new(&desc);
        GrB_Descriptor_set(desc, GrB_OUTP, GrB_REPLACE);
        GrB_Info info = GrB_mxm(c_grb, NULL, NULL, GxB_PLUS_TIMES_FC64, a_grb, b_grb, desc);
        GrB_Descriptor_free(&desc);
        if (info != GrB_SUCCESS) {
            MATX_ERROR("GrB_mxm error: %d", info);
            return MATX_ERR_INTERNAL;
        }
        if (alpha.real != 1.0 || alpha.imag != 0.0) {
            info = GrB_apply(c_grb, NULL, NULL, GxB_TIMES_FC64, c_grb, a, NULL);
            if (info != GrB_SUCCESS) {
                MATX_ERROR("GrB_apply scale error: %d", info);
                return MATX_ERR_INTERNAL;
            }
        }
    } else {
        GrB_Matrix temp;
        GrB_Info info = GrB_Matrix_new(&temp, GxB_FC64, C->nrows, C->ncols);
        if (info != GrB_SUCCESS) {
            MATX_ERROR("GrB_Matrix_new error: %d", info);
            return MATX_ERR_INTERNAL;
        }
        info = GrB_mxm(temp, NULL, NULL, GxB_PLUS_TIMES_FC64, a_grb, b_grb, NULL);
        if (info != GrB_SUCCESS) {
            MATX_ERROR("GrB_mxm alpha * A * B error: %d", info);
            return MATX_ERR_INTERNAL;
        }
        info = GrB_apply(temp, NULL, NULL, GxB_TIMES_FC64, temp, a, NULL);
        if (info != GrB_SUCCESS) {
            MATX_ERROR("GrB_apply error: %d", info);
            return MATX_ERR_INTERNAL;
        }
        info = GrB_apply(c_grb, NULL, NULL, GxB_TIMES_FC64, c_grb, b, NULL);
        if (info != GrB_SUCCESS) {
            MATX_ERROR("GrB_apply beta * gC error: %d", info);
            return MATX_ERR_INTERNAL;
        }
        info = GrB_eWiseAdd(c_grb, NULL, NULL, GxB_PLUS_FC64, temp, c_grb, NULL);
        if (info != GrB_SUCCESS) {
            MATX_ERROR("GrB_eWiseAdd temp + gC error: %d", info);
            GrB_Matrix_free(&temp);
            return MATX_ERR_INTERNAL;
        }
        GrB_Matrix_free(&temp);
    }
    grb_2_dense_z_i8(C);
#else
    (void) alpha; (void) A; (void) B; (void) beta; (void) C;
    MATX_ERROR("%s: GraphBLAS not available", __func__);
    return MATX_ERR_NOT_SUPPORTED;
#endif
    return MATX_OK;
}

static matx_status_t ref_transpose_d_i8_grb(matx_coo_d_i8_t A, matx_coo_d_i8_t out)
{
#ifdef MATX_ENABLE_GRAPHBLAS
    if (!A || !out || !A->rows || !A->columns || !A->values
        || !out->rows || !out->columns || !out->values) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }

    /* Direct COO transpose: swap row/col arrays, no GrB round-trips needed. */
    out->nrows = A->ncols;
    out->ncols = A->nrows;
    out->nnz = A->nnz;
    memcpy(out->rows, A->columns, (size_t) A->nnz * sizeof(matx_int64_t));
    memcpy(out->columns, A->rows, (size_t) A->nnz * sizeof(matx_int64_t));
    memcpy(out->values, A->values, (size_t) A->nnz * sizeof(matx_double));

    /* Invalidate cached GrB handle since COO data changed. */
    {
        matx_handle_t* h = MATX_HANDLE(out, MATX_HANDLE_TYPE_GRB_MATRIX);
        if (h) h->valid = 0;
    }
#else
    (void) A; (void) out;
    MATX_ERROR("%s: GraphBLAS not available", __func__);
    return MATX_ERR_NOT_SUPPORTED;
#endif
    return MATX_OK;
}

static matx_status_t ref_transpose_z_i8_grb(matx_coo_z_i8_t A, matx_coo_z_i8_t out)
{
#ifdef MATX_ENABLE_GRAPHBLAS
    if (!A || !out || !A->rows || !A->columns || !A->values
        || !out->rows || !out->columns || !out->values) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }

    /* Direct COO transpose. */
    out->nrows = A->ncols;
    out->ncols = A->nrows;
    out->nnz = A->nnz;
    memcpy(out->rows, A->columns, (size_t) A->nnz * sizeof(matx_int64_t));
    memcpy(out->columns, A->rows, (size_t) A->nnz * sizeof(matx_int64_t));
    memcpy(out->values, A->values, (size_t) A->nnz * sizeof(matx_complex_d_t));

    {
        matx_handle_t* h = MATX_HANDLE(out, MATX_HANDLE_TYPE_GRB_MATRIX);
        if (h) h->valid = 0;
    }
#else
    (void) A; (void) out;
    MATX_ERROR("%s: GraphBLAS not available", __func__);
    return MATX_ERR_NOT_SUPPORTED;
#endif
    return MATX_OK;
}

static matx_status_t ref_conj_trans_z_i8_grb(matx_coo_z_i8_t A, matx_coo_z_i8_t out)
{
#ifdef MATX_ENABLE_GRAPHBLAS
    if (!A || !out || !A->rows || !A->columns || !A->values
        || !out->rows || !out->columns || !out->values) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }

    /* Direct COO conjugate-transpose: swap rows/cols, negate imag parts. */
    out->nrows = A->ncols;
    out->ncols = A->nrows;
    out->nnz = A->nnz;
    memcpy(out->rows, A->columns, (size_t) A->nnz * sizeof(matx_int64_t));
    memcpy(out->columns, A->rows, (size_t) A->nnz * sizeof(matx_int64_t));
    for (matx_int64_t i = 0; i < A->nnz; ++i) {
        out->values[i].real = A->values[i].real;
        out->values[i].imag = -A->values[i].imag;
    }

    {
        matx_handle_t* h = MATX_HANDLE(out, MATX_HANDLE_TYPE_GRB_MATRIX);
        if (h) h->valid = 0;
    }
#else
    (void) A; (void) out;
    MATX_ERROR("%s: GraphBLAS not available", __func__);
    return MATX_ERR_NOT_SUPPORTED;
#endif
    return MATX_OK;
}

static matx_status_t ref_finalize_grb()
{
#ifdef MATX_ENABLE_GRAPHBLAS
    GrB_finalize();
#else
    MATX_ERROR("%s: GraphBLAS not available", __func__);
    return MATX_ERR_NOT_SUPPORTED;
#endif
    return MATX_OK;
}

// ---- Sparse matrix norms ----

static matx_status_t ref_norm1_mat_grb(matx_coo_d_i8_t A, matx_double* out)
{
    if (!A || !out) return MATX_ERR_INVALID_ARG;
    if (A->nnz <= MATX_SPARSE_DIRECT_NORM1_REAL_MAX_NNZ
        && A->nrows <= MATX_SPARSE_DIRECT_NORM_MAX_DIMENSION
        && A->ncols <= MATX_SPARSE_DIRECT_NORM_MAX_DIMENSION) {
        const matx_status_t status = ref_sparse_norm_coo(&A->alloc,
                                                          A->nrows,
                                                          A->ncols,
                                                          A->nnz,
                                                          A->rows,
                                                          A->columns,
                                                          A->values,
                                                          NULL,
                                                          MATX_SPARSE_NORM_ONE,
                                                          out);
        if (status != MATX_ERR_NOT_SUPPORTED) return status;
    }
#ifdef MATX_ENABLE_GRAPHBLAS
    matx_handle_t* h = MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX);
    if (h->valid <= 0) coo_2_grb_d_i8(A);

    GrB_Matrix src = (GrB_Matrix) h->impl;
    GrB_Matrix tmp;
    GrB_Matrix_new(&tmp, GrB_FP64, A->nrows, A->ncols);
    GrB_apply(tmp, NULL, NULL, GrB_ABS_FP64, src, NULL);
    GrB_Vector col_sums;
    GrB_Vector_new(&col_sums, GrB_FP64, A->ncols);
    GrB_reduce(col_sums, NULL, NULL, GrB_PLUS_MONOID_FP64, tmp, GrB_DESC_T0);
    GrB_Matrix_free(&tmp);
    GrB_reduce(out, NULL, GrB_MAX_MONOID_FP64, col_sums, NULL);
    GrB_Vector_free(&col_sums);
#else
    (void) A; (void) out;
    MATX_ERROR("%s: GraphBLAS not available", __func__);
    return MATX_ERR_NOT_SUPPORTED;
#endif
    return MATX_OK;
}

static matx_status_t ref_norminf_mat_grb(matx_coo_d_i8_t A, matx_double* out)
{
    if (!A || !out) return MATX_ERR_INVALID_ARG;
    if (A->nnz <= MATX_SPARSE_DIRECT_NORM_MAX_NNZ
        && A->nrows <= MATX_SPARSE_DIRECT_NORM_MAX_DIMENSION
        && A->ncols <= MATX_SPARSE_DIRECT_NORM_MAX_DIMENSION) {
        const matx_status_t status = ref_sparse_norm_coo(&A->alloc,
                                                          A->nrows,
                                                          A->ncols,
                                                          A->nnz,
                                                          A->rows,
                                                          A->columns,
                                                          A->values,
                                                          NULL,
                                                          MATX_SPARSE_NORM_INFINITY,
                                                          out);
        if (status != MATX_ERR_NOT_SUPPORTED) return status;
    }
#ifdef MATX_ENABLE_GRAPHBLAS
    matx_handle_t* h = MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX);
    if (h->valid <= 0) coo_2_grb_d_i8(A);

    GrB_Matrix src = (GrB_Matrix) h->impl;
    GrB_Matrix tmp;
    GrB_Matrix_new(&tmp, GrB_FP64, A->nrows, A->ncols);
    GrB_apply(tmp, NULL, NULL, GrB_ABS_FP64, src, NULL);
    GrB_Vector row_sums;
    GrB_Vector_new(&row_sums, GrB_FP64, A->nrows);
    GrB_reduce(row_sums, NULL, NULL, GrB_PLUS_MONOID_FP64, tmp, NULL);
    GrB_Matrix_free(&tmp);
    GrB_reduce(out, NULL, GrB_MAX_MONOID_FP64, row_sums, NULL);
    GrB_Vector_free(&row_sums);
#else
    (void) A; (void) out;
    MATX_ERROR("%s: GraphBLAS not available", __func__);
    return MATX_ERR_NOT_SUPPORTED;
#endif
    return MATX_OK;
}

static matx_status_t ref_normfro_mat_grb(matx_coo_d_i8_t A, matx_double* out)
{
    if (!A || !out) return MATX_ERR_INVALID_ARG;
    if (A->nnz <= MATX_SPARSE_DIRECT_NORM_MAX_NNZ
        && A->nrows <= MATX_SPARSE_DIRECT_NORM_MAX_DIMENSION
        && A->ncols <= MATX_SPARSE_DIRECT_NORM_MAX_DIMENSION) {
        const matx_status_t status = ref_sparse_norm_coo(&A->alloc,
                                                          A->nrows,
                                                          A->ncols,
                                                          A->nnz,
                                                          A->rows,
                                                          A->columns,
                                                          A->values,
                                                          NULL,
                                                          MATX_SPARSE_NORM_FROBENIUS,
                                                          out);
        if (status != MATX_ERR_NOT_SUPPORTED) return status;
    }
#ifdef MATX_ENABLE_GRAPHBLAS
    matx_handle_t* h = MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX);
    if (h->valid <= 0) coo_2_grb_d_i8(A);

    GrB_Matrix src = (GrB_Matrix) h->impl;
    GrB_Matrix tmp;
    GrB_Matrix_new(&tmp, GrB_FP64, A->nrows, A->ncols);
    GrB_eWiseMult(tmp, NULL, NULL, GrB_TIMES_FP64, src, src, NULL);
    double sumsq = 0.0;
    GrB_reduce(&sumsq, NULL, GrB_PLUS_MONOID_FP64, tmp, NULL);
    GrB_Matrix_free(&tmp);
    *out = sqrt(sumsq);
#else
    (void) A; (void) out;
    MATX_ERROR("%s: GraphBLAS not available", __func__);
    return MATX_ERR_NOT_SUPPORTED;
#endif
    return MATX_OK;
}

// ---- Sparse matrix norms (c64) ----

static matx_status_t ref_norm1_mat_z_i8_grb(matx_coo_z_i8_t A, matx_double* out)
{
    if (!A || !out) return MATX_ERR_INVALID_ARG;
    if (A->nnz <= MATX_SPARSE_DIRECT_NORM_MAX_NNZ
        && A->nrows <= MATX_SPARSE_DIRECT_NORM_MAX_DIMENSION
        && A->ncols <= MATX_SPARSE_DIRECT_NORM_MAX_DIMENSION) {
        const matx_status_t status = ref_sparse_norm_coo(&A->alloc,
                                                          A->nrows,
                                                          A->ncols,
                                                          A->nnz,
                                                          A->rows,
                                                          A->columns,
                                                          NULL,
                                                          A->values,
                                                          MATX_SPARSE_NORM_ONE,
                                                          out);
        if (status != MATX_ERR_NOT_SUPPORTED) return status;
    }
#ifdef MATX_ENABLE_GRAPHBLAS
    if (MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->valid <= 0)
        coo_2_grb_z_i8(A);

    GrB_Matrix abs_mat;
    GrB_Matrix_new(&abs_mat, GrB_FP64, A->nrows, A->ncols);
    GrB_apply(abs_mat, NULL, NULL, GxB_ABS_FC64, (GrB_Matrix) MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl, NULL);
    GrB_Vector col_sums;
    GrB_Vector_new(&col_sums, GrB_FP64, A->ncols);
    GrB_reduce(col_sums, NULL, NULL, GrB_PLUS_MONOID_FP64, abs_mat, GrB_DESC_T0);
    GrB_Matrix_free(&abs_mat);
    GrB_reduce(out, NULL, GrB_MAX_MONOID_FP64, col_sums, NULL);
    GrB_Vector_free(&col_sums);
#else
    (void) A; (void) out;
    MATX_ERROR("%s: GraphBLAS not available", __func__);
    return MATX_ERR_NOT_SUPPORTED;
#endif
    return MATX_OK;
}

static matx_status_t ref_norminf_mat_z_i8_grb(matx_coo_z_i8_t A, matx_double* out)
{
    if (!A || !out) return MATX_ERR_INVALID_ARG;
    if (A->nnz <= MATX_SPARSE_DIRECT_NORM_MAX_NNZ
        && A->nrows <= MATX_SPARSE_DIRECT_NORM_MAX_DIMENSION
        && A->ncols <= MATX_SPARSE_DIRECT_NORM_MAX_DIMENSION) {
        const matx_status_t status = ref_sparse_norm_coo(&A->alloc,
                                                          A->nrows,
                                                          A->ncols,
                                                          A->nnz,
                                                          A->rows,
                                                          A->columns,
                                                          NULL,
                                                          A->values,
                                                          MATX_SPARSE_NORM_INFINITY,
                                                          out);
        if (status != MATX_ERR_NOT_SUPPORTED) return status;
    }
#ifdef MATX_ENABLE_GRAPHBLAS
    if (MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->valid <= 0)
        coo_2_grb_z_i8(A);

    GrB_Matrix abs_mat;
    GrB_Matrix_new(&abs_mat, GrB_FP64, A->nrows, A->ncols);
    GrB_apply(abs_mat, NULL, NULL, GxB_ABS_FC64, (GrB_Matrix) MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl, NULL);
    GrB_Vector row_sums;
    GrB_Vector_new(&row_sums, GrB_FP64, A->nrows);
    GrB_reduce(row_sums, NULL, NULL, GrB_PLUS_MONOID_FP64, abs_mat, NULL);
    GrB_Matrix_free(&abs_mat);
    GrB_reduce(out, NULL, GrB_MAX_MONOID_FP64, row_sums, NULL);
    GrB_Vector_free(&row_sums);
#else
    (void) A; (void) out;
    MATX_ERROR("%s: GraphBLAS not available", __func__);
    return MATX_ERR_NOT_SUPPORTED;
#endif
    return MATX_OK;
}

static matx_status_t ref_normfro_mat_z_i8_grb(matx_coo_z_i8_t A, matx_double* out)
{
    if (!A || !out) return MATX_ERR_INVALID_ARG;
    if (A->nnz <= MATX_SPARSE_DIRECT_NORM_MAX_NNZ
        && A->nrows <= MATX_SPARSE_DIRECT_NORM_MAX_DIMENSION
        && A->ncols <= MATX_SPARSE_DIRECT_NORM_MAX_DIMENSION) {
        const matx_status_t status = ref_sparse_norm_coo(&A->alloc,
                                                          A->nrows,
                                                          A->ncols,
                                                          A->nnz,
                                                          A->rows,
                                                          A->columns,
                                                          NULL,
                                                          A->values,
                                                          MATX_SPARSE_NORM_FROBENIUS,
                                                          out);
        if (status != MATX_ERR_NOT_SUPPORTED) return status;
    }
#ifdef MATX_ENABLE_GRAPHBLAS
    if (MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->valid <= 0)
        coo_2_grb_z_i8(A);

    GrB_Matrix abs_mat;
    GrB_Matrix_new(&abs_mat, GrB_FP64, A->nrows, A->ncols);
    GrB_apply(abs_mat, NULL, NULL, GxB_ABS_FC64, (GrB_Matrix) MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl, NULL);
    GrB_eWiseMult(abs_mat, NULL, NULL, GrB_TIMES_FP64, abs_mat, abs_mat, NULL);
    double sumsq = 0.0;
    GrB_reduce(&sumsq, NULL, GrB_PLUS_MONOID_FP64, abs_mat, NULL);
    GrB_Matrix_free(&abs_mat);
    *out = sqrt(sumsq);
#else
    (void) A; (void) out;
    MATX_ERROR("%s: GraphBLAS not available", __func__);
    return MATX_ERR_NOT_SUPPORTED;
#endif
    return MATX_OK;
}

// ---- Sparse-sparse addition ----

static matx_status_t ref_spadd_d_i8_grb(
    matx_double alpha, matx_coo_d_i8_t A, matx_double beta, matx_coo_d_i8_t B, matx_coo_d_i8_t out)
{
#ifdef MATX_ENABLE_GRAPHBLAS
    if (!A || !B || !out) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    if (A->nrows != B->nrows || A->ncols != B->ncols) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    if (MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->valid <= 0)
        coo_2_grb_d_i8(A);
    if (MATX_HANDLE(B, MATX_HANDLE_TYPE_GRB_MATRIX)->valid <= 0)
        coo_2_grb_d_i8(B);

    GrB_Matrix temp_a = NULL, temp_b = NULL;
    GrB_Matrix a_op, b_op;

    if (alpha == 1.0) {
        a_op = (GrB_Matrix) MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl;
    } else {
        GrB_Matrix_dup(&temp_a, (GrB_Matrix) MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl);
        GrB_apply(temp_a, NULL, NULL, GrB_TIMES_FP64, temp_a, alpha, NULL);
        a_op = temp_a;
    }

    if (beta == 1.0) {
        b_op = (GrB_Matrix) MATX_HANDLE(B, MATX_HANDLE_TYPE_GRB_MATRIX)->impl;
    } else {
        GrB_Matrix_dup(&temp_b, (GrB_Matrix) MATX_HANDLE(B, MATX_HANDLE_TYPE_GRB_MATRIX)->impl);
        GrB_apply(temp_b, NULL, NULL, GrB_TIMES_FP64, temp_b, beta, NULL);
        b_op = temp_b;
    }

    create_empty_grb_d_i8(out);
    GrB_Info info = GrB_eWiseAdd((GrB_Matrix) MATX_HANDLE(out, MATX_HANDLE_TYPE_GRB_MATRIX)->impl,
                                 NULL, NULL, GrB_PLUS_FP64, a_op, b_op, NULL);
    if (temp_a) GrB_Matrix_free(&temp_a);
    if (temp_b) GrB_Matrix_free(&temp_b);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("GrB_eWiseAdd spadd_d_i8 error: %d", info);
        return MATX_ERR_INTERNAL;
    }
    grb_2_coo_d_i8(out);
#else
    (void) alpha; (void) A; (void) beta; (void) B; (void) out;
    MATX_ERROR("%s: GraphBLAS not available", __func__);
    return MATX_ERR_NOT_SUPPORTED;
#endif
    return MATX_OK;
}

static matx_status_t ref_spadd_z_i8_grb(matx_complex_d_t alpha,
                                        matx_coo_z_i8_t A,
                                        matx_complex_d_t beta,
                                        matx_coo_z_i8_t B,
                                        matx_coo_z_i8_t out)
{
#ifdef MATX_ENABLE_GRAPHBLAS
    if (!A || !B || !out) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    if (A->nrows != B->nrows || A->ncols != B->ncols) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    if (MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->valid <= 0)
        coo_2_grb_z_i8(A);
    if (MATX_HANDLE(B, MATX_HANDLE_TYPE_GRB_MATRIX)->valid <= 0)
        coo_2_grb_z_i8(B);

    GxB_FC64_t a = GxB_CMPLX(alpha.real, alpha.imag);
    GxB_FC64_t b = GxB_CMPLX(beta.real, beta.imag);
    GrB_Matrix temp_a = NULL, temp_b = NULL;
    GrB_Matrix a_op, b_op;

    if (alpha.real == 1.0 && alpha.imag == 0.0) {
        a_op = (GrB_Matrix) MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl;
    } else {
        GrB_Matrix_dup(&temp_a, (GrB_Matrix) MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl);
        GrB_apply(temp_a, NULL, NULL, GxB_TIMES_FC64, temp_a, a, NULL);
        a_op = temp_a;
    }

    if (beta.real == 1.0 && beta.imag == 0.0) {
        b_op = (GrB_Matrix) MATX_HANDLE(B, MATX_HANDLE_TYPE_GRB_MATRIX)->impl;
    } else {
        GrB_Matrix_dup(&temp_b, (GrB_Matrix) MATX_HANDLE(B, MATX_HANDLE_TYPE_GRB_MATRIX)->impl);
        GrB_apply(temp_b, NULL, NULL, GxB_TIMES_FC64, temp_b, b, NULL);
        b_op = temp_b;
    }

    create_empty_grb_z_i8(out);
    GrB_Info info = GrB_eWiseAdd((GrB_Matrix) MATX_HANDLE(out, MATX_HANDLE_TYPE_GRB_MATRIX)->impl,
                                 NULL, NULL, GxB_PLUS_FC64, a_op, b_op, NULL);
    if (temp_a) GrB_Matrix_free(&temp_a);
    if (temp_b) GrB_Matrix_free(&temp_b);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("GrB_eWiseAdd spadd_z_i8 error: %d", info);
        return MATX_ERR_INTERNAL;
    }
    grb_2_coo_z_i8(out);
#else
    (void) alpha; (void) A; (void) beta; (void) B; (void) out;
    MATX_ERROR("%s: GraphBLAS not available", __func__);
    return MATX_ERR_NOT_SUPPORTED;
#endif
    return MATX_OK;
}

// ---- Non-zero count per row/column ----

static matx_status_t ref_spnnz_rows_d_i8_grb(matx_coo_d_i8_t A, matx_vec_d_i8_t out)
{
    if (!A || !out || !A->values) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    memset(out->data, 0, sizeof(matx_double) * (size_t) A->nrows);
    for (matx_int64_t i = 0; i < A->nnz; ++i)
        out->data[A->rows[i]] += 1.0;
    return MATX_OK;
}

static matx_status_t ref_spnnz_cols_d_i8_grb(matx_coo_d_i8_t A, matx_vec_d_i8_t out)
{
    if (!A || !out || !A->values) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    memset(out->data, 0, sizeof(matx_double) * (size_t) A->ncols);
    for (matx_int64_t i = 0; i < A->nnz; ++i)
        out->data[A->columns[i]] += 1.0;
    return MATX_OK;
}

static matx_status_t ref_spnnz_rows_z_i8_grb(matx_coo_z_i8_t A, matx_vec_z_i8_t out)
{
    if (!A || !out || !A->values) {
        MATX_ERROR("invalid argument");
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

static matx_status_t ref_spnnz_cols_z_i8_grb(matx_coo_z_i8_t A, matx_vec_z_i8_t out)
{
    if (!A || !out || !A->values) {
        MATX_ERROR("invalid argument");
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

static matx_status_t ref_sprowsums_d_i8_grb(matx_coo_d_i8_t A, matx_vec_d_i8_t out)
{
    if (!A || !out || !A->values) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    memset(out->data, 0, sizeof(matx_double) * (size_t) A->nrows);
    for (matx_int64_t i = 0; i < A->nnz; ++i)
        out->data[A->rows[i]] += fabs(A->values[i]);
    return MATX_OK;
}

static matx_status_t ref_spcolsums_d_i8_grb(matx_coo_d_i8_t A, matx_vec_d_i8_t out)
{
    if (!A || !out || !A->values) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    memset(out->data, 0, sizeof(matx_double) * (size_t) A->ncols);
    for (matx_int64_t i = 0; i < A->nnz; ++i)
        out->data[A->columns[i]] += fabs(A->values[i]);
    return MATX_OK;
}

static matx_status_t ref_sprowsums_z_i8_grb(matx_coo_z_i8_t A, matx_vec_z_i8_t out)
{
    if (!A || !out || !A->values) {
        MATX_ERROR("invalid argument");
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

static matx_status_t ref_spcolsums_z_i8_grb(matx_coo_z_i8_t A, matx_vec_z_i8_t out)
{
    if (!A || !out || !A->values) {
        MATX_ERROR("invalid argument");
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

static matx_status_t ref_spdiag_d_i8_grb(matx_coo_d_i8_t A, matx_int64_t offset, matx_vec_d_i8_t out)
{
    if (!A || !out) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    matx_int64_t diag_len = (offset >= 0)
                                ? ((A->ncols - offset < A->nrows) ? A->ncols - offset : A->nrows)
                                : ((A->nrows + offset < A->ncols) ? A->nrows + offset : A->ncols);
    if (diag_len < 0)
        diag_len = 0;
    if (out->n < diag_len) {
        MATX_ERROR("invalid argument");
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

static matx_status_t ref_spdiag_z_i8_grb(matx_coo_z_i8_t A, matx_int64_t offset, matx_vec_z_i8_t out)
{
    if (!A || !out) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    matx_int64_t diag_len = (offset >= 0)
                                ? ((A->ncols - offset < A->nrows) ? A->ncols - offset : A->nrows)
                                : ((A->nrows + offset < A->ncols) ? A->nrows + offset : A->ncols);
    if (diag_len < 0)
        diag_len = 0;
    if (out->n < diag_len) {
        MATX_ERROR("invalid argument");
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

static matx_status_t ref_scale_rows_d_i8_grb(matx_coo_d_i8_t A, const matx_vec_d_i8_t s)
{
    if (!A || !s || !A->values || !s->data) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    if (s->n < A->nrows) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    for (matx_int64_t i = 0; i < A->nnz; ++i)
        A->values[i] *= s->data[A->rows[i]];
    return MATX_OK;
}

static matx_status_t ref_scale_cols_d_i8_grb(matx_coo_d_i8_t A, const matx_vec_d_i8_t s)
{
    if (!A || !s || !A->values || !s->data) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    if (s->n < A->ncols) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    for (matx_int64_t i = 0; i < A->nnz; ++i)
        A->values[i] *= s->data[A->columns[i]];
    return MATX_OK;
}

static matx_status_t ref_scale_rows_z_i8_grb(matx_coo_z_i8_t A, const matx_vec_z_i8_t s)
{
    if (!A || !s || !A->values || !s->data) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    if (s->n < A->nrows) {
        MATX_ERROR("invalid argument");
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

static matx_status_t ref_scale_cols_z_i8_grb(matx_coo_z_i8_t A, const matx_vec_z_i8_t s)
{
    if (!A || !s || !A->values || !s->data) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    if (s->n < A->ncols) {
        MATX_ERROR("invalid argument");
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

static matx_bool grb_init_ok = false;

static void do_grb_init(void)
{
#ifdef MATX_ENABLE_GRAPHBLAS
    GrB_Info info = GrB_init(GrB_NONBLOCKING);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("GraphBLAS initialization failed with error code %d", info);
    } else {
        grb_init_ok = true;
    }
#endif
}
#ifdef _WIN32
#include <windows.h>
static INIT_ONCE grb_init_flag = INIT_ONCE_STATIC_INIT;
static BOOL CALLBACK do_grb_init_win(PINIT_ONCE InitOnce, PVOID Parameter, PVOID* Context)
{
    (void) InitOnce;
    (void) Parameter;
    (void) Context;
    do_grb_init();
    return TRUE;
}
#define matx_call_once(flag, func) InitOnceExecuteOnce(flag, do_grb_init_win, NULL, NULL)
#else
#include <threads.h>
static once_flag grb_init_flag = ONCE_FLAG_INIT;
#define matx_call_once(flag, func) call_once(flag, func)
#endif

matx_sparse_backend_t matx_sparse_make_reference_grb(void)
{
    matx_call_once(&grb_init_flag, do_grb_init);
    if (!grb_init_ok) {
        MATX_ERROR("GraphBLAS initialization failed");
    }
    matx_sparse_backend_t b = {.kind = MATX_SPARSE_BACKEND_GRAPHBLAS,
                               .vt = {
                               .spmm_z_i8 = ref_spmm_z_i8_grb,
                               .spmv_z_i8 = ref_spmv_z_i8_grb,
                               .spmm_d_i8 = ref_spmm_d_i8_grb,
                               .spmv_d_i8 = ref_spmv_d_i8_grb,
                               .dsp2md_d_i8 = ref_dsp2md_d_i8_grb,
                               .zsp2md_z_i8 = ref_zsp2md_z_i8_grb,
                               .transpose_d_i8 = ref_transpose_d_i8_grb,
                               .transpose_z_i8 = ref_transpose_z_i8_grb,
                               .conj_trans_z_i8 = ref_conj_trans_z_i8_grb,
                               .finalize = ref_finalize_grb,
                               .norm1_mat_d_i8 = ref_norm1_mat_grb,
                               .norminf_mat_d_i8 = ref_norminf_mat_grb,
                               .normfro_mat_d_i8 = ref_normfro_mat_grb,
                               .norm1_mat_z_i8 = ref_norm1_mat_z_i8_grb,
                               .norminf_mat_z_i8 = ref_norminf_mat_z_i8_grb,
                               .normfro_mat_z_i8 = ref_normfro_mat_z_i8_grb,
                               .spadd_d_i8 = ref_spadd_d_i8_grb,
                               .spadd_z_i8 = ref_spadd_z_i8_grb,
                               .spnnz_rows_d_i8 = ref_spnnz_rows_d_i8_grb,
                               .spnnz_cols_d_i8 = ref_spnnz_cols_d_i8_grb,
                               .spnnz_rows_z_i8 = ref_spnnz_rows_z_i8_grb,
                               .spnnz_cols_z_i8 = ref_spnnz_cols_z_i8_grb,
                               .sprowsums_d_i8 = ref_sprowsums_d_i8_grb,
                               .spcolsums_d_i8 = ref_spcolsums_d_i8_grb,
                               .sprowsums_z_i8 = ref_sprowsums_z_i8_grb,
                               .spcolsums_z_i8 = ref_spcolsums_z_i8_grb,
                               .spdiag_d_i8 = ref_spdiag_d_i8_grb,
                               .spdiag_z_i8 = ref_spdiag_z_i8_grb,
                               .scale_rows_d_i8 = ref_scale_rows_d_i8_grb,
                               .scale_cols_d_i8 = ref_scale_cols_d_i8_grb,
                               .scale_rows_z_i8 = ref_scale_rows_z_i8_grb,
                               .scale_cols_z_i8 = ref_scale_cols_z_i8_grb,
                               }};
    return b;
}
