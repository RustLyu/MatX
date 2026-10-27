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
#define MATX_SPARSE_SPMM_COO_K_TILE 256
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

    /* ── fast path: already sorted and duplicate-free? ──
     * Many internally-built COO matrices enter this function with their
     * entries column-major sorted and deduplicated.  Computing the norm
     * directly from the COO arrays avoids an O(nnz) hash-table build +
     * O(2*nnz) full-table scan. */
    int sorted_dedup = 1;
    for (matx_int64_t k = 1; k < nnz; ++k) {
        if (columns[k] < columns[k - 1]
            || (columns[k] == columns[k - 1]
                && rows[k] < rows[k - 1])
            || (columns[k] == columns[k - 1]
                && rows[k] == rows[k - 1])) {
            sorted_dedup = 0;
            break;
        }
    }
    if (sorted_dedup) {
        /* validate all indices once */
        for (matx_int64_t k = 0; k < nnz; ++k) {
            if (rows[k] < 0 || rows[k] >= nrows
                || columns[k] < 0 || columns[k] >= ncols) {
                return MATX_ERR_INVALID_ARG;
            }
            matx_double rv = complex_values ? complex_values[k].real : real_values[k];
            matx_double iv = complex_values ? complex_values[k].imag : 0.0;
            if (isnan(rv) || isnan(iv)) return MATX_ERR_NOT_SUPPORTED;
        }
        if (kind == MATX_SPARSE_NORM_FROBENIUS) {
            matx_double sum_sq = 0.0;
            #pragma omp simd reduction(+:sum_sq)
            for (matx_int64_t k = 0; k < nnz; ++k) {
                matx_double rv = complex_values ? complex_values[k].real : real_values[k];
                matx_double iv = complex_values ? complex_values[k].imag : 0.0;
                sum_sq += rv * rv + iv * iv;
            }
            *out = sqrt(sum_sq);
        } else {
            const matx_int64_t count = kind == MATX_SPARSE_NORM_ONE ? ncols : nrows;
            if ((uint64_t) count > SIZE_MAX / sizeof(matx_double))
                return MATX_ERR_INVALID_ARG;
            matx_double* sums = (matx_double*) matx_malloc(
                alloc, (size_t) count * sizeof(*sums));
            if (!sums) return MATX_ERR_OUT_OF_MEMORY;
            memset(sums, 0, (size_t) count * sizeof(*sums));
            for (matx_int64_t k = 0; k < nnz; ++k) {
                matx_double rv = complex_values ? complex_values[k].real : real_values[k];
                matx_double iv = complex_values ? complex_values[k].imag : 0.0;
                matx_double mag = complex_values ? hypot(rv, iv) : fabs(rv);
                const matx_int64_t idx
                    = (kind == MATX_SPARSE_NORM_ONE) ? columns[k] : rows[k];
                sums[idx] += mag;
            }
            matx_double maximum = 0.0;
            for (matx_int64_t i = 0; i < count; ++i) {
                if (sums[i] > maximum) maximum = sums[i];
            }
            *out = maximum;
            matx_free(alloc, sums);
        }
        return MATX_OK;
    }

    /* ── slow path: duplicates exist → hash-table coalesce ── */
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
    matx_double  x_stack[1024];
    const matx_double* x_data = x->data;
    matx_int64_t x_stride = x->stride;
    if (x == y) {
        if (x->n <= 1024) {
            for (matx_int64_t i = 0; i < x->n; ++i)
                x_stack[i] = x->data[i * x->stride];
            x_data = x_stack;
            x_stride = 1;
        } else {
            x_copy = (matx_double*) malloc((size_t) x->n * sizeof(*x_copy));
            if (!x_copy) return MATX_ERR_OUT_OF_MEMORY;
            for (matx_int64_t i = 0; i < x->n; ++i)
                x_copy[i] = x->data[i * x->stride];
            x_data = x_copy;
            x_stride = 1;
        }
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
    matx_complex_d_t  x_stack[1024];
    const matx_complex_d_t* x_data = x->data;
    matx_int64_t x_stride = x->stride;
    if (x == y) {
        if (x->n <= 1024) {
            for (matx_int64_t i = 0; i < x->n; ++i)
                x_stack[i] = x->data[i * x->stride];
            x_data = x_stack;
            x_stride = 1;
        } else {
            x_copy = (matx_complex_d_t*) malloc((size_t) x->n * sizeof(*x_copy));
            if (!x_copy) return MATX_ERR_OUT_OF_MEMORY;
            for (matx_int64_t i = 0; i < x->n; ++i)
                x_copy[i] = x->data[i * x->stride];
            x_data = x_copy;
            x_stride = 1;
        }
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
    if (alpha != 0.0 && b_layout == MATX_COL_MAJOR
        && C->layout == MATX_COL_MAJOR && B->ncols >= 4) {
        /* Reuse a small COO slice while finishing one RHS column at a time. */
        for (matx_int64_t k_begin = 0; k_begin < A->nnz;) {
            const matx_int64_t remaining = A->nnz - k_begin;
            const matx_int64_t k_end
                = k_begin + (remaining < MATX_SPARSE_SPMM_COO_K_TILE
                                 ? remaining
                                 : MATX_SPARSE_SPMM_COO_K_TILE);
            for (matx_int64_t col = 0; col < B->ncols; ++col) {
                const size_t b_col_offset = (size_t) col * (size_t) b_stride;
                const size_t c_col_offset = (size_t) col * (size_t) C->stride;
                for (matx_int64_t k = k_begin; k < k_end; ++k) {
                    const matx_int64_t row = A->rows[k];
                    const matx_int64_t inner = A->columns[k];
                    const matx_double product
                        = A->values[k] * b_data[(size_t) inner + b_col_offset];
                    const size_t c_index = (size_t) row + c_col_offset;
                    C->data[c_index] += alpha == 1.0 ? product : alpha * product;
                }
            }
            k_begin = k_end;
        }
    } else if (alpha != 0.0) {
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
    if ((alpha.real != 0.0 || alpha.imag != 0.0)
        && b_layout == MATX_COL_MAJOR && C->layout == MATX_COL_MAJOR
        && B->ncols >= 4) {
        /* The active RHS column is contiguous while this COO tile stays hot. */
        for (matx_int64_t k_begin = 0; k_begin < A->nnz;) {
            const matx_int64_t remaining = A->nnz - k_begin;
            const matx_int64_t k_end
                = k_begin + (remaining < MATX_SPARSE_SPMM_COO_K_TILE
                                 ? remaining
                                 : MATX_SPARSE_SPMM_COO_K_TILE);
            for (matx_int64_t col = 0; col < B->ncols; ++col) {
                const size_t b_col_offset = (size_t) col * (size_t) b_stride;
                const size_t c_col_offset = (size_t) col * (size_t) C->stride;
                for (matx_int64_t k = k_begin; k < k_end; ++k) {
                    const matx_int64_t row = A->rows[k];
                    const matx_int64_t inner = A->columns[k];
                    const matx_complex_d_t product
                        = complex_multiply(A->values[k],
                                           b_data[(size_t) inner + b_col_offset]);
                    const matx_complex_d_t scaled
                        = alpha.real == 1.0 && alpha.imag == 0.0
                              ? product
                              : complex_multiply(alpha, product);
                    matx_complex_d_t* output
                        = &C->data[(size_t) row + c_col_offset];
                    output->real += scaled.real;
                    output->imag += scaled.imag;
                }
            }
            k_begin = k_end;
        }
    } else if (alpha.real != 0.0 || alpha.imag != 0.0) {
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

    /* Early exit: alpha == 0 or nnz == 0 — just scale y by beta */
    if ((alpha.real == 0.0 && alpha.imag == 0.0) || A->nnz == 0) {
        if (beta.real == 0.0 && beta.imag == 0.0) {
            for (matx_int64_t i = 0; i < y->n; ++i)
                y->data[i * y->stride] = (matx_complex_d_t){0.0, 0.0};
        } else {
            for (matx_int64_t i = 0; i < y->n; ++i) {
                matx_complex_d_t v = y->data[i * y->stride];
                y->data[i * y->stride].real = beta.real * v.real - beta.imag * v.imag;
                y->data[i * y->stride].imag = beta.real * v.imag + beta.imag * v.real;
            }
        }
        return MATX_OK;
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
        return MATX_ERR_INTERNAL;
    }

    GxB_FC64_t a = GxB_CMPLX(alpha.real, alpha.imag);
    GxB_FC64_t b = GxB_CMPLX(beta.real, beta.imag);
    GrB_Vector y_grb = (GrB_Vector) MATX_HANDLE(y, MATX_HANDLE_TYPE_GRB_VECTOR)->impl;
    GrB_Vector x_grb = (GrB_Vector) MATX_HANDLE(x, MATX_HANDLE_TYPE_GRB_VECTOR)->impl;
    GrB_Matrix a_grb = (GrB_Matrix) MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl;

    if (beta.real == 0.0 && beta.imag == 0.0) {
        /* y = alpha * A * x — write directly into y, no temp needed */
        GrB_Descriptor desc;
        GrB_Descriptor_new(&desc);
        GrB_Descriptor_set(desc, GrB_OUTP, GrB_REPLACE);
        GrB_Info info = GrB_mxv(y_grb, NULL, NULL, GxB_PLUS_TIMES_FC64, a_grb, x_grb, desc);
        GrB_Descriptor_free(&desc);
        if (info != GrB_SUCCESS) {
            MATX_ERROR("GrB_mxv error: %d", info);
            grb_2_vec_z_i8(y);
            return MATX_ERR_INTERNAL;
        }
        if (alpha.real != 1.0 || alpha.imag != 0.0) {
            info = GrB_apply(y_grb, NULL, NULL, GxB_TIMES_FC64, y_grb, a, NULL);
            if (info != GrB_SUCCESS) {
                MATX_ERROR("GrB_apply scale error: %d", info);
                grb_2_vec_z_i8(y);
                return MATX_ERR_INTERNAL;
            }
        }
    } else {
        /* y = alpha * A * x + beta * y */
        GrB_Info info = GrB_apply(y_grb, NULL, NULL, GxB_TIMES_FC64, y_grb, b, NULL);
        if (info != GrB_SUCCESS) {
            MATX_ERROR("GrB_apply beta*gy error: %d", info);
            grb_2_vec_z_i8(y);
            return MATX_ERR_INTERNAL;
        }
        GrB_Vector temp;
        info = GrB_Vector_new(&temp, GxB_FC64, y->n);
        if (info != GrB_SUCCESS) {
            grb_2_vec_z_i8(y);
            return MATX_ERR_INTERNAL;
        }
        info = GrB_mxv(temp, NULL, NULL, GxB_PLUS_TIMES_FC64, a_grb, x_grb, NULL);
        if (info != GrB_SUCCESS) {
            GrB_Vector_free(&temp);
            MATX_ERROR("GrB_mxv error: %d", info);
            grb_2_vec_z_i8(y);
            return MATX_ERR_INTERNAL;
        }
        info = GrB_apply(temp, NULL, NULL, GxB_TIMES_FC64, temp, a, NULL);
        if (info != GrB_SUCCESS) {
            GrB_Vector_free(&temp);
            MATX_ERROR("GrB_apply alpha*temp error: %d", info);
            grb_2_vec_z_i8(y);
            return MATX_ERR_INTERNAL;
        }
        info = GrB_eWiseAdd(y_grb, NULL, NULL, GxB_PLUS_FC64, y_grb, temp, NULL);
        GrB_Vector_free(&temp);
        if (info != GrB_SUCCESS) {
            MATX_ERROR("GrB_eWiseAdd error: %d", info);
            grb_2_vec_z_i8(y);
            return MATX_ERR_INTERNAL;
        }
    }

    /* Export only modified output y (x is read-only) */
    if (grb_2_vec_z_i8(y) != 0)
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
        && A->nnz <= MATX_SPARSE_DIRECT_SPMV_MAX_NNZ
        && A->nnz <= MATX_SPARSE_DIRECT_SPMM_MAX_PRODUCTS / B->ncols) {
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

    /* Early exit: alpha == 0 or nnz == 0 — just scale C by beta */
    if ((alpha.real == 0.0 && alpha.imag == 0.0) || A->nnz == 0) {
        if (beta.real == 0.0 && beta.imag == 0.0) {
            memset(C->data, 0, (size_t) C->nrows * C->ncols * C->stride * sizeof(matx_complex_d_t));
        } else {
            const matx_int64_t minor = (C->layout == MATX_COL_MAJOR) ? C->nrows : C->ncols;
            const matx_int64_t major = (C->layout == MATX_COL_MAJOR) ? C->ncols : C->nrows;
            if (C->stride == minor) {
                const size_t count = (size_t) C->nrows * (size_t) C->ncols;
                for (size_t i = 0; i < count; ++i) {
                    matx_complex_d_t v = C->data[i];
                    C->data[i].real = beta.real * v.real - beta.imag * v.imag;
                    C->data[i].imag = beta.real * v.imag + beta.imag * v.real;
                }
            } else {
                for (matx_int64_t j = 0; j < major; ++j) {
                    for (matx_int64_t i = 0; i < minor; ++i) {
                        matx_complex_d_t* c = &C->data[i + j * C->stride];
                        matx_complex_d_t v = *c;
                        c->real = beta.real * v.real - beta.imag * v.imag;
                        c->imag = beta.real * v.imag + beta.imag * v.real;
                    }
                }
            }
        }
        return MATX_OK;
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
        return MATX_ERR_INTERNAL;
    }

    /* C = alpha*A*B + beta*C */
    GxB_FC64_t a = GxB_CMPLX(alpha.real, alpha.imag);
    GxB_FC64_t b = GxB_CMPLX(beta.real, beta.imag);
    GrB_Matrix c_grb = (GrB_Matrix) MATX_HANDLE(C, MATX_HANDLE_TYPE_GRB_MATRIX)->impl;
    GrB_Matrix b_grb = (GrB_Matrix) MATX_HANDLE(B, MATX_HANDLE_TYPE_GRB_MATRIX)->impl;
    GrB_Matrix a_grb = (GrB_Matrix) MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl;

    if (beta.real == 0.0 && beta.imag == 0.0) {
        /* C = alpha * A * B — write directly into C, no temp needed */
        GrB_Descriptor desc;
        GrB_Descriptor_new(&desc);
        GrB_Descriptor_set(desc, GrB_OUTP, GrB_REPLACE);
        GrB_Info info = GrB_mxm(c_grb, NULL, NULL, GxB_PLUS_TIMES_FC64, a_grb, b_grb, desc);
        GrB_Descriptor_free(&desc);
        if (info != GrB_SUCCESS) {
            MATX_ERROR("GrB_mxm error: %d", info);
            grb_2_dense_z_i8(C);
            return MATX_ERR_INTERNAL;
        }
        if (alpha.real != 1.0 || alpha.imag != 0.0) {
            info = GrB_apply(c_grb, NULL, NULL, GxB_TIMES_FC64, c_grb, a, NULL);
            if (info != GrB_SUCCESS) {
                MATX_ERROR("GrB_apply scale error: %d", info);
                grb_2_dense_z_i8(C);
                return MATX_ERR_INTERNAL;
            }
        }
    } else {
        /* C = alpha * A * B + beta * C */
        GrB_Matrix temp;
        GrB_Info info = GrB_Matrix_new(&temp, GxB_FC64, C->nrows, C->ncols);
        if (info != GrB_SUCCESS) {
            grb_2_dense_z_i8(C);
            return MATX_ERR_INTERNAL;
        }
        info = GrB_mxm(temp, NULL, NULL, GxB_PLUS_TIMES_FC64, a_grb, b_grb, NULL);
        if (info != GrB_SUCCESS) {
            GrB_Matrix_free(&temp);
            MATX_ERROR("GrB_mxm error: %d", info);
            grb_2_dense_z_i8(C);
            return MATX_ERR_INTERNAL;
        }
        info = GrB_apply(temp, NULL, NULL, GxB_TIMES_FC64, temp, a, NULL);
        if (info != GrB_SUCCESS) {
            GrB_Matrix_free(&temp);
            MATX_ERROR("GrB_apply error: %d", info);
            grb_2_dense_z_i8(C);
            return MATX_ERR_INTERNAL;
        }
        info = GrB_apply(c_grb, NULL, NULL, GxB_TIMES_FC64, c_grb, b, NULL);
        if (info != GrB_SUCCESS) {
            GrB_Matrix_free(&temp);
            MATX_ERROR("GrB_apply beta * gC error: %d", info);
            grb_2_dense_z_i8(C);
            return MATX_ERR_INTERNAL;
        }
        info = GrB_eWiseAdd(c_grb, NULL, NULL, GxB_PLUS_FC64, temp, c_grb, NULL);
        GrB_Matrix_free(&temp);
        if (info != GrB_SUCCESS) {
            MATX_ERROR("GrB_eWiseAdd temp + gC error: %d", info);
            grb_2_dense_z_i8(C);
            return MATX_ERR_INTERNAL;
        }
    }

    /* Export only modified output C (B is read-only) */
    if (grb_2_dense_z_i8(C) != 0)
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

    /* Early exit: alpha == 0 or nnz == 0 — just scale y by beta */
    if (alpha == 0.0 || A->nnz == 0) {
        if (beta == 0.0) {
            for (matx_int64_t i = 0; i < y->n; ++i)
                y->data[i * y->stride] = 0.0;
        } else {
            for (matx_int64_t i = 0; i < y->n; ++i)
                y->data[i * y->stride] *= beta;
        }
        return MATX_OK;
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
        return MATX_ERR_INTERNAL;
    }

    GrB_Vector y_grb = (GrB_Vector) MATX_HANDLE(y, MATX_HANDLE_TYPE_GRB_VECTOR)->impl;
    GrB_Vector x_grb = (GrB_Vector) MATX_HANDLE(x, MATX_HANDLE_TYPE_GRB_VECTOR)->impl;
    GrB_Matrix a_grb = (GrB_Matrix) MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl;

    if (beta == 0.0) {
        /* y = alpha * A * x — write directly into y, no temp needed */
        GrB_Descriptor desc;
        GrB_Descriptor_new(&desc);
        GrB_Descriptor_set(desc, GrB_OUTP, GrB_REPLACE);
        GrB_Info info = GrB_mxv(y_grb, NULL, NULL, GxB_PLUS_TIMES_FP64, a_grb, x_grb, desc);
        GrB_Descriptor_free(&desc);
        if (info != GrB_SUCCESS) {
            MATX_ERROR("GrB_mxv error: %d", info);
            grb_2_vec_d_i8(y);
            return MATX_ERR_INTERNAL;
        }
        if (alpha != 1.0) {
            info = GrB_apply(y_grb, NULL, NULL, GrB_TIMES_FP64, y_grb, alpha, NULL);
            if (info != GrB_SUCCESS) {
                MATX_ERROR("GrB_apply scale error: %d", info);
                grb_2_vec_d_i8(y);
                return MATX_ERR_INTERNAL;
            }
        }
    } else {
        /* y = alpha * A * x + beta * y */
        GrB_Info info = GrB_apply(y_grb, NULL, NULL, GrB_TIMES_FP64, y_grb, beta, NULL);
        if (info != GrB_SUCCESS) {
            MATX_ERROR("GrB_apply beta*gy error: %d", info);
            grb_2_vec_d_i8(y);
            return MATX_ERR_INTERNAL;
        }
        GrB_Vector temp;
        info = GrB_Vector_new(&temp, GrB_FP64, y->n);
        if (info != GrB_SUCCESS) {
            grb_2_vec_d_i8(y);
            return MATX_ERR_INTERNAL;
        }
        info = GrB_mxv(temp, NULL, NULL, GxB_PLUS_TIMES_FP64, a_grb, x_grb, NULL);
        if (info != GrB_SUCCESS) {
            GrB_Vector_free(&temp);
            MATX_ERROR("GrB_mxv A*x error: %d", info);
            grb_2_vec_d_i8(y);
            return MATX_ERR_INTERNAL;
        }
        info = GrB_apply(temp, NULL, NULL, GrB_TIMES_FP64, temp, alpha, NULL);
        if (info != GrB_SUCCESS) {
            GrB_Vector_free(&temp);
            MATX_ERROR("GrB_apply alpha*temp error: %d", info);
            grb_2_vec_d_i8(y);
            return MATX_ERR_INTERNAL;
        }
        info = GrB_eWiseAdd(y_grb, NULL, NULL, GrB_PLUS_FP64, y_grb, temp, NULL);
        GrB_Vector_free(&temp);
        if (info != GrB_SUCCESS) {
            MATX_ERROR("GrB_eWiseAdd temp + gy error: %d", info);
            grb_2_vec_d_i8(y);
            return MATX_ERR_INTERNAL;
        }
    }

    /* Export only modified output y (x is read-only) */
    if (grb_2_vec_d_i8(y) != 0)
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

    /* Early exit: alpha == 0 or nnz == 0 — just scale C by beta */
    if (alpha == 0.0 || A->nnz == 0) {
        if (beta == 0.0) {
            memset(C->data, 0, (size_t) C->nrows * C->ncols * C->stride * sizeof(matx_double));
        } else {
            const matx_int64_t minor = (C->layout == MATX_COL_MAJOR) ? C->nrows : C->ncols;
            if (C->stride == minor) {
                const size_t count = (size_t) C->nrows * (size_t) C->ncols;
                for (size_t i = 0; i < count; ++i)
                    C->data[i] *= beta;
            } else {
                const matx_int64_t major = (C->layout == MATX_COL_MAJOR) ? C->ncols : C->nrows;
                for (matx_int64_t j = 0; j < major; ++j)
                    for (matx_int64_t i = 0; i < minor; ++i)
                        C->data[i + j * C->stride] *= beta;
            }
        }
        return MATX_OK;
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
        return MATX_ERR_INTERNAL;
    }

    /* C = alpha*A*B + beta*C */
    GrB_Matrix c_grb = (GrB_Matrix) MATX_HANDLE(C, MATX_HANDLE_TYPE_GRB_MATRIX)->impl;
    GrB_Matrix b_grb = (GrB_Matrix) MATX_HANDLE(B, MATX_HANDLE_TYPE_GRB_MATRIX)->impl;
    GrB_Matrix a_grb = (GrB_Matrix) MATX_HANDLE(A, MATX_HANDLE_TYPE_GRB_MATRIX)->impl;

    if (beta == 0.0) {
        /* C = alpha * A * B — write directly into C, no temp needed */
        GrB_Descriptor desc;
        GrB_Descriptor_new(&desc);
        GrB_Descriptor_set(desc, GrB_OUTP, GrB_REPLACE);
        GrB_Info info = GrB_mxm(c_grb, NULL, NULL, GxB_PLUS_TIMES_FP64, a_grb, b_grb, desc);
        GrB_Descriptor_free(&desc);
        if (info != GrB_SUCCESS) {
            MATX_ERROR("GrB_mxm error: %d", info);
            grb_2_dense_d_i8(C);
            return MATX_ERR_INTERNAL;
        }
        if (alpha != 1.0) {
            info = GrB_apply(c_grb, NULL, NULL, GrB_TIMES_FP64, c_grb, alpha, NULL);
            if (info != GrB_SUCCESS) {
                MATX_ERROR("GrB_apply scale error: %d", info);
                grb_2_dense_d_i8(C);
                return MATX_ERR_INTERNAL;
            }
        }
    } else {
        /* C = alpha * A * B + beta * C */
        GrB_Matrix temp;
        GrB_Info info = GrB_Matrix_new(&temp, GrB_FP64, C->nrows, C->ncols);
        if (info != GrB_SUCCESS) {
            grb_2_dense_d_i8(C);
            return MATX_ERR_INTERNAL;
        }
        info = GrB_mxm(temp, NULL, NULL, GxB_PLUS_TIMES_FP64, a_grb, b_grb, NULL);
        if (info != GrB_SUCCESS) {
            GrB_Matrix_free(&temp);
            MATX_ERROR("GrB_mxm alpha * A * B error: %d", info);
            grb_2_dense_d_i8(C);
            return MATX_ERR_INTERNAL;
        }
        info = GrB_apply(temp, NULL, NULL, GrB_TIMES_FP64, temp, alpha, NULL);
        if (info != GrB_SUCCESS) {
            GrB_Matrix_free(&temp);
            MATX_ERROR("GrB_apply alpha * temp error: %d", info);
            grb_2_dense_d_i8(C);
            return MATX_ERR_INTERNAL;
        }
        info = GrB_apply(c_grb, NULL, NULL, GrB_TIMES_FP64, c_grb, beta, NULL);
        if (info != GrB_SUCCESS) {
            GrB_Matrix_free(&temp);
            MATX_ERROR("GrB_apply beta * gC error: %d", info);
            grb_2_dense_d_i8(C);
            return MATX_ERR_INTERNAL;
        }
        info = GrB_eWiseAdd(c_grb, NULL, NULL, GrB_PLUS_FP64, temp, c_grb, NULL);
        GrB_Matrix_free(&temp);
        if (info != GrB_SUCCESS) {
            MATX_ERROR("GrB_eWiseAdd temp + gC error: %d", info);
            grb_2_dense_d_i8(C);
            return MATX_ERR_INTERNAL;
        }
    }

    /* Export only modified output C (B is read-only) */
    if (grb_2_dense_d_i8(C) != 0)
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

// ---- SpGEMM → COO (reference) ----

/* Simple open-addressing hash table for SpGEMM accumulation. */
#define SPGEMM_HASH_EMPTY (-1)

static matx_status_t ref_spgemm_d_i8_grb(matx_double alpha,
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

    if (annz == 0 || bnnz == 0 || alpha == 0.0) {
        out->nnz = 0;
        return MATX_OK;
    }

    /* Estimate result size and allocate hash table. */
    matx_int64_t cap = annz + bnnz;
    if (cap < 64) cap = 64;
    /* Next power of two for fast mod. */
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
    /* Mark all slots as empty. */
    for (matx_int64_t i = 0; i < cap; ++i)
        hrow[i] = SPGEMM_HASH_EMPTY;

    /* Build CSR-style row-index for B to avoid O(nnz²) linear scan:
       for each row k, b_row_start[k..k+1) indexes b_perm to all B
       entries with B->rows[ib] == k. */
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

    /* Accumulate C(i,j) += A(i,k) * B(k,j) for matching k. */
    for (matx_int64_t ia = 0; ia < annz; ++ia) {
        matx_int64_t k  = A->columns[ia];
        matx_int64_t i  = A->rows[ia];
        double       av = alpha * A->values[ia];

        /* Only iterate B entries in row k (CSR row-index lookup). */
        for (matx_int64_t p = b_row_start[k]; p < b_row_start[k + 1]; ++p) {
            matx_int64_t ib = b_perm[p];
            matx_int64_t j = B->columns[ib];
            double bv = B->values[ib];
            double prod = av * bv;

            /* Insert/accumulate into hash table. */
            matx_int64_t idx = (i * 2654435761ULL + (unsigned long long)j) & (cap - 1);
            while (hrow[idx] != SPGEMM_HASH_EMPTY) {
                if (hrow[idx] == i && hcol[idx] == j) {
                    hval[idx] += prod;
                    goto next_b;
                }
                idx = (idx + 1) & (cap - 1);
            }
            hrow[idx] = i;
            hcol[idx] = j;
            hval[idx] = prod;
            next_b:;
        }
    }
    free(b_row_start); free(b_perm);

    /* Count non-zeros. */
    matx_int64_t nnz_out = 0;
    for (matx_int64_t i = 0; i < cap; ++i)
        if (hrow[i] != SPGEMM_HASH_EMPTY && hval[i] != 0.0)
            nnz_out++;

    /* Write output. */
    matx_int64_t pos = 0;
    for (matx_int64_t i = 0; i < cap; ++i) {
        if (hrow[i] != SPGEMM_HASH_EMPTY && hval[i] != 0.0) {
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

static matx_status_t ref_spgemm_z_i8_grb(matx_complex_d_t alpha,
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

    matx_int64_t m = A->nrows;
    matx_int64_t n = B->ncols;
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
        hrow[i] = SPGEMM_HASH_EMPTY;

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
        /* av *= alpha */
        double avr = alpha.real * av.real - alpha.imag * av.imag;
        double avi = alpha.real * av.imag + alpha.imag * av.real;

        /* Only iterate B entries in row k (CSR row-index lookup). */
        for (matx_int64_t p = b_row_start[k]; p < b_row_start[k + 1]; ++p) {
            matx_int64_t ib = b_perm[p];
            matx_int64_t     j  = B->columns[ib];
            matx_complex_d_t bv = B->values[ib];
            /* prod = av * bv */
            double pr = avr * bv.real - avi * bv.imag;
            double pi = avr * bv.imag + avi * bv.real;

            matx_int64_t idx = (i * 2654435761ULL + (unsigned long long)j) & (cap - 1);
            while (hrow[idx] != SPGEMM_HASH_EMPTY) {
                if (hrow[idx] == i && hcol[idx] == j) {
                    hval[idx].real += pr;
                    hval[idx].imag += pi;
                    goto next_bz;
                }
                idx = (idx + 1) & (cap - 1);
            }
            hrow[idx] = i;
            hcol[idx] = j;
            hval[idx].real = pr;
            hval[idx].imag = pi;
            next_bz:;
        }
    }
    free(b_row_start); free(b_perm);

    matx_int64_t nnz_out = 0;
    for (matx_int64_t i = 0; i < cap; ++i)
        if (hrow[i] != SPGEMM_HASH_EMPTY &&
            (hval[i].real != 0.0 || hval[i].imag != 0.0))
            nnz_out++;

    matx_int64_t pos = 0;
    for (matx_int64_t i = 0; i < cap; ++i) {
        if (hrow[i] != SPGEMM_HASH_EMPTY &&
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

static matx_status_t ref_sptrsv_d_i8_grb(matx_uplo_t uplo,
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

    /* Build row pointers: for each row, first[i] = index of first entry. */
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
        row_cnt[i] = 0; /* will reuse as write cursor */
    }
    row_start[n] = total;

    /* Permute entries into row-major order. */
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
        /* Forward substitution: i = 0 .. n-1 */
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
        /* Backward substitution: i = n-1 .. 0 */
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

static matx_status_t ref_sptrsv_z_i8_grb(matx_uplo_t uplo,
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
                               .spgemm_d_i8 = ref_spgemm_d_i8_grb,
                               .spgemm_z_i8 = ref_spgemm_z_i8_grb,
                               .sptrsv_d_i8 = ref_sptrsv_d_i8_grb,
                               .sptrsv_z_i8 = ref_sptrsv_z_i8_grb,
                               }};
    return b;
}
