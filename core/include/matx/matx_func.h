#ifndef MATX_FUNC_H
#define MATX_FUNC_H

#include "matx/matx_types.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

MATX_API const char* matx_status_string(matx_status_t st);

MATX_API matx_alloc_t matx_alloc_default(void);
MATX_API void* matx_malloc(const matx_alloc_t* a, size_t size);
MATX_API void matx_free(const matx_alloc_t* a, void* ptr);

MATX_API matx_status_t matx_vec_d_i8_create(const matx_alloc_t* alloc,
                                            matx_vec_d_i8_t* out,
                                            matx_double* data,
                                            matx_int64_t n);

/**
	 * @brief Duplicate a real vector
	 * @formula out[i] := in[i],  i = 0, 1, ..., n-1
	 */
MATX_API matx_status_t matx_vec_d_i8_dup(const matx_alloc_t* alloc,
                                         const matx_vec_d_i8_t in,
                                         matx_vec_d_i8_t* out);

MATX_API matx_status_t matx_vec_d_i8_wrap(const matx_alloc_t* alloc,
                                          matx_vec_d_i8_t* out,
                                          matx_int64_t n,
                                          matx_int64_t stride,
                                          matx_double* data);

MATX_API void matx_vec_d_i8_destroy(const matx_alloc_t* alloc, matx_vec_d_i8_t v);

MATX_API matx_status_t matx_dense_d_i8_create(const matx_alloc_t* alloc,
                                              matx_dense_d_i8_t* out,
                                              matx_layout_t layout,
                                              matx_int64_t rows,
                                              matx_int64_t cols,
                                              matx_double* data);

/**
	 * @brief Duplicate a real dense matrix
	 * @formula out[i][j] := in[i][j],  i = 0,...,rows-1; j = 0,...,cols-1
	 */
MATX_API matx_status_t matx_dense_d_i8_dup(const matx_alloc_t* alloc,
                                           const matx_dense_d_i8_t in,
                                           matx_dense_d_i8_t* out);

MATX_API matx_status_t matx_dense_d_i8_wrap(const matx_alloc_t* alloc,
                                            matx_dense_d_i8_t* out,
                                            matx_int64_t rows,
                                            matx_int64_t cols,
                                            matx_int64_t stride,
                                            matx_layout_t layout,
                                            matx_double* data);

MATX_API void matx_dense_d_i8_destroy(const matx_alloc_t* alloc, matx_dense_d_i8_t m);

MATX_API matx_status_t matx_vec_z_i8_create(const matx_alloc_t* alloc,
                                            matx_vec_z_i8_t* out,
                                            matx_complex_d_t* data,
                                            matx_int64_t n);

/**
	 * @brief Duplicate a complex vector
	 * @formula out[i] := in[i],  i = 0, 1, ..., n-1
	 */
MATX_API matx_status_t matx_vec_z_i8_dup(const matx_alloc_t* alloc,
                                         const matx_vec_z_i8_t in,
                                         matx_vec_z_i8_t* out);

MATX_API matx_status_t matx_vec_z_i8_wrap(const matx_alloc_t* alloc,
                                          matx_vec_z_i8_t* out,
                                          matx_int64_t n,
                                          matx_int64_t stride,
                                          matx_complex_d_t* data);

MATX_API void matx_vec_z_i8_destroy(const matx_alloc_t* alloc, matx_vec_z_i8_t v);

MATX_API matx_status_t matx_dense_z_i8_create(const matx_alloc_t* alloc,
                                              matx_dense_z_i8_t* out,
                                              matx_layout_t layout,
                                              matx_int64_t rows,
                                              matx_int64_t cols,
                                              matx_complex_d_t* data);

/**
	 * @brief Duplicate a complex dense matrix
	 * @formula out[i][j] := in[i][j],  i = 0,...,rows-1; j = 0,...,cols-1
	 */
MATX_API matx_status_t matx_dense_z_i8_dup(const matx_alloc_t* alloc,
                                           const matx_dense_z_i8_t in,
                                           matx_dense_z_i8_t* out);

MATX_API matx_status_t matx_dense_z_i8_wrap(const matx_alloc_t* alloc,
                                            matx_dense_z_i8_t* out,
                                            matx_int64_t rows,
                                            matx_int64_t cols,
                                            matx_int64_t stride,
                                            matx_layout_t layout,
                                            matx_complex_d_t* data);

MATX_API void matx_dense_z_i8_destroy(const matx_alloc_t* alloc, matx_dense_z_i8_t m);

MATX_API matx_status_t matx_coo_sparse_d_i8_create(const matx_alloc_t* alloc,
                                                   matx_coo_d_i8_t* out,
                                                   matx_int64_t nrows,
                                                   matx_int64_t ncols,
                                                   matx_int64_t nnz,
                                                   matx_int64_t* ap,
                                                   matx_int64_t* ai,
                                                   matx_double* ax);

/**
	 * @brief Duplicate a real COO sparse matrix
	 * @formula out.rows := in.rows; out.cols := in.cols; out.values := in.values  (copy all nnz entries)
	 */
MATX_API matx_status_t matx_coo_d_i8_dup(const matx_alloc_t* alloc,
                                         const matx_coo_d_i8_t in,
                                         matx_coo_d_i8_t* out);

MATX_API matx_status_t matx_csc_sparse_d_i8_create(const matx_alloc_t* alloc,
                                                   matx_csc_d_i8_t* out,
                                                   matx_int64_t nrows,
                                                   matx_int64_t ncols,
                                                   matx_int64_t nnz);

MATX_API matx_status_t matx_csc_sparse_d_i8_wrap(const matx_alloc_t* alloc,
                                                 matx_csc_d_i8_t* out,
                                                 matx_int64_t nrows,
                                                 matx_int64_t ncols,
                                                 matx_int64_t nnz,
                                                 const matx_int64_t* col_ptr,
                                                 const matx_int64_t* row_ind,
                                                 const matx_double* values);

MATX_API void matx_coo_sparse_d_i8_destroy(const matx_alloc_t* alloc, matx_coo_d_i8_t m);

MATX_API matx_status_t matx_coo_sparse_z_i8_create(const matx_alloc_t* alloc,
                                                   matx_coo_z_i8_t* out,
                                                   matx_int64_t nrows,
                                                   matx_int64_t ncols,
                                                   matx_int64_t nnz,
                                                   matx_int64_t* ap,
                                                   matx_int64_t* ai,
                                                   matx_complex_d_t* ax);

/**
	 * @brief Duplicate a complex COO sparse matrix
	 * @formula out.rows := in.rows; out.cols := in.cols; out.values := in.values  (copy all nnz entries)
	 */
MATX_API matx_status_t matx_coo_z_i8_dup(const matx_alloc_t* alloc,
                                         const matx_coo_z_i8_t in,
                                         matx_coo_z_i8_t* out);

MATX_API matx_status_t matx_coo_sparse_z_i8_wrap(const matx_alloc_t* alloc,
                                                 matx_coo_z_i8_t* out,
                                                 matx_int64_t nrows,
                                                 matx_int64_t ncols,
                                                 matx_int64_t nnz,
                                                 const matx_int64_t* rows,
                                                 const matx_int64_t* cols,
                                                 const matx_complex_d_t* values);

MATX_API void matx_coo_sparse_z_i8_destroy(const matx_alloc_t* alloc, matx_coo_z_i8_t m);

MATX_API matx_status_t matx_csc_sparse_z_i8_create(const matx_alloc_t* alloc,
                                                   matx_csc_z_i8_t* out,
                                                   matx_int64_t nrows,
                                                   matx_int64_t ncols,
                                                   matx_int64_t nnz);
MATX_API void matx_csc_sparse_z_i8_destroy(const matx_alloc_t* alloc, matx_csc_z_i8_t m);
MATX_API void matx_csc_sparse_d_i8_destroy(const matx_alloc_t* alloc, matx_csc_d_i8_t m);

// ---- Vector fill / init ----

/**
	 * @brief Fill a real vector with a scalar value
	 * @formula v[i] := val,  i = 0, 1, ..., n-1
	 */
MATX_API matx_status_t matx_vec_d_i8_fill(matx_vec_d_i8_t v, matx_double val);

/**
	 * @brief Fill a complex vector with a scalar value
	 * @formula v[i] := val,  i = 0, 1, ..., n-1
	 */
MATX_API matx_status_t matx_vec_z_i8_fill(matx_vec_z_i8_t v, matx_complex_d_t val);

/**
	 * @brief Set all elements of a real vector to zero
	 * @formula v[i] := 0,  i = 0, 1, ..., n-1
	 */
MATX_API matx_status_t matx_vec_d_i8_zeros(matx_vec_d_i8_t v);

/**
	 * @brief Set all elements of a complex vector to zero
	 * @formula v[i] := 0 + 0i,  i = 0, 1, ..., n-1
	 */
MATX_API matx_status_t matx_vec_z_i8_zeros(matx_vec_z_i8_t v);

/**
	 * @brief Set all elements of a real vector to one
	 * @formula v[i] := 1,  i = 0, 1, ..., n-1
	 */
MATX_API matx_status_t matx_vec_d_i8_ones(matx_vec_d_i8_t v);

// ---- Dense matrix fill / init ----

/**
	 * @brief Fill a real dense matrix with a scalar value
	 * @formula m[i][j] := val,  i = 0,...,rows-1; j = 0,...,cols-1
	 */
MATX_API matx_status_t matx_dense_d_i8_fill(matx_dense_d_i8_t m, matx_double val);

/**
	 * @brief Fill a complex dense matrix with a scalar value
	 * @formula m[i][j] := val,  i = 0,...,rows-1; j = 0,...,cols-1
	 */
MATX_API matx_status_t matx_dense_z_i8_fill(matx_dense_z_i8_t m, matx_complex_d_t val);

/**
	 * @brief Set all elements of a real dense matrix to zero
	 * @formula m[i][j] := 0,  i = 0,...,rows-1; j = 0,...,cols-1
	 */
MATX_API matx_status_t matx_dense_d_i8_zeros(matx_dense_d_i8_t m);

/**
	 * @brief Set all elements of a complex dense matrix to zero
	 * @formula m[i][j] := 0 + 0i,  i = 0,...,rows-1; j = 0,...,cols-1
	 */
MATX_API matx_status_t matx_dense_z_i8_zeros(matx_dense_z_i8_t m);

/**
	 * @brief Set all elements of a real dense matrix to one
	 * @formula m[i][j] := 1,  i = 0,...,rows-1; j = 0,...,cols-1
	 */
MATX_API matx_status_t matx_dense_d_i8_ones(matx_dense_d_i8_t m);

// ---- Dense matrix trace ----

/**
	 * @brief Trace of a real dense matrix (sum of diagonal elements)
	 * @formula out := sum_{i=0}^{n-1} A[i][i]
	 *          where n = min(rows, cols)
	 */
MATX_API matx_status_t matx_dense_d_i8_trace(const matx_dense_d_i8_t A, matx_double* out);

/**
	 * @brief Trace of a complex dense matrix (sum of diagonal elements)
	 * @formula out := sum_{i=0}^{n-1} A[i][i]
	 *          where n = min(rows, cols)
	 */
MATX_API matx_status_t matx_dense_z_i8_trace(const matx_dense_z_i8_t A, matx_complex_d_t* out);

// ---- Type conversion ----

/**
	 * @brief Convert a real dense matrix to complex (zero imaginary part)
	 * @formula out[i][j] := A[i][j] + 0i,  i = 0,...,rows-1; j = 0,...,cols-1
	 */
MATX_API matx_status_t matx_dense_d_i8_to_z_i8(const matx_alloc_t* alloc,
                                               const matx_dense_d_i8_t A,
                                               matx_dense_z_i8_t* out);

/**
	 * @brief Convert a real vector to complex (zero imaginary part)
	 * @formula out[i] := v[i] + 0i,  i = 0, 1, ..., n-1
	 */
MATX_API matx_status_t matx_vec_d_i8_to_z_i8(const matx_alloc_t* alloc,
                                             const matx_vec_d_i8_t v,
                                             matx_vec_z_i8_t* out);

// ---- Element-wise math: vectors ----

#define MATX_DECL_VEC_MATH(name) \
    MATX_API matx_status_t matx_vec_d_i8_##name(const matx_alloc_t* alloc, \
                                                const matx_vec_d_i8_t v, \
                                                matx_vec_d_i8_t* out); \
    MATX_API matx_status_t matx_vec_z_i8_##name(const matx_alloc_t* alloc, \
                                                const matx_vec_z_i8_t v, \
                                                matx_vec_z_i8_t* out)

MATX_DECL_VEC_MATH(exp);
MATX_DECL_VEC_MATH(log);
MATX_DECL_VEC_MATH(sqrt);
MATX_DECL_VEC_MATH(sin);
MATX_DECL_VEC_MATH(cos);

#undef MATX_DECL_VEC_MATH

MATX_API matx_status_t matx_vec_d_i8_abs(const matx_alloc_t* alloc,
                                         const matx_vec_d_i8_t v,
                                         matx_vec_d_i8_t* out);
MATX_API matx_status_t matx_vec_z_i8_abs(const matx_alloc_t* alloc,
                                         const matx_vec_z_i8_t v,
                                         matx_vec_d_i8_t* out);
MATX_API matx_status_t matx_vec_d_i8_pow(const matx_alloc_t* alloc,
                                         const matx_vec_d_i8_t v,
                                         matx_double exp,
                                         matx_vec_d_i8_t* out);
MATX_API matx_status_t matx_vec_z_i8_pow(const matx_alloc_t* alloc,
                                         const matx_vec_z_i8_t v,
                                         matx_complex_d_t exp,
                                         matx_vec_z_i8_t* out);

// ---- Element-wise math: dense matrices ----

#define MATX_DECL_DENSE_MATH(name) \
    MATX_API matx_status_t matx_dense_d_i8_##name(const matx_alloc_t* alloc, \
                                                  const matx_dense_d_i8_t A, \
                                                  matx_dense_d_i8_t* out); \
    MATX_API matx_status_t matx_dense_z_i8_##name(const matx_alloc_t* alloc, \
                                                  const matx_dense_z_i8_t A, \
                                                  matx_dense_z_i8_t* out)

MATX_DECL_DENSE_MATH(exp);
MATX_DECL_DENSE_MATH(log);
MATX_DECL_DENSE_MATH(sqrt);
MATX_DECL_DENSE_MATH(sin);
MATX_DECL_DENSE_MATH(cos);

#undef MATX_DECL_DENSE_MATH

MATX_API matx_status_t matx_dense_d_i8_abs(const matx_alloc_t* alloc,
                                           const matx_dense_d_i8_t A,
                                           matx_dense_d_i8_t* out);
MATX_API matx_status_t matx_dense_z_i8_abs(const matx_alloc_t* alloc,
                                           const matx_dense_z_i8_t A,
                                           matx_dense_d_i8_t* out);
MATX_API matx_status_t matx_dense_d_i8_pow(const matx_alloc_t* alloc,
                                           const matx_dense_d_i8_t A,
                                           matx_double exp,
                                           matx_dense_d_i8_t* out);
MATX_API matx_status_t matx_dense_z_i8_pow(const matx_alloc_t* alloc,
                                           const matx_dense_z_i8_t A,
                                           matx_complex_d_t exp,
                                           matx_dense_z_i8_t* out);

// ---- Vector element-wise arithmetic ----

#define MATX_DECL_VEC_ARITH(name, op_desc) \
    MATX_API matx_status_t matx_vec_d_i8_##name(const matx_alloc_t* alloc, \
                                                const matx_vec_d_i8_t a, \
                                                const matx_vec_d_i8_t b, \
                                                matx_vec_d_i8_t* out); \
    MATX_API matx_status_t matx_vec_z_i8_##name(const matx_alloc_t* alloc, \
                                                const matx_vec_z_i8_t a, \
                                                const matx_vec_z_i8_t b, \
                                                matx_vec_z_i8_t* out)

MATX_DECL_VEC_ARITH(add, C = A + B);
MATX_DECL_VEC_ARITH(sub, C = A - B);
MATX_DECL_VEC_ARITH(mul, C = A * B);
MATX_DECL_VEC_ARITH(div, C = A / B);

#undef MATX_DECL_VEC_ARITH

// ---- Dense element-wise arithmetic ----

#define MATX_DECL_DENSE_ARITH(name) \
    MATX_API matx_status_t matx_dense_d_i8_##name(const matx_alloc_t* alloc, \
                                                  const matx_dense_d_i8_t A, \
                                                  const matx_dense_d_i8_t B, \
                                                  matx_dense_d_i8_t* out); \
    MATX_API matx_status_t matx_dense_z_i8_##name(const matx_alloc_t* alloc, \
                                                  const matx_dense_z_i8_t A, \
                                                  const matx_dense_z_i8_t B, \
                                                  matx_dense_z_i8_t* out)

MATX_DECL_DENSE_ARITH(add);
MATX_DECL_DENSE_ARITH(sub);
MATX_DECL_DENSE_ARITH(mul);
MATX_DECL_DENSE_ARITH(div);

#undef MATX_DECL_DENSE_ARITH

// ---- In-place scalar operations ----

MATX_API matx_status_t matx_vec_d_i8_add_scalar(matx_vec_d_i8_t v, matx_double val);
MATX_API matx_status_t matx_vec_z_i8_add_scalar(matx_vec_z_i8_t v, matx_complex_d_t val);
MATX_API matx_status_t matx_vec_d_i8_mul_scalar(matx_vec_d_i8_t v, matx_double val);
MATX_API matx_status_t matx_vec_z_i8_mul_scalar(matx_vec_z_i8_t v, matx_complex_d_t val);

MATX_API matx_status_t matx_dense_d_i8_add_scalar(matx_dense_d_i8_t m, matx_double val);
MATX_API matx_status_t matx_dense_z_i8_add_scalar(matx_dense_z_i8_t m, matx_complex_d_t val);
MATX_API matx_status_t matx_dense_d_i8_mul_scalar(matx_dense_d_i8_t m, matx_double val);
MATX_API matx_status_t matx_dense_z_i8_mul_scalar(matx_dense_z_i8_t m, matx_complex_d_t val);

// ---- Diagonal matrix creation / extraction ----

MATX_API matx_status_t matx_diag_d_i8_create(const matx_alloc_t* alloc,
                                             const matx_vec_d_i8_t diag,
                                             matx_dense_d_i8_t* out);
MATX_API matx_status_t matx_diag_z_i8_create(const matx_alloc_t* alloc,
                                             const matx_vec_z_i8_t diag,
                                             matx_dense_z_i8_t* out);
MATX_API matx_status_t matx_dense_d_i8_get_diag(const matx_alloc_t* alloc,
                                                const matx_dense_d_i8_t A,
                                                matx_vec_d_i8_t* out);
MATX_API matx_status_t matx_dense_z_i8_get_diag(const matx_alloc_t* alloc,
                                                const matx_dense_z_i8_t A,
                                                matx_vec_z_i8_t* out);

// ---- Submatrix block copy ----


/**
 * @brief Extract a contiguous block from a real dense matrix (deep copy)
 * @formula out[i][j] := A[rs + i][cs + j],  i = 0,...,(re-rs)-1; j = 0,...,(ce-cs)-1
 */
MATX_API matx_status_t matx_dense_d_i8_get_block(const matx_alloc_t* alloc,
                                                  const matx_dense_d_i8_t A,
                                                  matx_int64_t rs, matx_int64_t re,
                                                  matx_int64_t cs, matx_int64_t ce,
                                                  matx_dense_d_i8_t* out);

/**
 * @brief Extract a contiguous block from a complex dense matrix (deep copy)
 * @formula out[i][j] := A[rs + i][cs + j],  i = 0,...,(re-rs)-1; j = 0,...,(ce-cs)-1
 */
MATX_API matx_status_t matx_dense_z_i8_get_block(const matx_alloc_t* alloc,
                                                  const matx_dense_z_i8_t A,
                                                  matx_int64_t rs, matx_int64_t re,
                                                  matx_int64_t cs, matx_int64_t ce,
                                                  matx_dense_z_i8_t* out);

/**
 * @brief Copy a block from a real dense matrix into another at a target position
 * @formula B[dr + i][dc + j] := A[rs + i][cs + j],  i = 0,...,(re-rs)-1; j = 0,...,(ce-cs)-1
 */
MATX_API matx_status_t matx_dense_d_i8_set_block(
    const matx_dense_d_i8_t A,
    matx_int64_t rs, matx_int64_t re,
    matx_int64_t cs, matx_int64_t ce,
    matx_dense_d_i8_t B,
    matx_int64_t dr, matx_int64_t dc);

/**
 * @brief Copy a block from a complex dense matrix into another at a target position
 * @formula B[dr + i][dc + j] := A[rs + i][cs + j],  i = 0,...,(re-rs)-1; j = 0,...,(ce-cs)-1
 */
MATX_API matx_status_t matx_dense_z_i8_set_block(
    const matx_dense_z_i8_t A,
    matx_int64_t rs, matx_int64_t re,
    matx_int64_t cs, matx_int64_t ce,
    matx_dense_z_i8_t B,
    matx_int64_t dr, matx_int64_t dc);

// ---- Cumulative sum ----

MATX_API matx_status_t matx_vec_d_i8_cumsum(const matx_alloc_t* alloc,
                                            const matx_vec_d_i8_t v,
                                            matx_vec_d_i8_t* out);

// ---- Random number generation ----

MATX_API matx_status_t matx_vec_d_i8_rand_uniform(matx_vec_d_i8_t v,
                                                  matx_double low,
                                                  matx_double high,
                                                  unsigned int seed);
MATX_API matx_status_t matx_dense_d_i8_rand_uniform(matx_dense_d_i8_t m,
                                                    matx_double low,
                                                    matx_double high,
                                                    unsigned int seed);
MATX_API matx_status_t matx_vec_d_i8_rand_normal(matx_vec_d_i8_t v,
                                                 matx_double mean,
                                                 matx_double stddev,
                                                 unsigned int seed);

#ifdef __cplusplus
}
#endif
#endif // MATX_FUNC_H