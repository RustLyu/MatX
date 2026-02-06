#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---- Version ----
#define MATX_VERSION_MAJOR 0
#define MATX_VERSION_MINOR 1
#define MATX_VERSION_PATCH 0

typedef enum matx_status_t {
  MATX_OK = 0,
  MATX_ERR_INVALID_ARG = 1,
  MATX_ERR_OUT_OF_MEMORY = 2,
  MATX_ERR_NOT_SUPPORTED = 3,
  MATX_ERR_INTERNAL = 4
} matx_status_t;

const char* matx_status_string(matx_status_t st);
const char* matx_version_string(void);

// ---- Alloc ----
typedef void* (*matx_malloc_fn)(size_t size, void* user);
typedef void (*matx_free_fn)(void* ptr, void* user);

typedef struct matx_alloc_t {
  matx_malloc_fn malloc_fn;
  matx_free_fn free_fn;
  void* user;
} matx_alloc_t;

matx_alloc_t matx_alloc_default(void);
void* matx_malloc(const matx_alloc_t* a, size_t size);
void matx_free(const matx_alloc_t* a, void* ptr);

// ---- Layout ----
typedef enum matx_layout_t {
  MATX_COL_MAJOR = 0,
  MATX_ROW_MAJOR = 1
} matx_layout_t;

// ---- Real/complex scalar ----
typedef struct matx_complex_f64_t {
  double real;
  double imag;
} matx_complex_f64;

// ---- Dense vector (double) ----
typedef struct matx_vec_f64_t {
  size_t n;
  size_t stride;
  double* data;
  uint32_t flags;
} matx_vec_f64_t;

matx_status_t matx_vec_f64_create(matx_vec_f64_t* out,
                                  size_t n,
                                  const matx_alloc_t* alloc);

matx_status_t matx_vec_f64_wrap(matx_vec_f64_t* out,
                                size_t n,
                                size_t stride,
                                double* data);

void matx_vec_f64_destroy(matx_vec_f64_t* v,
                          const matx_alloc_t* alloc);

// ---- Dense matrix (double) ----
typedef struct matx_dense_f64_t {
  size_t rows;
  size_t cols;
  size_t stride;     // leading dimension: if col-major => ld = stride (>= rows); if row-major => ld = stride (>= cols)
  matx_layout_t layout;
  double* data;
  uint32_t flags;    // reserved for future (ownership, alignment, etc.)
} matx_dense_f64_t;

matx_status_t matx_dense_f64_create(matx_dense_f64_t* out,
                                   size_t rows,
                                   size_t cols,
                                   matx_layout_t layout,
                                   const matx_alloc_t* alloc);

matx_status_t matx_dense_f64_wrap(matx_dense_f64_t* out,
                                 size_t rows,
                                 size_t cols,
                                 size_t stride,
                                 matx_layout_t layout,
                                 double* data);

void matx_dense_f64_destroy(matx_dense_f64_t* m, const matx_alloc_t* alloc);

// ---- Dense vector (complex) ----
typedef struct matx_vec_c64_t {
  size_t n;
  size_t stride;
  matx_complex_f64* data;
  uint32_t flags;
} matx_vec_c64_t;

matx_status_t matx_vec_c64_create(matx_vec_c64_t* out,
                                  size_t n,
                                  const matx_alloc_t* alloc);

matx_status_t matx_vec_c64_wrap(matx_vec_c64_t* out,
                                size_t n,
                                size_t stride,
                                matx_complex_f64* data);

void matx_vec_c64_destroy(matx_vec_c64_t* v,
                          const matx_alloc_t* alloc);

// ---- Dense matrix (complex) ----
typedef struct matx_dense_c64_t {
  size_t rows;
  size_t cols;
  size_t stride;
  matx_layout_t layout;
  matx_complex_f64* data;
  uint32_t flags;
} matx_dense_c64_t;

matx_status_t matx_dense_c64_create(matx_dense_c64_t* out,
                                    size_t rows,
                                    size_t cols,
                                    matx_layout_t layout,
                                    const matx_alloc_t* alloc);

matx_status_t matx_dense_c64_wrap(matx_dense_c64_t* out,
                                  size_t rows,
                                  size_t cols,
                                  size_t stride,
                                  matx_layout_t layout,
                                  matx_complex_f64* data);

void matx_dense_c64_destroy(matx_dense_c64_t* m,
                            const matx_alloc_t* alloc);

// ---- Sparse CSC (real/complex) ----
typedef struct matx_csc_f64_t {
  size_t nrows;
  size_t ncols;
  size_t nnz;
  const int* col_ptr;   // size ncols+1, 0-based
  const int* row_ind;   // size nnz, 0-based
  const double* values; // size nnz
} matx_csc_f64_t;

typedef struct matx_csc_c64_t {
  size_t nrows;
  size_t ncols;
  size_t nnz;
  const int* col_ptr;
  const int* row_ind;
  const matx_complex_f64* values;
} matx_csc_c64_t;

#ifdef __cplusplus
}
#endif

