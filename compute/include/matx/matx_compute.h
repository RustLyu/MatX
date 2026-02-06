#pragma once

#include "matx/matx.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum matx_blas_backend_kind_t {
  MATX_BLAS_BACKEND_REFERENCE = 0,
  MATX_BLAS_BACKEND_OPENBLAS = 1,
  MATX_BLAS_BACKEND_BLIS = 2
} matx_blas_backend_kind_t;

typedef struct matx_blas_vtable_t {
  matx_status_t (*dgemm)(matx_layout_t layout,
                         int trans_a,
                         int trans_b,
                         size_t m,
                         size_t n,
                         size_t k,
                         double alpha,
                         const double* a,
                         size_t lda,
                         const double* b,
                         size_t ldb,
                         double beta,
                         double* c,
                         size_t ldc);
  matx_status_t(*zgemm)(matx_layout_t layout,
                          int trans_a,
                          int trans_b,
                          size_t m,
                          size_t n,
                          size_t k,
                          const void* alpha,
                          const void* A,
                          size_t lda,
                          const void* B,
                          size_t ldb,
                          const void* beta,
                          void* C,
                          size_t ldc);
  matx_status_t(*zgemv)(matx_layout_t layout,
                          int trans_a,
                          size_t m,
                          size_t n,
                          const void* alpha,
                          const void* A,
                          size_t lda,
                          const void* B,
                          size_t ldb,
                          const void* beta,
                          void* C,
                          size_t ldc);

  matx_status_t(*dgemv)(matx_layout_t layout,
                          int trans_a,
                          size_t m,
                          size_t n,
                          double alpha,
                          const double* A,
                          size_t lda,
                          double* B,
                          size_t ldb,
                          double beta,
                          double* C,
                          size_t ldc);

  matx_status_t(*daxpy)(size_t n,
                          double alpha,
                          const double* x,
                          size_t lda,
                          const void* y,
                          size_t ldy);
  matx_status_t(*zaxpy)(size_t n,
                          const void* alpha,
                          const void* x,
                          size_t lda,
                          const void* y,
                          size_t ldy);
} matx_blas_vtable_t;

typedef struct matx_blas_t {
  matx_blas_backend_kind_t kind;
  matx_blas_vtable_t vt;
} matx_blas_t;

matx_blas_t matx_blas_make_reference(void);

// Initialize default backend based on MATX_BLAS_BACKEND (AUTO picks a reasonable default at build time).
matx_blas_t matx_blas_default(void);
const char* matx_blas_backend_name(matx_blas_backend_kind_t k);

// Convenience API operating on MatX dense types.
matx_status_t matx_gemm_f64(const matx_blas_t* blas,
                            int trans_a,
                            int trans_b,
                            double alpha,
                            const matx_dense_f64_t* A,
                            const matx_dense_f64_t* B,
                            double beta,
                            matx_dense_f64_t* C);

matx_status_t matx_gemm_c64(const matx_blas_t* blas,
                            int trans_a,
                            int trans_b,
                            matx_complex_f64 alpha,
                            const matx_dense_c64_t* A,
                            const matx_dense_c64_t* B,
                            matx_complex_f64 beta,
                            matx_dense_c64_t* C);

// ---- Level 1: vector ops ----
// y := alpha * x + y
matx_status_t matx_axpy_f64(const matx_blas_t* blas,
                            double alpha,
                            const matx_vec_f64_t* x,
                            matx_vec_f64_t* y);

matx_status_t matx_axpy_c64(
                            const matx_blas_t* blas,
                            matx_complex_f64 alpha,
                            const matx_vec_c64_t* x,
                            matx_vec_c64_t* y);

// ---- Level 2: matrix-vector ----
// y := alpha * op(A) * x + beta * y  (dense)
matx_status_t matx_gemv_f64(const matx_blas_t* blas,
                            int trans_a,
                            double alpha,
                            const matx_dense_f64_t* A,
                            const matx_vec_f64_t* x,
                            double beta,
                            matx_vec_f64_t* y);

matx_status_t matx_gemv_c64(const matx_blas_t* blas,
                            int trans_a,
                            matx_complex_f64 alpha,
                            const matx_dense_c64_t* A,
                            const matx_vec_c64_t* x,
                            matx_complex_f64 beta,
                            matx_vec_c64_t* y);

// y := alpha * A * x + beta * y  (sparse CSC, op(A)=A for now)
matx_status_t matx_spmv_csc_f64(double alpha,
                                const matx_csc_f64_t* A,
                                const matx_vec_f64_t* x,
                                double beta,
                                matx_vec_f64_t* y);

matx_status_t matx_spmv_csc_c64(matx_complex_f64 alpha,
                                const matx_csc_c64_t* A,
                                const matx_vec_c64_t* x,
                                matx_complex_f64 beta,
                                matx_vec_c64_t* y);

// ---- Level 3: matrix-matrix ----
// C := alpha * op(A) * op(B) + beta * C  (dense)
matx_status_t matx_gemm_f64(const matx_blas_t* blas,
                            int trans_a,
                            int trans_b,
                            double alpha,
                            const matx_dense_f64_t* A,
                            const matx_dense_f64_t* B,
                            double beta,
                            matx_dense_f64_t* C);

// C := alpha * A * B + beta * C  (sparse CSC * dense, op() = I for now)
matx_status_t matx_spmm_csc_f64(double alpha,
                                const matx_csc_f64_t* A,
                                const matx_dense_f64_t* B,
                                double beta,
                                matx_dense_f64_t* C);

matx_status_t matx_spmm_csc_c64(matx_complex_f64 alpha,
                                const matx_csc_c64_t* A,
                                const matx_dense_c64_t* B,
                                matx_complex_f64 beta,
                                matx_dense_c64_t* C);

#ifdef __cplusplus
}
#endif

