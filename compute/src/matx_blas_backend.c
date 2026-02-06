#include "matx/matx_compute.h"

#include <string.h>

// Forward decls
matx_blas_t matx_blas_make_reference(void);

const char* matx_blas_backend_name(matx_blas_backend_kind_t k) {
  switch (k) {
    case MATX_BLAS_BACKEND_REFERENCE: return "REFERENCE";
    case MATX_BLAS_BACKEND_OPENBLAS: return "OPENBLAS";
    case MATX_BLAS_BACKEND_BLIS: return "BLIS";
    default: return "UNKNOWN";
  }
}

static matx_blas_t choose_default_backend(void) {
  // Build-time selection (simple & portable). Can be extended to runtime CPUID switching later.
  // If user wants strict control: set -DMATX_BLAS_BACKEND=OPENBLAS/BLIS/REFERENCE
#if defined(MATX_BLAS_BACKEND_REFERENCE_ONLY)
  return matx_blas_make_reference();
#else
  // If no external backend is wired in, fall back to reference.
  return matx_blas_make_reference();
#endif
}

matx_blas_t matx_blas_default(void) {
  return choose_default_backend();
}

matx_status_t matx_gemm_f64(const matx_blas_t* blas,
                            int trans_a,
                            int trans_b,
                            double alpha,
                            const matx_dense_f64_t* A,
                            const matx_dense_f64_t* B,
                            double beta,
                            matx_dense_f64_t* C) {
  if (!blas || !A || !B || !C) return MATX_ERR_INVALID_ARG;
  if (!blas->vt.dgemm) return MATX_ERR_NOT_SUPPORTED;

  if (A->layout != B->layout || A->layout != C->layout) return MATX_ERR_INVALID_ARG;
  if (A->layout != MATX_COL_MAJOR && A->layout != MATX_ROW_MAJOR) return MATX_ERR_INVALID_ARG;

  const size_t a_rows = A->rows;
  const size_t a_cols = A->cols;
  const size_t b_rows = B->rows;
  const size_t b_cols = B->cols;

  const size_t m = (trans_a ? a_cols : a_rows);
  const size_t kA = (trans_a ? a_rows : a_cols);
  const size_t kB = (trans_b ? b_cols : b_rows);
  const size_t n = (trans_b ? b_rows : b_cols);

  if (kA != kB) return MATX_ERR_INVALID_ARG;
  if (C->rows != m || C->cols != n) return MATX_ERR_INVALID_ARG;

  return blas->vt.dgemm(A->layout,
                        trans_a,
                        trans_b,
                        m,
                        n,
                        kA,
                        alpha,
                        A->data,
                        A->stride,
                        B->data,
                        B->stride,
                        beta,
                        C->data,
                        C->stride);
}


matx_status_t matx_gemm_c64(const matx_blas_t* blas,
    int trans_a,
    int trans_b,
    matx_complex_f64 alpha,
    const matx_dense_c64_t* A,
    const matx_dense_c64_t* B,
    matx_complex_f64 beta,
    matx_dense_c64_t* C) {
    if (!A || !B || !C || !A->data || !B->data || !C->data)
        return MATX_ERR_INVALID_ARG;
    if (A->layout != B->layout || A->layout != C->layout)
        return MATX_ERR_INVALID_ARG;

    const size_t a_rows = A->rows;
    const size_t a_cols = A->cols;
    const size_t b_rows = B->rows;
    const size_t b_cols = B->cols;
    const size_t m = trans_a ? a_cols : a_rows;
    const size_t kA = trans_a ? a_rows : a_cols;
    const size_t kB = trans_b ? b_cols : b_rows;
    const size_t n = trans_b ? b_rows : b_cols;

    if (kA != kB)
        return MATX_ERR_INVALID_ARG;
    if (C->rows != m || C->cols != n)
        return MATX_ERR_INVALID_ARG;

    return blas->vt.zgemm(A->layout,
        trans_a,
        trans_b,
        m,
        n,
        kA,
        (const void*)&alpha,
        A->data,
        A->stride,
        B->data,
        B->stride,
        (const void*)&beta,
        C->data,
        C->stride);
}

matx_status_t matx_gemv_c64(const matx_blas_t* blas,
    int trans_a,
    matx_complex_f64 alpha,
    const matx_dense_c64_t* A,
    const matx_vec_c64_t* x,
    matx_complex_f64 beta,
    matx_vec_c64_t* y) {
    (void)blas;
    if (!A || !x || !y || !A->data || !x->data || !y->data)
        return MATX_ERR_INVALID_ARG;

    const size_t m = A->rows;
    const size_t n = A->cols;
    const size_t len_x = trans_a ? m : n;
    const size_t len_y = trans_a ? n : m;

    if (x->n != len_x || y->n != len_y)
        return MATX_ERR_INVALID_ARG;

    return blas->vt.zgemv(A->layout, trans_a, (int)m, (int)n, (const void*)&alpha,
        (const void*)A->data, (int)A->stride, (const void*)x->data,
        (int)x->stride, (const void*)&beta, (void*)y->data,
        (int)y->stride);
}

matx_status_t matx_gemv_f64(const matx_blas_t* blas,
    int trans_a,
    double alpha,
    const matx_dense_f64_t* A,
    const matx_vec_f64_t* x,
    double beta,
    matx_vec_f64_t* y) {
    (void)blas;
    if (!A || !x || !y || !A->data || !x->data || !y->data)
        return MATX_ERR_INVALID_ARG;

    const size_t m = A->rows;
    const size_t n = A->cols;
    const size_t len_x = trans_a ? m : n;
    const size_t len_y = trans_a ? n : m;

    if (x->n != len_x || y->n != len_y)
        return MATX_ERR_INVALID_ARG;

    return blas->vt.dgemv(A->layout, trans_a, (int)m, (int)n, alpha,
        A->data, (int)A->stride, x->data,
        (int)x->stride, beta, y->data,
        (int)y->stride);
}