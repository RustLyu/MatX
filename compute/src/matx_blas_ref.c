#include "matx/matx_compute.h"

#include <limits.h>

#if defined(MATX_HAVE_OPENBLAS) || defined(MATX_HAVE_BLIS)
  #include <cblas.h>
#endif

static matx_status_t ref_dgemm(matx_layout_t layout,
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
                              size_t ldc) {
  if (!a || !b || !c) return MATX_ERR_INVALID_ARG;
  if (layout != MATX_COL_MAJOR && layout != MATX_ROW_MAJOR) return MATX_ERR_INVALID_ARG;

  if (m > (size_t)INT_MAX || n > (size_t)INT_MAX || k > (size_t)INT_MAX ||
      lda > (size_t)INT_MAX || ldb > (size_t)INT_MAX || ldc > (size_t)INT_MAX) {
    return MATX_ERR_NOT_SUPPORTED;
  }

  const enum CBLAS_ORDER order =
      (layout == MATX_COL_MAJOR) ? CblasColMajor : CblasRowMajor;

  const enum CBLAS_TRANSPOSE ta =
      (trans_a != 0) ? CblasTrans : CblasNoTrans;
  const enum CBLAS_TRANSPOSE tb =
      (trans_b != 0) ? CblasTrans : CblasNoTrans;

  cblas_dgemm(order,
              ta,
              tb,
              (int)m,
              (int)n,
              (int)k,
              alpha,
              a,
              (int)lda,
              b,
              (int)ldb,
              beta,
              c,
              (int)ldc);
  return MATX_OK;
}

static matx_status_t ref_zgemm(matx_layout_t layout,
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
    size_t ldc) {
    if (!A || !B || !C || !alpha || !beta)
        return MATX_ERR_INVALID_ARG;

    if (m > INT_MAX || n > INT_MAX || k > INT_MAX ||
        lda > INT_MAX || ldb > INT_MAX || ldc > INT_MAX)
        return MATX_ERR_NOT_SUPPORTED;

    const enum CBLAS_ORDER order =
        (layout == MATX_COL_MAJOR) ? CblasColMajor : CblasRowMajor;

    const enum CBLAS_TRANSPOSE ta =
        trans_a ? CblasTrans : CblasNoTrans;

    const enum CBLAS_TRANSPOSE tb =
        trans_b ? CblasTrans : CblasNoTrans;

    cblas_zgemm(order, ta, tb,
        (int)m, (int)n, (int)k,
        alpha,
        A, (int)lda,
        B, (int)ldb,
        beta,
        C, (int)ldc);

    return MATX_OK;
}

static matx_status_t ref_zgemv(matx_layout_t layout,
    int trans_a,
    size_t m,
    size_t n,
    const void* alpha,
    const void* A,
    size_t lda,
    const void* X,
    size_t ldx,
    const void* beta,
    void* C,
    size_t ldc) {
    if (!A || !X || !C || !alpha || !beta)
        return MATX_ERR_INVALID_ARG;

    if (m > INT_MAX || n > INT_MAX ||
        lda > INT_MAX || ldx > INT_MAX || ldc > INT_MAX)
        return MATX_ERR_NOT_SUPPORTED;

    const enum CBLAS_ORDER order =
        (layout == MATX_COL_MAJOR) ? CblasColMajor : CblasRowMajor;

    const enum CBLAS_TRANSPOSE ta =
        trans_a ? CblasTrans : CblasNoTrans;

    cblas_zgemv(order, ta,
        (int)m, (int)n,
        alpha,
        A, (int)lda,
        X, (int)ldx,
        beta,
        C, (int)ldc);

    return MATX_OK;
}

static matx_status_t ref_daxpy(
    size_t n,
    double alpha,
    const double* x,
    size_t lda,
    const void* y,
    size_t ldy) {
    if (!x || !y)
        return MATX_ERR_INVALID_ARG;
    if (lda != ldy)
        return MATX_ERR_INVALID_ARG;

    cblas_daxpy((int)n, alpha, x, (int)lda, y,
                (int)ldy);
    return MATX_OK;
}

static matx_status_t ref_zaxpy(
    size_t n,
    const void* alpha,
    const void* x,
    size_t lda,
    const void* y,
    size_t ldy) {
    if (!x || !y)
        return MATX_ERR_INVALID_ARG;
    if (lda != ldy)
        return MATX_ERR_INVALID_ARG;

    cblas_zaxpy((int)n, alpha, x, (int)lda, y,
        (int)ldy);
    return MATX_OK;
}

static matx_status_t ref_dgemv(matx_layout_t layout,
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
    size_t ldc)
{
    if (!A || !C || !alpha || !beta)
        return MATX_ERR_INVALID_ARG;

    if (m > INT_MAX || n > INT_MAX ||
        lda > INT_MAX || ldc > INT_MAX)
        return MATX_ERR_NOT_SUPPORTED;

    const enum CBLAS_ORDER order =
        (layout == MATX_COL_MAJOR) ? CblasColMajor : CblasRowMajor;

    const enum CBLAS_TRANSPOSE ta =
        trans_a ? CblasTrans : CblasNoTrans;

    cblas_dgemv(order, ta,
        (int)m, (int)n,
        alpha,
        A, (int)lda,
        B, (int)ldb,
        beta,
        C, (int)ldc);

    return MATX_OK;
}

static matx_status_t ref_dgeadd(matx_layout_t trans_a,
    size_t rows,
    size_t cols,
    double alpha,
    const double* A,
    size_t lda,
    double beta,
    double* B,
    size_t ldb)
{
    if (!A || !B || !alpha || !beta)
        return MATX_ERR_INVALID_ARG;

    if (rows > INT_MAX || cols > INT_MAX ||
        lda > INT_MAX || ldb > INT_MAX)
        return MATX_ERR_NOT_SUPPORTED;

    const enum CBLAS_ORDER order =
        (trans_a == MATX_COL_MAJOR) ? CblasColMajor : CblasRowMajor;

    cblas_dgeadd(
        order,
        (int)rows, (int)cols,
        alpha,
        A, lda,
        beta,
        B, ldb
    );
    return MATX_OK;
}

static matx_status_t ref_zgeadd(matx_layout_t trans_a,
    size_t rows,
    size_t cols,
    const void* alpha,
    const void* A,
    size_t lda,
    const void* beta,
    void* B,
    size_t ldb)
{
    if (!A || !B || !alpha || !beta)
        return MATX_ERR_INVALID_ARG;

    if (rows > INT_MAX || cols > INT_MAX ||
        lda > INT_MAX || ldb > INT_MAX)
        return MATX_ERR_NOT_SUPPORTED;

    const enum CBLAS_ORDER order =
        (trans_a == MATX_COL_MAJOR) ? CblasColMajor : CblasRowMajor;

    cblas_zgeadd(
        order,
        (int)rows, (int)cols,
        &alpha,
        A, lda,
        &beta,
        B, ldb
    );
    return MATX_OK;
}

matx_blas_t matx_blas_make_reference(void) {
  matx_blas_t b;
  b.kind = MATX_BLAS_BACKEND_REFERENCE;
  b.vt.dgemm = &ref_dgemm;
  b.vt.dgemv = &ref_dgemv;
  b.vt.zgemm = &ref_zgemm;
  b.vt.zgemv = &ref_zgemv;
  b.vt.daxpy = &ref_daxpy;
  b.vt.zaxpy = &ref_zaxpy;
  b.vt.dgeadd = &ref_dgeadd;
  b.vt.zgeadd = &ref_zgeadd;
  return b;
}

