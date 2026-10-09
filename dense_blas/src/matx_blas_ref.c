#if __linux__
#define _XOPEN_SOURCE 600
#endif

#include "matx/matx_dense_compute.h"
#include "matx/matx_log.h"

#include <math.h>
#include <string.h>

#if MATX_ENABLE_OPENBLAS
#include "cblas.h"
#include "lapacke.h"
#elif MATX_ENABLE_BLIS
#include "blis.h"
#include "lapacke.h"
#endif

static matx_status_t ref_dgemm(matx_layout_t layout,
                               matx_int64_t trans_a,
                               matx_int64_t trans_b,
                               matx_int64_t m,
                               matx_int64_t n,
                               matx_int64_t k,
                               matx_double alpha,
                               const matx_double* a,
                               matx_int64_t lda,
                               const matx_double* b,
                               matx_int64_t ldb,
                               matx_double beta,
                               matx_double* c,
                               matx_int64_t ldc)
{
    if (!a || !b || !c) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (layout != MATX_COL_MAJOR && layout != MATX_ROW_MAJOR) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    if (m > INT_MAX || n > INT_MAX || k > INT_MAX || lda > INT_MAX || ldb > INT_MAX
        || ldc > INT_MAX) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }

    /* Small-n direct path: bypass CBLAS overhead for very small matrices. */
    if (m <= 4 && n <= 4 && k <= 4 && trans_a == MATX_NO_TRANS
        && trans_b == MATX_NO_TRANS) {
        if (layout == MATX_COL_MAJOR) {
            for (matx_int64_t j = 0; j < n; ++j) {
                for (matx_int64_t i = 0; i < m; ++i) {
                    matx_double sum = 0.0;
                    for (matx_int64_t p = 0; p < k; ++p)
                        sum += a[i + p * lda] * b[p + j * ldb];
                    const matx_int64_t idx = i + j * ldc;
                    if (beta == 0.0)
                        c[idx] = alpha * sum;
                    else if (beta == 1.0)
                        c[idx] = alpha * sum + c[idx];
                    else
                        c[idx] = alpha * sum + beta * c[idx];
                }
            }
        } else {
            for (matx_int64_t i = 0; i < m; ++i) {
                for (matx_int64_t j = 0; j < n; ++j) {
                    matx_double sum = 0.0;
                    for (matx_int64_t p = 0; p < k; ++p)
                        sum += a[i * lda + p] * b[p * ldb + j];
                    const matx_int64_t idx = i * ldc + j;
                    if (beta == 0.0)
                        c[idx] = alpha * sum;
                    else if (beta == 1.0)
                        c[idx] = alpha * sum + c[idx];
                    else
                        c[idx] = alpha * sum + beta * c[idx];
                }
            }
        }
        return MATX_OK;
    }

    const enum CBLAS_ORDER order = (layout == MATX_COL_MAJOR) ? CblasColMajor : CblasRowMajor;

    const enum CBLAS_TRANSPOSE ta = (trans_a == MATX_NO_TRANS) ? CblasNoTrans : CblasTrans;
    const enum CBLAS_TRANSPOSE tb = (trans_b == MATX_NO_TRANS) ? CblasNoTrans : CblasTrans;
    cblas_dgemm(order, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    return MATX_OK;
}

static matx_status_t ref_zgemm(matx_layout_t layout,
                               matx_int64_t trans_a,
                               matx_int64_t trans_b,
                               matx_int64_t m,
                               matx_int64_t n,
                               matx_int64_t k,
                               const void* alpha,
                               const void* A,
                               matx_int64_t lda,
                               const void* B,
                               matx_int64_t ldb,
                               const void* beta,
                               void* C,
                               matx_int64_t ldc)
{
    if (!A || !B || !C || !alpha || !beta) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    if (m > INT_MAX || n > INT_MAX || k > INT_MAX || lda > INT_MAX || ldb > INT_MAX
        || ldc > INT_MAX) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }

    const matx_complex_d_t* a = (const matx_complex_d_t*) A;
    const matx_complex_d_t* b = (const matx_complex_d_t*) B;
    matx_complex_d_t* c = (matx_complex_d_t*) C;
    const matx_double alpha_re = ((const matx_complex_d_t*)alpha)->real;
    const matx_double alpha_im = ((const matx_complex_d_t*)alpha)->imag;
    const matx_double beta_re  = ((const matx_complex_d_t*)beta)->real;
    const matx_double beta_im  = ((const matx_complex_d_t*)beta)->imag;

    /* Small-n direct path: bypass CBLAS overhead for very small matrices. */
    if (m <= 4 && n <= 4 && k <= 4 && trans_a == MATX_NO_TRANS
        && trans_b == MATX_NO_TRANS) {
        if (layout == MATX_COL_MAJOR) {
            for (matx_int64_t j = 0; j < n; ++j) {
                for (matx_int64_t i = 0; i < m; ++i) {
                    matx_double sum_re = 0.0, sum_im = 0.0;
                    for (matx_int64_t p = 0; p < k; ++p) {
                        const matx_complex_d_t aik = a[i + p * lda];
                        const matx_complex_d_t bkj = b[p + j * ldb];
                        sum_re += aik.real * bkj.real - aik.imag * bkj.imag;
                        sum_im += aik.real * bkj.imag + aik.imag * bkj.real;
                    }
                    const matx_int64_t idx = i + j * ldc;
                    const matx_double r = alpha_re * sum_re - alpha_im * sum_im;
                    const matx_double im = alpha_re * sum_im + alpha_im * sum_re;
                    if (beta_re == 0.0 && beta_im == 0.0) {
                        c[idx].real = r;
                        c[idx].imag = im;
                    } else if (beta_re == 1.0 && beta_im == 0.0) {
                        c[idx].real = r + c[idx].real;
                        c[idx].imag = im + c[idx].imag;
                    } else {
                        const matx_double bcr = beta_re * c[idx].real - beta_im * c[idx].imag;
                        const matx_double bci = beta_re * c[idx].imag + beta_im * c[idx].real;
                        c[idx].real = r + bcr;
                        c[idx].imag = im + bci;
                    }
                }
            }
        } else {
            for (matx_int64_t i = 0; i < m; ++i) {
                for (matx_int64_t j = 0; j < n; ++j) {
                    matx_double sum_re = 0.0, sum_im = 0.0;
                    for (matx_int64_t p = 0; p < k; ++p) {
                        const matx_complex_d_t aik = a[i * lda + p];
                        const matx_complex_d_t bkj = b[p * ldb + j];
                        sum_re += aik.real * bkj.real - aik.imag * bkj.imag;
                        sum_im += aik.real * bkj.imag + aik.imag * bkj.real;
                    }
                    const matx_int64_t idx = i * ldc + j;
                    const matx_double r = alpha_re * sum_re - alpha_im * sum_im;
                    const matx_double im = alpha_re * sum_im + alpha_im * sum_re;
                    if (beta_re == 0.0 && beta_im == 0.0) {
                        c[idx].real = r;
                        c[idx].imag = im;
                    } else if (beta_re == 1.0 && beta_im == 0.0) {
                        c[idx].real = r + c[idx].real;
                        c[idx].imag = im + c[idx].imag;
                    } else {
                        const matx_double bcr = beta_re * c[idx].real - beta_im * c[idx].imag;
                        const matx_double bci = beta_re * c[idx].imag + beta_im * c[idx].real;
                        c[idx].real = r + bcr;
                        c[idx].imag = im + bci;
                    }
                }
            }
        }
        return MATX_OK;
    }

    const enum CBLAS_ORDER order = (layout == MATX_COL_MAJOR) ? CblasColMajor : CblasRowMajor;

    const enum CBLAS_TRANSPOSE ta = (trans_a == MATX_NO_TRANS) ? CblasNoTrans : CblasTrans;

    const enum CBLAS_TRANSPOSE tb = (trans_b == MATX_NO_TRANS) ? CblasNoTrans : CblasTrans;
    cblas_zgemm(order, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

    return MATX_OK;
}

static matx_status_t ref_zgemv(matx_layout_t layout,
                               matx_int64_t trans_a,
                               matx_int64_t m,
                               matx_int64_t n,
                               const void* alpha,
                               const void* A,
                               matx_int64_t lda,
                               const void* X,
                               matx_int64_t ldx,
                               const void* beta,
                               void* C,
                               matx_int64_t ldc)
{
    if (!A || !X || !C || !alpha || !beta) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    if (m > INT_MAX || n > INT_MAX || lda > INT_MAX || ldx > INT_MAX || ldc > INT_MAX) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }

    const matx_complex_d_t* a = (const matx_complex_d_t*) A;
    const matx_complex_d_t* x = (const matx_complex_d_t*) X;
    matx_complex_d_t* c = (matx_complex_d_t*) C;
    const matx_double alpha_re = ((const matx_complex_d_t*)alpha)->real;
    const matx_double alpha_im = ((const matx_complex_d_t*)alpha)->imag;
    const matx_double beta_re  = ((const matx_complex_d_t*)beta)->real;
    const matx_double beta_im  = ((const matx_complex_d_t*)beta)->imag;

    /* Small-n direct path: bypass CBLAS overhead. */
    if (m <= 16 && n <= 16 && trans_a == MATX_NO_TRANS) {
        if (layout == MATX_COL_MAJOR) {
            for (matx_int64_t i = 0; i < m; ++i) {
                matx_double sum_re = 0.0, sum_im = 0.0;
                for (matx_int64_t j = 0; j < n; ++j) {
                    const matx_complex_d_t aij = a[i + j * lda];
                    const matx_complex_d_t xj  = x[j * ldx];
                    sum_re += aij.real * xj.real - aij.imag * xj.imag;
                    sum_im += aij.real * xj.imag + aij.imag * xj.real;
                }
                const matx_double r = alpha_re * sum_re - alpha_im * sum_im;
                const matx_double im = alpha_re * sum_im + alpha_im * sum_re;
                const matx_int64_t idx = i * ldc;
                if (beta_re == 0.0 && beta_im == 0.0) {
                    c[idx].real = r;
                    c[idx].imag = im;
                } else if (beta_re == 1.0 && beta_im == 0.0) {
                    c[idx].real = r + c[idx].real;
                    c[idx].imag = im + c[idx].imag;
                } else {
                    const matx_double bcr = beta_re * c[idx].real - beta_im * c[idx].imag;
                    const matx_double bci = beta_re * c[idx].imag + beta_im * c[idx].real;
                    c[idx].real = r + bcr;
                    c[idx].imag = im + bci;
                }
            }
        } else {
            for (matx_int64_t i = 0; i < m; ++i) {
                matx_double sum_re = 0.0, sum_im = 0.0;
                for (matx_int64_t j = 0; j < n; ++j) {
                    const matx_complex_d_t aij = a[i * lda + j];
                    const matx_complex_d_t xj  = x[j * ldx];
                    sum_re += aij.real * xj.real - aij.imag * xj.imag;
                    sum_im += aij.real * xj.imag + aij.imag * xj.real;
                }
                const matx_double r = alpha_re * sum_re - alpha_im * sum_im;
                const matx_double im = alpha_re * sum_im + alpha_im * sum_re;
                const matx_int64_t idx = i * ldc;
                if (beta_re == 0.0 && beta_im == 0.0) {
                    c[idx].real = r;
                    c[idx].imag = im;
                } else if (beta_re == 1.0 && beta_im == 0.0) {
                    c[idx].real = r + c[idx].real;
                    c[idx].imag = im + c[idx].imag;
                } else {
                    const matx_double bcr = beta_re * c[idx].real - beta_im * c[idx].imag;
                    const matx_double bci = beta_re * c[idx].imag + beta_im * c[idx].real;
                    c[idx].real = r + bcr;
                    c[idx].imag = im + bci;
                }
            }
        }
        return MATX_OK;
    }

    const enum CBLAS_ORDER order = (layout == MATX_COL_MAJOR) ? CblasColMajor : CblasRowMajor;

    const enum CBLAS_TRANSPOSE ta = (trans_a == MATX_NO_TRANS) ? CblasNoTrans : CblasTrans;
    cblas_zgemv(order, ta, m, n, alpha, A, lda, X, ldx, beta, C, ldc);

    return MATX_OK;
}

static matx_status_t ref_dgemv(matx_layout_t layout,
                               matx_int64_t trans_a,
                               matx_int64_t m,
                               matx_int64_t n,
                               matx_double alpha,
                               const matx_double* A,
                               matx_int64_t lda,
                               matx_double* B,
                               matx_int64_t ldb,
                               matx_double beta,
                               matx_double* C,
                               matx_int64_t ldc)
{
    if (!A || !B || !C) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    if (m > INT_MAX || n > INT_MAX || lda > INT_MAX || ldb > INT_MAX || ldc > INT_MAX) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }

    /* Small-n direct path: bypass CBLAS overhead. */
    if (m <= 16 && n <= 16 && trans_a == MATX_NO_TRANS) {
        if (layout == MATX_COL_MAJOR) {
            for (matx_int64_t i = 0; i < m; ++i) {
                matx_double sum = 0.0;
                for (matx_int64_t j = 0; j < n; ++j)
                    sum += A[i + j * lda] * B[j * ldb];
                const matx_int64_t idx = i * ldc;
                if (beta == 0.0)
                    C[idx] = alpha * sum;
                else if (beta == 1.0)
                    C[idx] = alpha * sum + C[idx];
                else
                    C[idx] = alpha * sum + beta * C[idx];
            }
        } else {
            for (matx_int64_t i = 0; i < m; ++i) {
                matx_double sum = 0.0;
                for (matx_int64_t j = 0; j < n; ++j)
                    sum += A[i * lda + j] * B[j * ldb];
                const matx_int64_t idx = i * ldc;
                if (beta == 0.0)
                    C[idx] = alpha * sum;
                else if (beta == 1.0)
                    C[idx] = alpha * sum + C[idx];
                else
                    C[idx] = alpha * sum + beta * C[idx];
            }
        }
        return MATX_OK;
    }

    const enum CBLAS_ORDER order = (layout == MATX_COL_MAJOR) ? CblasColMajor : CblasRowMajor;

    const enum CBLAS_TRANSPOSE ta = (trans_a == MATX_NO_TRANS) ? CblasNoTrans : CblasTrans;

    cblas_dgemv(order, ta, m, n, alpha, A, lda, B, ldb, beta, C, ldc);

    return MATX_OK;
}

static matx_status_t ref_dgeadd(matx_layout_t trans_a,
                                matx_int64_t rows,
                                matx_int64_t cols,
                                matx_double alpha,
                                const matx_double* A,
                                matx_int64_t lda,
                                matx_double beta,
                                matx_double* B,
                                matx_int64_t ldb)
{
    if (!A || !B) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    if (rows > INT_MAX || cols > INT_MAX || lda > INT_MAX || ldb > INT_MAX) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }

#if MATX_ENABLE_OPENBLAS
    const enum CBLAS_ORDER order = (trans_a == MATX_COL_MAJOR) ? CblasColMajor : CblasRowMajor;

    cblas_dgeadd(order, rows, cols, alpha, A, lda, beta, B, ldb);
#elif MATX_ENABLE_BLIS

    if (trans_a == MATX_ROW_MAJOR) {
        if (lda == cols && ldb == cols) {
            const matx_int64_t len = rows * cols;
            for (matx_int64_t i = 0; i < len; ++i)
                B[i] = alpha * A[i] + beta * B[i];
            return MATX_OK;
        }
    } else {
        if (lda == rows && ldb == rows) {
            const matx_int64_t len = rows * cols;
            for (matx_int64_t i = 0; i < len; ++i)
                B[i] = alpha * A[i] + beta * B[i];
            return MATX_OK;
        }
    }

    for (matx_int64_t j = 0; j < cols; ++j) {
        const matx_int64_t a_off = j * lda, b_off = j * ldb;
        for (matx_int64_t i = 0; i < rows; ++i)
            B[b_off + i] = alpha * A[a_off + i] + beta * B[b_off + i];
    }
#else
    /* Pure-C fallback: B = alpha * A + beta * B */
    for (matx_int64_t j = 0; j < cols; ++j) {
        for (matx_int64_t i = 0; i < rows; ++i) {
            B[i + j * ldb] = alpha * A[i + j * lda] + beta * B[i + j * ldb];
        }
    }
#endif
    return MATX_OK;
}

static matx_status_t ref_zgeadd(matx_layout_t trans_a,
                                matx_int64_t rows,
                                matx_int64_t cols,
                                const void* alpha,
                                const void* A,
                                matx_int64_t lda,
                                const void* beta,
                                void* B,
                                matx_int64_t ldb)
{
    if (!A || !B) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    if (rows > INT_MAX || cols > INT_MAX || lda > INT_MAX || ldb > INT_MAX) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
#if MATX_ENABLE_OPENBLAS
    const enum CBLAS_ORDER order = (trans_a == MATX_COL_MAJOR) ? CblasColMajor : CblasRowMajor;

    cblas_zgeadd(order, rows, cols, alpha, A, lda, beta, B, ldb);
#elif MATX_ENABLE_BLIS

    const matx_complex_d_t* a = (const matx_complex_d_t*) alpha;
    const matx_complex_d_t* b = (const matx_complex_d_t*) beta;
    const matx_complex_d_t* Ad = (const matx_complex_d_t*) A;
    matx_complex_d_t* Bd = (matx_complex_d_t*) B;

    if (trans_a == MATX_ROW_MAJOR) {
        if (lda == cols && ldb == cols) {
            const matx_int64_t len = rows * cols;
            for (matx_int64_t i = 0; i < len; ++i) {
                Bd[i].real = a->real * Ad[i].real - a->imag * Ad[i].imag
                           + b->real * Bd[i].real - b->imag * Bd[i].imag;
                Bd[i].imag = a->real * Ad[i].imag + a->imag * Ad[i].real
                           + b->real * Bd[i].imag + b->imag * Bd[i].real;
            }
            return MATX_OK;
        }
    } else {
        if (lda == rows && ldb == rows) {
            const matx_int64_t len = rows * cols;
            for (matx_int64_t i = 0; i < len; ++i) {
                Bd[i].real = a->real * Ad[i].real - a->imag * Ad[i].imag
                           + b->real * Bd[i].real - b->imag * Bd[i].imag;
                Bd[i].imag = a->real * Ad[i].imag + a->imag * Ad[i].real
                           + b->real * Bd[i].imag + b->imag * Bd[i].real;
            }
            return MATX_OK;
        }
    }

    for (matx_int64_t j = 0; j < cols; ++j) {
        for (matx_int64_t i = 0; i < rows; ++i) {
            const matx_int64_t aidx = i + j * lda, bidx = i + j * ldb;
            Bd[bidx].real = a->real * Ad[aidx].real - a->imag * Ad[aidx].imag
                          + b->real * Bd[bidx].real - b->imag * Bd[bidx].imag;
            Bd[bidx].imag = a->real * Ad[aidx].imag + a->imag * Ad[aidx].real
                          + b->real * Bd[bidx].imag + b->imag * Bd[bidx].real;
        }
    }
#else
    /* Pure-C fallback: B = alpha * A + beta * B */
    {
        const matx_complex_d_t* a = (const matx_complex_d_t*) alpha;
        const matx_complex_d_t* b = (const matx_complex_d_t*) beta;
        const matx_complex_d_t* Ad = (const matx_complex_d_t*) A;
        matx_complex_d_t* Bd = (matx_complex_d_t*) B;
        for (matx_int64_t j = 0; j < cols; ++j) {
            for (matx_int64_t i = 0; i < rows; ++i) {
                matx_double ar = a->real, ai = a->imag;
                matx_double br = b->real, bi = b->imag;
                matx_double Ar = Ad[i + j * lda].real;
                matx_double Ai = Ad[i + j * lda].imag;
                matx_double Br = Bd[i + j * ldb].real;
                matx_double Bi = Bd[i + j * ldb].imag;
                Bd[i + j * ldb].real = ar * Ar - ai * Ai + br * Br - bi * Bi;
                Bd[i + j * ldb].imag = ar * Ai + ai * Ar + br * Bi + bi * Br;
            }
        }
    }
#endif
    return MATX_OK;
}

static matx_status_t ref_inv_dense_d_i8(matx_layout_t layout,
                                        matx_int64_t rows,
                                        matx_int64_t cols,
                                        const matx_double* A,
                                        matx_int64_t lda,
                                        matx_double* out_Ainv,
                                        matx_int64_t ldout)
{
    if (!A || !out_Ainv) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (rows != cols) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (layout != MATX_COL_MAJOR) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }

    /* Small-n explicit formula paths avoid LAPACK call overhead. */
    if (rows == 2) {
        const matx_double a = A[0 + 0 * lda];
        const matx_double b = A[0 + 1 * lda];
        const matx_double c = A[1 + 0 * lda];
        const matx_double d = A[1 + 1 * lda];
        const matx_double det = a * d - b * c;
        if (det == 0.0) {
            MATX_ERROR("%s: singular 2x2 matrix", __func__);
            return MATX_ERR_INTERNAL;
        }
        const matx_double inv_det = 1.0 / det;
        out_Ainv[0 + 0 * ldout] =  d * inv_det;
        out_Ainv[0 + 1 * ldout] = -b * inv_det;
        out_Ainv[1 + 0 * ldout] = -c * inv_det;
        out_Ainv[1 + 1 * ldout] =  a * inv_det;
        return MATX_OK;
    }
    if (rows == 3) {
        const matx_double a11 = A[0 + 0 * lda], a12 = A[0 + 1 * lda], a13 = A[0 + 2 * lda];
        const matx_double a21 = A[1 + 0 * lda], a22 = A[1 + 1 * lda], a23 = A[1 + 2 * lda];
        const matx_double a31 = A[2 + 0 * lda], a32 = A[2 + 1 * lda], a33 = A[2 + 2 * lda];
        /* Cofactors */
        const matx_double c11 = a22 * a33 - a23 * a32;
        const matx_double c12 = a13 * a32 - a12 * a33;
        const matx_double c13 = a12 * a23 - a13 * a22;
        const matx_double c21 = a23 * a31 - a21 * a33;
        const matx_double c22 = a11 * a33 - a13 * a31;
        const matx_double c23 = a13 * a21 - a11 * a23;
        const matx_double c31 = a21 * a32 - a22 * a31;
        const matx_double c32 = a12 * a31 - a11 * a32;
        const matx_double c33 = a11 * a22 - a12 * a21;
        const matx_double det = a11 * c11 + a12 * c21 + a13 * c31;
        if (det == 0.0) {
            MATX_ERROR("%s: singular 3x3 matrix", __func__);
            return MATX_ERR_INTERNAL;
        }
        const matx_double inv_det = 1.0 / det;
        out_Ainv[0 + 0 * ldout] = c11 * inv_det; out_Ainv[0 + 1 * ldout] = c21 * inv_det; out_Ainv[0 + 2 * ldout] = c31 * inv_det;
        out_Ainv[1 + 0 * ldout] = c12 * inv_det; out_Ainv[1 + 1 * ldout] = c22 * inv_det; out_Ainv[1 + 2 * ldout] = c32 * inv_det;
        out_Ainv[2 + 0 * ldout] = c13 * inv_det; out_Ainv[2 + 1 * ldout] = c23 * inv_det; out_Ainv[2 + 2 * ldout] = c33 * inv_det;
        return MATX_OK;
    }
    if (rows == 4) {
        const matx_double a11 = A[0 + 0 * lda], a12 = A[0 + 1 * lda], a13 = A[0 + 2 * lda], a14 = A[0 + 3 * lda];
        const matx_double a21 = A[1 + 0 * lda], a22 = A[1 + 1 * lda], a23 = A[1 + 2 * lda], a24 = A[1 + 3 * lda];
        const matx_double a31 = A[2 + 0 * lda], a32 = A[2 + 1 * lda], a33 = A[2 + 2 * lda], a34 = A[2 + 3 * lda];
        const matx_double a41 = A[3 + 0 * lda], a42 = A[3 + 1 * lda], a43 = A[3 + 2 * lda], a44 = A[3 + 3 * lda];
        /* 3x3 minors of row 1 */
        const matx_double m11 = a22*a33*a44 + a23*a34*a42 + a24*a32*a43 - a22*a34*a43 - a23*a32*a44 - a24*a33*a42;
        const matx_double m12 = a21*a34*a43 + a23*a31*a44 + a24*a33*a41 - a21*a33*a44 - a23*a34*a41 - a24*a31*a43;
        const matx_double m13 = a21*a32*a44 + a22*a34*a41 + a24*a31*a42 - a21*a34*a42 - a22*a31*a44 - a24*a32*a41;
        const matx_double m14 = a21*a33*a42 + a22*a31*a43 + a23*a32*a41 - a21*a32*a43 - a22*a33*a41 - a23*a31*a42;
        const matx_double det = a11*m11 - a12*m12 + a13*m13 - a14*m14;
        if (det == 0.0) {
            MATX_ERROR("%s: singular 4x4 matrix", __func__);
            return MATX_ERR_INTERNAL;
        }
        const matx_double inv_det = 1.0 / det;
        out_Ainv[0 + 0 * ldout] =  m11 * inv_det;
        out_Ainv[1 + 0 * ldout] = -m12 * inv_det;
        out_Ainv[2 + 0 * ldout] =  m13 * inv_det;
        out_Ainv[3 + 0 * ldout] = -m14 * inv_det;
        /* 3x3 minors of row 2 */
        const matx_double m21 = a12*a34*a43 + a13*a32*a44 + a14*a33*a42 - a12*a33*a44 - a13*a34*a42 - a14*a32*a43;
        const matx_double m22 = a11*a33*a44 + a13*a34*a41 + a14*a31*a43 - a11*a34*a43 - a13*a31*a44 - a14*a33*a41;
        const matx_double m23 = a11*a34*a42 + a12*a31*a44 + a14*a32*a41 - a11*a32*a44 - a12*a34*a41 - a14*a31*a42;
        const matx_double m24 = a11*a32*a43 + a12*a33*a41 + a13*a31*a42 - a11*a33*a42 - a12*a31*a43 - a13*a32*a41;
        out_Ainv[0 + 1 * ldout] = -m21 * inv_det;
        out_Ainv[1 + 1 * ldout] =  m22 * inv_det;
        out_Ainv[2 + 1 * ldout] = -m23 * inv_det;
        out_Ainv[3 + 1 * ldout] =  m24 * inv_det;
        /* 3x3 minors of row 3 */
        const matx_double m31 = a12*a23*a44 + a13*a24*a42 + a14*a22*a43 - a12*a24*a43 - a13*a22*a44 - a14*a23*a42;
        const matx_double m32 = a11*a24*a43 + a13*a21*a44 + a14*a23*a41 - a11*a23*a44 - a13*a24*a41 - a14*a21*a43;
        const matx_double m33 = a11*a22*a44 + a12*a24*a41 + a14*a21*a42 - a11*a24*a42 - a12*a21*a44 - a14*a22*a41;
        const matx_double m34 = a11*a23*a42 + a12*a21*a43 + a13*a22*a41 - a11*a22*a43 - a12*a23*a41 - a13*a21*a42;
        out_Ainv[0 + 2 * ldout] =  m31 * inv_det;
        out_Ainv[1 + 2 * ldout] = -m32 * inv_det;
        out_Ainv[2 + 2 * ldout] =  m33 * inv_det;
        out_Ainv[3 + 2 * ldout] = -m34 * inv_det;
        /* 3x3 minors of row 4 */
        const matx_double m41 = a12*a24*a33 + a13*a22*a34 + a14*a23*a32 - a12*a23*a34 - a13*a24*a32 - a14*a22*a33;
        const matx_double m42 = a11*a23*a34 + a13*a24*a31 + a14*a21*a33 - a11*a24*a33 - a13*a21*a34 - a14*a23*a31;
        const matx_double m43 = a11*a24*a32 + a12*a21*a34 + a14*a22*a31 - a11*a22*a34 - a12*a24*a31 - a14*a21*a32;
        const matx_double m44 = a11*a22*a33 + a12*a23*a31 + a13*a21*a32 - a11*a23*a32 - a12*a21*a33 - a13*a22*a31;
        out_Ainv[0 + 3 * ldout] = -m41 * inv_det;
        out_Ainv[1 + 3 * ldout] =  m42 * inv_det;
        out_Ainv[2 + 3 * ldout] = -m43 * inv_det;
        out_Ainv[3 + 3 * ldout] =  m44 * inv_det;
        return MATX_OK;
    }

    /* n >= 4: use LAPACK (standard path). */
    // Copy with stride-awareness
    for (matx_int64_t j = 0; j < cols; ++j)
        for (matx_int64_t i = 0; i < rows; ++i) {
            out_Ainv[i + j * ldout] = A[i + j * lda];
        }

    matx_int64_t* piv = (matx_int64_t*) malloc(rows * sizeof(matx_int64_t));
    if (!piv) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    matx_int64_t status = LAPACKE_dgetrf(layout == MATX_COL_MAJOR ? LAPACK_COL_MAJOR
                                                                  : LAPACK_ROW_MAJOR,
                                         rows,
                                         cols,
                                         out_Ainv,
                                         ldout,
                                         piv);
    if (status != 0) {
        MATX_ERROR("LAPACKE_dgetrf error:%d", status);
        free(piv);
        return MATX_ERR_INTERNAL;
    }

    status = LAPACKE_dgetri(layout == MATX_COL_MAJOR ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR,
                            rows,
                            out_Ainv,
                            ldout,
                            piv);
    if (status != 0) {
        MATX_ERROR("LAPACKE_dgetri error:%d", status);
        free(piv);
        return MATX_ERR_INTERNAL;
    }
    free(piv);
    return MATX_OK;
}

static matx_status_t ref_inv_dense_z_i8(
    matx_layout_t layout, matx_int64_t rows, matx_int64_t cols,
    const void* A, matx_int64_t lda, void* out_Ainv, matx_int64_t ldout)
{
    if (!A || !out_Ainv) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (rows != cols) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (layout != MATX_COL_MAJOR) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }

    matx_complex_d_t* out = (matx_complex_d_t*) out_Ainv;
    const matx_complex_d_t* in = (const matx_complex_d_t*) A;

    /* Small-n explicit formula paths. */
    if (rows == 2) {
        const matx_complex_d_t a = in[0 + 0 * lda];
        const matx_complex_d_t b = in[0 + 1 * lda];
        const matx_complex_d_t c = in[1 + 0 * lda];
        const matx_complex_d_t d = in[1 + 1 * lda];
        /* det = a*d - b*c */
        const matx_double det_re = a.real * d.real - a.imag * d.imag
                                 - (b.real * c.real - b.imag * c.imag);
        const matx_double det_im = a.real * d.imag + a.imag * d.real
                                 - (b.real * c.imag + b.imag * c.real);
        const matx_double den = det_re * det_re + det_im * det_im;
        if (den == 0.0) {
            MATX_ERROR("%s: singular 2x2 complex matrix", __func__);
            return MATX_ERR_INTERNAL;
        }
        const matx_double inv_re =  det_re / den;
        const matx_double inv_im = -det_im / den;
        /* out = adjugate * inv_det */
        out[0 + 0 * ldout].real =  d.real * inv_re - d.imag * inv_im;
        out[0 + 0 * ldout].imag =  d.real * inv_im + d.imag * inv_re;
        out[0 + 1 * ldout].real = -b.real * inv_re + b.imag * inv_im;
        out[0 + 1 * ldout].imag = -b.real * inv_im - b.imag * inv_re;
        out[1 + 0 * ldout].real = -c.real * inv_re + c.imag * inv_im;
        out[1 + 0 * ldout].imag = -c.real * inv_im - c.imag * inv_re;
        out[1 + 1 * ldout].real =  a.real * inv_re - a.imag * inv_im;
        out[1 + 1 * ldout].imag =  a.real * inv_im + a.imag * inv_re;
        return MATX_OK;
    }

    if (rows == 3) {
        /* 3x3 complex inverse via cofactor expansion. */
        #define Z_MUL(r, a, b) do { \
            const matx_double ar = (a).real, ai = (a).imag, br = (b).real, bi = (b).imag; \
            (r).real = ar * br - ai * bi; (r).imag = ar * bi + ai * br; \
        } while (0)
        #define Z_ADD(r, a) do { (r).real += (a).real; (r).imag += (a).imag; } while (0)
        #define Z_SUB(r, a) do { (r).real -= (a).real; (r).imag -= (a).imag; } while (0)
        matx_complex_d_t t1, t2;
        /* Co-det approach: compute cofactors as complex numbers directly */
        /* c11 = a22*a33 - a23*a32 */
        Z_MUL(t1, in[(1+1*lda)], in[(2+2*lda)]); Z_MUL(t2, in[(1+2*lda)], in[(2+1*lda)]);
        matx_complex_d_t c11 = t1; Z_SUB(c11, t2);
        /* c21 = a23*a31 - a21*a33 */
        Z_MUL(t1, in[(1+2*lda)], in[(2+0*lda)]); Z_MUL(t2, in[(1+0*lda)], in[(2+2*lda)]);
        matx_complex_d_t c21 = t1; Z_SUB(c21, t2);
        /* c31 = a21*a32 - a22*a31 */
        Z_MUL(t1, in[(1+0*lda)], in[(2+1*lda)]); Z_MUL(t2, in[(1+1*lda)], in[(2+0*lda)]);
        matx_complex_d_t c31 = t1; Z_SUB(c31, t2);
        /* det = a11*c11 + a12*c21 + a13*c31 */
        Z_MUL(t1, in[(0+0*lda)], c11); Z_MUL(t2, in[(0+1*lda)], c21);
        matx_complex_d_t det = t1; Z_ADD(det, t2);
        Z_MUL(t1, in[(0+2*lda)], c31); Z_ADD(det, t1);
        const matx_double den = det.real * det.real + det.imag * det.imag;
        if (den == 0.0) { MATX_ERROR("%s: singular 3x3 complex matrix", __func__); return MATX_ERR_INTERNAL; }
        const matx_double inv_re =  det.real / den;
        const matx_double inv_im = -det.imag / den;
        #define Z_INVSCALE(r, s) do { \
            const matx_double sr = (s).real, si = (s).imag; \
            (r).real = sr * inv_re - si * inv_im; (r).imag = sr * inv_im + si * inv_re; \
        } while (0)
        /* Remaining cofactors */
        /* c12 = a13*a32 - a12*a33 */
        Z_MUL(t1, in[(0+2*lda)], in[(2+1*lda)]); Z_MUL(t2, in[(0+1*lda)], in[(2+2*lda)]);
        matx_complex_d_t c12 = t1; Z_SUB(c12, t2);
        /* c22 = a11*a33 - a13*a31 */
        Z_MUL(t1, in[(0+0*lda)], in[(2+2*lda)]); Z_MUL(t2, in[(0+2*lda)], in[(2+0*lda)]);
        matx_complex_d_t c22 = t1; Z_SUB(c22, t2);
        /* c32 = a12*a31 - a11*a32 */
        Z_MUL(t1, in[(0+1*lda)], in[(2+0*lda)]); Z_MUL(t2, in[(0+0*lda)], in[(2+1*lda)]);
        matx_complex_d_t c32 = t1; Z_SUB(c32, t2);
        /* c13 = a12*a23 - a13*a22 */
        Z_MUL(t1, in[(0+1*lda)], in[(1+2*lda)]); Z_MUL(t2, in[(0+2*lda)], in[(1+1*lda)]);
        matx_complex_d_t c13 = t1; Z_SUB(c13, t2);
        /* c23 = a13*a21 - a11*a23 */
        Z_MUL(t1, in[(0+2*lda)], in[(1+0*lda)]); Z_MUL(t2, in[(0+0*lda)], in[(1+2*lda)]);
        matx_complex_d_t c23 = t1; Z_SUB(c23, t2);
        /* c33 = a11*a22 - a12*a21 */
        Z_MUL(t1, in[(0+0*lda)], in[(1+1*lda)]); Z_MUL(t2, in[(0+1*lda)], in[(1+0*lda)]);
        matx_complex_d_t c33 = t1; Z_SUB(c33, t2);
        /* Output: cofactor transpose / det */
        #define Z_STORE(r, c, s) Z_INVSCALE(out[(r)+(c)*ldout], s)
        Z_STORE(0, 0, c11); Z_STORE(1, 0, c12); Z_STORE(2, 0, c13);
        Z_STORE(0, 1, c21); Z_STORE(1, 1, c22); Z_STORE(2, 1, c23);
        Z_STORE(0, 2, c31); Z_STORE(1, 2, c32); Z_STORE(2, 2, c33);
        #undef Z_MUL
        #undef Z_ADD
        #undef Z_SUB
        #undef Z_INVSCALE
        #undef Z_STORE
        return MATX_OK;
    }

    /* n >= 4: use explicit formula for 4x4, LAPACK for n > 4. */
    if (rows == 4) {
        matx_complex_d_t t1, t2, t3, acc;
        #define Z_MUL(r, a, b) do { \
            const matx_double ar=(a).real,ai=(a).imag,br=(b).real,bi=(b).imag; \
            (r).real=ar*br-ai*bi; (r).imag=ar*bi+ai*br; } while(0)
        #define Z_LOAD(r, i, j) r = in[(i)+(j)*lda]
        #define Z_DET3(d, x1,y1, x2,y2, x3,y3) do { \
            matx_complex_d_t a_,b_,c_,d1,d2,d3; \
            Z_LOAD(a_,x1,y1); Z_LOAD(b_,x2,y2); Z_LOAD(c_,x3,y3); Z_MUL(d1,a_,b_); Z_MUL(d1,d1,c_); \
            Z_LOAD(a_,x1,y2); Z_LOAD(b_,x2,y3); Z_LOAD(c_,x3,y1); Z_MUL(d2,a_,b_); Z_MUL(d2,d2,c_); \
            Z_LOAD(a_,x1,y3); Z_LOAD(b_,x2,y1); Z_LOAD(c_,x3,y2); Z_MUL(d3,a_,b_); Z_MUL(d3,d3,c_); \
            d.real = d1.real+d2.real+d3.real; d.imag = d1.imag+d2.imag+d3.imag; \
            Z_LOAD(a_,x1,y3); Z_LOAD(b_,x2,y2); Z_LOAD(c_,x3,y1); Z_MUL(d1,a_,b_); Z_MUL(d1,d1,c_); \
            Z_LOAD(a_,x1,y1); Z_LOAD(b_,x2,y3); Z_LOAD(c_,x3,y2); Z_MUL(d2,a_,b_); Z_MUL(d2,d2,c_); \
            Z_LOAD(a_,x1,y2); Z_LOAD(b_,x2,y1); Z_LOAD(c_,x3,y3); Z_MUL(d3,a_,b_); Z_MUL(d3,d3,c_); \
            d.real = d.real - d1.real - d2.real - d3.real; d.imag = d.imag - d1.imag - d2.imag - d3.imag; \
        } while(0)
        /* Row 1 minors */
        matx_complex_d_t m11,m12,m13,m14;
        Z_DET3(m11, 1,1, 2,2, 3,3);
        Z_DET3(m12, 1,0, 2,2, 3,3);
        Z_DET3(m13, 1,0, 2,1, 3,3);
        Z_DET3(m14, 1,0, 2,1, 3,2);
        /* det = a11*m11 - a12*m12 + a13*m13 - a14*m14 */
        matx_complex_d_t det;
        Z_MUL(t1, in[0], m11);
        Z_MUL(t2, in[(0+1*lda)], m12);
        Z_MUL(t3, in[(0+2*lda)], m13);
        det.real = t1.real - t2.real + t3.real; det.imag = t1.imag - t2.imag + t3.imag;
        Z_MUL(t1, in[(0+3*lda)], m14);
        det.real -= t1.real; det.imag -= t1.imag;
        const matx_double den = det.real*det.real + det.imag*det.imag;
        if (den == 0.0) { MATX_ERROR("%s: singular 4x4 complex matrix", __func__); return MATX_ERR_INTERNAL; }
        const matx_double inv_re =  det.real/den;
        const matx_double inv_im = -det.imag/den;
        #define Z_SCALE(r, s) do { \
            const matx_double sr=(s).real,si=(s).imag; \
            (r).real=sr*inv_re-si*inv_im; (r).imag=sr*inv_im+si*inv_re; } while(0)
        #define Z_STO(r, c, s) do { matx_complex_d_t _tmp; Z_SCALE(_tmp, s); out[(r)+(c)*ldout]=_tmp; } while(0)
        Z_STO(0,0, m11); Z_SCALE(acc,m12); acc.real=-acc.real;acc.imag=-acc.imag; out[(1)+(0)*ldout]=acc;
        Z_STO(2,0, m13); Z_SCALE(acc,m14); acc.real=-acc.real;acc.imag=-acc.imag; out[(3)+(0)*ldout]=acc;
        /* Row 2 minors */
        matx_complex_d_t m21,m22,m23,m24;
        Z_DET3(m21, 0,1, 2,2, 3,3);
        Z_DET3(m22, 0,0, 2,2, 3,3);
        Z_DET3(m23, 0,0, 2,1, 3,3);
        Z_DET3(m24, 0,0, 2,1, 3,2);
        Z_SCALE(acc,m21); acc.real=-acc.real;acc.imag=-acc.imag; out[(0)+(1)*ldout]=acc;
        Z_STO(1,1, m22); Z_SCALE(acc,m23); acc.real=-acc.real;acc.imag=-acc.imag; out[(2)+(1)*ldout]=acc;
        Z_STO(3,1, m24);
        /* Row 3 minors */
        matx_complex_d_t m31,m32,m33,m34;
        Z_DET3(m31, 0,1, 1,2, 3,3);
        Z_DET3(m32, 0,0, 1,2, 3,3);
        Z_DET3(m33, 0,0, 1,1, 3,3);
        Z_DET3(m34, 0,0, 1,1, 3,2);
        Z_STO(0,2, m31); Z_SCALE(acc,m32); acc.real=-acc.real;acc.imag=-acc.imag; out[(1)+(2)*ldout]=acc;
        Z_STO(2,2, m33); Z_SCALE(acc,m34); acc.real=-acc.real;acc.imag=-acc.imag; out[(3)+(2)*ldout]=acc;
        /* Row 4 minors */
        matx_complex_d_t m41,m42,m43,m44;
        Z_DET3(m41, 0,1, 1,2, 2,3);
        Z_DET3(m42, 0,0, 1,2, 2,3);
        Z_DET3(m43, 0,0, 1,1, 2,3);
        Z_DET3(m44, 0,0, 1,1, 2,2);
        Z_SCALE(acc,m41); acc.real=-acc.real;acc.imag=-acc.imag; out[(0)+(3)*ldout]=acc;
        Z_STO(1,3, m42); Z_SCALE(acc,m43); acc.real=-acc.real;acc.imag=-acc.imag; out[(2)+(3)*ldout]=acc;
        Z_STO(3,3, m44);
        #undef Z_MUL
        #undef Z_LOAD
        #undef Z_DET3
        #undef Z_SCALE
        #undef Z_STO
        return MATX_OK;
    }

    /* n > 4: use LAPACK. */
    for (matx_int64_t j = 0; j < cols; ++j)
        for (matx_int64_t i = 0; i < rows; ++i) {
            out[i + j * ldout] = in[i + j * lda];
        }

    matx_int64_t* piv = (matx_int64_t*) malloc(rows * sizeof(matx_int64_t));
    if (!piv) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    matx_int64_t status = LAPACKE_zgetrf(layout == MATX_COL_MAJOR ? LAPACK_COL_MAJOR
                                                                  : LAPACK_ROW_MAJOR,
                                         rows,
                                         cols,
                                         out_Ainv,
                                         ldout,
                                         piv);
    if (status != 0) {
        MATX_ERROR("LAPACKE_zgetrf error:%d", status);
        free(piv);
        return MATX_ERR_INTERNAL;
    }

    status = LAPACKE_zgetri(layout == MATX_COL_MAJOR ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR,
                            rows,
                            out_Ainv,
                            ldout,
                            piv);
    if (status != 0) {
        MATX_ERROR("LAPACKE_zgetri error:%d", status);
        free(piv);
        return MATX_ERR_INTERNAL;
    }
    free(piv);
    return MATX_OK;
}

// ---- Matrix exponential (scaling-and-squaring, Pade(6,6)) ----

static matx_status_t ref_expm_dense_d_i8(matx_layout_t layout,
                                         matx_int64_t n,
                                         const matx_double* A,
                                         matx_int64_t lda,
                                         matx_double* out,
                                         matx_int64_t ldout)
{
    if (!A || !out || n <= 0) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (layout != MATX_COL_MAJOR) {
        MATX_ERROR("%s: only column-major supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    if (n > INT_MAX) {
        MATX_ERROR("%s: matrix too large", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }

    const matx_int64_t total = n * n;
    const size_t sz = (size_t) total * sizeof(matx_double);
    const int lapack_layout = LAPACK_COL_MAJOR;
    const matx_int64_t n_int = n;
    const matx_int64_t work_lda = n; /* leading dim for contiguous working buffers */

    /* 1. Compute ||A||_1 */
    matx_double anorm = LAPACKE_dlange(lapack_layout, '1', n_int, n_int, A, lda);
    if (anorm == 0.0) {
        /* exp(0) = I */
        for (matx_int64_t j = 0; j < n; ++j)
            for (matx_int64_t i = 0; i < n; ++i)
                out[i + j * ldout] = (i == j) ? 1.0 : 0.0;
        return MATX_OK;
    }

    /* 2. Determine scaling factor */
    const matx_double theta6 = 3.014350724601486; /* threshold for Pade degree 6 */
    matx_int64_t s = 0;
    matx_double scaled_norm = anorm;
    while (scaled_norm > theta6) {
        scaled_norm /= 2.0;
        ++s;
    }

    /* 3. Allocate working buffers: As, B1, B2 (power ping-pong), N, D = 5 buffers */
    matx_double* As = (matx_double*) malloc(sz);
    matx_double* B1 = (matx_double*) malloc(sz);
    matx_double* B2 = (matx_double*) malloc(sz);
    matx_double* N_mat = (matx_double*) malloc(sz);
    matx_double* D_mat = (matx_double*) malloc(sz);
    if (!As || !B1 || !B2 || !N_mat || !D_mat) {
        MATX_ERROR("%s: out of memory", __func__);
        free(As);
        free(B1);
        free(B2);
        free(N_mat);
        free(D_mat);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    /* 4. Scale A: As = A / 2^s */
    matx_double scale = 1.0;
    for (matx_int64_t j = 0; j < s; ++j)
        scale /= 2.0;
    for (matx_int64_t j = 0; j < n; ++j)
        for (matx_int64_t i = 0; i < n; ++i)
            As[i + j * work_lda] = A[i + j * lda] * scale;

    /* 5. Pade(6,6) with interleaved accumulation — only 3 power buffers needed */
    {
        const matx_double c0 = 1.0;
        const matx_double c1 = 1.0 / 2.0;
        const matx_double c2 = 5.0 / 44.0;
        const matx_double c3 = 1.0 / 66.0;
        const matx_double c4 = 1.0 / 792.0;
        const matx_double c5 = 1.0 / 15840.0;
        const matx_double c6 = 1.0 / 665280.0;

        /* Init N and D with c0*I, then accumulate c1*As */
        for (matx_int64_t i = 0; i < total; ++i) {
            matx_double eye = (i % (n + 1) == 0) ? c0 : 0.0;
            matx_double t1 = c1 * As[i];
            N_mat[i] = eye + t1;
            D_mat[i] = eye - t1;
        }

        /* A2 = As * As → B1, accumulate +c2 */
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    n_int, n_int, n_int, 1.0, As, work_lda, As, work_lda, 0.0, B1, work_lda);
        for (matx_int64_t i = 0; i < total; ++i) {
            matx_double t = c2 * B1[i];
            N_mat[i] += t;
            D_mat[i] += t;
        }

        /* A3 = A2 * As → B2, accumulate +c3 (N) / -c3 (D) */
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    n_int, n_int, n_int, 1.0, B1, work_lda, As, work_lda, 0.0, B2, work_lda);
        for (matx_int64_t i = 0; i < total; ++i) {
            matx_double t = c3 * B2[i];
            N_mat[i] += t;
            D_mat[i] -= t;
        }

        /* A4 = A3 * As → B1 (overwrite A2), accumulate +c4 */
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    n_int, n_int, n_int, 1.0, B2, work_lda, As, work_lda, 0.0, B1, work_lda);
        for (matx_int64_t i = 0; i < total; ++i) {
            matx_double t = c4 * B1[i];
            N_mat[i] += t;
            D_mat[i] += t;
        }

        /* A5 = A4 * As → B2 (overwrite A3), accumulate +c5 (N) / -c5 (D) */
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    n_int, n_int, n_int, 1.0, B1, work_lda, As, work_lda, 0.0, B2, work_lda);
        for (matx_int64_t i = 0; i < total; ++i) {
            matx_double t = c5 * B2[i];
            N_mat[i] += t;
            D_mat[i] -= t;
        }

        /* A6 = A5 * As → B1 (overwrite A4), accumulate +c6 */
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    n_int, n_int, n_int, 1.0, B2, work_lda, As, work_lda, 0.0, B1, work_lda);
        for (matx_int64_t i = 0; i < total; ++i) {
            matx_double t = c6 * B1[i];
            N_mat[i] += t;
            D_mat[i] += t;
        }
    }
    /* As, B1, B2 no longer needed — free early */
    free(As);
    free(B1);
    free(B2);

    /* 7. Solve D * X = N by LU factorization */
    {
        matx_int64_t* piv = (matx_int64_t*) malloc((size_t) n * sizeof(matx_int64_t));
        if (!piv) {
            MATX_ERROR("%s: out of memory", __func__);
            free(N_mat);
            free(D_mat);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        matx_int64_t info = LAPACKE_dgetrf(lapack_layout, n_int, n_int, D_mat, work_lda, piv);
        if (info != 0) {
            MATX_ERROR("%s: dgetrf failed, info=%d", __func__, (int) info);
            free(piv);
            free(N_mat);
            free(D_mat);
            return MATX_ERR_INTERNAL;
        }
        /* D_mat now contains LU, N_mat is the RHS; solve in-place */
        matx_int64_t nrhs = n_int;
        info = LAPACKE_dgetrs(lapack_layout, 'N', n_int, nrhs, D_mat, work_lda, piv, N_mat, work_lda);
        if (info != 0) {
            MATX_ERROR("%s: dgetrs failed, info=%d", __func__, (int) info);
            free(piv);
            free(N_mat);
            free(D_mat);
            return MATX_ERR_INTERNAL;
        }
        free(piv);
    }

    /* 8. Square s times, alternating source/dst to avoid memcpy */
    {
        matx_double* src = N_mat;   /* holds current result (Pade approx) */
        matx_double* dst = D_mat;   /* scratch for next square */
        for (matx_int64_t k = 0; k < s; ++k) {
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                        n_int, n_int, n_int,
                        1.0, src, work_lda, src, work_lda, 0.0, dst, work_lda);
            /* swap roles */
            matx_double* tmp = src;
            src = dst;
            dst = tmp;
        }
        /* After loop, result is in 'src'; if it ended up in D_mat, copy back */
        if (src == D_mat) {
            memcpy(N_mat, D_mat, sz);
        }
    }

    /* 9. Copy result to output (stride-aware) */
    for (matx_int64_t j = 0; j < n; ++j)
        for (matx_int64_t i = 0; i < n; ++i)
            out[i + j * ldout] = N_mat[i + j * work_lda];

    free(N_mat);
    free(D_mat);
    return MATX_OK;
}

// ---- Level 2 implementations ----

static matx_status_t ref_dger(matx_layout_t layout,
                              matx_int64_t m,
                              matx_int64_t n,
                              matx_double alpha,
                              const matx_double* x,
                              matx_int64_t incx,
                              const matx_double* y,
                              matx_int64_t incy,
                              matx_double* A,
                              matx_int64_t lda)
{
    if (!x || !y || !A) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const enum CBLAS_ORDER order = (layout == MATX_COL_MAJOR) ? CblasColMajor : CblasRowMajor;
    cblas_dger(order, m, n, alpha, x, incx, y, incy, A, lda);
    return MATX_OK;
}

static matx_status_t ref_zgeru(matx_layout_t layout,
                               matx_int64_t m,
                               matx_int64_t n,
                               const void* alpha,
                               const void* x,
                               matx_int64_t incx,
                               const void* y,
                               matx_int64_t incy,
                               void* A,
                               matx_int64_t lda)
{
    if (!x || !y || !A || !alpha) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const enum CBLAS_ORDER order = (layout == MATX_COL_MAJOR) ? CblasColMajor : CblasRowMajor;
    cblas_zgeru(order, m, n, alpha, x, incx, y, incy, A, lda);
    return MATX_OK;
}

static matx_status_t ref_zgerc(matx_layout_t layout,
                               matx_int64_t m,
                               matx_int64_t n,
                               const void* alpha,
                               const void* x,
                               matx_int64_t incx,
                               const void* y,
                               matx_int64_t incy,
                               void* A,
                               matx_int64_t lda)
{
    if (!x || !y || !A || !alpha) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const enum CBLAS_ORDER order = (layout == MATX_COL_MAJOR) ? CblasColMajor : CblasRowMajor;
    cblas_zgerc(order, m, n, alpha, x, incx, y, incy, A, lda);
    return MATX_OK;
}

static matx_status_t ref_dtrsv(matx_layout_t layout,
                               matx_uplo_t uplo,
                               matx_trans_t trans,
                               matx_diag_t diag,
                               matx_int64_t n,
                               const matx_double* A,
                               matx_int64_t lda,
                               matx_double* x,
                               matx_int64_t incx)
{
    if (!A || !x) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const enum CBLAS_ORDER order = (layout == MATX_COL_MAJOR) ? CblasColMajor : CblasRowMajor;
    cblas_dtrsv(order,
                (enum CBLAS_UPLO) uplo,
                (enum CBLAS_TRANSPOSE) trans,
                (enum CBLAS_DIAG) diag,
                n,
                A,
                lda,
                x,
                incx);
    return MATX_OK;
}

static matx_status_t ref_ztrsv(matx_layout_t layout,
                               matx_uplo_t uplo,
                               matx_trans_t trans,
                               matx_diag_t diag,
                               matx_int64_t n,
                               const void* A,
                               matx_int64_t lda,
                               void* x,
                               matx_int64_t incx)
{
    if (!A || !x) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const enum CBLAS_ORDER order = (layout == MATX_COL_MAJOR) ? CblasColMajor : CblasRowMajor;
    cblas_ztrsv(order,
                (enum CBLAS_UPLO) uplo,
                (enum CBLAS_TRANSPOSE) trans,
                (enum CBLAS_DIAG) diag,
                n,
                A,
                lda,
                x,
                incx);
    return MATX_OK;
}

// ---- Level 3 implementations ----

static matx_status_t ref_dtrsm(matx_layout_t layout,
                               matx_side_t side,
                               matx_uplo_t uplo,
                               matx_trans_t trans,
                               matx_diag_t diag,
                               matx_int64_t m,
                               matx_int64_t n,
                               matx_double alpha,
                               const matx_double* A,
                               matx_int64_t lda,
                               matx_double* B,
                               matx_int64_t ldb)
{
    if (!A || !B) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const enum CBLAS_ORDER order = (layout == MATX_COL_MAJOR) ? CblasColMajor : CblasRowMajor;
    cblas_dtrsm(order,
                (enum CBLAS_SIDE) side,
                (enum CBLAS_UPLO) uplo,
                (enum CBLAS_TRANSPOSE) trans,
                (enum CBLAS_DIAG) diag,
                m,
                n,
                alpha,
                A,
                lda,
                B,
                ldb);
    return MATX_OK;
}

static matx_status_t ref_ztrsm(matx_layout_t layout,
                               matx_side_t side,
                               matx_uplo_t uplo,
                               matx_trans_t trans,
                               matx_diag_t diag,
                               matx_int64_t m,
                               matx_int64_t n,
                               const void* alpha,
                               const void* A,
                               matx_int64_t lda,
                               void* B,
                               matx_int64_t ldb)
{
    if (!A || !B || !alpha) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const enum CBLAS_ORDER order = (layout == MATX_COL_MAJOR) ? CblasColMajor : CblasRowMajor;
    cblas_ztrsm(order,
                (enum CBLAS_SIDE) side,
                (enum CBLAS_UPLO) uplo,
                (enum CBLAS_TRANSPOSE) trans,
                (enum CBLAS_DIAG) diag,
                m,
                n,
                alpha,
                A,
                lda,
                B,
                ldb);
    return MATX_OK;
}

static matx_status_t ref_dsyrk(matx_layout_t layout,
                               matx_uplo_t uplo,
                               matx_trans_t trans,
                               matx_int64_t n,
                               matx_int64_t k,
                               matx_double alpha,
                               const matx_double* A,
                               matx_int64_t lda,
                               matx_double beta,
                               matx_double* C,
                               matx_int64_t ldc)
{
    if (!A || !C) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const enum CBLAS_ORDER order = (layout == MATX_COL_MAJOR) ? CblasColMajor : CblasRowMajor;
    cblas_dsyrk(order,
                (enum CBLAS_UPLO) uplo,
                (enum CBLAS_TRANSPOSE) trans,
                n,
                k,
                alpha,
                A,
                lda,
                beta,
                C,
                ldc);
    return MATX_OK;
}

static matx_status_t ref_zherk(matx_layout_t layout,
                               matx_uplo_t uplo,
                               matx_trans_t trans,
                               matx_int64_t n,
                               matx_int64_t k,
                               matx_double alpha,
                               const void* A,
                               matx_int64_t lda,
                               matx_double beta,
                               void* C,
                               matx_int64_t ldc)
{
    if (!A || !C) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const enum CBLAS_ORDER order = (layout == MATX_COL_MAJOR) ? CblasColMajor : CblasRowMajor;
    cblas_zherk(order,
                (enum CBLAS_UPLO) uplo,
                (enum CBLAS_TRANSPOSE) trans,
                n,
                k,
                alpha,
                A,
                lda,
                beta,
                C,
                ldc);
    return MATX_OK;
}

static matx_status_t ref_dsyr2k(matx_layout_t layout,
                                matx_uplo_t uplo,
                                matx_trans_t trans,
                                matx_int64_t n,
                                matx_int64_t k,
                                matx_double alpha,
                                const matx_double* A,
                                matx_int64_t lda,
                                const matx_double* B,
                                matx_int64_t ldb,
                                matx_double beta,
                                matx_double* C,
                                matx_int64_t ldc)
{
    if (!A || !B || !C) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const enum CBLAS_ORDER order = (layout == MATX_COL_MAJOR) ? CblasColMajor : CblasRowMajor;
    cblas_dsyr2k(order,
                 (enum CBLAS_UPLO) uplo,
                 (enum CBLAS_TRANSPOSE) trans,
                 n,
                 k,
                 alpha,
                 A,
                 lda,
                 B,
                 ldb,
                 beta,
                 C,
                 ldc);
    return MATX_OK;
}

static matx_status_t ref_zher2k(matx_layout_t layout,
                                matx_uplo_t uplo,
                                matx_trans_t trans,
                                matx_int64_t n,
                                matx_int64_t k,
                                const void* alpha,
                                const void* A,
                                matx_int64_t lda,
                                const void* B,
                                matx_int64_t ldb,
                                matx_double beta,
                                void* C,
                                matx_int64_t ldc)
{
    if (!A || !B || !C || !alpha) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const enum CBLAS_ORDER order = (layout == MATX_COL_MAJOR) ? CblasColMajor : CblasRowMajor;
    cblas_zher2k(order,
                 (enum CBLAS_UPLO) uplo,
                 (enum CBLAS_TRANSPOSE) trans,
                 n,
                 k,
                 alpha,
                 A,
                 lda,
                 B,
                 ldb,
                 beta,
                 C,
                 ldc);
    return MATX_OK;
}

// ---- Transpose implementations ----

static matx_status_t ref_transpose_d_i8(matx_layout_t layout,
                                        matx_int64_t rows,
                                        matx_int64_t cols,
                                        const matx_double* A,
                                        matx_int64_t lda,
                                        matx_double* out,
                                        matx_int64_t ldc)
{
    if (!A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (rows > INT_MAX || cols > INT_MAX || lda > INT_MAX || ldc > INT_MAX) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
#if MATX_ENABLE_OPENBLAS
    const enum CBLAS_ORDER order = (layout == MATX_COL_MAJOR) ? CblasColMajor : CblasRowMajor;
    cblas_domatcopy(order, CblasTrans, rows, cols, 1.0, A, lda, out, ldc);
#elif MATX_ENABLE_BLIS
    if (layout == MATX_COL_MAJOR) {
        for (matx_int64_t i = 0; i < rows; ++i)
            for (matx_int64_t j = 0; j < cols; ++j)
                out[j + i * ldc] = A[i + j * lda];
    } else {
        for (matx_int64_t i = 0; i < rows; ++i)
            for (matx_int64_t j = 0; j < cols; ++j)
                out[j * ldc + i] = A[i * lda + j];
    }
#else
    if (layout == MATX_COL_MAJOR) {
        for (matx_int64_t i = 0; i < rows; ++i)
            for (matx_int64_t j = 0; j < cols; ++j)
                out[j + i * ldc] = A[i + j * lda];
    } else {
        for (matx_int64_t i = 0; i < rows; ++i)
            for (matx_int64_t j = 0; j < cols; ++j)
                out[j * ldc + i] = A[i * lda + j];
    }
#endif
    return MATX_OK;
}

static matx_status_t ref_transpose_z_i8(matx_layout_t layout,
                                        matx_int64_t rows,
                                        matx_int64_t cols,
                                        const void* A,
                                        matx_int64_t lda,
                                        void* out,
                                        matx_int64_t ldc)
{
    if (!A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (rows > INT_MAX || cols > INT_MAX || lda > INT_MAX || ldc > INT_MAX) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
#if MATX_ENABLE_OPENBLAS
    const enum CBLAS_ORDER order = (layout == MATX_COL_MAJOR) ? CblasColMajor : CblasRowMajor;
    const matx_double alpha[2] = {1.0, 0.0};
    cblas_zomatcopy(order, CblasTrans, rows, cols, alpha, A, lda, out, ldc);
#elif MATX_ENABLE_BLIS
    const matx_complex_d_t* a_data = (const matx_complex_d_t*) A;
    matx_complex_d_t* o_data = (matx_complex_d_t*) out;
    if (layout == MATX_COL_MAJOR) {
        for (matx_int64_t i = 0; i < rows; ++i)
            for (matx_int64_t j = 0; j < cols; ++j)
                o_data[j + i * ldc] = a_data[i + j * lda];
    } else {
        for (matx_int64_t i = 0; i < rows; ++i)
            for (matx_int64_t j = 0; j < cols; ++j)
                o_data[j * ldc + i] = a_data[i * lda + j];
    }
#else
    const matx_complex_d_t* a_data = (const matx_complex_d_t*) A;
    matx_complex_d_t* o_data = (matx_complex_d_t*) out;
    if (layout == MATX_COL_MAJOR) {
        for (matx_int64_t i = 0; i < rows; ++i)
            for (matx_int64_t j = 0; j < cols; ++j)
                o_data[j + i * ldc] = a_data[i + j * lda];
    } else {
        for (matx_int64_t i = 0; i < rows; ++i)
            for (matx_int64_t j = 0; j < cols; ++j)
                o_data[j * ldc + i] = a_data[i * lda + j];
    }
#endif
    return MATX_OK;
}

static matx_status_t ref_conj_transpose_z_i8(matx_layout_t layout,
                                             matx_int64_t rows,
                                             matx_int64_t cols,
                                             const void* A,
                                             matx_int64_t lda,
                                             void* out,
                                             matx_int64_t ldc)
{
    if (!A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (rows > INT_MAX || cols > INT_MAX || lda > INT_MAX || ldc > INT_MAX) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
#if MATX_ENABLE_OPENBLAS
    const enum CBLAS_ORDER order = (layout == MATX_COL_MAJOR) ? CblasColMajor : CblasRowMajor;
    const matx_double alpha[2] = {1.0, 0.0};
    cblas_zomatcopy(order, CblasConjTrans, rows, cols, alpha, A, lda, out, ldc);
#elif MATX_ENABLE_BLIS
    const matx_complex_d_t* a_data = (const matx_complex_d_t*) A;
    matx_complex_d_t* o_data = (matx_complex_d_t*) out;
    if (layout == MATX_COL_MAJOR) {
        for (matx_int64_t i = 0; i < rows; ++i)
            for (matx_int64_t j = 0; j < cols; ++j) {
                const matx_int64_t src = i + j * lda;
                const matx_int64_t dst = j + i * ldc;
                o_data[dst].real = a_data[src].real;
                o_data[dst].imag = -a_data[src].imag;
            }
    } else {
        for (matx_int64_t i = 0; i < rows; ++i)
            for (matx_int64_t j = 0; j < cols; ++j) {
                const matx_int64_t src = i * lda + j;
                const matx_int64_t dst = j * ldc + i;
                o_data[dst].real = a_data[src].real;
                o_data[dst].imag = -a_data[src].imag;
            }
    }
#else
    const matx_complex_d_t* a_data = (const matx_complex_d_t*) A;
    matx_complex_d_t* o_data = (matx_complex_d_t*) out;
    if (layout == MATX_COL_MAJOR) {
        for (matx_int64_t i = 0; i < rows; ++i)
            for (matx_int64_t j = 0; j < cols; ++j) {
                const matx_int64_t src = i + j * lda;
                const matx_int64_t dst = j + i * ldc;
                o_data[dst].real = a_data[src].real;
                o_data[dst].imag = -a_data[src].imag;
            }
    } else {
        for (matx_int64_t i = 0; i < rows; ++i)
            for (matx_int64_t j = 0; j < cols; ++j) {
                const matx_int64_t src = i * lda + j;
                const matx_int64_t dst = j * ldc + i;
                o_data[dst].real = a_data[src].real;
                o_data[dst].imag = -a_data[src].imag;
            }
    }
#endif
    return MATX_OK;
}

// ---- Norm implementations ----

#define MATX_NORM_STACK_LIMIT 1024

static matx_status_t ref_norm_abs_sum_d(matx_layout_t layout,
                                        matx_int64_t rows,
                                        matx_int64_t cols,
                                        const matx_double* A,
                                        matx_int64_t lda,
                                        matx_double* out,
                                        int norm_one)
{
    if (!A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (rows < 0 || cols < 0 || lda < 0
        || (layout != MATX_COL_MAJOR && layout != MATX_ROW_MAJOR)
        || (layout == MATX_COL_MAJOR && lda < rows)
        || (layout == MATX_ROW_MAJOR && lda < cols)) {
        MATX_ERROR("%s: invalid matrix dimensions or layout", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (rows > INT_MAX || cols > INT_MAX || lda > INT_MAX) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    if (rows == 0 || cols == 0) {
        *out = 0.0;
        return MATX_OK;
    }

    const matx_int64_t reduction_count = norm_one ? cols : rows;
    const matx_int64_t inner_count = norm_one ? rows : cols;
    const int contiguous_reductions
        = (norm_one && layout == MATX_COL_MAJOR)
          || (!norm_one && layout == MATX_ROW_MAJOR);
    matx_double maximum = 0.0;

    if (contiguous_reductions) {
        /* Inner loop walks contiguous memory — stride is 1. */
        for (matx_int64_t reduction = 0; reduction < reduction_count; ++reduction) {
            matx_double sum = 0.0;
            const matx_int64_t base = reduction * lda;
            for (matx_int64_t inner = 0; inner < inner_count; ++inner)
                sum += fabs(A[base + inner]);
            if (sum > maximum || isnan(sum)) maximum = sum;
        }
    } else {
        if ((uint64_t) reduction_count > SIZE_MAX / sizeof(matx_double)) {
            return MATX_ERR_INVALID_ARG;
        }
        matx_double sums_buf[MATX_NORM_STACK_LIMIT];
        matx_double* sums = sums_buf;
        int free_sums = 0;
        if ((uint64_t) reduction_count > MATX_NORM_STACK_LIMIT) {
            sums = (matx_double*) calloc((size_t) reduction_count, sizeof(*sums));
            if (!sums) return MATX_ERR_OUT_OF_MEMORY;
            free_sums = 1;
        } else {
            memset(sums_buf, 0, (size_t) reduction_count * sizeof(matx_double));
        }
        const int column_major = layout == MATX_COL_MAJOR;
        const matx_int64_t outer_count = column_major ? cols : rows;
        const matx_int64_t inner_count_by_layout = column_major ? rows : cols;
        for (matx_int64_t outer = 0; outer < outer_count; ++outer) {
            for (matx_int64_t inner = 0; inner < inner_count_by_layout; ++inner) {
                const matx_int64_t row = column_major ? inner : outer;
                const matx_int64_t col = column_major ? outer : inner;
                const size_t index = column_major
                                         ? (size_t) row + (size_t) col * (size_t) lda
                                         : (size_t) row * (size_t) lda + (size_t) col;
                sums[norm_one ? col : row] += fabs(A[index]);
            }
        }
        for (matx_int64_t i = 0; i < reduction_count; ++i) {
            if (sums[i] > maximum || isnan(sums[i])) maximum = sums[i];
        }
        if (free_sums) free(sums);
    }

    *out = maximum;
    return MATX_OK;
}

static matx_status_t ref_norm_abs_sum_z(matx_layout_t layout,
                                        matx_int64_t rows,
                                        matx_int64_t cols,
                                        const void* A,
                                        matx_int64_t lda,
                                        matx_double* out,
                                        int norm_one)
{
    if (!A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (rows < 0 || cols < 0 || lda < 0
        || (layout != MATX_COL_MAJOR && layout != MATX_ROW_MAJOR)
        || (layout == MATX_COL_MAJOR && lda < rows)
        || (layout == MATX_ROW_MAJOR && lda < cols)) {
        MATX_ERROR("%s: invalid matrix dimensions or layout", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (rows > INT_MAX || cols > INT_MAX || lda > INT_MAX) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    if (rows == 0 || cols == 0) {
        *out = 0.0;
        return MATX_OK;
    }

    const matx_complex_d_t* a = (const matx_complex_d_t*) A;
    const matx_int64_t reduction_count = norm_one ? cols : rows;
    const matx_int64_t inner_count = norm_one ? rows : cols;
    const int contiguous_reductions
        = (norm_one && layout == MATX_COL_MAJOR)
          || (!norm_one && layout == MATX_ROW_MAJOR);
    matx_double maximum = 0.0;

    if (contiguous_reductions) {
        for (matx_int64_t reduction = 0; reduction < reduction_count; ++reduction) {
            matx_double sum = 0.0;
            const matx_int64_t base = reduction * lda;
            for (matx_int64_t inner = 0; inner < inner_count; ++inner)
                sum += hypot(a[base + inner].real, a[base + inner].imag);
            if (sum > maximum || isnan(sum)) maximum = sum;
        }
    } else {
        if ((uint64_t) reduction_count > SIZE_MAX / sizeof(matx_double)) {
            return MATX_ERR_INVALID_ARG;
        }
        matx_double sums_buf[MATX_NORM_STACK_LIMIT];
        matx_double* sums = sums_buf;
        int free_sums = 0;
        if ((uint64_t) reduction_count > MATX_NORM_STACK_LIMIT) {
            sums = (matx_double*) calloc((size_t) reduction_count, sizeof(*sums));
            if (!sums) return MATX_ERR_OUT_OF_MEMORY;
            free_sums = 1;
        } else {
            memset(sums_buf, 0, (size_t) reduction_count * sizeof(matx_double));
        }
        const int column_major = layout == MATX_COL_MAJOR;
        const matx_int64_t outer_count = column_major ? cols : rows;
        const matx_int64_t inner_count_by_layout = column_major ? rows : cols;
        for (matx_int64_t outer = 0; outer < outer_count; ++outer) {
            for (matx_int64_t inner = 0; inner < inner_count_by_layout; ++inner) {
                const matx_int64_t row = column_major ? inner : outer;
                const matx_int64_t col = column_major ? outer : inner;
                const size_t index = column_major
                                         ? (size_t) row + (size_t) col * (size_t) lda
                                         : (size_t) row * (size_t) lda + (size_t) col;
                sums[norm_one ? col : row] += hypot(a[index].real, a[index].imag);
            }
        }
        for (matx_int64_t i = 0; i < reduction_count; ++i) {
            if (sums[i] > maximum || isnan(sums[i])) maximum = sums[i];
        }
        if (free_sums) free(sums);
    }

    *out = maximum;
    return MATX_OK;
}

static matx_status_t ref_norm1_d_i8(matx_layout_t layout,
                                    matx_int64_t rows,
                                    matx_int64_t cols,
                                    const matx_double* A,
                                    matx_int64_t lda,
                                    matx_double* out)
{
    return ref_norm_abs_sum_d(layout, rows, cols, A, lda, out, 1);
}

static matx_status_t ref_norminf_d_i8(matx_layout_t layout,
                                      matx_int64_t rows,
                                      matx_int64_t cols,
                                      const matx_double* A,
                                      matx_int64_t lda,
                                      matx_double* out)
{
    return ref_norm_abs_sum_d(layout, rows, cols, A, lda, out, 0);
}

static matx_status_t ref_normfro_d_i8(matx_layout_t layout,
                                      matx_int64_t rows,
                                      matx_int64_t cols,
                                      const matx_double* A,
                                      matx_int64_t lda,
                                      matx_double* out)
{
    if (!A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (rows > INT_MAX || cols > INT_MAX || lda > INT_MAX) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    *out = LAPACKE_dlange(layout == MATX_COL_MAJOR ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR,
                          'F',
                          (lapack_int) rows,
                          (lapack_int) cols,
                          A,
                          (lapack_int) lda);
    return MATX_OK;
}

static matx_status_t ref_norm1_z_i8(matx_layout_t layout,
                                    matx_int64_t rows,
                                    matx_int64_t cols,
                                    const void* A,
                                    matx_int64_t lda,
                                    matx_double* out)
{
    return ref_norm_abs_sum_z(layout, rows, cols, A, lda, out, 1);
}

static matx_status_t ref_norminf_z_i8(matx_layout_t layout,
                                      matx_int64_t rows,
                                      matx_int64_t cols,
                                      const void* A,
                                      matx_int64_t lda,
                                      matx_double* out)
{
    return ref_norm_abs_sum_z(layout, rows, cols, A, lda, out, 0);
}

static matx_status_t ref_normfro_z_i8(matx_layout_t layout,
                                      matx_int64_t rows,
                                      matx_int64_t cols,
                                      const void* A,
                                      matx_int64_t lda,
                                      matx_double* out)
{
    if (!A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (rows > INT_MAX || cols > INT_MAX || lda > INT_MAX) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    *out = LAPACKE_zlange(layout == MATX_COL_MAJOR ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR,
                          'F',
                          (lapack_int) rows,
                          (lapack_int) cols,
                          A,
                          (lapack_int) lda);
    return MATX_OK;
}

static matx_status_t ref_hadamard_d_i8(matx_layout_t layout,
                                       matx_int64_t rows,
                                       matx_int64_t cols,
                                       const matx_double* restrict A,
                                       matx_int64_t lda,
                                       const matx_double* restrict B,
                                       matx_int64_t ldb,
                                       matx_double* restrict C,
                                       matx_int64_t ldc)
{
    if (!A || !B || !C) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (layout == MATX_COL_MAJOR) {
        for (matx_int64_t j = 0; j < cols; ++j) {
            const matx_int64_t a_col = j * lda;
            const matx_int64_t b_col = j * ldb;
            const matx_int64_t c_col = j * ldc;
            for (matx_int64_t i = 0; i < rows; ++i)
                C[c_col + i] = A[a_col + i] * B[b_col + i];
        }
    } else {
        for (matx_int64_t i = 0; i < rows; ++i) {
            const matx_int64_t a_row = i * lda;
            const matx_int64_t b_row = i * ldb;
            const matx_int64_t c_row = i * ldc;
            for (matx_int64_t j = 0; j < cols; ++j)
                C[c_row + j] = A[a_row + j] * B[b_row + j];
        }
    }
    return MATX_OK;
}

static matx_status_t ref_hadamard_z_i8(matx_layout_t layout,
                                       matx_int64_t rows,
                                       matx_int64_t cols,
                                       const void* A,
                                       matx_int64_t lda,
                                       const void* B,
                                       matx_int64_t ldb,
                                       void* C,
                                       matx_int64_t ldc)
{
    if (!A || !B || !C) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const matx_complex_d_t* restrict a_data = (const matx_complex_d_t*) A;
    const matx_complex_d_t* restrict b_data = (const matx_complex_d_t*) B;
    matx_complex_d_t* restrict c_data = (matx_complex_d_t*) C;
    if (layout == MATX_COL_MAJOR) {
        for (matx_int64_t j = 0; j < cols; ++j) {
            const matx_int64_t a_col = j * lda;
            const matx_int64_t b_col = j * ldb;
            const matx_int64_t c_col = j * ldc;
            for (matx_int64_t i = 0; i < rows; ++i) {
                c_data[c_col + i].real = a_data[a_col + i].real * b_data[b_col + i].real
                    - a_data[a_col + i].imag * b_data[b_col + i].imag;
                c_data[c_col + i].imag = a_data[a_col + i].real * b_data[b_col + i].imag
                    + a_data[a_col + i].imag * b_data[b_col + i].real;
            }
        }
    } else {
        for (matx_int64_t i = 0; i < rows; ++i) {
            const matx_int64_t a_row = i * lda;
            const matx_int64_t b_row = i * ldb;
            const matx_int64_t c_row = i * ldc;
            for (matx_int64_t j = 0; j < cols; ++j) {
                c_data[c_row + j].real = a_data[a_row + j].real * b_data[b_row + j].real
                    - a_data[a_row + j].imag * b_data[b_row + j].imag;
                c_data[c_row + j].imag = a_data[a_row + j].real * b_data[b_row + j].imag
                    + a_data[a_row + j].imag * b_data[b_row + j].real;
            }
        }
    }
    return MATX_OK;
}

matx_dense_backend_t matx_blas_make_reference(void)
{
    matx_dense_backend_t b = {
#if MATX_ENABLE_BLIS
        .kind = MATX_BLAS_BACKEND_BLIS,
#elif MATX_ENABLE_OPENBLAS
        .kind = MATX_BLAS_BACKEND_OPENBLAS,
#else
        .kind = MATX_BLAS_BACKEND_REFERENCE,
#endif
        .vt = {
        .dgemm = &ref_dgemm,
        .zgemm = &ref_zgemm,
        .dgemv = &ref_dgemv,
        .zgemv = &ref_zgemv,
        .dgeadd = &ref_dgeadd,
        .zgeadd = &ref_zgeadd,
        .inv_dense_d_i8 = &ref_inv_dense_d_i8,
        .inv_dense_z_i8 = &ref_inv_dense_z_i8,
        .expm_dense_d_i8 = &ref_expm_dense_d_i8,
        .dger = &ref_dger,
        .zgeru = &ref_zgeru,
        .zgerc = &ref_zgerc,
        .dtrsv = &ref_dtrsv,
        .ztrsv = &ref_ztrsv,
        .dtrsm = &ref_dtrsm,
        .ztrsm = &ref_ztrsm,
        .dsyrk = &ref_dsyrk,
        .zherk = &ref_zherk,
        .dsyr2k = &ref_dsyr2k,
        .zher2k = &ref_zher2k,
        .transpose_d_i8 = &ref_transpose_d_i8,
        .transpose_z_i8 = &ref_transpose_z_i8,
        .conj_transpose_z_i8 = &ref_conj_transpose_z_i8,
        .hadamard_d_i8 = &ref_hadamard_d_i8,
        .hadamard_z_i8 = &ref_hadamard_z_i8,
        .norm1_d_i8 = &ref_norm1_d_i8,
        .norminf_d_i8 = &ref_norminf_d_i8,
        .normfro_d_i8 = &ref_normfro_d_i8,
        .norm1_z_i8 = &ref_norm1_z_i8,
        .norminf_z_i8 = &ref_norminf_z_i8,
        .normfro_z_i8 = &ref_normfro_z_i8,
        }};

    return b;
}
