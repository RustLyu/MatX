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

    const enum CBLAS_ORDER order = (layout == MATX_COL_MAJOR) ? CblasColMajor : CblasRowMajor;

    const enum CBLAS_TRANSPOSE ta = (enum CBLAS_TRANSPOSE) trans_a;

    const enum CBLAS_TRANSPOSE tb = (enum CBLAS_TRANSPOSE) trans_b;
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

    const enum CBLAS_ORDER order = (layout == MATX_COL_MAJOR) ? CblasColMajor : CblasRowMajor;

    const enum CBLAS_TRANSPOSE ta = (enum CBLAS_TRANSPOSE) trans_a;
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
    if (!A || !C) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    if (m > INT_MAX || n > INT_MAX || lda > INT_MAX || ldc > INT_MAX) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
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
            matx_int64_t len = rows * cols;

            if (beta != 1.0)
                cblas_dscal(len, beta, B, 1);

            if (alpha != 0.0)
                cblas_daxpy(len, alpha, A, 1, B, 1);

            return MATX_OK;
        }
    } else {
        if (lda == rows && ldb == rows) {
            matx_int64_t len = rows * cols;

            if (beta != 1.0)
                cblas_dscal(len, beta, B, 1);

            if (alpha != 0.0)
                cblas_daxpy(len, alpha, A, 1, B, 1);

            return MATX_OK;
        }
    }

    for (size_t j = 0; j < cols; ++j) {
        cblas_dscal(rows, beta, B + j * ldb, 1);
        cblas_daxpy(rows, alpha, A + j * lda, 1, B + j * ldb, 1);
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

    const void* alpha_p = alpha;
    const void* beta_p = beta;

    size_t len;

    if (trans_a == MATX_ROW_MAJOR) {
        if (lda == cols && ldb == cols) {
            len = rows * cols;
            cblas_zscal(len, beta_p, B, 1);
            cblas_zaxpy(len, alpha_p, A, 1, B, 1);

            return MATX_OK;
        }
    } else {
        if (lda == rows && ldb == rows) {
            len = rows * cols;

            cblas_zscal((matx_int64_t) len, beta_p, B, 1);
            cblas_zaxpy((matx_int64_t) len, alpha_p, A, 1, B, 1);

            return MATX_OK;
        }
    }

    for (matx_int64_t j = 0; j < cols; ++j) {
        void* Bcol = (char*) B + j * ldb * sizeof(matx_double) * 2;
        const void* Acol = (const char*) A + j * lda * sizeof(matx_double) * 2;

        cblas_zscal(rows, beta_p, Bcol, 1);
        cblas_zaxpy(rows, alpha_p, Acol, 1, Bcol, 1);
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
    // Copy with stride-awareness
    for (matx_int64_t j = 0; j < cols; ++j)
        for (matx_int64_t i = 0; i < rows; ++i) {
            out_Ainv[i + j * ldout] = A[i + j * lda];
        }

    matx_int64_t N = rows;
    matx_int64_t info = 0;

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
    // Copy with stride-awareness
    matx_complex_d_t* out = (matx_complex_d_t*) out_Ainv;
    const matx_complex_d_t* in = (const matx_complex_d_t*) A;
    for (matx_int64_t j = 0; j < cols; ++j)
        for (matx_int64_t i = 0; i < rows; ++i) {
            out[i + j * ldout] = in[i + j * lda];
        }

    matx_int64_t N = rows;
    matx_int64_t info = 0;

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

    /* 3. Allocate working buffers: As, A2, A3, A4, A5, A6, N, D, work */
    matx_double* As = (matx_double*) malloc(sz);
    matx_double* A2 = (matx_double*) malloc(sz);
    matx_double* A3 = (matx_double*) malloc(sz);
    matx_double* A4 = (matx_double*) malloc(sz);
    matx_double* A5 = (matx_double*) malloc(sz);
    matx_double* A6 = (matx_double*) malloc(sz);
    matx_double* N_mat = (matx_double*) malloc(sz);
    matx_double* D_mat = (matx_double*) malloc(sz);
    if (!As || !A2 || !A3 || !A4 || !A5 || !A6 || !N_mat || !D_mat) {
        MATX_ERROR("%s: out of memory", __func__);
        free(As);
        free(A2);
        free(A3);
        free(A4);
        free(A5);
        free(A6);
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

    /* 5. Compute powers: A2=As*As, A3=A2*As, A4=A3*As, A5=A4*As, A6=A5*As */
    cblas_dgemm(CblasColMajor,
                CblasNoTrans,
                CblasNoTrans,
                n_int,
                n_int,
                n_int,
                1.0,
                As,
                work_lda,
                As,
                work_lda,
                0.0,
                A2,
                work_lda);
    cblas_dgemm(CblasColMajor,
                CblasNoTrans,
                CblasNoTrans,
                n_int,
                n_int,
                n_int,
                1.0,
                A2,
                work_lda,
                As,
                work_lda,
                0.0,
                A3,
                work_lda);
    cblas_dgemm(CblasColMajor,
                CblasNoTrans,
                CblasNoTrans,
                n_int,
                n_int,
                n_int,
                1.0,
                A3,
                work_lda,
                As,
                work_lda,
                0.0,
                A4,
                work_lda);
    cblas_dgemm(CblasColMajor,
                CblasNoTrans,
                CblasNoTrans,
                n_int,
                n_int,
                n_int,
                1.0,
                A4,
                work_lda,
                As,
                work_lda,
                0.0,
                A5,
                work_lda);
    cblas_dgemm(CblasColMajor,
                CblasNoTrans,
                CblasNoTrans,
                n_int,
                n_int,
                n_int,
                1.0,
                A5,
                work_lda,
                As,
                work_lda,
                0.0,
                A6,
                work_lda);

    /* 6. Pade(6,6) coefficients */
    {
        const matx_double c0 = 1.0;
        const matx_double c1 = 1.0 / 2.0;
        const matx_double c2 = 5.0 / 44.0;
        const matx_double c3 = 1.0 / 66.0;
        const matx_double c4 = 1.0 / 792.0;
        const matx_double c5 = 1.0 / 15840.0;
        const matx_double c6 = 1.0 / 665280.0;

        /* N = c0*I + c1*As + c2*A2 + c3*A3 + c4*A4 + c5*A5 + c6*A6 */
        /* D = c0*I - c1*As + c2*A2 - c3*A3 + c4*A4 - c5*A5 + c6*A6 */
        for (matx_int64_t i = 0; i < total; ++i) {
            matx_double t2 = c2 * A2[i];
            matx_double t4 = c4 * A4[i];
            matx_double t6 = c6 * A6[i];
            matx_double odd = c1 * As[i] + c3 * A3[i] + c5 * A5[i];
            matx_double even = c0 * (i % (n + 1) == 0 ? 1.0 : 0.0) + t2 + t4 + t6;
            N_mat[i] = even + odd;
            D_mat[i] = even - odd;
        }
    }

    /* 7. Solve D * X = N by LU factorization */
    {
        matx_int64_t* piv = (matx_int64_t*) malloc((size_t) n * sizeof(matx_int64_t));
        if (!piv) {
            MATX_ERROR("%s: out of memory", __func__);
            free(As);
            free(A2);
            free(A3);
            free(A4);
            free(A5);
            free(A6);
            free(N_mat);
            free(D_mat);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        matx_int64_t info = LAPACKE_dgetrf(lapack_layout, n_int, n_int, D_mat, work_lda, piv);
        if (info != 0) {
            MATX_ERROR("%s: dgetrf failed, info=%d", __func__, (int) info);
            free(piv);
            free(As);
            free(A2);
            free(A3);
            free(A4);
            free(A5);
            free(A6);
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
            free(As);
            free(A2);
            free(A3);
            free(A4);
            free(A5);
            free(A6);
            free(N_mat);
            free(D_mat);
            return MATX_ERR_INTERNAL;
        }
        free(piv);
    }

    /* 8. Square s times: N_mat = N_mat * N_mat repeatedly */
    for (matx_int64_t k = 0; k < s; ++k) {
        /* Copy N_mat to D_mat as source */
        memcpy(D_mat, N_mat, sz);
        cblas_dgemm(CblasColMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    n_int,
                    n_int,
                    n_int,
                    1.0,
                    D_mat,
                    work_lda,
                    D_mat,
                    work_lda,
                    0.0,
                    N_mat,
                    work_lda);
    }

    /* 9. Copy result to output (stride-aware) */
    for (matx_int64_t j = 0; j < n; ++j)
        for (matx_int64_t i = 0; i < n; ++i)
            out[i + j * ldout] = N_mat[i + j * work_lda];

    free(As);
    free(A2);
    free(A3);
    free(A4);
    free(A5);
    free(A6);
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
    for (matx_int64_t i = 0; i < rows; ++i)
        for (matx_int64_t j = 0; j < cols; ++j) {
            matx_int64_t src = (layout == MATX_COL_MAJOR) ? i + j * lda : i * lda + j;
            matx_int64_t dst = (layout == MATX_COL_MAJOR) ? j + i * ldc : j * ldc + i;
            out[dst] = A[src];
        }
#else
    for (matx_int64_t i = 0; i < rows; ++i)
        for (matx_int64_t j = 0; j < cols; ++j) {
            matx_int64_t src = (layout == MATX_COL_MAJOR) ? i + j * lda : i * lda + j;
            matx_int64_t dst = (layout == MATX_COL_MAJOR) ? j + i * ldc : j * ldc + i;
            out[dst] = A[src];
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
    for (matx_int64_t i = 0; i < rows; ++i)
        for (matx_int64_t j = 0; j < cols; ++j) {
            matx_int64_t src = (layout == MATX_COL_MAJOR) ? i + j * lda : i * lda + j;
            matx_int64_t dst = (layout == MATX_COL_MAJOR) ? j + i * ldc : j * ldc + i;
            o_data[dst] = a_data[src];
        }
#else
    const matx_complex_d_t* a_data = (const matx_complex_d_t*) A;
    matx_complex_d_t* o_data = (matx_complex_d_t*) out;
    for (matx_int64_t i = 0; i < rows; ++i)
        for (matx_int64_t j = 0; j < cols; ++j) {
            matx_int64_t src = (layout == MATX_COL_MAJOR) ? i + j * lda : i * lda + j;
            matx_int64_t dst = (layout == MATX_COL_MAJOR) ? j + i * ldc : j * ldc + i;
            o_data[dst] = a_data[src];
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
    for (matx_int64_t i = 0; i < rows; ++i)
        for (matx_int64_t j = 0; j < cols; ++j) {
            matx_int64_t src = (layout == MATX_COL_MAJOR) ? i + j * lda : i * lda + j;
            matx_int64_t dst = (layout == MATX_COL_MAJOR) ? j + i * ldc : j * ldc + i;
            o_data[dst].real = a_data[src].real;
            o_data[dst].imag = -a_data[src].imag;
        }
#else
    const matx_complex_d_t* a_data = (const matx_complex_d_t*) A;
    matx_complex_d_t* o_data = (matx_complex_d_t*) out;
    for (matx_int64_t i = 0; i < rows; ++i)
        for (matx_int64_t j = 0; j < cols; ++j) {
            matx_int64_t src = (layout == MATX_COL_MAJOR) ? i + j * lda : i * lda + j;
            matx_int64_t dst = (layout == MATX_COL_MAJOR) ? j + i * ldc : j * ldc + i;
            o_data[dst].real = a_data[src].real;
            o_data[dst].imag = -a_data[src].imag;
        }
#endif
    return MATX_OK;
}

// ---- Norm implementations ----

static matx_status_t ref_norm_abs_sum(matx_layout_t layout,
                                      matx_int64_t rows,
                                      matx_int64_t cols,
                                      const void* A,
                                      matx_int64_t lda,
                                      matx_double* out,
                                      int norm_one,
                                      int complex_values)
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
        for (matx_int64_t reduction = 0; reduction < reduction_count; ++reduction) {
            matx_double sum = 0.0;
            for (matx_int64_t inner = 0; inner < inner_count; ++inner) {
                const matx_int64_t row = norm_one ? inner : reduction;
                const matx_int64_t col = norm_one ? reduction : inner;
                const size_t index = layout == MATX_COL_MAJOR
                                         ? (size_t) row + (size_t) col * (size_t) lda
                                         : (size_t) row * (size_t) lda + (size_t) col;
                matx_double magnitude;
                if (complex_values) {
                    const matx_complex_d_t value = ((const matx_complex_d_t*) A)[index];
                    magnitude = hypot(value.real, value.imag);
                } else {
                    magnitude = fabs(((const matx_double*) A)[index]);
                }
                sum += magnitude;
            }
            if (sum > maximum || isnan(sum)) maximum = sum;
        }
    } else {
        if ((uint64_t) reduction_count > SIZE_MAX / sizeof(matx_double)) {
            return MATX_ERR_INVALID_ARG;
        }
        matx_double* sums = (matx_double*) calloc((size_t) reduction_count, sizeof(*sums));
        if (!sums) return MATX_ERR_OUT_OF_MEMORY;
        const int column_major = layout == MATX_COL_MAJOR;
        const matx_int64_t outer_count = column_major ? cols : rows;
        const matx_int64_t inner_count_by_layout = column_major ? rows : cols;
        for (matx_int64_t outer = 0; outer < outer_count; ++outer) {
            for (matx_int64_t inner = 0; inner < inner_count_by_layout; ++inner) {
                const matx_int64_t row = column_major ? inner : outer;
                const matx_int64_t col = column_major ? outer : inner;
                const size_t index = layout == MATX_COL_MAJOR
                                         ? (size_t) row + (size_t) col * (size_t) lda
                                         : (size_t) row * (size_t) lda + (size_t) col;
                matx_double magnitude;
                if (complex_values) {
                    const matx_complex_d_t value = ((const matx_complex_d_t*) A)[index];
                    magnitude = hypot(value.real, value.imag);
                } else {
                    magnitude = fabs(((const matx_double*) A)[index]);
                }
                sums[norm_one ? col : row] += magnitude;
            }
        }
        for (matx_int64_t i = 0; i < reduction_count; ++i) {
            if (sums[i] > maximum || isnan(sums[i])) maximum = sums[i];
        }
        free(sums);
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
    return ref_norm_abs_sum(layout, rows, cols, A, lda, out, 1, 0);
}

static matx_status_t ref_norminf_d_i8(matx_layout_t layout,
                                      matx_int64_t rows,
                                      matx_int64_t cols,
                                      const matx_double* A,
                                      matx_int64_t lda,
                                      matx_double* out)
{
    return ref_norm_abs_sum(layout, rows, cols, A, lda, out, 0, 0);
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
    return ref_norm_abs_sum(layout, rows, cols, A, lda, out, 1, 1);
}

static matx_status_t ref_norminf_z_i8(matx_layout_t layout,
                                      matx_int64_t rows,
                                      matx_int64_t cols,
                                      const void* A,
                                      matx_int64_t lda,
                                      matx_double* out)
{
    return ref_norm_abs_sum(layout, rows, cols, A, lda, out, 0, 1);
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
                                       const matx_double* A,
                                       matx_int64_t lda,
                                       const matx_double* B,
                                       matx_int64_t ldb,
                                       matx_double* C,
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
    const matx_complex_d_t* a_data = (const matx_complex_d_t*) A;
    const matx_complex_d_t* b_data = (const matx_complex_d_t*) B;
    matx_complex_d_t* c_data = (matx_complex_d_t*) C;
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
