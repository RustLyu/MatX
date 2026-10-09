#include "matx/matx_dense_solve.h"
#include "matx/matx_log.h"
#include "matx/matx_types_internal.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#if MATX_ENABLE_OPENBLAS
#include "cblas.h"
#include "lapacke.h"
#elif MATX_ENABLE_LIBFLAME
//#include "FLAME.h"
#include "lapacke.h"
#endif

// Dense factorization
struct matx_factor_dense_d_i8_t
{
    matx_int64_t n;
    matx_int64_t lda;
    matx_double* lu;
    matx_int64_t* piv;
    matx_layout_t layout;
    matx_uplo_t uplo;
    matx_alloc_t alloc;
};

struct matx_factor_dense_z_i8_t
{
    matx_int64_t n;
    matx_int64_t lda;
    matx_double* lu;
    matx_int64_t* piv;
    matx_layout_t layout;
    matx_uplo_t uplo;
    matx_alloc_t alloc;
};

static void ss_factor_dense_d_i8_destroy(const matx_alloc_t* alloc, matx_factor_dense_d_i8_t* F)
{
    if (!F)
        return;
    matx_free(alloc, F->lu);
    matx_free(alloc, F->piv);
    matx_free(alloc, F);
}

static void ss_factor_dense_z_i8_destroy(const matx_alloc_t* alloc, matx_factor_dense_z_i8_t* F)
{
    if (!F)
        return;
    matx_free(alloc, F->lu);
    matx_free(alloc, F->piv);
    matx_free(alloc, F);
}

// ---- Stride-aware packing helpers ----
// Pack a strided dense matrix into a contiguous buffer with lda = min dimension.
// Returns lda for the packed buffer.

static matx_int64_t ss_packed_lda(matx_layout_t layout, matx_int64_t nrows, matx_int64_t ncols)
{
    return (layout == MATX_COL_MAJOR) ? nrows : ncols;
}

static void ss_pack_d_i8(matx_layout_t layout,
                         matx_int64_t nrows,
                         matx_int64_t ncols,
                         matx_int64_t src_stride,
                         const matx_double* src,
                         matx_double* dst)
{
    matx_int64_t lda = ss_packed_lda(layout, nrows, ncols);
    if (src_stride == lda) {
        memcpy(dst, src, (size_t) nrows * (size_t) ncols * sizeof(matx_double));
    } else {
        for (matx_int64_t j = 0; j < ncols; ++j)
            for (matx_int64_t i = 0; i < nrows; ++i) {
                matx_int64_t si = (layout == MATX_COL_MAJOR) ? i + j * src_stride
                                                             : j + i * src_stride;
                matx_int64_t di = (layout == MATX_COL_MAJOR) ? i + j * lda : j + i * lda;
                dst[di] = src[si];
            }
    }
}

static void ss_pack_z_i8(matx_layout_t layout,
                         matx_int64_t nrows,
                         matx_int64_t ncols,
                         matx_int64_t src_stride,
                         const matx_complex_d_t* src,
                         matx_complex_d_t* dst)
{
    matx_int64_t lda = ss_packed_lda(layout, nrows, ncols);
    if (src_stride == lda) {
        memcpy(dst, src, (size_t) nrows * (size_t) ncols * sizeof(matx_complex_d_t));
    } else {
        for (matx_int64_t j = 0; j < ncols; ++j)
            for (matx_int64_t i = 0; i < nrows; ++i) {
                matx_int64_t si = (layout == MATX_COL_MAJOR) ? i + j * src_stride
                                                             : j + i * src_stride;
                matx_int64_t di = (layout == MATX_COL_MAJOR) ? i + j * lda : j + i * lda;
                dst[di] = src[si];
            }
    }
}

static int ss_layout_to_lapack(matx_layout_t layout)
{
    return (layout == MATX_COL_MAJOR) ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR;
}

#define MATX_DENSE_SMALL_SOLVE_LIMIT 32

static matx_int64_t ss_dense_index(matx_layout_t layout,
                                   matx_int64_t lda,
                                   matx_int64_t row,
                                   matx_int64_t col)
{
    return (layout == MATX_COL_MAJOR) ? row + col * lda : row * lda + col;
}

static void ss_solve_small_lu_d(const matx_factor_dense_d_i8_t* F, matx_double* restrict rhs)
{
    const matx_int64_t n = F->n;
    const matx_double* restrict lu = F->lu;
    const matx_int64_t lda = F->lda;

    for (matx_int64_t i = 0; i < n; ++i) {
        const matx_int64_t pivot = F->piv[i] - 1;
        if (pivot != i) {
            const matx_double tmp = rhs[i];
            rhs[i] = rhs[pivot];
            rhs[pivot] = tmp;
        }
    }

    if (F->layout == MATX_COL_MAJOR) {
        for (matx_int64_t col = 0; col < n; ++col) {
            const matx_double rcol = rhs[col];
            #pragma omp simd
            for (matx_int64_t row = col + 1; row < n; ++row)
                rhs[row] -= lu[row + col * lda] * rcol;
        }
        for (matx_int64_t col = n; col-- > 0;) {
            rhs[col] /= lu[col + col * lda];
            const matx_double rcol = rhs[col];
            #pragma omp simd
            for (matx_int64_t row = 0; row < col; ++row)
                rhs[row] -= lu[row + col * lda] * rcol;
        }
    } else {
        for (matx_int64_t row = 0; row < n; ++row) {
            matx_double sum = rhs[row];
            #pragma omp simd
            for (matx_int64_t col = 0; col < row; ++col)
                sum -= lu[row * lda + col] * rhs[col];
            rhs[row] = sum;
        }
        for (matx_int64_t row = n; row-- > 0;) {
            matx_double sum = rhs[row];
            #pragma omp simd
            for (matx_int64_t col = row + 1; col < n; ++col)
                sum -= lu[row * lda + col] * rhs[col];
            rhs[row] = sum / lu[row * lda + row];
        }
    }
}

static inline matx_complex_d_t ss_complex_sub(matx_complex_d_t a, matx_complex_d_t b)
{
    return (matx_complex_d_t) {a.real - b.real, a.imag - b.imag};
}

static inline matx_complex_d_t ss_complex_mul(matx_complex_d_t a, matx_complex_d_t b)
{
    return (matx_complex_d_t) {a.real * b.real - a.imag * b.imag,
                               a.real * b.imag + a.imag * b.real};
}

static inline matx_complex_d_t ss_complex_div(matx_complex_d_t a, matx_complex_d_t b)
{
    if (fabs(b.real) >= fabs(b.imag)) {
        const matx_double ratio = b.imag / b.real;
        const matx_double denominator = b.real + b.imag * ratio;
        return (matx_complex_d_t) {(a.real + a.imag * ratio) / denominator,
                                   (a.imag - a.real * ratio) / denominator};
    }
    const matx_double ratio = b.real / b.imag;
    const matx_double denominator = b.imag + b.real * ratio;
    return (matx_complex_d_t) {(a.real * ratio + a.imag) / denominator,
                               (a.imag * ratio - a.real) / denominator};
}

static inline matx_complex_d_t ss_complex_conj(matx_complex_d_t a)
{
    a.imag = -a.imag;
    return a;
}

static void ss_solve_small_chol_d(const matx_factor_dense_d_i8_t* F, matx_double* restrict rhs)
{
    const matx_int64_t n = F->n;
    const matx_double* restrict lu = F->lu;
    const matx_int64_t lda = F->lda;
    const matx_layout_t layout = F->layout;

    if (F->uplo != MATX_UPPER) {
        if (layout == MATX_COL_MAJOR) {
            for (matx_int64_t row = 0; row < n; ++row) {
                matx_double sum = rhs[row];
                #pragma omp simd
                for (matx_int64_t col = 0; col < row; ++col)
                    sum -= lu[row + col * lda] * rhs[col];
                rhs[row] = sum / lu[row + row * lda];
            }
            for (matx_int64_t row = n; row-- > 0;) {
                matx_double sum = rhs[row];
                #pragma omp simd
                for (matx_int64_t col = row + 1; col < n; ++col)
                    sum -= lu[col + row * lda] * rhs[col];
                rhs[row] = sum / lu[row + row * lda];
            }
        } else {
            for (matx_int64_t row = 0; row < n; ++row) {
                matx_double sum = rhs[row];
                #pragma omp simd
                for (matx_int64_t col = 0; col < row; ++col)
                    sum -= lu[row * lda + col] * rhs[col];
                rhs[row] = sum / lu[row * lda + row];
            }
            for (matx_int64_t row = n; row-- > 0;) {
                matx_double sum = rhs[row];
                #pragma omp simd
                for (matx_int64_t col = row + 1; col < n; ++col)
                    sum -= lu[col * lda + row] * rhs[col];
                rhs[row] = sum / lu[row * lda + row];
            }
        }
    } else {
        if (layout == MATX_COL_MAJOR) {
            for (matx_int64_t row = 0; row < n; ++row) {
                matx_double sum = rhs[row];
                #pragma omp simd
                for (matx_int64_t col = 0; col < row; ++col)
                    sum -= lu[col + row * lda] * rhs[col];
                rhs[row] = sum / lu[row + row * lda];
            }
            for (matx_int64_t row = n; row-- > 0;) {
                matx_double sum = rhs[row];
                #pragma omp simd
                for (matx_int64_t col = row + 1; col < n; ++col)
                    sum -= lu[row + col * lda] * rhs[col];
                rhs[row] = sum / lu[row + row * lda];
            }
        } else {
            for (matx_int64_t row = 0; row < n; ++row) {
                matx_double sum = rhs[row];
                #pragma omp simd
                for (matx_int64_t col = 0; col < row; ++col)
                    sum -= lu[col * lda + row] * rhs[col];
                rhs[row] = sum / lu[row * lda + row];
            }
            for (matx_int64_t row = n; row-- > 0;) {
                matx_double sum = rhs[row];
                #pragma omp simd
                for (matx_int64_t col = row + 1; col < n; ++col)
                    sum -= lu[row * lda + col] * rhs[col];
                rhs[row] = sum / lu[row * lda + row];
            }
        }
    }
}

static void ss_solve_small_chol_z(const matx_factor_dense_z_i8_t* F,
                                  matx_complex_d_t* restrict rhs)
{
    const matx_int64_t n = F->n;
    const matx_complex_d_t* restrict factor = (const matx_complex_d_t*) F->lu;
    const matx_int64_t lda = F->lda;
    const matx_layout_t layout = F->layout;

    if (F->uplo != MATX_UPPER) {
        if (layout == MATX_COL_MAJOR) {
            for (matx_int64_t row = 0; row < n; ++row) {
                matx_double sum_re = rhs[row].real, sum_im = rhs[row].imag;
                #pragma omp simd
                for (matx_int64_t col = 0; col < row; ++col) {
                    const matx_complex_d_t a = factor[row + col * lda];
                    const matx_complex_d_t b = rhs[col];
                    sum_re -= a.real * b.real - a.imag * b.imag;
                    sum_im -= a.real * b.imag + a.imag * b.real;
                }
                const matx_complex_d_t d = factor[row + row * lda];
                rhs[row] = ss_complex_div((matx_complex_d_t){sum_re, sum_im}, d);
            }
            for (matx_int64_t row = n; row-- > 0;) {
                matx_double sum_re = rhs[row].real, sum_im = rhs[row].imag;
                #pragma omp simd
                for (matx_int64_t col = row + 1; col < n; ++col) {
                    const matx_complex_d_t a = factor[col + row * lda];
                    const matx_complex_d_t b = rhs[col];
                    sum_re -= a.real * b.real + a.imag * b.imag;  /* conj(a) * b */
                    sum_im -= a.real * b.imag - a.imag * b.real;
                }
                const matx_complex_d_t d = factor[row + row * lda];
                rhs[row] = ss_complex_div((matx_complex_d_t){sum_re, sum_im},
                                          (matx_complex_d_t){d.real, -d.imag});
            }
        } else {
            for (matx_int64_t row = 0; row < n; ++row) {
                matx_double sum_re = rhs[row].real, sum_im = rhs[row].imag;
                #pragma omp simd
                for (matx_int64_t col = 0; col < row; ++col) {
                    const matx_complex_d_t a = factor[row * lda + col];
                    const matx_complex_d_t b = rhs[col];
                    sum_re -= a.real * b.real - a.imag * b.imag;
                    sum_im -= a.real * b.imag + a.imag * b.real;
                }
                const matx_complex_d_t d = factor[row * lda + row];
                rhs[row] = ss_complex_div((matx_complex_d_t){sum_re, sum_im}, d);
            }
            for (matx_int64_t row = n; row-- > 0;) {
                matx_double sum_re = rhs[row].real, sum_im = rhs[row].imag;
                #pragma omp simd
                for (matx_int64_t col = row + 1; col < n; ++col) {
                    const matx_complex_d_t a = factor[col * lda + row];
                    const matx_complex_d_t b = rhs[col];
                    sum_re -= a.real * b.real + a.imag * b.imag;
                    sum_im -= a.real * b.imag - a.imag * b.real;
                }
                const matx_complex_d_t d = factor[row * lda + row];
                rhs[row] = ss_complex_div((matx_complex_d_t){sum_re, sum_im},
                                          (matx_complex_d_t){d.real, -d.imag});
            }
        }
    } else {
        if (layout == MATX_COL_MAJOR) {
            for (matx_int64_t row = 0; row < n; ++row) {
                matx_double sum_re = rhs[row].real, sum_im = rhs[row].imag;
                #pragma omp simd
                for (matx_int64_t col = 0; col < row; ++col) {
                    const matx_complex_d_t a = factor[col + row * lda];
                    const matx_complex_d_t b = rhs[col];
                    sum_re -= a.real * b.real + a.imag * b.imag;
                    sum_im -= a.real * b.imag - a.imag * b.real;
                }
                const matx_complex_d_t d = factor[row + row * lda];
                rhs[row] = ss_complex_div((matx_complex_d_t){sum_re, sum_im},
                                          (matx_complex_d_t){d.real, -d.imag});
            }
            for (matx_int64_t row = n; row-- > 0;) {
                matx_double sum_re = rhs[row].real, sum_im = rhs[row].imag;
                #pragma omp simd
                for (matx_int64_t col = row + 1; col < n; ++col) {
                    const matx_complex_d_t a = factor[row + col * lda];
                    const matx_complex_d_t b = rhs[col];
                    sum_re -= a.real * b.real - a.imag * b.imag;
                    sum_im -= a.real * b.imag + a.imag * b.real;
                }
                const matx_complex_d_t d = factor[row + row * lda];
                rhs[row] = ss_complex_div((matx_complex_d_t){sum_re, sum_im}, d);
            }
        } else {
            for (matx_int64_t row = 0; row < n; ++row) {
                matx_double sum_re = rhs[row].real, sum_im = rhs[row].imag;
                #pragma omp simd
                for (matx_int64_t col = 0; col < row; ++col) {
                    const matx_complex_d_t a = factor[col * lda + row];
                    const matx_complex_d_t b = rhs[col];
                    sum_re -= a.real * b.real + a.imag * b.imag;
                    sum_im -= a.real * b.imag - a.imag * b.real;
                }
                const matx_complex_d_t d = factor[row * lda + row];
                rhs[row] = ss_complex_div((matx_complex_d_t){sum_re, sum_im},
                                          (matx_complex_d_t){d.real, -d.imag});
            }
            for (matx_int64_t row = n; row-- > 0;) {
                matx_double sum_re = rhs[row].real, sum_im = rhs[row].imag;
                #pragma omp simd
                for (matx_int64_t col = row + 1; col < n; ++col) {
                    const matx_complex_d_t a = factor[row * lda + col];
                    const matx_complex_d_t b = rhs[col];
                    sum_re -= a.real * b.real - a.imag * b.imag;
                    sum_im -= a.real * b.imag + a.imag * b.real;
                }
                const matx_complex_d_t d = factor[row * lda + row];
                rhs[row] = ss_complex_div((matx_complex_d_t){sum_re, sum_im}, d);
            }
        }
    }
}

static void ss_solve_small_lu_z(const matx_factor_dense_z_i8_t* F,
                                matx_complex_d_t* restrict rhs)
{
    const matx_int64_t n = F->n;
    const matx_complex_d_t* restrict lu = (const matx_complex_d_t*) F->lu;
    const matx_int64_t lda = F->lda;

    for (matx_int64_t i = 0; i < n; ++i) {
        const matx_int64_t pivot = F->piv[i] - 1;
        if (pivot != i) {
            const matx_complex_d_t tmp = rhs[i];
            rhs[i] = rhs[pivot];
            rhs[pivot] = tmp;
        }
    }

    if (F->layout == MATX_COL_MAJOR) {
        for (matx_int64_t col = 0; col < n; ++col) {
            const matx_complex_d_t rcol = rhs[col];
            #pragma omp simd
            for (matx_int64_t row = col + 1; row < n; ++row) {
                const matx_complex_d_t a = lu[row + col * lda];
                rhs[row].real -= a.real * rcol.real - a.imag * rcol.imag;
                rhs[row].imag -= a.real * rcol.imag + a.imag * rcol.real;
            }
        }
        for (matx_int64_t col = n; col-- > 0;) {
            rhs[col] = ss_complex_div(rhs[col], lu[col + col * lda]);
            const matx_complex_d_t rcol = rhs[col];
            #pragma omp simd
            for (matx_int64_t row = 0; row < col; ++row) {
                const matx_complex_d_t a = lu[row + col * lda];
                rhs[row].real -= a.real * rcol.real - a.imag * rcol.imag;
                rhs[row].imag -= a.real * rcol.imag + a.imag * rcol.real;
            }
        }
    } else {
        for (matx_int64_t row = 0; row < n; ++row) {
            matx_double sum_re = rhs[row].real, sum_im = rhs[row].imag;
            #pragma omp simd
            for (matx_int64_t col = 0; col < row; ++col) {
                const matx_complex_d_t a = lu[row * lda + col];
                const matx_complex_d_t b = rhs[col];
                sum_re -= a.real * b.real - a.imag * b.imag;
                sum_im -= a.real * b.imag + a.imag * b.real;
            }
            rhs[row].real = sum_re;
            rhs[row].imag = sum_im;
        }
        for (matx_int64_t row = n; row-- > 0;) {
            matx_double sum_re = rhs[row].real, sum_im = rhs[row].imag;
            #pragma omp simd
            for (matx_int64_t col = row + 1; col < n; ++col) {
                const matx_complex_d_t a = lu[row * lda + col];
                const matx_complex_d_t b = rhs[col];
                sum_re -= a.real * b.real - a.imag * b.imag;
                sum_im -= a.real * b.imag + a.imag * b.real;
            }
            const matx_complex_d_t d = lu[row * lda + row];
            rhs[row] = ss_complex_div((matx_complex_d_t){sum_re, sum_im}, d);
        }
    }
}

// ---- Dense real LU ----

static matx_status_t ss_factor_dense_d_i8(const matx_alloc_t* alloc,
                                          const matx_dense_d_i8_t A,
                                          matx_factor_dense_d_i8_t** out_F)
{
    if (!A || !out_F) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    if (A->nrows != A->ncols) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    if (*out_F != NULL)
        ss_factor_dense_d_i8_destroy(alloc, *out_F);

    const matx_int64_t n = A->nrows;
    matx_int64_t lda = ss_packed_lda(A->layout, n, n);

    matx_factor_dense_d_i8_t* F = (matx_factor_dense_d_i8_t*) matx_malloc(alloc, sizeof(*F));
    if (!F) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(F, 0, sizeof(*F));
    F->n = n;
    F->lda = lda;
    F->layout = A->layout;
    F->alloc = *alloc;

    F->lu = (matx_double*) matx_malloc(alloc, (size_t) n * (size_t) n * sizeof(matx_double));
    F->piv = (matx_int64_t*) matx_malloc(alloc, n * sizeof(matx_int64_t));
    if (!F->lu || !F->piv) {
        matx_free(alloc, F->lu);
        matx_free(alloc, F->piv);
        matx_free(alloc, F);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    ss_pack_d_i8(A->layout, n, n, A->stride, A->data, F->lu);

    matx_int64_t info = LAPACKE_dgetrf(ss_layout_to_lapack(A->layout), n, n, F->lu, lda, F->piv);
    if (info != 0) {
        MATX_ERROR("LAPACKE_dgetrf error:%d", info);
        matx_free(alloc, F->lu);
        matx_free(alloc, F->piv);
        matx_free(alloc, F);
        return MATX_ERR_INTERNAL;
    }

    *out_F = F;
    return MATX_OK;
}

static matx_status_t ss_solve_dense_d_i8(const matx_alloc_t* alloc,
                                         const matx_factor_dense_d_i8_t* F,
                                         const matx_double* b,
                                         matx_double* x)
{
    if (!F || !b || !x) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (x != b)
        memmove(x, b, (size_t) F->n * sizeof(matx_double));

    if (F->n <= MATX_DENSE_SMALL_SOLVE_LIMIT) {
        ss_solve_small_lu_d(F, x);
        return MATX_OK;
    }

    matx_int64_t ldb = (F->layout == MATX_COL_MAJOR) ? F->n : 1;
    matx_int64_t info
        = LAPACKE_dgetrs(ss_layout_to_lapack(F->layout), 'N', F->n, 1, F->lu, F->lda, F->piv, x, ldb);
    if (info != 0) {
        MATX_ERROR("LAPACKE_dgetrs error:%d", info);
        return MATX_ERR_INTERNAL;
    }
    return MATX_OK;
}

// ---- Dense complex LU ----

static matx_status_t ss_factor_dense_z_i8(const matx_alloc_t* alloc,
                                          const matx_dense_z_i8_t A,
                                          matx_factor_dense_z_i8_t** out_F)
{
#if !(defined(MATX_HAVE_OPENBLAS) || defined(MATX_HAVE_LIBFLAME))
    MATX_ERROR("%s: operation not supported", __func__);
    return MATX_ERR_NOT_SUPPORTED;
#else
    if (!A || !out_F) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (A->nrows != A->ncols) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    if (*out_F != NULL)
        ss_factor_dense_z_i8_destroy(alloc, *out_F);

    const matx_int64_t n = A->nrows;
    matx_int64_t lda = ss_packed_lda(A->layout, n, n);

    matx_factor_dense_z_i8_t* F = (matx_factor_dense_z_i8_t*) matx_malloc(alloc, sizeof(*F));
    if (!F) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(F, 0, sizeof(*F));
    F->n = n;
    F->lda = lda;
    F->layout = A->layout;
    F->alloc = *alloc;

    F->lu = (matx_double*) matx_malloc(alloc, (size_t) n * (size_t) n * sizeof(matx_complex_d_t));
    F->piv = (matx_int64_t*) matx_malloc(alloc, n * sizeof(matx_int64_t));
    if (!F->lu || !F->piv) {
        matx_free(alloc, F->lu);
        matx_free(alloc, F->piv);
        matx_free(alloc, F);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    ss_pack_z_i8(A->layout, n, n, A->stride, A->data, (matx_complex_d_t*) F->lu);

    matx_int64_t info = LAPACKE_zgetrf(ss_layout_to_lapack(A->layout),
                                       n,
                                       n,
                                       (lapack_complex_double*) F->lu,
                                       lda,
                                       F->piv);
    if (info != 0) {
        MATX_ERROR("LAPACKE_zgetrf error:%d", info);
        matx_free(alloc, F->lu);
        matx_free(alloc, F->piv);
        matx_free(alloc, F);
        return MATX_ERR_INTERNAL;
    }

    *out_F = F;
    return MATX_OK;
#endif
}

static matx_status_t ss_solve_dense_z_i8(const matx_alloc_t* alloc,
                                         const matx_factor_dense_z_i8_t* F,
                                         const matx_vec_z_i8_t b,
                                         matx_vec_z_i8_t x)
{
    if (!F || !b || !x || !alloc || !alloc->malloc_fn || !alloc->free_fn
        || F->n <= 0 || b->n < F->n || x->n < F->n
        || !b->data || !x->data || b->stride <= 0 || x->stride <= 0
        || (uint64_t) F->n > SIZE_MAX / sizeof(matx_complex_d_t)) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const size_t bytes = (size_t) F->n * sizeof(matx_complex_d_t);
    const int can_solve_in_output
        = x->stride == 1 && (b->data != x->data || b->stride == 1);
    matx_complex_d_t* rhs = x->data;
    int allocated_rhs = 0;
    if (can_solve_in_output) {
        if (b->data != x->data || b->stride != 1) {
            if (b->stride == 1)
                memmove(rhs, b->data, bytes);
            else {
                for (matx_int64_t i = 0; i < F->n; ++i)
                    rhs[i] = b->data[i * b->stride];
            }
        }
    } else {
        rhs = (matx_complex_d_t*) matx_malloc(alloc, bytes);
        if (!rhs) {
            MATX_ERROR("%s: out of memory", __func__);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        allocated_rhs = 1;
        for (matx_int64_t i = 0; i < F->n; ++i)
            rhs[i] = b->data[i * b->stride];
    }

    if (F->n <= MATX_DENSE_SMALL_SOLVE_LIMIT) {
        ss_solve_small_lu_z(F, rhs);
        if (allocated_rhs) {
            for (matx_int64_t i = 0; i < F->n; ++i)
                x->data[i * x->stride] = rhs[i];
            matx_free(alloc, rhs);
        }
        return MATX_OK;
    }

    matx_int64_t ldb = (F->layout == MATX_COL_MAJOR) ? F->n : 1;
    matx_int64_t info = LAPACKE_zgetrs(ss_layout_to_lapack(F->layout),
                                       'N',
                                       F->n,
                                       1,
                                       (lapack_complex_double*) F->lu,
                                       F->lda,
                                       F->piv,
                                       (lapack_complex_double*) rhs,
                                       ldb);
    if (info != 0) {
        MATX_ERROR("LAPACKE_zgetrs error:%d", info);
        if (allocated_rhs)
            matx_free(alloc, rhs);
        return MATX_ERR_INTERNAL;
    }
    if (allocated_rhs) {
        for (matx_int64_t i = 0; i < F->n; ++i)
            x->data[i * x->stride] = rhs[i];
        matx_free(alloc, rhs);
    }
    return MATX_OK;
}

// ---- Cholesky ----

static inline char matx_uplo_to_lapack(matx_uplo_t uplo)
{
    return (uplo == MATX_UPPER) ? 'U' : 'L';
}

static matx_status_t ss_potrf_d_i8(const matx_alloc_t* alloc,
                                   const matx_dense_d_i8_t A,
                                   matx_uplo_t uplo,
                                   matx_factor_dense_d_i8_t** out_F)
{
    if (!A || !out_F) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (A->nrows != A->ncols) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    if (*out_F)
        ss_factor_dense_d_i8_destroy(alloc, *out_F);

    const matx_int64_t n = A->nrows;
    matx_int64_t lda = ss_packed_lda(A->layout, n, n);

    matx_factor_dense_d_i8_t* F = (matx_factor_dense_d_i8_t*) matx_malloc(alloc, sizeof(*F));
    if (!F) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(F, 0, sizeof(*F));
    F->n = n;
    F->lda = lda;
    F->layout = A->layout;
    F->uplo = uplo;
    F->alloc = *alloc;

    F->lu = (matx_double*) matx_malloc(alloc, (size_t) n * (size_t) n * sizeof(matx_double));
    if (!F->lu) {
        MATX_ERROR("%s: out of memory", __func__);
        matx_free(alloc, F);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    ss_pack_d_i8(A->layout, n, n, A->stride, A->data, F->lu);

    matx_int64_t info = LAPACKE_dpotrf(ss_layout_to_lapack(A->layout),
                                       matx_uplo_to_lapack(uplo),
                                       n,
                                       F->lu,
                                       lda);
    if (info != 0) {
        MATX_ERROR("LAPACKE_dpotrf error: %d", info);
        matx_free(alloc, F->lu);
        matx_free(alloc, F);
        return MATX_ERR_INTERNAL;
    }
    *out_F = F;
    return MATX_OK;
}

static matx_status_t ss_potrs_d_i8(const matx_alloc_t* alloc,
                                   const matx_factor_dense_d_i8_t* F,
                                   const matx_double* b,
                                   matx_double* x)
{
    if (!F || !b || !x) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (x != b)
        memmove(x, b, (size_t) F->n * sizeof(matx_double));

    if (F->n <= MATX_DENSE_SMALL_SOLVE_LIMIT) {
        ss_solve_small_chol_d(F, x);
        return MATX_OK;
    }

    matx_int64_t ldb = (F->layout == MATX_COL_MAJOR) ? F->n : 1;
    matx_int64_t info = LAPACKE_dpotrs(ss_layout_to_lapack(F->layout),
                                       matx_uplo_to_lapack(F->uplo),
                                       F->n,
                                       1,
                                       F->lu,
                                       F->lda,
                                       x,
                                       ldb);
    if (info != 0) {
        MATX_ERROR("LAPACKE_dpotrs error: %d", info);
        return MATX_ERR_INTERNAL;
    }
    return MATX_OK;
}

static matx_status_t ss_potrf_z_i8(const matx_alloc_t* alloc,
                                   const matx_dense_z_i8_t A,
                                   matx_uplo_t uplo,
                                   matx_factor_dense_z_i8_t** out_F)
{
    if (!A || !out_F) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (A->nrows != A->ncols) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    if (*out_F)
        ss_factor_dense_z_i8_destroy(alloc, *out_F);

    const matx_int64_t n = A->nrows;
    matx_int64_t lda = ss_packed_lda(A->layout, n, n);

    matx_factor_dense_z_i8_t* F = (matx_factor_dense_z_i8_t*) matx_malloc(alloc, sizeof(*F));
    if (!F) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(F, 0, sizeof(*F));
    F->n = n;
    F->lda = lda;
    F->layout = A->layout;
    F->uplo = uplo;
    F->alloc = *alloc;

    F->lu = (matx_double*) matx_malloc(alloc, (size_t) n * (size_t) n * sizeof(matx_complex_d_t));
    if (!F->lu) {
        MATX_ERROR("%s: out of memory", __func__);
        matx_free(alloc, F);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    ss_pack_z_i8(A->layout, n, n, A->stride, A->data, (matx_complex_d_t*) F->lu);

    matx_int64_t info = LAPACKE_zpotrf(ss_layout_to_lapack(A->layout),
                                       matx_uplo_to_lapack(uplo),
                                       n,
                                       (lapack_complex_double*) F->lu,
                                       lda);
    if (info != 0) {
        MATX_ERROR("LAPACKE_zpotrf error: %d", info);
        matx_free(alloc, F->lu);
        matx_free(alloc, F);
        return MATX_ERR_INTERNAL;
    }
    *out_F = F;
    return MATX_OK;
}

static matx_status_t ss_potrs_z_i8(const matx_alloc_t* alloc,
                                   const matx_factor_dense_z_i8_t* F,
                                   const matx_vec_z_i8_t b,
                                   matx_vec_z_i8_t x)
{
    if (!F || !b || !x || !alloc || !alloc->malloc_fn || !alloc->free_fn
        || F->n <= 0 || b->n < F->n || x->n < F->n
        || !b->data || !x->data || b->stride <= 0 || x->stride <= 0
        || (uint64_t) F->n > SIZE_MAX / sizeof(matx_complex_d_t)) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const size_t bytes = (size_t) F->n * sizeof(matx_complex_d_t);
    const int can_solve_in_output
        = x->stride == 1 && (b->data != x->data || b->stride == 1);
    matx_complex_d_t* rhs = x->data;
    int allocated_rhs = 0;
    if (can_solve_in_output) {
        if (b->data != x->data || b->stride != 1) {
            if (b->stride == 1)
                memmove(rhs, b->data, bytes);
            else {
                for (matx_int64_t i = 0; i < F->n; ++i)
                    rhs[i] = b->data[i * b->stride];
            }
        }
    } else {
        rhs = (matx_complex_d_t*) matx_malloc(alloc, bytes);
        if (!rhs) {
            MATX_ERROR("%s: out of memory", __func__);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        allocated_rhs = 1;
        for (matx_int64_t i = 0; i < F->n; ++i)
            rhs[i] = b->data[i * b->stride];
    }

    if (F->n <= MATX_DENSE_SMALL_SOLVE_LIMIT) {
        ss_solve_small_chol_z(F, rhs);
        if (allocated_rhs) {
            for (matx_int64_t i = 0; i < F->n; ++i)
                x->data[i * x->stride] = rhs[i];
            matx_free(alloc, rhs);
        }
        return MATX_OK;
    }

    matx_int64_t ldb = (F->layout == MATX_COL_MAJOR) ? F->n : 1;
    matx_int64_t info = LAPACKE_zpotrs(ss_layout_to_lapack(F->layout),
                                       matx_uplo_to_lapack(F->uplo),
                                       F->n,
                                       1,
                                       (lapack_complex_double*) F->lu,
                                       F->lda,
                                       (lapack_complex_double*) rhs,
                                       ldb);
    if (info != 0) {
        MATX_ERROR("LAPACKE_zpotrs error: %d", info);
        if (allocated_rhs)
            matx_free(alloc, rhs);
        return MATX_ERR_INTERNAL;
    }
    if (allocated_rhs) {
        for (matx_int64_t i = 0; i < F->n; ++i)
            x->data[i * x->stride] = rhs[i];
        matx_free(alloc, rhs);
    }
    return MATX_OK;
}

// ---- GELS ----

static matx_status_t ss_gels_d_i8(const matx_alloc_t* alloc, const matx_dense_d_i8_t A, const matx_double* b, matx_double* x)
{
    if (!A || !b || !x) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const matx_int64_t m = A->nrows, n = A->ncols;
    const matx_int64_t lda = ss_packed_lda(A->layout, m, n);
    const matx_int64_t blen = (m > n) ? m : n;
    const matx_int64_t ldb = (A->layout == MATX_COL_MAJOR) ? blen : 1;

    matx_double* Acopy = (matx_double*) matx_malloc(alloc, (size_t) m * (size_t) n * sizeof(matx_double));
    matx_double* bcopy = (matx_double*) matx_malloc(alloc, blen * sizeof(matx_double));
    if (!Acopy || !bcopy) {
        MATX_ERROR("%s: out of memory", __func__);
        matx_free(alloc, Acopy);
        matx_free(alloc, bcopy);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(bcopy, 0, blen * sizeof(matx_double));

    ss_pack_d_i8(A->layout, m, n, A->stride, A->data, Acopy);
    memcpy(bcopy, b, m * sizeof(matx_double));

    matx_int64_t info
        = LAPACKE_dgels(ss_layout_to_lapack(A->layout), 'N', m, n, 1, Acopy, lda, bcopy, ldb);
    if (info != 0) {
        MATX_ERROR("LAPACKE_dgels error: %d", info);
        matx_free(alloc, Acopy);
        matx_free(alloc, bcopy);
        return MATX_ERR_INTERNAL;
    }
    memcpy(x, bcopy, n * sizeof(matx_double));
    matx_free(alloc, Acopy);
    matx_free(alloc, bcopy);
    return MATX_OK;
}

static matx_status_t ss_gels_z_i8(const matx_alloc_t* alloc,
                                  const matx_dense_z_i8_t A,
                                  const matx_vec_z_i8_t b,
                                  matx_vec_z_i8_t x)
{
    if (!A || !b || !x || !alloc || !alloc->malloc_fn || !alloc->free_fn
        || A->nrows <= 0 || A->ncols <= 0 || !A->data
        || b->n < A->nrows || x->n < A->ncols
        || !b->data || !x->data || b->stride <= 0 || x->stride <= 0) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const matx_int64_t m = A->nrows, n = A->ncols;
    const matx_int64_t lda = ss_packed_lda(A->layout, m, n);
    const matx_int64_t blen = (m > n) ? m : n;
    const matx_int64_t ldb = (A->layout == MATX_COL_MAJOR) ? blen : 1;

    if ((uint64_t) m > SIZE_MAX / (uint64_t) n
        || (size_t) m * (size_t) n > SIZE_MAX / sizeof(matx_complex_d_t)
        || (uint64_t) blen > SIZE_MAX / sizeof(matx_complex_d_t)) {
        MATX_ERROR("%s: matrix size overflow", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_complex_d_t* Acopy = (matx_complex_d_t*) matx_malloc(alloc, (size_t) m * (size_t) n
                                                         * sizeof(matx_complex_d_t));
    matx_complex_d_t* bcopy = (matx_complex_d_t*) matx_malloc(alloc, (size_t) blen * sizeof(matx_complex_d_t));
    if (!Acopy || !bcopy) {
        MATX_ERROR("%s: out of memory", __func__);
        matx_free(alloc, Acopy);
        matx_free(alloc, bcopy);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(bcopy, 0, blen * sizeof(matx_complex_d_t));

    ss_pack_z_i8(A->layout, m, n, A->stride, A->data, Acopy);
    for (matx_int64_t i = 0; i < m; ++i)
        bcopy[i] = b->data[i * b->stride];

    matx_int64_t info = LAPACKE_zgels(ss_layout_to_lapack(A->layout),
                                      'N',
                                      m,
                                      n,
                                      1,
                                      (lapack_complex_double*) Acopy,
                                      lda,
                                      (lapack_complex_double*) bcopy,
                                      ldb);
    if (info != 0) {
        MATX_ERROR("LAPACKE_zgels error: %d", info);
        matx_free(alloc, Acopy);
        matx_free(alloc, bcopy);
        return MATX_ERR_INTERNAL;
    }
    for (matx_int64_t i = 0; i < n; ++i)
        x->data[i * x->stride] = bcopy[i];
    matx_free(alloc, Acopy);
    matx_free(alloc, bcopy);
    return MATX_OK;
}

// ---- SYEV ----

static matx_status_t ss_syev_d_i8(const matx_alloc_t* alloc,
                                  const matx_dense_d_i8_t A,
                                  matx_vec_d_i8_t eigenvalues,
                                  matx_dense_d_i8_t* eigenvectors)
{
    if (!A || !eigenvalues) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (A->nrows != A->ncols) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const matx_int64_t n = A->nrows;
    if (eigenvalues->n != n) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    const char jobz = eigenvectors ? 'V' : 'N';
    matx_int64_t lda = ss_packed_lda(A->layout, n, n);

    matx_double* Acopy = (matx_double*) matx_malloc(alloc, (size_t) n * (size_t) n * sizeof(matx_double));
    if (!Acopy) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    ss_pack_d_i8(A->layout, n, n, A->stride, A->data, Acopy);

    matx_int64_t info
        = LAPACKE_dsyev(ss_layout_to_lapack(A->layout), jobz, 'L', n, Acopy, lda, eigenvalues->data);
    if (info != 0) {
        MATX_ERROR("LAPACKE_dsyev error: %d", info);
        matx_free(alloc, Acopy);
        return MATX_ERR_INTERNAL;
    }
    if (eigenvectors) {
        matx_dense_d_i8_opaque_t* ev = (matx_dense_d_i8_opaque_t*) matx_malloc(alloc,
            sizeof(matx_dense_d_i8_opaque_t));
        if (!ev) {
            MATX_ERROR("%s: out of memory", __func__);
            matx_free(alloc, Acopy);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        memset(ev, 0, sizeof(*ev));
        ev->alloc = *alloc;
        ev->nrows = n;
        ev->ncols = n;
        ev->layout = A->layout;
        ev->stride = lda;
        ev->flags = 1u;
        ev->data = Acopy;
        if (*eigenvectors) {
            if ((*eigenvectors)->flags & 1u)
                matx_free(alloc, (*eigenvectors)->data);
            matx_free(alloc, *eigenvectors);
        }
        *eigenvectors = ev;
    } else {
        matx_free(alloc, Acopy);
    }
    return MATX_OK;
}

// ---- GESVD ----

static matx_status_t ss_gesvd_d_i8(const matx_alloc_t* alloc,
                                   const matx_dense_d_i8_t A,
                                   matx_vec_d_i8_t S,
                                   matx_dense_d_i8_t* U,
                                   matx_dense_d_i8_t* Vt)
{
    if (!A || !S) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const matx_int64_t m = A->nrows, n = A->ncols;
    const matx_int64_t k = (m < n) ? m : n;
    if (S->n != k) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    matx_int64_t lda = ss_packed_lda(A->layout, m, n);

    matx_double* Acopy = (matx_double*) matx_malloc(alloc, (size_t) m * (size_t) n * sizeof(matx_double));
    matx_double* u_data = U ? (matx_double*) matx_malloc(alloc, (size_t) m * (size_t) m * sizeof(matx_double))
                            : NULL;
    matx_double* vt_data = Vt ? (matx_double*) matx_malloc(alloc, (size_t) n * (size_t) n * sizeof(matx_double))
                              : NULL;
    matx_double* superb = (matx_double*) matx_malloc(alloc, k * sizeof(matx_double));
    if (!Acopy || !superb || (U && !u_data) || (Vt && !vt_data)) {
        matx_free(alloc, Acopy);
        matx_free(alloc, u_data);
        matx_free(alloc, vt_data);
        matx_free(alloc, superb);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    ss_pack_d_i8(A->layout, m, n, A->stride, A->data, Acopy);

    const char jobu = U ? 'A' : 'N';
    const char jobvt = Vt ? 'A' : 'N';
    matx_int64_t ldu = m, ldvt = n;
    matx_int64_t info = LAPACKE_dgesvd(ss_layout_to_lapack(A->layout),
                                       jobu,
                                       jobvt,
                                       m,
                                       n,
                                       Acopy,
                                       lda,
                                       S->data,
                                       u_data,
                                       ldu,
                                       vt_data,
                                       ldvt,
                                       superb);
    matx_free(alloc, Acopy);
    matx_free(alloc, superb);
    if (info != 0) {
        MATX_ERROR("LAPACKE_dgesvd error: %d", info);
        matx_free(alloc, u_data);
        matx_free(alloc, vt_data);
        return MATX_ERR_INTERNAL;
    }
    if (U) {
        matx_int64_t u_lda = ss_packed_lda(A->layout, m, m);
        matx_dense_d_i8_opaque_t* ev = (matx_dense_d_i8_opaque_t*) matx_malloc(alloc,
            sizeof(matx_dense_d_i8_opaque_t));
        if (!ev) {
            MATX_ERROR("%s: out of memory", __func__);
            matx_free(alloc, u_data);
            matx_free(alloc, vt_data);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        memset(ev, 0, sizeof(*ev));
        ev->alloc = *alloc;
        ev->nrows = m;
        ev->ncols = m;
        ev->layout = A->layout;
        ev->stride = u_lda;
        ev->flags = 1u;
        ev->data = u_data;
        if (*U) {
            if ((*U)->flags & 1u)
                matx_free(alloc, (*U)->data);
            matx_free(alloc, *U);
        }
        *U = ev;
    }
    if (Vt) {
        matx_int64_t vt_lda = ss_packed_lda(A->layout, n, n);
        matx_dense_d_i8_opaque_t* ev = (matx_dense_d_i8_opaque_t*) matx_malloc(alloc,
            sizeof(matx_dense_d_i8_opaque_t));
        if (!ev) {
            MATX_ERROR("%s: out of memory", __func__);
            matx_free(alloc, vt_data);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        memset(ev, 0, sizeof(*ev));
        ev->alloc = *alloc;
        ev->nrows = n;
        ev->ncols = n;
        ev->layout = A->layout;
        ev->stride = vt_lda;
        ev->flags = 1u;
        ev->data = vt_data;
        if (*Vt) {
            if ((*Vt)->flags & 1u)
                matx_free(alloc, (*Vt)->data);
            matx_free(alloc, *Vt);
        }
        *Vt = ev;
    }
    return MATX_OK;
}

static matx_status_t ss_gesvd_z_i8(const matx_alloc_t* alloc,
                                   const matx_dense_z_i8_t A,
                                   matx_vec_d_i8_t S,
                                   matx_dense_z_i8_t* U,
                                   matx_dense_z_i8_t* Vt)
{
    if (!A || !S) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const matx_int64_t m = A->nrows, n = A->ncols;
    const matx_int64_t k = (m < n) ? m : n;
    if (S->n != k) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    matx_int64_t lda = ss_packed_lda(A->layout, m, n);

    matx_complex_d_t* Acopy = (matx_complex_d_t*) matx_malloc(alloc, (size_t) m * (size_t) n
                                                         * sizeof(matx_complex_d_t));
    matx_complex_d_t* u_data = U ? (matx_complex_d_t*) matx_malloc(alloc, (size_t) m * (size_t) m
                                                              * sizeof(matx_complex_d_t))
                                 : NULL;
    matx_complex_d_t* vt_data = Vt ? (matx_complex_d_t*) matx_malloc(alloc, (size_t) n * (size_t) n
                                                                * sizeof(matx_complex_d_t))
                                   : NULL;
    matx_double* superb = (matx_double*) matx_malloc(alloc, k * sizeof(matx_double));
    if (!Acopy || !superb || (U && !u_data) || (Vt && !vt_data)) {
        matx_free(alloc, Acopy);
        matx_free(alloc, u_data);
        matx_free(alloc, vt_data);
        matx_free(alloc, superb);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    ss_pack_z_i8(A->layout, m, n, A->stride, A->data, Acopy);

    const char jobu = U ? 'A' : 'N';
    const char jobvt = Vt ? 'A' : 'N';
    matx_int64_t ldu = m, ldvt = n;
    matx_int64_t info = LAPACKE_zgesvd(ss_layout_to_lapack(A->layout),
                                       jobu,
                                       jobvt,
                                       m,
                                       n,
                                       (lapack_complex_double*) Acopy,
                                       lda,
                                       S->data,
                                       (lapack_complex_double*) u_data,
                                       ldu,
                                       (lapack_complex_double*) vt_data,
                                       ldvt,
                                       superb);
    matx_free(alloc, Acopy);
    matx_free(alloc, superb);
    if (info != 0) {
        MATX_ERROR("LAPACKE_zgesvd error: %d", info);
        matx_free(alloc, u_data);
        matx_free(alloc, vt_data);
        return MATX_ERR_INTERNAL;
    }
    if (U) {
        matx_int64_t u_lda = ss_packed_lda(A->layout, m, m);
        matx_dense_z_i8_opaque_t* ev = (matx_dense_z_i8_opaque_t*) matx_malloc(alloc,
            sizeof(matx_dense_z_i8_opaque_t));
        if (!ev) {
            MATX_ERROR("%s: out of memory", __func__);
            matx_free(alloc, u_data);
            matx_free(alloc, vt_data);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        memset(ev, 0, sizeof(*ev));
        ev->alloc = *alloc;
        ev->nrows = m;
        ev->ncols = m;
        ev->layout = A->layout;
        ev->stride = u_lda;
        ev->flags = 1u;
        ev->data = u_data;
        if (*U) {
            if ((*U)->flags & 1u)
                matx_free(alloc, (*U)->data);
            matx_free(alloc, *U);
        }
        *U = ev;
    }
    if (Vt) {
        matx_int64_t vt_lda = ss_packed_lda(A->layout, n, n);
        matx_dense_z_i8_opaque_t* ev = (matx_dense_z_i8_opaque_t*) matx_malloc(alloc,
            sizeof(matx_dense_z_i8_opaque_t));
        if (!ev) {
            MATX_ERROR("%s: out of memory", __func__);
            matx_free(alloc, vt_data);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        memset(ev, 0, sizeof(*ev));
        ev->alloc = *alloc;
        ev->nrows = n;
        ev->ncols = n;
        ev->layout = A->layout;
        ev->stride = vt_lda;
        ev->flags = 1u;
        ev->data = vt_data;
        if (*Vt) {
            if ((*Vt)->flags & 1u)
                matx_free(alloc, (*Vt)->data);
            matx_free(alloc, *Vt);
        }
        *Vt = ev;
    }
    return MATX_OK;
}

// ---- SYEV (complex Hermitian) ----

static matx_status_t ss_syev_z_i8(const matx_alloc_t* alloc,
                                  const matx_dense_z_i8_t A,
                                  matx_vec_d_i8_t eigenvalues,
                                  matx_dense_z_i8_t* eigenvectors)
{
    if (!A || !eigenvalues) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (A->nrows != A->ncols) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const matx_int64_t n = A->nrows;
    if (eigenvalues->n != n) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    const char jobz = eigenvectors ? 'V' : 'N';
    matx_int64_t lda = ss_packed_lda(A->layout, n, n);

    matx_complex_d_t* Acopy = (matx_complex_d_t*) matx_malloc(alloc, (size_t) n * (size_t) n
                                                         * sizeof(matx_complex_d_t));
    if (!Acopy) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    ss_pack_z_i8(A->layout, n, n, A->stride, A->data, Acopy);

    matx_int64_t info = LAPACKE_zheev(ss_layout_to_lapack(A->layout),
                                      jobz,
                                      'L',
                                      n,
                                      (lapack_complex_double*) Acopy,
                                      lda,
                                      eigenvalues->data);
    if (info != 0) {
        MATX_ERROR("LAPACKE_zheev error: %d", (int) info);
        matx_free(alloc, Acopy);
        return MATX_ERR_INTERNAL;
    }
    if (eigenvectors) {
        matx_dense_z_i8_opaque_t* ev = (matx_dense_z_i8_opaque_t*) matx_malloc(alloc,
            sizeof(matx_dense_z_i8_opaque_t));
        if (!ev) {
            matx_free(alloc, Acopy);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        memset(ev, 0, sizeof(*ev));
        ev->alloc = *alloc;
        ev->nrows = n;
        ev->ncols = n;
        ev->layout = A->layout;
        ev->stride = lda;
        ev->flags = 1u;
        ev->data = Acopy;
        if (*eigenvectors) {
            if ((*eigenvectors)->flags & 1u)
                matx_free(alloc, (*eigenvectors)->data);
            matx_free(alloc, *eigenvectors);
        }
        *eigenvectors = ev;
    } else {
        matx_free(alloc, Acopy);
    }
    return MATX_OK;
}

// ---- GEEV (general eigenvalues, real) ----

static matx_status_t ss_geev_d_i8(const matx_alloc_t* alloc,
                                  const matx_dense_d_i8_t A,
                                  matx_vec_z_i8_t eigenvalues,
                                  matx_dense_d_i8_t* vr,
                                  matx_dense_d_i8_t* vl)
{
    if (!A || !eigenvalues) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (A->nrows != A->ncols) {
        MATX_ERROR("%s: matrix must be square", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const matx_int64_t n = A->nrows;
    if (eigenvalues->n != n) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    const char jobvl = vl ? 'V' : 'N';
    const char jobvr = vr ? 'V' : 'N';
    matx_int64_t lda = ss_packed_lda(A->layout, n, n);

    matx_double* Acopy = (matx_double*) matx_malloc(alloc, (size_t) n * (size_t) n * sizeof(matx_double));
    matx_double* wr = (matx_double*) matx_malloc(alloc, (size_t) n * sizeof(matx_double));
    matx_double* wi = (matx_double*) matx_malloc(alloc, (size_t) n * sizeof(matx_double));
    matx_double* vl_data = vl ? (matx_double*) matx_malloc(alloc, (size_t) n * (size_t) n * sizeof(matx_double))
                              : NULL;
    matx_double* vr_data = vr ? (matx_double*) matx_malloc(alloc, (size_t) n * (size_t) n * sizeof(matx_double))
                              : NULL;
    if (!Acopy || !wr || !wi || (vl && !vl_data) || (vr && !vr_data)) {
        matx_free(alloc, Acopy);
        matx_free(alloc, wr);
        matx_free(alloc, wi);
        matx_free(alloc, vl_data);
        matx_free(alloc, vr_data);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    ss_pack_d_i8(A->layout, n, n, A->stride, A->data, Acopy);

    matx_int64_t info = LAPACKE_dgeev(
        ss_layout_to_lapack(A->layout), jobvl, jobvr, n, Acopy, lda, wr, wi, vl_data, n, vr_data, n);
    if (info != 0) {
        MATX_ERROR("LAPACKE_dgeev error: %d", (int) info);
        matx_free(alloc, Acopy);
        matx_free(alloc, wr);
        matx_free(alloc, wi);
        matx_free(alloc, vl_data);
        matx_free(alloc, vr_data);
        return MATX_ERR_INTERNAL;
    }

    /* Pack wr + wi into complex eigenvalues */
    for (matx_int64_t i = 0; i < n; ++i) {
        eigenvalues->data[i * eigenvalues->stride].real = wr[i];
        eigenvalues->data[i * eigenvalues->stride].imag = wi[i];
    }

    /* Build vr (right eigenvectors) */
    if (vr) {
        matx_dense_d_i8_opaque_t* ev = (matx_dense_d_i8_opaque_t*) matx_malloc(alloc,
            sizeof(matx_dense_d_i8_opaque_t));
        if (!ev) {
            matx_free(alloc, vr_data);
            matx_free(alloc, Acopy);
            matx_free(alloc, wr);
            matx_free(alloc, wi);
            matx_free(alloc, vl_data);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        memset(ev, 0, sizeof(*ev));
        ev->alloc = *alloc;
        ev->nrows = n;
        ev->ncols = n;
        ev->layout = A->layout;
        ev->stride = n;
        ev->flags = 1u;
        ev->data = vr_data;
        if (*vr) {
            if ((*vr)->flags & 1u)
                matx_free(alloc, (*vr)->data);
            matx_free(alloc, *vr);
        }
        *vr = ev;
    }
    if (vl) {
        matx_dense_d_i8_opaque_t* ev = (matx_dense_d_i8_opaque_t*) matx_malloc(alloc,
            sizeof(matx_dense_d_i8_opaque_t));
        if (!ev) {
            if (vr && vr_data)
                matx_free(alloc, vr_data);
            matx_free(alloc, Acopy);
            matx_free(alloc, wr);
            matx_free(alloc, wi);
            matx_free(alloc, vl_data);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        memset(ev, 0, sizeof(*ev));
        ev->alloc = *alloc;
        ev->nrows = n;
        ev->ncols = n;
        ev->layout = A->layout;
        ev->stride = n;
        ev->flags = 1u;
        ev->data = vl_data;
        if (*vl) {
            if ((*vl)->flags & 1u)
                matx_free(alloc, (*vl)->data);
            matx_free(alloc, *vl);
        }
        *vl = ev;
    }
    matx_free(alloc, Acopy);
    matx_free(alloc, wr);
    matx_free(alloc, wi);
    return MATX_OK;
}

// ---- GEEV (general eigenvalues, complex) ----

static matx_status_t ss_geev_z_i8(const matx_alloc_t* alloc,
                                  const matx_dense_z_i8_t A,
                                  matx_vec_z_i8_t eigenvalues,
                                  matx_dense_z_i8_t* vr,
                                  matx_dense_z_i8_t* vl)
{
    if (!A || !eigenvalues) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (A->nrows != A->ncols) {
        MATX_ERROR("%s: matrix must be square", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const matx_int64_t n = A->nrows;
    if (eigenvalues->n != n) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    const char jobvl = vl ? 'V' : 'N';
    const char jobvr = vr ? 'V' : 'N';
    matx_int64_t lda = ss_packed_lda(A->layout, n, n);

    matx_complex_d_t* Acopy = (matx_complex_d_t*) matx_malloc(alloc, (size_t) n * (size_t) n
                                                         * sizeof(matx_complex_d_t));
    matx_complex_d_t* w = (matx_complex_d_t*) matx_malloc(alloc, (size_t) n * sizeof(matx_complex_d_t));
    matx_complex_d_t* vl_data = vl ? (matx_complex_d_t*) matx_malloc(alloc, (size_t) n * (size_t) n
                                                                * sizeof(matx_complex_d_t))
                                   : NULL;
    matx_complex_d_t* vr_data = vr ? (matx_complex_d_t*) matx_malloc(alloc, (size_t) n * (size_t) n
                                                                * sizeof(matx_complex_d_t))
                                   : NULL;
    if (!Acopy || !w || (vl && !vl_data) || (vr && !vr_data)) {
        matx_free(alloc, Acopy);
        matx_free(alloc, w);
        matx_free(alloc, vl_data);
        matx_free(alloc, vr_data);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    ss_pack_z_i8(A->layout, n, n, A->stride, A->data, Acopy);

    matx_int64_t info = LAPACKE_zgeev(ss_layout_to_lapack(A->layout),
                                      jobvl,
                                      jobvr,
                                      n,
                                      (lapack_complex_double*) Acopy,
                                      lda,
                                      (lapack_complex_double*) w,
                                      (lapack_complex_double*) vl_data,
                                      n,
                                      (lapack_complex_double*) vr_data,
                                      n);
    if (info != 0) {
        MATX_ERROR("LAPACKE_zgeev error: %d", (int) info);
        matx_free(alloc, Acopy);
        matx_free(alloc, w);
        matx_free(alloc, vl_data);
        matx_free(alloc, vr_data);
        return MATX_ERR_INTERNAL;
    }

    for (matx_int64_t i = 0; i < n; ++i)
        eigenvalues->data[i * eigenvalues->stride] = w[i];

    if (vr) {
        matx_dense_z_i8_opaque_t* ev = (matx_dense_z_i8_opaque_t*) matx_malloc(alloc,
            sizeof(matx_dense_z_i8_opaque_t));
        if (!ev) {
            matx_free(alloc, vr_data);
            matx_free(alloc, Acopy);
            matx_free(alloc, w);
            matx_free(alloc, vl_data);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        memset(ev, 0, sizeof(*ev));
        ev->alloc = *alloc;
        ev->nrows = n;
        ev->ncols = n;
        ev->layout = A->layout;
        ev->stride = n;
        ev->flags = 1u;
        ev->data = vr_data;
        if (*vr) {
            if ((*vr)->flags & 1u)
                matx_free(alloc, (*vr)->data);
            matx_free(alloc, *vr);
        }
        *vr = ev;
    }
    if (vl) {
        matx_dense_z_i8_opaque_t* ev = (matx_dense_z_i8_opaque_t*) matx_malloc(alloc,
            sizeof(matx_dense_z_i8_opaque_t));
        if (!ev) {
            if (vr && vr_data)
                matx_free(alloc, vr_data);
            matx_free(alloc, Acopy);
            matx_free(alloc, w);
            matx_free(alloc, vl_data);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        memset(ev, 0, sizeof(*ev));
        ev->alloc = *alloc;
        ev->nrows = n;
        ev->ncols = n;
        ev->layout = A->layout;
        ev->stride = n;
        ev->flags = 1u;
        ev->data = vl_data;
        if (*vl) {
            if ((*vl)->flags & 1u)
                matx_free(alloc, (*vl)->data);
            matx_free(alloc, *vl);
        }
        *vl = ev;
    }
    matx_free(alloc, Acopy);
    matx_free(alloc, w);
    return MATX_OK;
}

// ---- QR Factorization ----

static matx_status_t ss_qr_d_i8(const matx_alloc_t* alloc,
                                const matx_dense_d_i8_t A,
                                matx_dense_d_i8_t* Q,
                                matx_dense_d_i8_t* R)
{
    if (!A || !Q || !R) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const matx_int64_t m = A->nrows, n = A->ncols;
    const matx_int64_t k = (m < n) ? m : n;

    matx_int64_t lda = ss_packed_lda(A->layout, m, n);

    matx_double* Acopy = (matx_double*) matx_malloc(alloc, (size_t) m * (size_t) n * sizeof(matx_double));
    matx_double* tau = (matx_double*) matx_malloc(alloc, (size_t) k * sizeof(matx_double));
    if (!Acopy || !tau) {
        matx_free(alloc, Acopy);
        matx_free(alloc, tau);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    ss_pack_d_i8(A->layout, m, n, A->stride, A->data, Acopy);

    matx_int64_t info = LAPACKE_dgeqrf(ss_layout_to_lapack(A->layout), m, n, Acopy, lda, tau);
    if (info != 0) {
        MATX_ERROR("LAPACKE_dgeqrf error: %d", (int) info);
        matx_free(alloc, Acopy);
        matx_free(alloc, tau);
        return MATX_ERR_INTERNAL;
    }

    /* Extract R: upper triangular part */
    matx_double* r_data = (matx_double*) matx_malloc(alloc, (size_t) n * (size_t) n * sizeof(matx_double));
    if (!r_data) {
        matx_free(alloc, Acopy);
        matx_free(alloc, tau);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(r_data, 0, (size_t) n * (size_t) n * sizeof(matx_double));
    for (matx_int64_t j = 0; j < n; ++j)
        for (matx_int64_t i = 0; i <= j && i < m; ++i) {
            matx_int64_t idx = (A->layout == MATX_COL_MAJOR) ? i + j * lda : j + i * lda;
            matx_int64_t ridx = (A->layout == MATX_COL_MAJOR) ? i + j * n : j + i * n;
            r_data[ridx] = Acopy[idx];
        }

    /* Generate Q */
    info = LAPACKE_dorgqr(ss_layout_to_lapack(A->layout), m, k, k, Acopy, lda, tau);
    if (info != 0) {
        MATX_ERROR("LAPACKE_dorgqr error: %d", (int) info);
        matx_free(alloc, Acopy);
        matx_free(alloc, tau);
        matx_free(alloc, r_data);
        return MATX_ERR_INTERNAL;
    }
    matx_free(alloc, tau);

    /* Q = Acopy (m x k, leading columns) */
    {
        matx_dense_d_i8_opaque_t* qe = (matx_dense_d_i8_opaque_t*) matx_malloc(alloc,
            sizeof(matx_dense_d_i8_opaque_t));
        if (!qe) {
            matx_free(alloc, Acopy);
            matx_free(alloc, r_data);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        memset(qe, 0, sizeof(*qe));
        qe->alloc = *alloc;
        qe->nrows = m;
        qe->ncols = k;
        qe->layout = A->layout;
        qe->stride = lda;
        qe->flags = 1u;
        qe->data = Acopy;
        if (*Q) {
            if ((*Q)->flags & 1u)
                matx_free(alloc, (*Q)->data);
            matx_free(alloc, *Q);
        }
        *Q = qe;
    }

    /* R = r_data (k x n) */
    {
        matx_dense_d_i8_opaque_t* re = (matx_dense_d_i8_opaque_t*) matx_malloc(alloc,
            sizeof(matx_dense_d_i8_opaque_t));
        if (!re) {
            matx_free(alloc, r_data);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        memset(re, 0, sizeof(*re));
        re->alloc = *alloc;
        re->nrows = k;
        re->ncols = n;
        re->layout = A->layout;
        re->stride = (A->layout == MATX_COL_MAJOR) ? k : n;
        re->flags = 1u;
        re->data = r_data;
        if (*R) {
            if ((*R)->flags & 1u)
                matx_free(alloc, (*R)->data);
            matx_free(alloc, *R);
        }
        *R = re;
    }
    return MATX_OK;
}

// ---- QR Factorization (complex) ----

static matx_status_t ss_qr_z_i8(const matx_alloc_t* alloc,
                                const matx_dense_z_i8_t A,
                                matx_dense_z_i8_t* Q,
                                matx_dense_z_i8_t* R)
{
    if (!A || !Q || !R) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const matx_int64_t m = A->nrows, n = A->ncols;
    const matx_int64_t k = (m < n) ? m : n;

    matx_int64_t lda = ss_packed_lda(A->layout, m, n);

    matx_complex_d_t* Acopy = (matx_complex_d_t*) matx_malloc(alloc, (size_t) m * (size_t) n
                                                         * sizeof(matx_complex_d_t));
    matx_complex_d_t* tau = (matx_complex_d_t*) matx_malloc(alloc, (size_t) k * sizeof(matx_complex_d_t));
    if (!Acopy || !tau) {
        matx_free(alloc, Acopy);
        matx_free(alloc, tau);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    ss_pack_z_i8(A->layout, m, n, A->stride, A->data, Acopy);

    matx_int64_t info = LAPACKE_zgeqrf(ss_layout_to_lapack(A->layout),
                                       m,
                                       n,
                                       (lapack_complex_double*) Acopy,
                                       lda,
                                       (lapack_complex_double*) tau);
    if (info != 0) {
        MATX_ERROR("LAPACKE_zgeqrf error: %d", (int) info);
        matx_free(alloc, Acopy);
        matx_free(alloc, tau);
        return MATX_ERR_INTERNAL;
    }

    matx_complex_d_t* r_data = (matx_complex_d_t*) matx_malloc(alloc, (size_t) n * (size_t) n
                                                          * sizeof(matx_complex_d_t));
    if (!r_data) {
        matx_free(alloc, Acopy);
        matx_free(alloc, tau);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(r_data, 0, (size_t) n * (size_t) n * sizeof(matx_complex_d_t));
    for (matx_int64_t j = 0; j < n; ++j)
        for (matx_int64_t i = 0; i <= j && i < m; ++i) {
            matx_int64_t idx = (A->layout == MATX_COL_MAJOR) ? i + j * lda : j + i * lda;
            matx_int64_t ridx = (A->layout == MATX_COL_MAJOR) ? i + j * n : j + i * n;
            r_data[ridx] = Acopy[idx];
        }

    info = LAPACKE_zungqr(ss_layout_to_lapack(A->layout),
                          m,
                          k,
                          k,
                          (lapack_complex_double*) Acopy,
                          lda,
                          (lapack_complex_double*) tau);
    if (info != 0) {
        MATX_ERROR("LAPACKE_zungqr error: %d", (int) info);
        matx_free(alloc, Acopy);
        matx_free(alloc, tau);
        matx_free(alloc, r_data);
        return MATX_ERR_INTERNAL;
    }
    matx_free(alloc, tau);

    {
        matx_dense_z_i8_opaque_t* qe = (matx_dense_z_i8_opaque_t*) matx_malloc(alloc,
            sizeof(matx_dense_z_i8_opaque_t));
        if (!qe) {
            matx_free(alloc, Acopy);
            matx_free(alloc, r_data);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        memset(qe, 0, sizeof(*qe));
        qe->alloc = *alloc;
        qe->nrows = m;
        qe->ncols = k;
        qe->layout = A->layout;
        qe->stride = lda;
        qe->flags = 1u;
        qe->data = Acopy;
        if (*Q) {
            if ((*Q)->flags & 1u)
                matx_free(alloc, (*Q)->data);
            matx_free(alloc, *Q);
        }
        *Q = qe;
    }

    {
        matx_dense_z_i8_opaque_t* re = (matx_dense_z_i8_opaque_t*) matx_malloc(alloc,
            sizeof(matx_dense_z_i8_opaque_t));
        if (!re) {
            matx_free(alloc, r_data);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        memset(re, 0, sizeof(*re));
        re->alloc = *alloc;
        re->nrows = k;
        re->ncols = n;
        re->layout = A->layout;
        re->stride = (A->layout == MATX_COL_MAJOR) ? k : n;
        re->flags = 1u;
        re->data = r_data;
        if (*R) {
            if ((*R)->flags & 1u)
                matx_free(alloc, (*R)->data);
            matx_free(alloc, *R);
        }
        *R = re;
    }
    return MATX_OK;
}

// ---- Matrix determinant (via LU) ----

static matx_status_t ss_det_dense_d_i8(const matx_alloc_t* alloc, const matx_dense_d_i8_t A, matx_double* det)
{
    if (!A || !det) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (A->nrows != A->ncols) {
        MATX_ERROR("%s: matrix must be square", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const matx_int64_t n = A->nrows;

    /* Small-n explicit formulas avoid LAPACK allocation overhead. */
    if (n == 2) {
        const matx_double a00 = (A->layout == MATX_COL_MAJOR)
            ? A->data[0] : A->data[0];
        const matx_double a01 = (A->layout == MATX_COL_MAJOR)
            ? A->data[A->stride] : A->data[1];
        const matx_double a10 = (A->layout == MATX_COL_MAJOR)
            ? A->data[1] : A->data[A->stride];
        const matx_double a11 = (A->layout == MATX_COL_MAJOR)
            ? A->data[1 + A->stride] : A->data[A->stride + 1];
        *det = a00 * a11 - a01 * a10;
        return MATX_OK;
    }
    if (n == 3) {
        const matx_int64_t s = A->stride;
        const matx_double a00 = (A->layout == MATX_COL_MAJOR) ? A->data[0]      : A->data[0];
        const matx_double a01 = (A->layout == MATX_COL_MAJOR) ? A->data[s]      : A->data[1];
        const matx_double a02 = (A->layout == MATX_COL_MAJOR) ? A->data[2*s]    : A->data[2];
        const matx_double a10 = (A->layout == MATX_COL_MAJOR) ? A->data[1]      : A->data[s];
        const matx_double a11 = (A->layout == MATX_COL_MAJOR) ? A->data[1+s]    : A->data[s+1];
        const matx_double a12 = (A->layout == MATX_COL_MAJOR) ? A->data[1+2*s]  : A->data[s+2];
        const matx_double a20 = (A->layout == MATX_COL_MAJOR) ? A->data[2]      : A->data[2*s];
        const matx_double a21 = (A->layout == MATX_COL_MAJOR) ? A->data[2+s]    : A->data[2*s+1];
        const matx_double a22 = (A->layout == MATX_COL_MAJOR) ? A->data[2+2*s]  : A->data[2*s+2];
        *det = a00 * (a11 * a22 - a12 * a21)
             - a01 * (a10 * a22 - a12 * a20)
             + a02 * (a10 * a21 - a11 * a20);
        return MATX_OK;
    }
    matx_int64_t lda = ss_packed_lda(A->layout, n, n);

    matx_double* Acopy = (matx_double*) matx_malloc(alloc, (size_t) n * (size_t) n * sizeof(matx_double));
    matx_int64_t* piv = (matx_int64_t*) matx_malloc(alloc, (size_t) n * sizeof(matx_int64_t));
    if (!Acopy || !piv) {
        matx_free(alloc, Acopy);
        matx_free(alloc, piv);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    ss_pack_d_i8(A->layout, n, n, A->stride, A->data, Acopy);

    matx_int64_t info = LAPACKE_dgetrf(ss_layout_to_lapack(A->layout), n, n, Acopy, lda, piv);
    if (info < 0) {
        MATX_ERROR("LAPACKE_dgetrf error: %d", (int) info);
        matx_free(alloc, Acopy);
        matx_free(alloc, piv);
        return MATX_ERR_INTERNAL;
    }
    if (info > 0) {
        /* Singular matrix, determinant is zero */
        *det = 0.0;
        matx_free(alloc, Acopy);
        matx_free(alloc, piv);
        return MATX_OK;
    }

    matx_double d = 1.0;
    matx_int64_t sign = 1;
    for (matx_int64_t i = 0; i < n; ++i) {
        if (piv[i] != i + 1)
            sign = -sign;
        d *= Acopy[i + i * lda];
    }
    *det = (matx_double) sign * d;
    matx_free(alloc, Acopy);
    matx_free(alloc, piv);
    return MATX_OK;
}

// ---- Matrix determinant (complex) ----

static matx_status_t ss_det_dense_z_i8(const matx_alloc_t* alloc, const matx_dense_z_i8_t A, matx_complex_d_t* det)
{
    if (!A || !det) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (A->nrows != A->ncols) {
        MATX_ERROR("%s: matrix must be square", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const matx_int64_t n = A->nrows;

    /* Small-n explicit formulas avoid LAPACK allocation overhead. */
    if (n == 2) {
        const matx_complex_d_t a00 = (A->layout == MATX_COL_MAJOR)
            ? A->data[0] : A->data[0];
        const matx_complex_d_t a01 = (A->layout == MATX_COL_MAJOR)
            ? A->data[A->stride] : A->data[1];
        const matx_complex_d_t a10 = (A->layout == MATX_COL_MAJOR)
            ? A->data[1] : A->data[A->stride];
        const matx_complex_d_t a11 = (A->layout == MATX_COL_MAJOR)
            ? A->data[1 + A->stride] : A->data[A->stride + 1];
        det->real = a00.real * a11.real - a00.imag * a11.imag
                  - (a01.real * a10.real - a01.imag * a10.imag);
        det->imag = a00.real * a11.imag + a00.imag * a11.real
                  - (a01.real * a10.imag + a01.imag * a10.real);
        return MATX_OK;
    }
    matx_int64_t lda = ss_packed_lda(A->layout, n, n);

    matx_complex_d_t* Acopy = (matx_complex_d_t*) matx_malloc(alloc, (size_t) n * (size_t) n
                                                         * sizeof(matx_complex_d_t));
    matx_int64_t* piv = (matx_int64_t*) matx_malloc(alloc, (size_t) n * sizeof(matx_int64_t));
    if (!Acopy || !piv) {
        matx_free(alloc, Acopy);
        matx_free(alloc, piv);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    ss_pack_z_i8(A->layout, n, n, A->stride, A->data, Acopy);

    matx_int64_t info = LAPACKE_zgetrf(ss_layout_to_lapack(A->layout),
                                       n,
                                       n,
                                       (lapack_complex_double*) Acopy,
                                       lda,
                                       piv);
    if (info < 0) {
        MATX_ERROR("LAPACKE_zgetrf error: %d", (int) info);
        matx_free(alloc, Acopy);
        matx_free(alloc, piv);
        return MATX_ERR_INTERNAL;
    }
    if (info > 0) {
        det->real = 0.0;
        det->imag = 0.0;
        matx_free(alloc, Acopy);
        matx_free(alloc, piv);
        return MATX_OK;
    }

    matx_complex_d_t d = {1.0, 0.0};
    matx_double sign = 1.0;
    for (matx_int64_t i = 0; i < n; ++i) {
        if (piv[i] != i + 1)
            sign = -sign;
        matx_double re = d.real, im = d.imag;
        matx_double u_re = Acopy[i + i * lda].real, u_im = Acopy[i + i * lda].imag;
        d.real = re * u_re - im * u_im;
        d.imag = re * u_im + im * u_re;
    }
    d.real *= sign;
    d.imag *= sign;
    *det = d;
    matx_free(alloc, Acopy);
    matx_free(alloc, piv);
    return MATX_OK;
}

// ---- Condition number estimation (1-norm) ----

static matx_status_t ss_cond_dense_d_i8(const matx_alloc_t* alloc, const matx_dense_d_i8_t A, matx_double* cond)
{
    if (!A || !cond) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (A->nrows != A->ncols) {
        MATX_ERROR("%s: matrix must be square", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const matx_int64_t n = A->nrows;
    matx_int64_t lda = ss_packed_lda(A->layout, n, n);

    /* Compute 1-norm of A */
    matx_double anorm = LAPACKE_dlange(ss_layout_to_lapack(A->layout), '1', n, n, A->data, A->stride);

    matx_double* Acopy = (matx_double*) matx_malloc(alloc, (size_t) n * (size_t) n * sizeof(matx_double));
    matx_int64_t* piv = (matx_int64_t*) matx_malloc(alloc, (size_t) n * sizeof(matx_int64_t));
    if (!Acopy || !piv) {
        matx_free(alloc, Acopy);
        matx_free(alloc, piv);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    ss_pack_d_i8(A->layout, n, n, A->stride, A->data, Acopy);

    matx_int64_t info = LAPACKE_dgetrf(ss_layout_to_lapack(A->layout), n, n, Acopy, lda, piv);
    if (info != 0) {
        MATX_ERROR("LAPACKE_dgetrf error: %d", (int) info);
        matx_free(alloc, Acopy);
        matx_free(alloc, piv);
        return MATX_ERR_INTERNAL;
    }

    matx_double rcond = 0.0;
    info = LAPACKE_dgecon(ss_layout_to_lapack(A->layout), '1', n, Acopy, lda, anorm, &rcond);
    if (info != 0) {
        MATX_ERROR("LAPACKE_dgecon error: %d", (int) info);
        matx_free(alloc, Acopy);
        matx_free(alloc, piv);
        return MATX_ERR_INTERNAL;
    }

    *cond = (rcond > 0.0) ? (1.0 / rcond) : (matx_double)INFINITY;
    matx_free(alloc, Acopy);
    matx_free(alloc, piv);
    return MATX_OK;
}

// ---- Condition number estimation (complex, 1-norm) ----

static matx_status_t ss_cond_dense_z_i8(const matx_alloc_t* alloc, const matx_dense_z_i8_t A, matx_double* cond)
{
    if (!A || !cond) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (A->nrows != A->ncols) {
        MATX_ERROR("%s: matrix must be square", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const matx_int64_t n = A->nrows;
    matx_int64_t lda = ss_packed_lda(A->layout, n, n);

    matx_double anorm = LAPACKE_zlange(ss_layout_to_lapack(A->layout),
                                       '1',
                                       n,
                                       n,
                                       (const lapack_complex_double*) A->data,
                                       A->stride);

    matx_complex_d_t* Acopy = (matx_complex_d_t*) matx_malloc(alloc, (size_t) n * (size_t) n
                                                         * sizeof(matx_complex_d_t));
    matx_int64_t* piv = (matx_int64_t*) matx_malloc(alloc, (size_t) n * sizeof(matx_int64_t));
    if (!Acopy || !piv) {
        matx_free(alloc, Acopy);
        matx_free(alloc, piv);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    ss_pack_z_i8(A->layout, n, n, A->stride, A->data, Acopy);

    matx_int64_t info = LAPACKE_zgetrf(ss_layout_to_lapack(A->layout),
                                       n,
                                       n,
                                       (lapack_complex_double*) Acopy,
                                       lda,
                                       piv);
    if (info != 0) {
        MATX_ERROR("LAPACKE_zgetrf error: %d", (int) info);
        matx_free(alloc, Acopy);
        matx_free(alloc, piv);
        return MATX_ERR_INTERNAL;
    }

    matx_double rcond = 0.0;
    info = LAPACKE_zgecon(ss_layout_to_lapack(A->layout),
                          '1',
                          n,
                          (lapack_complex_double*) Acopy,
                          lda,
                          anorm,
                          &rcond);
    if (info != 0) {
        MATX_ERROR("LAPACKE_zgecon error: %d", (int) info);
        matx_free(alloc, Acopy);
        matx_free(alloc, piv);
        return MATX_ERR_INTERNAL;
    }

    *cond = (rcond > 0.0) ? (1.0 / rcond) : (matx_double)INFINITY;
    matx_free(alloc, Acopy);
    matx_free(alloc, piv);
    return MATX_OK;
}

// ---- Multi-RHS dense solve (DGETRS with nrhs > 1) ----

static matx_status_t ss_solve_dense_mrhs_d_i8(const matx_alloc_t* alloc,
                                              const matx_factor_dense_d_i8_t* F,
                                              const matx_dense_d_i8_t B,
                                              matx_dense_d_i8_t* X)
{
    if (!F || !B || !X) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (B->nrows != F->n) {
        MATX_ERROR("%s: dimension mismatch", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const matx_int64_t n = F->n, nrhs = B->ncols;
    const matx_int64_t ldb = ss_packed_lda(B->layout, n, nrhs);

    matx_double* Bcopy = (matx_double*) matx_malloc(alloc, (size_t) n * (size_t) nrhs * sizeof(matx_double));
    if (!Bcopy) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    ss_pack_d_i8(B->layout, n, nrhs, B->stride, B->data, Bcopy);

    matx_int64_t info = LAPACKE_dgetrs(ss_layout_to_lapack(B->layout),
                                       'N',
                                       n,
                                       nrhs,
                                       F->lu,
                                       F->lda,
                                       F->piv,
                                       Bcopy,
                                       ldb);
    if (info != 0) {
        MATX_ERROR("LAPACKE_dgetrs error: %d", (int) info);
        matx_free(alloc, Bcopy);
        return MATX_ERR_INTERNAL;
    }

    matx_dense_d_i8_opaque_t* xe = (matx_dense_d_i8_opaque_t*) matx_malloc(alloc,
        sizeof(matx_dense_d_i8_opaque_t));
    if (!xe) {
        matx_free(alloc, Bcopy);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(xe, 0, sizeof(*xe));
    xe->alloc = *alloc;
    xe->nrows = n;
    xe->ncols = nrhs;
    xe->layout = B->layout;
    xe->stride = ldb;
    xe->flags = 1u;
    xe->data = Bcopy;
    if (*X) {
        if ((*X)->flags & 1u)
            matx_free(alloc, (*X)->data);
        matx_free(alloc, *X);
    }
    *X = xe;
    return MATX_OK;
}

static matx_status_t ss_solve_dense_mrhs_z_i8(const matx_alloc_t* alloc,
                                              const matx_factor_dense_z_i8_t* F,
                                              const matx_dense_z_i8_t B,
                                              matx_dense_z_i8_t* X)
{
    if (!F || !B || !X) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (B->nrows != F->n) {
        MATX_ERROR("%s: dimension mismatch", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const matx_int64_t n = F->n, nrhs = B->ncols;
    const matx_int64_t ldb = ss_packed_lda(B->layout, n, nrhs);

    matx_complex_d_t* Bcopy = (matx_complex_d_t*) matx_malloc(alloc,
        (size_t) n * (size_t) nrhs * sizeof(matx_complex_d_t));
    if (!Bcopy) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    ss_pack_z_i8(B->layout, n, nrhs, B->stride, B->data, Bcopy);

    matx_int64_t info = LAPACKE_zgetrs(ss_layout_to_lapack(B->layout),
                                       'N',
                                       n,
                                       nrhs,
                                       (lapack_complex_double*) F->lu,
                                       F->lda,
                                       F->piv,
                                       (lapack_complex_double*) Bcopy,
                                       ldb);
    if (info != 0) {
        MATX_ERROR("LAPACKE_zgetrs error: %d", (int) info);
        matx_free(alloc, Bcopy);
        return MATX_ERR_INTERNAL;
    }

    matx_dense_z_i8_opaque_t* xe = (matx_dense_z_i8_opaque_t*) matx_malloc(alloc,
        sizeof(matx_dense_z_i8_opaque_t));
    if (!xe) {
        matx_free(alloc, Bcopy);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(xe, 0, sizeof(*xe));
    xe->alloc = *alloc;
    xe->nrows = n;
    xe->ncols = nrhs;
    xe->layout = B->layout;
    xe->stride = ldb;
    xe->flags = 1u;
    xe->data = (matx_complex_d_t*) Bcopy;
    if (*X) {
        if ((*X)->flags & 1u)
            matx_free(alloc, (*X)->data);
        matx_free(alloc, *X);
    }
    *X = xe;
    return MATX_OK;
}

// ---- Multi-RHS Cholesky solve (DPOTRS with nrhs > 1) ----

static matx_status_t ss_potrs_mrhs_d_i8(const matx_alloc_t* alloc,
                                        const matx_factor_dense_d_i8_t* F,
                                        const matx_dense_d_i8_t B,
                                        matx_dense_d_i8_t* X)
{
    if (!F || !B || !X) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (B->nrows != F->n) {
        MATX_ERROR("%s: dimension mismatch", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const matx_int64_t n = F->n, nrhs = B->ncols;
    const matx_int64_t ldb = ss_packed_lda(B->layout, n, nrhs);

    matx_double* Bcopy = (matx_double*) matx_malloc(alloc, (size_t) n * (size_t) nrhs * sizeof(matx_double));
    if (!Bcopy) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    ss_pack_d_i8(B->layout, n, nrhs, B->stride, B->data, Bcopy);

    matx_int64_t info = LAPACKE_dpotrs(ss_layout_to_lapack(B->layout),
                                       matx_uplo_to_lapack(F->uplo),
                                       n,
                                       nrhs,
                                       F->lu,
                                       F->lda,
                                       Bcopy,
                                       ldb);
    if (info != 0) {
        MATX_ERROR("LAPACKE_dpotrs error: %d", (int) info);
        matx_free(alloc, Bcopy);
        return MATX_ERR_INTERNAL;
    }

    matx_dense_d_i8_opaque_t* xe = (matx_dense_d_i8_opaque_t*) matx_malloc(alloc,
        sizeof(matx_dense_d_i8_opaque_t));
    if (!xe) {
        matx_free(alloc, Bcopy);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(xe, 0, sizeof(*xe));
    xe->alloc = *alloc;
    xe->nrows = n;
    xe->ncols = nrhs;
    xe->layout = B->layout;
    xe->stride = ldb;
    xe->flags = 1u;
    xe->data = Bcopy;
    if (*X) {
        if ((*X)->flags & 1u)
            matx_free(alloc, (*X)->data);
        matx_free(alloc, *X);
    }
    *X = xe;
    return MATX_OK;
}

static matx_status_t ss_potrs_mrhs_z_i8(const matx_alloc_t* alloc,
                                        const matx_factor_dense_z_i8_t* F,
                                        const matx_dense_z_i8_t B,
                                        matx_dense_z_i8_t* X)
{
    if (!F || !B || !X) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (B->nrows != F->n) {
        MATX_ERROR("%s: dimension mismatch", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const matx_int64_t n = F->n, nrhs = B->ncols;
    const matx_int64_t ldb = ss_packed_lda(B->layout, n, nrhs);

    matx_complex_d_t* Bcopy = (matx_complex_d_t*) matx_malloc(alloc,
        (size_t) n * (size_t) nrhs * sizeof(matx_complex_d_t));
    if (!Bcopy) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    ss_pack_z_i8(B->layout, n, nrhs, B->stride, B->data, Bcopy);

    matx_int64_t info = LAPACKE_zpotrs(ss_layout_to_lapack(B->layout),
                                       matx_uplo_to_lapack(F->uplo),
                                       n,
                                       nrhs,
                                       (lapack_complex_double*) F->lu,
                                       F->lda,
                                       (lapack_complex_double*) Bcopy,
                                       ldb);
    if (info != 0) {
        MATX_ERROR("LAPACKE_zpotrs error: %d", (int) info);
        matx_free(alloc, Bcopy);
        return MATX_ERR_INTERNAL;
    }

    matx_dense_z_i8_opaque_t* xe = (matx_dense_z_i8_opaque_t*) matx_malloc(alloc,
        sizeof(matx_dense_z_i8_opaque_t));
    if (!xe) {
        matx_free(alloc, Bcopy);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(xe, 0, sizeof(*xe));
    xe->alloc = *alloc;
    xe->nrows = n;
    xe->ncols = nrhs;
    xe->layout = B->layout;
    xe->stride = ldb;
    xe->flags = 1u;
    xe->data = (matx_complex_d_t*) Bcopy;
    if (*X) {
        if ((*X)->flags & 1u)
            matx_free(alloc, (*X)->data);
        matx_free(alloc, *X);
    }
    *X = xe;
    return MATX_OK;
}

// ---- LDL^T factorization (symmetric indefinite) ----

static matx_status_t ss_sytrf_d_i8(const matx_alloc_t* alloc,
                                   const matx_dense_d_i8_t A,
                                   matx_uplo_t uplo,
                                   matx_factor_dense_d_i8_t** out_F)
{
    if (!A || !out_F) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (A->nrows != A->ncols) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    if (*out_F)
        ss_factor_dense_d_i8_destroy(alloc, *out_F);

    const matx_int64_t n = A->nrows;
    matx_int64_t lda = ss_packed_lda(A->layout, n, n);

    matx_factor_dense_d_i8_t* F = (matx_factor_dense_d_i8_t*) matx_malloc(alloc, sizeof(*F));
    if (!F) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(F, 0, sizeof(*F));
    F->n = n;
    F->lda = lda;
    F->layout = A->layout;
    F->uplo = uplo;
    F->alloc = *alloc;

    F->lu = (matx_double*) matx_malloc(alloc, (size_t) n * (size_t) n * sizeof(matx_double));
    F->piv = (matx_int64_t*) matx_malloc(alloc, (size_t) n * sizeof(matx_int64_t));
    if (!F->lu || !F->piv) {
        MATX_ERROR("%s: out of memory", __func__);
        matx_free(alloc, F->lu);
        matx_free(alloc, F->piv);
        matx_free(alloc, F);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    ss_pack_d_i8(A->layout, n, n, A->stride, A->data, F->lu);

    matx_int64_t info = LAPACKE_dsytrf(ss_layout_to_lapack(A->layout),
                                       matx_uplo_to_lapack(uplo),
                                       n,
                                       F->lu,
                                       lda,
                                       F->piv);
    if (info != 0) {
        MATX_ERROR("LAPACKE_dsytrf error: %d", (int) info);
        matx_free(alloc, F->lu);
        matx_free(alloc, F->piv);
        matx_free(alloc, F);
        return MATX_ERR_INTERNAL;
    }
    *out_F = F;
    return MATX_OK;
}

static matx_status_t ss_sytrs_d_i8(const matx_alloc_t* alloc,
                                   const matx_factor_dense_d_i8_t* F,
                                   const matx_double* b,
                                   matx_double* x)
{
    if (!F || !b || !x) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (x != b)
        memmove(x, b, (size_t) F->n * sizeof(matx_double));

    matx_int64_t ldb = (F->layout == MATX_COL_MAJOR) ? F->n : 1;
    matx_int64_t info = LAPACKE_dsytrs(ss_layout_to_lapack(F->layout),
                                       matx_uplo_to_lapack(F->uplo),
                                       F->n,
                                       1,
                                       F->lu,
                                       F->lda,
                                       F->piv,
                                       x,
                                       ldb);
    if (info != 0) {
        MATX_ERROR("LAPACKE_dsytrs error: %d", (int) info);
        return MATX_ERR_INTERNAL;
    }
    return MATX_OK;
}

static void ss_sytrf_destroy_d(const matx_alloc_t* alloc, matx_factor_dense_d_i8_t* F)
{
    ss_factor_dense_d_i8_destroy(alloc, F);
}

static matx_status_t ss_sytrf_z_i8(const matx_alloc_t* alloc,
                                   const matx_dense_z_i8_t A,
                                   matx_uplo_t uplo,
                                   matx_factor_dense_z_i8_t** out_F)
{
    if (!A || !out_F) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (A->nrows != A->ncols) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    if (*out_F)
        ss_factor_dense_z_i8_destroy(alloc, *out_F);

    const matx_int64_t n = A->nrows;
    matx_int64_t lda = ss_packed_lda(A->layout, n, n);

    matx_factor_dense_z_i8_t* F = (matx_factor_dense_z_i8_t*) matx_malloc(alloc, sizeof(*F));
    if (!F) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(F, 0, sizeof(*F));
    F->n = n;
    F->lda = lda;
    F->layout = A->layout;
    F->uplo = uplo;
    F->alloc = *alloc;

    F->lu = (matx_double*) matx_malloc(alloc, (size_t) n * (size_t) n * sizeof(matx_complex_d_t));
    F->piv = (matx_int64_t*) matx_malloc(alloc, (size_t) n * sizeof(matx_int64_t));
    if (!F->lu || !F->piv) {
        MATX_ERROR("%s: out of memory", __func__);
        matx_free(alloc, F->lu);
        matx_free(alloc, F->piv);
        matx_free(alloc, F);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    ss_pack_z_i8(A->layout, n, n, A->stride, A->data, (matx_complex_d_t*) F->lu);

    matx_int64_t info = LAPACKE_zhetrf(ss_layout_to_lapack(A->layout),
                                       matx_uplo_to_lapack(uplo),
                                       n,
                                       (lapack_complex_double*) F->lu,
                                       lda,
                                       F->piv);
    if (info != 0) {
        MATX_ERROR("LAPACKE_zhetrf error: %d", (int) info);
        matx_free(alloc, F->lu);
        matx_free(alloc, F->piv);
        matx_free(alloc, F);
        return MATX_ERR_INTERNAL;
    }
    *out_F = F;
    return MATX_OK;
}

static matx_status_t ss_sytrs_z_i8(const matx_alloc_t* alloc,
                                   const matx_factor_dense_z_i8_t* F,
                                   const matx_vec_z_i8_t b,
                                   matx_vec_z_i8_t x)
{
    if (!F || !b || !x) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    matx_complex_d_t* rhs = (matx_complex_d_t*) matx_malloc(alloc, (size_t) F->n * sizeof(matx_complex_d_t));
    if (!rhs) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    for (matx_int64_t i = 0; i < F->n; ++i)
        rhs[i] = b->data[i * b->stride];

    matx_int64_t ldb = (F->layout == MATX_COL_MAJOR) ? F->n : 1;
    matx_int64_t info = LAPACKE_zhetrs(ss_layout_to_lapack(F->layout),
                                       matx_uplo_to_lapack(F->uplo),
                                       F->n,
                                       1,
                                       (lapack_complex_double*) F->lu,
                                       F->lda,
                                       F->piv,
                                       (lapack_complex_double*) rhs,
                                       ldb);
    if (info != 0) {
        MATX_ERROR("LAPACKE_zhetrs error: %d", (int) info);
        matx_free(alloc, rhs);
        return MATX_ERR_INTERNAL;
    }
    for (matx_int64_t i = 0; i < F->n; ++i)
        x->data[i * x->stride] = rhs[i];
    matx_free(alloc, rhs);
    return MATX_OK;
}

static void ss_sytrf_destroy_z(const matx_alloc_t* alloc, matx_factor_dense_z_i8_t* F)
{
    ss_factor_dense_z_i8_destroy(alloc, F);
}

// ---- QR with column pivoting ----

static matx_status_t ss_qrp_d_i8(const matx_alloc_t* alloc,
                                 const matx_dense_d_i8_t A,
                                 matx_dense_d_i8_t* Q,
                                 matx_dense_d_i8_t* R,
                                 matx_vec_d_i8_t* jpvt)
{
    if (!A || !Q || !R || !jpvt) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const matx_int64_t m = A->nrows, n = A->ncols;
    const matx_int64_t k = (m < n) ? m : n;

    matx_int64_t lda = ss_packed_lda(A->layout, m, n);

    matx_double* Acopy = (matx_double*) matx_malloc(alloc, (size_t) m * (size_t) n * sizeof(matx_double));
    matx_double* tau = (matx_double*) matx_malloc(alloc, (size_t) k * sizeof(matx_double));
    matx_int64_t* jpvt_buf = (matx_int64_t*) matx_malloc(alloc, (size_t) n * sizeof(matx_int64_t));
    if (!Acopy || !tau || !jpvt_buf) {
        matx_free(alloc, Acopy);
        matx_free(alloc, tau);
        matx_free(alloc, jpvt_buf);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    for (matx_int64_t i = 0; i < n; ++i)
        jpvt_buf[i] = 0;

    ss_pack_d_i8(A->layout, m, n, A->stride, A->data, Acopy);

    matx_int64_t info = LAPACKE_dgeqp3(ss_layout_to_lapack(A->layout), m, n, Acopy, lda, jpvt_buf, tau);
    if (info != 0) {
        MATX_ERROR("LAPACKE_dgeqp3 error: %d", (int) info);
        matx_free(alloc, Acopy);
        matx_free(alloc, tau);
        matx_free(alloc, jpvt_buf);
        return MATX_ERR_INTERNAL;
    }

    matx_double* r_data = (matx_double*) matx_malloc(alloc, (size_t) k * (size_t) n * sizeof(matx_double));
    if (!r_data) {
        matx_free(alloc, Acopy);
        matx_free(alloc, tau);
        matx_free(alloc, jpvt_buf);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(r_data, 0, (size_t) k * (size_t) n * sizeof(matx_double));
    for (matx_int64_t j = 0; j < n; ++j)
        for (matx_int64_t i = 0; i <= j && i < m; ++i) {
            matx_int64_t idx = (A->layout == MATX_COL_MAJOR) ? i + j * lda : j + i * lda;
            matx_int64_t ridx = (A->layout == MATX_COL_MAJOR) ? i + j * k : j + i * k;
            r_data[ridx] = Acopy[idx];
        }

    info = LAPACKE_dorgqr(ss_layout_to_lapack(A->layout), m, k, k, Acopy, lda, tau);
    if (info != 0) {
        MATX_ERROR("LAPACKE_dorgqr error: %d", (int) info);
        matx_free(alloc, Acopy);
        matx_free(alloc, tau);
        matx_free(alloc, jpvt_buf);
        matx_free(alloc, r_data);
        return MATX_ERR_INTERNAL;
    }
    matx_free(alloc, tau);

    /* Q = Acopy (m x k) */
    {
        matx_dense_d_i8_opaque_t* qe = (matx_dense_d_i8_opaque_t*) matx_malloc(alloc,
            sizeof(matx_dense_d_i8_opaque_t));
        if (!qe) {
            matx_free(alloc, Acopy);
            matx_free(alloc, r_data);
            matx_free(alloc, jpvt_buf);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        memset(qe, 0, sizeof(*qe));
        qe->alloc = *alloc;
        qe->nrows = m;
        qe->ncols = k;
        qe->layout = A->layout;
        qe->stride = lda;
        qe->flags = 1u;
        qe->data = Acopy;
        if (*Q) {
            if ((*Q)->flags & 1u)
                matx_free(alloc, (*Q)->data);
            matx_free(alloc, *Q);
        }
        *Q = qe;
    }

    /* R = r_data (k x n) */
    {
        matx_dense_d_i8_opaque_t* re = (matx_dense_d_i8_opaque_t*) matx_malloc(alloc,
            sizeof(matx_dense_d_i8_opaque_t));
        if (!re) {
            matx_free(alloc, r_data);
            matx_free(alloc, jpvt_buf);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        memset(re, 0, sizeof(*re));
        re->alloc = *alloc;
        re->nrows = k;
        re->ncols = n;
        re->layout = A->layout;
        re->stride = (A->layout == MATX_COL_MAJOR) ? k : n;
        re->flags = 1u;
        re->data = r_data;
        if (*R) {
            if ((*R)->flags & 1u)
                matx_free(alloc, (*R)->data);
            matx_free(alloc, *R);
        }
        *R = re;
    }

    /* jpvt output */
    {
        matx_vec_d_i8_opaque_t* jv = (matx_vec_d_i8_opaque_t*) matx_malloc(alloc,
            sizeof(matx_vec_d_i8_opaque_t));
        if (!jv) {
            matx_free(alloc, jpvt_buf);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        memset(jv, 0, sizeof(*jv));
        jv->alloc = *alloc;
        jv->n = n;
        jv->stride = 1;
        jv->flags = 1u;
        jv->data = (matx_double*) jpvt_buf;
        /* Convert jpvt indices from Fortran 1-based to C 0-based */
        for (matx_int64_t i = 0; i < n; ++i)
            jv->data[i] = (matx_double) (jpvt_buf[i] - 1);
        if (*jpvt) {
            if ((*jpvt)->flags & 1u)
                matx_free(alloc, (*jpvt)->data);
            matx_free(alloc, *jpvt);
        }
        *jpvt = jv;
    }
    return MATX_OK;
}

static matx_status_t ss_qrp_z_i8(const matx_alloc_t* alloc,
                                 const matx_dense_z_i8_t A,
                                 matx_dense_z_i8_t* Q,
                                 matx_dense_z_i8_t* R,
                                 matx_vec_d_i8_t* jpvt)
{
    if (!A || !Q || !R || !jpvt) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const matx_int64_t m = A->nrows, n = A->ncols;
    const matx_int64_t k = (m < n) ? m : n;

    matx_int64_t lda = ss_packed_lda(A->layout, m, n);

    matx_complex_d_t* Acopy = (matx_complex_d_t*) matx_malloc(alloc,
        (size_t) m * (size_t) n * sizeof(matx_complex_d_t));
    matx_complex_d_t* tau_c = (matx_complex_d_t*) matx_malloc(alloc,
        (size_t) k * sizeof(matx_complex_d_t));
    matx_int64_t* jpvt_buf = (matx_int64_t*) matx_malloc(alloc, (size_t) n * sizeof(matx_int64_t));
    if (!Acopy || !tau_c || !jpvt_buf) {
        matx_free(alloc, Acopy);
        matx_free(alloc, tau_c);
        matx_free(alloc, jpvt_buf);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    for (matx_int64_t i = 0; i < n; ++i)
        jpvt_buf[i] = 0;

    ss_pack_z_i8(A->layout, m, n, A->stride, A->data, Acopy);

    matx_int64_t info = LAPACKE_zgeqp3(ss_layout_to_lapack(A->layout), m, n,
                                       (lapack_complex_double*) Acopy, lda, jpvt_buf,
                                       (lapack_complex_double*) tau_c);
    if (info != 0) {
        MATX_ERROR("LAPACKE_zgeqp3 error: %d", (int) info);
        matx_free(alloc, Acopy);
        matx_free(alloc, tau_c);
        matx_free(alloc, jpvt_buf);
        return MATX_ERR_INTERNAL;
    }

    matx_complex_d_t* r_data = (matx_complex_d_t*) matx_malloc(alloc,
        (size_t) k * (size_t) n * sizeof(matx_complex_d_t));
    if (!r_data) {
        matx_free(alloc, Acopy);
        matx_free(alloc, tau_c);
        matx_free(alloc, jpvt_buf);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(r_data, 0, (size_t) k * (size_t) n * sizeof(matx_complex_d_t));
    for (matx_int64_t j = 0; j < n; ++j)
        for (matx_int64_t i = 0; i <= j && i < m; ++i) {
            matx_int64_t idx = (A->layout == MATX_COL_MAJOR) ? i + j * lda : j + i * lda;
            matx_int64_t ridx = (A->layout == MATX_COL_MAJOR) ? i + j * k : j + i * k;
            r_data[ridx] = Acopy[idx];
        }

    info = LAPACKE_zungqr(ss_layout_to_lapack(A->layout), m, k, k,
                          (lapack_complex_double*) Acopy, lda,
                          (lapack_complex_double*) tau_c);
    if (info != 0) {
        MATX_ERROR("LAPACKE_zungqr error: %d", (int) info);
        matx_free(alloc, Acopy);
        matx_free(alloc, tau_c);
        matx_free(alloc, jpvt_buf);
        matx_free(alloc, r_data);
        return MATX_ERR_INTERNAL;
    }
    matx_free(alloc, tau_c);

    /* Q = Acopy (m x k) */
    {
        matx_dense_z_i8_opaque_t* qe = (matx_dense_z_i8_opaque_t*) matx_malloc(alloc,
            sizeof(matx_dense_z_i8_opaque_t));
        if (!qe) {
            matx_free(alloc, Acopy);
            matx_free(alloc, r_data);
            matx_free(alloc, jpvt_buf);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        memset(qe, 0, sizeof(*qe));
        qe->alloc = *alloc;
        qe->nrows = m;
        qe->ncols = k;
        qe->layout = A->layout;
        qe->stride = lda;
        qe->flags = 1u;
        qe->data = (matx_complex_d_t*) Acopy;
        if (*Q) {
            if ((*Q)->flags & 1u)
                matx_free(alloc, (*Q)->data);
            matx_free(alloc, *Q);
        }
        *Q = qe;
    }

    /* R = r_data (k x n) */
    {
        matx_dense_z_i8_opaque_t* re = (matx_dense_z_i8_opaque_t*) matx_malloc(alloc,
            sizeof(matx_dense_z_i8_opaque_t));
        if (!re) {
            matx_free(alloc, r_data);
            matx_free(alloc, jpvt_buf);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        memset(re, 0, sizeof(*re));
        re->alloc = *alloc;
        re->nrows = k;
        re->ncols = n;
        re->layout = A->layout;
        re->stride = (A->layout == MATX_COL_MAJOR) ? k : n;
        re->flags = 1u;
        re->data = (matx_complex_d_t*) r_data;
        if (*R) {
            if ((*R)->flags & 1u)
                matx_free(alloc, (*R)->data);
            matx_free(alloc, *R);
        }
        *R = re;
    }

    /* jpvt output */
    {
        matx_vec_d_i8_opaque_t* jv = (matx_vec_d_i8_opaque_t*) matx_malloc(alloc,
            sizeof(matx_vec_d_i8_opaque_t));
        if (!jv) {
            matx_free(alloc, jpvt_buf);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        memset(jv, 0, sizeof(*jv));
        jv->alloc = *alloc;
        jv->n = n;
        jv->stride = 1;
        jv->flags = 1u;
        jv->data = (matx_double*) jpvt_buf;
        for (matx_int64_t i = 0; i < n; ++i)
            jv->data[i] = (matx_double) (jpvt_buf[i] - 1);
        if (*jpvt) {
            if ((*jpvt)->flags & 1u)
                matx_free(alloc, (*jpvt)->data);
            matx_free(alloc, *jpvt);
        }
        *jpvt = jv;
    }
    return MATX_OK;
}

// ---- Pseudo-inverse (via SVD) ----

static matx_status_t ss_pinv_d_i8(const matx_alloc_t* alloc,
                                  const matx_dense_d_i8_t A,
                                  matx_double rcond,
                                  matx_dense_d_i8_t* out)
{
    if (!A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const matx_int64_t m = A->nrows, n = A->ncols;
    const matx_int64_t k = (m < n) ? m : n;

    matx_int64_t lda = ss_packed_lda(A->layout, m, n);

    matx_double* Acopy = (matx_double*) matx_malloc(alloc, (size_t) m * (size_t) n * sizeof(matx_double));
    matx_double* u_data = (matx_double*) matx_malloc(alloc, (size_t) m * (size_t) m * sizeof(matx_double));
    matx_double* vt_data = (matx_double*) matx_malloc(alloc, (size_t) n * (size_t) n * sizeof(matx_double));
    matx_double* S = (matx_double*) matx_malloc(alloc, (size_t) k * sizeof(matx_double));
    matx_double* superb = (matx_double*) matx_malloc(alloc, k * sizeof(matx_double));
    if (!Acopy || !u_data || !vt_data || !S || !superb) {
        matx_free(alloc, Acopy);
        matx_free(alloc, u_data);
        matx_free(alloc, vt_data);
        matx_free(alloc, S);
        matx_free(alloc, superb);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    ss_pack_d_i8(A->layout, m, n, A->stride, A->data, Acopy);

    matx_int64_t info = LAPACKE_dgesvd(ss_layout_to_lapack(A->layout),
                                       'A', 'A', m, n, Acopy, lda,
                                       S, u_data, m, vt_data, n, superb);
    matx_free(alloc, Acopy);
    matx_free(alloc, superb);
    if (info != 0) {
        MATX_ERROR("LAPACKE_dgesvd error: %d", (int) info);
        matx_free(alloc, u_data);
        matx_free(alloc, vt_data);
        matx_free(alloc, S);
        return MATX_ERR_INTERNAL;
    }

    /* Compute threshold */
    matx_double smax = 0.0;
    for (matx_int64_t i = 0; i < k; ++i)
        if (S[i] > smax) smax = S[i];
    const matx_double tol = rcond * smax;

    /* pinv = V * diag(1/S) * U^T, computed as:
     *   temp(i,j) = V(i,j) * (1/S(j)) if S(j) > tol, else 0   for i=0..n-1, j=0..k-1
     *   pinv(i,j) = sum_l temp(i,l) * U(j,l)   for i=0..n-1, j=0..m-1
     * V is stored in vt_data as V^T, so V(i,j) = vt_data(j,i) in col-major.
     */
    matx_double* pinv_data = (matx_double*) matx_malloc(alloc, (size_t) n * (size_t) m * sizeof(matx_double));
    if (!pinv_data) {
        matx_free(alloc, u_data);
        matx_free(alloc, vt_data);
        matx_free(alloc, S);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(pinv_data, 0, (size_t) n * (size_t) m * sizeof(matx_double));

    if (smax > 0.0) {
        /* Vt is n x n col-major, U is m x m col-major */
        /* temp[n][k] = V * diag(S_inv): temp[i][j] = Vt[j][i] / S[j] if S[j] > tol */
        /* pinv[n][m] = temp * U^T: pinv[i][j] = sum_l temp[i][l] * U[j][l] */
        matx_double* temp = (matx_double*) matx_malloc(alloc, (size_t) n * (size_t) k * sizeof(matx_double));
        if (!temp) {
            matx_free(alloc, u_data);
            matx_free(alloc, vt_data);
            matx_free(alloc, S);
            matx_free(alloc, pinv_data);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        for (matx_int64_t i = 0; i < n; ++i) {
            for (matx_int64_t j = 0; j < k; ++j) {
                const matx_double sinv = (S[j] > tol) ? (1.0 / S[j]) : 0.0;
                temp[i + j * n] = vt_data[j + i * n] * sinv; /* Vt[j][i] = vt_data[j + i*n] in col-major */
            }
        }
        /* pinv[n][m] = temp * U^T; pinv[i][j] = sum_l temp[i][l] * U[j][l] */
        for (matx_int64_t i = 0; i < n; ++i) {
            for (matx_int64_t j = 0; j < m; ++j) {
                matx_double s = 0.0;
                for (matx_int64_t l = 0; l < k; ++l) {
                    s += temp[i + l * n] * u_data[j + l * m]; /* U[j][l] in col-major */
                }
                pinv_data[i + j * n] = s;
            }
        }
        matx_free(alloc, temp);
    }

    matx_free(alloc, u_data);
    matx_free(alloc, vt_data);
    matx_free(alloc, S);

    matx_dense_d_i8_opaque_t* oe = (matx_dense_d_i8_opaque_t*) matx_malloc(alloc,
        sizeof(matx_dense_d_i8_opaque_t));
    if (!oe) {
        matx_free(alloc, pinv_data);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(oe, 0, sizeof(*oe));
    oe->alloc = *alloc;
    oe->nrows = n;
    oe->ncols = m;
    oe->layout = MATX_COL_MAJOR;
    oe->stride = n;
    oe->flags = 1u;
    oe->data = pinv_data;
    if (*out) {
        if ((*out)->flags & 1u)
            matx_free(alloc, (*out)->data);
        matx_free(alloc, *out);
    }
    *out = oe;
    return MATX_OK;
}

static matx_status_t ss_pinv_z_i8(const matx_alloc_t* alloc,
                                  const matx_dense_z_i8_t A,
                                  matx_double rcond,
                                  matx_dense_z_i8_t* out)
{
    if (!A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const matx_int64_t m = A->nrows, n = A->ncols;
    const matx_int64_t k = (m < n) ? m : n;

    matx_int64_t lda = ss_packed_lda(A->layout, m, n);

    matx_complex_d_t* Acopy = (matx_complex_d_t*) matx_malloc(alloc,
        (size_t) m * (size_t) n * sizeof(matx_complex_d_t));
    matx_complex_d_t* u_data = (matx_complex_d_t*) matx_malloc(alloc,
        (size_t) m * (size_t) m * sizeof(matx_complex_d_t));
    matx_complex_d_t* vt_data = (matx_complex_d_t*) matx_malloc(alloc,
        (size_t) n * (size_t) n * sizeof(matx_complex_d_t));
    matx_double* S = (matx_double*) matx_malloc(alloc, (size_t) k * sizeof(matx_double));
    matx_double* superb = (matx_double*) matx_malloc(alloc, k * sizeof(matx_double));
    if (!Acopy || !u_data || !vt_data || !S || !superb) {
        matx_free(alloc, Acopy);
        matx_free(alloc, u_data);
        matx_free(alloc, vt_data);
        matx_free(alloc, S);
        matx_free(alloc, superb);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    ss_pack_z_i8(A->layout, m, n, A->stride, A->data, Acopy);

    matx_int64_t info = LAPACKE_zgesvd(ss_layout_to_lapack(A->layout),
                                       'A', 'A', m, n,
                                       (lapack_complex_double*) Acopy, lda,
                                       S,
                                       (lapack_complex_double*) u_data, m,
                                       (lapack_complex_double*) vt_data, n,
                                       superb);
    matx_free(alloc, Acopy);
    matx_free(alloc, superb);
    if (info != 0) {
        MATX_ERROR("LAPACKE_zgesvd error: %d", (int) info);
        matx_free(alloc, u_data);
        matx_free(alloc, vt_data);
        matx_free(alloc, S);
        return MATX_ERR_INTERNAL;
    }

    matx_double smax = 0.0;
    for (matx_int64_t i = 0; i < k; ++i)
        if (S[i] > smax) smax = S[i];
    const matx_double tol = rcond * smax;

    matx_complex_d_t* pinv_data = (matx_complex_d_t*) matx_malloc(alloc,
        (size_t) n * (size_t) m * sizeof(matx_complex_d_t));
    if (!pinv_data) {
        matx_free(alloc, u_data);
        matx_free(alloc, vt_data);
        matx_free(alloc, S);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(pinv_data, 0, (size_t) n * (size_t) m * sizeof(matx_complex_d_t));

    if (smax > 0.0) {
        matx_complex_d_t* temp = (matx_complex_d_t*) matx_malloc(alloc,
            (size_t) n * (size_t) k * sizeof(matx_complex_d_t));
        if (!temp) {
            matx_free(alloc, u_data);
            matx_free(alloc, vt_data);
            matx_free(alloc, S);
            matx_free(alloc, pinv_data);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        for (matx_int64_t i = 0; i < n; ++i) {
            for (matx_int64_t j = 0; j < k; ++j) {
                const matx_double sinv = (S[j] > tol) ? (1.0 / S[j]) : 0.0;
                /* Vt[j][i] = conj(V[i][j]) — for real SVD of complex, V^H, not V^T */
                matx_complex_d_t vh = vt_data[j + i * n]; /* V^H stored as V^T for ZGESVD */
                temp[i + j * n].real = vh.real * sinv;
                temp[i + j * n].imag = -vh.imag * sinv; /* conj for V^H */
            }
        }
        /* pinv[n][m] = temp * U^H; pinv[i][j] = sum_l temp[i][l] * conj(U[j][l]) */
        for (matx_int64_t i = 0; i < n; ++i) {
            for (matx_int64_t j = 0; j < m; ++j) {
                matx_double sr = 0.0, si = 0.0;
                for (matx_int64_t l = 0; l < k; ++l) {
                    matx_complex_d_t t = temp[i + l * n];
                    matx_complex_d_t u_jl = u_data[j + l * m];
                    /* t * conj(u_jl) */
                    sr += t.real * u_jl.real + t.imag * u_jl.imag;
                    si += t.imag * u_jl.real - t.real * u_jl.imag;
                }
                pinv_data[i + j * n].real = sr;
                pinv_data[i + j * n].imag = si;
            }
        }
        matx_free(alloc, temp);
    }

    matx_free(alloc, u_data);
    matx_free(alloc, vt_data);
    matx_free(alloc, S);

    matx_dense_z_i8_opaque_t* oe = (matx_dense_z_i8_opaque_t*) matx_malloc(alloc,
        sizeof(matx_dense_z_i8_opaque_t));
    if (!oe) {
        matx_free(alloc, pinv_data);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(oe, 0, sizeof(*oe));
    oe->alloc = *alloc;
    oe->nrows = n;
    oe->ncols = m;
    oe->layout = MATX_COL_MAJOR;
    oe->stride = n;
    oe->flags = 1u;
    oe->data = (matx_complex_d_t*) pinv_data;
    if (*out) {
        if ((*out)->flags & 1u)
            matx_free(alloc, (*out)->data);
        matx_free(alloc, *out);
    }
    *out = oe;
    return MATX_OK;
}

// ---- Matrix rank (via SVD) ----

static matx_status_t ss_rank_d_i8(const matx_alloc_t* alloc,
                                  const matx_dense_d_i8_t A,
                                  matx_double tol,
                                  matx_int64_t* rank)
{
    if (!A || !rank) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const matx_int64_t m = A->nrows, n = A->ncols;
    const matx_int64_t k = (m < n) ? m : n;

    matx_int64_t lda = ss_packed_lda(A->layout, m, n);

    matx_double* Acopy = (matx_double*) matx_malloc(alloc, (size_t) m * (size_t) n * sizeof(matx_double));
    matx_double* S = (matx_double*) matx_malloc(alloc, (size_t) k * sizeof(matx_double));
    matx_double* superb = (matx_double*) matx_malloc(alloc, k * sizeof(matx_double));
    if (!Acopy || !S || !superb) {
        matx_free(alloc, Acopy);
        matx_free(alloc, S);
        matx_free(alloc, superb);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    ss_pack_d_i8(A->layout, m, n, A->stride, A->data, Acopy);

    matx_int64_t info = LAPACKE_dgesvd(ss_layout_to_lapack(A->layout),
                                       'N', 'N', m, n, Acopy, lda,
                                       S, NULL, m, NULL, n, superb);
    matx_free(alloc, Acopy);
    matx_free(alloc, superb);
    if (info != 0) {
        MATX_ERROR("LAPACKE_dgesvd error: %d", (int) info);
        matx_free(alloc, S);
        return MATX_ERR_INTERNAL;
    }

    matx_double smax = 0.0;
    for (matx_int64_t i = 0; i < k; ++i)
        if (S[i] > smax) smax = S[i];

    const matx_double threshold = (tol > 0.0) ? tol * smax : smax * (matx_double) k * DBL_EPSILON;
    matx_int64_t r = 0;
    for (matx_int64_t i = 0; i < k; ++i)
        if (S[i] > threshold) ++r;

    *rank = r;
    matx_free(alloc, S);
    return MATX_OK;
}

static matx_status_t ss_rank_z_i8(const matx_alloc_t* alloc,
                                  const matx_dense_z_i8_t A,
                                  matx_double tol,
                                  matx_int64_t* rank)
{
    if (!A || !rank) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const matx_int64_t m = A->nrows, n = A->ncols;
    const matx_int64_t k = (m < n) ? m : n;

    matx_int64_t lda = ss_packed_lda(A->layout, m, n);

    matx_complex_d_t* Acopy = (matx_complex_d_t*) matx_malloc(alloc,
        (size_t) m * (size_t) n * sizeof(matx_complex_d_t));
    matx_double* S = (matx_double*) matx_malloc(alloc, (size_t) k * sizeof(matx_double));
    matx_double* superb = (matx_double*) matx_malloc(alloc, k * sizeof(matx_double));
    if (!Acopy || !S || !superb) {
        matx_free(alloc, Acopy);
        matx_free(alloc, S);
        matx_free(alloc, superb);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    ss_pack_z_i8(A->layout, m, n, A->stride, A->data, Acopy);

    matx_int64_t info = LAPACKE_zgesvd(ss_layout_to_lapack(A->layout),
                                       'N', 'N', m, n,
                                       (lapack_complex_double*) Acopy, lda,
                                       S, NULL, m, NULL, n, superb);
    matx_free(alloc, Acopy);
    matx_free(alloc, superb);
    if (info != 0) {
        MATX_ERROR("LAPACKE_zgesvd error: %d", (int) info);
        matx_free(alloc, S);
        return MATX_ERR_INTERNAL;
    }

    matx_double smax = 0.0;
    for (matx_int64_t i = 0; i < k; ++i)
        if (S[i] > smax) smax = S[i];

    const matx_double threshold = (tol > 0.0) ? tol * smax : smax * (matx_double) k * DBL_EPSILON;
    matx_int64_t r = 0;
    for (matx_int64_t i = 0; i < k; ++i)
        if (S[i] > threshold) ++r;

    *rank = r;
    matx_free(alloc, S);
    return MATX_OK;
}

// ---- Generalized symmetric eigenvalue (SYGV/HEGV) ----

static matx_status_t ss_sygv_d_i8(const matx_alloc_t* alloc,
                                  const matx_dense_d_i8_t A,
                                  const matx_dense_d_i8_t B,
                                  matx_vec_d_i8_t eigenvalues,
                                  matx_dense_d_i8_t* eigenvectors)
{
    if (!A || !B || !eigenvalues) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (A->nrows != A->ncols || B->nrows != B->ncols || A->nrows != B->nrows) {
        MATX_ERROR("%s: dimension mismatch", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const matx_int64_t n = A->nrows;
    if (eigenvalues->n != n) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    const char jobz = eigenvectors ? 'V' : 'N';
    matx_int64_t lda = ss_packed_lda(A->layout, n, n);
    matx_int64_t ldb = ss_packed_lda(B->layout, n, n);

    matx_double* Acopy = (matx_double*) matx_malloc(alloc, (size_t) n * (size_t) n * sizeof(matx_double));
    matx_double* Bcopy = (matx_double*) matx_malloc(alloc, (size_t) n * (size_t) n * sizeof(matx_double));
    if (!Acopy || !Bcopy) {
        matx_free(alloc, Acopy);
        matx_free(alloc, Bcopy);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    ss_pack_d_i8(A->layout, n, n, A->stride, A->data, Acopy);
    ss_pack_d_i8(B->layout, n, n, B->stride, B->data, Bcopy);

    /* itype=1: A*v = lambda*B*v */
    matx_int64_t info = LAPACKE_dsygv(ss_layout_to_lapack(A->layout),
                                      1, jobz, 'L', n,
                                      Acopy, lda,
                                      Bcopy, ldb,
                                      eigenvalues->data);
    matx_free(alloc, Bcopy);
    if (info != 0) {
        MATX_ERROR("LAPACKE_dsygv error: %d", (int) info);
        matx_free(alloc, Acopy);
        return MATX_ERR_INTERNAL;
    }

    if (eigenvectors) {
        matx_dense_d_i8_opaque_t* ev = (matx_dense_d_i8_opaque_t*) matx_malloc(alloc,
            sizeof(matx_dense_d_i8_opaque_t));
        if (!ev) {
            matx_free(alloc, Acopy);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        memset(ev, 0, sizeof(*ev));
        ev->alloc = *alloc;
        ev->nrows = n;
        ev->ncols = n;
        ev->layout = A->layout;
        ev->stride = lda;
        ev->flags = 1u;
        ev->data = Acopy;
        if (*eigenvectors) {
            if ((*eigenvectors)->flags & 1u)
                matx_free(alloc, (*eigenvectors)->data);
            matx_free(alloc, *eigenvectors);
        }
        *eigenvectors = ev;
    } else {
        matx_free(alloc, Acopy);
    }
    return MATX_OK;
}

static matx_status_t ss_sygv_z_i8(const matx_alloc_t* alloc,
                                  const matx_dense_z_i8_t A,
                                  const matx_dense_z_i8_t B,
                                  matx_vec_d_i8_t eigenvalues,
                                  matx_dense_z_i8_t* eigenvectors)
{
    if (!A || !B || !eigenvalues) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (A->nrows != A->ncols || B->nrows != B->ncols || A->nrows != B->nrows) {
        MATX_ERROR("%s: dimension mismatch", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const matx_int64_t n = A->nrows;
    if (eigenvalues->n != n) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }

    const char jobz = eigenvectors ? 'V' : 'N';
    matx_int64_t lda = ss_packed_lda(A->layout, n, n);
    matx_int64_t ldb = ss_packed_lda(B->layout, n, n);

    matx_complex_d_t* Acopy = (matx_complex_d_t*) matx_malloc(alloc,
        (size_t) n * (size_t) n * sizeof(matx_complex_d_t));
    matx_complex_d_t* Bcopy = (matx_complex_d_t*) matx_malloc(alloc,
        (size_t) n * (size_t) n * sizeof(matx_complex_d_t));
    if (!Acopy || !Bcopy) {
        matx_free(alloc, Acopy);
        matx_free(alloc, Bcopy);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    ss_pack_z_i8(A->layout, n, n, A->stride, A->data, Acopy);
    ss_pack_z_i8(B->layout, n, n, B->stride, B->data, Bcopy);

    /* itype=1: A*v = lambda*B*v */
    matx_int64_t info = LAPACKE_zhegv(ss_layout_to_lapack(A->layout),
                                      1, jobz, 'L', n,
                                      (lapack_complex_double*) Acopy, lda,
                                      (lapack_complex_double*) Bcopy, ldb,
                                      eigenvalues->data);
    matx_free(alloc, Bcopy);
    if (info != 0) {
        MATX_ERROR("LAPACKE_zhegv error: %d", (int) info);
        matx_free(alloc, Acopy);
        return MATX_ERR_INTERNAL;
    }

    if (eigenvectors) {
        matx_dense_z_i8_opaque_t* ev = (matx_dense_z_i8_opaque_t*) matx_malloc(alloc,
            sizeof(matx_dense_z_i8_opaque_t));
        if (!ev) {
            matx_free(alloc, Acopy);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        memset(ev, 0, sizeof(*ev));
        ev->alloc = *alloc;
        ev->nrows = n;
        ev->ncols = n;
        ev->layout = A->layout;
        ev->stride = lda;
        ev->flags = 1u;
        ev->data = (matx_complex_d_t*) Acopy;
        if (*eigenvectors) {
            if ((*eigenvectors)->flags & 1u)
                matx_free(alloc, (*eigenvectors)->data);
            matx_free(alloc, *eigenvectors);
        }
        *eigenvectors = ev;
    } else {
        matx_free(alloc, Acopy);
    }
    return MATX_OK;
}

// ---- LQ factorization ----

static matx_status_t ss_lq_d_i8(const matx_alloc_t* alloc,
                                const matx_dense_d_i8_t A,
                                matx_dense_d_i8_t* L,
                                matx_dense_d_i8_t* Q)
{
    if (!A || !L || !Q) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const matx_int64_t m = A->nrows, n = A->ncols;
    const matx_int64_t k = (m < n) ? m : n;

    matx_int64_t lda = ss_packed_lda(A->layout, m, n);

    matx_double* Acopy = (matx_double*) matx_malloc(alloc, (size_t) m * (size_t) n * sizeof(matx_double));
    matx_double* tau = (matx_double*) matx_malloc(alloc, (size_t) k * sizeof(matx_double));
    if (!Acopy || !tau) {
        matx_free(alloc, Acopy);
        matx_free(alloc, tau);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    ss_pack_d_i8(A->layout, m, n, A->stride, A->data, Acopy);

    matx_int64_t info = LAPACKE_dgelqf(ss_layout_to_lapack(A->layout), m, n, Acopy, lda, tau);
    if (info != 0) {
        MATX_ERROR("LAPACKE_dgelqf error: %d", (int) info);
        matx_free(alloc, Acopy);
        matx_free(alloc, tau);
        return MATX_ERR_INTERNAL;
    }

    /* Extract L: lower triangular part (m x n, only lower filled to min(m,n)) */
    matx_double* l_data = (matx_double*) matx_malloc(alloc, (size_t) m * (size_t) n * sizeof(matx_double));
    if (!l_data) {
        matx_free(alloc, Acopy);
        matx_free(alloc, tau);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(l_data, 0, (size_t) m * (size_t) n * sizeof(matx_double));
    for (matx_int64_t j = 0; j < n; ++j)
        for (matx_int64_t i = j; i < m; ++i) {
            matx_int64_t idx = (A->layout == MATX_COL_MAJOR) ? i + j * lda : j + i * lda;
            matx_int64_t lidx = (A->layout == MATX_COL_MAJOR) ? i + j * m : j + i * m;
            l_data[lidx] = Acopy[idx];
        }

    /* Generate Q */
    info = LAPACKE_dorglq(ss_layout_to_lapack(A->layout), k, n, k, Acopy, lda, tau);
    if (info != 0) {
        MATX_ERROR("LAPACKE_dorglq error: %d", (int) info);
        matx_free(alloc, Acopy);
        matx_free(alloc, tau);
        matx_free(alloc, l_data);
        return MATX_ERR_INTERNAL;
    }
    matx_free(alloc, tau);

    /* L = l_data (m x n) */
    {
        matx_dense_d_i8_opaque_t* le = (matx_dense_d_i8_opaque_t*) matx_malloc(alloc,
            sizeof(matx_dense_d_i8_opaque_t));
        if (!le) {
            matx_free(alloc, Acopy);
            matx_free(alloc, l_data);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        memset(le, 0, sizeof(*le));
        le->alloc = *alloc;
        le->nrows = m;
        le->ncols = n;
        le->layout = A->layout;
        le->stride = (A->layout == MATX_COL_MAJOR) ? m : n;
        le->flags = 1u;
        le->data = l_data;
        if (*L) {
            if ((*L)->flags & 1u)
                matx_free(alloc, (*L)->data);
            matx_free(alloc, *L);
        }
        *L = le;
    }

    /* Q = Acopy (k x n, leading rows) */
    {
        matx_dense_d_i8_opaque_t* qe = (matx_dense_d_i8_opaque_t*) matx_malloc(alloc,
            sizeof(matx_dense_d_i8_opaque_t));
        if (!qe) {
            matx_free(alloc, Acopy);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        memset(qe, 0, sizeof(*qe));
        qe->alloc = *alloc;
        qe->nrows = k;
        qe->ncols = n;
        qe->layout = A->layout;
        qe->stride = lda;
        qe->flags = 1u;
        qe->data = Acopy;
        if (*Q) {
            if ((*Q)->flags & 1u)
                matx_free(alloc, (*Q)->data);
            matx_free(alloc, *Q);
        }
        *Q = qe;
    }
    return MATX_OK;
}

static matx_status_t ss_lq_z_i8(const matx_alloc_t* alloc,
                                const matx_dense_z_i8_t A,
                                matx_dense_z_i8_t* L,
                                matx_dense_z_i8_t* Q)
{
    if (!A || !L || !Q) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const matx_int64_t m = A->nrows, n = A->ncols;
    const matx_int64_t k = (m < n) ? m : n;

    matx_int64_t lda = ss_packed_lda(A->layout, m, n);

    matx_complex_d_t* Acopy = (matx_complex_d_t*) matx_malloc(alloc,
        (size_t) m * (size_t) n * sizeof(matx_complex_d_t));
    matx_complex_d_t* tau_c = (matx_complex_d_t*) matx_malloc(alloc,
        (size_t) k * sizeof(matx_complex_d_t));
    if (!Acopy || !tau_c) {
        matx_free(alloc, Acopy);
        matx_free(alloc, tau_c);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    ss_pack_z_i8(A->layout, m, n, A->stride, A->data, Acopy);

    matx_int64_t info = LAPACKE_zgelqf(ss_layout_to_lapack(A->layout), m, n,
                                       (lapack_complex_double*) Acopy, lda,
                                       (lapack_complex_double*) tau_c);
    if (info != 0) {
        MATX_ERROR("LAPACKE_zgelqf error: %d", (int) info);
        matx_free(alloc, Acopy);
        matx_free(alloc, tau_c);
        return MATX_ERR_INTERNAL;
    }

    matx_complex_d_t* l_data = (matx_complex_d_t*) matx_malloc(alloc,
        (size_t) m * (size_t) n * sizeof(matx_complex_d_t));
    if (!l_data) {
        matx_free(alloc, Acopy);
        matx_free(alloc, tau_c);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(l_data, 0, (size_t) m * (size_t) n * sizeof(matx_complex_d_t));
    for (matx_int64_t j = 0; j < n; ++j)
        for (matx_int64_t i = j; i < m; ++i) {
            matx_int64_t idx = (A->layout == MATX_COL_MAJOR) ? i + j * lda : j + i * lda;
            matx_int64_t lidx = (A->layout == MATX_COL_MAJOR) ? i + j * m : j + i * m;
            l_data[lidx] = Acopy[idx];
        }

    info = LAPACKE_zunglq(ss_layout_to_lapack(A->layout), k, n, k,
                          (lapack_complex_double*) Acopy, lda,
                          (lapack_complex_double*) tau_c);
    if (info != 0) {
        MATX_ERROR("LAPACKE_zunglq error: %d", (int) info);
        matx_free(alloc, Acopy);
        matx_free(alloc, tau_c);
        matx_free(alloc, l_data);
        return MATX_ERR_INTERNAL;
    }
    matx_free(alloc, tau_c);

    /* L = l_data (m x n) */
    {
        matx_dense_z_i8_opaque_t* le = (matx_dense_z_i8_opaque_t*) matx_malloc(alloc,
            sizeof(matx_dense_z_i8_opaque_t));
        if (!le) {
            matx_free(alloc, Acopy);
            matx_free(alloc, l_data);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        memset(le, 0, sizeof(*le));
        le->alloc = *alloc;
        le->nrows = m;
        le->ncols = n;
        le->layout = A->layout;
        le->stride = (A->layout == MATX_COL_MAJOR) ? m : n;
        le->flags = 1u;
        le->data = (matx_complex_d_t*) l_data;
        if (*L) {
            if ((*L)->flags & 1u)
                matx_free(alloc, (*L)->data);
            matx_free(alloc, *L);
        }
        *L = le;
    }

    /* Q = Acopy (k x n) */
    {
        matx_dense_z_i8_opaque_t* qe = (matx_dense_z_i8_opaque_t*) matx_malloc(alloc,
            sizeof(matx_dense_z_i8_opaque_t));
        if (!qe) {
            matx_free(alloc, Acopy);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        memset(qe, 0, sizeof(*qe));
        qe->alloc = *alloc;
        qe->nrows = k;
        qe->ncols = n;
        qe->layout = A->layout;
        qe->stride = lda;
        qe->flags = 1u;
        qe->data = (matx_complex_d_t*) Acopy;
        if (*Q) {
            if ((*Q)->flags & 1u)
                matx_free(alloc, (*Q)->data);
            matx_free(alloc, *Q);
        }
        *Q = qe;
    }
    return MATX_OK;
}

matx_dense_linsolve_t matx_dense_linsolve_make_cblas(matx_alloc_t alloc)
{
    matx_dense_linsolve_t ls = {.kind = MATX_LINSOLVE_BACKEND_CBLAS,
                                .alloc = alloc,
                                .vt = {
                                .factor_dense_d_i8 = &ss_factor_dense_d_i8,
                                .solve_dense_d_i8 = &ss_solve_dense_d_i8,
                                .factor_dense_d_i8_destroy = &ss_factor_dense_d_i8_destroy,
                                .factor_dense_z_i8 = &ss_factor_dense_z_i8,
                                .solve_dense_z_i8 = &ss_solve_dense_z_i8,
                                .factor_dense_z_i8_destroy = &ss_factor_dense_z_i8_destroy,
                                .potrf_d_i8 = &ss_potrf_d_i8,
                                .potrs_d_i8 = &ss_potrs_d_i8,
                                .potrf_z_i8 = &ss_potrf_z_i8,
                                .potrs_z_i8 = &ss_potrs_z_i8,
                                .gels_d_i8 = &ss_gels_d_i8,
                                .gels_z_i8 = &ss_gels_z_i8,
                                .syev_d_i8 = &ss_syev_d_i8,
                                .syev_z_i8 = &ss_syev_z_i8,
                                .geev_d_i8 = &ss_geev_d_i8,
                                .geev_z_i8 = &ss_geev_z_i8,
                                .qr_d_i8 = &ss_qr_d_i8,
                                .qr_z_i8 = &ss_qr_z_i8,
                                .det_dense_d_i8 = &ss_det_dense_d_i8,
                                .det_dense_z_i8 = &ss_det_dense_z_i8,
                                .cond_dense_d_i8 = &ss_cond_dense_d_i8,
                                .cond_dense_z_i8 = &ss_cond_dense_z_i8,
                                .gesvd_d_i8 = &ss_gesvd_d_i8,
                                .gesvd_z_i8 = &ss_gesvd_z_i8,
                                .solve_dense_mrhs_d_i8 = &ss_solve_dense_mrhs_d_i8,
                                .solve_dense_mrhs_z_i8 = &ss_solve_dense_mrhs_z_i8,
                                .potrs_mrhs_d_i8 = &ss_potrs_mrhs_d_i8,
                                .potrs_mrhs_z_i8 = &ss_potrs_mrhs_z_i8,
                                .sytrf_d_i8 = &ss_sytrf_d_i8,
                                .sytrs_d_i8 = &ss_sytrs_d_i8,
                                .sytrf_destroy_d = &ss_sytrf_destroy_d,
                                .sytrf_z_i8 = &ss_sytrf_z_i8,
                                .sytrs_z_i8 = &ss_sytrs_z_i8,
                                .sytrf_destroy_z = &ss_sytrf_destroy_z,
                                .qrp_d_i8 = &ss_qrp_d_i8,
                                .qrp_z_i8 = &ss_qrp_z_i8,
                                .pinv_d_i8 = &ss_pinv_d_i8,
                                .pinv_z_i8 = &ss_pinv_z_i8,
                                .rank_d_i8 = &ss_rank_d_i8,
                                .rank_z_i8 = &ss_rank_z_i8,
                                .sygv_d_i8 = &ss_sygv_d_i8,
                                .sygv_z_i8 = &ss_sygv_z_i8,
                                .lq_d_i8 = &ss_lq_d_i8,
                                .lq_z_i8 = &ss_lq_z_i8,
                                }};

    return ls;
}
