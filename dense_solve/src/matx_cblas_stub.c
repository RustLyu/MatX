#include "matx/matx_dense_solve.h"
#include "matx/matx_log.h"
#include "matx/matx_types_internal.h"

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
};

struct matx_factor_dense_z_i8_t
{
    matx_int64_t n;
    matx_int64_t lda;
    matx_double* lu;
    matx_int64_t* piv;
    matx_layout_t layout;
    matx_uplo_t uplo;
};

static void ss_factor_dense_d_i8_destroy(matx_factor_dense_d_i8_t* F)
{
    if (!F)
        return;
    free(F->lu);
    free(F->piv);
    free(F);
}

static void ss_factor_dense_z_i8_destroy(matx_factor_dense_z_i8_t* F)
{
    if (!F)
        return;
    free(F->lu);
    free(F->piv);
    free(F);
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

// ---- Dense real LU ----

static matx_status_t ss_factor_dense_d_i8(const matx_dense_d_i8_t A,
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
        ss_factor_dense_d_i8_destroy(*out_F);

    const matx_int64_t n = A->nrows;
    matx_int64_t lda = ss_packed_lda(A->layout, n, n);

    matx_factor_dense_d_i8_t* F = (matx_factor_dense_d_i8_t*) malloc(sizeof(*F));
    if (!F) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(F, 0, sizeof(*F));
    F->n = n;
    F->lda = lda;
    F->layout = A->layout;

    F->lu = (matx_double*) malloc((size_t) n * (size_t) n * sizeof(matx_double));
    F->piv = (matx_int64_t*) malloc(n * sizeof(matx_int64_t));
    if (!F->lu || !F->piv) {
        free(F->lu);
        free(F->piv);
        free(F);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    ss_pack_d_i8(A->layout, n, n, A->stride, A->data, F->lu);

    matx_int64_t info = LAPACKE_dgetrf(ss_layout_to_lapack(A->layout), n, n, F->lu, lda, F->piv);
    if (info != 0) {
        MATX_ERROR("LAPACKE_dgetrf error:%d", info);
        free(F->lu);
        free(F->piv);
        free(F);
        return MATX_ERR_INTERNAL;
    }

    *out_F = F;
    return MATX_OK;
}

static matx_status_t ss_solve_dense_d_i8(const matx_factor_dense_d_i8_t* F,
                                         const matx_double* b,
                                         matx_double* x)
{
    if (!F || !b || !x) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    memcpy(x, b, F->n * sizeof(matx_double));

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

static matx_status_t ss_factor_dense_z_i8(const matx_dense_z_i8_t A,
                                          matx_factor_dense_z_i8_t** out_F)
{
#if !(defined(MATX_HAVE_OPENBLAS) || defined(MATX_HAVE_BLIS))
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
        ss_factor_dense_z_i8_destroy(*out_F);

    const matx_int64_t n = A->nrows;
    matx_int64_t lda = ss_packed_lda(A->layout, n, n);

    matx_factor_dense_z_i8_t* F = (matx_factor_dense_z_i8_t*) malloc(sizeof(*F));
    if (!F) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(F, 0, sizeof(*F));
    F->n = n;
    F->lda = lda;
    F->layout = A->layout;

    F->lu = (matx_double*) malloc((size_t) n * (size_t) n * sizeof(matx_complex_d_t));
    F->piv = (matx_int64_t*) malloc(n * sizeof(matx_int64_t));
    if (!F->lu || !F->piv) {
        free(F->lu);
        free(F->piv);
        free(F);
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
        free(F->lu);
        free(F->piv);
        free(F);
        return MATX_ERR_INTERNAL;
    }

    *out_F = F;
    return MATX_OK;
#endif
}

static matx_status_t ss_solve_dense_z_i8(const matx_factor_dense_z_i8_t* F,
                                         const matx_vec_z_i8_t b,
                                         matx_vec_z_i8_t x)
{
    if (!F || !b || !x) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    memcpy(x->data, b->data, F->n * sizeof(matx_complex_d_t));

    matx_int64_t ldb = (F->layout == MATX_COL_MAJOR) ? F->n : 1;
    matx_int64_t info = LAPACKE_zgetrs(ss_layout_to_lapack(F->layout),
                                       'N',
                                       F->n,
                                       1,
                                       (lapack_complex_double*) F->lu,
                                       F->lda,
                                       F->piv,
                                       (lapack_complex_double*) x->data,
                                       ldb);
    if (info != 0) {
        MATX_ERROR("LAPACKE_zgetrs error:%d", info);
        return MATX_ERR_INTERNAL;
    }
    return MATX_OK;
}

// ---- Cholesky ----

static inline char matx_uplo_to_lapack(matx_uplo_t uplo)
{
    return (uplo == MATX_UPPER) ? 'U' : 'L';
}

static matx_status_t ss_potrf_d_i8(const matx_dense_d_i8_t A,
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
        ss_factor_dense_d_i8_destroy(*out_F);

    const matx_int64_t n = A->nrows;
    matx_int64_t lda = ss_packed_lda(A->layout, n, n);

    matx_factor_dense_d_i8_t* F = (matx_factor_dense_d_i8_t*) malloc(sizeof(*F));
    if (!F) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(F, 0, sizeof(*F));
    F->n = n;
    F->lda = lda;
    F->layout = A->layout;
    F->uplo = uplo;

    F->lu = (matx_double*) malloc((size_t) n * (size_t) n * sizeof(matx_double));
    MATX_ERROR("%s: out of memory", __func__);
    if (!F->lu) {
        free(F);
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
        free(F->lu);
        free(F);
        return MATX_ERR_INTERNAL;
    }
    *out_F = F;
    return MATX_OK;
}

static matx_status_t ss_potrs_d_i8(const matx_factor_dense_d_i8_t* F,
                                   const matx_double* b,
                                   matx_double* x)
{
    if (!F || !b || !x) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    memcpy(x, b, F->n * sizeof(matx_double));

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

static matx_status_t ss_potrf_z_i8(const matx_dense_z_i8_t A,
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
        ss_factor_dense_z_i8_destroy(*out_F);

    const matx_int64_t n = A->nrows;
    matx_int64_t lda = ss_packed_lda(A->layout, n, n);

    matx_factor_dense_z_i8_t* F = (matx_factor_dense_z_i8_t*) malloc(sizeof(*F));
    if (!F) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    memset(F, 0, sizeof(*F));
    F->n = n;
    F->lda = lda;
    F->layout = A->layout;
    F->uplo = uplo;

    F->lu = (matx_double*) malloc((size_t) n * (size_t) n * sizeof(matx_complex_d_t));
    MATX_ERROR("%s: out of memory", __func__);
    if (!F->lu) {
        free(F);
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
        free(F->lu);
        free(F);
        return MATX_ERR_INTERNAL;
    }
    *out_F = F;
    return MATX_OK;
}

static matx_status_t ss_potrs_z_i8(const matx_factor_dense_z_i8_t* F,
                                   const matx_vec_z_i8_t b,
                                   matx_vec_z_i8_t x)
{
    if (!F || !b || !x) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    memcpy(x->data, b->data, F->n * sizeof(matx_complex_d_t));

    matx_int64_t ldb = (F->layout == MATX_COL_MAJOR) ? F->n : 1;
    matx_int64_t info = LAPACKE_zpotrs(ss_layout_to_lapack(F->layout),
                                       matx_uplo_to_lapack(F->uplo),
                                       F->n,
                                       1,
                                       (lapack_complex_double*) F->lu,
                                       F->lda,
                                       (lapack_complex_double*) x->data,
                                       ldb);
    if (info != 0) {
        MATX_ERROR("LAPACKE_zpotrs error: %d", info);
        return MATX_ERR_INTERNAL;
    }
    return MATX_OK;
}

// ---- GELS ----

static matx_status_t ss_gels_d_i8(const matx_dense_d_i8_t A, const matx_double* b, matx_double* x)
{
    if (!A || !b || !x) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const matx_int64_t m = A->nrows, n = A->ncols;
    const matx_int64_t lda = ss_packed_lda(A->layout, m, n);
    const matx_int64_t blen = (m > n) ? m : n;
    const matx_int64_t ldb = (A->layout == MATX_COL_MAJOR) ? blen : 1;

    matx_double* Acopy = (matx_double*) malloc((size_t) m * (size_t) n * sizeof(matx_double));
    matx_double* bcopy = (matx_double*) calloc(blen, sizeof(matx_double));
    MATX_ERROR("%s: out of memory", __func__);
    if (!Acopy || !bcopy) {
        free(Acopy);
        free(bcopy);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    ss_pack_d_i8(A->layout, m, n, A->stride, A->data, Acopy);
    memcpy(bcopy, b, m * sizeof(matx_double));

    matx_int64_t info
        = LAPACKE_dgels(ss_layout_to_lapack(A->layout), 'N', m, n, 1, Acopy, lda, bcopy, ldb);
    if (info != 0) {
        MATX_ERROR("LAPACKE_dgels error: %d", info);
        free(Acopy);
        free(bcopy);
        return MATX_ERR_INTERNAL;
    }
    memcpy(x, bcopy, n * sizeof(matx_double));
    free(Acopy);
    free(bcopy);
    return MATX_OK;
}

static matx_status_t ss_gels_z_i8(const matx_dense_z_i8_t A,
                                  const matx_vec_z_i8_t b,
                                  matx_vec_z_i8_t x)
{
    if (!A || !b || !x) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    const matx_int64_t m = A->nrows, n = A->ncols;
    const matx_int64_t lda = ss_packed_lda(A->layout, m, n);
    const matx_int64_t blen = (m > n) ? m : n;
    const matx_int64_t ldb = (A->layout == MATX_COL_MAJOR) ? blen : 1;

    matx_complex_d_t* Acopy = (matx_complex_d_t*) malloc((size_t) m * (size_t) n
                                                         * sizeof(matx_complex_d_t));
    matx_complex_d_t* bcopy = (matx_complex_d_t*) calloc(blen, sizeof(matx_complex_d_t));
    MATX_ERROR("%s: out of memory", __func__);
    if (!Acopy || !bcopy) {
        free(Acopy);
        free(bcopy);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    ss_pack_z_i8(A->layout, m, n, A->stride, A->data, Acopy);
    memcpy(bcopy, b->data, m * sizeof(matx_complex_d_t));

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
        free(Acopy);
        free(bcopy);
        return MATX_ERR_INTERNAL;
    }
    memcpy(x->data, bcopy, n * sizeof(matx_complex_d_t));
    free(Acopy);
    free(bcopy);
    return MATX_OK;
}

// ---- SYEV ----

static matx_status_t ss_syev_d_i8(const matx_dense_d_i8_t A,
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

    matx_double* Acopy = (matx_double*) malloc((size_t) n * (size_t) n * sizeof(matx_double));
    if (!Acopy) {
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    ss_pack_d_i8(A->layout, n, n, A->stride, A->data, Acopy);

    matx_int64_t info
        = LAPACKE_dsyev(ss_layout_to_lapack(A->layout), jobz, 'L', n, Acopy, lda, eigenvalues->data);
    if (info != 0) {
        MATX_ERROR("LAPACKE_dsyev error: %d", info);
        free(Acopy);
        return MATX_ERR_INTERNAL;
    }
    if (eigenvectors) {
        matx_dense_d_i8_opaque_t* ev = (matx_dense_d_i8_opaque_t*) malloc(
            sizeof(matx_dense_d_i8_opaque_t));
        MATX_ERROR("%s: out of memory", __func__);
        if (!ev) {
            free(Acopy);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        memset(ev, 0, sizeof(*ev));
        ev->nrows = n;
        ev->ncols = n;
        ev->layout = A->layout;
        ev->stride = lda;
        ev->flags = 1u;
        ev->data = Acopy;
        if (*eigenvectors) {
            if ((*eigenvectors)->flags & 1u)
                free((*eigenvectors)->data);
            free(*eigenvectors);
        }
        *eigenvectors = ev;
    } else {
        free(Acopy);
    }
    return MATX_OK;
}

// ---- GESVD ----

static matx_status_t ss_gesvd_d_i8(const matx_dense_d_i8_t A,
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

    matx_double* Acopy = (matx_double*) malloc((size_t) m * (size_t) n * sizeof(matx_double));
    matx_double* u_data = U ? (matx_double*) malloc((size_t) m * (size_t) m * sizeof(matx_double))
                            : NULL;
    matx_double* vt_data = Vt ? (matx_double*) malloc((size_t) n * (size_t) n * sizeof(matx_double))
                              : NULL;
    matx_double* superb = (matx_double*) malloc(k * sizeof(matx_double));
    if (!Acopy || !superb || (U && !u_data) || (Vt && !vt_data)) {
        free(Acopy);
        free(u_data);
        free(vt_data);
        free(superb);
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
    free(Acopy);
    free(superb);
    if (info != 0) {
        MATX_ERROR("LAPACKE_dgesvd error: %d", info);
        free(u_data);
        free(vt_data);
        return MATX_ERR_INTERNAL;
    }
    if (U) {
        matx_int64_t u_lda = ss_packed_lda(A->layout, m, m);
        matx_dense_d_i8_opaque_t* ev = (matx_dense_d_i8_opaque_t*) malloc(
            sizeof(matx_dense_d_i8_opaque_t));
        MATX_ERROR("%s: out of memory", __func__);
        if (!ev) {
            free(u_data);
            free(vt_data);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        memset(ev, 0, sizeof(*ev));
        ev->nrows = m;
        ev->ncols = m;
        ev->layout = A->layout;
        ev->stride = u_lda;
        ev->flags = 1u;
        ev->data = u_data;
        if (*U) {
            if ((*U)->flags & 1u)
                free((*U)->data);
            free(*U);
        }
        *U = ev;
    }
    if (Vt) {
        matx_int64_t vt_lda = ss_packed_lda(A->layout, n, n);
        matx_dense_d_i8_opaque_t* ev = (matx_dense_d_i8_opaque_t*) malloc(
            sizeof(matx_dense_d_i8_opaque_t));
        MATX_ERROR("%s: out of memory", __func__);
        if (!ev) {
            free(vt_data);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        memset(ev, 0, sizeof(*ev));
        ev->nrows = n;
        ev->ncols = n;
        ev->layout = A->layout;
        ev->stride = vt_lda;
        ev->flags = 1u;
        ev->data = vt_data;
        if (*Vt) {
            if ((*Vt)->flags & 1u)
                free((*Vt)->data);
            free(*Vt);
        }
        *Vt = ev;
    }
    return MATX_OK;
}

static matx_status_t ss_gesvd_z_i8(const matx_dense_z_i8_t A,
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

    matx_complex_d_t* Acopy = (matx_complex_d_t*) malloc((size_t) m * (size_t) n
                                                         * sizeof(matx_complex_d_t));
    matx_complex_d_t* u_data = U ? (matx_complex_d_t*) malloc((size_t) m * (size_t) m
                                                              * sizeof(matx_complex_d_t))
                                 : NULL;
    matx_complex_d_t* vt_data = Vt ? (matx_complex_d_t*) malloc((size_t) n * (size_t) n
                                                                * sizeof(matx_complex_d_t))
                                   : NULL;
    matx_double* superb = (matx_double*) malloc(k * sizeof(matx_double));
    if (!Acopy || !superb || (U && !u_data) || (Vt && !vt_data)) {
        free(Acopy);
        free(u_data);
        free(vt_data);
        free(superb);
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
    free(Acopy);
    free(superb);
    if (info != 0) {
        MATX_ERROR("LAPACKE_zgesvd error: %d", info);
        free(u_data);
        free(vt_data);
        return MATX_ERR_INTERNAL;
    }
    if (U) {
        matx_int64_t u_lda = ss_packed_lda(A->layout, m, m);
        matx_dense_z_i8_opaque_t* ev = (matx_dense_z_i8_opaque_t*) malloc(
            sizeof(matx_dense_z_i8_opaque_t));
        MATX_ERROR("%s: out of memory", __func__);
        if (!ev) {
            free(u_data);
            free(vt_data);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        memset(ev, 0, sizeof(*ev));
        ev->nrows = m;
        ev->ncols = m;
        ev->layout = A->layout;
        ev->stride = u_lda;
        ev->flags = 1u;
        ev->data = u_data;
        if (*U) {
            if ((*U)->flags & 1u)
                free((*U)->data);
            free(*U);
        }
        *U = ev;
    }
    if (Vt) {
        matx_int64_t vt_lda = ss_packed_lda(A->layout, n, n);
        matx_dense_z_i8_opaque_t* ev = (matx_dense_z_i8_opaque_t*) malloc(
            sizeof(matx_dense_z_i8_opaque_t));
        MATX_ERROR("%s: out of memory", __func__);
        if (!ev) {
            free(vt_data);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        memset(ev, 0, sizeof(*ev));
        ev->nrows = n;
        ev->ncols = n;
        ev->layout = A->layout;
        ev->stride = vt_lda;
        ev->flags = 1u;
        ev->data = vt_data;
        if (*Vt) {
            if ((*Vt)->flags & 1u)
                free((*Vt)->data);
            free(*Vt);
        }
        *Vt = ev;
    }
    return MATX_OK;
}

// ---- SYEV (complex Hermitian) ----

static matx_status_t ss_syev_z_i8(const matx_dense_z_i8_t A,
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

    matx_complex_d_t* Acopy = (matx_complex_d_t*) malloc((size_t) n * (size_t) n
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
        free(Acopy);
        return MATX_ERR_INTERNAL;
    }
    if (eigenvectors) {
        matx_dense_z_i8_opaque_t* ev = (matx_dense_z_i8_opaque_t*) malloc(
            sizeof(matx_dense_z_i8_opaque_t));
        if (!ev) {
            free(Acopy);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        memset(ev, 0, sizeof(*ev));
        ev->nrows = n;
        ev->ncols = n;
        ev->layout = A->layout;
        ev->stride = lda;
        ev->flags = 1u;
        ev->data = Acopy;
        if (*eigenvectors) {
            if ((*eigenvectors)->flags & 1u)
                free((*eigenvectors)->data);
            free(*eigenvectors);
        }
        *eigenvectors = ev;
    } else {
        free(Acopy);
    }
    return MATX_OK;
}

// ---- GEEV (general eigenvalues, real) ----

static matx_status_t ss_geev_d_i8(const matx_dense_d_i8_t A,
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

    matx_double* Acopy = (matx_double*) malloc((size_t) n * (size_t) n * sizeof(matx_double));
    matx_double* wr = (matx_double*) malloc((size_t) n * sizeof(matx_double));
    matx_double* wi = (matx_double*) malloc((size_t) n * sizeof(matx_double));
    matx_double* vl_data = vl ? (matx_double*) malloc((size_t) n * (size_t) n * sizeof(matx_double))
                              : NULL;
    matx_double* vr_data = vr ? (matx_double*) malloc((size_t) n * (size_t) n * sizeof(matx_double))
                              : NULL;
    if (!Acopy || !wr || !wi || (vl && !vl_data) || (vr && !vr_data)) {
        free(Acopy);
        free(wr);
        free(wi);
        free(vl_data);
        free(vr_data);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    ss_pack_d_i8(A->layout, n, n, A->stride, A->data, Acopy);

    matx_int64_t info = LAPACKE_dgeev(
        ss_layout_to_lapack(A->layout), jobvl, jobvr, n, Acopy, lda, wr, wi, vl_data, n, vr_data, n);
    if (info != 0) {
        MATX_ERROR("LAPACKE_dgeev error: %d", (int) info);
        free(Acopy);
        free(wr);
        free(wi);
        free(vl_data);
        free(vr_data);
        return MATX_ERR_INTERNAL;
    }

    /* Pack wr + wi into complex eigenvalues */
    for (matx_int64_t i = 0; i < n; ++i) {
        eigenvalues->data[i * eigenvalues->stride].real = wr[i];
        eigenvalues->data[i * eigenvalues->stride].imag = wi[i];
    }

    /* Build vr (right eigenvectors) */
    if (vr) {
        matx_dense_d_i8_opaque_t* ev = (matx_dense_d_i8_opaque_t*) malloc(
            sizeof(matx_dense_d_i8_opaque_t));
        if (!ev) {
            free(vr_data);
            free(Acopy);
            free(wr);
            free(wi);
            free(vl_data);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        memset(ev, 0, sizeof(*ev));
        ev->nrows = n;
        ev->ncols = n;
        ev->layout = A->layout;
        ev->stride = n;
        ev->flags = 1u;
        ev->data = vr_data;
        if (*vr) {
            if ((*vr)->flags & 1u)
                free((*vr)->data);
            free(*vr);
        }
        *vr = ev;
    }
    if (vl) {
        matx_dense_d_i8_opaque_t* ev = (matx_dense_d_i8_opaque_t*) malloc(
            sizeof(matx_dense_d_i8_opaque_t));
        if (!ev) {
            if (vr && vr_data)
                free(vr_data);
            free(Acopy);
            free(wr);
            free(wi);
            free(vl_data);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        memset(ev, 0, sizeof(*ev));
        ev->nrows = n;
        ev->ncols = n;
        ev->layout = A->layout;
        ev->stride = n;
        ev->flags = 1u;
        ev->data = vl_data;
        if (*vl) {
            if ((*vl)->flags & 1u)
                free((*vl)->data);
            free(*vl);
        }
        *vl = ev;
    }
    free(Acopy);
    free(wr);
    free(wi);
    return MATX_OK;
}

// ---- GEEV (general eigenvalues, complex) ----

static matx_status_t ss_geev_z_i8(const matx_dense_z_i8_t A,
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

    matx_complex_d_t* Acopy = (matx_complex_d_t*) malloc((size_t) n * (size_t) n
                                                         * sizeof(matx_complex_d_t));
    matx_complex_d_t* w = (matx_complex_d_t*) malloc((size_t) n * sizeof(matx_complex_d_t));
    matx_complex_d_t* vl_data = vl ? (matx_complex_d_t*) malloc((size_t) n * (size_t) n
                                                                * sizeof(matx_complex_d_t))
                                   : NULL;
    matx_complex_d_t* vr_data = vr ? (matx_complex_d_t*) malloc((size_t) n * (size_t) n
                                                                * sizeof(matx_complex_d_t))
                                   : NULL;
    if (!Acopy || !w || (vl && !vl_data) || (vr && !vr_data)) {
        free(Acopy);
        free(w);
        free(vl_data);
        free(vr_data);
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
        free(Acopy);
        free(w);
        free(vl_data);
        free(vr_data);
        return MATX_ERR_INTERNAL;
    }

    for (matx_int64_t i = 0; i < n; ++i)
        eigenvalues->data[i * eigenvalues->stride] = w[i];

    if (vr) {
        matx_dense_z_i8_opaque_t* ev = (matx_dense_z_i8_opaque_t*) malloc(
            sizeof(matx_dense_z_i8_opaque_t));
        if (!ev) {
            free(vr_data);
            free(Acopy);
            free(w);
            free(vl_data);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        memset(ev, 0, sizeof(*ev));
        ev->nrows = n;
        ev->ncols = n;
        ev->layout = A->layout;
        ev->stride = n;
        ev->flags = 1u;
        ev->data = vr_data;
        if (*vr) {
            if ((*vr)->flags & 1u)
                free((*vr)->data);
            free(*vr);
        }
        *vr = ev;
    }
    if (vl) {
        matx_dense_z_i8_opaque_t* ev = (matx_dense_z_i8_opaque_t*) malloc(
            sizeof(matx_dense_z_i8_opaque_t));
        if (!ev) {
            if (vr && vr_data)
                free(vr_data);
            free(Acopy);
            free(w);
            free(vl_data);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        memset(ev, 0, sizeof(*ev));
        ev->nrows = n;
        ev->ncols = n;
        ev->layout = A->layout;
        ev->stride = n;
        ev->flags = 1u;
        ev->data = vl_data;
        if (*vl) {
            if ((*vl)->flags & 1u)
                free((*vl)->data);
            free(*vl);
        }
        *vl = ev;
    }
    free(Acopy);
    free(w);
    return MATX_OK;
}

// ---- QR Factorization ----

static matx_status_t ss_qr_d_i8(const matx_dense_d_i8_t A,
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

    matx_double* Acopy = (matx_double*) malloc((size_t) m * (size_t) n * sizeof(matx_double));
    matx_double* tau = (matx_double*) malloc((size_t) k * sizeof(matx_double));
    if (!Acopy || !tau) {
        free(Acopy);
        free(tau);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    ss_pack_d_i8(A->layout, m, n, A->stride, A->data, Acopy);

    matx_int64_t info = LAPACKE_dgeqrf(ss_layout_to_lapack(A->layout), m, n, Acopy, lda, tau);
    if (info != 0) {
        MATX_ERROR("LAPACKE_dgeqrf error: %d", (int) info);
        free(Acopy);
        free(tau);
        return MATX_ERR_INTERNAL;
    }

    /* Extract R: upper triangular part */
    matx_double* r_data = (matx_double*) calloc((size_t) n * (size_t) n, sizeof(matx_double));
    if (!r_data) {
        free(Acopy);
        free(tau);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
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
        free(Acopy);
        free(tau);
        free(r_data);
        return MATX_ERR_INTERNAL;
    }
    free(tau);

    /* Q = Acopy (m x k, leading columns) */
    {
        matx_dense_d_i8_opaque_t* qe = (matx_dense_d_i8_opaque_t*) malloc(
            sizeof(matx_dense_d_i8_opaque_t));
        if (!qe) {
            free(Acopy);
            free(r_data);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        memset(qe, 0, sizeof(*qe));
        qe->nrows = m;
        qe->ncols = k;
        qe->layout = A->layout;
        qe->stride = lda;
        qe->flags = 1u;
        qe->data = Acopy;
        if (*Q) {
            if ((*Q)->flags & 1u)
                free((*Q)->data);
            free(*Q);
        }
        *Q = qe;
    }

    /* R = r_data (k x n) */
    {
        matx_dense_d_i8_opaque_t* re = (matx_dense_d_i8_opaque_t*) malloc(
            sizeof(matx_dense_d_i8_opaque_t));
        if (!re) {
            free(r_data);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        memset(re, 0, sizeof(*re));
        re->nrows = k;
        re->ncols = n;
        re->layout = A->layout;
        re->stride = (A->layout == MATX_COL_MAJOR) ? k : n;
        re->flags = 1u;
        re->data = r_data;
        if (*R) {
            if ((*R)->flags & 1u)
                free((*R)->data);
            free(*R);
        }
        *R = re;
    }
    return MATX_OK;
}

// ---- QR Factorization (complex) ----

static matx_status_t ss_qr_z_i8(const matx_dense_z_i8_t A,
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

    matx_complex_d_t* Acopy = (matx_complex_d_t*) malloc((size_t) m * (size_t) n
                                                         * sizeof(matx_complex_d_t));
    matx_complex_d_t* tau = (matx_complex_d_t*) malloc((size_t) k * sizeof(matx_complex_d_t));
    if (!Acopy || !tau) {
        free(Acopy);
        free(tau);
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
        free(Acopy);
        free(tau);
        return MATX_ERR_INTERNAL;
    }

    matx_complex_d_t* r_data = (matx_complex_d_t*) calloc((size_t) n * (size_t) n,
                                                          sizeof(matx_complex_d_t));
    if (!r_data) {
        free(Acopy);
        free(tau);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }
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
        free(Acopy);
        free(tau);
        free(r_data);
        return MATX_ERR_INTERNAL;
    }
    free(tau);

    {
        matx_dense_z_i8_opaque_t* qe = (matx_dense_z_i8_opaque_t*) malloc(
            sizeof(matx_dense_z_i8_opaque_t));
        if (!qe) {
            free(Acopy);
            free(r_data);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        memset(qe, 0, sizeof(*qe));
        qe->nrows = m;
        qe->ncols = k;
        qe->layout = A->layout;
        qe->stride = lda;
        qe->flags = 1u;
        qe->data = Acopy;
        if (*Q) {
            if ((*Q)->flags & 1u)
                free((*Q)->data);
            free(*Q);
        }
        *Q = qe;
    }

    {
        matx_dense_z_i8_opaque_t* re = (matx_dense_z_i8_opaque_t*) malloc(
            sizeof(matx_dense_z_i8_opaque_t));
        if (!re) {
            free(r_data);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        memset(re, 0, sizeof(*re));
        re->nrows = k;
        re->ncols = n;
        re->layout = A->layout;
        re->stride = (A->layout == MATX_COL_MAJOR) ? k : n;
        re->flags = 1u;
        re->data = r_data;
        if (*R) {
            if ((*R)->flags & 1u)
                free((*R)->data);
            free(*R);
        }
        *R = re;
    }
    return MATX_OK;
}

// ---- Matrix determinant (via LU) ----

static matx_status_t ss_det_dense_d_i8(const matx_dense_d_i8_t A, matx_double* det)
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
    matx_int64_t lda = ss_packed_lda(A->layout, n, n);

    matx_double* Acopy = (matx_double*) malloc((size_t) n * (size_t) n * sizeof(matx_double));
    matx_int64_t* piv = (matx_int64_t*) malloc((size_t) n * sizeof(matx_int64_t));
    if (!Acopy || !piv) {
        free(Acopy);
        free(piv);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    ss_pack_d_i8(A->layout, n, n, A->stride, A->data, Acopy);

    matx_int64_t info = LAPACKE_dgetrf(ss_layout_to_lapack(A->layout), n, n, Acopy, lda, piv);
    if (info < 0) {
        MATX_ERROR("LAPACKE_dgetrf error: %d", (int) info);
        free(Acopy);
        free(piv);
        return MATX_ERR_INTERNAL;
    }
    if (info > 0) {
        /* Singular matrix, determinant is zero */
        *det = 0.0;
        free(Acopy);
        free(piv);
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
    free(Acopy);
    free(piv);
    return MATX_OK;
}

// ---- Matrix determinant (complex) ----

static matx_status_t ss_det_dense_z_i8(const matx_dense_z_i8_t A, matx_complex_d_t* det)
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
    matx_int64_t lda = ss_packed_lda(A->layout, n, n);

    matx_complex_d_t* Acopy = (matx_complex_d_t*) malloc((size_t) n * (size_t) n
                                                         * sizeof(matx_complex_d_t));
    matx_int64_t* piv = (matx_int64_t*) malloc((size_t) n * sizeof(matx_int64_t));
    if (!Acopy || !piv) {
        free(Acopy);
        free(piv);
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
        free(Acopy);
        free(piv);
        return MATX_ERR_INTERNAL;
    }
    if (info > 0) {
        det->real = 0.0;
        det->imag = 0.0;
        free(Acopy);
        free(piv);
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
    free(Acopy);
    free(piv);
    return MATX_OK;
}

// ---- Condition number estimation (1-norm) ----

static matx_status_t ss_cond_dense_d_i8(const matx_dense_d_i8_t A, matx_double* cond)
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

    matx_double* Acopy = (matx_double*) malloc((size_t) n * (size_t) n * sizeof(matx_double));
    matx_int64_t* piv = (matx_int64_t*) malloc((size_t) n * sizeof(matx_int64_t));
    if (!Acopy || !piv) {
        free(Acopy);
        free(piv);
        MATX_ERROR("%s: out of memory", __func__);
        return MATX_ERR_OUT_OF_MEMORY;
    }

    ss_pack_d_i8(A->layout, n, n, A->stride, A->data, Acopy);

    matx_int64_t info = LAPACKE_dgetrf(ss_layout_to_lapack(A->layout), n, n, Acopy, lda, piv);
    if (info != 0) {
        MATX_ERROR("LAPACKE_dgetrf error: %d", (int) info);
        free(Acopy);
        free(piv);
        return MATX_ERR_INTERNAL;
    }

    matx_double rcond = 0.0;
    info = LAPACKE_dgecon(ss_layout_to_lapack(A->layout), '1', n, Acopy, lda, anorm, &rcond);
    if (info != 0) {
        MATX_ERROR("LAPACKE_dgecon error: %d", (int) info);
        free(Acopy);
        free(piv);
        return MATX_ERR_INTERNAL;
    }

    *cond = (rcond > 0.0) ? (1.0 / rcond) : (matx_double)INFINITY;
    free(Acopy);
    free(piv);
    return MATX_OK;
}

// ---- Condition number estimation (complex, 1-norm) ----

static matx_status_t ss_cond_dense_z_i8(const matx_dense_z_i8_t A, matx_double* cond)
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

    matx_complex_d_t* Acopy = (matx_complex_d_t*) malloc((size_t) n * (size_t) n
                                                         * sizeof(matx_complex_d_t));
    matx_int64_t* piv = (matx_int64_t*) malloc((size_t) n * sizeof(matx_int64_t));
    if (!Acopy || !piv) {
        free(Acopy);
        free(piv);
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
        free(Acopy);
        free(piv);
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
        free(Acopy);
        free(piv);
        return MATX_ERR_INTERNAL;
    }

    *cond = (rcond > 0.0) ? (1.0 / rcond) : (matx_double)INFINITY;
    free(Acopy);
    free(piv);
    return MATX_OK;
}

matx_dense_linsolve_t matx_dense_linsolve_make_cblas(void)
{
    matx_dense_linsolve_t ls = {.kind = MATX_LINSOLVE_BACKEND_CBLAS,
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
                                }};

    return ls;
}
