#include "matx/matx_dense_compute.h"
#include "matx/matx_types_internal.h"

// Forward decls
matx_dense_backend_t matx_blas_make_reference(void);

const char* matx_blas_backend_name(matx_dense_backend_kind_t k)
{
    switch (k) {
    case MATX_BLAS_BACKEND_REFERENCE:
        return "REFERENCE";
    case MATX_BLAS_BACKEND_OPENBLAS:
        return "OPENBLAS";
    case MATX_BLAS_BACKEND_BLIS:
        return "BLIS";
    default:
        return "UNKNOWN";
    }
}

static matx_dense_backend_t choose_default_backend(void)
{
    return matx_blas_make_reference();
}

matx_dense_backend_t matx_blas_default(void)
{
    return choose_default_backend();
}

matx_status_t matx_gemm_d_i8(const matx_dense_backend_t* blas,
                             matx_int64_t trans_a,
                             matx_int64_t trans_b,
                             matx_double alpha,
                             const matx_dense_d_i8_t A,
                             const matx_dense_d_i8_t B,
                             matx_double beta,
                             matx_dense_d_i8_t C)
{
    if (!blas || !A || !B || !C)
        return MATX_ERR_INVALID_ARG;
    if (!blas->vt.dgemm)
        return MATX_ERR_NOT_SUPPORTED;

    if (A->layout != B->layout || A->layout != C->layout)
        return MATX_ERR_INVALID_ARG;
    if (A->layout != MATX_COL_MAJOR && A->layout != MATX_ROW_MAJOR)
        return MATX_ERR_INVALID_ARG;

    const matx_int64_t a_rows = A->nrows;
    const matx_int64_t a_cols = A->ncols;
    const matx_int64_t b_rows = B->nrows;
    const matx_int64_t b_cols = B->ncols;

    const matx_int64_t m = (trans_a ? a_cols : a_rows);
    const matx_int64_t kA = (trans_a ? a_rows : a_cols);
    const matx_int64_t kB = (trans_b ? b_cols : b_rows);
    const matx_int64_t n = (trans_b ? b_rows : b_cols);

    if (kA != kB)
        return MATX_ERR_INVALID_ARG;
    if (C->nrows != m || C->ncols != n)
        return MATX_ERR_INVALID_ARG;

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

matx_status_t matx_gemm_z_i8(const matx_dense_backend_t* blas,
                             matx_int64_t trans_a,
                             matx_int64_t trans_b,
                             matx_complex_d_i8_t alpha,
                             const matx_dense_z_i8_t A,
                             const matx_dense_z_i8_t B,
                             matx_complex_d_i8_t beta,
                             matx_dense_z_i8_t C)
{
    if (!A || !B || !C || !A->data || !B->data || !C->data)
        return MATX_ERR_INVALID_ARG;
    if (A->layout != B->layout || A->layout != C->layout)
        return MATX_ERR_INVALID_ARG;

    const matx_int64_t a_rows = A->nrows;
    const matx_int64_t a_cols = A->ncols;
    const matx_int64_t b_rows = B->nrows;
    const matx_int64_t b_cols = B->ncols;
    const matx_int64_t m = trans_a ? a_cols : a_rows;
    const matx_int64_t kA = trans_a ? a_rows : a_cols;
    const matx_int64_t kB = trans_b ? b_cols : b_rows;
    const matx_int64_t n = trans_b ? b_rows : b_cols;

    if (kA != kB)
        return MATX_ERR_INVALID_ARG;
    if (C->nrows != m || C->ncols != n)
        return MATX_ERR_INVALID_ARG;

    return blas->vt.zgemm(A->layout,
                          trans_a,
                          trans_b,
                          m,
                          n,
                          kA,
                          &alpha,
                          A->data,
                          A->stride,
                          B->data,
                          B->stride,
                          &beta,
                          C->data,
                          C->stride);
}

matx_status_t matx_gemv_z_i8(const matx_dense_backend_t* blas,
                             matx_int64_t trans_a,
                             matx_complex_d_i8_t alpha,
                             const matx_dense_z_i8_t A,
                             const matx_vec_z_i8_t x,
                             matx_complex_d_i8_t beta,
                             matx_vec_z_i8_t y)
{
    (void) blas;
    if (!A || !x || !y || !A->data || !x->data || !y->data)
        return MATX_ERR_INVALID_ARG;

    const matx_int64_t m = A->nrows;
    const matx_int64_t n = A->ncols;
    const matx_int64_t len_x = trans_a ? m : n;
    const matx_int64_t len_y = trans_a ? n : m;

    if (x->n != len_x || y->n != len_y)
        return MATX_ERR_INVALID_ARG;

    return blas->vt.zgemv(A->layout,
                          trans_a,
                          m,
                          n,
                          &alpha,
                          A->data,
                          A->stride,
                          x->data,
                          x->stride,
                          &beta,
                          y->data,
                          y->stride);
}

matx_status_t matx_gemv_d_i8(const matx_dense_backend_t* blas,
                             matx_int64_t trans_a,
                             matx_double alpha,
                             const matx_dense_d_i8_t A,
                             const matx_vec_d_i8_t x,
                             matx_double beta,
                             matx_vec_d_i8_t y)
{
    (void) blas;
    if (!A || !x || !y || !A->data || !x->data || !y->data)
        return MATX_ERR_INVALID_ARG;

    const size_t m = A->nrows;
    const size_t n = A->ncols;
    const size_t len_x = trans_a ? m : n;
    const size_t len_y = trans_a ? n : m;

    if (x->n != len_x || y->n != len_y)
        return MATX_ERR_INVALID_ARG;

    return blas->vt.dgemv(A->layout,
                          trans_a,
                          m,
                          n,
                          alpha,
                          A->data,
                          A->stride,
                          x->data,
                          x->stride,
                          beta,
                          y->data,
                          y->stride);
}

matx_status_t matx_geadd_z_i8(const matx_dense_backend_t* blas,
                              matx_complex_d_i8_t alpha,
                              const matx_dense_z_i8_t A,
                              matx_complex_d_i8_t beta,
                              matx_dense_z_i8_t B)
{
    if (!A || !B || !A->data || !B->data)
        return MATX_ERR_INVALID_ARG;
    if (A->nrows != B->nrows || A->ncols != B->ncols)
        return MATX_ERR_INVALID_ARG;
    if (A->layout != B->layout)
        return MATX_ERR_INVALID_ARG;

    const size_t m = A->nrows;
    const size_t n = A->ncols;
    return blas->vt.zgeadd(A->layout, m, n, &alpha, A->data, A->stride, &beta, B->data, B->stride);
}

matx_status_t matx_geadd_d_i8(const matx_dense_backend_t* blas,
                              matx_double alpha,
                              const matx_dense_d_i8_t A,
                              matx_double beta,
                              matx_dense_d_i8_t B)
{
    if (!A || !B || !A->data || !B->data)
        return MATX_ERR_INVALID_ARG;
    if (A->nrows != B->nrows || A->ncols != B->ncols)
        return MATX_ERR_INVALID_ARG;
    if (A->layout != B->layout)
        return MATX_ERR_INVALID_ARG;

    const size_t m = A->nrows;
    const size_t n = A->ncols;
    return blas->vt.dgeadd(A->layout, m, n, alpha, A->data, A->stride, beta, B->data, B->stride);
}

// ---- Level 2 wrappers ----

matx_status_t matx_ger_d_i8(const matx_dense_backend_t* blas,
                            matx_double alpha,
                            const matx_vec_d_i8_t x,
                            const matx_vec_d_i8_t y,
                            matx_dense_d_i8_t A)
{
    if (!blas || !x || !y || !A || !x->data || !y->data || !A->data)
        return MATX_ERR_INVALID_ARG;
    if (A->nrows != x->n || A->ncols != y->n)
        return MATX_ERR_INVALID_ARG;
    if (!blas->vt.dger)
        return MATX_ERR_NOT_SUPPORTED;
    return blas->vt.dger(A->layout,
                         A->nrows,
                         A->ncols,
                         alpha,
                         x->data,
                         x->stride,
                         y->data,
                         y->stride,
                         A->data,
                         A->stride);
}

matx_status_t matx_geru_z_i8(const matx_dense_backend_t* blas,
                             matx_complex_d_i8_t alpha,
                             const matx_vec_z_i8_t x,
                             const matx_vec_z_i8_t y,
                             matx_dense_z_i8_t A)
{
    if (!blas || !x || !y || !A || !x->data || !y->data || !A->data)
        return MATX_ERR_INVALID_ARG;
    if (A->nrows != x->n || A->ncols != y->n)
        return MATX_ERR_INVALID_ARG;
    if (!blas->vt.zgeru)
        return MATX_ERR_NOT_SUPPORTED;
    return blas->vt.zgeru(A->layout,
                          A->nrows,
                          A->ncols,
                          &alpha,
                          x->data,
                          x->stride,
                          y->data,
                          y->stride,
                          A->data,
                          A->stride);
}

matx_status_t matx_gerc_z_i8(const matx_dense_backend_t* blas,
                             matx_complex_d_i8_t alpha,
                             const matx_vec_z_i8_t x,
                             const matx_vec_z_i8_t y,
                             matx_dense_z_i8_t A)
{
    if (!blas || !x || !y || !A || !x->data || !y->data || !A->data)
        return MATX_ERR_INVALID_ARG;
    if (A->nrows != x->n || A->ncols != y->n)
        return MATX_ERR_INVALID_ARG;
    if (!blas->vt.zgerc)
        return MATX_ERR_NOT_SUPPORTED;
    return blas->vt.zgerc(A->layout,
                          A->nrows,
                          A->ncols,
                          &alpha,
                          x->data,
                          x->stride,
                          y->data,
                          y->stride,
                          A->data,
                          A->stride);
}

matx_status_t matx_trsv_d_i8(const matx_dense_backend_t* blas,
                             int uplo,
                             int trans,
                             int diag,
                             const matx_dense_d_i8_t A,
                             matx_vec_d_i8_t x)
{
    if (!blas || !A || !x || !A->data || !x->data)
        return MATX_ERR_INVALID_ARG;
    if (A->nrows != A->ncols || A->ncols != x->n)
        return MATX_ERR_INVALID_ARG;
    if (!blas->vt.dtrsv)
        return MATX_ERR_NOT_SUPPORTED;
    return blas->vt
        .dtrsv(A->layout, uplo, trans, diag, A->nrows, A->data, A->stride, x->data, x->stride);
}

matx_status_t matx_trsv_z_i8(const matx_dense_backend_t* blas,
                             int uplo,
                             int trans,
                             int diag,
                             const matx_dense_z_i8_t A,
                             matx_vec_z_i8_t x)
{
    if (!blas || !A || !x || !A->data || !x->data)
        return MATX_ERR_INVALID_ARG;
    if (A->nrows != A->ncols || A->ncols != x->n)
        return MATX_ERR_INVALID_ARG;
    if (!blas->vt.ztrsv)
        return MATX_ERR_NOT_SUPPORTED;
    return blas->vt
        .ztrsv(A->layout, uplo, trans, diag, A->nrows, A->data, A->stride, x->data, x->stride);
}

// ---- Level 3 wrappers ----

matx_status_t matx_trsm_d_i8(const matx_dense_backend_t* blas,
                             int side,
                             int uplo,
                             int trans,
                             int diag,
                             matx_double alpha,
                             const matx_dense_d_i8_t A,
                             matx_dense_d_i8_t B)
{
    if (!blas || !A || !B || !A->data || !B->data)
        return MATX_ERR_INVALID_ARG;
    if (A->layout != B->layout)
        return MATX_ERR_INVALID_ARG;
    if (!blas->vt.dtrsm)
        return MATX_ERR_NOT_SUPPORTED;
    return blas->vt.dtrsm(A->layout,
                          side,
                          uplo,
                          trans,
                          diag,
                          B->nrows,
                          B->ncols,
                          alpha,
                          A->data,
                          A->stride,
                          B->data,
                          B->stride);
}

matx_status_t matx_trsm_z_i8(const matx_dense_backend_t* blas,
                             int side,
                             int uplo,
                             int trans,
                             int diag,
                             matx_complex_d_i8_t alpha,
                             const matx_dense_z_i8_t A,
                             matx_dense_z_i8_t B)
{
    if (!blas || !A || !B || !A->data || !B->data)
        return MATX_ERR_INVALID_ARG;
    if (A->layout != B->layout)
        return MATX_ERR_INVALID_ARG;
    if (!blas->vt.ztrsm)
        return MATX_ERR_NOT_SUPPORTED;
    return blas->vt.ztrsm(A->layout,
                          side,
                          uplo,
                          trans,
                          diag,
                          B->nrows,
                          B->ncols,
                          &alpha,
                          A->data,
                          A->stride,
                          B->data,
                          B->stride);
}

matx_status_t matx_syrk_d_i8(const matx_dense_backend_t* blas,
                             int uplo,
                             int trans,
                             matx_double alpha,
                             const matx_dense_d_i8_t A,
                             matx_double beta,
                             matx_dense_d_i8_t C)
{
    if (!blas || !A || !C || !A->data || !C->data)
        return MATX_ERR_INVALID_ARG;
    if (A->layout != C->layout)
        return MATX_ERR_INVALID_ARG;
    if (C->nrows != C->ncols)
        return MATX_ERR_INVALID_ARG;
    if (!blas->vt.dsyrk)
        return MATX_ERR_NOT_SUPPORTED;
    const matx_int64_t n = C->nrows;
    const matx_int64_t k = trans ? A->nrows : A->ncols;
    return blas->vt
        .dsyrk(A->layout, uplo, trans, n, k, alpha, A->data, A->stride, beta, C->data, C->stride);
}

matx_status_t matx_herk_z_i8(const matx_dense_backend_t* blas,
                             int uplo,
                             int trans,
                             matx_double alpha,
                             const matx_dense_z_i8_t A,
                             matx_double beta,
                             matx_dense_z_i8_t C)
{
    if (!blas || !A || !C || !A->data || !C->data)
        return MATX_ERR_INVALID_ARG;
    if (A->layout != C->layout)
        return MATX_ERR_INVALID_ARG;
    if (C->nrows != C->ncols)
        return MATX_ERR_INVALID_ARG;
    if (!blas->vt.zherk)
        return MATX_ERR_NOT_SUPPORTED;
    const matx_int64_t n = C->nrows;
    const matx_int64_t k = trans ? A->nrows : A->ncols;
    return blas->vt
        .zherk(A->layout, uplo, trans, n, k, alpha, A->data, A->stride, beta, C->data, C->stride);
}

matx_status_t matx_syr2k_d_i8(const matx_dense_backend_t* blas,
                              int uplo,
                              int trans,
                              matx_double alpha,
                              const matx_dense_d_i8_t A,
                              const matx_dense_d_i8_t B,
                              matx_double beta,
                              matx_dense_d_i8_t C)
{
    if (!blas || !A || !B || !C || !A->data || !B->data || !C->data)
        return MATX_ERR_INVALID_ARG;
    if (A->layout != C->layout || B->layout != C->layout)
        return MATX_ERR_INVALID_ARG;
    if (C->nrows != C->ncols)
        return MATX_ERR_INVALID_ARG;
    if (!blas->vt.dsyr2k)
        return MATX_ERR_NOT_SUPPORTED;
    const matx_int64_t n = C->nrows;
    const matx_int64_t k = trans ? A->nrows : A->ncols;
    return blas->vt.dsyr2k(A->layout,
                           uplo,
                           trans,
                           n,
                           k,
                           alpha,
                           A->data,
                           A->stride,
                           B->data,
                           B->stride,
                           beta,
                           C->data,
                           C->stride);
}

matx_status_t matx_her2k_z_i8(const matx_dense_backend_t* blas,
                              int uplo,
                              int trans,
                              matx_complex_d_i8_t alpha,
                              const matx_dense_z_i8_t A,
                              const matx_dense_z_i8_t B,
                              matx_double beta,
                              matx_dense_z_i8_t C)
{
    if (!blas || !A || !B || !C || !A->data || !B->data || !C->data)
        return MATX_ERR_INVALID_ARG;
    if (A->layout != C->layout || B->layout != C->layout)
        return MATX_ERR_INVALID_ARG;
    if (C->nrows != C->ncols)
        return MATX_ERR_INVALID_ARG;
    if (!blas->vt.zher2k)
        return MATX_ERR_NOT_SUPPORTED;
    const matx_int64_t n = C->nrows;
    const matx_int64_t k = trans ? A->nrows : A->ncols;
    return blas->vt.zher2k(A->layout,
                           uplo,
                           trans,
                           n,
                           k,
                           &alpha,
                           A->data,
                           A->stride,
                           B->data,
                           B->stride,
                           beta,
                           C->data,
                           C->stride);
}

// ---- Transpose wrappers ----

matx_status_t matx_transpose_d_i8(const matx_dense_backend_t* blas,
                                  const matx_dense_d_i8_t A,
                                  matx_dense_d_i8_t out)
{
    if (!blas || !A || !out || !A->data || !out->data)
        return MATX_ERR_INVALID_ARG;
    if (out->nrows != A->ncols || out->ncols != A->nrows)
        return MATX_ERR_INVALID_ARG;
    if (A->layout != out->layout)
        return MATX_ERR_INVALID_ARG;
    if (!blas->vt.transpose_d_i8)
        return MATX_ERR_NOT_SUPPORTED;
    return blas->vt
        .transpose_d_i8(A->layout, A->nrows, A->ncols, A->data, A->stride, out->data, out->stride);
}

matx_status_t matx_transpose_z_i8(const matx_dense_backend_t* blas,
                                  const matx_dense_z_i8_t A,
                                  matx_dense_z_i8_t out)
{
    if (!blas || !A || !out || !A->data || !out->data)
        return MATX_ERR_INVALID_ARG;
    if (out->nrows != A->ncols || out->ncols != A->nrows)
        return MATX_ERR_INVALID_ARG;
    if (A->layout != out->layout)
        return MATX_ERR_INVALID_ARG;
    if (!blas->vt.transpose_z_i8)
        return MATX_ERR_NOT_SUPPORTED;
    return blas->vt
        .transpose_z_i8(A->layout, A->nrows, A->ncols, A->data, A->stride, out->data, out->stride);
}

matx_status_t matx_conj_transpose_z_i8(const matx_dense_backend_t* blas,
                                       const matx_dense_z_i8_t A,
                                       matx_dense_z_i8_t out)
{
    if (!blas || !A || !out || !A->data || !out->data)
        return MATX_ERR_INVALID_ARG;
    if (out->nrows != A->ncols || out->ncols != A->nrows)
        return MATX_ERR_INVALID_ARG;
    if (A->layout != out->layout)
        return MATX_ERR_INVALID_ARG;
    if (!blas->vt.conj_transpose_z_i8)
        return MATX_ERR_NOT_SUPPORTED;
    return blas->vt.conj_transpose_z_i8(A->layout,
                                        A->nrows,
                                        A->ncols,
                                        A->data,
                                        A->stride,
                                        out->data,
                                        out->stride);
}

// ---- Norm wrappers ----

matx_status_t matx_mat_norm1_d_i8(const matx_dense_backend_t* blas,
                                  const matx_dense_d_i8_t A,
                                  matx_double* out)
{
    if (!blas || !A || !A->data || !out)
        return MATX_ERR_INVALID_ARG;
    if (!blas->vt.norm1_d_i8)
        return MATX_ERR_NOT_SUPPORTED;
    return blas->vt.norm1_d_i8(A->layout, A->nrows, A->ncols, A->data, A->stride, out);
}

matx_status_t matx_mat_norminf_d_i8(const matx_dense_backend_t* blas,
                                    const matx_dense_d_i8_t A,
                                    matx_double* out)
{
    if (!blas || !A || !A->data || !out)
        return MATX_ERR_INVALID_ARG;
    if (!blas->vt.norminf_d_i8)
        return MATX_ERR_NOT_SUPPORTED;
    return blas->vt.norminf_d_i8(A->layout, A->nrows, A->ncols, A->data, A->stride, out);
}

matx_status_t matx_mat_normfro_d_i8(const matx_dense_backend_t* blas,
                                    const matx_dense_d_i8_t A,
                                    matx_double* out)
{
    if (!blas || !A || !A->data || !out)
        return MATX_ERR_INVALID_ARG;
    if (!blas->vt.normfro_d_i8)
        return MATX_ERR_NOT_SUPPORTED;
    return blas->vt.normfro_d_i8(A->layout, A->nrows, A->ncols, A->data, A->stride, out);
}

matx_status_t matx_mat_norm1_z_i8(const matx_dense_backend_t* blas,
                                  const matx_dense_z_i8_t A,
                                  matx_double* out)
{
    if (!blas || !A || !A->data || !out)
        return MATX_ERR_INVALID_ARG;
    if (!blas->vt.norm1_z_i8)
        return MATX_ERR_NOT_SUPPORTED;
    return blas->vt.norm1_z_i8(A->layout, A->nrows, A->ncols, A->data, A->stride, out);
}

matx_status_t matx_mat_norminf_z_i8(const matx_dense_backend_t* blas,
                                    const matx_dense_z_i8_t A,
                                    matx_double* out)
{
    if (!blas || !A || !A->data || !out)
        return MATX_ERR_INVALID_ARG;
    if (!blas->vt.norminf_z_i8)
        return MATX_ERR_NOT_SUPPORTED;
    return blas->vt.norminf_z_i8(A->layout, A->nrows, A->ncols, A->data, A->stride, out);
}

matx_status_t matx_mat_normfro_z_i8(const matx_dense_backend_t* blas,
                                    const matx_dense_z_i8_t A,
                                    matx_double* out)
{
    if (!blas || !A || !A->data || !out)
        return MATX_ERR_INVALID_ARG;
    if (!blas->vt.normfro_z_i8)
        return MATX_ERR_NOT_SUPPORTED;
    return blas->vt.normfro_z_i8(A->layout, A->nrows, A->ncols, A->data, A->stride, out);
}

// ---- Hadamard (element-wise) wrappers ----

matx_status_t matx_hadamard_d_i8(const matx_dense_backend_t* blas,
                                 const matx_dense_d_i8_t A,
                                 const matx_dense_d_i8_t B,
                                 matx_dense_d_i8_t C)
{
    if (!blas || !A || !B || !C || !A->data || !B->data || !C->data)
        return MATX_ERR_INVALID_ARG;
    if (A->nrows != B->nrows || A->ncols != B->ncols || A->nrows != C->nrows || A->ncols != C->ncols)
        return MATX_ERR_INVALID_ARG;
    if (A->layout != B->layout || A->layout != C->layout)
        return MATX_ERR_INVALID_ARG;
    if (!blas->vt.hadamard_d_i8)
        return MATX_ERR_NOT_SUPPORTED;
    return blas->vt.hadamard_d_i8(A->layout,
                                  A->nrows,
                                  A->ncols,
                                  A->data,
                                  A->stride,
                                  B->data,
                                  B->stride,
                                  C->data,
                                  C->stride);
}

matx_status_t matx_hadamard_z_i8(const matx_dense_backend_t* blas,
                                 const matx_dense_z_i8_t A,
                                 const matx_dense_z_i8_t B,
                                 matx_dense_z_i8_t C)
{
    if (!blas || !A || !B || !C || !A->data || !B->data || !C->data)
        return MATX_ERR_INVALID_ARG;
    if (A->nrows != B->nrows || A->ncols != B->ncols || A->nrows != C->nrows || A->ncols != C->ncols)
        return MATX_ERR_INVALID_ARG;
    if (A->layout != B->layout || A->layout != C->layout)
        return MATX_ERR_INVALID_ARG;
    if (!blas->vt.hadamard_z_i8)
        return MATX_ERR_NOT_SUPPORTED;
    return blas->vt.hadamard_z_i8(A->layout,
                                  A->nrows,
                                  A->ncols,
                                  A->data,
                                  A->stride,
                                  B->data,
                                  B->stride,
                                  C->data,
                                  C->stride);
}