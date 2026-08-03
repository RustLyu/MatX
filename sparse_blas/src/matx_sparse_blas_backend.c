#include "matx/matx_log.h"
#include "matx/matx_sparse_compute.h"
#include "matx/matx_types_internal.h"

// Forward decls
matx_sparse_backend_t matx_sparse_make_reference_grb(void);
matx_sparse_backend_t matx_sparse_make_reference_aocl(void);

const char* matx_sparse_backend_name(matx_sparse_backend_kind_t k)
{
    switch (k) {
    case MATX_SPARSE_BACKEND_REFERENCE:
        return "REFERENCE";
    case MATX_SPARSE_BACKEND_GRAPHBLAS:
        return "GRAPHBLAS";
    default:
        return "UNKNOWN";
    }
}

static matx_sparse_backend_t choose_default_backend(void)
{
    return matx_sparse_make_reference_grb();
}

matx_sparse_backend_t matx_sparse_default(void)
{
    return choose_default_backend();
}

MATX_API matx_sparse_backend_t matx_sparse_by_type(matx_sparse_backend_kind_t k)
{
    switch (k) {
    case MATX_SPARSE_BACKEND_AOCL_CPARSE:
        return matx_sparse_make_reference_aocl();
    default:
        return matx_sparse_make_reference_grb();
    }
}

matx_status_t matx_spmv_coo_z_i8(const matx_sparse_backend_t* backend,
                                 matx_complex_d_t alpha,
                                 matx_coo_z_i8_t A,
                                 matx_vec_z_i8_t x,
                                 matx_complex_d_t beta,
                                 matx_vec_z_i8_t y)
{
    if (!backend || !A || !x || !y) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!backend->vt.spmv_z_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    if (!A->values || !x->data || !y->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    return backend->vt.spmv_z_i8(alpha, A, x, beta, y);
}

matx_status_t matx_spmm_coo_z_i8(const matx_sparse_backend_t* backend,
                                 matx_complex_d_t alpha,
                                 matx_coo_z_i8_t A,
                                 matx_dense_z_i8_t B,
                                 matx_complex_d_t beta,
                                 matx_dense_z_i8_t C)
{
    if (!backend || !A || !B || !C) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!backend->vt.spmm_z_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    if (!A->values || !B->data || !C->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    return backend->vt.spmm_z_i8(alpha, A, B, beta, C);
}

matx_status_t matx_spmv_coo_d_i8(const matx_sparse_backend_t* backend,
                                 matx_double alpha,
                                 matx_coo_d_i8_t A,
                                 matx_vec_d_i8_t x,
                                 matx_double beta,
                                 matx_vec_d_i8_t y)
{
    if (!backend || !A || !x || !y) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!backend->vt.spmv_d_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    if (!A->values || !x->data || !y->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    return backend->vt.spmv_d_i8(alpha, A, x, beta, y);
}

matx_status_t matx_spmm_coo_d_i8(const matx_sparse_backend_t* backend,
                                 matx_double alpha,
                                 matx_coo_d_i8_t A,
                                 matx_dense_d_i8_t B,
                                 matx_double beta,
                                 matx_dense_d_i8_t C)
{
    if (!backend || !A || !B || !C) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!backend->vt.spmm_d_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    if (!A->values || !B->data || !C->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    return backend->vt.spmm_d_i8(alpha, A, B, beta, C);
}

matx_status_t matx_dsp2md_coo_d_i8(const matx_sparse_backend_t* backend,
                                   matx_double alpha,
                                   matx_coo_d_i8_t A,
                                   matx_coo_d_i8_t B,
                                   matx_double beta,
                                   matx_dense_d_i8_t C)
{
    if (!backend || !A || !B || !C) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!backend->vt.dsp2md_d_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    if (!A->values || !B->values || !C->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    return backend->vt.dsp2md_d_i8(alpha, A, B, beta, C);
}

matx_status_t matx_zsp2md_coo_z_i8(const matx_sparse_backend_t* backend,
                                   matx_complex_d_t alpha,
                                   matx_coo_z_i8_t A,
                                   matx_coo_z_i8_t B,
                                   matx_complex_d_t beta,
                                   matx_dense_z_i8_t C)
{
    if (!backend || !A || !B || !C) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!backend->vt.zsp2md_z_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    if (!A->values || !B->values || !C->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    return backend->vt.zsp2md_z_i8(alpha, A, B, beta, C);
}

matx_status_t matx_transpose_coo_d_i8(const matx_sparse_backend_t* backend,
                                      matx_coo_d_i8_t A,
                                      matx_coo_d_i8_t out)
{
    if (!backend || !A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!backend->vt.transpose_d_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    if (!A->values || !out->values) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    return backend->vt.transpose_d_i8(A, out);
}

matx_status_t matx_transpose_coo_z_i8(const matx_sparse_backend_t* backend,
                                      matx_coo_z_i8_t A,
                                      matx_coo_z_i8_t out)
{
    if (!backend || !A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!backend->vt.transpose_z_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    if (!A->values || !out->values) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    return backend->vt.transpose_z_i8(A, out);
}

matx_status_t matx_conj_coo_z_i8(const matx_sparse_backend_t* backend,
                                 matx_coo_z_i8_t A,
                                 matx_coo_z_i8_t out)
{
    if (!backend || !A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!backend->vt.conj_trans_z_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    if (!A->values || !out->values) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    return backend->vt.conj_trans_z_i8(A, out);
}

matx_status_t matx_finalize(const matx_sparse_backend_t* backend)
{
    if (!backend) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!backend->vt.finalize) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return backend->vt.finalize();
}

matx_status_t matx_norm1_mat_coo_d_i8(const matx_sparse_backend_t* backend,
                                      matx_coo_d_i8_t A,
                                      matx_double* out)
{
    if (!backend || !A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!backend->vt.norm1_mat_d_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return backend->vt.norm1_mat_d_i8(A, out);
}

matx_status_t matx_norminf_mat_coo_d_i8(const matx_sparse_backend_t* backend,
                                        matx_coo_d_i8_t A,
                                        matx_double* out)
{
    if (!backend || !A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!backend->vt.norminf_mat_d_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return backend->vt.norminf_mat_d_i8(A, out);
}

matx_status_t matx_normfro_mat_coo_d_i8(const matx_sparse_backend_t* backend,
                                        matx_coo_d_i8_t A,
                                        matx_double* out)
{
    if (!backend || !A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!backend->vt.normfro_mat_d_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return backend->vt.normfro_mat_d_i8(A, out);
}

matx_status_t matx_norm1_mat_coo_z_i8(const matx_sparse_backend_t* backend,
                                      matx_coo_z_i8_t A,
                                      matx_double* out)
{
    if (!backend || !A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!backend->vt.norm1_mat_z_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return backend->vt.norm1_mat_z_i8(A, out);
}

matx_status_t matx_norminf_mat_coo_z_i8(const matx_sparse_backend_t* backend,
                                        matx_coo_z_i8_t A,
                                        matx_double* out)
{
    if (!backend || !A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!backend->vt.norminf_mat_z_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return backend->vt.norminf_mat_z_i8(A, out);
}

matx_status_t matx_normfro_mat_coo_z_i8(const matx_sparse_backend_t* backend,
                                        matx_coo_z_i8_t A,
                                        matx_double* out)
{
    if (!backend || !A || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!backend->vt.normfro_mat_z_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return backend->vt.normfro_mat_z_i8(A, out);
}

matx_status_t matx_spadd_coo_d_i8(const matx_sparse_backend_t* backend,
                                  matx_double alpha,
                                  matx_coo_d_i8_t A,
                                  matx_double beta,
                                  matx_coo_d_i8_t B,
                                  matx_coo_d_i8_t out)
{
    if (!backend || !A || !B || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!backend->vt.spadd_d_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return backend->vt.spadd_d_i8(alpha, A, beta, B, out);
}

matx_status_t matx_spadd_coo_z_i8(const matx_sparse_backend_t* backend,
                                  matx_complex_d_t alpha,
                                  matx_coo_z_i8_t A,
                                  matx_complex_d_t beta,
                                  matx_coo_z_i8_t B,
                                  matx_coo_z_i8_t out)
{
    if (!backend || !A || !B || !out) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!backend->vt.spadd_z_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return backend->vt.spadd_z_i8(alpha, A, beta, B, out);
}

// ---- Non-zero count per row/column ----

matx_status_t matx_spnnz_rows_coo_d_i8(const matx_sparse_backend_t* backend,
                                       matx_coo_d_i8_t A,
                                       matx_vec_d_i8_t out)
{
    if (!backend || !A || !out || !out->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!backend->vt.spnnz_rows_d_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return backend->vt.spnnz_rows_d_i8(A, out);
}

matx_status_t matx_spnnz_cols_coo_d_i8(const matx_sparse_backend_t* backend,
                                       matx_coo_d_i8_t A,
                                       matx_vec_d_i8_t out)
{
    if (!backend || !A || !out || !out->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!backend->vt.spnnz_cols_d_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return backend->vt.spnnz_cols_d_i8(A, out);
}

matx_status_t matx_spnnz_rows_coo_z_i8(const matx_sparse_backend_t* backend,
                                       matx_coo_z_i8_t A,
                                       matx_vec_z_i8_t out)
{
    if (!backend || !A || !out || !out->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!backend->vt.spnnz_rows_z_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return backend->vt.spnnz_rows_z_i8(A, out);
}

matx_status_t matx_spnnz_cols_coo_z_i8(const matx_sparse_backend_t* backend,
                                       matx_coo_z_i8_t A,
                                       matx_vec_z_i8_t out)
{
    if (!backend || !A || !out || !out->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!backend->vt.spnnz_cols_z_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return backend->vt.spnnz_cols_z_i8(A, out);
}

// ---- Row / column sums ----

matx_status_t matx_sprowsums_coo_d_i8(const matx_sparse_backend_t* backend,
                                      matx_coo_d_i8_t A,
                                      matx_vec_d_i8_t out)
{
    if (!backend || !A || !out || !out->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!backend->vt.sprowsums_d_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return backend->vt.sprowsums_d_i8(A, out);
}

matx_status_t matx_spcolsums_coo_d_i8(const matx_sparse_backend_t* backend,
                                      matx_coo_d_i8_t A,
                                      matx_vec_d_i8_t out)
{
    if (!backend || !A || !out || !out->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!backend->vt.spcolsums_d_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return backend->vt.spcolsums_d_i8(A, out);
}

matx_status_t matx_sprowsums_coo_z_i8(const matx_sparse_backend_t* backend,
                                      matx_coo_z_i8_t A,
                                      matx_vec_z_i8_t out)
{
    if (!backend || !A || !out || !out->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!backend->vt.sprowsums_z_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return backend->vt.sprowsums_z_i8(A, out);
}

matx_status_t matx_spcolsums_coo_z_i8(const matx_sparse_backend_t* backend,
                                      matx_coo_z_i8_t A,
                                      matx_vec_z_i8_t out)
{
    if (!backend || !A || !out || !out->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!backend->vt.spcolsums_z_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return backend->vt.spcolsums_z_i8(A, out);
}

// ---- Diagonal extraction ----

matx_status_t matx_spdiag_coo_d_i8(const matx_sparse_backend_t* backend,
                                   matx_coo_d_i8_t A,
                                   matx_int64_t offset,
                                   matx_vec_d_i8_t out)
{
    if (!backend || !A || !out || !out->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!backend->vt.spdiag_d_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return backend->vt.spdiag_d_i8(A, offset, out);
}

matx_status_t matx_spdiag_coo_z_i8(const matx_sparse_backend_t* backend,
                                   matx_coo_z_i8_t A,
                                   matx_int64_t offset,
                                   matx_vec_z_i8_t out)
{
    if (!backend || !A || !out || !out->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!backend->vt.spdiag_z_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return backend->vt.spdiag_z_i8(A, offset, out);
}

// ---- In-place row/column scaling ----

matx_status_t matx_scale_rows_coo_d_i8(const matx_sparse_backend_t* backend,
                                       matx_coo_d_i8_t A,
                                       const matx_vec_d_i8_t s)
{
    if (!backend || !A || !s || !A->values || !s->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!backend->vt.scale_rows_d_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return backend->vt.scale_rows_d_i8(A, s);
}

matx_status_t matx_scale_cols_coo_d_i8(const matx_sparse_backend_t* backend,
                                       matx_coo_d_i8_t A,
                                       const matx_vec_d_i8_t s)
{
    if (!backend || !A || !s || !A->values || !s->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!backend->vt.scale_cols_d_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return backend->vt.scale_cols_d_i8(A, s);
}

matx_status_t matx_scale_rows_coo_z_i8(const matx_sparse_backend_t* backend,
                                       matx_coo_z_i8_t A,
                                       const matx_vec_z_i8_t s)
{
    if (!backend || !A || !s || !A->values || !s->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!backend->vt.scale_rows_z_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return backend->vt.scale_rows_z_i8(A, s);
}

matx_status_t matx_scale_cols_coo_z_i8(const matx_sparse_backend_t* backend,
                                       matx_coo_z_i8_t A,
                                       const matx_vec_z_i8_t s)
{
    if (!backend || !A || !s || !A->values || !s->data) {
        MATX_ERROR("%s: invalid argument", __func__);
        return MATX_ERR_INVALID_ARG;
    }
    if (!backend->vt.scale_cols_z_i8) {
        MATX_ERROR("%s: operation not supported", __func__);
        return MATX_ERR_NOT_SUPPORTED;
    }
    return backend->vt.scale_cols_z_i8(A, s);
}
