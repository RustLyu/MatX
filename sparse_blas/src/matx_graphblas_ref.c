#include "matx/matx_func.h"
#include "matx/matx_log.h"
#include "matx/matx_sparse_compute.h"
#include "matx/matx_tm.h"
#include "matx/matx_types.h"
#include "matx/matx_types_internal.h"

#ifdef MATX_ENABLE_GRAPHBLAS
#include "GraphBLAS.h"
#endif

#include <limits.h>
#include <math.h>

//C(i,j)=k⨁​(A(i,k)⊗B(k,j))

static matx_status_t ref_spmv_z_i8_grb(matx_complex_d_t alpha,
                                       matx_coo_z_i8_t A,
                                       matx_vec_z_i8_t x,
                                       matx_complex_d_t beta,
                                       matx_vec_z_i8_t y)
{
#ifdef MATX_ENABLE_GRAPHBLAS
    if (!A || !x || !y) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }

    if (A->ncols != x->n || A->nrows != y->n) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }

    if (A->handle_grb.valid <= 0)
        coo_2_grb_z_i8(A);
    if (x->handle_grb.valid <= 0)
        vec_2_grb_z_i8(x);
    if (y->handle_grb.valid <= 0)
        vec_2_grb_z_i8(y);

    GxB_FC64_t a = {alpha.real, alpha.imag};
    GxB_FC64_t b = {beta.real, beta.imag};

    // gy = beta * gy
    GrB_Info info = GrB_apply((GrB_Vector) y->handle_grb.impl,
                              NULL,
                              NULL,
                              GxB_TIMES_FC64,
                              (GrB_Vector) y->handle_grb.impl,
                              b,
                              NULL);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("GrB_apply beta*gy error: %d", info);
        return MATX_ERR_INTERNAL;
    }
    // gy += alpha * A * x  (accumulate into gy)
    GrB_Vector temp;
    GrB_Vector_new(&temp, GxB_FC64, y->n);
    info = GrB_mxv(temp,
                   NULL,
                   NULL,
                   GxB_PLUS_TIMES_FC64,
                   (GrB_Matrix) A->handle_grb.impl,
                   (GrB_Vector) x->handle_grb.impl,
                   NULL);
    if (info != GrB_SUCCESS) {
        GrB_Vector_free(&temp);
        MATX_ERROR("GrB_mxv error: %d", info);
        return MATX_ERR_INTERNAL;
    }
    info = GrB_apply(temp, NULL, NULL, GxB_TIMES_FC64, temp, a, NULL);
    if (info != GrB_SUCCESS) {
        GrB_Vector_free(&temp);
        MATX_ERROR("GrB_apply alpha*temp error: %d", info);
        return MATX_ERR_INTERNAL;
    }
    info = GrB_eWiseAdd((GrB_Vector) y->handle_grb.impl,
                        NULL,
                        NULL,
                        GxB_PLUS_FC64,
                        (GrB_Vector) y->handle_grb.impl,
                        temp,
                        NULL);
    GrB_Vector_free(&temp);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("GrB_eWiseAdd error: %d", info);
        return MATX_ERR_INTERNAL;
    }
    grb_2_vec_z_i8(y);
#endif
    return MATX_OK;
}

static matx_status_t ref_spmm_z_i8_grb(matx_complex_d_t alpha,
                                       matx_coo_z_i8_t A,
                                       matx_dense_z_i8_t B,
                                       matx_complex_d_t beta,
                                       matx_dense_z_i8_t C)
{
#ifdef MATX_ENABLE_GRAPHBLAS
    if (!A || !B || !C) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }

    if (A->ncols != B->nrows || A->nrows != C->nrows || B->ncols != C->ncols) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }

    /* build A */
    if (A->handle_grb.valid <= 0) {
        coo_2_grb_z_i8(A);
    }
    if (B->handle_grb.valid <= 0) {
        dense_2_grb_z_i8(B);
    }
    if (C->handle_grb.valid <= 0) {
        dense_2_grb_z_i8(C);
    }

    /* C = alpha*A*B + beta*C */
    GxB_FC64_t a = {alpha.real, alpha.imag};
    GxB_FC64_t b = {beta.real, beta.imag};

    // 1. temp = alpha * A * B
    GrB_Matrix temp;
    GrB_Matrix_new(&temp, GxB_FC64, C->nrows, C->ncols);
    GrB_Info info = GrB_mxm(temp,
                            NULL,
                            NULL,
                            GxB_PLUS_TIMES_FC64,
                            (GrB_Matrix) A->handle_grb.impl,
                            (GrB_Matrix) B->handle_grb.impl,
                            NULL);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("GrB_mxm error: %d", info);
        return MATX_ERR_INTERNAL;
    }
    info = GrB_apply(temp, NULL, NULL, GxB_TIMES_FC64, temp, a, NULL);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("GrB_apply error: %d", info);
        return MATX_ERR_INTERNAL;
    }
    //2. gC = beta * gC
    info = GrB_apply((GrB_Matrix) C->handle_grb.impl,
                     NULL,
                     NULL,
                     GxB_TIMES_FC64,
                     (GrB_Matrix) C->handle_grb.impl,
                     b,
                     NULL);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("GrB_apply beta * gC error: %d", info);
        return MATX_ERR_INTERNAL;
    }
    //3. gC = temp + gC
    info = GrB_eWiseAdd((GrB_Matrix) C->handle_grb.impl,
                        NULL,
                        NULL,
                        GxB_PLUS_FC64,
                        temp,
                        (GrB_Matrix) C->handle_grb.impl,
                        NULL);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("GrB_eWiseAdd temp + gC error: %d", info);
        return MATX_ERR_INTERNAL;
    }
    GrB_Matrix_free(&temp);
    grb_2_dense_z_i8(C);
#endif
    return MATX_OK;
}

static matx_status_t ref_spmv_d_i8_grb(
    matx_double alpha, matx_coo_d_i8_t A, matx_vec_d_i8_t x, matx_double beta, matx_vec_d_i8_t y)
{
#ifdef MATX_ENABLE_GRAPHBLAS
    if (!A || !x || !y) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }

    if (A->ncols != x->n || A->nrows != y->n) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }

    if (A->handle_grb.valid <= 0)
        coo_2_grb_d_i8(A);
    if (x->handle_grb.valid <= 0)
        vec_2_grb_d_i8(x);
    if (y->handle_grb.valid <= 0)
        vec_2_grb_d_i8(y);

    // gy = beta * gy
    GrB_Info info = GrB_apply((GrB_Vector) y->handle_grb.impl,
                              NULL,
                              NULL,
                              GrB_TIMES_FP64,
                              (GrB_Vector) y->handle_grb.impl,
                              beta,
                              NULL);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("GrB_apply beta*gy error: %d", info);
        return MATX_ERR_INTERNAL;
    }
    // temp = A*x, then gy += alpha*temp
    matx_int64_t t0 = matx_tm_now(MATX_TM_MICROSECOND);
    GrB_Vector temp;
    GrB_Vector_new(&temp, GrB_FP64, y->n);
    info = GrB_mxv(temp,
                   NULL,
                   NULL,
                   GxB_PLUS_TIMES_FP64,
                   (GrB_Matrix) A->handle_grb.impl,
                   (GrB_Vector) x->handle_grb.impl,
                   NULL);
    if (info != GrB_SUCCESS) {
        GrB_Vector_free(&temp);
        MATX_ERROR("GrB_mxv A*x error: %d", info);
        return MATX_ERR_INTERNAL;
    }
    info = GrB_apply(temp, NULL, NULL, GrB_TIMES_FP64, temp, alpha, NULL);
    if (info != GrB_SUCCESS) {
        GrB_Vector_free(&temp);
        MATX_ERROR("GrB_apply alpha*temp error: %d", info);
        return MATX_ERR_INTERNAL;
    }
    info = GrB_eWiseAdd((GrB_Vector) y->handle_grb.impl,
                        NULL,
                        NULL,
                        GrB_PLUS_FP64,
                        (GrB_Vector) y->handle_grb.impl,
                        temp,
                        NULL);
    GrB_Vector_free(&temp);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("GrB_eWiseAdd temp + gy error: %d", info);
        return MATX_ERR_INTERNAL;
    }
    matx_int64_t t1 = matx_tm_now(MATX_TM_MICROSECOND);
    MATX_TRACE("GraphBLAS SpMV time: %ld micro.s", t1 - t0);
    grb_2_vec_d_i8(y);
#endif
    return MATX_OK;
}

static matx_status_t ref_spmm_d_i8_grb(matx_double alpha,
                                       matx_coo_d_i8_t A,
                                       matx_dense_d_i8_t B,
                                       matx_double beta,
                                       matx_dense_d_i8_t C)
{
#ifdef MATX_ENABLE_GRAPHBLAS
    if (!A || !B || !C) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }

    /* build A */
    if (A->handle_grb.valid <= 0) {
        coo_2_grb_d_i8(A);
    }
    if (B->handle_grb.valid <= 0) {
        dense_2_grb_d_i8(B);
    }
    if (C->handle_grb.valid <= 0) {
        dense_2_grb_d_i8(C);
    }

    /* C = alpha*A*B + beta*C */

    // 1. temp = alpha * A * B
    GrB_Matrix temp;
    GrB_Info info = GrB_Matrix_new(&temp, GrB_FP64, C->nrows, C->ncols);
    info = GrB_mxm(temp,
                   NULL,
                   NULL,
                   GxB_PLUS_TIMES_FP64,
                   (GrB_Matrix) A->handle_grb.impl,
                   (GrB_Matrix) B->handle_grb.impl,
                   NULL);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("GrB_mxm alpha * A * B error: %d", info);
        return MATX_ERR_INTERNAL;
    }
    info = GrB_apply(temp, NULL, NULL, GrB_TIMES_FP64, temp, alpha, NULL);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("GrB_apply  error: %d", info);
        return MATX_ERR_INTERNAL;
    }
    //2. gC = beta * gC
    info = GrB_apply((GrB_Matrix) C->handle_grb.impl,
                     NULL,
                     NULL,
                     GrB_TIMES_FP64,
                     (GrB_Matrix) C->handle_grb.impl,
                     beta,
                     NULL);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("GrB_apply beta * gC error: %d", info);
        return MATX_ERR_INTERNAL;
    }
    //3. gC = temp + gC
    info = GrB_eWiseAdd((GrB_Matrix) C->handle_grb.impl,
                        NULL,
                        NULL,
                        GrB_PLUS_FP64,
                        temp,
                        (GrB_Matrix) C->handle_grb.impl,
                        NULL);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("GrB_eWiseAdd temp + gC error: %d", info);
        return MATX_ERR_INTERNAL;
    }
    GrB_Matrix_free(&temp);
    grb_2_dense_d_i8(C);
#endif
    return MATX_OK;
}

static matx_status_t ref_dsp2md_d_i8_grb(
    matx_double alpha, matx_coo_d_i8_t A, matx_coo_d_i8_t B, matx_double beta, matx_dense_d_i8_t C)
{
#ifdef MATX_ENABLE_GRAPHBLAS
    if (!A || !B || !C) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }

    /* build A */
    if (A->handle_grb.valid <= 0) {
        coo_2_grb_d_i8(A);
    }
    if (B->handle_grb.valid <= 0) {
        coo_2_grb_d_i8(B);
    }
    if (C->handle_grb.valid <= 0) {
        dense_2_grb_d_i8(C);
    }

    /* C = alpha*A*B + beta*C */

    // 1. temp = alpha * A * B
    GrB_Matrix temp;
    GrB_Info info = GrB_Matrix_new(&temp, GrB_FP64, C->nrows, C->ncols);
    info = GrB_mxm(temp,
                   NULL,
                   NULL,
                   GxB_PLUS_TIMES_FP64,
                   (GrB_Matrix) A->handle_grb.impl,
                   (GrB_Matrix) B->handle_grb.impl,
                   NULL);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("GrB_mxm alpha * A * B error: %d", info);
        return MATX_ERR_INTERNAL;
    }
    info = GrB_apply(temp, NULL, NULL, GrB_TIMES_FP64, temp, alpha, NULL);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("GrB_apply alpha * A * B error: %d", info);
        return MATX_ERR_INTERNAL;
    }
    //2. gC = beta * gC
    info = GrB_apply((GrB_Matrix) C->handle_grb.impl,
                     NULL,
                     NULL,
                     GrB_TIMES_FP64,
                     (GrB_Matrix) C->handle_grb.impl,
                     beta,
                     NULL);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("GrB_apply beta * gC error: %d", info);
        return MATX_ERR_INTERNAL;
    }
    //3. gC = temp + gC
    info = GrB_eWiseAdd((GrB_Matrix) C->handle_grb.impl,
                        NULL,
                        NULL,
                        GrB_PLUS_FP64,
                        temp,
                        (GrB_Matrix) C->handle_grb.impl,
                        NULL);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("GrB_eWiseAdd temp + gC error: %d", info);
        return MATX_ERR_INTERNAL;
    }
    GrB_Matrix_free(&temp);
    grb_2_dense_d_i8(C);
#endif
    return MATX_OK;
}

static matx_status_t ref_zsp2md_z_i8_grb(matx_complex_d_t alpha,
                                         matx_coo_z_i8_t A,
                                         matx_coo_z_i8_t B,
                                         matx_complex_d_t beta,
                                         matx_dense_z_i8_t C)
{
#ifdef MATX_ENABLE_GRAPHBLAS
    if (!A || !B || !C) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }

    /* build A */
    if (A->handle_grb.valid <= 0) {
        coo_2_grb_z_i8(A);
    }
    if (B->handle_grb.valid <= 0) {
        coo_2_grb_z_i8(B);
    }
    if (C->handle_grb.valid <= 0) {
        dense_2_grb_z_i8(C);
    }

    /* C = alpha*A*B + beta*C */
    GxB_FC64_t a = {alpha.real, alpha.imag};
    GxB_FC64_t b = {beta.real, beta.imag};
    // 1. temp = alpha * A * B
    GrB_Matrix temp;
    GrB_Info info = GrB_Matrix_new(&temp, GxB_FC64, C->nrows, C->ncols);
    info = GrB_mxm(temp,
                   NULL,
                   NULL,
                   GxB_PLUS_TIMES_FC64,
                   (GrB_Matrix) A->handle_grb.impl,
                   (GrB_Matrix) B->handle_grb.impl,
                   NULL);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("GrB_mxm alpha * A * B error: %d", info);
        return MATX_ERR_INTERNAL;
    }
    info = GrB_apply(temp, NULL, NULL, GxB_TIMES_FC64, temp, a, NULL);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("GrB_apply error: %d", info);
        return MATX_ERR_INTERNAL;
    }
    //2. gC = beta * gC
    info = GrB_apply((GrB_Matrix) C->handle_grb.impl,
                     NULL,
                     NULL,
                     GxB_TIMES_FC64,
                     (GrB_Matrix) C->handle_grb.impl,
                     b,
                     NULL);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("GrB_apply beta * gC error: %d", info);
        return MATX_ERR_INTERNAL;
    }
    //3. gC = temp + gC
    info = GrB_eWiseAdd((GrB_Matrix) C->handle_grb.impl,
                        NULL,
                        NULL,
                        GxB_PLUS_FC64,
                        temp,
                        (GrB_Matrix) C->handle_grb.impl,
                        NULL);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("GrB_eWiseAdd temp + gC error: %d", info);
        return MATX_ERR_INTERNAL;
    }
    GrB_Matrix_free(&temp);
    grb_2_dense_z_i8(C);
#endif
    return MATX_OK;
}

static matx_status_t ref_transpose_d_i8_grb(matx_coo_d_i8_t A, matx_coo_d_i8_t out)
{
#ifdef MATX_ENABLE_GRAPHBLAS
    if (!A) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }

    /* build A */
    if (A->handle_grb.valid <= 0) {
        coo_2_grb_d_i8(A);
    }
    create_empty_grb_d_i8(out);
    GrB_Info info = GrB_transpose((GrB_Matrix) out->handle_grb.impl,
                                  NULL,
                                  NULL,
                                  (GrB_Matrix) A->handle_grb.impl,
                                  NULL);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("GrB_transpose error: %d", info);
        return MATX_ERR_INTERNAL;
    }
    grb_2_coo_d_i8(out);
#endif
    return MATX_OK;
}

static matx_status_t ref_transpose_z_i8_grb(matx_coo_z_i8_t A, matx_coo_z_i8_t out)
{
#ifdef MATX_ENABLE_GRAPHBLAS
    if (!A || !out) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }

    /* build A */
    if (A->handle_grb.valid <= 0) {
        coo_2_grb_z_i8(A);
    }
    if (!out->handle_grb.impl)
        create_empty_grb_z_i8(out);
    GrB_Info info = GrB_transpose((GrB_Matrix) (out->handle_grb.impl),
                                  NULL,
                                  NULL,
                                  (GrB_Matrix) (A->handle_grb.impl),
                                  NULL);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("GrB_transpose error: %d", info);
        return MATX_ERR_INTERNAL;
    }
    grb_2_coo_z_i8(out);
#endif
    return MATX_OK;
}

static matx_status_t ref_conj_trans_z_i8_grb(matx_coo_z_i8_t A, matx_coo_z_i8_t out)
{
#ifdef MATX_ENABLE_GRAPHBLAS
    if (!A) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }

    /* build A */
    if (A->handle_grb.valid <= 0) {
        coo_2_grb_z_i8(A);
    }
    if (!out->handle_grb.impl)
        create_empty_grb_z_i8(out);
    //1. transpose
    matx_status_t trans_status = ref_transpose_z_i8_grb(A, out);
    if (trans_status != MATX_OK) {
        MATX_ERROR("GrB_transpose error: %d", trans_status);
        return MATX_ERR_INTERNAL;
    }
    //2.  conj
    GrB_Info info = GrB_apply((GrB_Matrix) out->handle_grb.impl,
                              NULL,
                              NULL,
                              GxB_CONJ_FC64,
                              (GrB_Matrix) out->handle_grb.impl,
                              NULL);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("GrB_CONJ error: %d", info);
        return MATX_ERR_INTERNAL;
    }
    grb_2_coo_z_i8(out);
#endif
    return MATX_OK;
}

static matx_status_t ref_finalize_grb()
{
#ifdef MATX_ENABLE_GRAPHBLAS
    GrB_finalize();
#endif
    return MATX_OK;
}

// ---- Sparse matrix norms ----

static matx_status_t ref_norm1_mat_grb(matx_coo_d_i8_t A, matx_double* out)
{
#ifdef MATX_ENABLE_GRAPHBLAS
    if (!A || !out) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    if (A->handle_grb.valid <= 0)
        coo_2_grb_d_i8(A);

    GrB_Matrix tmp;
    GrB_Matrix_dup(&tmp, (GrB_Matrix) A->handle_grb.impl);
    GrB_apply(tmp, NULL, NULL, GrB_ABS_FP64, tmp, NULL);
    GrB_Vector col_sums;
    GrB_Vector_new(&col_sums, GrB_FP64, A->ncols);
    GrB_reduce(col_sums, NULL, NULL, GrB_PLUS_MONOID_FP64, tmp, NULL);
    GrB_Matrix_free(&tmp);
    GrB_reduce(out, NULL, GrB_MAX_MONOID_FP64, col_sums, NULL);
    GrB_Vector_free(&col_sums);
#endif
    return MATX_OK;
}

static matx_status_t ref_norminf_mat_grb(matx_coo_d_i8_t A, matx_double* out)
{
#ifdef MATX_ENABLE_GRAPHBLAS
    if (!A || !out) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    if (A->handle_grb.valid <= 0)
        coo_2_grb_d_i8(A);

    GrB_Matrix tmp;
    GrB_Matrix_dup(&tmp, (GrB_Matrix) A->handle_grb.impl);
    GrB_apply(tmp, NULL, NULL, GrB_ABS_FP64, tmp, NULL);
    GrB_Vector row_sums;
    GrB_Vector_new(&row_sums, GrB_FP64, A->nrows);
    GrB_reduce(row_sums, NULL, NULL, GrB_PLUS_MONOID_FP64, tmp, GrB_DESC_T0);
    GrB_Matrix_free(&tmp);
    GrB_reduce(out, NULL, GrB_MAX_MONOID_FP64, row_sums, NULL);
    GrB_Vector_free(&row_sums);
#endif
    return MATX_OK;
}

static matx_status_t ref_normfro_mat_grb(matx_coo_d_i8_t A, matx_double* out)
{
#ifdef MATX_ENABLE_GRAPHBLAS
    if (!A || !out) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    if (A->handle_grb.valid <= 0)
        coo_2_grb_d_i8(A);

    GrB_Matrix tmp;
    GrB_Matrix_dup(&tmp, (GrB_Matrix) A->handle_grb.impl);
    GrB_eWiseMult(tmp, NULL, NULL, GrB_TIMES_FP64, tmp, tmp, NULL);
    double sumsq = 0.0;
    GrB_reduce(&sumsq, NULL, GrB_PLUS_MONOID_FP64, tmp, NULL);
    GrB_Matrix_free(&tmp);
    *out = sqrt(sumsq);
#endif
    return MATX_OK;
}

// ---- Sparse matrix norms (c64) ----

static matx_status_t ref_norm1_mat_z_i8_grb(matx_coo_z_i8_t A, matx_double* out)
{
#ifdef MATX_ENABLE_GRAPHBLAS
    if (!A || !out) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    if (A->handle_grb.valid <= 0)
        coo_2_grb_z_i8(A);

    GrB_Matrix abs_mat;
    GrB_Matrix_new(&abs_mat, GrB_FP64, A->nrows, A->ncols);
    GrB_apply(abs_mat, NULL, NULL, GxB_ABS_FC64, (GrB_Matrix) A->handle_grb.impl, NULL);
    GrB_Vector col_sums;
    GrB_Vector_new(&col_sums, GrB_FP64, A->ncols);
    GrB_reduce(col_sums, NULL, NULL, GrB_PLUS_MONOID_FP64, abs_mat, NULL);
    GrB_Matrix_free(&abs_mat);
    GrB_reduce(out, NULL, GrB_MAX_MONOID_FP64, col_sums, NULL);
    GrB_Vector_free(&col_sums);
#endif
    return MATX_OK;
}

static matx_status_t ref_norminf_mat_z_i8_grb(matx_coo_z_i8_t A, matx_double* out)
{
#ifdef MATX_ENABLE_GRAPHBLAS
    if (!A || !out) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    if (A->handle_grb.valid <= 0)
        coo_2_grb_z_i8(A);

    GrB_Matrix abs_mat;
    GrB_Matrix_new(&abs_mat, GrB_FP64, A->nrows, A->ncols);
    GrB_apply(abs_mat, NULL, NULL, GxB_ABS_FC64, (GrB_Matrix) A->handle_grb.impl, NULL);
    GrB_Vector row_sums;
    GrB_Vector_new(&row_sums, GrB_FP64, A->nrows);
    GrB_reduce(row_sums, NULL, NULL, GrB_PLUS_MONOID_FP64, abs_mat, GrB_DESC_T0);
    GrB_Matrix_free(&abs_mat);
    GrB_reduce(out, NULL, GrB_MAX_MONOID_FP64, row_sums, NULL);
    GrB_Vector_free(&row_sums);
#endif
    return MATX_OK;
}

static matx_status_t ref_normfro_mat_z_i8_grb(matx_coo_z_i8_t A, matx_double* out)
{
#ifdef MATX_ENABLE_GRAPHBLAS
    if (!A || !out) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    if (A->handle_grb.valid <= 0)
        coo_2_grb_z_i8(A);

    GrB_Matrix abs_mat;
    GrB_Matrix_new(&abs_mat, GrB_FP64, A->nrows, A->ncols);
    GrB_apply(abs_mat, NULL, NULL, GxB_ABS_FC64, (GrB_Matrix) A->handle_grb.impl, NULL);
    GrB_eWiseMult(abs_mat, NULL, NULL, GrB_TIMES_FP64, abs_mat, abs_mat, NULL);
    double sumsq = 0.0;
    GrB_reduce(&sumsq, NULL, GrB_PLUS_MONOID_FP64, abs_mat, NULL);
    GrB_Matrix_free(&abs_mat);
    *out = sqrt(sumsq);
#endif
    return MATX_OK;
}

// ---- Sparse-sparse addition ----

static matx_status_t ref_spadd_d_i8_grb(
    matx_double alpha, matx_coo_d_i8_t A, matx_double beta, matx_coo_d_i8_t B, matx_coo_d_i8_t out)
{
#ifdef MATX_ENABLE_GRAPHBLAS
    if (!A || !B || !out) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    if (A->nrows != B->nrows || A->ncols != B->ncols) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    if (A->handle_grb.valid <= 0)
        coo_2_grb_d_i8(A);
    if (B->handle_grb.valid <= 0)
        coo_2_grb_d_i8(B);

    GrB_Matrix temp_a;
    GrB_Matrix_dup(&temp_a, (GrB_Matrix) A->handle_grb.impl);
    GrB_apply(temp_a, NULL, NULL, GrB_TIMES_FP64, temp_a, alpha, NULL);
    GrB_Matrix temp_b;
    GrB_Matrix_dup(&temp_b, (GrB_Matrix) B->handle_grb.impl);
    GrB_apply(temp_b, NULL, NULL, GrB_TIMES_FP64, temp_b, beta, NULL);

    create_empty_grb_d_i8(out);
    GrB_Info info = GrB_eWiseAdd((GrB_Matrix) out->handle_grb.impl,
                                 NULL,
                                 NULL,
                                 GrB_PLUS_FP64,
                                 temp_a,
                                 temp_b,
                                 NULL);
    GrB_Matrix_free(&temp_a);
    GrB_Matrix_free(&temp_b);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("GrB_eWiseAdd spadd_d_i8 error: %d", info);
        return MATX_ERR_INTERNAL;
    }
    grb_2_coo_d_i8(out);
#endif
    return MATX_OK;
}

static matx_status_t ref_spadd_z_i8_grb(matx_complex_d_t alpha,
                                        matx_coo_z_i8_t A,
                                        matx_complex_d_t beta,
                                        matx_coo_z_i8_t B,
                                        matx_coo_z_i8_t out)
{
#ifdef MATX_ENABLE_GRAPHBLAS
    if (!A || !B || !out) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    if (A->nrows != B->nrows || A->ncols != B->ncols) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    if (A->handle_grb.valid <= 0)
        coo_2_grb_z_i8(A);
    if (B->handle_grb.valid <= 0)
        coo_2_grb_z_i8(B);

    GxB_FC64_t a = {alpha.real, alpha.imag};
    GxB_FC64_t b = {beta.real, beta.imag};
    GrB_Matrix temp_a;
    GrB_Matrix_dup(&temp_a, (GrB_Matrix) A->handle_grb.impl);
    GrB_apply(temp_a, NULL, NULL, GxB_TIMES_FC64, temp_a, a, NULL);
    GrB_Matrix temp_b;
    GrB_Matrix_dup(&temp_b, (GrB_Matrix) B->handle_grb.impl);
    GrB_apply(temp_b, NULL, NULL, GxB_TIMES_FC64, temp_b, b, NULL);

    create_empty_grb_z_i8(out);
    GrB_Info info = GrB_eWiseAdd((GrB_Matrix) out->handle_grb.impl,
                                 NULL,
                                 NULL,
                                 GxB_PLUS_FC64,
                                 temp_a,
                                 temp_b,
                                 NULL);
    GrB_Matrix_free(&temp_a);
    GrB_Matrix_free(&temp_b);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("GrB_eWiseAdd spadd_z_i8 error: %d", info);
        return MATX_ERR_INTERNAL;
    }
    grb_2_coo_z_i8(out);
#endif
    return MATX_OK;
}

// ---- Non-zero count per row/column ----

static matx_status_t ref_spnnz_rows_d_i8_grb(matx_coo_d_i8_t A, matx_vec_d_i8_t out)
{
    if (!A || !out || !A->values) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    memset(out->data, 0, sizeof(matx_double) * (size_t) A->nrows);
    for (matx_int64_t i = 0; i < A->nnz; ++i)
        out->data[A->rows[i]] += 1.0;
    return MATX_OK;
}

static matx_status_t ref_spnnz_cols_d_i8_grb(matx_coo_d_i8_t A, matx_vec_d_i8_t out)
{
    if (!A || !out || !A->values) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    memset(out->data, 0, sizeof(matx_double) * (size_t) A->ncols);
    for (matx_int64_t i = 0; i < A->nnz; ++i)
        out->data[A->columns[i]] += 1.0;
    return MATX_OK;
}

static matx_status_t ref_spnnz_rows_z_i8_grb(matx_coo_z_i8_t A, matx_vec_z_i8_t out)
{
    if (!A || !out || !A->values) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    for (matx_int64_t i = 0; i < A->nrows; ++i) {
        out->data[i].real = 0.0;
        out->data[i].imag = 0.0;
    }
    for (matx_int64_t i = 0; i < A->nnz; ++i)
        out->data[A->rows[i]].real += 1.0;
    return MATX_OK;
}

static matx_status_t ref_spnnz_cols_z_i8_grb(matx_coo_z_i8_t A, matx_vec_z_i8_t out)
{
    if (!A || !out || !A->values) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    for (matx_int64_t j = 0; j < A->ncols; ++j) {
        out->data[j].real = 0.0;
        out->data[j].imag = 0.0;
    }
    for (matx_int64_t i = 0; i < A->nnz; ++i)
        out->data[A->columns[i]].real += 1.0;
    return MATX_OK;
}

// ---- Row / column sums ----

static matx_status_t ref_sprowsums_d_i8_grb(matx_coo_d_i8_t A, matx_vec_d_i8_t out)
{
    if (!A || !out || !A->values) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    memset(out->data, 0, sizeof(matx_double) * (size_t) A->nrows);
    for (matx_int64_t i = 0; i < A->nnz; ++i)
        out->data[A->rows[i]] += fabs(A->values[i]);
    return MATX_OK;
}

static matx_status_t ref_spcolsums_d_i8_grb(matx_coo_d_i8_t A, matx_vec_d_i8_t out)
{
    if (!A || !out || !A->values) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    memset(out->data, 0, sizeof(matx_double) * (size_t) A->ncols);
    for (matx_int64_t i = 0; i < A->nnz; ++i)
        out->data[A->columns[i]] += fabs(A->values[i]);
    return MATX_OK;
}

static matx_status_t ref_sprowsums_z_i8_grb(matx_coo_z_i8_t A, matx_vec_z_i8_t out)
{
    if (!A || !out || !A->values) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    for (matx_int64_t i = 0; i < A->nrows; ++i) {
        out->data[i].real = 0.0;
        out->data[i].imag = 0.0;
    }
    for (matx_int64_t k = 0; k < A->nnz; ++k) {
        matx_double re = A->values[k].real;
        matx_double im = A->values[k].imag;
        out->data[A->rows[k]].real += sqrt(re * re + im * im);
    }
    return MATX_OK;
}

static matx_status_t ref_spcolsums_z_i8_grb(matx_coo_z_i8_t A, matx_vec_z_i8_t out)
{
    if (!A || !out || !A->values) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    for (matx_int64_t j = 0; j < A->ncols; ++j) {
        out->data[j].real = 0.0;
        out->data[j].imag = 0.0;
    }
    for (matx_int64_t k = 0; k < A->nnz; ++k) {
        matx_double re = A->values[k].real;
        matx_double im = A->values[k].imag;
        out->data[A->columns[k]].real += sqrt(re * re + im * im);
    }
    return MATX_OK;
}

// ---- Diagonal extraction ----

static matx_status_t ref_spdiag_d_i8_grb(matx_coo_d_i8_t A, matx_int64_t offset, matx_vec_d_i8_t out)
{
    if (!A || !out) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    matx_int64_t diag_len = (offset >= 0)
                                ? ((A->ncols - offset < A->nrows) ? A->ncols - offset : A->nrows)
                                : ((A->nrows + offset < A->ncols) ? A->nrows + offset : A->ncols);
    if (diag_len < 0)
        diag_len = 0;
    if (out->n < diag_len) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    memset(out->data, 0, sizeof(matx_double) * (size_t) diag_len);
    for (matx_int64_t i = 0; i < A->nnz; ++i) {
        matx_int64_t d = A->columns[i] - A->rows[i];
        if (d == offset) {
            matx_int64_t idx = (offset >= 0) ? A->rows[i] : A->columns[i];
            if (idx >= 0 && idx < diag_len)
                out->data[idx] = A->values[i];
        }
    }
    return MATX_OK;
}

static matx_status_t ref_spdiag_z_i8_grb(matx_coo_z_i8_t A, matx_int64_t offset, matx_vec_z_i8_t out)
{
    if (!A || !out) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    matx_int64_t diag_len = (offset >= 0)
                                ? ((A->ncols - offset < A->nrows) ? A->ncols - offset : A->nrows)
                                : ((A->nrows + offset < A->ncols) ? A->nrows + offset : A->ncols);
    if (diag_len < 0)
        diag_len = 0;
    if (out->n < diag_len) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    for (matx_int64_t i = 0; i < diag_len; ++i) {
        out->data[i].real = 0.0;
        out->data[i].imag = 0.0;
    }
    for (matx_int64_t i = 0; i < A->nnz; ++i) {
        matx_int64_t d = A->columns[i] - A->rows[i];
        if (d == offset) {
            matx_int64_t idx = (offset >= 0) ? A->rows[i] : A->columns[i];
            if (idx >= 0 && idx < diag_len)
                out->data[idx] = A->values[i];
        }
    }
    return MATX_OK;
}

// ---- In-place scaling ----

static matx_status_t ref_scale_rows_d_i8_grb(matx_coo_d_i8_t A, const matx_vec_d_i8_t s)
{
    if (!A || !s || !A->values || !s->data) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    if (s->n < A->nrows) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    for (matx_int64_t i = 0; i < A->nnz; ++i)
        A->values[i] *= s->data[A->rows[i]];
    return MATX_OK;
}

static matx_status_t ref_scale_cols_d_i8_grb(matx_coo_d_i8_t A, const matx_vec_d_i8_t s)
{
    if (!A || !s || !A->values || !s->data) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    if (s->n < A->ncols) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    for (matx_int64_t i = 0; i < A->nnz; ++i)
        A->values[i] *= s->data[A->columns[i]];
    return MATX_OK;
}

static matx_status_t ref_scale_rows_z_i8_grb(matx_coo_z_i8_t A, const matx_vec_z_i8_t s)
{
    if (!A || !s || !A->values || !s->data) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    if (s->n < A->nrows) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    for (matx_int64_t i = 0; i < A->nnz; ++i) {
        matx_double sr = s->data[A->rows[i]].real;
        matx_double si = s->data[A->rows[i]].imag;
        matx_double ar = A->values[i].real;
        matx_double ai = A->values[i].imag;
        A->values[i].real = ar * sr - ai * si;
        A->values[i].imag = ar * si + ai * sr;
    }
    return MATX_OK;
}

static matx_status_t ref_scale_cols_z_i8_grb(matx_coo_z_i8_t A, const matx_vec_z_i8_t s)
{
    if (!A || !s || !A->values || !s->data) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    if (s->n < A->ncols) {
        MATX_ERROR("invalid argument");
        return MATX_ERR_INVALID_ARG;
    }
    for (matx_int64_t i = 0; i < A->nnz; ++i) {
        matx_double sr = s->data[A->columns[i]].real;
        matx_double si = s->data[A->columns[i]].imag;
        matx_double ar = A->values[i].real;
        matx_double ai = A->values[i].imag;
        A->values[i].real = ar * sr - ai * si;
        A->values[i].imag = ar * si + ai * sr;
    }
    return MATX_OK;
}

static matx_bool grb_init_ok = false;

static void do_grb_init(void)
{
#ifdef MATX_ENABLE_GRAPHBLAS
    GrB_Info info = GrB_init(GrB_NONBLOCKING);
    if (info != GrB_SUCCESS) {
        MATX_ERROR("GraphBLAS initialization failed with error code %d", info);
    } else {
        grb_init_ok = true;
    }
#endif
}
#ifdef _WIN32
#include <windows.h>
static INIT_ONCE grb_init_flag = INIT_ONCE_STATIC_INIT;
static BOOL CALLBACK do_grb_init_win(PINIT_ONCE InitOnce, PVOID Parameter, PVOID* Context)
{
    (void) InitOnce;
    (void) Parameter;
    (void) Context;
    do_grb_init();
    return TRUE;
}
#define matx_call_once(flag, func) InitOnceExecuteOnce(flag, do_grb_init_win, NULL, NULL)
#else
#include <threads.h>
static once_flag grb_init_flag = ONCE_FLAG_INIT;
#define matx_call_once(flag, func) call_once(flag, func)
#endif

matx_sparse_backend_t matx_sparse_make_reference_grb(void)
{
    matx_call_once(&grb_init_flag, do_grb_init);
    if (!grb_init_ok) {
        MATX_ERROR("GraphBLAS initialization failed");
    }
    matx_sparse_backend_t b = {.kind = MATX_SPARSE_BACKEND_GRAPHBLAS,
                               .vt = {
                               .spmm_z_i8 = ref_spmm_z_i8_grb,
                               .spmv_z_i8 = ref_spmv_z_i8_grb,
                               .spmm_d_i8 = ref_spmm_d_i8_grb,
                               .spmv_d_i8 = ref_spmv_d_i8_grb,
                               .dsp2md_d_i8 = ref_dsp2md_d_i8_grb,
                               .zsp2md_z_i8 = ref_zsp2md_z_i8_grb,
                               .transpose_d_i8 = ref_transpose_d_i8_grb,
                               .transpose_z_i8 = ref_transpose_z_i8_grb,
                               .conj_trans_z_i8 = ref_conj_trans_z_i8_grb,
                               .finalize = ref_finalize_grb,
                               .norm1_mat_d_i8 = ref_norm1_mat_grb,
                               .norminf_mat_d_i8 = ref_norminf_mat_grb,
                               .normfro_mat_d_i8 = ref_normfro_mat_grb,
                               .norm1_mat_z_i8 = ref_norm1_mat_z_i8_grb,
                               .norminf_mat_z_i8 = ref_norminf_mat_z_i8_grb,
                               .normfro_mat_z_i8 = ref_normfro_mat_z_i8_grb,
                               .spadd_d_i8 = ref_spadd_d_i8_grb,
                               .spadd_z_i8 = ref_spadd_z_i8_grb,
                               .spnnz_rows_d_i8 = ref_spnnz_rows_d_i8_grb,
                               .spnnz_cols_d_i8 = ref_spnnz_cols_d_i8_grb,
                               .spnnz_rows_z_i8 = ref_spnnz_rows_z_i8_grb,
                               .spnnz_cols_z_i8 = ref_spnnz_cols_z_i8_grb,
                               .sprowsums_d_i8 = ref_sprowsums_d_i8_grb,
                               .spcolsums_d_i8 = ref_spcolsums_d_i8_grb,
                               .sprowsums_z_i8 = ref_sprowsums_z_i8_grb,
                               .spcolsums_z_i8 = ref_spcolsums_z_i8_grb,
                               .spdiag_d_i8 = ref_spdiag_d_i8_grb,
                               .spdiag_z_i8 = ref_spdiag_z_i8_grb,
                               .scale_rows_d_i8 = ref_scale_rows_d_i8_grb,
                               .scale_cols_d_i8 = ref_scale_cols_d_i8_grb,
                               .scale_rows_z_i8 = ref_scale_rows_z_i8_grb,
                               .scale_cols_z_i8 = ref_scale_cols_z_i8_grb,
                               }};
    return b;
}
