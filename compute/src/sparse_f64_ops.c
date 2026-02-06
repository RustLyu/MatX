/* Sparse real f64: spmv and spmm. Uses CXSparse when available. */
#include "matx/matx_compute.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#if defined(MATX_HAVE_CXSPARSE)
#include <suitesparse/GraphBLAS.h>
#endif

matx_status_t matx_spmv_csc_f64(double alpha,
    const matx_csc_f64_t* A,
    const matx_vec_f64_t* x,
    double beta,
    matx_vec_f64_t* y)
{
    if (!A || !x || !y) return MATX_ERR_INVALID_ARG;
    if (A->ncols != x->n || A->nrows != y->n)
        return MATX_ERR_INVALID_ARG;

    /* y = beta*y */
    for (size_t i = 0; i < y->n; i++)
        y->data[i * y->stride] *= beta;

    /* ---------- GraphBLAS backend ---------- */

    GrB_Vector gx, gy;
    GrB_Matrix gA;

    GrB_Vector_new(&gx, GrB_FP64, x->n);
    GrB_Vector_new(&gy, GrB_FP64, y->n);
    GrB_Matrix_new(&gA, GrB_FP64, A->nrows, A->ncols);

    /* build vector */
    for (size_t i = 0; i < x->n; i++)
        GrB_Vector_setElement_FP64(gx, x->data[i * x->stride], i);

    /* build CSC matrix */
    for (size_t j = 0; j < A->ncols; j++)
    {
        for (int k = A->col_ptr[j]; k < A->col_ptr[j + 1]; k++)
        {
            GrB_Matrix_setElement_FP64(gA,
                A->values[k],
                A->row_ind[k],
                j);
        }
    }

    /* tmp = A*x */
    GrB_mxv(gy, NULL, NULL,
        GrB_PLUS_TIMES_SEMIRING_FP64,
        gA, gx, NULL);

    /* y += alpha*tmp */
    for (size_t i = 0; i < y->n; i++)
    {
        double v;
        if (GrB_Vector_extractElement_FP64(&v, gy, i) == GrB_SUCCESS)
            y->data[i * y->stride] += alpha * v;
    }

    GrB_free(&gx);
    GrB_free(&gy);
    GrB_free(&gA);

    return MATX_OK;
}


matx_status_t matx_spmm_csc_f64(double alpha,
                                const matx_csc_f64_t* A,
                                const matx_dense_f64_t* B,
                                double beta,
                                matx_dense_f64_t* C) {
    if (!A || !B || !C) return MATX_ERR_INVALID_ARG;

    if (B->layout != MATX_COL_MAJOR ||
        C->layout != MATX_COL_MAJOR)
        return MATX_ERR_NOT_SUPPORTED;

    size_t m = C->rows;
    size_t n = C->cols;

    for (size_t j = 0; j < n; j++)
        for (size_t i = 0; i < m; i++)
            C->data[i + j * C->stride] *= beta;

    matx_vec_f64_t bj;
    matx_vec_f64_t cj;

    bj.n = A->ncols;
    bj.stride = 1;

    cj.n = A->nrows;
    cj.stride = 1;

    for (size_t j = 0; j < n; j++)
    {
        bj.data = B->data + j * B->stride;
        cj.data = C->data + j * C->stride;

        matx_spmv_csc_f64(alpha, A, &bj, 1.0, &cj);
    }

    return MATX_OK;
}
