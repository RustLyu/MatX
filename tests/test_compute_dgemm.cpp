#include <gtest/gtest.h>

extern "C" {
#include "matx/matx_dense_compute.h"
#include "matx/matx_func.h"
#include "matx/matx_types.h"
#include "matx/matx_types_internal.h"
}

static void fill_col_major(matx_dense_d_i8_t M, double base)
{
    for (size_t j = 0; j < M->ncols; ++j) {
        for (size_t i = 0; i < M->nrows; ++i) {
            M->data[i + j * M->stride] = base + (double) (i + 10 * j);
        }
    }
}

TEST(compute, dgemm_reference)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t A = NULL, B = NULL, C = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_COL_MAJOR, 2, 3, NULL), MATX_OK);
    ASSERT_EQ(matx_dense_d_i8_create(&a, &B, MATX_COL_MAJOR, 3, 4, NULL), MATX_OK);
    ASSERT_EQ(matx_dense_d_i8_create(&a, &C, MATX_COL_MAJOR, 2, 4, NULL), MATX_OK);

    fill_col_major(A, 1.0);
    fill_col_major(B, 2.0);
    for (size_t i = 0; i < 2 * 4; ++i)
        C->data[i] = 0.0;

    matx_dense_backend_t blas = matx_blas_default();
    ASSERT_EQ(matx_gemm_d_i8(&blas, 0, 0, 1.0, A, B, 0.0, C), MATX_OK);

    // Spot-check a couple values against manual computation.
    // C(0,0) = sum_{p=0..2} A(0,p)*B(p,0)
    const double a00 = A->data[0 + 0 * A->stride];
    const double a01 = A->data[0 + 1 * A->stride];
    const double a02 = A->data[0 + 2 * A->stride];
    const double b00 = B->data[0 + 0 * B->stride];
    const double b10 = B->data[1 + 0 * B->stride];
    const double b20 = B->data[2 + 0 * B->stride];
    const double expected00 = a00 * b00 + a01 * b10 + a02 * b20;
    EXPECT_NEAR(C->data[0 + 0 * C->stride], expected00, 1e-12);

    matx_dense_d_i8_destroy(&a, A);
    matx_dense_d_i8_destroy(&a, B);
    matx_dense_d_i8_destroy(&a, C);
}
