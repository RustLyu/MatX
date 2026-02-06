#include <gtest/gtest.h>

extern "C" {
#include "matx/matx.h"
#include "matx/matx_compute.h"
}

static void fill_col_major(matx_dense_f64_t* M, double base) {
  for (size_t j = 0; j < M->cols; ++j) {
    for (size_t i = 0; i < M->rows; ++i) {
      M->data[i + j * M->stride] = base + (double)(i + 10 * j);
    }
  }
}

TEST(compute, dgemm_reference) {
  matx_alloc_t a = matx_alloc_default();
  matx_dense_f64_t A, B, C;
  ASSERT_EQ(matx_dense_f64_create(&A, 2, 3, MATX_COL_MAJOR, &a), MATX_OK);
  ASSERT_EQ(matx_dense_f64_create(&B, 3, 4, MATX_COL_MAJOR, &a), MATX_OK);
  ASSERT_EQ(matx_dense_f64_create(&C, 2, 4, MATX_COL_MAJOR, &a), MATX_OK);

  fill_col_major(&A, 1.0);
  fill_col_major(&B, 2.0);
  for (size_t i = 0; i < 2 * 4; ++i) C.data[i] = 0.0;

  matx_blas_t blas = matx_blas_make_reference();
  ASSERT_EQ(matx_gemm_f64(&blas, 0, 0, 1.0, &A, &B, 0.0, &C), MATX_OK);

  // Spot-check a couple values against manual computation.
  // C(0,0) = sum_{p=0..2} A(0,p)*B(p,0)
  const double a00 = A.data[0 + 0 * A.stride];
  const double a01 = A.data[0 + 1 * A.stride];
  const double a02 = A.data[0 + 2 * A.stride];
  const double b00 = B.data[0 + 0 * B.stride];
  const double b10 = B.data[1 + 0 * B.stride];
  const double b20 = B.data[2 + 0 * B.stride];
  const double expected00 = a00 * b00 + a01 * b10 + a02 * b20;
  EXPECT_NEAR(C.data[0 + 0 * C.stride], expected00, 1e-12);

  matx_dense_f64_destroy(&A, &a);
  matx_dense_f64_destroy(&B, &a);
  matx_dense_f64_destroy(&C, &a);
}

