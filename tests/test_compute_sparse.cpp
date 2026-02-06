#include <gtest/gtest.h>

extern "C" {
#include "matx/matx.h"
#include "matx/matx_compute.h"
}

/* 4x4 sparse CSC: full matrix for simplicity. col_ptr[0..4], row_ind[0..16], values[16] */
TEST(compute_sparse, spmv_csc_f64_4x4) {
  matx_alloc_t a = matx_alloc_default();
  int col_ptr[5] = {0, 4, 8, 12, 16};
  int row_ind[16] = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
  double values[16];
  for (int i = 0; i < 16; ++i) values[i] = (i % 4 == i / 4) ? 2.0 : 0.5;

  matx_csc_f64_t A = {4, 4, 16, col_ptr, row_ind, values};

  matx_vec_f64_t x, y;
  ASSERT_EQ(matx_vec_f64_create(&x, 4, &a), MATX_OK);
  ASSERT_EQ(matx_vec_f64_create(&y, 4, &a), MATX_OK);
  x.data[0] = 1.0;
  x.data[1] = 1.0;
  x.data[2] = 1.0;
  x.data[3] = 1.0;
  y.data[0] = y.data[1] = y.data[2] = y.data[3] = 0.0;

  matx_status_t st = matx_spmv_csc_f64(1.0, &A, &x, 0.0, &y);
  ASSERT_EQ(st, MATX_OK);
  /* y = A*x; A has diagonal 2, off-diag 0.5. So y_i = 2*1 + 0.5*3 = 3.5 */
  EXPECT_NEAR(y.data[0], 3.5, 1e-12);
  EXPECT_NEAR(y.data[3], 3.5, 1e-12);

  matx_vec_f64_destroy(&x, &a);
  matx_vec_f64_destroy(&y, &a);
}

TEST(compute_sparse, spmv_csc_c64_4x4) {
  matx_alloc_t a = matx_alloc_default();
  int col_ptr[5] = {0, 4, 8, 12, 16};
  int row_ind[16] = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
  matx_complex_f64 values[16];
  for (int i = 0; i < 16; ++i) {
    values[i].real = (i % 4 == i / 4) ? 2.0 : 0.5;
    values[i].imag = 0.0;
  }
  matx_csc_c64_t A = {4, 4, 16, col_ptr, row_ind, values};

  matx_vec_c64_t x, y;
  ASSERT_EQ(matx_vec_c64_create(&x, 4, &a), MATX_OK);
  ASSERT_EQ(matx_vec_c64_create(&y, 4, &a), MATX_OK);
  x.data[0] = x.data[1] = x.data[2] = x.data[3] = {1.0, 0.0};
  y.data[0] = y.data[1] = y.data[2] = y.data[3] = {0.0, 0.0};

  matx_complex_f64 alpha = {1.0, 0.0};
  matx_complex_f64 beta = {0.0, 0.0};
  matx_status_t st = matx_spmv_csc_c64(alpha, &A, &x, beta, &y);
  ASSERT_EQ(st, MATX_OK);
  EXPECT_NEAR(y.data[0].real, 3.5, 1e-12);
  EXPECT_NEAR(y.data[0].imag, 0.0, 1e-12);

  matx_vec_c64_destroy(&x, &a);
  matx_vec_c64_destroy(&y, &a);
}

TEST(compute_sparse, spmm_csc_f64_4x4) {
  matx_alloc_t a = matx_alloc_default();
  int col_ptr[5] = {0, 4, 8, 12, 16};
  int row_ind[16] = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
  double values[16];
  for (int i = 0; i < 16; ++i) values[i] = (i % 4 == i / 4) ? 1.0 : 0.0;
  matx_csc_f64_t A = {4, 4, 16, col_ptr, row_ind, values};

  matx_dense_f64_t B, C;
  ASSERT_EQ(matx_dense_f64_create(&B, 4, 4, MATX_COL_MAJOR, &a), MATX_OK);
  ASSERT_EQ(matx_dense_f64_create(&C, 4, 4, MATX_COL_MAJOR, &a), MATX_OK);
  for (size_t i = 0; i < 16; ++i) B.data[i] = (i % 4 == i / 4) ? 1.0 : 0.0;
  for (size_t i = 0; i < 16; ++i) C.data[i] = 0.0;

  matx_status_t st = matx_spmm_csc_f64(1.0, &A, &B, 0.0, &C);
  ASSERT_EQ(st, MATX_OK);
  EXPECT_NEAR(C.data[0], 1.0, 1e-12);
  EXPECT_NEAR(C.data[5], 1.0, 1e-12);

  matx_dense_f64_destroy(&B, &a);
  matx_dense_f64_destroy(&C, &a);
}

TEST(compute_sparse, spmm_csc_c64_4x4) {
  matx_alloc_t a = matx_alloc_default();
  int col_ptr[5] = {0, 4, 8, 12, 16};
  int row_ind[16] = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
  matx_complex_f64 values[16];
  for (int i = 0; i < 16; ++i) {
    values[i].real = (i % 4 == i / 4) ? 1.0 : 0.0;
    values[i].imag = 0.0;
  }
  matx_csc_c64_t A = {4, 4, 16, col_ptr, row_ind, values};

  matx_dense_c64_t B, C;
  ASSERT_EQ(matx_dense_c64_create(&B, 4, 4, MATX_COL_MAJOR, &a), MATX_OK);
  ASSERT_EQ(matx_dense_c64_create(&C, 4, 4, MATX_COL_MAJOR, &a), MATX_OK);
  for (size_t i = 0; i < 16; ++i) {
    B.data[i].real = (i % 4 == i / 4) ? 1.0 : 0.0;
    B.data[i].imag = 0.0;
    C.data[i].real = C.data[i].imag = 0.0;
  }

  matx_complex_f64 alpha = {1.0, 0.0};
  matx_complex_f64 beta = {0.0, 0.0};
  matx_status_t st = matx_spmm_csc_c64(alpha, &A, &B, beta, &C);
  ASSERT_EQ(st, MATX_OK);
  EXPECT_NEAR(C.data[0].real, 1.0, 1e-12);

  matx_dense_c64_destroy(&B, &a);
  matx_dense_c64_destroy(&C, &a);
}
