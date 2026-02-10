#include <gtest/gtest.h>

extern "C" {
#include "matx/matx.h"
#include "matx/matx_compute.h"
}

static void fill_dense_f64_4x4(matx_dense_f64_t* M, double base) {
  for (size_t j = 0; j < 4; ++j)
    for (size_t i = 0; i < 4; ++i)
      M->data[i + j * M->stride] = base + (double)(i + 4 * j);
}

TEST(compute_dense, geadd_f64_4x4) {
  matx_alloc_t a = matx_alloc_default();
  matx_dense_f64_t A, B;
  ASSERT_EQ(matx_dense_f64_create(&A, 4, 4, MATX_COL_MAJOR, &a), MATX_OK);
  ASSERT_EQ(matx_dense_f64_create(&B, 4, 4, MATX_COL_MAJOR, &a), MATX_OK);
  fill_dense_f64_4x4(&A, 1.0);
  fill_dense_f64_4x4(&B, 2.0);
  matx_dense_backend_t blas = matx_blas_make_reference();
  matx_status_t st = matx_geadd_f64(&blas, 3.0, &A, 0.0, &B);
  ASSERT_EQ(st, MATX_OK);
  EXPECT_NEAR(B.data[0], 3.0 * 1.0, 1e-12);
  EXPECT_NEAR(B.data[5], 3.0 * 6.0, 1e-12);

  matx_dense_f64_destroy(&A, &a);
  matx_dense_f64_destroy(&B, &a);
}

TEST(compute_dense, geadd_c64_4x4) {
  matx_alloc_t a = matx_alloc_default();
  matx_dense_c64_t A, B;
  ASSERT_EQ(matx_dense_c64_create(&A, 4, 4, MATX_COL_MAJOR, &a), MATX_OK);
  ASSERT_EQ(matx_dense_c64_create(&B, 4, 4, MATX_COL_MAJOR, &a), MATX_OK);
  for (size_t i = 0; i < 16; ++i) {
    A.data[i].real = (double)i;
    A.data[i].imag = 0.0;
    B.data[i].real = 1.0;
    B.data[i].imag = 0.0;
  }

  matx_dense_backend_t blas = matx_blas_make_reference();
  matx_complex_f64 alpha = {2.0, 0.0};
  matx_complex_f64 beta = {0.0, 0.0};
  matx_status_t st = matx_geadd_c64(&blas, alpha, &A, beta, &B);
  ASSERT_EQ(st, MATX_OK);
  EXPECT_NEAR(B.data[0].real, 0.0, 1e-12);
  EXPECT_NEAR(B.data[1].real, 2.0, 1e-12);

  matx_dense_c64_destroy(&A, &a);
  matx_dense_c64_destroy(&B, &a);
}

TEST(compute_dense, gemv_f64_4x4) {
  matx_alloc_t a = matx_alloc_default();
  matx_dense_f64_t A;
  matx_vec_f64_t x, y;
  ASSERT_EQ(matx_dense_f64_create(&A, 4, 4, MATX_COL_MAJOR, &a), MATX_OK);
  ASSERT_EQ(matx_vec_f64_create(&x, 4, &a), MATX_OK);
  ASSERT_EQ(matx_vec_f64_create(&y, 4, &a), MATX_OK);
  fill_dense_f64_4x4(&A, 1.0);
  x.data[0] = 1.0;
  x.data[1] = 0.0;
  x.data[2] = 0.0;
  x.data[3] = 0.0;
  y.data[0] = y.data[1] = y.data[2] = y.data[3] = 0.0;

  matx_dense_backend_t blas = matx_blas_default();
  matx_status_t st = matx_gemv_f64(&blas, 0, 1.0, &A, &x, 0.0, &y);
  if (st == MATX_ERR_NOT_SUPPORTED) {
    matx_dense_f64_destroy(&A, &a);
    matx_vec_f64_destroy(&x, &a);
    matx_vec_f64_destroy(&y, &a);
    return;
  }
  ASSERT_EQ(st, MATX_OK);
  EXPECT_NEAR(y.data[0], A.data[0], 1e-12);
  EXPECT_NEAR(y.data[1], A.data[1], 1e-12);

  matx_dense_f64_destroy(&A, &a);
  matx_vec_f64_destroy(&x, &a);
  matx_vec_f64_destroy(&y, &a);
}

TEST(compute_dense, gemm_f64_4x4) {
  matx_alloc_t a = matx_alloc_default();
  matx_dense_f64_t A, B, C;
  ASSERT_EQ(matx_dense_f64_create(&A, 4, 4, MATX_COL_MAJOR, &a), MATX_OK);
  ASSERT_EQ(matx_dense_f64_create(&B, 4, 4, MATX_COL_MAJOR, &a), MATX_OK);
  ASSERT_EQ(matx_dense_f64_create(&C, 4, 4, MATX_COL_MAJOR, &a), MATX_OK);
  fill_dense_f64_4x4(&A, 1.0);
  fill_dense_f64_4x4(&B, 0.5);
  for (size_t i = 0; i < 16; ++i) C.data[i] = 0.0;

  matx_dense_backend_t blas = matx_blas_default();
  matx_status_t st = matx_gemm_f64(&blas, 0, 0, 1.0, &A, &B, 0.0, &C);
  ASSERT_EQ(st, MATX_OK);
  double c00 = 0.0;
  for (size_t k = 0; k < 4; ++k) c00 += A.data[0 + k * 4] * B.data[k + 0 * 4];
  EXPECT_NEAR(C.data[0], c00, 1e-10);

  matx_dense_f64_destroy(&A, &a);
  matx_dense_f64_destroy(&B, &a);
  matx_dense_f64_destroy(&C, &a);
}

TEST(compute_dense, gemm_c64_4x4) {
  matx_alloc_t a = matx_alloc_default();
  matx_dense_c64_t A, B, C;
  ASSERT_EQ(matx_dense_c64_create(&A, 4, 4, MATX_COL_MAJOR, &a), MATX_OK);
  ASSERT_EQ(matx_dense_c64_create(&B, 4, 4, MATX_COL_MAJOR, &a), MATX_OK);
  ASSERT_EQ(matx_dense_c64_create(&C, 4, 4, MATX_COL_MAJOR, &a), MATX_OK);
  for (size_t i = 0; i < 16; ++i) {
    A.data[i].real = (double)i;
    A.data[i].imag = 0.0;
    B.data[i].real = (i == 0 ? 1.0 : 0.0);
    B.data[i].imag = 0.0;
    C.data[i].real = C.data[i].imag = 0.0;
  }

  matx_dense_backend_t blas = matx_blas_default();
  matx_complex_f64 alpha = {1.0, 0.0};
  matx_complex_f64 beta = {0.0, 0.0};
  matx_status_t st = matx_gemm_c64(&blas, 0, 0, alpha, &A, &B, beta, &C);
  if (st == MATX_ERR_NOT_SUPPORTED) {
    matx_dense_c64_destroy(&A, &a);
    matx_dense_c64_destroy(&B, &a);
    matx_dense_c64_destroy(&C, &a);
    return;
  }
  ASSERT_EQ(st, MATX_OK);
  EXPECT_NEAR(C.data[0].real, A.data[0].real, 1e-10);

  matx_dense_c64_destroy(&A, &a);
  matx_dense_c64_destroy(&B, &a);
  matx_dense_c64_destroy(&C, &a);
}
