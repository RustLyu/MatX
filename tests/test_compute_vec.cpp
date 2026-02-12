#include <gtest/gtest.h>

extern "C" {
#include "matx/matx.h"
#include "matx/matx_dense_compute.h"
}

TEST(compute_vec, axpy_f64) {
  matx_alloc_t a = matx_alloc_default();
  matx_vec_f64_t x, y;
  ASSERT_EQ(matx_vec_f64_create(&x, 4, &a), MATX_OK);
  ASSERT_EQ(matx_vec_f64_create(&y, 4, &a), MATX_OK);
  x.data[0] = 1.0;
  x.data[1] = 2.0;
  x.data[2] = 3.0;
  x.data[3] = 4.0;
  y.data[0] = 0.1;
  y.data[1] = 0.2;
  y.data[2] = 0.3;
  y.data[3] = 0.4;

  matx_dense_backend_t blas = matx_blas_make_reference();
  matx_status_t st = matx_axpy_f64(&blas, 2.0, &x, &y);
  if (st == MATX_ERR_NOT_SUPPORTED) {
    matx_vec_f64_destroy(&x, &a);
    matx_vec_f64_destroy(&y, &a);
    return;
  }
  ASSERT_EQ(st, MATX_OK);
  EXPECT_NEAR(y.data[0], 2.0 * 1.0 + 0.1, 1e-12);
  EXPECT_NEAR(y.data[1], 2.0 * 2.0 + 0.2, 1e-12);
  EXPECT_NEAR(y.data[3], 2.0 * 4.0 + 0.4, 1e-12);

  matx_vec_f64_destroy(&x, &a);
  matx_vec_f64_destroy(&y, &a);
}

TEST(compute_vec, axpy_c64) {
  matx_alloc_t a = matx_alloc_default();
  matx_vec_c64_t x, y;
  ASSERT_EQ(matx_vec_c64_create(&x, 4, &a), MATX_OK);
  ASSERT_EQ(matx_vec_c64_create(&y, 4, &a), MATX_OK);
  x.data[0] = {1.0, 0.0};
  x.data[1] = {0.0, 1.0};
  x.data[2] = {1.0, 1.0};
  x.data[3] = {0.0, 0.0};
  y.data[0] = {0.5, 0.0};
  y.data[1] = {0.0, 0.5};
  y.data[2] = {0.0, 0.0};
  y.data[3] = {1.0, 1.0};

  matx_dense_backend_t blas = matx_blas_make_reference();
  matx_complex_f64 alpha = {2.0, 0.0};
  matx_status_t st = matx_axpy_c64(&blas, alpha, &x, &y);
  if (st == MATX_ERR_NOT_SUPPORTED) {
    matx_vec_c64_destroy(&x, &a);
    matx_vec_c64_destroy(&y, &a);
    return;
  }
  ASSERT_EQ(st, MATX_OK);
  EXPECT_NEAR(y.data[0].real, 2.5, 1e-12);
  EXPECT_NEAR(y.data[0].imag, 0.0, 1e-12);
  EXPECT_NEAR(y.data[1].real, 0.0, 1e-12);
  EXPECT_NEAR(y.data[1].imag, 2.5, 1e-12);

  matx_vec_c64_destroy(&x, &a);
  matx_vec_c64_destroy(&y, &a);
}
