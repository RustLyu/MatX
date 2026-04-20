#include <gtest/gtest.h>

extern "C" {
#include "matx/matx_types.h"
#include "matx/matx_func.h"
#include "matx/matx_dense_compute.h"
#include "matx/matx_types_internal.h"
}

TEST(compute_vec, axpy_f64) {
  matx_alloc_t a = matx_alloc_default();
  matx_vec_f64_t x = NULL, y = NULL;
  ASSERT_EQ(matx_vec_f64_create(&a, &x, NULL, 4), MATX_OK);
  ASSERT_EQ(matx_vec_f64_create(&a, &y, NULL, 4), MATX_OK);
  x->data[0] = 1.0;
  x->data[1] = 2.0;
  x->data[2] = 3.0;
  x->data[3] = 4.0;
  y->data[0] = 0.1;
  y->data[1] = 0.2;
  y->data[2] = 0.3;
  y->data[3] = 0.4;

  matx_dense_backend_t blas = matx_blas_default();
  matx_status_t st = matx_axpy_f64(&blas, 2.0, x, y);
  if (st == MATX_ERR_NOT_SUPPORTED) {
    matx_vec_f64_destroy(&a, x);
    matx_vec_f64_destroy(&a, y);
    return;
  }
  ASSERT_EQ(st, MATX_OK);
  EXPECT_NEAR(y->data[0], 2.0 * 1.0 + 0.1, 1e-12);
  EXPECT_NEAR(y->data[1], 2.0 * 2.0 + 0.2, 1e-12);
  EXPECT_NEAR(y->data[3], 2.0 * 4.0 + 0.4, 1e-12);

  matx_vec_f64_destroy(&a, x);
  matx_vec_f64_destroy(&a, y);
}

TEST(compute_vec, axpy_c64) {
  matx_alloc_t a = matx_alloc_default();
  matx_vec_c64_t x = NULL, y = NULL;
  ASSERT_EQ(matx_vec_c64_create(&a, &x, NULL, 4), MATX_OK);
  ASSERT_EQ(matx_vec_c64_create(&a, &y, NULL, 4), MATX_OK);
  x->data[0] = {1.0, 0.0};
  x->data[1] = {0.0, 1.0};
  x->data[2] = {1.0, 1.0};
  x->data[3] = {0.0, 0.0};
  y->data[0] = {0.5, 0.0};
  y->data[1] = {0.0, 0.5};
  y->data[2] = {0.0, 0.0};
  y->data[3] = {1.0, 1.0};

  matx_dense_backend_t blas = matx_blas_default();
  matx_complex_f64 alpha = {2.0, 0.0};
  matx_status_t st = matx_axpy_c64(&blas, alpha, x, y);
  if (st == MATX_ERR_NOT_SUPPORTED) {
    matx_vec_c64_destroy(&a, x);
    matx_vec_c64_destroy(&a, y);
    return;
  }
  ASSERT_EQ(st, MATX_OK);
  EXPECT_NEAR(y->data[0].real, 2.5, 1e-12);
  EXPECT_NEAR(y->data[0].imag, 0.0, 1e-12);
  EXPECT_NEAR(y->data[1].real, 0.0, 1e-12);
  EXPECT_NEAR(y->data[1].imag, 2.5, 1e-12);

  matx_vec_c64_destroy(&a, x);
  matx_vec_c64_destroy(&a, y);
}
