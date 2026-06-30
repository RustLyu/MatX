#include <gtest/gtest.h>

extern "C" {
#include "matx/matx_types.h"
#include "matx/matx_func.h"
#include "matx/matx_vec_compute.h"
#include "matx/matx_types_internal.h"
}

TEST(compute_vec, axpy_d_i8) {
  matx_alloc_t a = matx_alloc_default();
  matx_vec_d_i8_t x = NULL, y = NULL;
  ASSERT_EQ(matx_vec_d_i8_create(&a, &x, NULL, 4), MATX_OK);
  ASSERT_EQ(matx_vec_d_i8_create(&a, &y, NULL, 4), MATX_OK);
  x->data[0] = 1.0;
  x->data[1] = 2.0;
  x->data[2] = 3.0;
  x->data[3] = 4.0;
  y->data[0] = 0.1;
  y->data[1] = 0.2;
  y->data[2] = 0.3;
  y->data[3] = 0.4;

  matx_vec_backend_t vblas = matx_vec_default();
  matx_status_t st = matx_vec_axpy_d_i8(&vblas, 2.0, x, y);
  if (st == MATX_ERR_NOT_SUPPORTED) {
    matx_vec_d_i8_destroy(&a, x);
    matx_vec_d_i8_destroy(&a, y);
    return;
  }
  ASSERT_EQ(st, MATX_OK);
  EXPECT_NEAR(y->data[0], 2.0 * 1.0 + 0.1, 1e-12);
  EXPECT_NEAR(y->data[1], 2.0 * 2.0 + 0.2, 1e-12);
  EXPECT_NEAR(y->data[3], 2.0 * 4.0 + 0.4, 1e-12);

  matx_vec_d_i8_destroy(&a, x);
  matx_vec_d_i8_destroy(&a, y);
}

TEST(compute_vec, axpy_z_i8) {
  matx_alloc_t a = matx_alloc_default();
  matx_vec_z_i8_t x = NULL, y = NULL;
  ASSERT_EQ(matx_vec_z_i8_create(&a, &x, NULL, 4), MATX_OK);
  ASSERT_EQ(matx_vec_z_i8_create(&a, &y, NULL, 4), MATX_OK);
  x->data[0] = {1.0, 0.0};
  x->data[1] = {0.0, 1.0};
  x->data[2] = {1.0, 1.0};
  x->data[3] = {0.0, 0.0};
  y->data[0] = {0.5, 0.0};
  y->data[1] = {0.0, 0.5};
  y->data[2] = {0.0, 0.0};
  y->data[3] = {1.0, 1.0};

  matx_vec_backend_t vblas = matx_vec_default();
  matx_complex_d_i8_t alpha = {2.0, 0.0};
  matx_status_t st = matx_vec_axpy_z_i8(&vblas, alpha, x, y);
  if (st == MATX_ERR_NOT_SUPPORTED) {
    matx_vec_z_i8_destroy(&a, x);
    matx_vec_z_i8_destroy(&a, y);
    return;
  }
  ASSERT_EQ(st, MATX_OK);
  EXPECT_NEAR(y->data[0].real, 2.5, 1e-12);
  EXPECT_NEAR(y->data[0].imag, 0.0, 1e-12);
  EXPECT_NEAR(y->data[1].real, 0.0, 1e-12);
  EXPECT_NEAR(y->data[1].imag, 2.5, 1e-12);

  matx_vec_z_i8_destroy(&a, x);
  matx_vec_z_i8_destroy(&a, y);
}
