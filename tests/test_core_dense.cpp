#include <gtest/gtest.h>
#include <math.h>

extern "C" {
#include "matx/matx_types.h"
#include "matx/matx_log.h"
#include "matx/matx_func.h"
#include "matx/matx_types_internal.h"
}

TEST(core_dense, create_destroy) {
  matx_alloc_t a = matx_alloc_default();
  matx_dense_d_i8_t M = NULL;
  ASSERT_EQ(matx_dense_d_i8_create(&a, &M, MATX_COL_MAJOR, 3, 4, NULL), MATX_OK);
  ASSERT_NE(M->data, nullptr);
  EXPECT_EQ(M->nrows, 3u);
  EXPECT_EQ(M->ncols, 4u);
  EXPECT_EQ(M->stride, 3u);
  matx_dense_d_i8_destroy(&a, M);
}

TEST(core_dense, wrap_no_ownership) {
  double buf[6] = {0};
  matx_dense_d_i8_t V = NULL;
  matx_alloc_t a = matx_alloc_default();
  ASSERT_EQ(matx_dense_d_i8_wrap(&a, &V, 2, 3, 3, MATX_COL_MAJOR, buf), MATX_OK);
  EXPECT_EQ(V->data, buf);
  matx_dense_d_i8_destroy(&a, V);
}

TEST(core_dense, create_4x4) {
  matx_alloc_t a = matx_alloc_default();
  matx_dense_d_i8_t M = NULL;
  ASSERT_EQ(matx_dense_d_i8_create(&a, &M, MATX_COL_MAJOR, 4, 4, NULL), MATX_OK);
  ASSERT_NE(M->data, nullptr);
  EXPECT_EQ(M->nrows, 4u);
  EXPECT_EQ(M->ncols, 4u);
  EXPECT_EQ(M->stride, 4u);
  matx_dense_d_i8_destroy(&a, M);
}

TEST(core_dense, dense_z_i8_create_4x4) {
  matx_alloc_t a = matx_alloc_default();
  matx_dense_z_i8_t M = NULL;
  ASSERT_EQ(matx_dense_z_i8_create(&a, &M, MATX_COL_MAJOR, 4, 4, NULL), MATX_OK);
  ASSERT_NE(M->data, nullptr);
  EXPECT_EQ(M->nrows, 4u);
  EXPECT_EQ(M->ncols, 4u);
  matx_dense_z_i8_destroy(&a, M);
}

TEST(core_dense, dup_f4x4) {
	matx_alloc_t a = matx_alloc_default();
	matx_dense_d_i8_t M = NULL;
	ASSERT_EQ(matx_dense_d_i8_create(&a, &M, MATX_COL_MAJOR, 4, 4, NULL), MATX_OK);
	matx_dense_d_i8_t copy = NULL;
	ASSERT_EQ(matx_dense_d_i8_dup(&a, M, &copy), MATX_OK);
	ASSERT_NE(copy->data, nullptr);
	EXPECT_EQ(copy->nrows, 4u);
	EXPECT_EQ(copy->ncols, 4u);
	EXPECT_EQ(copy->stride, 4u);
	matx_dense_d_i8_destroy(&a, M);
	matx_dense_d_i8_destroy(&a, copy);
}

TEST(core_dense, dup_c4x4) {
	matx_alloc_t a = matx_alloc_default();
	matx_dense_z_i8_t M = NULL;
	ASSERT_EQ(matx_dense_z_i8_create(&a, &M, MATX_COL_MAJOR, 4, 4, NULL), MATX_OK);
	matx_dense_z_i8_t copy = NULL;
	ASSERT_EQ(matx_dense_z_i8_dup(&a, M, &copy), MATX_OK);
	ASSERT_NE(copy->data, nullptr);
	EXPECT_EQ(copy->nrows, 4u);
	EXPECT_EQ(copy->ncols, 4u);
	matx_dense_z_i8_destroy(&a, M);
	matx_dense_z_i8_destroy(&a, copy);
}

TEST(core_dense, f64_fill_zeros_ones) {
	matx_alloc_t a = matx_alloc_default();
	matx_dense_d_i8_t M = NULL;
	ASSERT_EQ(matx_dense_d_i8_create(&a, &M, MATX_COL_MAJOR, 3, 3, NULL), MATX_OK);
	ASSERT_EQ(matx_dense_d_i8_fill(M, 5.0), MATX_OK);
	EXPECT_NEAR(M->data[0], 5.0, 1e-12);
	ASSERT_EQ(matx_dense_d_i8_zeros(M), MATX_OK);
	EXPECT_NEAR(M->data[4], 0.0, 1e-12);
	ASSERT_EQ(matx_dense_d_i8_ones(M), MATX_OK);
	EXPECT_NEAR(M->data[8], 1.0, 1e-12);
	matx_dense_d_i8_destroy(&a, M);
}

TEST(core_dense, f64_trace) {
	matx_alloc_t a = matx_alloc_default();
	double data[4] = {1,0,0,4}; // 2x2 col-major: diag = {1,4}
	matx_dense_d_i8_t A = NULL;
	ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_COL_MAJOR, 2, 2, data), MATX_OK);
	double tr = 0.0;
	ASSERT_EQ(matx_dense_d_i8_trace(A, &tr), MATX_OK);
	EXPECT_NEAR(tr, 5.0, 1e-12);
	matx_dense_d_i8_destroy(&a, A);
}

TEST(core_dense, dense_z_i8_fill_zeros) {
	matx_alloc_t a = matx_alloc_default();
	matx_dense_z_i8_t M = NULL;
	ASSERT_EQ(matx_dense_z_i8_create(&a, &M, MATX_COL_MAJOR, 3, 3, NULL), MATX_OK);
	matx_complex_d_t val = { 5.0, 2.0 };
	ASSERT_EQ(matx_dense_z_i8_fill(M, val), MATX_OK);
	EXPECT_NEAR(M->data[0].real, 5.0, 1e-12);
	EXPECT_NEAR(M->data[0].imag, 2.0, 1e-12);
	ASSERT_EQ(matx_dense_z_i8_zeros(M), MATX_OK);
	EXPECT_NEAR(M->data[4].real, 0.0, 1e-12);
	EXPECT_NEAR(M->data[4].imag, 0.0, 1e-12);
	matx_dense_z_i8_destroy(&a, M);
}

TEST(core_dense, dense_z_i8_trace) {
	matx_alloc_t a = matx_alloc_default();
	matx_complex_d_t data[4] = { {1,0}, {0,0}, {0,0}, {4,0} };
	matx_dense_z_i8_t A = NULL;
	ASSERT_EQ(matx_dense_z_i8_create(&a, &A, MATX_COL_MAJOR, 2, 2, data), MATX_OK);
	matx_complex_d_t tr = { 0, 0 };
	ASSERT_EQ(matx_dense_z_i8_trace(A, &tr), MATX_OK);
	EXPECT_NEAR(tr.real, 5.0, 1e-12);
	EXPECT_NEAR(tr.imag, 0.0, 1e-12);
	matx_dense_z_i8_destroy(&a, A);
}