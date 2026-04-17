#include <gtest/gtest.h>

extern "C" {
#include "matx/matx_types.h"
#include "matx/matx_log.h"
#include "matx/matx_func.h"
}

TEST(core_dense, create_destroy) {
  matx_alloc_t a = matx_alloc_default();
  matx_dense_f64_t M;
  ASSERT_EQ(matx_dense_f64_create(&a, &M, MATX_COL_MAJOR, 3, 4, NULL), MATX_OK);
  ASSERT_NE(M.data, nullptr);
  EXPECT_EQ(M.nrows, 3u);
  EXPECT_EQ(M.ncols, 4u);
  EXPECT_EQ(M.stride, 3u);
  matx_dense_f64_destroy(&a, &M);
  EXPECT_EQ(M.data, nullptr);
}

TEST(core_dense, wrap_no_ownership) {
  double buf[6] = {0};
  matx_dense_f64_t V;
  ASSERT_EQ(matx_dense_f64_wrap(&V, 2, 3, 3, MATX_COL_MAJOR, buf), MATX_OK);
  EXPECT_EQ(V.data, buf);
  matx_dense_f64_destroy(nullptr , &V);
  EXPECT_EQ(V.data, nullptr);
}

TEST(core_dense, create_4x4) {
  matx_alloc_t a = matx_alloc_default();
  matx_dense_f64_t M;
  ASSERT_EQ(matx_dense_f64_create(&a, &M, MATX_COL_MAJOR, 4, 4, NULL), MATX_OK);
  ASSERT_NE(M.data, nullptr);
  EXPECT_EQ(M.nrows, 4u);
  EXPECT_EQ(M.ncols, 4u);
  EXPECT_EQ(M.stride, 4u);
  matx_dense_f64_destroy(&a, &M);
}

TEST(core_dense, dense_c64_create_4x4) {
  matx_alloc_t a = matx_alloc_default();
  matx_dense_c64_t M;
  ASSERT_EQ(matx_dense_c64_create(&a, &M, MATX_COL_MAJOR, 4, 4, NULL), MATX_OK);
  ASSERT_NE(M.data, nullptr);
  EXPECT_EQ(M.nrows, 4u);
  EXPECT_EQ(M.ncols, 4u);
  matx_dense_c64_destroy(&a, &M);
}

TEST(core_dense, dump_f4x4) {
	matx_alloc_t a = matx_alloc_default();
	matx_dense_f64_t M;
	ASSERT_EQ(matx_dense_f64_create(&a, &M, MATX_COL_MAJOR, 4, 4, NULL), MATX_OK);
	matx_dense_f64_t copy;
	ASSERT_EQ(matx_dense_f64_dup(&a, &M, &copy), MATX_OK);
	ASSERT_NE(copy.data, nullptr);
	EXPECT_EQ(copy.nrows, 4u);
	EXPECT_EQ(copy.ncols, 4u);
	EXPECT_EQ(copy.stride, 4u);
	matx_dense_f64_destroy(&a, &M);
	matx_dense_f64_destroy(&a, &copy);
}

TEST(core_dense, dump_c4x4) {
	matx_alloc_t a = matx_alloc_default();
	matx_dense_c64_t M;
	ASSERT_EQ(matx_dense_c64_create(&a, &M, MATX_COL_MAJOR, 4, 4, NULL), MATX_OK);
	matx_dense_c64_t copy;
	ASSERT_EQ(matx_dense_c64_dup(&a, &M, &copy), MATX_OK);
	ASSERT_NE(copy.data, nullptr);
	EXPECT_EQ(copy.nrows, 4u);
	EXPECT_EQ(copy.ncols, 4u);
	matx_dense_c64_destroy(&a, &M);
	matx_dense_c64_destroy(&a, &copy);
}

