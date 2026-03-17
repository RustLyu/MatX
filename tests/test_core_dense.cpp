#include <gtest/gtest.h>

extern "C" {
#include "matx/matx_types.h"
#include "matx/matx_func.h"
}

TEST(core_dense, create_destroy) {
  matx_alloc_t a = matx_alloc_default();
  matx_dense_f64_t M;
  ASSERT_EQ(matx_dense_f64_create(&M, 3, 4, MATX_COL_MAJOR, &a), MATX_OK);
  ASSERT_NE(M.data, nullptr);
  EXPECT_EQ(M.rows, 3u);
  EXPECT_EQ(M.cols, 4u);
  EXPECT_EQ(M.stride, 3u);
  matx_dense_f64_destroy(&M, &a);
  EXPECT_EQ(M.data, nullptr);
}

TEST(core_dense, wrap_no_ownership) {
  double buf[6] = {0};
  matx_dense_f64_t V;
  ASSERT_EQ(matx_dense_f64_wrap(&V, 2, 3, 3, MATX_COL_MAJOR, buf), MATX_OK);
  EXPECT_EQ(V.data, buf);
  matx_dense_f64_destroy(&V, nullptr);
  EXPECT_EQ(V.data, nullptr);
}

TEST(core_dense, create_4x4) {
  matx_alloc_t a = matx_alloc_default();
  matx_dense_f64_t M;
  ASSERT_EQ(matx_dense_f64_create(&M, 4, 4, MATX_COL_MAJOR, &a), MATX_OK);
  ASSERT_NE(M.data, nullptr);
  EXPECT_EQ(M.rows, 4u);
  EXPECT_EQ(M.cols, 4u);
  EXPECT_EQ(M.stride, 4u);
  matx_dense_f64_destroy(&M, &a);
}

TEST(core_dense, dense_c64_create_4x4) {
  matx_alloc_t a = matx_alloc_default();
  matx_dense_c64_t M;
  ASSERT_EQ(matx_dense_c64_create(&M, 4, 4, MATX_COL_MAJOR, &a), MATX_OK);
  ASSERT_NE(M.data, nullptr);
  EXPECT_EQ(M.rows, 4u);
  EXPECT_EQ(M.cols, 4u);
  matx_dense_c64_destroy(&M, &a);
}

