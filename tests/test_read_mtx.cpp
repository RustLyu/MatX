#include <gtest/gtest.h>

extern "C" {
#include "matx/matx_types.h"
#include "matx/matx_func.h"
#include "matx/matx_sparse_solve.h"
#include "matx/matx_dense_solve.h"
#include "matx/matx_read.h"
#include "matx/matx_types_internal.h"
}

TEST(read, dense_real_4x4) {
	matx_alloc_t a = matx_alloc_default();
	matx_dense_d_i8_t mtx = NULL;
	ASSERT_EQ(matx_read_dense_mtx_d_i8(&a, &mtx, "dense_real_4x4_print.txt"), MATX_OK);
	matx_dense_d_i8_destroy(&a, mtx);
}

TEST(read, dense_complex_4x4) {
	matx_alloc_t a = matx_alloc_default();
	matx_dense_z_i8_t mtx = NULL;
	ASSERT_EQ(matx_read_dense_mtx_z_i8(&a, &mtx, "dense_complex_4x4_print.txt"), MATX_OK);
	matx_dense_z_i8_destroy(&a, mtx);
}


TEST(read, sparse_real_4x4) {
	matx_alloc_t a = matx_alloc_default();
	matx_coo_d_i8_t mtx = NULL;
	ASSERT_EQ(matx_read_sparse_mtx_d_i8(&a, &mtx, "sparse_real_4x4_print.txt"), MATX_OK);
	matx_coo_sparse_d_i8_destroy(&a, mtx);
}

TEST(read, sparse_complex_4x4) {
	matx_alloc_t a = matx_alloc_default();
	matx_coo_z_i8_t mtx = NULL;
	ASSERT_EQ(matx_read_sparse_mtx_z_i8(&a, &mtx, "sparse_complex_4x4_print.txt"), MATX_OK);
	matx_coo_sparse_z_i8_destroy(&a, mtx);
}


TEST(read, vec_real_4x4_read) {
	matx_alloc_t a = matx_alloc_default();
	matx_vec_d_i8_t vec = NULL;
	ASSERT_EQ(matx_read_vec_d_i8(&a, &vec, "vec_real_4x4_print.txt"), MATX_OK);
	matx_vec_d_i8_destroy(&a, vec);
}

TEST(read, vec_complex_4x4_read) {
	matx_alloc_t a = matx_alloc_default();
	matx_vec_z_i8_t vec = NULL;
	ASSERT_EQ(matx_read_vec_z_i8(&a, &vec, "vec_complex_4x4_print.txt"), MATX_OK);
	matx_vec_z_i8_destroy(&a, vec);
}