#include <gtest/gtest.h>

extern "C" {
#include "matx/matx_types.h"
#include "matx/matx_func.h"
#include "matx/matx_sparse_solve.h"
#include "matx/matx_dense_solve.h"
#include "matx/matx_read.h"
}

TEST(read, dense_real_4x4) {
	matx_alloc_t a = matx_alloc_default();
	matx_dense_f64_t mtx;
	ASSERT_EQ(matx_read_dense_mtx_f64(&a, &mtx, "dense_real_4x4_print.txt"), MATX_OK);
	matx_dense_f64_destroy(&a, &mtx);
}

TEST(read, dense_complex_4x4) {
	matx_alloc_t a = matx_alloc_default();
	matx_dense_c64_t mtx;
	ASSERT_EQ(matx_read_dense_mtx_c64(&a, &mtx, "dense_complex_4x4_print.txt"), MATX_OK);
	matx_dense_c64_destroy(&a, &mtx);
}


TEST(read, sparse_real_4x4) {
	matx_alloc_t a = matx_alloc_default();
	matx_coo_f64_t mtx;
	ASSERT_EQ(matx_read_sparse_mtx_f64(&a, &mtx, "sparse_real_4x4_print.txt"), MATX_OK);
	matx_coo_sparse_f64_destroy(&a, &mtx);
}

TEST(read, sparse_complex_4x4) {
	matx_alloc_t a = matx_alloc_default();
	matx_coo_c64_t mtx;
	ASSERT_EQ(matx_read_sparse_mtx_c64(&a, &mtx, "sparse_complex_4x4_print.txt"), MATX_OK);
	matx_coo_sparse_c64_destroy(&a, &mtx);
}


TEST(read, vec_real_4x4_read) {
	matx_alloc_t a = matx_alloc_default();
	matx_vec_f64_t vec;
	ASSERT_EQ(matx_read_vec_f64(&a, &vec, "vec_real_4x4_print.txt"), MATX_OK);
	matx_vec_f64_destroy(&a, &vec);
}

TEST(read, vec_complex_4x4_read) {
	matx_alloc_t a = matx_alloc_default();
	matx_vec_c64_t vec;
	ASSERT_EQ(matx_read_vec_c64(&a, &vec, "vec_complex_4x4_print.txt"), MATX_OK);
	matx_vec_c64_destroy(&a, &vec);
}