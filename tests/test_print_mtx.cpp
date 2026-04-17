#include <gtest/gtest.h>

extern "C" {
#include "matx/matx_types.h"
#include "matx/matx_func.h"
#include "matx/matx_sparse_solve.h"
#include "matx/matx_dense_solve.h"
#include "matx/matx_print.h"
}

TEST(print, sparse_real_4x4_print) {


	matx_int64_t nnz = 4;
	matx_int64_t coo_rows[4] = { 0, 1, 2, 3 };
	matx_int64_t coo_cols[4] = { 0, 1, 2, 3 };
	matx_double coo_values[4] = { 2.0, 2.0, 2.0, 2.0 };

	matx_coo_f64_t coo_A = {
		.nrows = 4,
		.ncols = 4,
		.nnz = nnz,
		.rows = coo_rows,
		.columns = coo_cols,
		.values = coo_values,
		.flags = 0,
		.handle_grb = nullptr,
		.handle_mkl = nullptr
	};
	matx_print_sparse_mtx_f64(&coo_A, "sparse_real_4x4_print.txt");
}

TEST(print, sparse_complex_4x4_print) {
	const int rows = 4, cols = 4;
	const int nnz = 16;

	matx_int64_t coo_rows[16];
	matx_int64_t coo_cols[16];
	matx_complex_f64 coo_values[16];

	for (matx_int64_t i = 0; i < nnz; ++i) {
		matx_int64_t row = i % 4;
		matx_int64_t col = i / 4;

		coo_rows[i] = row;
		coo_cols[i] = col;

		coo_values[i].real = (row == col) ? 2.0 : 0.5;
		coo_values[i].imag = 0.0;
	}
	matx_coo_c64_t A = {
		.nrows = rows,
		.ncols = cols,
		.nnz = nnz,
		.rows = coo_rows,
		.columns = coo_cols,
		.values = coo_values,
		.flags = 0,
		.handle_grb = {.impl = NULL, .type = MATX_HANDLE_TYPE_GRB_MATRIX, .valid = -1},
		.handle_aocl = {.impl = NULL, .type = MATX_HANDLE_TYPE_AOCL_MATRIX, .valid = -1}
	};
	matx_print_sparse_mtx_c64(&A, "sparse_complex_4x4_print.txt");
}

TEST(print, dense_complex_4x4_print) {
	matx_alloc_t a = matx_alloc_default();
	matx_dense_c64_t A;
	ASSERT_EQ(matx_dense_c64_create(&a, &A, MATX_COL_MAJOR, 4, 4, NULL), MATX_OK);
	for (size_t i = 0; i < 16; ++i) {
		A.data[i].real = (double)i;
		A.data[i].imag = 0.0;
	}
	matx_print_dense_mtx_c64(&A, "dense_complex_4x4_print.txt");
	matx_dense_c64_destroy(&a, &A);
}

TEST(print, dense_real_4x4_print) {

	auto fill_dense_f64_4x4 = [](matx_dense_f64_t * M, double base){
		for (size_t j = 0; j < 4; ++j)
			for (size_t i = 0; i < 4; ++i)
				M->data[i + j * M->stride] = base + (double)(i + 4 * j);
	};

	matx_alloc_t a = matx_alloc_default();
	matx_dense_f64_t A;
	ASSERT_EQ(matx_dense_f64_create(&a, &A, MATX_COL_MAJOR, 4, 4, NULL), MATX_OK);
	fill_dense_f64_4x4(&A, 1.0);
	matx_print_dense_mtx_f64(&A, "dense_real_4x4_print.txt");
	matx_dense_f64_destroy(&a, &A);
}

TEST(print, vec_real_4x4_print) {
	matx_alloc_t a = matx_alloc_default();
	matx_vec_f64_t x;
	ASSERT_EQ(matx_vec_f64_create(&a, &x, NULL, 4), MATX_OK);
	x.data[0] = 1.0;
	x.data[1] = 2.0;
	x.data[2] = 3.0;
	x.data[3] = 9.0;
	matx_print_vec_f64(&x, "vec_real_4x4_print.txt");
	matx_vec_f64_destroy(&a, &x);
}

TEST(print, vec_complex_4x4_print) {
	matx_alloc_t a = matx_alloc_default();
	matx_vec_c64_t x;
	ASSERT_EQ(matx_vec_c64_create(&a, &x, NULL, 4), MATX_OK);

	for (matx_int64_t i = 0; i < 4; ++i)
	{
		x.data[i].real = 1.0 * i * 1.0f;
		x.data[i].imag = 1.0 * i * 3.0f;
	}
	matx_print_vec_c64(&x, "vec_complex_4x4_print.txt");
	matx_vec_c64_destroy(&a, &x);
}