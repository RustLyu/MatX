#include <gtest/gtest.h>
#include <fstream>
#include <filesystem>
#include <cmath>

extern "C" {
#include "matx/matx_types.h"
#include "matx/matx_func.h"
#include "matx/matx_sparse_compute.h"
#include "matx/matx_types_internal.h"
}

/* 4x4 sparse CSC: full matrix for simplicity. col_ptr[0..4], row_ind[0..16], values[16] */
TEST(compute_sparse, spmv_csc_d_i8_4x4) {
	matx_alloc_t a = matx_alloc_default();
	matx_int64_t I[16] = { 0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3 };  // 行索引
	matx_int64_t J[16] = { 0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3 };  // 列索引
	matx_double values[16];
	for (matx_int64_t i = 0; i < 16; ++i)
		values[i] = (i % 4 == i / 4) ? 2.0 : 0.5;
	matx_int64_t nnz = 16;
	// 构造 COO 矩阵
	matx_coo_d_i8_t A = NULL;
	matx_coo_sparse_d_i8_create(&a, &A, 4, 4, nnz, I, J, values);
	matx_vec_d_i8_t x = NULL, y = NULL;
	ASSERT_EQ(matx_vec_d_i8_create(&a, &x, NULL, 4), MATX_OK);
	ASSERT_EQ(matx_vec_d_i8_create(&a, &y, NULL, 4), MATX_OK);
	x->data[0] = 1.0;
	x->data[1] = 1.0;
	x->data[2] = 1.0;
	x->data[3] = 1.0;
	y->data[0] = y->data[1] = y->data[2] = y->data[3] = 0.0;

	auto backend = matx_sparse_default();
	matx_status_t st = matx_spmv_coo_d_i8(&backend, 1.0, A, x, 0.0, y);
	ASSERT_EQ(st, MATX_OK);
	/* y = A*x; A has diagonal 2, off-diag 0.5. So y_i = 2*1 + 0.5*3 = 3.5 */
	EXPECT_NEAR(y->data[0], 3.5, 1e-12);
	EXPECT_NEAR(y->data[3], 3.5, 1e-12);
	matx_coo_sparse_d_i8_destroy(&a, A);
	matx_vec_d_i8_destroy(&a, x);
	matx_vec_d_i8_destroy(&a, y);
	matx_finalize(&backend);
}

//TEST(compute_sparse, spmv_csc_d_i8_cd) {
//	std::cout << "Current path: " << std::filesystem::current_path() << "\n";
//    std::ifstream infile("../matrix.txt");
//    if (!infile) {
//        std::cerr << "Cannot open file\n";
//    }
//
//    size_t nrows, ncols;
//    infile >> nrows >> ncols;
//
//    std::vector<matx_int64_t> rows;
//    std::vector<matx_int64_t> cols;
//    std::vector<matx_double> vals;
//
//    matx_int64_t r, c;
//    double v;
//    while (infile >> r >> c >> v) {
//        rows.push_back(r);
//        cols.push_back(c);
//        vals.push_back(v);
//    }
//
//    infile.close();
//
//
//    matx_alloc_t a = matx_alloc_default();
//    // 构造 COO 矩阵
//    matx_coo_d_i8_t A = {
//        .nrows = 12,
//        .ncols = 12,
//        .nnz = (matx_int64_t)vals.size(),
//        .rows = rows.data(),
//        .columns = cols.data(),
//        .values = vals.data(),
//        .flags = 0,
//        .handle_grb = {.impl = NULL, .type = MATX_HANDLE_TYPE_GRB_MATRIX, .valid = -1},
//        .handle_aocl = {.impl = NULL, .type = MATX_HANDLE_TYPE_AOCL_MATRIX, .valid = -1}
//    };
//
//    matx_vec_d_i8_t x, y;
//    ASSERT_EQ(matx_vec_d_i8_create(&x, 12, &a), MATX_OK);
//    ASSERT_EQ(matx_vec_d_i8_create(&y, 12, &a), MATX_OK);
//    x->data[0] = 1.0;
//    x->data[1] = 0.99;
//    x->data[2] = 1.0;
//    x->data[3] = 0.99;
//    x->data[4] = 1.0;
//    x->data[5] = 0.99;
//    x->data[6] = 1.0;
//    x->data[7] = 0.99;
//    x->data[8] = 1.0;
//    x->data[9] = 0.99;
//    x->data[10] = 1.0;
//    x->data[11] = 0.99;
//    y->data[0] = y->data[1] = y->data[2] = y->data[3] = 0.0;
//
//    auto backend = matx_sparse_default();
//    matx_status_t st = matx_spmv_coo_d_i8(&backend, 1.0, A, &x, 0.0, &y);
//    ASSERT_EQ(st, MATX_OK);
//    /* y = A*x; A has diagonal 2, off-diag 0.5. So y_i = 2*1 + 0.5*3 = 3.5 */
//    //EXPECT_NEAR(y->data[0], 3.5, 1e-12);
//    //EXPECT_NEAR(y->data[3], 3.5, 1e-12);
//    matx_coo_sparse_d_i8_destroy(A, &a);
//    matx_vec_d_i8_destroy(&x, &a);
//    matx_vec_d_i8_destroy(&y, &a);
//}


TEST(compute_sparse, spmv_csc_z_i8_4x4) {

	matx_alloc_t a = matx_alloc_default();
	const int rows = 4, cols = 4;
	const int nnz = 16;

	matx_int64_t coo_rows[16];
	matx_int64_t coo_cols[16];
    matx_complex_d_t coo_values[16];

	for (matx_int64_t i = 0; i < nnz; ++i) {
		matx_int64_t row = i % 4;
		matx_int64_t col = i / 4;

		coo_rows[i] = row;
		coo_cols[i] = col;

		coo_values[i].real = (row == col) ? 2.0 : 0.5;
		coo_values[i].imag = 0.0;
	}
	matx_coo_z_i8_t A = NULL;
	matx_coo_sparse_z_i8_create(&a, &A, rows, cols, nnz, coo_rows, coo_cols, coo_values);
	matx_vec_z_i8_t x = NULL, y = NULL;
	ASSERT_EQ(matx_vec_z_i8_create(&a, &x, NULL, cols), MATX_OK);
	ASSERT_EQ(matx_vec_z_i8_create(&a, &y, NULL, rows), MATX_OK);

	for (int i = 0; i < cols; ++i) {
		x->data[i].real = 1.0;
		x->data[i].imag = 0.0;
	}
	for (int i = 0; i < rows; ++i) {
		y->data[i].real = 0.0;
		y->data[i].imag = 0.0;
	}

    matx_complex_d_t alpha = { 1.0, 0.0 };
    matx_complex_d_t beta = { 0.0, 0.0 };
	auto backend = matx_sparse_default();
	matx_status_t st = matx_spmv_coo_z_i8(&backend, alpha, A, x, beta, y);

	ASSERT_EQ(st, MATX_OK);

	const double expected_real = 3.5;
	const double expected_imag = 0.0;
	const double eps = 1e-12;

	for (int i = 0; i < rows; ++i) {
		EXPECT_NEAR(y->data[i].real, expected_real, eps) << "y[" << i << "] real part error";
		EXPECT_NEAR(y->data[i].imag, expected_imag, eps) << "y[" << i << "] imag part error";
	}
	matx_coo_sparse_z_i8_destroy(&a, A);
	matx_vec_z_i8_destroy(&a, x);
	matx_vec_z_i8_destroy(&a, y);
	matx_finalize(&backend);
}

TEST(compute_sparse, spmm_csc_d_i8_4x4) {
	matx_alloc_t a = matx_alloc_default();
	matx_int64_t I[16] = { 0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3 };
	matx_int64_t J[16] = { 0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3 };
	matx_double values[16];
	for (int i = 0; i < 16; ++i)
		values[i] = (i % 4 == i / 4) ? 1.0 : 0.0;
	//matx_csc_d_i8_t A = {4, 4, 16, col_ptr, row_ind, values};
	matx_int64_t nnz = 16;
	matx_coo_d_i8_t A = NULL;
	matx_coo_sparse_d_i8_create(&a, &A, 4, 4, nnz, I, J, values);
	matx_dense_d_i8_t B = NULL, C = NULL;
	ASSERT_EQ(matx_dense_d_i8_create(&a, &B, MATX_COL_MAJOR, 4, 4, NULL), MATX_OK);
	ASSERT_EQ(matx_dense_d_i8_create(&a, &C, MATX_COL_MAJOR, 4, 4, NULL), MATX_OK);
	for (size_t i = 0; i < 16; ++i) B->data[i] = (i % 4 == i / 4) ? 1.0 : 0.0;
	for (size_t i = 0; i < 16; ++i) C->data[i] = 0.0;
	auto backend = matx_sparse_default();
	matx_status_t st = matx_spmm_coo_d_i8(&backend, 1.0, A, B, 0.0, C);
	ASSERT_EQ(st, MATX_OK);
	EXPECT_NEAR(C->data[0], 1.0, 1e-12);
	EXPECT_NEAR(C->data[5], 1.0, 1e-12);

	matx_coo_sparse_d_i8_destroy(&a, A);
	matx_dense_d_i8_destroy(&a, B);
	matx_dense_d_i8_destroy(&a, C);
	matx_finalize(&backend);
}

TEST(compute_sparse, spmm_csc_z_i8_4x4) {
	matx_alloc_t a = matx_alloc_default();
	matx_int64_t nnz = 4;

	matx_int64_t rows[16] = { 0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3 };
	matx_int64_t cols[16] = { 0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3 };

    matx_complex_d_t values[16];
	for (int i = 0; i < 16; ++i) {
		values[i].real = (i % 4 == i / 4) ? 1.0 : 0.0;
		values[i].imag = 0.0;
	}

	matx_coo_z_i8_t A;
	matx_coo_sparse_z_i8_create(&a, &A, 4, 4, nnz, rows, cols, values);
	matx_dense_z_i8_t B = NULL, C = NULL;
	ASSERT_EQ(matx_dense_z_i8_create(&a, &B, MATX_COL_MAJOR, 4, 4, NULL), MATX_OK);
	ASSERT_EQ(matx_dense_z_i8_create(&a, &C, MATX_COL_MAJOR, 4, 4, NULL), MATX_OK);
	MATX_HANDLE(B, MATX_HANDLE_TYPE_GRB_MATRIX)->valid = -1;
	MATX_HANDLE(C, MATX_HANDLE_TYPE_GRB_MATRIX)->valid = -1;
	for (size_t i = 0; i < 16; ++i) {
		B->data[i].real = (i % 4 == i / 4) ? 1.0 : 0.0;
		B->data[i].imag = 0.0;
		C->data[i].real = C->data[i].imag = 0.0;
	}

    matx_complex_d_t alpha = { 1.0, 0.0 };
    matx_complex_d_t beta = { 0.0, 0.0 };
	auto backend = matx_sparse_default();
	matx_status_t st = matx_spmm_coo_z_i8(&backend, alpha, A, B, beta, C);
	ASSERT_EQ(st, MATX_OK);
	EXPECT_NEAR(C->data[0].real, 1.0, 1e-12);
	matx_coo_sparse_z_i8_destroy(&a, A);
	matx_dense_z_i8_destroy(&a, B);
	matx_dense_z_i8_destroy(&a, C);
	matx_finalize(&backend);
}

TEST(compute_sparse, dsp2md_coo_d_i8_4x4) {
	matx_alloc_t a = matx_alloc_default();

	matx_int64_t I_A[16] = { 0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3 };
	matx_int64_t J_A[16] = { 0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3 };
	matx_double values_A[16];
	for (int i = 0; i < 16; ++i) {
		values_A[i] = (i % 4 == i / 4) ? 1.0 : 0.0;
	}
	matx_int64_t nnz = 16;
	matx_coo_d_i8_t A = NULL;
	matx_coo_sparse_d_i8_create(&a, &A, 4, 4, nnz, I_A, J_A, values_A);

	matx_int64_t I_B[16] = { 0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3 };
	matx_int64_t J_B[16] = { 0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3 };
	matx_double values_B[16];
	for (int i = 0; i < 16; ++i) {
		values_B[i] = (i % 4 == i / 4) ? 2.0 : 0.5;
	}
	matx_coo_d_i8_t B = NULL;
	matx_coo_sparse_d_i8_create(&a, &B, 4, 4, nnz, I_B, J_B, values_B);
	matx_sparse_backend_t backend = matx_sparse_default();

	matx_dense_d_i8_t C = NULL;
	ASSERT_EQ(matx_dense_d_i8_create(&a, &C, MATX_COL_MAJOR, 4, 4, NULL), MATX_OK);
	for (size_t i = 0; i < 16; ++i) C->data[i] = 0.0;
	matx_status_t st = matx_dsp2md_coo_d_i8(
		&backend,
		1.0,
		A,
		B,
		0.0,
		C
	);

	ASSERT_EQ(st, MATX_OK);

	EXPECT_NEAR(C->data[0], 2.0, 1e-12);
	EXPECT_NEAR(C->data[1], 0.5, 1e-12);
	EXPECT_NEAR(C->data[14], 0.5, 1e-12);
	EXPECT_NEAR(C->data[15], 2.0, 1e-12);
	matx_coo_sparse_d_i8_destroy(&a, A);
	matx_coo_sparse_d_i8_destroy(&a, B);
	matx_dense_d_i8_destroy(&a, C);
	matx_finalize(&backend);
}


TEST(compute_sparse, transpose_coo_d_i8_2x2) {
	matx_alloc_t a = matx_alloc_default();
	matx_int64_t I_A[16] = { 0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3 };
	matx_int64_t J_A[16] = { 0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3 };
	matx_double values_A[16];
	for (int i = 0; i < 16; ++i) {
		values_A[i] = i;
	}
	matx_coo_d_i8_t A = NULL;
	matx_int64_t nnz = 16;
	matx_coo_sparse_d_i8_create(&a, &A, 4, 4, nnz, I_A, J_A, values_A);
	matx_coo_d_i8_t B = NULL;
	matx_coo_sparse_d_i8_create(&a, &B, 4, 4, 16, NULL, NULL, NULL);
	matx_sparse_backend_t backend = matx_sparse_default();

	matx_status_t st = matx_transpose_coo_d_i8(&backend, A, B);
	matx_finalize(&backend);
	ASSERT_EQ(st, MATX_OK);
}


TEST(compute_sparse, conj_z_i8_4x4) {
	matx_alloc_t a = matx_alloc_default();
	matx_int64_t nnz = 16;

	matx_int64_t rows[16] = { 0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3 };
	matx_int64_t cols[16] = { 0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3 };

    matx_complex_d_t values[16];
	for (int i = 0; i < 16; ++i) {
		values[i].real = i;
		values[i].imag = i + 1;
	}

	matx_coo_z_i8_t A = NULL;
	matx_coo_sparse_z_i8_create(&a, &A, 4, 4, nnz, rows, cols, values);
	matx_coo_z_i8_t B = NULL;
	ASSERT_EQ(matx_coo_sparse_z_i8_create(&a, &B, 4, 4, 16, NULL, NULL, NULL), MATX_OK);
	MATX_HANDLE(B, MATX_HANDLE_TYPE_GRB_MATRIX)->valid = -1;
	for (size_t i = 0; i < 16; ++i) {
		B->values[i].real = 0.0;
		B->values[i].imag = 0.0;
	}

	auto backend = matx_sparse_default();
	matx_status_t st = matx_conj_coo_z_i8(&backend, A, B);
	ASSERT_EQ(st, MATX_OK);
	matx_coo_sparse_z_i8_destroy(&a, A);
	matx_coo_sparse_z_i8_destroy(&a, B);
	matx_finalize(&backend);
}

// ---- Sparse matrix norm tests ----

TEST(compute_sparse, norm1_mat_coo_d_i8) {
	// 2x2 diagonal: A = diag(3, 4) => 1-norm = max col sum = 4
	matx_alloc_t a = matx_alloc_default();
	matx_int64_t rows[2] = {0, 1}, cols[2] = {0, 1};
	matx_double vals[2] = {3.0, 4.0};
	matx_coo_d_i8_t A = NULL;
	ASSERT_EQ(matx_coo_sparse_d_i8_create(&a, &A, 2, 2, 2, rows, cols, vals), MATX_OK);
	auto backend = matx_sparse_default();
	matx_double out = 0.0;
	matx_status_t st = matx_norm1_mat_coo_d_i8(&backend, A, &out);
	if (st == MATX_ERR_NOT_SUPPORTED) { matx_coo_sparse_d_i8_destroy(&a, A); return; }
	ASSERT_EQ(st, MATX_OK);
	EXPECT_NEAR(out, 4.0, 1e-12);
	matx_coo_sparse_d_i8_destroy(&a, A);
	matx_finalize(&backend);
}

TEST(compute_sparse, normfro_mat_coo_d_i8) {
	// A = diag(3, 4) => Frobenius = sqrt(9+16) = 5
	matx_alloc_t a = matx_alloc_default();
	matx_int64_t rows[2] = {0, 1}, cols[2] = {0, 1};
	matx_double vals[2] = {3.0, 4.0};
	matx_coo_d_i8_t A = NULL;
	ASSERT_EQ(matx_coo_sparse_d_i8_create(&a, &A, 2, 2, 2, rows, cols, vals), MATX_OK);
	auto backend = matx_sparse_default();
	matx_double out = 0.0;
	matx_status_t st = matx_normfro_mat_coo_d_i8(&backend, A, &out);
	if (st == MATX_ERR_NOT_SUPPORTED) { matx_coo_sparse_d_i8_destroy(&a, A); return; }
	ASSERT_EQ(st, MATX_OK);
	EXPECT_NEAR(out, 5.0, 1e-12);
	matx_coo_sparse_d_i8_destroy(&a, A);
	matx_finalize(&backend);
}

TEST(compute_sparse, spadd_coo_d_i8) {
	// A = diag(1,2), B = diag(3,4), C = 2*A + 1*B = diag(5,8)
	matx_alloc_t a = matx_alloc_default();
	matx_int64_t rows[2] = {0, 1}, cols[2] = {0, 1};
	matx_double valsA[2] = {1.0, 2.0}, valsB[2] = {3.0, 4.0};
	matx_coo_d_i8_t A = NULL, B = NULL, C = NULL;
	ASSERT_EQ(matx_coo_sparse_d_i8_create(&a, &A, 2, 2, 2, rows, cols, valsA), MATX_OK);
	ASSERT_EQ(matx_coo_sparse_d_i8_create(&a, &B, 2, 2, 2, rows, cols, valsB), MATX_OK);
	ASSERT_EQ(matx_coo_sparse_d_i8_create(&a, &C, 2, 2, 2, NULL, NULL, NULL), MATX_OK);
	auto backend = matx_sparse_default();
	matx_status_t st = matx_spadd_coo_d_i8(&backend, 2.0, A, 1.0, B, C);
	if (st == MATX_ERR_NOT_SUPPORTED) {
		matx_coo_sparse_d_i8_destroy(&a, A);
		matx_coo_sparse_d_i8_destroy(&a, B);
		matx_coo_sparse_d_i8_destroy(&a, C);
		return;
	}
	ASSERT_EQ(st, MATX_OK);
	// C should have 2 entries: (0,0)=5, (1,1)=8
	ASSERT_EQ(C->nnz, 2);
	// find values (order may vary)
	bool found5 = false, found8 = false;
	for (int i = 0; i < C->nnz; ++i) {
		if (C->rows[i] == 0 && C->columns[i] == 0) { EXPECT_NEAR(C->values[i], 5.0, 1e-12); found5 = true; }
		if (C->rows[i] == 1 && C->columns[i] == 1) { EXPECT_NEAR(C->values[i], 8.0, 1e-12); found8 = true; }
	}
	EXPECT_TRUE(found5);
	EXPECT_TRUE(found8);
	matx_coo_sparse_d_i8_destroy(&a, A);
	matx_coo_sparse_d_i8_destroy(&a, B);
	matx_coo_sparse_d_i8_destroy(&a, C);
	matx_finalize(&backend);
}// ---- Non-zero count per row/column tests ----

TEST(compute_sparse, spnnz_rows_cols_coo_d_i8) {
	matx_alloc_t a = matx_alloc_default();
	matx_int64_t rows[6] = {0, 0, 1, 1, 2, 2};
	matx_int64_t cols[6] = {0, 1, 1, 2, 0, 2};
	matx_double vals[6] = {1, 2, 3, 4, 5, 6};
	matx_coo_d_i8_t A = NULL;
	ASSERT_EQ(matx_coo_sparse_d_i8_create(&a, &A, 3, 3, 6, rows, cols, vals), MATX_OK);

	matx_vec_d_i8_t out_r = NULL, out_c = NULL;
	ASSERT_EQ(matx_vec_d_i8_create(&a, &out_r, NULL, 3), MATX_OK);
	ASSERT_EQ(matx_vec_d_i8_create(&a, &out_c, NULL, 3), MATX_OK);
	for (int i = 0; i < 3; ++i) { out_r->data[i] = 0.0; out_c->data[i] = 0.0; }

	auto backend = matx_sparse_default();

	matx_status_t st = matx_spnnz_rows_coo_d_i8(&backend, A, out_r);
	if (st == MATX_ERR_NOT_SUPPORTED) { /* skip */ }
	else {
		ASSERT_EQ(st, MATX_OK);
		EXPECT_NEAR(out_r->data[0], 2.0, 1e-12);
		EXPECT_NEAR(out_r->data[1], 2.0, 1e-12);
		EXPECT_NEAR(out_r->data[2], 2.0, 1e-12);
	}

	st = matx_spnnz_cols_coo_d_i8(&backend, A, out_c);
	if (st == MATX_ERR_NOT_SUPPORTED) { /* skip */ }
	else {
		ASSERT_EQ(st, MATX_OK);
		EXPECT_NEAR(out_c->data[0], 2.0, 1e-12);
		EXPECT_NEAR(out_c->data[1], 2.0, 1e-12);
		EXPECT_NEAR(out_c->data[2], 2.0, 1e-12);
	}

	matx_vec_d_i8_destroy(&a, out_r);
	matx_vec_d_i8_destroy(&a, out_c);
	matx_coo_sparse_d_i8_destroy(&a, A);
	matx_finalize(&backend);
}

TEST(compute_sparse, spnnz_rows_cols_coo_z_i8) {
	matx_alloc_t a = matx_alloc_default();
	matx_int64_t r[4] = {0, 0, 1, 2};
	matx_int64_t c[4] = {0, 1, 1, 2};
    matx_complex_d_t v[4] = {{1,0},{2,0},{3,0},{4,0}};
	matx_coo_z_i8_t A = NULL;
	ASSERT_EQ(matx_coo_sparse_z_i8_create(&a, &A, 3, 3, 4, r, c, v), MATX_OK);

	matx_vec_z_i8_t out_r = NULL, out_c = NULL;
	ASSERT_EQ(matx_vec_z_i8_create(&a, &out_r, NULL, 3), MATX_OK);
	ASSERT_EQ(matx_vec_z_i8_create(&a, &out_c, NULL, 3), MATX_OK);

	auto backend = matx_sparse_default();
	matx_status_t st = matx_spnnz_rows_coo_z_i8(&backend, A, out_r);
	if (st != MATX_ERR_NOT_SUPPORTED) {
		ASSERT_EQ(st, MATX_OK);
		EXPECT_NEAR(out_r->data[0].real, 2.0, 1e-12);
		EXPECT_NEAR(out_r->data[1].real, 1.0, 1e-12);
		EXPECT_NEAR(out_r->data[2].real, 1.0, 1e-12);
	}
	st = matx_spnnz_cols_coo_z_i8(&backend, A, out_c);
	if (st != MATX_ERR_NOT_SUPPORTED) {
		ASSERT_EQ(st, MATX_OK);
		EXPECT_NEAR(out_c->data[0].real, 1.0, 1e-12);
		EXPECT_NEAR(out_c->data[1].real, 2.0, 1e-12);
		EXPECT_NEAR(out_c->data[2].real, 1.0, 1e-12);
	}

	matx_vec_z_i8_destroy(&a, out_r);
	matx_vec_z_i8_destroy(&a, out_c);
	matx_coo_sparse_z_i8_destroy(&a, A);
	matx_finalize(&backend);
}

// ---- Row / column sums tests ----

TEST(compute_sparse, sprowsums_spcolsums_coo_d_i8) {
	matx_alloc_t a = matx_alloc_default();
	matx_int64_t rows[2] = {0, 1}, cols[2] = {0, 1};
	matx_double vals[2] = {3.0, 4.0};
	matx_coo_d_i8_t A = NULL;
	ASSERT_EQ(matx_coo_sparse_d_i8_create(&a, &A, 2, 2, 2, rows, cols, vals), MATX_OK);

	matx_vec_d_i8_t rs = NULL, cs = NULL;
	ASSERT_EQ(matx_vec_d_i8_create(&a, &rs, NULL, 2), MATX_OK);
	ASSERT_EQ(matx_vec_d_i8_create(&a, &cs, NULL, 2), MATX_OK);

	auto backend = matx_sparse_default();
	matx_status_t st = matx_sprowsums_coo_d_i8(&backend, A, rs);
	if (st != MATX_ERR_NOT_SUPPORTED) {
		ASSERT_EQ(st, MATX_OK);
		EXPECT_NEAR(rs->data[0], 3.0, 1e-12);
		EXPECT_NEAR(rs->data[1], 4.0, 1e-12);
	}
	st = matx_spcolsums_coo_d_i8(&backend, A, cs);
	if (st != MATX_ERR_NOT_SUPPORTED) {
		ASSERT_EQ(st, MATX_OK);
		EXPECT_NEAR(cs->data[0], 3.0, 1e-12);
		EXPECT_NEAR(cs->data[1], 4.0, 1e-12);
	}

	matx_vec_d_i8_destroy(&a, rs);
	matx_vec_d_i8_destroy(&a, cs);
	matx_coo_sparse_d_i8_destroy(&a, A);
	matx_finalize(&backend);
}

TEST(compute_sparse, sprowsums_spcolsums_coo_z_i8) {
	matx_alloc_t a = matx_alloc_default();
	matx_int64_t r[2] = {0, 1}, c[2] = {0, 1};
    matx_complex_d_t v[2] = {{3,4},{5,6}};
	matx_coo_z_i8_t A = NULL;
	ASSERT_EQ(matx_coo_sparse_z_i8_create(&a, &A, 2, 2, 2, r, c, v), MATX_OK);

	matx_vec_z_i8_t rs = NULL, cs = NULL;
	ASSERT_EQ(matx_vec_z_i8_create(&a, &rs, NULL, 2), MATX_OK);
	ASSERT_EQ(matx_vec_z_i8_create(&a, &cs, NULL, 2), MATX_OK);

	auto backend = matx_sparse_default();
    double e3 = std::sqrt(9+16), e5 = std::sqrt(25+36);

	matx_status_t st = matx_sprowsums_coo_z_i8(&backend, A, rs);
	if (st != MATX_ERR_NOT_SUPPORTED) {
		ASSERT_EQ(st, MATX_OK);
		EXPECT_NEAR(rs->data[0].real, e3, 1e-12);
		EXPECT_NEAR(rs->data[1].real, e5, 1e-12);
	}
	st = matx_spcolsums_coo_z_i8(&backend, A, cs);
	if (st != MATX_ERR_NOT_SUPPORTED) {
		ASSERT_EQ(st, MATX_OK);
		EXPECT_NEAR(cs->data[0].real, e3, 1e-12);
		EXPECT_NEAR(cs->data[1].real, e5, 1e-12);
	}

	matx_vec_z_i8_destroy(&a, rs);
	matx_vec_z_i8_destroy(&a, cs);
	matx_coo_sparse_z_i8_destroy(&a, A);
	matx_finalize(&backend);
}

// ---- Diagonal extraction tests ----

TEST(compute_sparse, spdiag_coo_d_i8_main_diag) {
	matx_alloc_t a = matx_alloc_default();
	matx_int64_t rows[3] = {0, 1, 2}, cols[3] = {0, 1, 2};
	matx_double vals[3] = {10.0, 20.0, 30.0};
	matx_coo_d_i8_t A = NULL;
	ASSERT_EQ(matx_coo_sparse_d_i8_create(&a, &A, 3, 3, 3, rows, cols, vals), MATX_OK);

	matx_vec_d_i8_t d = NULL;
	ASSERT_EQ(matx_vec_d_i8_create(&a, &d, NULL, 3), MATX_OK);
	d->data[0] = d->data[1] = d->data[2] = -999.0;

	auto backend = matx_sparse_default();
	matx_status_t st = matx_spdiag_coo_d_i8(&backend, A, 0, d);
	if (st != MATX_ERR_NOT_SUPPORTED) {
		ASSERT_EQ(st, MATX_OK);
		EXPECT_NEAR(d->data[0], 10.0, 1e-12);
		EXPECT_NEAR(d->data[1], 20.0, 1e-12);
		EXPECT_NEAR(d->data[2], 30.0, 1e-12);
	}
	matx_vec_d_i8_destroy(&a, d);
	matx_coo_sparse_d_i8_destroy(&a, A);
	matx_finalize(&backend);
}

TEST(compute_sparse, spdiag_coo_d_i8_off_diag) {
	// offset=-1: need row-col=-1, i.e. (0,1),(1,2) for 3x3
	// matrix with entries at (0,1)=100, (1,2)=200
	matx_alloc_t a = matx_alloc_default();
	matx_int64_t rows[2] = {0, 1}, cols[2] = {1, 2};
	matx_double vals[2] = {100.0, 200.0};
	matx_coo_d_i8_t A = NULL;
	ASSERT_EQ(matx_coo_sparse_d_i8_create(&a, &A, 3, 3, 2, rows, cols, vals), MATX_OK);

	// diag_len for offset=-1 in 3x3: min(3, 3+(-1)) = 2
	matx_vec_d_i8_t d = NULL;
	ASSERT_EQ(matx_vec_d_i8_create(&a, &d, NULL, 2), MATX_OK);

	auto backend = matx_sparse_default();
    matx_status_t st = matx_spdiag_coo_d_i8(&backend, A, 1, d);
	if (st != MATX_ERR_NOT_SUPPORTED) {
		ASSERT_EQ(st, MATX_OK);
		EXPECT_NEAR(d->data[0], 100.0, 1e-12); // (0,1) on -1 diagonal: row-col=0-1=-1
		EXPECT_NEAR(d->data[1], 200.0, 1e-12); // (1,2) on -1 diagonal: row-col=1-2=-1
	}
	matx_vec_d_i8_destroy(&a, d);
	matx_coo_sparse_d_i8_destroy(&a, A);
	matx_finalize(&backend);
}

TEST(compute_sparse, spdiag_coo_z_i8_main_diag) {
    matx_alloc_t a = matx_alloc_default();

    matx_int64_t rows[3] = {0, 1, 2};
    matx_int64_t cols[3] = {0, 1, 2};

    matx_complex_d_t vals[3] = {
        {10.0, 1.0},
        {20.0, 2.0},
        {30.0, 3.0}
    };

    matx_coo_z_i8_t A = NULL;
    ASSERT_EQ(
        matx_coo_sparse_z_i8_create(&a, &A, 3, 3, 3, rows, cols, vals),
        MATX_OK);

    matx_vec_z_i8_t d = NULL;
    ASSERT_EQ(matx_vec_z_i8_create(&a, &d, NULL, 3), MATX_OK);

    for (int i = 0; i < 3; ++i) {
        d->data[i].real = -999.0;
        d->data[i].imag = -999.0;
    }

    auto backend = matx_sparse_default();
    matx_status_t st = matx_spdiag_coo_z_i8(&backend, A, 0, d);

    if (st != MATX_ERR_NOT_SUPPORTED) {
        ASSERT_EQ(st, MATX_OK);

        EXPECT_NEAR(d->data[0].real, 10.0, 1e-12);
        EXPECT_NEAR(d->data[0].imag, 1.0, 1e-12);

        EXPECT_NEAR(d->data[1].real, 20.0, 1e-12);
        EXPECT_NEAR(d->data[1].imag, 2.0, 1e-12);

        EXPECT_NEAR(d->data[2].real, 30.0, 1e-12);
        EXPECT_NEAR(d->data[2].imag, 3.0, 1e-12);
    }

    matx_vec_z_i8_destroy(&a, d);
    matx_coo_sparse_z_i8_destroy(&a, A);
    matx_finalize(&backend);
}

TEST(compute_sparse, spdiag_coo_z_i8_off_diag) {
    // offset = +1
    // (0,1), (1,2)

    matx_alloc_t a = matx_alloc_default();

    matx_int64_t rows[2] = {0, 1};
    matx_int64_t cols[2] = {1, 2};

    matx_complex_d_t vals[2] = {
        {100.0, 10.0},
        {200.0, 20.0}
    };

    matx_coo_z_i8_t A = NULL;
    ASSERT_EQ(
        matx_coo_sparse_z_i8_create(&a, &A, 3, 3, 2, rows, cols, vals),
        MATX_OK);

    matx_vec_z_i8_t d = NULL;
    ASSERT_EQ(matx_vec_z_i8_create(&a, &d, NULL, 2), MATX_OK);

    auto backend = matx_sparse_default();

    matx_status_t st = matx_spdiag_coo_z_i8(&backend, A, 1, d);

    if (st != MATX_ERR_NOT_SUPPORTED) {
        ASSERT_EQ(st, MATX_OK);

        EXPECT_NEAR(d->data[0].real, 100.0, 1e-12);
        EXPECT_NEAR(d->data[0].imag, 10.0, 1e-12);

        EXPECT_NEAR(d->data[1].real, 200.0, 1e-12);
        EXPECT_NEAR(d->data[1].imag, 20.0, 1e-12);
    }

    matx_vec_z_i8_destroy(&a, d);
    matx_coo_sparse_z_i8_destroy(&a, A);
    matx_finalize(&backend);
}

// ---- In-place scaling tests ----

TEST(compute_sparse, scale_rows_coo_d_i8) {
	matx_alloc_t a = matx_alloc_default();
	matx_int64_t rows[2] = {0, 1}, cols[2] = {0, 1};
	matx_double vals[2] = {1.0, 2.0};
	matx_coo_d_i8_t A = NULL;
	ASSERT_EQ(matx_coo_sparse_d_i8_create(&a, &A, 2, 2, 2, rows, cols, vals), MATX_OK);

	double s_data[2] = {10.0, 20.0};
	matx_vec_d_i8_t s = NULL;
	ASSERT_EQ(matx_vec_d_i8_create(&a, &s, s_data, 2), MATX_OK);

	auto backend = matx_sparse_default();
	matx_status_t st = matx_scale_rows_coo_d_i8(&backend, A, s);
	if (st != MATX_ERR_NOT_SUPPORTED) {
		ASSERT_EQ(st, MATX_OK);
		EXPECT_NEAR(A->values[0], 10.0, 1e-12);
		EXPECT_NEAR(A->values[1], 40.0, 1e-12);
	}
	matx_vec_d_i8_destroy(&a, s);
	matx_coo_sparse_d_i8_destroy(&a, A);
	matx_finalize(&backend);
}

TEST(compute_sparse, scale_cols_coo_d_i8) {
	matx_alloc_t a = matx_alloc_default();
	matx_int64_t rows[2] = {0, 1}, cols[2] = {0, 1};
	matx_double vals[2] = {1.0, 2.0};
	matx_coo_d_i8_t A = NULL;
	ASSERT_EQ(matx_coo_sparse_d_i8_create(&a, &A, 2, 2, 2, rows, cols, vals), MATX_OK);

	double s_data[2] = {10.0, 20.0};
	matx_vec_d_i8_t s = NULL;
	ASSERT_EQ(matx_vec_d_i8_create(&a, &s, s_data, 2), MATX_OK);

	auto backend = matx_sparse_default();
	matx_status_t st = matx_scale_cols_coo_d_i8(&backend, A, s);
	if (st != MATX_ERR_NOT_SUPPORTED) {
		ASSERT_EQ(st, MATX_OK);
		EXPECT_NEAR(A->values[0], 10.0, 1e-12);
		EXPECT_NEAR(A->values[1], 40.0, 1e-12);
	}
	matx_vec_d_i8_destroy(&a, s);
	matx_coo_sparse_d_i8_destroy(&a, A);
	matx_finalize(&backend);
}

TEST(compute_sparse, scale_rows_coo_z_i8) {
	matx_alloc_t a = matx_alloc_default();
	matx_int64_t r[2] = {0, 1}, c[2] = {0, 1};
    matx_complex_d_t v[2] = {{1,0},{2,0}};
	matx_coo_z_i8_t A = NULL;
	ASSERT_EQ(matx_coo_sparse_z_i8_create(&a, &A, 2, 2, 2, r, c, v), MATX_OK);

    matx_complex_d_t s_data[2] = {{10,0},{20,0}};
	matx_vec_z_i8_t s = NULL;
	ASSERT_EQ(matx_vec_z_i8_create(&a, &s, NULL, 2), MATX_OK);
	s->data[0] = s_data[0]; s->data[1] = s_data[1];

	auto backend = matx_sparse_default();
	matx_status_t st = matx_scale_rows_coo_z_i8(&backend, A, s);
	if (st != MATX_ERR_NOT_SUPPORTED) {
		ASSERT_EQ(st, MATX_OK);
		EXPECT_NEAR(A->values[0].real, 10.0, 1e-12);
		EXPECT_NEAR(A->values[0].imag, 0.0, 1e-12);
		EXPECT_NEAR(A->values[1].real, 40.0, 1e-12);
		EXPECT_NEAR(A->values[1].imag, 0.0, 1e-12);
	}
	matx_vec_z_i8_destroy(&a, s);
	matx_coo_sparse_z_i8_destroy(&a, A);
	matx_finalize(&backend);
}

TEST(compute_sparse, scale_cols_coo_z_i8) {
	matx_alloc_t a = matx_alloc_default();
	matx_int64_t r[2] = {0, 1}, c[2] = {0, 1};
    matx_complex_d_t v[2] = {{1,0},{2,0}};
	matx_coo_z_i8_t A = NULL;
	ASSERT_EQ(matx_coo_sparse_z_i8_create(&a, &A, 2, 2, 2, r, c, v), MATX_OK);

    matx_complex_d_t s_data[2] = {{10,0},{20,0}};
	matx_vec_z_i8_t s = NULL;
	ASSERT_EQ(matx_vec_z_i8_create(&a, &s, NULL, 2), MATX_OK);
	s->data[0] = s_data[0]; s->data[1] = s_data[1];

	auto backend = matx_sparse_default();
	matx_status_t st = matx_scale_cols_coo_z_i8(&backend, A, s);
	if (st != MATX_ERR_NOT_SUPPORTED) {
		ASSERT_EQ(st, MATX_OK);
		EXPECT_NEAR(A->values[0].real, 10.0, 1e-12);
		EXPECT_NEAR(A->values[0].imag, 0.0, 1e-12);
		EXPECT_NEAR(A->values[1].real, 40.0, 1e-12);
		EXPECT_NEAR(A->values[1].imag, 0.0, 1e-12);
	}
	matx_vec_z_i8_destroy(&a, s);
	matx_coo_sparse_z_i8_destroy(&a, A);
	matx_finalize(&backend);
}

TEST(compute_sparse, zsp2md_coo_z_i8_4x4) {
	matx_alloc_t a = matx_alloc_default();
	matx_int64_t I_A[16] = { 0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3 };
	matx_int64_t J_A[16] = { 0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3 };
	matx_complex_d_t values_A[16];
	for (int i = 0; i < 16; ++i) {
		values_A[i].real = (i % 4 == i / 4) ? 1.0 : 0.0;
		values_A[i].imag = 0.0;
	}
	matx_coo_z_i8_t A = NULL;
	matx_coo_sparse_z_i8_create(&a, &A, 4, 4, 16, I_A, J_A, values_A);

	matx_int64_t I_B[16] = { 0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3 };
	matx_int64_t J_B[16] = { 0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3 };
	matx_complex_d_t values_B[16];
	for (int i = 0; i < 16; ++i) {
		values_B[i].real = (i % 4 == i / 4) ? 2.0 : 0.5;
		values_B[i].imag = 0.0;
	}
	matx_coo_z_i8_t B = NULL;
	matx_coo_sparse_z_i8_create(&a, &B, 4, 4, 16, I_B, J_B, values_B);

	matx_dense_z_i8_t C = NULL;
	ASSERT_EQ(matx_dense_z_i8_create(&a, &C, MATX_COL_MAJOR, 4, 4, NULL), MATX_OK);
	for (size_t i = 0; i < 16; ++i) C->data[i].real = C->data[i].imag = 0.0;

	matx_complex_d_t alpha = {1.0, 0.0};
	matx_complex_d_t beta  = {0.0, 0.0};
	matx_sparse_backend_t backend = matx_sparse_default();
	matx_status_t st = matx_zsp2md_coo_z_i8(&backend, alpha, A, B, beta, C);
	if (st == MATX_ERR_NOT_SUPPORTED) {
		matx_coo_sparse_z_i8_destroy(&a, A);
		matx_coo_sparse_z_i8_destroy(&a, B);
		matx_dense_z_i8_destroy(&a, C);
		matx_finalize(&backend);
		return;
	}
	ASSERT_EQ(st, MATX_OK);
	EXPECT_NEAR(C->data[0].real, 2.0, 1e-12);
	EXPECT_NEAR(C->data[1].real, 0.5, 1e-12);
	EXPECT_NEAR(C->data[14].real, 0.5, 1e-12);
	EXPECT_NEAR(C->data[15].real, 2.0, 1e-12);

	matx_coo_sparse_z_i8_destroy(&a, A);
	matx_coo_sparse_z_i8_destroy(&a, B);
	matx_dense_z_i8_destroy(&a, C);
	matx_finalize(&backend);
}

TEST(compute_sparse, transpose_coo_z_i8_4x4) {
	matx_alloc_t a = matx_alloc_default();
	matx_int64_t I_A[16] = { 0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3 };
	matx_int64_t J_A[16] = { 0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3 };
	matx_complex_d_t values_A[16];
	for (int i = 0; i < 16; ++i) {
		values_A[i].real = i;
		values_A[i].imag = 0.0;
	}
	matx_coo_z_i8_t A = NULL, B = NULL;
	matx_coo_sparse_z_i8_create(&a, &A, 4, 4, 16, I_A, J_A, values_A);
	matx_coo_sparse_z_i8_create(&a, &B, 4, 4, 16, NULL, NULL, NULL);
	matx_sparse_backend_t backend = matx_sparse_default();

	matx_status_t st = matx_transpose_coo_z_i8(&backend, A, B);
	matx_finalize(&backend);
	ASSERT_EQ(st, MATX_OK);
	matx_coo_sparse_z_i8_destroy(&a, A);
	matx_coo_sparse_z_i8_destroy(&a, B);
}

TEST(compute_sparse, norm1_mat_coo_z_i8) {
	matx_alloc_t a = matx_alloc_default();
	matx_int64_t rows[2] = {0, 1}, cols[2] = {0, 1};
	matx_complex_d_t vals[2] = {{3.0, 4.0}, {4.0, 0.0}};
	matx_coo_z_i8_t A = NULL;
	ASSERT_EQ(matx_coo_sparse_z_i8_create(&a, &A, 2, 2, 2, rows, cols, vals), MATX_OK);
	auto backend = matx_sparse_default();
	matx_double out = 0.0;
	matx_status_t st = matx_norm1_mat_coo_z_i8(&backend, A, &out);
	if (st == MATX_ERR_NOT_SUPPORTED) { matx_coo_sparse_z_i8_destroy(&a, A); return; }
	ASSERT_EQ(st, MATX_OK);
	EXPECT_NEAR(out, 5.0, 1e-12);
	matx_coo_sparse_z_i8_destroy(&a, A);
	matx_finalize(&backend);
}

TEST(compute_sparse, norminf_mat_coo_z_i8) {
	matx_alloc_t a = matx_alloc_default();
	matx_int64_t rows[2] = {0, 1}, cols[2] = {0, 1};
	matx_complex_d_t vals[2] = {{3.0, 4.0}, {4.0, 0.0}};
	matx_coo_z_i8_t A = NULL;
	ASSERT_EQ(matx_coo_sparse_z_i8_create(&a, &A, 2, 2, 2, rows, cols, vals), MATX_OK);
	auto backend = matx_sparse_default();
	matx_double out = 0.0;
	matx_status_t st = matx_norminf_mat_coo_z_i8(&backend, A, &out);
	if (st == MATX_ERR_NOT_SUPPORTED) { matx_coo_sparse_z_i8_destroy(&a, A); return; }
	ASSERT_EQ(st, MATX_OK);
	EXPECT_NEAR(out, 5.0, 1e-12);
	matx_coo_sparse_z_i8_destroy(&a, A);
	matx_finalize(&backend);
}

TEST(compute_sparse, normfro_mat_coo_z_i8) {
	matx_alloc_t a = matx_alloc_default();
	matx_int64_t rows[1] = {0}, cols[1] = {0};
	matx_complex_d_t vals[1] = {{3.0, 4.0}};
	matx_coo_z_i8_t A = NULL;
	ASSERT_EQ(matx_coo_sparse_z_i8_create(&a, &A, 2, 2, 1, rows, cols, vals), MATX_OK);
	auto backend = matx_sparse_default();
	matx_double out = 0.0;
	matx_status_t st = matx_normfro_mat_coo_z_i8(&backend, A, &out);
	if (st == MATX_ERR_NOT_SUPPORTED) { matx_coo_sparse_z_i8_destroy(&a, A); return; }
	ASSERT_EQ(st, MATX_OK);
	EXPECT_NEAR(out, 5.0, 1e-12);
	matx_coo_sparse_z_i8_destroy(&a, A);
	matx_finalize(&backend);
}

TEST(compute_sparse, spadd_coo_z_i8) {
	matx_alloc_t a = matx_alloc_default();
	matx_int64_t rows[2] = {0, 1}, cols[2] = {0, 1};
	matx_complex_d_t valsA[2] = {{1.0, 2.0}, {2.0, 0.0}};
	matx_complex_d_t valsB[2] = {{3.0, 0.0}, {4.0, 0.0}};
	matx_coo_z_i8_t A = NULL, B = NULL, C = NULL;
	ASSERT_EQ(matx_coo_sparse_z_i8_create(&a, &A, 2, 2, 2, rows, cols, valsA), MATX_OK);
	ASSERT_EQ(matx_coo_sparse_z_i8_create(&a, &B, 2, 2, 2, rows, cols, valsB), MATX_OK);
	ASSERT_EQ(matx_coo_sparse_z_i8_create(&a, &C, 2, 2, 2, NULL, NULL, NULL), MATX_OK);
	auto backend = matx_sparse_default();
	matx_complex_d_t alpha = {2.0, 0.0};
	matx_complex_d_t beta  = {1.0, 0.0};
	matx_status_t st = matx_spadd_coo_z_i8(&backend, alpha, A, beta, B, C);
	if (st == MATX_ERR_NOT_SUPPORTED) {
		matx_coo_sparse_z_i8_destroy(&a, A);
		matx_coo_sparse_z_i8_destroy(&a, B);
		matx_coo_sparse_z_i8_destroy(&a, C);
		return;
	}
	ASSERT_EQ(st, MATX_OK);
	ASSERT_EQ(C->nnz, 2);
	bool found5 = false, found8 = false;
	for (int i = 0; i < C->nnz; ++i) {
		if (C->rows[i] == 0 && C->columns[i] == 0) {
			EXPECT_NEAR(C->values[i].real, 5.0, 1e-12);
			EXPECT_NEAR(C->values[i].imag, 4.0, 1e-12);
			found5 = true;
		}
		if (C->rows[i] == 1 && C->columns[i] == 1) {
			EXPECT_NEAR(C->values[i].real, 8.0, 1e-12);
			EXPECT_NEAR(C->values[i].imag, 0.0, 1e-12);
			found8 = true;
		}
	}
	EXPECT_TRUE(found5);
	EXPECT_TRUE(found8);
	matx_coo_sparse_z_i8_destroy(&a, A);
	matx_coo_sparse_z_i8_destroy(&a, B);
	matx_coo_sparse_z_i8_destroy(&a, C);
	matx_finalize(&backend);
}
