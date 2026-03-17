#include <gtest/gtest.h>

extern "C" {
#include "matx/matx.h"
#include "matx/matx_sparse_compute.h"
}

/* 4x4 sparse CSC: full matrix for simplicity. col_ptr[0..4], row_ind[0..16], values[16] */
TEST(compute_sparse, spmv_csc_f64_4x4) {
  matx_alloc_t a = matx_alloc_default();
  matx_int64_t I[16] = { 0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3 };  // 行索引
  matx_int64_t J[16] = { 0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3 };  // 列索引
  matx_double values[16];
  for (matx_int64_t i = 0; i < 16; ++i) 
      values[i] = (i % 4 == i / 4) ? 2.0 : 0.5;

  // 构造 COO 矩阵
  matx_coo_f64_t A = {
      .nrows = 4,
      .ncols = 4,
      .nnz = 16,
      .rows = I,
      .columns = J,
	  .values = values,
      .flags = 0,
	  .handle_grb = {.impl = NULL, .type = MATX_HANDLE_TYPE_GRB_MATRIX, .valid = -1},
      .handle_aocl = {.impl = NULL, .type = MATX_HANDLE_TYPE_AOCL_MATRIX, .valid = -1}
  };

  matx_vec_f64_t x, y;
  ASSERT_EQ(matx_vec_f64_create(&x, 4, &a), MATX_OK);
  ASSERT_EQ(matx_vec_f64_create(&y, 4, &a), MATX_OK);
  x.data[0] = 1.0;
  x.data[1] = 1.0;
  x.data[2] = 1.0;
  x.data[3] = 1.0;
  y.data[0] = y.data[1] = y.data[2] = y.data[3] = 0.0;

  auto backend = matx_sparse_default();
  matx_status_t st = matx_spmv_coo_f64(&backend, 1.0, &A, &x, 0.0, &y);
  ASSERT_EQ(st, MATX_OK);
  /* y = A*x; A has diagonal 2, off-diag 0.5. So y_i = 2*1 + 0.5*3 = 3.5 */
  EXPECT_NEAR(y.data[0], 3.5, 1e-12);
  EXPECT_NEAR(y.data[3], 3.5, 1e-12);
  matx_coo_sparse_f64_destroy(&A, &a);
  matx_vec_f64_destroy(&x, &a);
  matx_vec_f64_destroy(&y, &a);
}

#include <fstream>
#include <filesystem>
TEST(compute_sparse, spmv_csc_f64_cd) {
	std::cout << "Current path: " << std::filesystem::current_path() << "\n";
    std::ifstream infile("../matrix.txt");
    if (!infile) {
        std::cerr << "Cannot open file\n";
    }

    size_t nrows, ncols;
    infile >> nrows >> ncols;

    std::vector<matx_int64_t> rows;
    std::vector<matx_int64_t> cols;
    std::vector<matx_double> vals;

    matx_int64_t r, c;
    double v;
    while (infile >> r >> c >> v) {
        rows.push_back(r);
        cols.push_back(c);
        vals.push_back(v);
    }

    infile.close();


    matx_alloc_t a = matx_alloc_default();
    // 构造 COO 矩阵
    matx_coo_f64_t A = {
        .nrows = 12,
        .ncols = 12,
        .nnz = (matx_int64_t)vals.size(),
        .rows = rows.data(),
        .columns = cols.data(),
        .values = vals.data(),
        .flags = 0,
        .handle_grb = {.impl = NULL, .type = MATX_HANDLE_TYPE_GRB_MATRIX, .valid = -1},
        .handle_aocl = {.impl = NULL, .type = MATX_HANDLE_TYPE_AOCL_MATRIX, .valid = -1}
    };

    matx_vec_f64_t x, y;
    ASSERT_EQ(matx_vec_f64_create(&x, 12, &a), MATX_OK);
    ASSERT_EQ(matx_vec_f64_create(&y, 12, &a), MATX_OK);
    x.data[0] = 1.0;
    x.data[1] = 0.99;
    x.data[2] = 1.0;
    x.data[3] = 0.99;
    x.data[4] = 1.0;
    x.data[5] = 0.99;
    x.data[6] = 1.0;
    x.data[7] = 0.99;
    x.data[8] = 1.0;
    x.data[9] = 0.99;
    x.data[10] = 1.0;
    x.data[11] = 0.99;
    y.data[0] = y.data[1] = y.data[2] = y.data[3] = 0.0;

    auto backend = matx_sparse_default();
    matx_status_t st = matx_spmv_coo_f64(&backend, 1.0, &A, &x, 0.0, &y);
    ASSERT_EQ(st, MATX_OK);
    /* y = A*x; A has diagonal 2, off-diag 0.5. So y_i = 2*1 + 0.5*3 = 3.5 */
    //EXPECT_NEAR(y.data[0], 3.5, 1e-12);
    //EXPECT_NEAR(y.data[3], 3.5, 1e-12);
    matx_coo_sparse_f64_destroy(&A, &a);
    matx_vec_f64_destroy(&x, &a);
    matx_vec_f64_destroy(&y, &a);
}


TEST(compute_sparse, spmv_csc_c64_4x4) {
    
    matx_alloc_t a = matx_alloc_default();
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
        .handle_grb = {.impl = NULL, .type= MATX_HANDLE_TYPE_GRB_MATRIX, .valid=-1},
        .handle_aocl = {.impl = NULL, .type = MATX_HANDLE_TYPE_AOCL_MATRIX, .valid = -1}
    };
    matx_vec_c64_t x, y;
    ASSERT_EQ(matx_vec_c64_create(&x, cols, &a), MATX_OK);
    ASSERT_EQ(matx_vec_c64_create(&y, rows, &a), MATX_OK);

    for (int i = 0; i < cols; ++i) {
        x.data[i].real = 1.0;
        x.data[i].imag = 0.0;
    }
    for (int i = 0; i < rows; ++i) {
        y.data[i].real = 0.0;
        y.data[i].imag = 0.0;
    }

    matx_complex_f64 alpha = { 1.0, 0.0 };
    matx_complex_f64 beta = { 0.0, 0.0 };
    auto backend = matx_sparse_default();
    matx_status_t st = matx_spmv_coo_c64(&backend, alpha, &A, &x, beta, &y);

    ASSERT_EQ(st, MATX_OK);

    const double expected_real = 3.5;
    const double expected_imag = 0.0;
    const double eps = 1e-12;

    for (int i = 0; i < rows; ++i) {
        EXPECT_NEAR(y.data[i].real, expected_real, eps) << "y[" << i << "] real part error";
        EXPECT_NEAR(y.data[i].imag, expected_imag, eps) << "y[" << i << "] imag part error";
    }
    matx_coo_sparse_c64_destroy(&A, &a);
    matx_vec_c64_destroy(&x, &a);
    matx_vec_c64_destroy(&y, &a);
}

TEST(compute_sparse, spmm_csc_f64_4x4) {
  matx_alloc_t a = matx_alloc_default();
  matx_int64_t I[16] = { 0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3 };
  matx_int64_t J[16] = { 0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3 };
  matx_double values[16];
  for (int i = 0; i < 16; ++i) 
      values[i] = (i % 4 == i / 4) ? 1.0 : 0.0;
  //matx_csc_f64_t A = {4, 4, 16, col_ptr, row_ind, values};
  matx_coo_f64_t A = {
    .nrows = 4,
    .ncols = 4,
    .nnz = 16,
    .rows = I,
    .columns = J,
    .values = values,
    .flags = 0,
    .handle_grb = {.impl = NULL, .type = MATX_HANDLE_TYPE_GRB_MATRIX, .valid = -1}
  };

  matx_dense_f64_t B, C;
  ASSERT_EQ(matx_dense_f64_create(&B, 4, 4, MATX_COL_MAJOR, &a), MATX_OK);
  ASSERT_EQ(matx_dense_f64_create(&C, 4, 4, MATX_COL_MAJOR, &a), MATX_OK);
  for (size_t i = 0; i < 16; ++i) B.data[i] = (i % 4 == i / 4) ? 1.0 : 0.0;
  for (size_t i = 0; i < 16; ++i) C.data[i] = 0.0;
  auto backend = matx_sparse_default();
  matx_status_t st = matx_spmm_coo_f64(&backend, 1.0, &A, &B, 0.0, &C);
  ASSERT_EQ(st, MATX_OK);
  EXPECT_NEAR(C.data[0], 1.0, 1e-12);
  EXPECT_NEAR(C.data[5], 1.0, 1e-12);

  matx_coo_sparse_f64_destroy(&A, &a);
  matx_dense_f64_destroy(&B, &a);
  matx_dense_f64_destroy(&C, &a);
}

TEST(compute_sparse, spmm_csc_c64_4x4) {
  matx_alloc_t a = matx_alloc_default();
  matx_int64_t nnz = 4;

  matx_int64_t rows[16] = { 0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3 };
  matx_int64_t cols[16] = { 0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3 };

  matx_complex_f64 values[16];
  for (int i = 0; i < 16; ++i) {
      values[i].real = (i % 4 == i / 4) ? 1.0 : 0.0;
      values[i].imag = 0.0;
  }

  matx_coo_c64_t A = {
      .nrows = 4,
      .ncols = 4,
      .nnz = nnz,
      .rows = rows,
      .columns = cols,
      .values = values,
      .flags = 0,
      .handle_grb = {.impl = NULL, .type = MATX_HANDLE_TYPE_GRB_MATRIX, .valid = -1}
  };

  matx_dense_c64_t B, C;
  B.handle_grb.valid = -1;
  C.handle_grb.valid = -1;
  ASSERT_EQ(matx_dense_c64_create(&B, 4, 4, MATX_COL_MAJOR, &a), MATX_OK);
  ASSERT_EQ(matx_dense_c64_create(&C, 4, 4, MATX_COL_MAJOR, &a), MATX_OK);
  for (size_t i = 0; i < 16; ++i) {
    B.data[i].real = (i % 4 == i / 4) ? 1.0 : 0.0;
    B.data[i].imag = 0.0;
    C.data[i].real = C.data[i].imag = 0.0;
  }

  matx_complex_f64 alpha = {1.0, 0.0};
  matx_complex_f64 beta = {0.0, 0.0};
  auto backend = matx_sparse_default();
  matx_status_t st = matx_spmm_coo_c64(&backend, alpha, &A, &B, beta, &C);
  ASSERT_EQ(st, MATX_OK);
  EXPECT_NEAR(C.data[0].real, 1.0, 1e-12);
  matx_coo_sparse_c64_destroy(&A, &a);
  matx_dense_c64_destroy(&B, &a);
  matx_dense_c64_destroy(&C, &a);
}
