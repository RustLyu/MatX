#include "matx_test_harness.h"

extern "C" {
#include "matx/matx_func.h"
#include "matx/matx_read.h"
#include "matx/matx_types.h"
#include "matx/matx_types_internal.h"
#include "matx/matx_write.h"
}

// ---- Phase 5: MTX write (round-trip: write then read back) ----

TEST(io_write, write_dense_real_2x2)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t A = NULL;
    double vals[4] = {1.0, 2.0, 3.0, 4.0};
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_COL_MAJOR, 2, 2, vals), MATX_OK);
    matx_write_dense_mtx_d_i8(A, "write_test_dense_real.mtx");

    matx_dense_d_i8_t B = NULL;
    ASSERT_EQ(matx_read_dense_mtx_d_i8(&a, &B, "write_test_dense_real.mtx"), MATX_OK);
    EXPECT_EQ(B->nrows, 2);
    EXPECT_EQ(B->ncols, 2);
    EXPECT_NEAR(B->data[0], 1.0, 1e-12);
    EXPECT_NEAR(B->data[1], 2.0, 1e-12);
    EXPECT_NEAR(B->data[2], 3.0, 1e-12);
    EXPECT_NEAR(B->data[3], 4.0, 1e-12);
    matx_dense_d_i8_destroy(&a, B);
    matx_dense_d_i8_destroy(&a, A);
}

TEST(io_write, write_dense_complex_2x2)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_z_i8_t A = NULL;
    ASSERT_EQ(matx_dense_z_i8_create(&a, &A, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);
    A->data[0] = {1.0, 2.0};
    A->data[1] = {3.0, 4.0};
    A->data[2] = {5.0, 6.0};
    A->data[3] = {7.0, 8.0};
    matx_write_dense_mtx_z_i8(A, "write_test_dense_complex.mtx");

    matx_dense_z_i8_t B = NULL;
    ASSERT_EQ(matx_read_dense_mtx_z_i8(&a, &B, "write_test_dense_complex.mtx"), MATX_OK);
    EXPECT_EQ(B->nrows, 2);
    EXPECT_EQ(B->ncols, 2);
    EXPECT_NEAR(B->data[0].real, 1.0, 1e-12);
    EXPECT_NEAR(B->data[0].imag, 2.0, 1e-12);
    EXPECT_NEAR(B->data[3].real, 7.0, 1e-12);
    EXPECT_NEAR(B->data[3].imag, 8.0, 1e-12);
    matx_dense_z_i8_destroy(&a, B);
    matx_dense_z_i8_destroy(&a, A);
}

TEST(io_write, write_sparse_real_2x2)
{
    matx_alloc_t a = matx_alloc_default();
    matx_int64_t rows[3] = {0, 1, 0};
    matx_int64_t cols[3] = {0, 0, 1};
    double vals[3] = {10.0, 20.0, 30.0};
    matx_coo_d_i8_t A = NULL;
    ASSERT_EQ(matx_coo_sparse_d_i8_create(&a, &A, 2, 2, 3, rows, cols, vals), MATX_OK);
    matx_write_sparse_mtx_d_i8(A, "write_test_sparse_real.mtx");

    matx_coo_d_i8_t B = NULL;
    ASSERT_EQ(matx_read_sparse_mtx_d_i8(&a, &B, "write_test_sparse_real.mtx"), MATX_OK);
    EXPECT_EQ(B->nrows, 2);
    EXPECT_EQ(B->ncols, 2);
    EXPECT_EQ(B->nnz, 3);
    matx_coo_sparse_d_i8_destroy(&a, B);
    matx_coo_sparse_d_i8_destroy(&a, A);
}

TEST(io_write, write_sparse_complex_2x2)
{
    matx_alloc_t a = matx_alloc_default();
    matx_int64_t rows[3] = {0, 1, 1};
    matx_int64_t cols[3] = {0, 0, 1};
    matx_complex_d_t vals[3] = {{1.0, 0.5}, {2.0, 1.0}, {3.0, -0.5}};
    matx_coo_z_i8_t A = NULL;
    ASSERT_EQ(matx_coo_sparse_z_i8_create(&a, &A, 2, 2, 3, rows, cols, vals), MATX_OK);
    matx_write_sparse_mtx_z_i8(A, "write_test_sparse_complex.mtx");

    matx_coo_z_i8_t B = NULL;
    ASSERT_EQ(matx_read_sparse_mtx_z_i8(&a, &B, "write_test_sparse_complex.mtx"), MATX_OK);
    EXPECT_EQ(B->nrows, 2);
    EXPECT_EQ(B->ncols, 2);
    EXPECT_EQ(B->nnz, 3);
    matx_coo_sparse_z_i8_destroy(&a, B);
    matx_coo_sparse_z_i8_destroy(&a, A);
}

TEST(io_write, write_vec_real)
{
    matx_alloc_t a = matx_alloc_default();
    matx_vec_d_i8_t v = NULL;
    ASSERT_EQ(matx_vec_d_i8_create(&a, &v, NULL, 3), MATX_OK);
    v->data[0] = 10.0;
    v->data[1] = 20.0;
    v->data[2] = 30.0;
    matx_write_vec_d_i8(v, "write_test_vec_real.mtx");

    matx_vec_d_i8_t w = NULL;
    ASSERT_EQ(matx_read_vec_d_i8(&a, &w, "write_test_vec_real.mtx"), MATX_OK);
    EXPECT_EQ(w->n, 3);
    EXPECT_NEAR(w->data[0], 10.0, 1e-12);
    EXPECT_NEAR(w->data[1], 20.0, 1e-12);
    EXPECT_NEAR(w->data[2], 30.0, 1e-12);
    matx_vec_d_i8_destroy(&a, w);
    matx_vec_d_i8_destroy(&a, v);
}

TEST(io_write, write_vec_complex)
{
    matx_alloc_t a = matx_alloc_default();
    matx_vec_z_i8_t v = NULL;
    ASSERT_EQ(matx_vec_z_i8_create(&a, &v, NULL, 2), MATX_OK);
    v->data[0] = {1.5, -2.0};
    v->data[1] = {3.0, 4.5};
    matx_write_vec_z_i8(v, "write_test_vec_complex.mtx");

    matx_vec_z_i8_t w = NULL;
    ASSERT_EQ(matx_read_vec_z_i8(&a, &w, "write_test_vec_complex.mtx"), MATX_OK);
    EXPECT_EQ(w->n, 2);
    EXPECT_NEAR(w->data[0].real, 1.5, 1e-12);
    EXPECT_NEAR(w->data[0].imag, -2.0, 1e-12);
    EXPECT_NEAR(w->data[1].real, 3.0, 1e-12);
    EXPECT_NEAR(w->data[1].imag, 4.5, 1e-12);
    matx_vec_z_i8_destroy(&a, w);
    matx_vec_z_i8_destroy(&a, v);
}