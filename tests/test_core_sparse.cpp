#include "matx_test_harness.h"

extern "C" {
#include "matx/matx_func.h"
#include "matx/matx_log.h"
#include "matx/matx_types.h"
#include "matx/matx_types_internal.h"
}

/* ---- COO real (d_i8) ---- */

TEST(core_sparse, coo_d_i8_create_destroy)
{
    matx_alloc_t a = matx_alloc_default();
    matx_int64_t rows[3] = {0, 1, 2};
    matx_int64_t cols[3] = {0, 1, 2};
    matx_double vals[3] = {1.0, 2.0, 3.0};
    matx_coo_d_i8_t A = NULL;
    ASSERT_EQ(matx_coo_sparse_d_i8_create(&a, &A, 3, 3, 3, rows, cols, vals), MATX_OK);
    ASSERT_NE(A, nullptr);
    EXPECT_EQ(A->nrows, 3u);
    EXPECT_EQ(A->ncols, 3u);
    EXPECT_EQ(A->nnz, 3u);
    EXPECT_NE(A->rows, nullptr);
    EXPECT_NE(A->columns, nullptr);
    EXPECT_NE(A->values, nullptr);
    EXPECT_NE(A->rows, rows); /* deep copy, not alias */
    EXPECT_NE(A->columns, cols);
    EXPECT_NE(A->values, vals);
    EXPECT_EQ(A->rows[0], 0);
    EXPECT_EQ(A->columns[1], 1);
    EXPECT_NEAR(A->values[2], 3.0, 1e-12);
    EXPECT_NE(A->flags & 1u, 0u); /* ownership flag set */
    matx_coo_sparse_d_i8_destroy(&a, A);
}

TEST(core_sparse, coo_d_i8_create_null_arrays)
{
    matx_alloc_t a = matx_alloc_default();
    matx_coo_d_i8_t A = NULL;
    ASSERT_EQ(matx_coo_sparse_d_i8_create(&a, &A, 2, 2, 4, NULL, NULL, NULL), MATX_OK);
    ASSERT_NE(A, nullptr);
    EXPECT_EQ(A->nrows, 2u);
    EXPECT_EQ(A->ncols, 2u);
    EXPECT_EQ(A->nnz, 4u);
    EXPECT_NE(A->rows, nullptr);
    EXPECT_NE(A->columns, nullptr);
    EXPECT_NE(A->values, nullptr);
    matx_coo_sparse_d_i8_destroy(&a, A);
}

TEST(core_sparse, coo_d_i8_create_invalid_args)
{
    matx_alloc_t a = matx_alloc_default();
    matx_coo_d_i8_t A = NULL;
    EXPECT_NE(matx_coo_sparse_d_i8_create(&a, NULL, 3, 3, 3, NULL, NULL, NULL), MATX_OK);
    EXPECT_NE(matx_coo_sparse_d_i8_create(NULL, &A, 3, 3, 3, NULL, NULL, NULL), MATX_OK);
    EXPECT_NE(matx_coo_sparse_d_i8_create(&a, &A, 0, 3, 3, NULL, NULL, NULL), MATX_OK);
    EXPECT_NE(matx_coo_sparse_d_i8_create(&a, &A, 3, 0, 3, NULL, NULL, NULL), MATX_OK);
    EXPECT_NE(matx_coo_sparse_d_i8_create(&a, &A, 3, 3, 0, NULL, NULL, NULL), MATX_OK);
}

TEST(core_sparse, coo_d_i8_dup)
{
    matx_alloc_t a = matx_alloc_default();
    matx_int64_t rows[2] = {0, 1};
    matx_int64_t cols[2] = {1, 0};
    matx_double vals[2] = {5.0, 7.0};
    matx_coo_d_i8_t A = NULL;
    ASSERT_EQ(matx_coo_sparse_d_i8_create(&a, &A, 2, 2, 2, rows, cols, vals), MATX_OK);

    matx_coo_d_i8_t B = NULL;
    ASSERT_EQ(matx_coo_d_i8_dup(&a, A, &B), MATX_OK);
    ASSERT_NE(B, nullptr);
    EXPECT_EQ(B->nrows, 2u);
    EXPECT_EQ(B->ncols, 2u);
    EXPECT_EQ(B->nnz, 2u);
    EXPECT_NE(B->rows, A->rows); /* deep copy */
    EXPECT_NE(B->columns, A->columns);
    EXPECT_NE(B->values, A->values);
    EXPECT_EQ(B->rows[0], 0);
    EXPECT_EQ(B->columns[0], 1);
    EXPECT_NEAR(B->values[0], 5.0, 1e-12);
    EXPECT_EQ(B->rows[1], 1);
    EXPECT_EQ(B->columns[1], 0);
    EXPECT_NEAR(B->values[1], 7.0, 1e-12);

    matx_coo_sparse_d_i8_destroy(&a, A);
    matx_coo_sparse_d_i8_destroy(&a, B);
}

TEST(core_sparse, coo_d_i8_destroy_null_safe)
{
    matx_alloc_t a = matx_alloc_default();
    matx_coo_sparse_d_i8_destroy(&a, NULL); /* must not crash */
    matx_coo_sparse_d_i8_destroy(NULL, NULL);
}

/* ---- COO complex (z_i8) ---- */

TEST(core_sparse, coo_z_i8_create_destroy)
{
    matx_alloc_t a = matx_alloc_default();
    matx_int64_t rows[2] = {0, 1};
    matx_int64_t cols[2] = {0, 1};
    matx_complex_d_t vals[2] = {{1.0, 0.5}, {2.0, -0.5}};
    matx_coo_z_i8_t A = NULL;
    ASSERT_EQ(matx_coo_sparse_z_i8_create(&a, &A, 2, 2, 2, rows, cols, vals), MATX_OK);
    ASSERT_NE(A, nullptr);
    EXPECT_EQ(A->nrows, 2u);
    EXPECT_EQ(A->ncols, 2u);
    EXPECT_EQ(A->nnz, 2u);
    EXPECT_NE(A->rows, rows);
    EXPECT_NE(A->columns, cols);
    EXPECT_NE(A->values, vals);
    EXPECT_NEAR(A->values[0].real, 1.0, 1e-12);
    EXPECT_NEAR(A->values[0].imag, 0.5, 1e-12);
    EXPECT_NEAR(A->values[1].real, 2.0, 1e-12);
    EXPECT_NEAR(A->values[1].imag, -0.5, 1e-12);
    matx_coo_sparse_z_i8_destroy(&a, A);
}

TEST(core_sparse, coo_z_i8_dup)
{
    matx_alloc_t a = matx_alloc_default();
    matx_int64_t rows[2] = {0, 1};
    matx_int64_t cols[2] = {0, 1};
    matx_complex_d_t vals[2] = {{3.0, 1.0}, {4.0, 2.0}};
    matx_coo_z_i8_t A = NULL;
    ASSERT_EQ(matx_coo_sparse_z_i8_create(&a, &A, 2, 2, 2, rows, cols, vals), MATX_OK);

    matx_coo_z_i8_t B = NULL;
    ASSERT_EQ(matx_coo_z_i8_dup(&a, A, &B), MATX_OK);
    ASSERT_NE(B, nullptr);
    EXPECT_EQ(B->nnz, 2u);
    EXPECT_NE(B->rows, A->rows);
    EXPECT_NE(B->values, A->values);
    EXPECT_NEAR(B->values[0].real, 3.0, 1e-12);
    EXPECT_NEAR(B->values[0].imag, 1.0, 1e-12);
    EXPECT_NEAR(B->values[1].real, 4.0, 1e-12);
    EXPECT_NEAR(B->values[1].imag, 2.0, 1e-12);

    matx_coo_sparse_z_i8_destroy(&a, A);
    matx_coo_sparse_z_i8_destroy(&a, B);
}

TEST(core_sparse, coo_z_i8_wrap)
{
    matx_alloc_t a = matx_alloc_default();
    matx_int64_t rows[3] = {0, 1, 1};
    matx_int64_t cols[3] = {0, 0, 1};
    matx_complex_d_t vals[3] = {{1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}};
    matx_coo_z_i8_t A = NULL;
    ASSERT_EQ(matx_coo_sparse_z_i8_wrap(&a, &A, 2, 2, 3, rows, cols, vals), MATX_OK);
    ASSERT_NE(A, nullptr);
    EXPECT_EQ(A->nrows, 2u);
    EXPECT_EQ(A->ncols, 2u);
    EXPECT_EQ(A->nnz, 3u);
    EXPECT_EQ(A->rows, rows); /* wrap: no copy */
    EXPECT_EQ(A->columns, cols);
    EXPECT_EQ(A->values, vals);
    EXPECT_EQ(A->flags & 1u, 0u); /* no ownership */
    matx_coo_sparse_z_i8_destroy(&a, A);
}

/* ---- CSC real (d_i8) ---- */

TEST(core_sparse, csc_d_i8_create_destroy)
{
    matx_alloc_t a = matx_alloc_default();
    matx_csc_d_i8_t A = NULL;
    ASSERT_EQ(matx_csc_sparse_d_i8_create(&a, &A, 3, 3, 5), MATX_OK);
    ASSERT_NE(A, nullptr);
    EXPECT_EQ(A->nrows, 3u);
    EXPECT_EQ(A->ncols, 3u);
    EXPECT_EQ(A->nnz, 5u);
    EXPECT_NE(A->col_ptr, nullptr);
    EXPECT_NE(A->row_ind, nullptr);
    EXPECT_NE(A->values, nullptr);
    EXPECT_NE(A->coo_csc_index_map, nullptr);
    EXPECT_NE(A->flags & 1u, 0u);
    matx_csc_sparse_d_i8_destroy(&a, A);
}

TEST(core_sparse, csc_d_i8_create_invalid_args)
{
    matx_alloc_t a = matx_alloc_default();
    matx_csc_d_i8_t A = NULL;
    EXPECT_NE(matx_csc_sparse_d_i8_create(&a, NULL, 3, 3, 5), MATX_OK);
    EXPECT_NE(matx_csc_sparse_d_i8_create(NULL, &A, 3, 3, 5), MATX_OK);
    EXPECT_NE(matx_csc_sparse_d_i8_create(&a, &A, 0, 3, 5), MATX_OK);
    EXPECT_NE(matx_csc_sparse_d_i8_create(&a, &A, 3, 0, 5), MATX_OK);
    EXPECT_NE(matx_csc_sparse_d_i8_create(&a, &A, 3, 3, 0), MATX_OK);
}

TEST(core_sparse, csc_d_i8_wrap)
{
    matx_alloc_t a = matx_alloc_default();
    matx_int64_t col_ptr[4] = {0, 2, 3, 4};
    matx_int64_t row_ind[4] = {0, 1, 0, 1};
    matx_double vals[4] = {1.0, 2.0, 3.0, 4.0};
    matx_csc_d_i8_t A = NULL;
    ASSERT_EQ(matx_csc_sparse_d_i8_wrap(&a, &A, 2, 3, 4, col_ptr, row_ind, vals), MATX_OK);
    ASSERT_NE(A, nullptr);
    EXPECT_EQ(A->nrows, 2u);
    EXPECT_EQ(A->ncols, 3u);
    EXPECT_EQ(A->nnz, 4u);
    EXPECT_EQ(A->col_ptr, col_ptr); /* wrap: no copy */
    EXPECT_EQ(A->row_ind, row_ind);
    EXPECT_EQ(A->values, vals);
    EXPECT_EQ(A->flags & 1u, 0u); /* no ownership */
    matx_csc_sparse_d_i8_destroy(&a, A);
}

TEST(core_sparse, csc_d_i8_destroy_null_safe)
{
    matx_alloc_t a = matx_alloc_default();
    matx_csc_sparse_d_i8_destroy(&a, NULL); /* must not crash */
    matx_csc_sparse_d_i8_destroy(NULL, NULL);
}

/* ---- CSC complex (z_i8) ---- */

TEST(core_sparse, csc_z_i8_create_destroy)
{
    matx_alloc_t a = matx_alloc_default();
    matx_csc_z_i8_t A = NULL;
    ASSERT_EQ(matx_csc_sparse_z_i8_create(&a, &A, 4, 4, 10), MATX_OK);
    ASSERT_NE(A, nullptr);
    EXPECT_EQ(A->nrows, 4u);
    EXPECT_EQ(A->ncols, 4u);
    EXPECT_EQ(A->nnz, 10u);
    EXPECT_NE(A->col_ptr, nullptr);
    EXPECT_NE(A->row_ind, nullptr);
    EXPECT_NE(A->values, nullptr);
    matx_csc_sparse_z_i8_destroy(&a, A);
}

TEST(core_sparse, csc_z_i8_destroy_null_safe)
{
    matx_alloc_t a = matx_alloc_default();
    matx_csc_sparse_z_i8_destroy(&a, NULL);
    matx_csc_sparse_z_i8_destroy(NULL, NULL);
}