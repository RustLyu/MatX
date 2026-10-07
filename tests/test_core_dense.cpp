#include "matx_test_harness.h"
#include <math.h>

extern "C" {
#include "matx/matx_func.h"
#include "matx/matx_log.h"
#include "matx/matx_types.h"
#include "matx/matx_types_internal.h"
}

TEST(core_dense, create_destroy)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t M = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &M, MATX_COL_MAJOR, 3, 4, NULL), MATX_OK);
    ASSERT_NE(M->data, nullptr);
    EXPECT_EQ(M->nrows, 3u);
    EXPECT_EQ(M->ncols, 4u);
    EXPECT_EQ(M->stride, 3u);
    matx_dense_d_i8_destroy(&a, M);
}

TEST(core_dense, wrap_no_ownership)
{
    double buf[6] = {0};
    matx_dense_d_i8_t V = NULL;
    matx_alloc_t a = matx_alloc_default();
    ASSERT_EQ(matx_dense_d_i8_wrap(&a, &V, 2, 3, 3, MATX_COL_MAJOR, buf), MATX_OK);
    EXPECT_EQ(V->data, buf);
    matx_dense_d_i8_destroy(&a, V);
}

TEST(core_dense, row_major_padded_wrap_dup_and_abs)
{
    matx_alloc_t alloc = matx_alloc_default();
    double backing[10] = {1.0, -2.0, 3.0, 90.0, 91.0,
                          -4.0, 5.0, -6.0, 92.0, 93.0};
    matx_dense_d_i8_t wrapped = NULL, copy = NULL, magnitude = NULL;
    ASSERT_EQ(matx_dense_d_i8_wrap(&alloc, &wrapped, 2, 3, 5,
                                   MATX_ROW_MAJOR, backing), MATX_OK);
    ASSERT_EQ(matx_dense_d_i8_dup(&alloc, wrapped, &copy), MATX_OK);
    ASSERT_EQ(copy->stride, 3);
    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(copy->data[i], i < 3 ? backing[i] : backing[i + 2], 1e-14);

    ASSERT_EQ(matx_dense_d_i8_abs(&alloc, wrapped, &magnitude), MATX_OK);
    const double expected[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(magnitude->data[i], expected[i], 1e-14);

    matx_dense_d_i8_destroy(&alloc, magnitude);
    matx_dense_d_i8_destroy(&alloc, copy);
    matx_dense_d_i8_destroy(&alloc, wrapped);
}

TEST(core_dense, create_4x4)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t M = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &M, MATX_COL_MAJOR, 4, 4, NULL), MATX_OK);
    ASSERT_NE(M->data, nullptr);
    EXPECT_EQ(M->nrows, 4u);
    EXPECT_EQ(M->ncols, 4u);
    EXPECT_EQ(M->stride, 4u);
    matx_dense_d_i8_destroy(&a, M);
}

TEST(core_dense, dense_z_i8_create_4x4)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_z_i8_t M = NULL;
    ASSERT_EQ(matx_dense_z_i8_create(&a, &M, MATX_COL_MAJOR, 4, 4, NULL), MATX_OK);
    ASSERT_NE(M->data, nullptr);
    EXPECT_EQ(M->nrows, 4u);
    EXPECT_EQ(M->ncols, 4u);
    matx_dense_z_i8_destroy(&a, M);
}

TEST(core_dense, dup_f4x4)
{
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

TEST(core_dense, dup_c4x4)
{
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

TEST(core_dense, f64_fill_zeros_ones)
{
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

TEST(core_dense, uniform_random_seeded_fill_is_repeatable_and_respects_stride)
{
    matx_alloc_t alloc = matx_alloc_default();
    double backing[10];
    for (double& value : backing) value = -99.0;
    matx_dense_d_i8_t matrix = NULL;
    ASSERT_EQ(matx_dense_d_i8_wrap(&alloc, &matrix, 2, 3, 5,
                                   MATX_ROW_MAJOR, backing), MATX_OK);
    matx_vec_d_i8_t vector = NULL;
    ASSERT_EQ(matx_vec_d_i8_create(&alloc, &vector, NULL, 6), MATX_OK);

    ASSERT_EQ(matx_dense_d_i8_rand_uniform(matrix, -1.0, 1.0, 12345), MATX_OK);
    double first_fill[6];
    for (int row = 0; row < 2; ++row) {
        for (int col = 0; col < 3; ++col) {
            const double value = backing[row * 5 + col];
            EXPECT_TRUE(value >= -1.0 && value < 1.0);
            first_fill[row * 3 + col] = value;
        }
    }
    EXPECT_EQ(backing[3], -99.0);
    EXPECT_EQ(backing[4], -99.0);
    EXPECT_EQ(backing[8], -99.0);
    EXPECT_EQ(backing[9], -99.0);

    ASSERT_EQ(matx_dense_d_i8_rand_uniform(matrix, -1.0, 1.0, 12345), MATX_OK);
    ASSERT_EQ(matx_vec_d_i8_rand_uniform(vector, -1.0, 1.0, 12345), MATX_OK);
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(backing[(i / 3) * 5 + i % 3], first_fill[i], 0.0);
        EXPECT_NEAR(vector->data[i * vector->stride], first_fill[i], 0.0);
    }

    matx_vec_d_i8_destroy(&alloc, vector);
    matx_dense_d_i8_destroy(&alloc, matrix);
}

TEST(core_dense, f64_trace)
{
    matx_alloc_t a = matx_alloc_default();
    double data[4] = {1, 0, 0, 4}; // 2x2 col-major: diag = {1,4}
    matx_dense_d_i8_t A = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_COL_MAJOR, 2, 2, data), MATX_OK);
    double tr = 0.0;
    ASSERT_EQ(matx_dense_d_i8_trace(A, &tr), MATX_OK);
    EXPECT_NEAR(tr, 5.0, 1e-12);
    matx_dense_d_i8_destroy(&a, A);
}

TEST(core_dense, dense_z_i8_fill_zeros)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_z_i8_t M = NULL;
    ASSERT_EQ(matx_dense_z_i8_create(&a, &M, MATX_COL_MAJOR, 3, 3, NULL), MATX_OK);
    matx_complex_d_t val = {5.0, 2.0};
    ASSERT_EQ(matx_dense_z_i8_fill(M, val), MATX_OK);
    EXPECT_NEAR(M->data[0].real, 5.0, 1e-12);
    EXPECT_NEAR(M->data[0].imag, 2.0, 1e-12);
    ASSERT_EQ(matx_dense_z_i8_zeros(M), MATX_OK);
    EXPECT_NEAR(M->data[4].real, 0.0, 1e-12);
    EXPECT_NEAR(M->data[4].imag, 0.0, 1e-12);
    matx_dense_z_i8_destroy(&a, M);
}

TEST(core_dense, dense_z_i8_trace)
{
    matx_alloc_t a = matx_alloc_default();
    matx_complex_d_t data[4] = {{1, 0}, {0, 0}, {0, 0}, {4, 0}};
    matx_dense_z_i8_t A = NULL;
    ASSERT_EQ(matx_dense_z_i8_create(&a, &A, MATX_COL_MAJOR, 2, 2, data), MATX_OK);
    matx_complex_d_t tr = {0, 0};
    ASSERT_EQ(matx_dense_z_i8_trace(A, &tr), MATX_OK);
    EXPECT_NEAR(tr.real, 5.0, 1e-12);
    EXPECT_NEAR(tr.imag, 0.0, 1e-12);
    matx_dense_z_i8_destroy(&a, A);
}

TEST(core_dense, f64_get_block_col_major)
{
    matx_alloc_t a = matx_alloc_default();
    /* 4x4 col-major: data[i + j*4] */
    double data[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    matx_dense_d_i8_t A = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_COL_MAJOR, 4, 4, data), MATX_OK);

    /* extract A[1:3, 1:3] = [ [6,7], [10,11] ] */
    matx_dense_d_i8_t B = NULL;
    ASSERT_EQ(matx_dense_d_i8_get_block(&a, A, 1, 3, 1, 3, &B), MATX_OK);
    EXPECT_EQ(B->nrows, 2u);
    EXPECT_EQ(B->ncols, 2u);
    EXPECT_EQ(B->layout, MATX_COL_MAJOR);
    EXPECT_NEAR(B->data[0 + 0 * B->stride], 6.0, 1e-12);
    EXPECT_NEAR(B->data[1 + 0 * B->stride], 7.0, 1e-12);
    EXPECT_NEAR(B->data[0 + 1 * B->stride], 10.0, 1e-12);
    EXPECT_NEAR(B->data[1 + 1 * B->stride], 11.0, 1e-12);

    matx_dense_d_i8_destroy(&a, A);
    matx_dense_d_i8_destroy(&a, B);
}

TEST(core_dense, z64_get_block_col_major)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_z_i8_t A = NULL;
    ASSERT_EQ(matx_dense_z_i8_create(&a, &A, MATX_COL_MAJOR, 3, 3, NULL), MATX_OK);
    for (matx_int64_t i = 0; i < 9; ++i) {
        A->data[i].real = (matx_double) (i + 1);
        A->data[i].imag = (matx_double) (i + 1) * 0.1;
    }

    /* extract A[0:2, 0:2] = top-left 2x2 */
    matx_dense_z_i8_t B = NULL;
    ASSERT_EQ(matx_dense_z_i8_get_block(&a, A, 0, 2, 0, 2, &B), MATX_OK);
    EXPECT_EQ(B->nrows, 2u);
    EXPECT_EQ(B->ncols, 2u);
    EXPECT_NEAR(B->data[0 + 0 * B->stride].real, 1.0, 1e-12);
    EXPECT_NEAR(B->data[0 + 0 * B->stride].imag, 0.1, 1e-12);
    EXPECT_NEAR(B->data[1 + 1 * B->stride].real, 5.0, 1e-12);

    matx_dense_z_i8_destroy(&a, A);
    matx_dense_z_i8_destroy(&a, B);
}

TEST(core_dense, f64_get_block_row_major)
{
    matx_alloc_t a = matx_alloc_default();
    /* 3x3 row-major: data[i*3 + j] */
    double data[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    matx_dense_d_i8_t A = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_ROW_MAJOR, 3, 3, data), MATX_OK);

    /* extract A[0:2, 1:3] = [ [2,3], [5,6] ] */
    matx_dense_d_i8_t B = NULL;
    ASSERT_EQ(matx_dense_d_i8_get_block(&a, A, 0, 2, 1, 3, &B), MATX_OK);
    EXPECT_EQ(B->nrows, 2u);
    EXPECT_EQ(B->ncols, 2u);
    EXPECT_EQ(B->layout, MATX_ROW_MAJOR);
    EXPECT_NEAR(B->data[0 * B->stride + 0], 2.0, 1e-12);
    EXPECT_NEAR(B->data[0 * B->stride + 1], 3.0, 1e-12);
    EXPECT_NEAR(B->data[1 * B->stride + 0], 5.0, 1e-12);
    EXPECT_NEAR(B->data[1 * B->stride + 1], 6.0, 1e-12);

    matx_dense_d_i8_destroy(&a, A);
    matx_dense_d_i8_destroy(&a, B);
}

TEST(core_dense, get_block_invalid)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t A = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_COL_MAJOR, 4, 4, NULL), MATX_OK);

    matx_dense_d_i8_t B = NULL;
    EXPECT_NE(matx_dense_d_i8_get_block(&a, A, -1, 2, 0, 2, &B), MATX_OK);
    EXPECT_NE(matx_dense_d_i8_get_block(&a, A, 2, 1, 0, 2, &B), MATX_OK);
    EXPECT_NE(matx_dense_d_i8_get_block(&a, A, 0, 5, 0, 2, &B), MATX_OK);

    matx_dense_d_i8_destroy(&a, A);
}

TEST(core_dense, f64_set_block_col_major)
{
    matx_alloc_t a = matx_alloc_default();
    /* 3x3 col-major src, fill with 1..9 */
    matx_dense_d_i8_t A = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_COL_MAJOR, 3, 3, NULL), MATX_OK);
    for (matx_int64_t i = 0; i < 9; ++i)
        A->data[i] = (matx_double) (i + 1);

    /* 4x4 col-major dst, zeroed */
    matx_dense_d_i8_t B = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &B, MATX_COL_MAJOR, 4, 4, NULL), MATX_OK);
    ASSERT_EQ(matx_dense_d_i8_zeros(B), MATX_OK);

    /* copy A[0:2, 0:2] = [1,2;4,5] into B at (1,1) */
    ASSERT_EQ(matx_dense_d_i8_set_block(A, 0, 2, 0, 2, B, 1, 1), MATX_OK);

    EXPECT_NEAR(B->data[1 + 1 * B->stride], 1.0, 1e-12); // B(1,1)
    EXPECT_NEAR(B->data[2 + 1 * B->stride], 2.0, 1e-12); // B(2,1)
    EXPECT_NEAR(B->data[1 + 2 * B->stride], 4.0, 1e-12); // B(1,2)
    EXPECT_NEAR(B->data[2 + 2 * B->stride], 5.0, 1e-12); // B(2,2)
    EXPECT_NEAR(B->data[0 + 0 * B->stride], 0.0, 1e-12); // untouched

    matx_dense_d_i8_destroy(&a, A);
    matx_dense_d_i8_destroy(&a, B);
}

TEST(core_dense, z64_set_block_col_major)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_z_i8_t A = NULL;
    ASSERT_EQ(matx_dense_z_i8_create(&a, &A, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);
    A->data[0].real = 1.0;
    A->data[0].imag = 0.5;
    A->data[1].real = 2.0;
    A->data[1].imag = 0.0;
    A->data[2].real = 3.0;
    A->data[2].imag = -0.5;
    A->data[3].real = 4.0;
    A->data[3].imag = 0.0;

    matx_dense_z_i8_t B = NULL;
    ASSERT_EQ(matx_dense_z_i8_create(&a, &B, MATX_COL_MAJOR, 3, 3, NULL), MATX_OK);
    ASSERT_EQ(matx_dense_z_i8_zeros(B), MATX_OK);

    /* copy entire A into B at (0,0) */
    ASSERT_EQ(matx_dense_z_i8_set_block(A, 0, 2, 0, 2, B, 0, 0), MATX_OK);
    EXPECT_NEAR(B->data[0 + 0 * B->stride].real, 1.0, 1e-12);
    EXPECT_NEAR(B->data[0 + 0 * B->stride].imag, 0.5, 1e-12);
    EXPECT_NEAR(B->data[1 + 1 * B->stride].real, 4.0, 1e-12);
    EXPECT_NEAR(B->data[1 + 1 * B->stride].imag, 0.0, 1e-12);

    matx_dense_z_i8_destroy(&a, A);
    matx_dense_z_i8_destroy(&a, B);
}

TEST(core_dense, set_block_out_of_bounds)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t A = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_COL_MAJOR, 3, 3, NULL), MATX_OK);

    matx_dense_d_i8_t B = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &B, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);

    /* A[0:3, 0:3] cannot fit in B at (0,0) since B is only 2x2 */
    EXPECT_NE(matx_dense_d_i8_set_block(A, 0, 3, 0, 3, B, 0, 0), MATX_OK);

    matx_dense_d_i8_destroy(&a, A);
    matx_dense_d_i8_destroy(&a, B);
}

TEST(core_dense, set_block_layout_mismatch)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t A = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);

    matx_dense_d_i8_t B = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &B, MATX_ROW_MAJOR, 4, 4, NULL), MATX_OK);

    EXPECT_NE(matx_dense_d_i8_set_block(A, 0, 2, 0, 2, B, 0, 0), MATX_OK);

    matx_dense_d_i8_destroy(&a, A);
    matx_dense_d_i8_destroy(&a, B);
}
