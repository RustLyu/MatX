#include "matx_test_harness.h"
#include <math.h>

extern "C" {
#include "matx/matx_func.h"
#include "matx/matx_log.h"
#include "matx/matx_types.h"
#include "matx/matx_types_internal.h"
}

/* ======== Zero-copy submatrix views ======== */

TEST(core_submatrix, dense_slice_col_major)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t M = NULL;
    double data[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    ASSERT_EQ(matx_dense_d_i8_create(&a, &M, MATX_COL_MAJOR, 4, 3, data), MATX_OK);

    /* slice rows 1-3, cols 1-2: should give [[6,7],[7,8],[8,9]]? No...
     * Original: 4x3 col-major: data = [1,2,3,4, 5,6,7,8, 9,10,11,12]
     * A(1,1) = 6, A(1,2) = 10 -> slice (1, 3, 1, 2) yields [6,7],[10,11] */
    matx_dense_d_i8_t V = NULL;
    ASSERT_EQ(matx_dense_d_i8_slice(&a, M, 1, 3, 1, 2, &V), MATX_OK);
    EXPECT_EQ(V->nrows, 2);
    EXPECT_EQ(V->ncols, 1);
    EXPECT_EQ(V->stride, 4);
    EXPECT_NEAR(V->data[0], 6.0, 1e-14);
    EXPECT_NEAR(V->data[1], 7.0, 1e-14);
    /* Modify through the view */
    V->data[0] = 99.0;
    EXPECT_NEAR(M->data[1 + 1 * M->stride], 99.0, 1e-14);

    matx_dense_d_i8_destroy(&a, V);
    matx_dense_d_i8_destroy(&a, M);
}

TEST(core_submatrix, dense_slice_row_major)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t M = NULL;
    double data[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    ASSERT_EQ(matx_dense_d_i8_create(&a, &M, MATX_ROW_MAJOR, 3, 4, data), MATX_OK);

    /* 3x4 row-major: [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
     * slice rows 0-2, cols 1-3 -> [[2,3],[6,7]] */
    matx_dense_d_i8_t V = NULL;
    ASSERT_EQ(matx_dense_d_i8_slice(&a, M, 0, 2, 1, 3, &V), MATX_OK);
    EXPECT_EQ(V->nrows, 2);
    EXPECT_EQ(V->ncols, 2);
    EXPECT_EQ(V->stride, 4);
    /* view->data points to M->data[0*4+1] = M->data[1] = 2 */
    EXPECT_NEAR(V->data[0], 2.0, 1e-14); /* row=0,col=0 -> M[0][1] = 2 */
    EXPECT_NEAR(V->data[1], 3.0, 1e-14); /* row=0,col=1 -> M[0][2] = 3 */
    EXPECT_NEAR(V->data[4], 6.0, 1e-14); /* row=1,col=0 -> M[1][1] = 6 */
    EXPECT_NEAR(V->data[5], 7.0, 1e-14); /* row=1,col=1 -> M[1][2] = 7 */

    matx_dense_d_i8_destroy(&a, V);
    matx_dense_d_i8_destroy(&a, M);
}

TEST(core_submatrix, vec_slice)
{
    matx_alloc_t a = matx_alloc_default();
    matx_vec_d_i8_t v = NULL;
    double data[5] = {10, 20, 30, 40, 50};
    ASSERT_EQ(matx_vec_d_i8_create(&a, &v, data, 5), MATX_OK);

    matx_vec_d_i8_t s = NULL;
    ASSERT_EQ(matx_vec_d_i8_slice(&a, v, 1, 3, &s), MATX_OK);
    EXPECT_EQ(s->n, 3);
    EXPECT_NEAR(s->data[0], 20.0, 1e-14);
    EXPECT_NEAR(s->data[2], 40.0, 1e-14);
    /* Modify through view */
    s->data[1] = 999.0;
    EXPECT_NEAR(v->data[2], 999.0, 1e-14);

    matx_vec_d_i8_destroy(&a, s);
    matx_vec_d_i8_destroy(&a, v);
}

TEST(core_submatrix, complex_dense_slice)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_z_i8_t M = NULL;
    ASSERT_EQ(matx_dense_z_i8_create(&a, &M, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);
    M->data[0] = {1, 2};
    M->data[1] = {3, 4};
    M->data[2] = {5, 6};
    M->data[3] = {7, 8};

    matx_dense_z_i8_t V = NULL;
    ASSERT_EQ(matx_dense_z_i8_slice(&a, M, 1, 2, 1, 2, &V), MATX_OK);
    EXPECT_EQ(V->nrows, 1);
    EXPECT_EQ(V->ncols, 1);
    EXPECT_NEAR(V->data[0].real, 7.0, 1e-14);
    EXPECT_NEAR(V->data[0].imag, 8.0, 1e-14);

    matx_dense_z_i8_destroy(&a, V);
    matx_dense_z_i8_destroy(&a, M);
}

/* ======== Row / column extraction ======== */

TEST(core_submatrix, get_row_col_major)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t M = NULL;
    double data[6] = {1, 2, 3, 4, 5, 6};
    ASSERT_EQ(matx_dense_d_i8_create(&a, &M, MATX_COL_MAJOR, 3, 2, data), MATX_OK);

    matx_vec_d_i8_t row = NULL;
    ASSERT_EQ(matx_dense_d_i8_get_row(&a, M, 1, &row), MATX_OK);
    EXPECT_EQ(row->n, 2);
    EXPECT_NEAR(row->data[0], 2.0, 1e-14);
    EXPECT_NEAR(row->data[1], 5.0, 1e-14);
    matx_vec_d_i8_destroy(&a, row);
    matx_dense_d_i8_destroy(&a, M);
}

TEST(core_submatrix, get_col_row_major)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t M = NULL;
    double data[6] = {1, 2, 3, 4, 5, 6};
    ASSERT_EQ(matx_dense_d_i8_create(&a, &M, MATX_ROW_MAJOR, 2, 3, data), MATX_OK);

    matx_vec_d_i8_t col = NULL;
    ASSERT_EQ(matx_dense_d_i8_get_col(&a, M, 1, &col), MATX_OK);
    EXPECT_EQ(col->n, 2);
    EXPECT_NEAR(col->data[0], 2.0, 1e-14);
    EXPECT_NEAR(col->data[1], 5.0, 1e-14);
    matx_vec_d_i8_destroy(&a, col);
    matx_dense_d_i8_destroy(&a, M);
}

TEST(core_submatrix, set_row_and_verify)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t M = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &M, MATX_COL_MAJOR, 3, 2, NULL), MATX_OK);
    matx_dense_d_i8_fill(M, 0.0);

    matx_vec_d_i8_t r = NULL;
    double row_data[2] = {10, 20};
    ASSERT_EQ(matx_vec_d_i8_create(&a, &r, row_data, 2), MATX_OK);
    ASSERT_EQ(matx_dense_d_i8_set_row(M, 1, r), MATX_OK);
    EXPECT_NEAR(M->data[1 + 0 * M->stride], 10.0, 1e-14);
    EXPECT_NEAR(M->data[1 + 1 * M->stride], 20.0, 1e-14);

    matx_vec_d_i8_destroy(&a, r);
    matx_dense_d_i8_destroy(&a, M);
}

TEST(core_submatrix, set_col_and_verify)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t M = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &M, MATX_ROW_MAJOR, 2, 3, NULL), MATX_OK);
    matx_dense_d_i8_fill(M, 0.0);

    matx_vec_d_i8_t c = NULL;
    double col_data[2] = {100, 200};
    ASSERT_EQ(matx_vec_d_i8_create(&a, &c, col_data, 2), MATX_OK);
    ASSERT_EQ(matx_dense_d_i8_set_col(M, 2, c), MATX_OK);
    EXPECT_NEAR(M->data[0 * M->stride + 2], 100.0, 1e-14);
    EXPECT_NEAR(M->data[1 * M->stride + 2], 200.0, 1e-14);

    matx_vec_d_i8_destroy(&a, c);
    matx_dense_d_i8_destroy(&a, M);
}

/* ======== Multiple row / column extraction ======== */

TEST(core_submatrix, get_rows)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t M = NULL;
    double data[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    ASSERT_EQ(matx_dense_d_i8_create(&a, &M, MATX_COL_MAJOR, 4, 3, data), MATX_OK);

    matx_dense_d_i8_t out = NULL;
    ASSERT_EQ(matx_dense_d_i8_get_rows(&a, M, 1, 3, &out), MATX_OK);
    EXPECT_EQ(out->nrows, 2);
    EXPECT_EQ(out->ncols, 3);
    EXPECT_NEAR(out->data[0 + 0 * out->stride], 2.0, 1e-14);
    EXPECT_NEAR(out->data[1 + 0 * out->stride], 3.0, 1e-14);
    EXPECT_NEAR(out->data[1 + 2 * out->stride], 11.0, 1e-14);

    matx_dense_d_i8_destroy(&a, out);
    matx_dense_d_i8_destroy(&a, M);
}

TEST(core_submatrix, get_cols)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t M = NULL;
    double data[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    ASSERT_EQ(matx_dense_d_i8_create(&a, &M, MATX_COL_MAJOR, 3, 3, data), MATX_OK);

    matx_dense_d_i8_t out = NULL;
    ASSERT_EQ(matx_dense_d_i8_get_cols(&a, M, 1, 3, &out), MATX_OK);
    EXPECT_EQ(out->nrows, 3);
    EXPECT_EQ(out->ncols, 2);
    EXPECT_NEAR(out->data[0 + 0 * out->stride], 4.0, 1e-14);
    EXPECT_NEAR(out->data[2 + 1 * out->stride], 9.0, 1e-14);

    matx_dense_d_i8_destroy(&a, out);
    matx_dense_d_i8_destroy(&a, M);
}

/* ======== Triangular extraction ======== */

TEST(core_submatrix, tril)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t M = NULL;
    double data[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    ASSERT_EQ(matx_dense_d_i8_create(&a, &M, MATX_COL_MAJOR, 3, 3, data), MATX_OK);

    matx_dense_d_i8_t L = NULL;
    ASSERT_EQ(matx_dense_d_i8_tril(&a, M, &L), MATX_OK);
    /* 3x3 col-major: [1 4 7; 2 5 8; 3 6 9] -> tril [1 0 0; 2 5 0; 3 6 9] */
    EXPECT_NEAR(L->data[0], 1.0, 1e-14); /* (0,0) diag */
    EXPECT_NEAR(L->data[1], 2.0, 1e-14); /* (1,0) lower */
    EXPECT_NEAR(L->data[2], 3.0, 1e-14); /* (2,0) lower */
    EXPECT_NEAR(L->data[3], 0.0, 1e-14); /* (0,1) upper zeroed */
    EXPECT_NEAR(L->data[4], 5.0, 1e-14); /* (1,1) diag */
    EXPECT_NEAR(L->data[5], 6.0, 1e-14); /* (2,1) lower */
    EXPECT_NEAR(L->data[6], 0.0, 1e-14); /* (0,2) upper zeroed */
    EXPECT_NEAR(L->data[7], 0.0, 1e-14); /* (1,2) upper zeroed */
    EXPECT_NEAR(L->data[8], 9.0, 1e-14); /* (2,2) diag */

    matx_dense_d_i8_destroy(&a, L);
    matx_dense_d_i8_destroy(&a, M);
}

TEST(core_submatrix, triu)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t M = NULL;
    double data[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    ASSERT_EQ(matx_dense_d_i8_create(&a, &M, MATX_COL_MAJOR, 3, 3, data), MATX_OK);

    matx_dense_d_i8_t U = NULL;
    ASSERT_EQ(matx_dense_d_i8_triu(&a, M, &U), MATX_OK);
    EXPECT_NEAR(U->data[0 + 0 * U->stride], 1.0, 1e-14);
    EXPECT_NEAR(U->data[0 + 1 * U->stride], 4.0, 1e-14);
    EXPECT_NEAR(U->data[2 + 2 * U->stride], 9.0, 1e-14);
    EXPECT_NEAR(U->data[1 + 0 * U->stride], 0.0, 1e-14); /* lower part zeroed */

    matx_dense_d_i8_destroy(&a, U);
    matx_dense_d_i8_destroy(&a, M);
}

/* ======== Complex get_row / get_col ======== */

TEST(core_submatrix, complex_get_row_col)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_z_i8_t M = NULL;
    ASSERT_EQ(matx_dense_z_i8_create(&a, &M, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);
    M->data[0] = {1, 2};
    M->data[1] = {3, 4};
    M->data[2] = {5, 6};
    M->data[3] = {7, 8};

    matx_vec_z_i8_t row = NULL;
    ASSERT_EQ(matx_dense_z_i8_get_row(&a, M, 0, &row), MATX_OK);
    EXPECT_NEAR(row->data[0].real, 1.0, 1e-14);
    EXPECT_NEAR(row->data[0].imag, 2.0, 1e-14);
    EXPECT_NEAR(row->data[1].real, 5.0, 1e-14);
    matx_vec_z_i8_destroy(&a, row);

    matx_vec_z_i8_t col = NULL;
    ASSERT_EQ(matx_dense_z_i8_get_col(&a, M, 1, &col), MATX_OK);
    EXPECT_NEAR(col->data[0].real, 5.0, 1e-14);
    EXPECT_NEAR(col->data[1].real, 7.0, 1e-14);
    matx_vec_z_i8_destroy(&a, col);

    matx_dense_z_i8_destroy(&a, M);
}

/* ======== Complex set_row / set_col ======== */

TEST(core_submatrix, complex_set_row_col)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_z_i8_t M = NULL;
    ASSERT_EQ(matx_dense_z_i8_create(&a, &M, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);
    matx_dense_z_i8_zeros(M);

    matx_vec_z_i8_t r = NULL;
    matx_complex_d_t rdata[2] = {{10, 1}, {20, 2}};
    ASSERT_EQ(matx_vec_z_i8_create(&a, &r, rdata, 2), MATX_OK);
    ASSERT_EQ(matx_dense_z_i8_set_row(M, 1, r), MATX_OK);
    EXPECT_NEAR(M->data[1 + 0 * M->stride].real, 10.0, 1e-14);
    EXPECT_NEAR(M->data[1 + 1 * M->stride].real, 20.0, 1e-14);
    matx_vec_z_i8_destroy(&a, r);

    matx_vec_z_i8_t c = NULL;
    matx_complex_d_t cdata[2] = {{100, 3}, {200, 4}};
    ASSERT_EQ(matx_vec_z_i8_create(&a, &c, cdata, 2), MATX_OK);
    ASSERT_EQ(matx_dense_z_i8_set_col(M, 0, c), MATX_OK);
    EXPECT_NEAR(M->data[0 + 0 * M->stride].real, 100.0, 1e-14);
    EXPECT_NEAR(M->data[1 + 0 * M->stride].real, 200.0, 1e-14);
    matx_vec_z_i8_destroy(&a, c);

    matx_dense_z_i8_destroy(&a, M);
}

/* ======== Element-wise comparisons (dense) ======== */

TEST(core_comparison, dense_eq_complex)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_z_i8_t A = NULL, B = NULL;
    ASSERT_EQ(matx_dense_z_i8_create(&a, &A, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);
    ASSERT_EQ(matx_dense_z_i8_create(&a, &B, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);
    A->data[0] = {1, 2}; A->data[1] = {3, 4};
    A->data[2] = {5, 6}; A->data[3] = {7, 8};
    B->data[0] = {1, 2}; B->data[1] = {3, 5};
    B->data[2] = {5, 6}; B->data[3] = {0, 0};

    matx_dense_d_i8_t mask = NULL;
    ASSERT_EQ(matx_dense_z_i8_eq(&a, A, B, &mask), MATX_OK);
    EXPECT_NEAR(mask->data[0], 1.0, 1e-14); /* equals */
    EXPECT_NEAR(mask->data[1], 0.0, 1e-14); /* not equals */
    EXPECT_NEAR(mask->data[2], 1.0, 1e-14); /* equals */
    EXPECT_NEAR(mask->data[3], 0.0, 1e-14); /* not equals */
    matx_dense_d_i8_destroy(&a, mask);
    mask = NULL;

    ASSERT_EQ(matx_dense_z_i8_ne(&a, A, B, &mask), MATX_OK);
    EXPECT_NEAR(mask->data[0], 0.0, 1e-14);
    EXPECT_NEAR(mask->data[1], 1.0, 1e-14);
    matx_dense_d_i8_destroy(&a, mask);
    mask = NULL;

    ASSERT_EQ(matx_dense_z_i8_lt(&a, A, B, &mask), MATX_OK);
    EXPECT_NEAR(mask->data[0], 0.0, 1e-14); /* equal, not less */
    EXPECT_NEAR(mask->data[1], 1.0, 1e-14); /* |3+4i| < |3+5i| */
    matx_dense_d_i8_destroy(&a, mask);
    mask = NULL;

    matx_dense_z_i8_destroy(&a, B);
    matx_dense_z_i8_destroy(&a, A);
}

TEST(core_comparison, dense_real_all_comparisons)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t A = NULL, B = NULL;
    double adata[6] = {1, 2, 3, 4, 5, 6};
    double bdata[6] = {1, 0, 5, 4, 2, 6};
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_COL_MAJOR, 3, 2, adata), MATX_OK);
    ASSERT_EQ(matx_dense_d_i8_create(&a, &B, MATX_COL_MAJOR, 3, 2, bdata), MATX_OK);

    matx_dense_d_i8_t mask = NULL;

    ASSERT_EQ(matx_dense_d_i8_eq(&a, A, B, &mask), MATX_OK);
    EXPECT_NEAR(mask->data[0], 1.0, 1e-14);
    EXPECT_NEAR(mask->data[3], 1.0, 1e-14);
    EXPECT_NEAR(mask->data[5], 1.0, 1e-14);
    matx_dense_d_i8_destroy(&a, mask);
    mask = NULL;

    ASSERT_EQ(matx_dense_d_i8_lt(&a, A, B, &mask), MATX_OK);
    EXPECT_NEAR(mask->data[0], 0.0, 1e-14);
    EXPECT_NEAR(mask->data[1], 0.0, 1e-14); /* 2 > 0 */
    EXPECT_NEAR(mask->data[2], 1.0, 1e-14); /* 3 < 5 */
    matx_dense_d_i8_destroy(&a, mask);
    mask = NULL;

    ASSERT_EQ(matx_dense_d_i8_gt(&a, A, B, &mask), MATX_OK);
    EXPECT_NEAR(mask->data[1], 1.0, 1e-14); /* 2 > 0 */
    EXPECT_NEAR(mask->data[2], 0.0, 1e-14); /* 3 < 5 */
    matx_dense_d_i8_destroy(&a, mask);
    mask = NULL;

    ASSERT_EQ(matx_dense_d_i8_le(&a, A, B, &mask), MATX_OK);
    EXPECT_NEAR(mask->data[0], 1.0, 1e-14);
    EXPECT_NEAR(mask->data[2], 1.0, 1e-14);
    matx_dense_d_i8_destroy(&a, mask);
    mask = NULL;

    ASSERT_EQ(matx_dense_d_i8_ge(&a, A, B, &mask), MATX_OK);
    EXPECT_NEAR(mask->data[1], 1.0, 1e-14);
    EXPECT_NEAR(mask->data[5], 1.0, 1e-14);
    matx_dense_d_i8_destroy(&a, mask);
    mask = NULL;

    matx_dense_d_i8_destroy(&a, B);
    matx_dense_d_i8_destroy(&a, A);
}

/* ======== Element-wise comparisons (vector) ======== */

TEST(core_comparison, vec_comparisons)
{
    matx_alloc_t a = matx_alloc_default();
    matx_vec_d_i8_t x = NULL, y = NULL;
    double xdata[4] = {1, 2, 3, 4};
    double ydata[4] = {1, 0, 5, 4};
    ASSERT_EQ(matx_vec_d_i8_create(&a, &x, xdata, 4), MATX_OK);
    ASSERT_EQ(matx_vec_d_i8_create(&a, &y, ydata, 4), MATX_OK);

    matx_vec_d_i8_t mask = NULL;

    ASSERT_EQ(matx_vec_d_i8_eq(&a, x, y, &mask), MATX_OK);
    EXPECT_NEAR(mask->data[0], 1.0, 1e-14);
    EXPECT_NEAR(mask->data[3], 1.0, 1e-14);
    matx_vec_d_i8_destroy(&a, mask);
    mask = NULL;

    ASSERT_EQ(matx_vec_d_i8_ne(&a, x, y, &mask), MATX_OK);
    EXPECT_NEAR(mask->data[1], 1.0, 1e-14);
    EXPECT_NEAR(mask->data[2], 1.0, 1e-14);
    matx_vec_d_i8_destroy(&a, mask);
    mask = NULL;

    ASSERT_EQ(matx_vec_d_i8_lt(&a, x, y, &mask), MATX_OK);
    EXPECT_NEAR(mask->data[0], 0.0, 1e-14);
    EXPECT_NEAR(mask->data[2], 1.0, 1e-14);
    matx_vec_d_i8_destroy(&a, mask);
    mask = NULL;

    matx_vec_d_i8_destroy(&a, y);
    matx_vec_d_i8_destroy(&a, x);
}

/* ======== vec comparison complex ======== */

TEST(core_comparison, vec_complex_compare)
{
    matx_alloc_t a = matx_alloc_default();
    matx_vec_z_i8_t x = NULL, y = NULL;
    ASSERT_EQ(matx_vec_z_i8_create(&a, &x, NULL, 3), MATX_OK);
    ASSERT_EQ(matx_vec_z_i8_create(&a, &y, NULL, 3), MATX_OK);
    x->data[0] = {1, 0}; x->data[1] = {2, 2}; x->data[2] = {0, 5};
    y->data[0] = {1, 0}; y->data[1] = {1, 1}; y->data[2] = {5, 0};

    matx_vec_d_i8_t mask = NULL;
    ASSERT_EQ(matx_vec_z_i8_eq(&a, x, y, &mask), MATX_OK);
    EXPECT_NEAR(mask->data[0], 1.0, 1e-14);
    EXPECT_NEAR(mask->data[1], 0.0, 1e-14); /* |2+2i|!=|1+1i| */
    EXPECT_NEAR(mask->data[2], 1.0, 1e-14); /* |0+5i|==|5+0i| */
    matx_vec_d_i8_destroy(&a, mask);
    mask = NULL;

    ASSERT_EQ(matx_vec_z_i8_gt(&a, x, y, &mask), MATX_OK);
    EXPECT_NEAR(mask->data[1], 1.0, 1e-14); /* |2+2i| > |1+1i| */
    matx_vec_d_i8_destroy(&a, mask);
    mask = NULL;

    matx_vec_z_i8_destroy(&a, y);
    matx_vec_z_i8_destroy(&a, x);
}

/* ======== Dense normal random ======== */

TEST(core_random, dense_rand_normal_basic)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t M = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &M, MATX_COL_MAJOR, 100, 1, NULL), MATX_OK);
    ASSERT_EQ(matx_dense_d_i8_rand_normal(M, 0.0, 1.0, 42), MATX_OK);

    /* check that values are not all zero and within reasonable range */
    double sum = 0.0;
    for (matx_int64_t i = 0; i < 100; ++i) {
        EXPECT_TRUE(fabs(M->data[i]) < 6.0); /* 6 sigma is extremely unlikely */
        sum += M->data[i];
    }
    double mean = sum / 100.0;
    EXPECT_TRUE(fabs(mean) < 1.0); /* rough check */

    matx_dense_d_i8_destroy(&a, M);
}

TEST(core_random, dense_rand_normal_repeatable)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t A = NULL, B = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_COL_MAJOR, 10, 1, NULL), MATX_OK);
    ASSERT_EQ(matx_dense_d_i8_create(&a, &B, MATX_COL_MAJOR, 10, 1, NULL), MATX_OK);

    ASSERT_EQ(matx_dense_d_i8_rand_normal(A, 0.0, 1.0, 12345), MATX_OK);
    ASSERT_EQ(matx_dense_d_i8_rand_normal(B, 0.0, 1.0, 12345), MATX_OK);
    for (int i = 0; i < 10; ++i)
        EXPECT_NEAR(A->data[i], B->data[i], 0.0);

    matx_dense_d_i8_destroy(&a, B);
    matx_dense_d_i8_destroy(&a, A);
}

TEST(core_random, dense_rand_normal_complex)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_z_i8_t M = NULL;
    ASSERT_EQ(matx_dense_z_i8_create(&a, &M, MATX_COL_MAJOR, 50, 1, NULL), MATX_OK);
    ASSERT_EQ(matx_dense_z_i8_rand_normal(M, 5.0, -3.0, 1.0, 77), MATX_OK);

    double sum_re = 0.0, sum_im = 0.0;
    for (matx_int64_t i = 0; i < 50; ++i) {
        sum_re += M->data[i].real;
        sum_im += M->data[i].imag;
    }
    double mean_re = sum_re / 50.0;
    double mean_im = sum_im / 50.0;
    EXPECT_TRUE(fabs(mean_re - 5.0) < 2.0);
    EXPECT_TRUE(fabs(mean_im - (-3.0)) < 2.0);

    matx_dense_z_i8_destroy(&a, M);
}

/* ======== Dense comparisons error cases ======== */

TEST(core_comparison, dense_compare_dimension_mismatch)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t A = NULL, B = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);
    ASSERT_EQ(matx_dense_d_i8_create(&a, &B, MATX_COL_MAJOR, 3, 2, NULL), MATX_OK);

    matx_dense_d_i8_t mask = NULL;
    EXPECT_NE(matx_dense_d_i8_eq(&a, A, B, &mask), MATX_OK);
    /* mask should remain NULL on failure */
    EXPECT_TRUE(mask == NULL);

    matx_dense_d_i8_destroy(&a, B);
    matx_dense_d_i8_destroy(&a, A);
}