#include "matx_test_harness.h"

extern "C" {
#include "matx/matx_func.h"
#include "matx/matx_read.h"
#include "matx/matx_types.h"
#include "matx/matx_types_internal.h"
#include "matx/matx_write.h"
}

// ---- Phase 5: CSV write and read-back (round-trip) ----

TEST(io_csv, write_read_csv_dense_real_3x2)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t A = NULL;
    double vals[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_COL_MAJOR, 3, 2, vals), MATX_OK);
    matx_write_csv_d_i8(A, "csv_test_real.csv");

    matx_dense_d_i8_t B = NULL;
    ASSERT_EQ(matx_read_csv_d_i8(&a, &B, "csv_test_real.csv"), MATX_OK);
    EXPECT_EQ(B->nrows, 3);
    EXPECT_EQ(B->ncols, 2);
    for (size_t i = 0; i < 6; ++i)
        EXPECT_NEAR(B->data[i], vals[i], 1e-12);
    matx_dense_d_i8_destroy(&a, B);
    matx_dense_d_i8_destroy(&a, A);
}

TEST(io_csv, write_read_csv_dense_complex_2x2)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_z_i8_t A = NULL;
    ASSERT_EQ(matx_dense_z_i8_create(&a, &A, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);
    A->data[0] = {1.5, -2.5};
    A->data[1] = {3.0, 4.0};
    A->data[2] = {-1.0, 0.0};
    A->data[3] = {2.0, -3.0};
    matx_write_csv_z_i8(A, "csv_test_complex.csv");

    matx_dense_z_i8_t B = NULL;
    ASSERT_EQ(matx_read_csv_z_i8(&a, &B, "csv_test_complex.csv"), MATX_OK);
    EXPECT_EQ(B->nrows, 2);
    EXPECT_EQ(B->ncols, 2);
    EXPECT_NEAR(B->data[0].real, 1.5, 1e-12);
    EXPECT_NEAR(B->data[0].imag, -2.5, 1e-12);
    EXPECT_NEAR(B->data[3].real, 2.0, 1e-12);
    EXPECT_NEAR(B->data[3].imag, -3.0, 1e-12);
    matx_dense_z_i8_destroy(&a, B);
    matx_dense_z_i8_destroy(&a, A);
}

// ---- CSV vector write and read-back (round-trip) ----

TEST(io_csv, write_read_csv_vec_real)
{
    matx_alloc_t a = matx_alloc_default();
    matx_vec_d_i8_t v = NULL;
    double vals[5] = {1.0, 2.5, -3.0, 4.75, 5.125};
    ASSERT_EQ(matx_vec_d_i8_create(&a, &v, vals, 5), MATX_OK);
    matx_write_csv_vec_d_i8(v, "csv_vec_test_real.csv");

    matx_vec_d_i8_t w = NULL;
    ASSERT_EQ(matx_read_csv_vec_d_i8(&a, &w, "csv_vec_test_real.csv"), MATX_OK);
    EXPECT_EQ(w->n, 5);
    for (size_t i = 0; i < 5; ++i)
        EXPECT_NEAR(w->data[i], vals[i], 1e-12);
    matx_vec_d_i8_destroy(&a, w);
    matx_vec_d_i8_destroy(&a, v);
}

TEST(io_csv, write_read_csv_vec_complex)
{
    matx_alloc_t a = matx_alloc_default();
    matx_vec_z_i8_t v = NULL;
    ASSERT_EQ(matx_vec_z_i8_create(&a, &v, NULL, 4), MATX_OK);
    v->data[0] = {1.0, 2.0};
    v->data[1] = {-3.5, 4.5};
    v->data[2] = {0.0, -1.0};
    v->data[3] = {6.25, -7.75};
    matx_write_csv_vec_z_i8(v, "csv_vec_test_complex.csv");

    matx_vec_z_i8_t w = NULL;
    ASSERT_EQ(matx_read_csv_vec_z_i8(&a, &w, "csv_vec_test_complex.csv"), MATX_OK);
    EXPECT_EQ(w->n, 4);
    EXPECT_NEAR(w->data[0].real, 1.0, 1e-12);
    EXPECT_NEAR(w->data[0].imag, 2.0, 1e-12);
    EXPECT_NEAR(w->data[1].real, -3.5, 1e-12);
    EXPECT_NEAR(w->data[1].imag, 4.5, 1e-12);
    EXPECT_NEAR(w->data[2].real, 0.0, 1e-12);
    EXPECT_NEAR(w->data[2].imag, -1.0, 1e-12);
    EXPECT_NEAR(w->data[3].real, 6.25, 1e-12);
    EXPECT_NEAR(w->data[3].imag, -7.75, 1e-12);
    matx_vec_z_i8_destroy(&a, w);
    matx_vec_z_i8_destroy(&a, v);
}