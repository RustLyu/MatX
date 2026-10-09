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