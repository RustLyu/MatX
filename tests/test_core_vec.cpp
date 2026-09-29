#include "matx_test_harness.h"

extern "C" {
#include "matx/matx_func.h"
#include "matx/matx_types.h"
#include "matx/matx_types_internal.h"
}

TEST(core_vec, vec_d_i8_create_destroy)
{
    matx_alloc_t a = matx_alloc_default();
    matx_vec_d_i8_t v = NULL;
    ASSERT_EQ(matx_vec_d_i8_create(&a, &v, NULL, 4), MATX_OK);
    ASSERT_NE(v->data, nullptr);
    EXPECT_EQ(v->n, 4u);
    EXPECT_EQ(v->stride, 1u);
    matx_vec_d_i8_destroy(&a, v);
}

TEST(core_vec, vec_d_i8_wrap)
{
    matx_alloc_t a = matx_alloc_default();
    double buf[4] = {1.0, 2.0, 3.0, 4.0};
    matx_vec_d_i8_t v = NULL;
    ASSERT_EQ(matx_vec_d_i8_wrap(&a, &v, 4, 1, buf), MATX_OK);
    EXPECT_EQ(v->data, buf);
    EXPECT_EQ(v->n, 4u);
    matx_vec_d_i8_destroy(nullptr, v);
}

TEST(core_vec, vec_z_i8_create_destroy)
{
    matx_alloc_t a = matx_alloc_default();
    matx_vec_z_i8_t v = NULL;
    ASSERT_EQ(matx_vec_z_i8_create(&a, &v, NULL, 4), MATX_OK);
    ASSERT_NE(v->data, nullptr);
    EXPECT_EQ(v->n, 4u);
    matx_vec_z_i8_destroy(&a, v);
    //EXPECT_EQ(v->data, nullptr);
}

TEST(core_vec, vec_z_i8_wrap)
{
    matx_alloc_t a = matx_alloc_default();
    matx_complex_d_t buf[4] = {{1.0, 0.0}, {0.0, 1.0}, {2.0, -1.0}, {0.0, 0.0}};
    matx_vec_z_i8_t v = NULL;
    ASSERT_EQ(matx_vec_z_i8_wrap(&a, &v, 4, 1, buf), MATX_OK);
    EXPECT_EQ(v->data, buf);
    matx_vec_z_i8_destroy(nullptr, v);
}

TEST(core_vec, vec_d_i8_fill_zeros_ones)
{
    matx_alloc_t a = matx_alloc_default();
    matx_vec_d_i8_t v = NULL;
    ASSERT_EQ(matx_vec_d_i8_create(&a, &v, NULL, 4), MATX_OK);
    ASSERT_EQ(matx_vec_d_i8_fill(v, 7.0), MATX_OK);
    EXPECT_NEAR(v->data[0], 7.0, 1e-12);
    EXPECT_NEAR(v->data[3], 7.0, 1e-12);
    ASSERT_EQ(matx_vec_d_i8_zeros(v), MATX_OK);
    EXPECT_NEAR(v->data[0], 0.0, 1e-12);
    ASSERT_EQ(matx_vec_d_i8_ones(v), MATX_OK);
    EXPECT_NEAR(v->data[2], 1.0, 1e-12);
    matx_vec_d_i8_destroy(&a, v);
}

TEST(core_vec, vec_z_i8_fill_zeros)
{
    matx_alloc_t a = matx_alloc_default();
    matx_vec_z_i8_t v = NULL;
    ASSERT_EQ(matx_vec_z_i8_create(&a, &v, NULL, 4), MATX_OK);
    matx_complex_d_t val = {7.0, 3.0};
    ASSERT_EQ(matx_vec_z_i8_fill(v, val), MATX_OK);
    EXPECT_NEAR(v->data[0].real, 7.0, 1e-12);
    EXPECT_NEAR(v->data[0].imag, 3.0, 1e-12);
    EXPECT_NEAR(v->data[3].real, 7.0, 1e-12);
    EXPECT_NEAR(v->data[3].imag, 3.0, 1e-12);
    ASSERT_EQ(matx_vec_z_i8_zeros(v), MATX_OK);
    EXPECT_NEAR(v->data[0].real, 0.0, 1e-12);
    EXPECT_NEAR(v->data[0].imag, 0.0, 1e-12);
    matx_vec_z_i8_destroy(&a, v);
}
