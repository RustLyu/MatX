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

// ---- Phase 4: min/max/clip (vec) ----

TEST(core_vec, vec_d_i8_min_max_clip)
{
    matx_alloc_t a = matx_alloc_default();
    matx_vec_d_i8_t v1 = NULL, v2 = NULL, out = NULL;
    ASSERT_EQ(matx_vec_d_i8_create(&a, &v1, NULL, 3), MATX_OK);
    ASSERT_EQ(matx_vec_d_i8_create(&a, &v2, NULL, 3), MATX_OK);
    v1->data[0] = 1.0; v1->data[1] = 5.0; v1->data[2] = -2.0;
    v2->data[0] = 3.0; v2->data[1] = 2.0; v2->data[2] =  4.0;

    ASSERT_EQ(matx_vec_d_i8_min(&a, v1, v2, &out), MATX_OK);
    EXPECT_NEAR(out->data[0], 1.0, 1e-12);
    EXPECT_NEAR(out->data[1], 2.0, 1e-12);
    EXPECT_NEAR(out->data[2], -2.0, 1e-12);
    matx_vec_d_i8_destroy(&a, out);
    out = NULL;

    ASSERT_EQ(matx_vec_d_i8_max(&a, v1, v2, &out), MATX_OK);
    EXPECT_NEAR(out->data[0], 3.0, 1e-12);
    EXPECT_NEAR(out->data[1], 5.0, 1e-12);
    EXPECT_NEAR(out->data[2], 4.0, 1e-12);
    matx_vec_d_i8_destroy(&a, out);
    out = NULL;

    ASSERT_EQ(matx_vec_d_i8_clip(&a, v1, 0.0, 4.0, &out), MATX_OK);
    EXPECT_NEAR(out->data[0], 1.0, 1e-12);
    EXPECT_NEAR(out->data[1], 4.0, 1e-12);
    EXPECT_NEAR(out->data[2], 0.0, 1e-12);
    matx_vec_d_i8_destroy(&a, out);

    matx_vec_d_i8_destroy(&a, v1);
    matx_vec_d_i8_destroy(&a, v2);
}

// ---- Phase 4: Complex conjugate (vec) ----

TEST(core_vec, vec_z_i8_conj)
{
    matx_alloc_t a = matx_alloc_default();
    matx_vec_z_i8_t v = NULL, out = NULL;
    ASSERT_EQ(matx_vec_z_i8_create(&a, &v, NULL, 3), MATX_OK);
    v->data[0] = {1.0, 2.0};
    v->data[1] = {-3.0, 4.0};
    v->data[2] = {0.0, -5.0};

    ASSERT_EQ(matx_vec_z_i8_conj(&a, v, &out), MATX_OK);
    EXPECT_NEAR(out->data[0].real, 1.0, 1e-12);
    EXPECT_NEAR(out->data[0].imag, -2.0, 1e-12);
    EXPECT_NEAR(out->data[1].real, -3.0, 1e-12);
    EXPECT_NEAR(out->data[1].imag, -4.0, 1e-12);
    EXPECT_NEAR(out->data[2].real, 0.0, 1e-12);
    EXPECT_NEAR(out->data[2].imag, 5.0, 1e-12);
    matx_vec_z_i8_destroy(&a, out);

    ASSERT_EQ(matx_vec_z_i8_conj_inplace(v), MATX_OK);
    EXPECT_NEAR(v->data[0].real, 1.0, 1e-12);
    EXPECT_NEAR(v->data[0].imag, -2.0, 1e-12);

    matx_vec_z_i8_destroy(&a, v);
}

// ---- Phase 4: real/imag extraction (vec) ----

TEST(core_vec, vec_z_i8_real_imag)
{
    matx_alloc_t a = matx_alloc_default();
    matx_vec_z_i8_t v = NULL;
    ASSERT_EQ(matx_vec_z_i8_create(&a, &v, NULL, 2), MATX_OK);
    v->data[0] = {2.5, -1.5};
    v->data[1] = {0.0, 3.0};

    matx_vec_d_i8_t real_out = NULL;
    ASSERT_EQ(matx_vec_z_i8_real(&a, v, &real_out), MATX_OK);
    EXPECT_NEAR(real_out->data[0], 2.5, 1e-12);
    EXPECT_NEAR(real_out->data[1], 0.0, 1e-12);
    matx_vec_d_i8_destroy(&a, real_out);

    matx_vec_d_i8_t imag_out = NULL;
    ASSERT_EQ(matx_vec_z_i8_imag(&a, v, &imag_out), MATX_OK);
    EXPECT_NEAR(imag_out->data[0], -1.5, 1e-12);
    EXPECT_NEAR(imag_out->data[1], 3.0, 1e-12);
    matx_vec_d_i8_destroy(&a, imag_out);

    matx_vec_z_i8_destroy(&a, v);
}

// ---- Phase 4: cumsum z / linspace / logspace / complex rand ----

TEST(core_vec, vec_z_i8_cumsum)
{
    matx_alloc_t a = matx_alloc_default();
    matx_vec_z_i8_t v = NULL, out = NULL;
    ASSERT_EQ(matx_vec_z_i8_create(&a, &v, NULL, 3), MATX_OK);
    v->data[0] = {1.0, 2.0};
    v->data[1] = {3.0, 4.0};
    v->data[2] = {5.0, 6.0};

    ASSERT_EQ(matx_vec_z_i8_cumsum(&a, v, &out), MATX_OK);
    EXPECT_NEAR(out->data[0].real, 1.0, 1e-12);
    EXPECT_NEAR(out->data[0].imag, 2.0, 1e-12);
    EXPECT_NEAR(out->data[1].real, 4.0, 1e-12);
    EXPECT_NEAR(out->data[1].imag, 6.0, 1e-12);
    EXPECT_NEAR(out->data[2].real, 9.0, 1e-12);
    EXPECT_NEAR(out->data[2].imag, 12.0, 1e-12);

    matx_vec_z_i8_destroy(&a, out);
    matx_vec_z_i8_destroy(&a, v);
}

TEST(core_vec, linspace_logspace)
{
    matx_alloc_t a = matx_alloc_default();
    matx_vec_d_i8_t out = NULL;

    ASSERT_EQ(matx_vec_d_i8_linspace(&a, 0.0, 1.0, 5, &out), MATX_OK);
    EXPECT_EQ(out->n, 5);
    EXPECT_NEAR(out->data[0], 0.0, 1e-12);
    EXPECT_NEAR(out->data[1], 0.25, 1e-12);
    EXPECT_NEAR(out->data[2], 0.5, 1e-12);
    EXPECT_NEAR(out->data[3], 0.75, 1e-12);
    EXPECT_NEAR(out->data[4], 1.0, 1e-12);
    matx_vec_d_i8_destroy(&a, out);

    ASSERT_EQ(matx_vec_d_i8_logspace(&a, 0.0, 2.0, 3, &out), MATX_OK);
    EXPECT_EQ(out->n, 3);
    EXPECT_NEAR(out->data[0], 1.0, 1e-12);          // 10^0
    EXPECT_NEAR(out->data[1], 10.0, 1e-12);         // 10^1
    EXPECT_NEAR(out->data[2], 100.0, 1e-12);        // 10^2
    matx_vec_d_i8_destroy(&a, out);
}

TEST(core_vec, rand_uniform_z_i8)
{
    matx_alloc_t a = matx_alloc_default();
    matx_vec_z_i8_t v = NULL;
    ASSERT_EQ(matx_vec_z_i8_create(&a, &v, NULL, 4), MATX_OK);

    ASSERT_EQ(matx_vec_z_i8_rand_uniform(v, 0.0, 1.0, -1.0, 0.0, 42), MATX_OK);
    for (int i = 0; i < 4; ++i) {
        EXPECT_TRUE(v->data[i].real >= 0.0);
        EXPECT_TRUE(v->data[i].real <= 1.0);
        EXPECT_TRUE(v->data[i].imag >= -1.0);
        EXPECT_TRUE(v->data[i].imag <= 0.0);
    }

    matx_vec_z_i8_destroy(&a, v);
}
