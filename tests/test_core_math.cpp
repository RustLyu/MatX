#define _USE_MATH_DEFINES
#include "matx_test_harness.h"
#include <math.h>

extern "C" {
#include "matx/matx_dense_compute.h"
#include "matx/matx_dense_solve.h"
#include "matx/matx_func.h"
#include "matx/matx_sparse_compute.h"
#include "matx/matx_sparse_solve.h"
#include "matx/matx_types.h"
#include "matx/matx_types_internal.h"
#include "matx/matx_vec_compute.h"
}

/* ============================================================
 *  Phase 1: Core element-wise math (vectors)
 * ============================================================ */

TEST(core_math, vec_exp_d_i8)
{
    matx_alloc_t a = matx_alloc_default();
    matx_vec_d_i8_t x = NULL, y = NULL;
    ASSERT_EQ(matx_vec_d_i8_create(&a, &x, NULL, 3), MATX_OK);
    x->data[0] = 0.0;
    x->data[1] = 1.0;
    x->data[2] = 2.0;
    matx_status_t st = matx_vec_d_i8_exp(&a, x, &y);
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(y->data[0], 1.0, 1e-12);
    EXPECT_NEAR(y->data[1], exp(1.0), 1e-12);
    EXPECT_NEAR(y->data[2], exp(2.0), 1e-12);
    matx_vec_d_i8_destroy(&a, x);
    matx_vec_d_i8_destroy(&a, y);
}

TEST(core_math, vec_log_d_i8)
{
    matx_alloc_t a = matx_alloc_default();
    matx_vec_d_i8_t x = NULL, y = NULL;
    ASSERT_EQ(matx_vec_d_i8_create(&a, &x, NULL, 3), MATX_OK);
    x->data[0] = 1.0;
    x->data[1] = 2.0;
    x->data[2] = 4.0;
    matx_status_t st = matx_vec_d_i8_log(&a, x, &y);
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(y->data[0], 0.0, 1e-12);
    EXPECT_NEAR(y->data[1], log(2.0), 1e-12);
    EXPECT_NEAR(y->data[2], log(4.0), 1e-12);
    matx_vec_d_i8_destroy(&a, x);
    matx_vec_d_i8_destroy(&a, y);
}

TEST(core_math, vec_sqrt_d_i8)
{
    matx_alloc_t a = matx_alloc_default();
    matx_vec_d_i8_t x = NULL, y = NULL;
    ASSERT_EQ(matx_vec_d_i8_create(&a, &x, NULL, 3), MATX_OK);
    x->data[0] = 0.0;
    x->data[1] = 4.0;
    x->data[2] = 9.0;
    matx_status_t st = matx_vec_d_i8_sqrt(&a, x, &y);
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(y->data[0], 0.0, 1e-12);
    EXPECT_NEAR(y->data[1], 2.0, 1e-12);
    EXPECT_NEAR(y->data[2], 3.0, 1e-12);
    matx_vec_d_i8_destroy(&a, x);
    matx_vec_d_i8_destroy(&a, y);
}

TEST(core_math, vec_sin_cos_d_i8)
{
    matx_alloc_t a = matx_alloc_default();
    matx_vec_d_i8_t x = NULL, s = NULL, c = NULL;
    ASSERT_EQ(matx_vec_d_i8_create(&a, &x, NULL, 3), MATX_OK);
    x->data[0] = 0.0;
    x->data[1] = M_PI / 4.0;
    x->data[2] = M_PI / 2.0;
    ASSERT_EQ(matx_vec_d_i8_sin(&a, x, &s), MATX_OK);
    ASSERT_EQ(matx_vec_d_i8_cos(&a, x, &c), MATX_OK);
    EXPECT_NEAR(s->data[0], 0.0, 1e-12);
    EXPECT_NEAR(s->data[2], 1.0, 1e-12);
    EXPECT_NEAR(c->data[0], 1.0, 1e-12);
    EXPECT_NEAR(c->data[2], 0.0, 1e-12);
    matx_vec_d_i8_destroy(&a, x);
    matx_vec_d_i8_destroy(&a, s);
    matx_vec_d_i8_destroy(&a, c);
}

TEST(core_math, vec_abs_d_i8)
{
    matx_alloc_t a = matx_alloc_default();
    matx_vec_d_i8_t x = NULL, y = NULL;
    ASSERT_EQ(matx_vec_d_i8_create(&a, &x, NULL, 3), MATX_OK);
    x->data[0] = -3.0;
    x->data[1] = 0.0;
    x->data[2] = 5.0;
    matx_status_t st = matx_vec_d_i8_abs(&a, x, &y);
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(y->data[0], 3.0, 1e-12);
    EXPECT_NEAR(y->data[1], 0.0, 1e-12);
    EXPECT_NEAR(y->data[2], 5.0, 1e-12);
    matx_vec_d_i8_destroy(&a, x);
    matx_vec_d_i8_destroy(&a, y);
}

TEST(core_math, vec_pow_d_i8)
{
    matx_alloc_t a = matx_alloc_default();
    matx_vec_d_i8_t x = NULL, y = NULL;
    ASSERT_EQ(matx_vec_d_i8_create(&a, &x, NULL, 3), MATX_OK);
    x->data[0] = 1.0;
    x->data[1] = 2.0;
    x->data[2] = 3.0;
    matx_status_t st = matx_vec_d_i8_pow(&a, x, 2.0, &y);
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(y->data[0], 1.0, 1e-12);
    EXPECT_NEAR(y->data[1], 4.0, 1e-12);
    EXPECT_NEAR(y->data[2], 9.0, 1e-12);
    matx_vec_d_i8_destroy(&a, x);
    matx_vec_d_i8_destroy(&a, y);
}

/* ============================================================
 *  Phase 1: Core element-wise math (complex vectors)
 * ============================================================ */

TEST(core_math, vec_abs_z_i8)
{
    matx_alloc_t a = matx_alloc_default();
    matx_vec_z_i8_t x = NULL;
    matx_vec_d_i8_t y = NULL;
    ASSERT_EQ(matx_vec_z_i8_create(&a, &x, NULL, 3), MATX_OK);
    x->data[0] = {3.0, 4.0};
    x->data[1] = {0.0, 0.0};
    x->data[2] = {-5.0, 12.0};
    matx_status_t st = matx_vec_z_i8_abs(&a, x, &y);
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(y->data[0], 5.0, 1e-12);
    EXPECT_NEAR(y->data[1], 0.0, 1e-12);
    EXPECT_NEAR(y->data[2], 13.0, 1e-12);
    matx_vec_z_i8_destroy(&a, x);
    matx_vec_d_i8_destroy(&a, y);
}

/* ============================================================
 *  Phase 1: Core element-wise math (dense matrices)
 * ============================================================ */

TEST(core_math, dense_exp_d_i8)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t A = NULL, B = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);
    A->data[0] = 0.0;
    A->data[1] = 1.0;
    A->data[2] = 2.0;
    A->data[3] = -1.0;
    matx_status_t st = matx_dense_d_i8_exp(&a, A, &B);
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(B->data[0], 1.0, 1e-12);
    EXPECT_NEAR(B->data[1], exp(1.0), 1e-12);
    EXPECT_NEAR(B->data[2], exp(2.0), 1e-12);
    matx_dense_d_i8_destroy(&a, A);
    matx_dense_d_i8_destroy(&a, B);
}

TEST(core_math, dense_abs_d_i8)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t A = NULL, B = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);
    A->data[0] = -3.0;
    A->data[1] = 0.0;
    A->data[2] = 4.0;
    A->data[3] = -1.0;
    matx_status_t st = matx_dense_d_i8_abs(&a, A, &B);
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(B->data[0], 3.0, 1e-12);
    EXPECT_NEAR(B->data[1], 0.0, 1e-12);
    EXPECT_NEAR(B->data[2], 4.0, 1e-12);
    matx_dense_d_i8_destroy(&a, A);
    matx_dense_d_i8_destroy(&a, B);
}

/* ============================================================
 *  Phase 1: Vector arithmetic (add, sub, mul, div)
 * ============================================================ */

TEST(core_math, vec_add_d_i8)
{
    matx_alloc_t a = matx_alloc_default();
    matx_vec_d_i8_t x = NULL, y = NULL, z = NULL;
    ASSERT_EQ(matx_vec_d_i8_create(&a, &x, NULL, 3), MATX_OK);
    ASSERT_EQ(matx_vec_d_i8_create(&a, &y, NULL, 3), MATX_OK);
    x->data[0] = 1.0;
    x->data[1] = 2.0;
    x->data[2] = 3.0;
    y->data[0] = 4.0;
    y->data[1] = 5.0;
    y->data[2] = 6.0;
    matx_status_t st = matx_vec_d_i8_add(&a, x, y, &z);
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(z->data[0], 5.0, 1e-12);
    EXPECT_NEAR(z->data[1], 7.0, 1e-12);
    EXPECT_NEAR(z->data[2], 9.0, 1e-12);
    matx_vec_d_i8_destroy(&a, x);
    matx_vec_d_i8_destroy(&a, y);
    matx_vec_d_i8_destroy(&a, z);
}

TEST(core_math, vec_sub_d_i8)
{
    matx_alloc_t a = matx_alloc_default();
    matx_vec_d_i8_t x = NULL, y = NULL, z = NULL;
    ASSERT_EQ(matx_vec_d_i8_create(&a, &x, NULL, 3), MATX_OK);
    ASSERT_EQ(matx_vec_d_i8_create(&a, &y, NULL, 3), MATX_OK);
    x->data[0] = 1.0;
    x->data[1] = 2.0;
    x->data[2] = 3.0;
    y->data[0] = 4.0;
    y->data[1] = 6.0;
    y->data[2] = 8.0;
    matx_status_t st = matx_vec_d_i8_sub(&a, x, y, &z);
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(z->data[0], -3.0, 1e-12);
    EXPECT_NEAR(z->data[1], -4.0, 1e-12);
    EXPECT_NEAR(z->data[2], -5.0, 1e-12);
    matx_vec_d_i8_destroy(&a, x);
    matx_vec_d_i8_destroy(&a, y);
    matx_vec_d_i8_destroy(&a, z);
}

TEST(core_math, vec_mul_d_i8)
{
    matx_alloc_t a = matx_alloc_default();
    matx_vec_d_i8_t x = NULL, y = NULL, z = NULL;
    ASSERT_EQ(matx_vec_d_i8_create(&a, &x, NULL, 3), MATX_OK);
    ASSERT_EQ(matx_vec_d_i8_create(&a, &y, NULL, 3), MATX_OK);
    x->data[0] = 2.0;
    x->data[1] = 3.0;
    x->data[2] = 4.0;
    y->data[0] = 5.0;
    y->data[1] = 6.0;
    y->data[2] = 7.0;
    matx_status_t st = matx_vec_d_i8_mul(&a, x, y, &z);
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(z->data[0], 10.0, 1e-12);
    EXPECT_NEAR(z->data[1], 18.0, 1e-12);
    EXPECT_NEAR(z->data[2], 28.0, 1e-12);
    matx_vec_d_i8_destroy(&a, x);
    matx_vec_d_i8_destroy(&a, y);
    matx_vec_d_i8_destroy(&a, z);
}

TEST(core_math, vec_div_d_i8)
{
    matx_alloc_t a = matx_alloc_default();
    matx_vec_d_i8_t x = NULL, y = NULL, z = NULL;
    ASSERT_EQ(matx_vec_d_i8_create(&a, &x, NULL, 3), MATX_OK);
    ASSERT_EQ(matx_vec_d_i8_create(&a, &y, NULL, 3), MATX_OK);
    x->data[0] = 10.0;
    x->data[1] = 20.0;
    x->data[2] = 30.0;
    y->data[0] = 2.0;
    y->data[1] = 5.0;
    y->data[2] = 3.0;
    matx_status_t st = matx_vec_d_i8_div(&a, x, y, &z);
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(z->data[0], 5.0, 1e-12);
    EXPECT_NEAR(z->data[1], 4.0, 1e-12);
    EXPECT_NEAR(z->data[2], 10.0, 1e-12);
    matx_vec_d_i8_destroy(&a, x);
    matx_vec_d_i8_destroy(&a, y);
    matx_vec_d_i8_destroy(&a, z);
}

/* ============================================================
 *  Phase 1: Scalar-vector ops (in-place)
 * ============================================================ */

TEST(core_math, vec_add_scalar_d_i8)
{
    matx_alloc_t a = matx_alloc_default();
    matx_vec_d_i8_t x = NULL;
    ASSERT_EQ(matx_vec_d_i8_create(&a, &x, NULL, 3), MATX_OK);
    x->data[0] = 1.0;
    x->data[1] = 2.0;
    x->data[2] = 3.0;
    matx_status_t st = matx_vec_d_i8_add_scalar(x, 10.0);
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(x->data[0], 11.0, 1e-12);
    EXPECT_NEAR(x->data[1], 12.0, 1e-12);
    EXPECT_NEAR(x->data[2], 13.0, 1e-12);
    matx_vec_d_i8_destroy(&a, x);
}

TEST(core_math, vec_mul_scalar_d_i8)
{
    matx_alloc_t a = matx_alloc_default();
    matx_vec_d_i8_t x = NULL;
    ASSERT_EQ(matx_vec_d_i8_create(&a, &x, NULL, 3), MATX_OK);
    x->data[0] = 1.0;
    x->data[1] = 2.0;
    x->data[2] = 3.0;
    matx_status_t st = matx_vec_d_i8_mul_scalar(x, 0.5);
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(x->data[0], 0.5, 1e-12);
    EXPECT_NEAR(x->data[1], 1.0, 1e-12);
    EXPECT_NEAR(x->data[2], 1.5, 1e-12);
    matx_vec_d_i8_destroy(&a, x);
}

/* ============================================================
 *  Phase 1: Dense arithmetic (add, mul)
 * ============================================================ */

TEST(core_math, dense_add_d_i8)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t A = NULL, B = NULL, C = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);
    ASSERT_EQ(matx_dense_d_i8_create(&a, &B, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);
    A->data[0] = 1.0;
    A->data[1] = 2.0;
    A->data[2] = 3.0;
    A->data[3] = 4.0;
    B->data[0] = 5.0;
    B->data[1] = 6.0;
    B->data[2] = 7.0;
    B->data[3] = 8.0;
    matx_status_t st = matx_dense_d_i8_add(&a, A, B, &C);
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(C->data[0], 6.0, 1e-12);
    EXPECT_NEAR(C->data[3], 12.0, 1e-12);
    matx_dense_d_i8_destroy(&a, A);
    matx_dense_d_i8_destroy(&a, B);
    matx_dense_d_i8_destroy(&a, C);
}

TEST(core_math, dense_mul_d_i8)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t A = NULL, B = NULL, C = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);
    ASSERT_EQ(matx_dense_d_i8_create(&a, &B, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);
    A->data[0] = 1.0;
    A->data[1] = 2.0;
    A->data[2] = 3.0;
    A->data[3] = 4.0;
    B->data[0] = 2.0;
    B->data[1] = 3.0;
    B->data[2] = 4.0;
    B->data[3] = 5.0;
    matx_status_t st = matx_dense_d_i8_mul(&a, A, B, &C);
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(C->data[0], 2.0, 1e-12);
    EXPECT_NEAR(C->data[1], 6.0, 1e-12);
    EXPECT_NEAR(C->data[2], 12.0, 1e-12);
    EXPECT_NEAR(C->data[3], 20.0, 1e-12);
    matx_dense_d_i8_destroy(&a, A);
    matx_dense_d_i8_destroy(&a, B);
    matx_dense_d_i8_destroy(&a, C);
}

/* ============================================================
 *  Phase 1: Diagonal creation and extraction
 * ============================================================ */

TEST(core_math, diag_create_d_i8)
{
    matx_alloc_t a = matx_alloc_default();
    matx_vec_d_i8_t v = NULL;
    matx_dense_d_i8_t D = NULL;
    ASSERT_EQ(matx_vec_d_i8_create(&a, &v, NULL, 3), MATX_OK);
    v->data[0] = 1.0;
    v->data[1] = 2.0;
    v->data[2] = 3.0;
    matx_status_t st = matx_diag_d_i8_create(&a, v, &D);
    ASSERT_EQ(st, MATX_OK);
    EXPECT_EQ(D->nrows, 3u);
    EXPECT_EQ(D->ncols, 3u);
    EXPECT_NEAR(D->data[0 + 0 * D->stride], 1.0, 1e-12);
    EXPECT_NEAR(D->data[1 + 1 * D->stride], 2.0, 1e-12);
    EXPECT_NEAR(D->data[2 + 2 * D->stride], 3.0, 1e-12);
    EXPECT_NEAR(D->data[0 + 1 * D->stride], 0.0, 1e-12);
    matx_vec_d_i8_destroy(&a, v);
    matx_dense_d_i8_destroy(&a, D);
}

TEST(core_math, get_diag_d_i8)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t A = NULL;
    matx_vec_d_i8_t d = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_COL_MAJOR, 3, 3, NULL), MATX_OK);
    A->data[0 + 0 * A->stride] = 1.0;
    A->data[1 + 1 * A->stride] = 2.0;
    A->data[2 + 2 * A->stride] = 3.0;
    A->data[0 + 1 * A->stride] = 99.0;
    matx_status_t st = matx_dense_d_i8_get_diag(&a, A, &d);
    ASSERT_EQ(st, MATX_OK);
    EXPECT_EQ(d->n, 3u);
    EXPECT_NEAR(d->data[0], 1.0, 1e-12);
    EXPECT_NEAR(d->data[1], 2.0, 1e-12);
    EXPECT_NEAR(d->data[2], 3.0, 1e-12);
    matx_dense_d_i8_destroy(&a, A);
    matx_vec_d_i8_destroy(&a, d);
}

/* ============================================================
 *  Phase 1: Cumulative sum
 * ============================================================ */

TEST(core_math, vec_cumsum_d_i8)
{
    matx_alloc_t a = matx_alloc_default();
    matx_vec_d_i8_t x = NULL, y = NULL;
    ASSERT_EQ(matx_vec_d_i8_create(&a, &x, NULL, 4), MATX_OK);
    x->data[0] = 1.0;
    x->data[1] = 2.0;
    x->data[2] = 3.0;
    x->data[3] = 4.0;
    matx_status_t st = matx_vec_d_i8_cumsum(&a, x, &y);
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(y->data[0], 1.0, 1e-12);
    EXPECT_NEAR(y->data[1], 3.0, 1e-12);
    EXPECT_NEAR(y->data[2], 6.0, 1e-12);
    EXPECT_NEAR(y->data[3], 10.0, 1e-12);
    matx_vec_d_i8_destroy(&a, x);
    matx_vec_d_i8_destroy(&a, y);
}

/* ============================================================
 *  Phase 4: Cross product
 * ============================================================ */

TEST(vec_blas, cross_d_i8)
{
    matx_alloc_t a = matx_alloc_default();
    matx_vec_d_i8_t x = NULL, y = NULL, z = NULL;
    ASSERT_EQ(matx_vec_d_i8_create(&a, &x, NULL, 3), MATX_OK);
    ASSERT_EQ(matx_vec_d_i8_create(&a, &y, NULL, 3), MATX_OK);
    ASSERT_EQ(matx_vec_d_i8_create(&a, &z, NULL, 3), MATX_OK);
    x->data[0] = 1.0;
    x->data[1] = 0.0;
    x->data[2] = 0.0;
    y->data[0] = 0.0;
    y->data[1] = 1.0;
    y->data[2] = 0.0;
    matx_vec_backend_t blas = matx_vec_default();
    matx_status_t st = matx_vec_cross_d_i8(&blas, x, y, z);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_vec_d_i8_destroy(&a, x);
        matx_vec_d_i8_destroy(&a, y);
        matx_vec_d_i8_destroy(&a, z);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(z->data[0], 0.0, 1e-12);
    EXPECT_NEAR(z->data[1], 0.0, 1e-12);
    EXPECT_NEAR(z->data[2], 1.0, 1e-12);
    matx_vec_d_i8_destroy(&a, x);
    matx_vec_d_i8_destroy(&a, y);
    matx_vec_d_i8_destroy(&a, z);
}

TEST(vec_blas, cross_z_i8)
{
    matx_alloc_t a = matx_alloc_default();
    matx_vec_z_i8_t x = NULL, y = NULL, z = NULL;
    ASSERT_EQ(matx_vec_z_i8_create(&a, &x, NULL, 3), MATX_OK);
    ASSERT_EQ(matx_vec_z_i8_create(&a, &y, NULL, 3), MATX_OK);
    ASSERT_EQ(matx_vec_z_i8_create(&a, &z, NULL, 3), MATX_OK);
    x->data[0] = {1.0, 0.0};
    x->data[1] = {0.0, 0.0};
    x->data[2] = {0.0, 0.0};
    y->data[0] = {0.0, 0.0};
    y->data[1] = {1.0, 0.0};
    y->data[2] = {0.0, 0.0};
    matx_vec_backend_t blas = matx_vec_default();
    matx_status_t st = matx_vec_cross_z_i8(&blas, x, y, z);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_vec_z_i8_destroy(&a, x);
        matx_vec_z_i8_destroy(&a, y);
        matx_vec_z_i8_destroy(&a, z);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(z->data[0].real, 0.0, 1e-12);
    EXPECT_NEAR(z->data[1].real, 0.0, 1e-12);
    EXPECT_NEAR(z->data[2].real, 1.0, 1e-12);
    EXPECT_NEAR(z->data[2].imag, 0.0, 1e-12);
    matx_vec_z_i8_destroy(&a, x);
    matx_vec_z_i8_destroy(&a, y);
    matx_vec_z_i8_destroy(&a, z);
}

/* ============================================================
 *  Phase 2: Matrix inverse (uses pre-allocated output)
 * ============================================================ */

TEST(compute_dense, inv_d_i8)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t A = NULL, Ainv = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);
    ASSERT_EQ(matx_dense_d_i8_create(&a, &Ainv, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);
    A->data[0 + 0 * A->stride] = 4.0;
    A->data[1 + 0 * A->stride] = 3.0;
    A->data[0 + 1 * A->stride] = 3.0;
    A->data[1 + 1 * A->stride] = 2.0;
    matx_dense_backend_t blas = matx_blas_default();
    matx_status_t st = matx_inv_dense_d_i8(&blas, A, Ainv);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_dense_d_i8_destroy(&a, A);
        matx_dense_d_i8_destroy(&a, Ainv);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(Ainv->data[0 + 0 * Ainv->stride], -2.0, 1e-10);
    EXPECT_NEAR(Ainv->data[1 + 0 * Ainv->stride], 3.0, 1e-10);
    EXPECT_NEAR(Ainv->data[0 + 1 * Ainv->stride], 3.0, 1e-10);
    EXPECT_NEAR(Ainv->data[1 + 1 * Ainv->stride], -4.0, 1e-10);
    matx_dense_d_i8_destroy(&a, A);
    matx_dense_d_i8_destroy(&a, Ainv);
}

/* ============================================================
 *  Phase 3: Dense solve — QR, determinant, condition number
 * ============================================================ */

TEST(solve, qr_d_i8)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t A = NULL, Q = NULL, R = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_COL_MAJOR, 3, 2, NULL), MATX_OK);
    A->data[0] = 1.0;
    A->data[1] = 0.0;
    A->data[2] = 0.0;
    A->data[3] = 1.0;
    A->data[4] = 1.0;
    A->data[5] = 0.0;
    matx_dense_linsolve_t ls = matx_dense_linsolve_default(matx_alloc_default());
    matx_status_t st = matx_qr_d_i8(&ls, A, &Q, &R);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_dense_d_i8_destroy(&a, A);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    ASSERT_NE(Q, nullptr);
    ASSERT_NE(R, nullptr);
    matx_dense_d_i8_destroy(&a, A);
    matx_dense_d_i8_destroy(&a, Q);
    matx_dense_d_i8_destroy(&a, R);
}

TEST(solve, det_dense_d_i8)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t A = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);
    A->data[0] = 4.0;
    A->data[1] = 3.0;
    A->data[2] = 3.0;
    A->data[3] = 2.0;
    matx_dense_linsolve_t ls = matx_dense_linsolve_default(matx_alloc_default());
    matx_double det = 0.0;
    matx_status_t st = matx_det_dense_d_i8(&ls, A, &det);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_dense_d_i8_destroy(&a, A);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(det, -1.0, 1e-10);
    matx_dense_d_i8_destroy(&a, A);
}

TEST(solve, cond_dense_d_i8)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t A = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);
    A->data[0] = 1.0;
    A->data[1] = 0.0;
    A->data[2] = 0.0;
    A->data[3] = 1.0;
    matx_dense_linsolve_t ls = matx_dense_linsolve_default(matx_alloc_default());
    matx_double cond = 0.0;
    matx_status_t st = matx_cond_dense_d_i8(&ls, A, &cond);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_dense_d_i8_destroy(&a, A);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(cond, 1.0, 1e-10);
    matx_dense_d_i8_destroy(&a, A);
}

/* ============================================================
 *  Phase 5: COO helpers
 * ============================================================ */

TEST(sparse, coo_get_row_d_i8)
{
    matx_alloc_t a = matx_alloc_default();
    matx_int64_t rows[4] = {0, 1, 0, 1};
    matx_int64_t cols[4] = {0, 0, 1, 1};
    matx_double vals[4] = {10.0, 20.0, 30.0, 40.0};
    matx_coo_d_i8_t A = NULL;
    ASSERT_EQ(matx_coo_sparse_d_i8_create(&a, &A, 2, 2, 4, rows, cols, vals), MATX_OK);
    matx_vec_d_i8_t row = NULL;
    ASSERT_EQ(matx_vec_d_i8_create(&a, &row, NULL, 2), MATX_OK);
    matx_status_t st = matx_coo_get_row_d_i8(A, 0, row);
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(row->data[0], 10.0, 1e-12);
    EXPECT_NEAR(row->data[1], 30.0, 1e-12);
    matx_coo_sparse_d_i8_destroy(&a, A);
    matx_vec_d_i8_destroy(&a, row);
}

TEST(sparse, coo_to_dense_d_i8)
{
    matx_alloc_t a = matx_alloc_default();
    matx_int64_t rows[3] = {0, 1, 1};
    matx_int64_t cols[3] = {0, 0, 1};
    matx_double vals[3] = {1.0, 2.0, 3.0};
    matx_coo_d_i8_t A = NULL;
    ASSERT_EQ(matx_coo_sparse_d_i8_create(&a, &A, 2, 2, 3, rows, cols, vals), MATX_OK);
    matx_dense_d_i8_t D = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &D, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);
    matx_status_t st = matx_coo_to_dense_d_i8(A, D);
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(D->data[0], 1.0, 1e-12);
    EXPECT_NEAR(D->data[1], 2.0, 1e-12);
    EXPECT_NEAR(D->data[2], 0.0, 1e-12);
    EXPECT_NEAR(D->data[3], 3.0, 1e-12);
    matx_coo_sparse_d_i8_destroy(&a, A);
    matx_dense_d_i8_destroy(&a, D);
}

TEST(sparse, coo_to_dense_d_i8_row_major_padded)
{
    matx_alloc_t a = matx_alloc_default();
    matx_int64_t rows[3] = {0, 1, 1};
    matx_int64_t cols[3] = {0, 0, 2};
    matx_double vals[3] = {1.0, 2.0, 3.0};
    matx_coo_d_i8_t A = NULL;
    ASSERT_EQ(matx_coo_sparse_d_i8_create(&a, &A, 2, 3, 3, rows, cols, vals), MATX_OK);

    matx_double backing[8] = {9.0, 9.0, 9.0, 91.0,
                              9.0, 9.0, 9.0, 92.0};
    matx_dense_d_i8_t D = NULL;
    ASSERT_EQ(matx_dense_d_i8_wrap(&a, &D, 2, 3, 4, MATX_ROW_MAJOR, backing), MATX_OK);
    ASSERT_EQ(matx_coo_to_dense_d_i8(A, D), MATX_OK);

    EXPECT_NEAR(backing[0], 1.0, 1e-12);
    EXPECT_NEAR(backing[1], 0.0, 1e-12);
    EXPECT_NEAR(backing[2], 0.0, 1e-12);
    EXPECT_NEAR(backing[3], 91.0, 1e-12);
    EXPECT_NEAR(backing[4], 2.0, 1e-12);
    EXPECT_NEAR(backing[5], 0.0, 1e-12);
    EXPECT_NEAR(backing[6], 3.0, 1e-12);
    EXPECT_NEAR(backing[7], 92.0, 1e-12);

    matx_dense_d_i8_destroy(&a, D);
    matx_coo_sparse_d_i8_destroy(&a, A);
}

TEST(sparse, coo_to_dense_z_i8_col_major_padded_merge_duplicates)
{
    matx_alloc_t a = matx_alloc_default();
    matx_int64_t rows[4] = {0, 1, 1, 1};
    matx_int64_t cols[4] = {0, 0, 1, 0};
    matx_complex_d_t vals[4] = {{1.0, 0.5}, {2.0, 1.0},
                                {3.0, -0.5}, {0.5, -0.5}};
    matx_coo_z_i8_t A = NULL;
    ASSERT_EQ(matx_coo_sparse_z_i8_create(&a, &A, 2, 2, 4, rows, cols, vals), MATX_OK);

    matx_complex_d_t backing[6] = {{9.0, 9.0}, {9.0, 9.0}, {91.0, 91.0},
                                   {9.0, 9.0}, {9.0, 9.0}, {92.0, 92.0}};
    matx_dense_z_i8_t D = NULL;
    ASSERT_EQ(matx_dense_z_i8_wrap(&a, &D, 2, 2, 3, MATX_COL_MAJOR, backing), MATX_OK);
    ASSERT_EQ(matx_coo_to_dense_z_i8(A, D), MATX_OK);

    EXPECT_NEAR(backing[0].real, 1.0, 1e-12);
    EXPECT_NEAR(backing[0].imag, 0.5, 1e-12);
    EXPECT_NEAR(backing[1].real, 2.5, 1e-12);
    EXPECT_NEAR(backing[1].imag, 0.5, 1e-12);
    EXPECT_NEAR(backing[2].real, 91.0, 1e-12);
    EXPECT_NEAR(backing[2].imag, 91.0, 1e-12);
    EXPECT_NEAR(backing[3].real, 0.0, 1e-12);
    EXPECT_NEAR(backing[3].imag, 0.0, 1e-12);
    EXPECT_NEAR(backing[4].real, 3.0, 1e-12);
    EXPECT_NEAR(backing[4].imag, -0.5, 1e-12);
    EXPECT_NEAR(backing[5].real, 92.0, 1e-12);
    EXPECT_NEAR(backing[5].imag, 92.0, 1e-12);

    matx_dense_z_i8_destroy(&a, D);
    matx_coo_sparse_z_i8_destroy(&a, A);
}

/* ============================================================
 *  Phase 5: Sparse Cholesky
 * ============================================================ */

TEST(solve, sparse_chol_d_i8)
{
    matx_alloc_t a = matx_alloc_default();
    matx_int64_t nnz = 10;
    matx_int64_t rows[10] = {0, 1, 2, 3, 1, 2, 3, 0, 1, 2};
    matx_int64_t cols[10] = {0, 1, 2, 3, 0, 1, 2, 1, 2, 3};
    matx_double vals[10] = {4.0, 4.0, 4.0, 4.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0};
    matx_coo_d_i8_t A = NULL;
    matx_coo_sparse_d_i8_create(&a, &A, 4, 4, nnz, rows, cols, vals);
    double b[4] = {2.0, 4.0, 6.0, 13.0};
    double x[4] = {0.0, 0.0, 0.0, 0.0};
    matx_sparse_linsolve_t ls
        = matx_sparse_linsolve_by_type(MATX_LINSOLVE_BACKEND_SUITESPARSE_KLU,
                                       matx_alloc_default());
    matx_status_t st = matx_solve_chol_coo_d_i8(&ls, A, b, x);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_coo_sparse_d_i8_destroy(&a, A);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(x[0], 1.0, 1e-10);
    EXPECT_NEAR(x[1], 2.0, 1e-10);
    EXPECT_NEAR(x[2], 3.0, 1e-10);
    EXPECT_NEAR(x[3], 4.0, 1e-10);
    matx_coo_sparse_d_i8_destroy(&a, A);
}

TEST(core_math, row_major_dense_elementwise_and_scalars)
{
    matx_alloc_t alloc = matx_alloc_default();
    matx_dense_d_i8_t real_a = NULL, real_b = NULL, real_out = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&alloc, &real_a, MATX_ROW_MAJOR, 2, 3, NULL), MATX_OK);
    ASSERT_EQ(matx_dense_d_i8_create(&alloc, &real_b, MATX_ROW_MAJOR, 2, 3, NULL), MATX_OK);
    const double real_values_a[6] = {1, 2, 3, 4, 5, 6};
    const double real_values_b[6] = {2, 3, 4, 5, 6, 7};
    std::copy(real_values_a, real_values_a + 6, real_a->data);
    std::copy(real_values_b, real_values_b + 6, real_b->data);
    ASSERT_EQ(matx_dense_d_i8_add(&alloc, real_a, real_b, &real_out), MATX_OK);
    ASSERT_EQ(real_out->stride, 3);
    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(real_out->data[i], real_values_a[i] + real_values_b[i], 1e-14);
    ASSERT_EQ(matx_dense_d_i8_mul_scalar(real_out, 2.0), MATX_OK);
    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(real_out->data[i],
                    2.0 * (real_values_a[i] + real_values_b[i]), 1e-14);

    matx_dense_z_i8_t complex_a = NULL, complex_b = NULL, complex_out = NULL;
    ASSERT_EQ(matx_dense_z_i8_create(&alloc, &complex_a, MATX_ROW_MAJOR, 2, 3, NULL), MATX_OK);
    ASSERT_EQ(matx_dense_z_i8_create(&alloc, &complex_b, MATX_ROW_MAJOR, 2, 3, NULL), MATX_OK);
    for (int i = 0; i < 6; ++i) {
        complex_a->data[i] = {real_values_a[i], -0.5 * real_values_a[i]};
        complex_b->data[i] = {real_values_b[i], 0.25 * real_values_b[i]};
    }
    ASSERT_EQ(matx_dense_z_i8_add(&alloc, complex_a, complex_b, &complex_out), MATX_OK);
    ASSERT_EQ(complex_out->stride, 3);
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(complex_out->data[i].real,
                    real_values_a[i] + real_values_b[i], 1e-14);
        EXPECT_NEAR(complex_out->data[i].imag,
                    -0.5 * real_values_a[i] + 0.25 * real_values_b[i], 1e-14);
    }
    ASSERT_EQ(matx_dense_z_i8_mul_scalar(complex_out, {0.5, 0.25}), MATX_OK);
    for (int i = 0; i < 6; ++i) {
        const double real = real_values_a[i] + real_values_b[i];
        const double imag = -0.5 * real_values_a[i] + 0.25 * real_values_b[i];
        EXPECT_NEAR(complex_out->data[i].real, 0.5 * real - 0.25 * imag, 1e-14);
        EXPECT_NEAR(complex_out->data[i].imag, 0.25 * real + 0.5 * imag, 1e-14);
    }

    matx_dense_z_i8_destroy(&alloc, complex_out);
    matx_dense_z_i8_destroy(&alloc, complex_b);
    matx_dense_z_i8_destroy(&alloc, complex_a);
    matx_dense_d_i8_destroy(&alloc, real_out);
    matx_dense_d_i8_destroy(&alloc, real_b);
    matx_dense_d_i8_destroy(&alloc, real_a);
}
