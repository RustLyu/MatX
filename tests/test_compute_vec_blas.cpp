#include <gtest/gtest.h>

extern "C" {
#include "matx/matx_types.h"
#include "matx/matx_func.h"
#include "matx/matx_vec_compute.h"
#include "matx/matx_types_internal.h"
}
#include <math.h>

// ---- scal ----

TEST(vec_blas, scal_f64) {
  matx_alloc_t a = matx_alloc_default();
  matx_vec_f64_t x = NULL;
  ASSERT_EQ(matx_vec_f64_create(&a, &x, NULL, 4), MATX_OK);
  x->data[0] = 1.0; x->data[1] = 2.0; x->data[2] = 3.0; x->data[3] = 4.0;

  matx_vec_backend_t blas = matx_vec_default();
  matx_status_t st = matx_vec_scal_f64(&blas, 3.0, x);
  if (st == MATX_ERR_NOT_SUPPORTED) { matx_vec_f64_destroy(&a, x); return; }
  ASSERT_EQ(st, MATX_OK);
  EXPECT_NEAR(x->data[0], 3.0, 1e-12);
  EXPECT_NEAR(x->data[1], 6.0, 1e-12);
  EXPECT_NEAR(x->data[2], 9.0, 1e-12);
  EXPECT_NEAR(x->data[3], 12.0, 1e-12);
  matx_vec_f64_destroy(&a, x);
}

TEST(vec_blas, scal_c64) {
  matx_alloc_t a = matx_alloc_default();
  matx_vec_c64_t x = NULL;
  ASSERT_EQ(matx_vec_c64_create(&a, &x, NULL, 3), MATX_OK);
  x->data[0] = {1.0, 2.0}; x->data[1] = {3.0, 4.0}; x->data[2] = {0.0, -1.0};

  matx_vec_backend_t blas = matx_vec_default();
  matx_complex_f64_t alpha = {2.0, 1.0};
  matx_status_t st = matx_vec_scal_c64(&blas, alpha, x);
  if (st == MATX_ERR_NOT_SUPPORTED) { matx_vec_c64_destroy(&a, x); return; }
  ASSERT_EQ(st, MATX_OK);
  EXPECT_NEAR(x->data[0].real, 2.0*1.0 - 1.0*2.0, 1e-12);
  EXPECT_NEAR(x->data[0].imag, 2.0*2.0 + 1.0*1.0, 1e-12);
  EXPECT_NEAR(x->data[1].real, 2.0*3.0 - 1.0*4.0, 1e-12);
  EXPECT_NEAR(x->data[1].imag, 2.0*4.0 + 1.0*3.0, 1e-12);
  matx_vec_c64_destroy(&a, x);
}

// ---- copy ----

TEST(vec_blas, copy_f64) {
  matx_alloc_t a = matx_alloc_default();
  matx_vec_f64_t x = NULL, y = NULL;
  ASSERT_EQ(matx_vec_f64_create(&a, &x, NULL, 3), MATX_OK);
  ASSERT_EQ(matx_vec_f64_create(&a, &y, NULL, 3), MATX_OK);
  x->data[0] = 1.0; x->data[1] = 2.0; x->data[2] = 3.0;

  matx_vec_backend_t blas = matx_vec_default();
  matx_status_t st = matx_vec_copy_f64(&blas, x, y);
  if (st == MATX_ERR_NOT_SUPPORTED) { matx_vec_f64_destroy(&a, x); matx_vec_f64_destroy(&a, y); return; }
  ASSERT_EQ(st, MATX_OK);
  EXPECT_NEAR(y->data[0], 1.0, 1e-12);
  EXPECT_NEAR(y->data[1], 2.0, 1e-12);
  EXPECT_NEAR(y->data[2], 3.0, 1e-12);
  matx_vec_f64_destroy(&a, x);
  matx_vec_f64_destroy(&a, y);
}

TEST(vec_blas, copy_c64) {
  matx_alloc_t a = matx_alloc_default();
  matx_vec_c64_t x = NULL, y = NULL;
  ASSERT_EQ(matx_vec_c64_create(&a, &x, NULL, 2), MATX_OK);
  ASSERT_EQ(matx_vec_c64_create(&a, &y, NULL, 2), MATX_OK);
  x->data[0] = {1.0, 2.0}; x->data[1] = {3.0, -4.0};

  matx_vec_backend_t blas = matx_vec_default();
  matx_status_t st = matx_vec_copy_c64(&blas, x, y);
  if (st == MATX_ERR_NOT_SUPPORTED) { matx_vec_c64_destroy(&a, x); matx_vec_c64_destroy(&a, y); return; }
  ASSERT_EQ(st, MATX_OK);
  EXPECT_NEAR(y->data[0].real, 1.0, 1e-12);
  EXPECT_NEAR(y->data[0].imag, 2.0, 1e-12);
  EXPECT_NEAR(y->data[1].real, 3.0, 1e-12);
  EXPECT_NEAR(y->data[1].imag, -4.0, 1e-12);
  matx_vec_c64_destroy(&a, x);
  matx_vec_c64_destroy(&a, y);
}

// ---- swap ----

TEST(vec_blas, swap_f64) {
  matx_alloc_t a = matx_alloc_default();
  matx_vec_f64_t x = NULL, y = NULL;
  ASSERT_EQ(matx_vec_f64_create(&a, &x, NULL, 3), MATX_OK);
  ASSERT_EQ(matx_vec_f64_create(&a, &y, NULL, 3), MATX_OK);
  x->data[0] = 1.0; x->data[1] = 2.0; x->data[2] = 3.0;
  y->data[0] = 10.0; y->data[1] = 20.0; y->data[2] = 30.0;

  matx_vec_backend_t blas = matx_vec_default();
  matx_status_t st = matx_vec_swap_f64(&blas, x, y);
  if (st == MATX_ERR_NOT_SUPPORTED) { matx_vec_f64_destroy(&a, x); matx_vec_f64_destroy(&a, y); return; }
  ASSERT_EQ(st, MATX_OK);
  EXPECT_NEAR(x->data[0], 10.0, 1e-12);
  EXPECT_NEAR(x->data[1], 20.0, 1e-12);
  EXPECT_NEAR(x->data[2], 30.0, 1e-12);
  EXPECT_NEAR(y->data[0], 1.0, 1e-12);
  EXPECT_NEAR(y->data[1], 2.0, 1e-12);
  EXPECT_NEAR(y->data[2], 3.0, 1e-12);
  matx_vec_f64_destroy(&a, x);
  matx_vec_f64_destroy(&a, y);
}

TEST(vec_blas, swap_c64) {
  matx_alloc_t a = matx_alloc_default();
  matx_vec_c64_t x = NULL, y = NULL;
  ASSERT_EQ(matx_vec_c64_create(&a, &x, NULL, 2), MATX_OK);
  ASSERT_EQ(matx_vec_c64_create(&a, &y, NULL, 2), MATX_OK);
  x->data[0] = {1.0, 2.0}; x->data[1] = {3.0, 4.0};
  y->data[0] = {10.0, 20.0}; y->data[1] = {30.0, 40.0};

  matx_vec_backend_t blas = matx_vec_default();
  matx_status_t st = matx_vec_swap_c64(&blas, x, y);
  if (st == MATX_ERR_NOT_SUPPORTED) { matx_vec_c64_destroy(&a, x); matx_vec_c64_destroy(&a, y); return; }
  ASSERT_EQ(st, MATX_OK);
  EXPECT_NEAR(x->data[0].real, 10.0, 1e-12);
  EXPECT_NEAR(x->data[0].imag, 20.0, 1e-12);
  EXPECT_NEAR(y->data[0].real, 1.0, 1e-12);
  EXPECT_NEAR(y->data[0].imag, 2.0, 1e-12);
  matx_vec_c64_destroy(&a, x);
  matx_vec_c64_destroy(&a, y);
}

// ---- dot ----

TEST(vec_blas, dot_f64) {
  matx_alloc_t a = matx_alloc_default();
  matx_vec_f64_t x = NULL, y = NULL;
  ASSERT_EQ(matx_vec_f64_create(&a, &x, NULL, 3), MATX_OK);
  ASSERT_EQ(matx_vec_f64_create(&a, &y, NULL, 3), MATX_OK);
  x->data[0] = 1.0; x->data[1] = 2.0; x->data[2] = 3.0;
  y->data[0] = 4.0; y->data[1] = 5.0; y->data[2] = 6.0;

  matx_vec_backend_t blas = matx_vec_default();
  matx_double result = 0.0;
  matx_status_t st = matx_vec_dot_f64(&blas, x, y, &result);
  if (st == MATX_ERR_NOT_SUPPORTED) { matx_vec_f64_destroy(&a, x); matx_vec_f64_destroy(&a, y); return; }
  ASSERT_EQ(st, MATX_OK);
  EXPECT_NEAR(result, 1.0*4.0 + 2.0*5.0 + 3.0*6.0, 1e-12);
  matx_vec_f64_destroy(&a, x);
  matx_vec_f64_destroy(&a, y);
}

TEST(vec_blas, dotu_c64) {
  matx_alloc_t a = matx_alloc_default();
  matx_vec_c64_t x = NULL, y = NULL;
  ASSERT_EQ(matx_vec_c64_create(&a, &x, NULL, 2), MATX_OK);
  ASSERT_EQ(matx_vec_c64_create(&a, &y, NULL, 2), MATX_OK);
  x->data[0] = {1.0, 2.0}; x->data[1] = {3.0, 0.0};
  y->data[0] = {4.0, 5.0}; y->data[1] = {6.0, 0.0};

  matx_vec_backend_t blas = matx_vec_default();
  matx_complex_f64_t result = {0.0, 0.0};
  matx_status_t st = matx_vec_dotu_c64(&blas, x, y, &result);
  if (st == MATX_ERR_NOT_SUPPORTED) { matx_vec_c64_destroy(&a, x); matx_vec_c64_destroy(&a, y); return; }
  ASSERT_EQ(st, MATX_OK);
  EXPECT_NEAR(result.real, (1.0*4.0 - 2.0*5.0) + (3.0*6.0 - 0.0*0.0), 1e-12);
  EXPECT_NEAR(result.imag, (1.0*5.0 + 2.0*4.0) + (3.0*0.0 + 0.0*6.0), 1e-12);
  matx_vec_c64_destroy(&a, x);
  matx_vec_c64_destroy(&a, y);
}

TEST(vec_blas, dotc_c64) {
  matx_alloc_t a = matx_alloc_default();
  matx_vec_c64_t x = NULL, y = NULL;
  ASSERT_EQ(matx_vec_c64_create(&a, &x, NULL, 2), MATX_OK);
  ASSERT_EQ(matx_vec_c64_create(&a, &y, NULL, 2), MATX_OK);
  x->data[0] = {1.0, 2.0}; x->data[1] = {3.0, 0.0};
  y->data[0] = {4.0, 5.0}; y->data[1] = {6.0, 0.0};

  matx_vec_backend_t blas = matx_vec_default();
  matx_complex_f64_t result = {0.0, 0.0};
  matx_status_t st = matx_vec_dotc_c64(&blas, x, y, &result);
  if (st == MATX_ERR_NOT_SUPPORTED) { matx_vec_c64_destroy(&a, x); matx_vec_c64_destroy(&a, y); return; }
  ASSERT_EQ(st, MATX_OK);
  // dotc = conj(x) . y = (1-2i)*(4+5i) + (3)*(6) = (4+5i+8i-10i^2) + 18 = (4+13i+10)+18 = 14+18+13i
  EXPECT_NEAR(result.real, (1.0*4.0 + 2.0*5.0) + (3.0*6.0 + 0.0*0.0), 1e-12);
  EXPECT_NEAR(result.imag, (-2.0*4.0 + 1.0*5.0) + (-0.0*6.0 + 3.0*0.0), 1e-12);
  matx_vec_c64_destroy(&a, x);
  matx_vec_c64_destroy(&a, y);
}

// ---- nrm2 ----

TEST(vec_blas, nrm2_f64) {
  matx_alloc_t a = matx_alloc_default();
  matx_vec_f64_t x = NULL;
  ASSERT_EQ(matx_vec_f64_create(&a, &x, NULL, 3), MATX_OK);
  x->data[0] = 1.0; x->data[1] = 2.0; x->data[2] = 3.0;

  matx_vec_backend_t blas = matx_vec_default();
  matx_double result = 0.0;
  matx_status_t st = matx_vec_nrm2_f64(&blas, x, &result);
  if (st == MATX_ERR_NOT_SUPPORTED) { matx_vec_f64_destroy(&a, x); return; }
  ASSERT_EQ(st, MATX_OK);
  EXPECT_NEAR(result, sqrt(1.0*1.0 + 2.0*2.0 + 3.0*3.0), 1e-12);
  matx_vec_f64_destroy(&a, x);
}

TEST(vec_blas, nrm2_c64) {
  matx_alloc_t a = matx_alloc_default();
  matx_vec_c64_t x = NULL;
  ASSERT_EQ(matx_vec_c64_create(&a, &x, NULL, 2), MATX_OK);
  x->data[0] = {1.0, 2.0}; x->data[1] = {3.0, 4.0};

  matx_vec_backend_t blas = matx_vec_default();
  matx_double result = 0.0;
  matx_status_t st = matx_vec_nrm2_c64(&blas, x, &result);
  if (st == MATX_ERR_NOT_SUPPORTED) { matx_vec_c64_destroy(&a, x); return; }
  ASSERT_EQ(st, MATX_OK);
  EXPECT_NEAR(result, sqrt(5.0 + 25.0), 1e-12);
  matx_vec_c64_destroy(&a, x);
}

// ---- asum ----

TEST(vec_blas, asum_f64) {
  matx_alloc_t a = matx_alloc_default();
  matx_vec_f64_t x = NULL;
  ASSERT_EQ(matx_vec_f64_create(&a, &x, NULL, 4), MATX_OK);
  x->data[0] = 1.0; x->data[1] = -2.0; x->data[2] = 3.0; x->data[3] = -4.0;

  matx_vec_backend_t blas = matx_vec_default();
  matx_double result = 0.0;
  matx_status_t st = matx_vec_asum_f64(&blas, x, &result);
  if (st == MATX_ERR_NOT_SUPPORTED) { matx_vec_f64_destroy(&a, x); return; }
  ASSERT_EQ(st, MATX_OK);
  EXPECT_NEAR(result, 1.0 + 2.0 + 3.0 + 4.0, 1e-12);
  matx_vec_f64_destroy(&a, x);
}

TEST(vec_blas, asum_c64) {
  matx_alloc_t a = matx_alloc_default();
  matx_vec_c64_t x = NULL;
  ASSERT_EQ(matx_vec_c64_create(&a, &x, NULL, 2), MATX_OK);
  x->data[0] = {1.0, -2.0}; x->data[1] = {-3.0, 4.0};

  matx_vec_backend_t blas = matx_vec_default();
  matx_double result = 0.0;
  matx_status_t st = matx_vec_asum_c64(&blas, x, &result);
  if (st == MATX_ERR_NOT_SUPPORTED) { matx_vec_c64_destroy(&a, x); return; }
  ASSERT_EQ(st, MATX_OK);
  EXPECT_NEAR(result, (1.0 + 2.0) + (3.0 + 4.0), 1e-12);
  matx_vec_c64_destroy(&a, x);
}

// ---- iamax ----

TEST(vec_blas, iamax_f64) {
  matx_alloc_t a = matx_alloc_default();
  matx_vec_f64_t x = NULL;
  ASSERT_EQ(matx_vec_f64_create(&a, &x, NULL, 5), MATX_OK);
  x->data[0] = 1.0; x->data[1] = -5.0; x->data[2] = 3.0; x->data[3] = -2.0; x->data[4] = 4.0;

  matx_vec_backend_t blas = matx_vec_default();
  matx_int64_t result = -1;
  matx_status_t st = matx_vec_iamax_f64(&blas, x, &result);
  if (st == MATX_ERR_NOT_SUPPORTED) { matx_vec_f64_destroy(&a, x); return; }
  ASSERT_EQ(st, MATX_OK);
  EXPECT_EQ(result, 1);
  matx_vec_f64_destroy(&a, x);
}

TEST(vec_blas, iamax_c64) {
  matx_alloc_t a = matx_alloc_default();
  matx_vec_c64_t x = NULL;
  ASSERT_EQ(matx_vec_c64_create(&a, &x, NULL, 3), MATX_OK);
  x->data[0] = {1.0, 0.0}; x->data[1] = {0.0, 5.0}; x->data[2] = {3.0, 0.0};

  matx_vec_backend_t blas = matx_vec_default();
  matx_int64_t result = -1;
  matx_status_t st = matx_vec_iamax_c64(&blas, x, &result);
  if (st == MATX_ERR_NOT_SUPPORTED) { matx_vec_c64_destroy(&a, x); return; }
  ASSERT_EQ(st, MATX_OK);
  EXPECT_EQ(result, 1);
  matx_vec_c64_destroy(&a, x);
}

// ---- axpy ----

TEST(vec_blas, axpy_f64) {
  matx_alloc_t a = matx_alloc_default();
  matx_vec_f64_t x = NULL, y = NULL;
  ASSERT_EQ(matx_vec_f64_create(&a, &x, NULL, 4), MATX_OK);
  ASSERT_EQ(matx_vec_f64_create(&a, &y, NULL, 4), MATX_OK);
  x->data[0] = 1.0; x->data[1] = 2.0; x->data[2] = 3.0; x->data[3] = 4.0;
  y->data[0] = 0.1; y->data[1] = 0.2; y->data[2] = 0.3; y->data[3] = 0.4;

  matx_vec_backend_t blas = matx_vec_default();
  matx_status_t st = matx_vec_axpy_f64(&blas, 2.0, x, y);
  if (st == MATX_ERR_NOT_SUPPORTED) { matx_vec_f64_destroy(&a, x); matx_vec_f64_destroy(&a, y); return; }
  ASSERT_EQ(st, MATX_OK);
  EXPECT_NEAR(y->data[0], 2.0 * 1.0 + 0.1, 1e-12);
  EXPECT_NEAR(y->data[1], 2.0 * 2.0 + 0.2, 1e-12);
  EXPECT_NEAR(y->data[2], 2.0 * 3.0 + 0.3, 1e-12);
  EXPECT_NEAR(y->data[3], 2.0 * 4.0 + 0.4, 1e-12);
  matx_vec_f64_destroy(&a, x);
  matx_vec_f64_destroy(&a, y);
}

TEST(vec_blas, axpy_c64) {
  matx_alloc_t a = matx_alloc_default();
  matx_vec_c64_t x = NULL, y = NULL;
  ASSERT_EQ(matx_vec_c64_create(&a, &x, NULL, 4), MATX_OK);
  ASSERT_EQ(matx_vec_c64_create(&a, &y, NULL, 4), MATX_OK);
  x->data[0] = {1.0, 0.0}; x->data[1] = {0.0, 1.0}; x->data[2] = {1.0, 1.0}; x->data[3] = {0.0, 0.0};
  y->data[0] = {0.5, 0.0}; y->data[1] = {0.0, 0.5}; y->data[2] = {0.0, 0.0}; y->data[3] = {1.0, 1.0};

  matx_vec_backend_t blas = matx_vec_default();
  matx_complex_f64_t alpha = {2.0, 0.0};
  matx_status_t st = matx_vec_axpy_c64(&blas, alpha, x, y);
  if (st == MATX_ERR_NOT_SUPPORTED) { matx_vec_c64_destroy(&a, x); matx_vec_c64_destroy(&a, y); return; }
  ASSERT_EQ(st, MATX_OK);
  EXPECT_NEAR(y->data[0].real, 2.5, 1e-12);
  EXPECT_NEAR(y->data[0].imag, 0.0, 1e-12);
  EXPECT_NEAR(y->data[1].real, 0.0, 1e-12);
  EXPECT_NEAR(y->data[1].imag, 2.5, 1e-12);
  matx_vec_c64_destroy(&a, x);
  matx_vec_c64_destroy(&a, y);
}

// ---- vector norms ----

TEST(vec_blas, norm1_f64) {
  matx_alloc_t a = matx_alloc_default();
  matx_vec_f64_t v = NULL;
  ASSERT_EQ(matx_vec_f64_create(&a, &v, NULL, 4), MATX_OK);
  v->data[0] = 1.0; v->data[1] = -2.0; v->data[2] = 3.0; v->data[3] = -4.0;

  matx_vec_backend_t blas = matx_vec_default();
  matx_double out = 0.0;
  matx_status_t st = matx_vec_norm1_f64(&blas, v, &out);
  if (st == MATX_ERR_NOT_SUPPORTED) { matx_vec_f64_destroy(&a, v); return; }
  ASSERT_EQ(st, MATX_OK);
  EXPECT_NEAR(out, 1.0 + 2.0 + 3.0 + 4.0, 1e-12);
  matx_vec_f64_destroy(&a, v);
}

TEST(vec_blas, norm1_c64) {
  matx_alloc_t a = matx_alloc_default();
  matx_vec_c64_t v = NULL;
  ASSERT_EQ(matx_vec_c64_create(&a, &v, NULL, 2), MATX_OK);
  v->data[0] = {1.0, 2.0}; v->data[1] = {3.0, -4.0};

  matx_vec_backend_t blas = matx_vec_default();
  matx_double out = 0.0;
  matx_status_t st = matx_vec_norm1_c64(&blas, v, &out);
  if (st == MATX_ERR_NOT_SUPPORTED) { matx_vec_c64_destroy(&a, v); return; }
  ASSERT_EQ(st, MATX_OK);
  EXPECT_NEAR(out, sqrt(1.0 + 4.0) + sqrt(9.0 + 16.0), 1e-12);
  matx_vec_c64_destroy(&a, v);
}

TEST(vec_blas, norm2_f64) {
  matx_alloc_t a = matx_alloc_default();
  matx_vec_f64_t v = NULL;
  ASSERT_EQ(matx_vec_f64_create(&a, &v, NULL, 3), MATX_OK);
  v->data[0] = 1.0; v->data[1] = 2.0; v->data[2] = 3.0;

  matx_vec_backend_t blas = matx_vec_default();
  matx_double out = 0.0;
  matx_status_t st = matx_vec_norm2_f64(&blas, v, &out);
  if (st == MATX_ERR_NOT_SUPPORTED) { matx_vec_f64_destroy(&a, v); return; }
  ASSERT_EQ(st, MATX_OK);
  EXPECT_NEAR(out, sqrt(14.0), 1e-12);
  matx_vec_f64_destroy(&a, v);
}

TEST(vec_blas, norm2_c64) {
  matx_alloc_t a = matx_alloc_default();
  matx_vec_c64_t v = NULL;
  ASSERT_EQ(matx_vec_c64_create(&a, &v, NULL, 2), MATX_OK);
  v->data[0] = {1.0, 2.0}; v->data[1] = {3.0, 4.0};

  matx_vec_backend_t blas = matx_vec_default();
  matx_double out = 0.0;
  matx_status_t st = matx_vec_norm2_c64(&blas, v, &out);
  if (st == MATX_ERR_NOT_SUPPORTED) { matx_vec_c64_destroy(&a, v); return; }
  ASSERT_EQ(st, MATX_OK);
  EXPECT_NEAR(out, sqrt(5.0 + 25.0), 1e-12);
  matx_vec_c64_destroy(&a, v);
}

TEST(vec_blas, norminf_f64) {
  matx_alloc_t a = matx_alloc_default();
  matx_vec_f64_t v = NULL;
  ASSERT_EQ(matx_vec_f64_create(&a, &v, NULL, 4), MATX_OK);
  v->data[0] = 1.0; v->data[1] = -5.0; v->data[2] = 3.0; v->data[3] = 2.0;

  matx_vec_backend_t blas = matx_vec_default();
  matx_double out = 0.0;
  matx_status_t st = matx_vec_norminf_f64(&blas, v, &out);
  if (st == MATX_ERR_NOT_SUPPORTED) { matx_vec_f64_destroy(&a, v); return; }
  ASSERT_EQ(st, MATX_OK);
  EXPECT_NEAR(out, 5.0, 1e-12);
  matx_vec_f64_destroy(&a, v);
}

TEST(vec_blas, norminf_c64) {
  matx_alloc_t a = matx_alloc_default();
  matx_vec_c64_t v = NULL;
  ASSERT_EQ(matx_vec_c64_create(&a, &v, NULL, 3), MATX_OK);
  v->data[0] = {1.0, 0.0}; v->data[1] = {0.0, 5.0}; v->data[2] = {3.0, 0.0};

  matx_vec_backend_t blas = matx_vec_default();
  matx_double out = 0.0;
  matx_status_t st = matx_vec_norminf_c64(&blas, v, &out);
  if (st == MATX_ERR_NOT_SUPPORTED) { matx_vec_c64_destroy(&a, v); return; }
  ASSERT_EQ(st, MATX_OK);
  EXPECT_NEAR(out, 5.0, 1e-12);
  matx_vec_c64_destroy(&a, v);
}