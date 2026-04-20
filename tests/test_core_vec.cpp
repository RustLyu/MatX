#include <gtest/gtest.h>

extern "C" {
#include "matx/matx_types.h"
#include "matx/matx_func.h"
#include "matx/matx_types_internal.h"
}

TEST(core_vec, vec_f64_create_destroy) {
	matx_alloc_t a = matx_alloc_default();
	matx_vec_f64_t v = NULL;
	ASSERT_EQ(matx_vec_f64_create(&a, &v, NULL, 4), MATX_OK);
	ASSERT_NE(v->data, nullptr);
	EXPECT_EQ(v->n, 4u);
	EXPECT_EQ(v->stride, 1u);
	matx_vec_f64_destroy(&a, v);
	EXPECT_EQ(v->data, nullptr);
}

TEST(core_vec, vec_f64_wrap) {
	matx_alloc_t a = matx_alloc_default();
	double buf[4] = { 1.0, 2.0, 3.0, 4.0 };
	matx_vec_f64_t v = NULL;
	ASSERT_EQ(matx_vec_f64_wrap(&a, &v, 4, 1, buf), MATX_OK);
	EXPECT_EQ(v->data, buf);
	EXPECT_EQ(v->n, 4u);
	matx_vec_f64_destroy(nullptr, v);
}

TEST(core_vec, vec_c64_create_destroy) {
	matx_alloc_t a = matx_alloc_default();
	matx_vec_c64_t v = NULL;
	ASSERT_EQ(matx_vec_c64_create(&a, &v, NULL, 4), MATX_OK);
	ASSERT_NE(v->data, nullptr);
	EXPECT_EQ(v->n, 4u);
	matx_vec_c64_destroy(&a, v);
	EXPECT_EQ(v->data, nullptr);
}

TEST(core_vec, vec_c64_wrap) {
	matx_alloc_t a = matx_alloc_default();
	matx_complex_f64 buf[4] = { {1.0, 0.0}, {0.0, 1.0}, {2.0, -1.0}, {0.0, 0.0} };
	matx_vec_c64_t v = NULL;
	ASSERT_EQ(matx_vec_c64_wrap(&a, &v, 4, 1, buf), MATX_OK);
	EXPECT_EQ(v->data, buf);
	matx_vec_c64_destroy(nullptr, v);
}
