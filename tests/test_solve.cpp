#include <gtest/gtest.h>

extern "C" {
#include "matx/matx_types.h"
#include "matx/matx_func.h"
#include "matx/matx_sparse_solve.h"
#include "matx/matx_dense_solve.h"
#include "matx/matx_types_internal.h"
}

/* 4x4 identity-like sparse system: A = diag(2,2,2,2) in CSC */
TEST(solve, sparse_real_4x4_factor_solve) {
	/*matx_int64_t col_ptr[5] = { 0, 1, 2, 3, 4 };
	matx_int64_t row_ind[4] = { 0, 1, 2, 3 };
	matx_double values[4] = { 2.0, 2.0, 2.0, 2.0 };
	matx_csc_f64_t A = { 4, 4, 4, col_ptr, row_ind, values };*/

	matx_alloc_t a = matx_alloc_default();
	matx_int64_t nnz = 4;
	matx_int64_t coo_rows[4] = { 0, 1, 2, 3 };
	matx_int64_t coo_cols[4] = { 0, 1, 2, 3 };
	matx_double coo_values[4] = { 2.0, 2.0, 2.0, 2.0 };

	matx_coo_f64_t coo_A = NULL;
	matx_coo_sparse_f64_create(&a, &coo_A, 4, 4, nnz, coo_rows, coo_cols, coo_values);
	auto alloc = matx_alloc_default();
	matx_csc_sparse_f64_create(&alloc, &coo_A->handle_csc, 4, 4, nnz);
	double b[4] = { 4.0, 6.0, 8.0, 10.0 };
	double x[4] = { 0.0, 0.0, 0.0, 0.0 };

	matx_sparse_linsolve_t ls = matx_sparse_linsolve_default();
	matx_factor_sparse_f64_t F;// = NULL;
	F.reserved = NULL;
	matx_status_t st = matx_factor_csc_f64(&ls, coo_A, &F);
	if (st == MATX_ERR_NOT_SUPPORTED) {
		return;
	}
	ASSERT_EQ(st, MATX_OK);
	//ASSERT_NE(F, nullptr);

	st = matx_solve_csc_f64_factor(&ls, &F, b, x);
	matx_factor_csc_f64_destroy(&ls, &F);
	if (st == MATX_ERR_NOT_SUPPORTED) {
		return;
	}
	ASSERT_EQ(st, MATX_OK);
	EXPECT_NEAR(x[0], 2.0, 1e-10);
	EXPECT_NEAR(x[1], 3.0, 1e-10);
	EXPECT_NEAR(x[2], 4.0, 1e-10);
	EXPECT_NEAR(x[3], 5.0, 1e-10);
	matx_coo_sparse_f64_destroy(&alloc, coo_A);
}

TEST(solve, sparse_real_4x4_solve_one_shot) {
	matx_alloc_t a = matx_alloc_default();
	//matx_int64_t col_ptr[5] = { 0, 1, 2, 3, 4 };
	//matx_int64_t row_ind[4] = { 0, 1, 2, 3 };
	//matx_double values[4] = { 2.0, 2.0, 2.0, 2.0 };
	//matx_csc_f64_t A = { 4, 4, 4, col_ptr, row_ind, values };
	matx_int64_t nnz = 4;
	matx_int64_t coo_rows[4] = { 0, 1, 2, 3 };
	matx_int64_t coo_cols[4] = { 0, 1, 2, 3 };
	matx_double coo_values[4] = { 2.0, 2.0, 2.0, 2.0 };

	matx_coo_f64_t coo_A = NULL;
	matx_coo_sparse_f64_create(&a, &coo_A, 4, 4, nnz, coo_rows, coo_cols, coo_values);
	auto alloc = matx_alloc_default();
	matx_csc_sparse_f64_create(&alloc, &coo_A->handle_csc, 4, 4, nnz);

	double b[4] = { 2.0, 4.0, 6.0, 8.0 };
	double x[4] = { 0.0, 0.0, 0.0, 0.0 };

	matx_sparse_linsolve_t ls = matx_sparse_linsolve_default();
	matx_status_t st = matx_solve_csc_f64(&ls, coo_A, b, x);
	if (st == MATX_ERR_NOT_SUPPORTED) {
		return;
	}
	ASSERT_EQ(st, MATX_OK);
	EXPECT_NEAR(x[0], 1.0, 1e-10);
	EXPECT_NEAR(x[1], 2.0, 1e-10);
	EXPECT_NEAR(x[2], 3.0, 1e-10);
	EXPECT_NEAR(x[3], 4.0, 1e-10);
	matx_coo_sparse_f64_destroy(&alloc, coo_A);
}

TEST(solve, dense_real_4x4_factor_solve) {
	matx_alloc_t a = matx_alloc_default();
	matx_dense_f64_t A = NULL;
	ASSERT_EQ(matx_dense_f64_create(&a, &A, MATX_COL_MAJOR, 4, 4, NULL), MATX_OK);
	/* Simple 4x4: identity * 2 */
	for (size_t i = 0; i < 4; ++i)
		for (size_t j = 0; j < 4; ++j)
			A->data[i + j * 4] = (i == j) ? 2.0 : 0.0;

	double b[4] = { 2.0, 4.0, 6.0, 8.0 };
	double x[4] = { 0.0, 0.0, 0.0, 0.0 };

	matx_dense_linsolve_t ls = matx_dense_linsolve_default();
	matx_factor_dense_f64_t* F = NULL;
	matx_status_t st = matx_factor_dense_f64(&ls, A, &F);
	if (st == MATX_ERR_NOT_SUPPORTED) {
		matx_dense_f64_destroy(&a, A);
		return;
	}
	ASSERT_EQ(st, MATX_OK);
	ASSERT_NE(F, nullptr);

	st = matx_solve_dense_f64_factor(&ls, F, b, x);
	matx_factor_dense_f64_destroy(&ls, F);
	matx_dense_f64_destroy(&a, A);
	if (st == MATX_ERR_NOT_SUPPORTED) {
		return;
	}
	ASSERT_EQ(st, MATX_OK);
	EXPECT_NEAR(x[0], 1.0, 1e-10);
	EXPECT_NEAR(x[1], 2.0, 1e-10);
	EXPECT_NEAR(x[2], 3.0, 1e-10);
	EXPECT_NEAR(x[3], 4.0, 1e-10);
}

TEST(solve, dense_real_4x4_solve_one_shot) {
	matx_alloc_t a = matx_alloc_default();
	matx_dense_f64_t A = NULL;
	ASSERT_EQ(matx_dense_f64_create(&a, &A, MATX_COL_MAJOR, 4, 4, NULL), MATX_OK);
	for (size_t i = 0; i < 4; ++i)
		for (size_t j = 0; j < 4; ++j)
			A->data[i + j * 4] = (i == j) ? 3.0 : 0.0;

	double b[4] = { 3.0, 6.0, 9.0, 12.0 };
	double x[4] = { 0.0, 0.0, 0.0, 0.0 };

	matx_dense_linsolve_t ls = matx_dense_linsolve_default();
	matx_status_t st = matx_solve_dense_f64(&ls, A, b, x);
	matx_dense_f64_destroy(&a, A);
	if (st == MATX_ERR_NOT_SUPPORTED) {
		return;
	}
	ASSERT_EQ(st, MATX_OK);
	EXPECT_NEAR(x[0], 1.0, 1e-10);
	EXPECT_NEAR(x[3], 4.0, 1e-10);
}

TEST(solve, dense_complex_4x4_factor_solve) {
	matx_alloc_t a = matx_alloc_default();
	matx_dense_c64_t A = NULL;
	ASSERT_EQ(matx_dense_c64_create(&a, &A, MATX_COL_MAJOR, 4, 4, NULL), MATX_OK);
	for (size_t i = 0; i < 4; ++i) {
		for (size_t j = 0; j < 4; ++j) {
			A->data[i + j * 4].real = (i == j) ? 2.0 : 0.0;
			A->data[i + j * 4].imag = 0.0;
		}
	}

	matx_vec_c64_t b = NULL, x = NULL;
	ASSERT_EQ(matx_vec_c64_create(&a, &b, NULL, 4), MATX_OK);
	ASSERT_EQ(matx_vec_c64_create(&a, &x, NULL, 4), MATX_OK);
	b->data[0] = {4.0, 0.0};
	b->data[1] = {6.0, 0.0};
	b->data[2] = {8.0, 0.0};
	b->data[3] = {10.0, 0.0};

	matx_dense_linsolve_t ls = matx_dense_linsolve_default();
	matx_factor_dense_c64_t* F = NULL;
	matx_status_t st = matx_factor_dense_c64(&ls, A, &F);
	if (st == MATX_ERR_NOT_SUPPORTED) {
		matx_vec_c64_destroy(&a, b);
		matx_vec_c64_destroy(&a, x);
		matx_dense_c64_destroy(&a, A);
		return;
	}
	ASSERT_EQ(st, MATX_OK);

	st = matx_solve_dense_c64_factor(&ls, F, b, x);
	matx_factor_dense_c64_destroy(&ls, F);
	ASSERT_EQ(st, MATX_OK);
	EXPECT_NEAR(x->data[0].real, 2.0, 1e-10);
	EXPECT_NEAR(x->data[3].real, 5.0, 1e-10);

	matx_vec_c64_destroy(&a, b);
	matx_vec_c64_destroy(&a, x);
	matx_dense_c64_destroy(&a, A);
}

TEST(solve, sparse_complex_4x4_factor_solve) {
	matx_alloc_t a = matx_alloc_default();
	matx_int64_t nnz = 4;
	matx_int64_t coo_rows[4] = { 0, 1, 2, 3 };
	matx_int64_t coo_cols[4] = { 0, 1, 2, 3 };
	matx_complex_f64_t coo_values[4] = { {2.0,0.0},{2.0,0.0},{2.0,0.0},{2.0,0.0} };

	matx_coo_c64_t coo_A = NULL;
	matx_coo_sparse_c64_create(&a, &coo_A, 4, 4, nnz, coo_rows, coo_cols, coo_values);
	auto alloc = matx_alloc_default();
	matx_csc_sparse_c64_create(&alloc, &coo_A->handle_csc, 4, 4, nnz);

	matx_vec_c64_t b = NULL, x = NULL;
	ASSERT_EQ(matx_vec_c64_create(&alloc, &b, NULL, 4), MATX_OK);
	ASSERT_EQ(matx_vec_c64_create(&alloc, &x, NULL, 4), MATX_OK);
	b->data[0] = {4.0, 0.0};
	b->data[1] = {6.0, 0.0};
	b->data[2] = {8.0, 0.0};
	b->data[3] = {10.0, 0.0};

	matx_sparse_linsolve_t ls = matx_sparse_linsolve_default();
	matx_factor_sparse_c64_t F;
	F.reserved = NULL;
	matx_status_t st = matx_factor_csc_c64(&ls, coo_A, &F);
	if (st == MATX_ERR_NOT_SUPPORTED) {
		matx_vec_c64_destroy(&alloc, b);
		matx_vec_c64_destroy(&alloc, x);
		//matx_csc_sparse_c64_destroy(coo_A->handle_csc, &alloc);
		matx_coo_sparse_c64_destroy(&alloc, coo_A);
		return;
	}
	ASSERT_EQ(st, MATX_OK);

	st = matx_solve_csc_c64_factor(&ls, &F, b, x);
	matx_factor_csc_c64_destroy(&ls, &F);
	ASSERT_EQ(st, MATX_OK);
	EXPECT_NEAR(x->data[0].real, 2.0, 1e-10);
	EXPECT_NEAR(x->data[3].real, 5.0, 1e-10);

	matx_vec_c64_destroy(&alloc, b);
	matx_vec_c64_destroy(&alloc, x);
	matx_coo_sparse_c64_destroy(&alloc, coo_A);
}

// ---- Cholesky tests ----
TEST(solve, chol_f64_3x3) {
	// A = [[4,2,0],[2,5,2],[0,2,5]] — symmetric positive definite
	matx_alloc_t a = matx_alloc_default();
	double Adata[9] = {4,2,0, 2,5,2, 0,2,5};
	matx_dense_f64_t A = NULL;
	ASSERT_EQ(matx_dense_f64_create(&a, &A, MATX_ROW_MAJOR, 3, 3, Adata), MATX_OK);
	double b[3] = {8.0, 16.0, 14.0};
	double x[3] = {0.0, 0.0, 0.0};
	matx_dense_linsolve_t ls = matx_dense_linsolve_default();
	matx_status_t st = matx_solve_chol_f64_oneshot(&ls, A, 'U', b, x);
	if (st == MATX_ERR_NOT_SUPPORTED) { matx_dense_f64_destroy(&a, A); return; }
	ASSERT_EQ(st, MATX_OK);
	EXPECT_NEAR(x[0], 1.0, 1e-9);
	EXPECT_NEAR(x[1], 2.0, 1e-9);
	EXPECT_NEAR(x[2], 2.0, 1e-9);
	matx_dense_f64_destroy(&a, A);
}

// ---- GELS tests ----

TEST(solve, gels_f64_overdetermined) {
	// 3x2 overdetermined system: A*x = b, least squares
	matx_alloc_t a = matx_alloc_default();
	// A = [[1,1],[1,2],[1,3]], b = [6,5,7] => x ~ [5, 0.5]
	double Adata[6] = {1,1, 1,2, 1,3};
	matx_dense_f64_t A = NULL;
	ASSERT_EQ(matx_dense_f64_create(&a, &A, MATX_ROW_MAJOR, 3, 2, Adata), MATX_OK);
	double b[3] = {6.0, 5.0, 7.0};
	double x[2] = {0.0, 0.0};
	matx_dense_linsolve_t ls = matx_dense_linsolve_default();
	matx_status_t st = matx_gels_f64(&ls, A, b, x);
	if (st == MATX_ERR_NOT_SUPPORTED) { matx_dense_f64_destroy(&a, A); return; }
	ASSERT_EQ(st, MATX_OK);
	// least squares solution: x[0]=5, x[1]=0.5
	EXPECT_NEAR(x[0], 5.0, 1e-8);
	EXPECT_NEAR(x[1], 0.5, 1e-8);
	matx_dense_f64_destroy(&a, A);
}

// ---- SYEV tests ----

TEST(solve, syev_f64_2x2) {
	// A = [[2,1],[1,2]], eigenvalues = {1, 3}
	matx_alloc_t a = matx_alloc_default();
	double Adata[4] = {2,1, 1,2};
	matx_dense_f64_t A = NULL;
	ASSERT_EQ(matx_dense_f64_create(&a, &A, MATX_ROW_MAJOR, 2, 2, Adata), MATX_OK);
	matx_vec_f64_t evals = NULL;
	ASSERT_EQ(matx_vec_f64_create(&a, &evals, NULL, 2), MATX_OK);
	matx_dense_f64_t evecs = NULL;
	matx_dense_linsolve_t ls = matx_dense_linsolve_default();
	matx_status_t st = matx_syev_f64(&ls, A, evals, &evecs);
	if (st == MATX_ERR_NOT_SUPPORTED) {
		matx_dense_f64_destroy(&a, A);
		matx_vec_f64_destroy(&a, evals);
		return;
	}
	ASSERT_EQ(st, MATX_OK);
	EXPECT_NEAR(evals->data[0], 1.0, 1e-9);
	EXPECT_NEAR(evals->data[1], 3.0, 1e-9);
	ASSERT_NE(evecs, nullptr);
	// free evecs with default alloc (allocated with malloc)
	free(evecs->data);
	free(evecs);
	matx_dense_f64_destroy(&a, A);
	matx_vec_f64_destroy(&a, evals);
}

// ---- GESVD tests ----

TEST(solve, gesvd_f64_2x2) {
	// A = [[3,0],[0,2]], singular values = {3, 2}
	matx_alloc_t a = matx_alloc_default();
	double Adata[4] = {3,0, 0,2};
	matx_dense_f64_t A = NULL;
	ASSERT_EQ(matx_dense_f64_create(&a, &A, MATX_ROW_MAJOR, 2, 2, Adata), MATX_OK);
	matx_vec_f64_t S = NULL;
	ASSERT_EQ(matx_vec_f64_create(&a, &S, NULL, 2), MATX_OK);
	matx_dense_f64_t U = NULL, Vt = NULL;
	matx_dense_linsolve_t ls = matx_dense_linsolve_default();
	matx_status_t st = matx_gesvd_f64(&ls, A, S, &U, &Vt);
	if (st == MATX_ERR_NOT_SUPPORTED) {
		matx_dense_f64_destroy(&a, A);
		matx_vec_f64_destroy(&a, S);
		return;
	}
	ASSERT_EQ(st, MATX_OK);
	EXPECT_NEAR(S->data[0], 3.0, 1e-9);
	EXPECT_NEAR(S->data[1], 2.0, 1e-9);
	ASSERT_NE(U, nullptr);
	ASSERT_NE(Vt, nullptr);
	free(U->data); free(U);
	free(Vt->data); free(Vt);
	matx_dense_f64_destroy(&a, A);
	matx_vec_f64_destroy(&a, S);
}
