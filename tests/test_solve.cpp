#include "matx_test_harness.h"

extern "C" {
#include "matx/matx_dense_solve.h"
#include "matx/matx_func.h"
#include "matx/matx_sparse_solve.h"
#include "matx/matx_types.h"
#include "matx/matx_types_internal.h"
}

/* 4x4 identity-like sparse system: A = diag(2,2,2,2) in CSC */
TEST(solve, sparse_real_4x4_factor_solve)
{
    /*matx_int64_t col_ptr[5] = { 0, 1, 2, 3, 4 };
	matx_int64_t row_ind[4] = { 0, 1, 2, 3 };
	matx_double values[4] = { 2.0, 2.0, 2.0, 2.0 };
	matx_csc_d_i8_t A = { 4, 4, 4, col_ptr, row_ind, values };*/

    matx_alloc_t a = matx_alloc_default();
    matx_int64_t nnz = 4;
    matx_int64_t coo_rows[4] = {0, 1, 2, 3};
    matx_int64_t coo_cols[4] = {0, 1, 2, 3};
    matx_double coo_values[4] = {2.0, 2.0, 2.0, 2.0};

    matx_coo_d_i8_t coo_A = NULL;
    matx_coo_sparse_d_i8_create(&a, &coo_A, 4, 4, nnz, coo_rows, coo_cols, coo_values);
    auto alloc = matx_alloc_default();
    matx_csc_sparse_d_i8_create(&alloc, &coo_A->handle_csc, 4, 4, nnz);
    double b[4] = {4.0, 6.0, 8.0, 10.0};
    double x[4] = {0.0, 0.0, 0.0, 0.0};

    matx_sparse_linsolve_t ls = matx_sparse_linsolve_default(matx_alloc_default());
    matx_factor_sparse_d_i8_t F; // = NULL;
    F.reserved = NULL;
    matx_status_t st = matx_factor_csc_d_i8(&ls, coo_A, &F);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    //ASSERT_NE(F, nullptr);

    st = matx_solve_csc_d_i8_factor(&ls, &F, b, x);
    matx_factor_csc_d_i8_destroy(&ls, &F);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(x[0], 2.0, 1e-10);
    EXPECT_NEAR(x[1], 3.0, 1e-10);
    EXPECT_NEAR(x[2], 4.0, 1e-10);
    EXPECT_NEAR(x[3], 5.0, 1e-10);
    matx_coo_sparse_d_i8_destroy(&alloc, coo_A);
}

TEST(solve, sparse_real_4x4_solve_one_shot)
{
    matx_alloc_t a = matx_alloc_default();
    //matx_int64_t col_ptr[5] = { 0, 1, 2, 3, 4 };
    //matx_int64_t row_ind[4] = { 0, 1, 2, 3 };
    //matx_double values[4] = { 2.0, 2.0, 2.0, 2.0 };
    //matx_csc_d_i8_t A = { 4, 4, 4, col_ptr, row_ind, values };
    matx_int64_t nnz = 4;
    matx_int64_t coo_rows[4] = {0, 1, 2, 3};
    matx_int64_t coo_cols[4] = {0, 1, 2, 3};
    matx_double coo_values[4] = {2.0, 2.0, 2.0, 2.0};

    matx_coo_d_i8_t coo_A = NULL;
    matx_coo_sparse_d_i8_create(&a, &coo_A, 4, 4, nnz, coo_rows, coo_cols, coo_values);
    auto alloc = matx_alloc_default();
    matx_csc_sparse_d_i8_create(&alloc, &coo_A->handle_csc, 4, 4, nnz);

    double b[4] = {2.0, 4.0, 6.0, 8.0};
    double x[4] = {0.0, 0.0, 0.0, 0.0};

    matx_sparse_linsolve_t ls = matx_sparse_linsolve_default(matx_alloc_default());
    matx_status_t st = matx_solve_csc_d_i8(&ls, coo_A, b, x);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(x[0], 1.0, 1e-10);
    EXPECT_NEAR(x[1], 2.0, 1e-10);
    EXPECT_NEAR(x[2], 3.0, 1e-10);
    EXPECT_NEAR(x[3], 4.0, 1e-10);
    matx_coo_sparse_d_i8_destroy(&alloc, coo_A);
}

TEST(solve, dense_real_4x4_factor_solve)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t A = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_COL_MAJOR, 4, 4, NULL), MATX_OK);
    /* Simple 4x4: identity * 2 */
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            A->data[i + j * 4] = (i == j) ? 2.0 : 0.0;

    double b[4] = {2.0, 4.0, 6.0, 8.0};
    double x[4] = {0.0, 0.0, 0.0, 0.0};

    matx_dense_linsolve_t ls = matx_dense_linsolve_default(matx_alloc_default());
    matx_factor_dense_d_i8_t* F = NULL;
    matx_status_t st = matx_factor_dense_d_i8(&ls, A, &F);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_dense_d_i8_destroy(&a, A);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    ASSERT_NE(F, nullptr);

    st = matx_solve_dense_d_i8_factor(&ls, F, b, x);
    matx_factor_dense_d_i8_destroy(&ls, F);
    matx_dense_d_i8_destroy(&a, A);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(x[0], 1.0, 1e-10);
    EXPECT_NEAR(x[1], 2.0, 1e-10);
    EXPECT_NEAR(x[2], 3.0, 1e-10);
    EXPECT_NEAR(x[3], 4.0, 1e-10);
}

TEST(solve, dense_real_4x4_solve_one_shot)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t A = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_COL_MAJOR, 4, 4, NULL), MATX_OK);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            A->data[i + j * 4] = (i == j) ? 3.0 : 0.0;

    double b[4] = {3.0, 6.0, 9.0, 12.0};
    double x[4] = {0.0, 0.0, 0.0, 0.0};

    matx_dense_linsolve_t ls = matx_dense_linsolve_default(matx_alloc_default());
    matx_status_t st = matx_solve_dense_d_i8(&ls, A, b, x);
    matx_dense_d_i8_destroy(&a, A);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(x[0], 1.0, 1e-10);
    EXPECT_NEAR(x[3], 4.0, 1e-10);
}

TEST(solve, dense_complex_4x4_factor_solve)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_z_i8_t A = NULL;
    ASSERT_EQ(matx_dense_z_i8_create(&a, &A, MATX_COL_MAJOR, 4, 4, NULL), MATX_OK);
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            A->data[i + j * 4].real = (i == j) ? 2.0 : 0.0;
            A->data[i + j * 4].imag = 0.0;
        }
    }

    matx_vec_z_i8_t b = NULL, x = NULL;
    ASSERT_EQ(matx_vec_z_i8_create(&a, &b, NULL, 4), MATX_OK);
    ASSERT_EQ(matx_vec_z_i8_create(&a, &x, NULL, 4), MATX_OK);
    b->data[0] = {4.0, 0.0};
    b->data[1] = {6.0, 0.0};
    b->data[2] = {8.0, 0.0};
    b->data[3] = {10.0, 0.0};

    matx_dense_linsolve_t ls = matx_dense_linsolve_default(matx_alloc_default());
    matx_factor_dense_z_i8_t* F = NULL;
    matx_status_t st = matx_factor_dense_z_i8(&ls, A, &F);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_vec_z_i8_destroy(&a, b);
        matx_vec_z_i8_destroy(&a, x);
        matx_dense_z_i8_destroy(&a, A);
        return;
    }
    ASSERT_EQ(st, MATX_OK);

    st = matx_solve_dense_z_i8_factor(&ls, F, b, x);
    matx_factor_dense_z_i8_destroy(&ls, F);
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(x->data[0].real, 2.0, 1e-10);
    EXPECT_NEAR(x->data[3].real, 5.0, 1e-10);

    matx_vec_z_i8_destroy(&a, b);
    matx_vec_z_i8_destroy(&a, x);
    matx_dense_z_i8_destroy(&a, A);
}

TEST(solve, dense_real_small_lu_row_major_pivot)
{
    matx_alloc_t alloc = matx_alloc_default();
    double matrix_values[9] = {0.0, 2.0, 1.0,
                               1.0, 1.0, 0.0,
                               2.0, 0.0, 1.0};
    matx_dense_d_i8_t A = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&alloc, &A, MATX_ROW_MAJOR, 3, 3,
                                     matrix_values), MATX_OK);
    const double rhs[3] = {7.0, 3.0, 5.0};
    double solution[3] = {0.0, 0.0, 0.0};
    matx_dense_linsolve_t solver = matx_dense_linsolve_default(matx_alloc_default());
    matx_factor_dense_d_i8_t* factor = NULL;
    matx_status_t status = matx_factor_dense_d_i8(&solver, A, &factor);
    if (status == MATX_ERR_NOT_SUPPORTED) {
        matx_dense_d_i8_destroy(&alloc, A);
        return;
    }
    ASSERT_EQ(status, MATX_OK);
    status = matx_solve_dense_d_i8_factor(&solver, factor, rhs, solution);
    ASSERT_EQ(status, MATX_OK);
    EXPECT_NEAR(solution[0], 1.0, 1e-12);
    EXPECT_NEAR(solution[1], 2.0, 1e-12);
    EXPECT_NEAR(solution[2], 3.0, 1e-12);

    double in_place[3] = {7.0, 3.0, 5.0};
    status = matx_solve_dense_d_i8_factor(&solver, factor, in_place, in_place);
    EXPECT_EQ(status, MATX_OK);
    EXPECT_NEAR(in_place[0], 1.0, 1e-12);
    EXPECT_NEAR(in_place[1], 2.0, 1e-12);
    EXPECT_NEAR(in_place[2], 3.0, 1e-12);
    matx_factor_dense_d_i8_destroy(&solver, factor);
    matx_dense_d_i8_destroy(&alloc, A);
}

TEST(solve, dense_complex_small_lu_row_major_pivot)
{
    matx_alloc_t alloc = matx_alloc_default();
    matx_dense_z_i8_t A = NULL;
    ASSERT_EQ(matx_dense_z_i8_create(&alloc, &A, MATX_ROW_MAJOR, 3, 3, NULL), MATX_OK);
    const matx_complex_d_t matrix_values[9] = {
        {0.0, 0.0}, {2.0, 1.0}, {0.0, 0.0},
        {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {2.0, -1.0}};
    for (size_t i = 0; i < 9; ++i) A->data[i] = matrix_values[i];

    matx_vec_z_i8_t rhs = NULL, solution = NULL;
    ASSERT_EQ(matx_vec_z_i8_create(&alloc, &rhs, NULL, 3), MATX_OK);
    ASSERT_EQ(matx_vec_z_i8_create(&alloc, &solution, NULL, 3), MATX_OK);
    rhs->data[0] = {5.0, 0.0};
    rhs->data[1] = {3.0, -0.5};
    rhs->data[2] = {6.25, -2.5};
    matx_dense_linsolve_t solver = matx_dense_linsolve_default(matx_alloc_default());
    matx_factor_dense_z_i8_t* factor = NULL;
    matx_status_t status = matx_factor_dense_z_i8(&solver, A, &factor);
    if (status == MATX_ERR_NOT_SUPPORTED) {
        matx_vec_z_i8_destroy(&alloc, rhs);
        matx_vec_z_i8_destroy(&alloc, solution);
        matx_dense_z_i8_destroy(&alloc, A);
        return;
    }
    ASSERT_EQ(status, MATX_OK);
    status = matx_solve_dense_z_i8_factor(&solver, factor, rhs, solution);
    ASSERT_EQ(status, MATX_OK);
    EXPECT_NEAR(solution->data[0].real, 1.0, 1e-12);
    EXPECT_NEAR(solution->data[0].imag, 0.5, 1e-12);
    EXPECT_NEAR(solution->data[1].real, 2.0, 1e-12);
    EXPECT_NEAR(solution->data[1].imag, -1.0, 1e-12);
    EXPECT_NEAR(solution->data[2].real, 3.0, 1e-12);
    EXPECT_NEAR(solution->data[2].imag, 0.25, 1e-12);

    status = matx_solve_dense_z_i8_factor(&solver, factor, rhs, rhs);
    EXPECT_EQ(status, MATX_OK);
    EXPECT_NEAR(rhs->data[0].real, 1.0, 1e-12);
    EXPECT_NEAR(rhs->data[0].imag, 0.5, 1e-12);
    EXPECT_NEAR(rhs->data[1].real, 2.0, 1e-12);
    EXPECT_NEAR(rhs->data[1].imag, -1.0, 1e-12);
    EXPECT_NEAR(rhs->data[2].real, 3.0, 1e-12);
    EXPECT_NEAR(rhs->data[2].imag, 0.25, 1e-12);
    matx_factor_dense_z_i8_destroy(&solver, factor);
    matx_vec_z_i8_destroy(&alloc, rhs);
    matx_vec_z_i8_destroy(&alloc, solution);
    matx_dense_z_i8_destroy(&alloc, A);
}

TEST(solve, sparse_complex_4x4_factor_solve)
{
    matx_alloc_t a = matx_alloc_default();
    matx_int64_t nnz = 4;
    matx_int64_t coo_rows[4] = {0, 1, 2, 3};
    matx_int64_t coo_cols[4] = {0, 1, 2, 3};
    matx_complex_d_t coo_values[4] = {{2.0, 0.0}, {2.0, 0.0}, {2.0, 0.0}, {2.0, 0.0}};

    matx_coo_z_i8_t coo_A = NULL;
    matx_coo_sparse_z_i8_create(&a, &coo_A, 4, 4, nnz, coo_rows, coo_cols, coo_values);
    auto alloc = matx_alloc_default();
    matx_csc_sparse_z_i8_create(&alloc, &coo_A->handle_csc, 4, 4, nnz);

    matx_vec_z_i8_t b = NULL, x = NULL;
    ASSERT_EQ(matx_vec_z_i8_create(&alloc, &b, NULL, 4), MATX_OK);
    ASSERT_EQ(matx_vec_z_i8_create(&alloc, &x, NULL, 4), MATX_OK);
    b->data[0] = {4.0, 0.0};
    b->data[1] = {6.0, 0.0};
    b->data[2] = {8.0, 0.0};
    b->data[3] = {10.0, 0.0};

    matx_sparse_linsolve_t ls = matx_sparse_linsolve_default(matx_alloc_default());
    matx_factor_sparse_z_i8_t F;
    F.reserved = NULL;
    matx_status_t st = matx_factor_csc_z_i8(&ls, coo_A, &F);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_vec_z_i8_destroy(&alloc, b);
        matx_vec_z_i8_destroy(&alloc, x);
        //matx_csc_sparse_z_i8_destroy(coo_A->handle_csc, &alloc);
        matx_coo_sparse_z_i8_destroy(&alloc, coo_A);
        return;
    }
    ASSERT_EQ(st, MATX_OK);

    st = matx_solve_csc_z_i8_factor(&ls, &F, b, x);
    matx_factor_csc_z_i8_destroy(&ls, &F);
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(x->data[0].real, 2.0, 1e-10);
    EXPECT_NEAR(x->data[3].real, 5.0, 1e-10);

    matx_vec_z_i8_destroy(&alloc, b);
    matx_vec_z_i8_destroy(&alloc, x);
    matx_coo_sparse_z_i8_destroy(&alloc, coo_A);
}

TEST(solve, sparse_complex_factor_solve_strided_vectors)
{
    matx_alloc_t alloc = matx_alloc_default();
    matx_int64_t rows[2] = {0, 1};
    matx_int64_t columns[2] = {0, 1};
    matx_complex_d_t values[2] = {{2.0, 0.0}, {4.0, 0.0}};
    matx_coo_z_i8_t matrix = NULL;
    ASSERT_EQ(matx_coo_sparse_z_i8_create(&alloc, &matrix, 2, 2, 2,
                                          rows, columns, values), MATX_OK);

    matx_complex_d_t rhs_storage[3] = {{2.0, 2.0}, {0.0, 0.0}, {8.0, -2.0}};
    matx_complex_d_t solution_storage[3] = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
    matx_vec_z_i8_t rhs = NULL, solution = NULL;
    ASSERT_EQ(matx_vec_z_i8_wrap(&alloc, &rhs, 2, 2, rhs_storage), MATX_OK);
    ASSERT_EQ(matx_vec_z_i8_wrap(&alloc, &solution, 2, 2, solution_storage), MATX_OK);

    const matx_sparse_linsolve_backend_kind_t backends[] = {
        MATX_LINSOLVE_BACKEND_UMFPACK,
        MATX_LINSOLVE_BACKEND_CXSPARSE,
        MATX_LINSOLVE_BACKEND_MUMPS,
        MATX_LINSOLVE_BACKEND_SUITESPARSE_KLU};
    int supported_backends = 0;
    for (matx_sparse_linsolve_backend_kind_t kind : backends) {
        matx_sparse_linsolve_t solver = matx_sparse_linsolve_by_type(kind, alloc);
        matx_factor_sparse_z_i8_t factor{};
        matx_status_t status = matx_factor_csc_z_i8(&solver, matrix, &factor);
        if (status == MATX_ERR_NOT_SUPPORTED) continue;
        ASSERT_EQ(status, MATX_OK);
        status = matx_solve_csc_z_i8_factor(&solver, &factor, rhs, solution);
        matx_factor_csc_z_i8_destroy(&solver, &factor);
        if (status == MATX_ERR_NOT_SUPPORTED) continue;
        ASSERT_EQ(status, MATX_OK);
        ++supported_backends;
        EXPECT_NEAR(solution_storage[0].real, 1.0, 1e-12);
        EXPECT_NEAR(solution_storage[0].imag, 1.0, 1e-12);
        EXPECT_NEAR(solution_storage[2].real, 2.0, 1e-12);
        EXPECT_NEAR(solution_storage[2].imag, -0.5, 1e-12);
        solution_storage[0] = {0.0, 0.0};
        solution_storage[2] = {0.0, 0.0};
    }
    EXPECT_TRUE(supported_backends >= 1);

    matx_vec_z_i8_destroy(&alloc, solution);
    matx_vec_z_i8_destroy(&alloc, rhs);
    matx_coo_sparse_z_i8_destroy(&alloc, matrix);
}

// ---- Cholesky tests ----
TEST(solve, chol_d_i8_3x3)
{
    // A = [[4,2,0],[2,5,2],[0,2,5]] — symmetric positive definite
    matx_alloc_t a = matx_alloc_default();
    double Adata[9] = {4, 2, 0, 2, 5, 2, 0, 2, 5};
    matx_dense_d_i8_t A = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_ROW_MAJOR, 3, 3, Adata), MATX_OK);
    double b[3] = {8.0, 16.0, 14.0};
    double x[3] = {0.0, 0.0, 0.0};
    matx_dense_linsolve_t ls = matx_dense_linsolve_default(matx_alloc_default());
    matx_status_t st = matx_solve_chol_d_i8_oneshot(&ls, A, MATX_UPPER, b, x);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_dense_d_i8_destroy(&a, A);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(x[0], 1.0, 1e-9);
    EXPECT_NEAR(x[1], 2.0, 1e-9);
    EXPECT_NEAR(x[2], 2.0, 1e-9);
    matx_dense_d_i8_destroy(&a, A);
}

// ---- GELS tests ----

TEST(solve, gels_d_i8_overdetermined)
{
    // 3x2 overdetermined system: A*x = b, least squares
    matx_alloc_t a = matx_alloc_default();
    // A = [[1,1],[1,2],[1,3]], b = [6,5,7] => x ~ [5, 0.5]
    double Adata[6] = {1, 1, 1, 2, 1, 3};
    matx_dense_d_i8_t A = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_ROW_MAJOR, 3, 2, Adata), MATX_OK);
    double b[3] = {6.0, 5.0, 7.0};
    double x[2] = {0.0, 0.0};
    matx_dense_linsolve_t ls = matx_dense_linsolve_default(matx_alloc_default());
    matx_status_t st = matx_gels_d_i8(&ls, A, b, x);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_dense_d_i8_destroy(&a, A);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    // least squares solution: x[0]=5, x[1]=0.5
    EXPECT_NEAR(x[0], 5.0, 1e-8);
    EXPECT_NEAR(x[1], 0.5, 1e-8);
    matx_dense_d_i8_destroy(&a, A);
}

// ---- SYEV tests ----

TEST(solve, syev_d_i8_2x2)
{
    // A = [[2,1],[1,2]], eigenvalues = {1, 3}
    matx_alloc_t a = matx_alloc_default();
    double Adata[4] = {2, 1, 1, 2};
    matx_dense_d_i8_t A = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_ROW_MAJOR, 2, 2, Adata), MATX_OK);
    matx_vec_d_i8_t evals = NULL;
    ASSERT_EQ(matx_vec_d_i8_create(&a, &evals, NULL, 2), MATX_OK);
    matx_dense_d_i8_t evecs = NULL;
    matx_dense_linsolve_t ls = matx_dense_linsolve_default(matx_alloc_default());
    matx_status_t st = matx_syev_d_i8(&ls, A, evals, &evecs);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_dense_d_i8_destroy(&a, A);
        matx_vec_d_i8_destroy(&a, evals);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(evals->data[0], 1.0, 1e-9);
    EXPECT_NEAR(evals->data[1], 3.0, 1e-9);
    ASSERT_NE(evecs, nullptr);
    matx_dense_d_i8_destroy(&ls.alloc, evecs);
    matx_dense_d_i8_destroy(&a, A);
    matx_vec_d_i8_destroy(&a, evals);
}

// ---- GESVD tests ----

TEST(solve, gesvd_d_i8_2x2)
{
    // A = [[3,0],[0,2]], singular values = {3, 2}
    matx_alloc_t a = matx_alloc_default();
    double Adata[4] = {3, 0, 0, 2};
    matx_dense_d_i8_t A = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_ROW_MAJOR, 2, 2, Adata), MATX_OK);
    matx_vec_d_i8_t S = NULL;
    ASSERT_EQ(matx_vec_d_i8_create(&a, &S, NULL, 2), MATX_OK);
    matx_dense_d_i8_t U = NULL, Vt = NULL;
    matx_dense_linsolve_t ls = matx_dense_linsolve_default(matx_alloc_default());
    matx_status_t st = matx_gesvd_d_i8(&ls, A, S, &U, &Vt);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_dense_d_i8_destroy(&a, A);
        matx_vec_d_i8_destroy(&a, S);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(S->data[0], 3.0, 1e-9);
    EXPECT_NEAR(S->data[1], 2.0, 1e-9);
    ASSERT_NE(U, nullptr);
    ASSERT_NE(Vt, nullptr);
    matx_dense_d_i8_destroy(&ls.alloc, U);
    matx_dense_d_i8_destroy(&ls.alloc, Vt);
    matx_dense_d_i8_destroy(&a, A);
    matx_vec_d_i8_destroy(&a, S);
}

TEST(solve, dense_complex_4x4_solve_one_shot)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_z_i8_t A = NULL;
    ASSERT_EQ(matx_dense_z_i8_create(&a, &A, MATX_COL_MAJOR, 4, 4, NULL), MATX_OK);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j) {
            A->data[i + j * 4].real = (i == j) ? 3.0 : 0.0;
            A->data[i + j * 4].imag = 0.0;
        }

    matx_vec_z_i8_t b = NULL, x = NULL;
    ASSERT_EQ(matx_vec_z_i8_create(&a, &b, NULL, 4), MATX_OK);
    ASSERT_EQ(matx_vec_z_i8_create(&a, &x, NULL, 4), MATX_OK);
    b->data[0] = {3.0, 0.0};
    b->data[1] = {6.0, 0.0};
    b->data[2] = {9.0, 0.0};
    b->data[3] = {12.0, 0.0};

    matx_dense_linsolve_t ls = matx_dense_linsolve_default(matx_alloc_default());
    matx_status_t st = matx_solve_dense_z_i8(&ls, A, b, x);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_vec_z_i8_destroy(&a, b);
        matx_vec_z_i8_destroy(&a, x);
        matx_dense_z_i8_destroy(&a, A);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(x->data[0].real, 1.0, 1e-10);
    EXPECT_NEAR(x->data[3].real, 4.0, 1e-10);

    matx_vec_z_i8_destroy(&a, b);
    matx_vec_z_i8_destroy(&a, x);
    matx_dense_z_i8_destroy(&a, A);
}

TEST(solve, sparse_complex_4x4_solve_one_shot)
{
    matx_alloc_t a = matx_alloc_default();
    matx_int64_t nnz = 4;
    matx_int64_t coo_rows[4] = {0, 1, 2, 3};
    matx_int64_t coo_cols[4] = {0, 1, 2, 3};
    matx_complex_d_t coo_values[4] = {{2.0, 0.0}, {2.0, 0.0}, {2.0, 0.0}, {2.0, 0.0}};

    matx_coo_z_i8_t coo_A = NULL;
    matx_coo_sparse_z_i8_create(&a, &coo_A, 4, 4, nnz, coo_rows, coo_cols, coo_values);
    auto alloc = matx_alloc_default();
    matx_csc_sparse_z_i8_create(&alloc, &coo_A->handle_csc, 4, 4, nnz);

    matx_vec_z_i8_t b = NULL, x = NULL;
    ASSERT_EQ(matx_vec_z_i8_create(&alloc, &b, NULL, 4), MATX_OK);
    ASSERT_EQ(matx_vec_z_i8_create(&alloc, &x, NULL, 4), MATX_OK);
    b->data[0] = {2.0, 0.0};
    b->data[1] = {4.0, 0.0};
    b->data[2] = {6.0, 0.0};
    b->data[3] = {8.0, 0.0};

    matx_sparse_linsolve_t ls = matx_sparse_linsolve_default(matx_alloc_default());
    matx_status_t st = matx_solve_csc_z_i8(&ls, coo_A, b, x);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_vec_z_i8_destroy(&alloc, b);
        matx_vec_z_i8_destroy(&alloc, x);
        matx_coo_sparse_z_i8_destroy(&alloc, coo_A);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(x->data[0].real, 1.0, 1e-10);
    EXPECT_NEAR(x->data[1].real, 2.0, 1e-10);
    EXPECT_NEAR(x->data[2].real, 3.0, 1e-10);
    EXPECT_NEAR(x->data[3].real, 4.0, 1e-10);

    matx_vec_z_i8_destroy(&alloc, b);
    matx_vec_z_i8_destroy(&alloc, x);
    matx_coo_sparse_z_i8_destroy(&alloc, coo_A);
}

TEST(solve, chol_z_i8_3x3)
{
    // A = [[4,2,0],[2,5,2],[0,2,5]] — Hermitian positive definite (all real)
    matx_alloc_t a = matx_alloc_default();
    matx_dense_z_i8_t A = NULL;
    ASSERT_EQ(matx_dense_z_i8_create(&a, &A, MATX_ROW_MAJOR, 3, 3, NULL), MATX_OK);
    double Adata[9] = {4, 2, 0, 2, 5, 2, 0, 2, 5};
    for (int i = 0; i < 9; ++i) {
        A->data[i].real = Adata[i];
        A->data[i].imag = 0.0;
    }
    matx_vec_z_i8_t b = NULL, x = NULL;
    ASSERT_EQ(matx_vec_z_i8_create(&a, &b, NULL, 3), MATX_OK);
    ASSERT_EQ(matx_vec_z_i8_create(&a, &x, NULL, 3), MATX_OK);
    b->data[0] = {8.0, 0.0};
    b->data[1] = {16.0, 0.0};
    b->data[2] = {14.0, 0.0};

    matx_dense_linsolve_t ls = matx_dense_linsolve_default(matx_alloc_default());
    matx_factor_dense_z_i8_t* F = NULL;
    matx_status_t st = matx_factor_chol_z_i8(&ls, A, MATX_UPPER, &F);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_vec_z_i8_destroy(&a, b);
        matx_vec_z_i8_destroy(&a, x);
        matx_dense_z_i8_destroy(&a, A);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    st = matx_solve_chol_z_i8(&ls, F, b, x);
    matx_factor_dense_z_i8_destroy(&ls, F);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_vec_z_i8_destroy(&a, b);
        matx_vec_z_i8_destroy(&a, x);
        matx_dense_z_i8_destroy(&a, A);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(x->data[0].real, 1.0, 1e-9);
    EXPECT_NEAR(x->data[1].real, 2.0, 1e-9);
    EXPECT_NEAR(x->data[2].real, 2.0, 1e-9);

    matx_vec_z_i8_destroy(&a, b);
    matx_vec_z_i8_destroy(&a, x);
    matx_dense_z_i8_destroy(&a, A);
}

TEST(solve, gels_z_i8_overdetermined)
{
    // 3x2 overdetermined: A*x = b, least squares
    matx_alloc_t a = matx_alloc_default();
    matx_dense_z_i8_t A = NULL;
    ASSERT_EQ(matx_dense_z_i8_create(&a, &A, MATX_ROW_MAJOR, 3, 2, NULL), MATX_OK);
    double Adata[6] = {1, 1, 1, 2, 1, 3};
    for (int i = 0; i < 6; ++i) {
        A->data[i].real = Adata[i];
        A->data[i].imag = 0.0;
    }
    matx_vec_z_i8_t b = NULL, x = NULL;
    ASSERT_EQ(matx_vec_z_i8_create(&a, &b, NULL, 3), MATX_OK);
    ASSERT_EQ(matx_vec_z_i8_create(&a, &x, NULL, 2), MATX_OK);
    b->data[0] = {6.0, 0.0};
    b->data[1] = {5.0, 0.0};
    b->data[2] = {7.0, 0.0};

    matx_dense_linsolve_t ls = matx_dense_linsolve_default(matx_alloc_default());
    matx_status_t st = matx_gels_z_i8(&ls, A, b, x);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_vec_z_i8_destroy(&a, b);
        matx_vec_z_i8_destroy(&a, x);
        matx_dense_z_i8_destroy(&a, A);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(x->data[0].real, 5.0, 1e-8);
    EXPECT_NEAR(x->data[1].real, 0.5, 1e-8);

    matx_vec_z_i8_destroy(&a, b);
    matx_vec_z_i8_destroy(&a, x);
    matx_dense_z_i8_destroy(&a, A);
}

TEST(solve, gesvd_z_i8_2x2)
{
    // A = [[3,0],[0,2]], singular values = {3, 2}
    matx_alloc_t a = matx_alloc_default();
    matx_dense_z_i8_t A = NULL;
    ASSERT_EQ(matx_dense_z_i8_create(&a, &A, MATX_ROW_MAJOR, 2, 2, NULL), MATX_OK);
    A->data[0] = {3, 0};
    A->data[1] = {0, 0};
    A->data[2] = {0, 0};
    A->data[3] = {2, 0};

    matx_vec_d_i8_t S = NULL;
    ASSERT_EQ(matx_vec_d_i8_create(&a, &S, NULL, 2), MATX_OK);
    matx_dense_z_i8_t U = NULL, Vt = NULL;
    matx_dense_linsolve_t ls = matx_dense_linsolve_default(matx_alloc_default());
    matx_status_t st = matx_gesvd_z_i8(&ls, A, S, &U, &Vt);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_dense_z_i8_destroy(&a, A);
        matx_vec_d_i8_destroy(&a, S);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(S->data[0], 3.0, 1e-9);
    EXPECT_NEAR(S->data[1], 2.0, 1e-9);
    ASSERT_NE(U, nullptr);
    ASSERT_NE(Vt, nullptr);
    matx_dense_z_i8_destroy(&ls.alloc, U);
    matx_dense_z_i8_destroy(&ls.alloc, Vt);
    matx_dense_z_i8_destroy(&a, A);
    matx_vec_d_i8_destroy(&a, S);
}
