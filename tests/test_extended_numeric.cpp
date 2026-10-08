#include "matx_test_harness.h"

#include <cmath>

extern "C" {
#include "matx/matx_dense_compute.h"
#include "matx/matx_dense_solve.h"
#include "matx/matx_func.h"
#include "matx/matx_types.h"
#include "matx/matx_types_internal.h"
}

static size_t dense_index(matx_layout_t layout, matx_int64_t stride,
                          matx_int64_t row, matx_int64_t col)
{
    return layout == MATX_COL_MAJOR ? (size_t) (row + col * stride)
                                    : (size_t) (col + row * stride);
}

TEST(solve_extended, chol_d_factor_and_solve)
{
    matx_alloc_t alloc = matx_alloc_default();
    matx_dense_d_i8_t A = NULL;
    double values[4] = {4.0, 1.0, 1.0, 3.0};
    ASSERT_EQ(matx_dense_d_i8_create(&alloc, &A, MATX_COL_MAJOR, 2, 2, values), MATX_OK);

    matx_dense_linsolve_t ls = matx_dense_linsolve_default(matx_alloc_default());
    matx_factor_dense_d_i8_t* factor = NULL;
    matx_status_t st = matx_factor_chol_d_i8(&ls, A, MATX_LOWER, &factor);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_dense_d_i8_destroy(&alloc, A);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    const double b[2] = {6.0, 7.0};
    double x[2] = {};
    st = matx_solve_chol_d_i8(&ls, factor, b, x);
    matx_factor_dense_d_i8_destroy(&ls, factor);
    matx_dense_d_i8_destroy(&alloc, A);
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(x[0], 1.0, 1e-10);
    EXPECT_NEAR(x[1], 2.0, 1e-10);
}

TEST(solve_extended, syev_z_hermitian)
{
    matx_alloc_t alloc = matx_alloc_default();
    matx_dense_z_i8_t A = NULL;
    ASSERT_EQ(matx_dense_z_i8_create(&alloc, &A, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);
    A->data[0] = {2.0, 0.0};
    A->data[1] = {1.0, -1.0};
    A->data[2] = {1.0, 1.0};
    A->data[3] = {2.0, 0.0};
    matx_vec_d_i8_t eigenvalues = NULL;
    ASSERT_EQ(matx_vec_d_i8_create(&alloc, &eigenvalues, NULL, 2), MATX_OK);
    matx_dense_z_i8_t eigenvectors = NULL;
    matx_dense_linsolve_t ls = matx_dense_linsolve_default(matx_alloc_default());
    matx_status_t st = matx_syev_z_i8(&ls, A, eigenvalues, &eigenvectors);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_vec_d_i8_destroy(&alloc, eigenvalues);
        matx_dense_z_i8_destroy(&alloc, A);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(eigenvalues->data[0], 2.0 - std::sqrt(2.0), 1e-10);
    EXPECT_NEAR(eigenvalues->data[1], 2.0 + std::sqrt(2.0), 1e-10);
    ASSERT_NE(eigenvectors, nullptr);
    matx_dense_z_i8_destroy(&ls.alloc, eigenvectors);
    matx_vec_d_i8_destroy(&alloc, eigenvalues);
    matx_dense_z_i8_destroy(&alloc, A);
}

TEST(solve_extended, geev_d_complex_pair)
{
    matx_alloc_t alloc = matx_alloc_default();
    matx_dense_d_i8_t A = NULL;
    double values[4] = {0.0, 1.0, -1.0, 0.0};
    ASSERT_EQ(matx_dense_d_i8_create(&alloc, &A, MATX_COL_MAJOR, 2, 2, values), MATX_OK);
    matx_vec_z_i8_t eigenvalues = NULL;
    ASSERT_EQ(matx_vec_z_i8_create(&alloc, &eigenvalues, NULL, 2), MATX_OK);
    matx_dense_linsolve_t ls = matx_dense_linsolve_default(matx_alloc_default());
    matx_status_t st = matx_geev_d_i8(&ls, A, eigenvalues, NULL, NULL);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_vec_z_i8_destroy(&alloc, eigenvalues);
        matx_dense_d_i8_destroy(&alloc, A);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(eigenvalues->data[0].real, 0.0, 1e-10);
    EXPECT_NEAR(eigenvalues->data[1].real, 0.0, 1e-10);
    EXPECT_NEAR(std::abs(eigenvalues->data[0].imag), 1.0, 1e-10);
    EXPECT_NEAR(std::abs(eigenvalues->data[1].imag), 1.0, 1e-10);
    EXPECT_NEAR(eigenvalues->data[0].imag + eigenvalues->data[1].imag, 0.0, 1e-10);
    matx_vec_z_i8_destroy(&alloc, eigenvalues);
    matx_dense_d_i8_destroy(&alloc, A);
}

TEST(solve_extended, geev_z_eigenvalues_and_vectors)
{
    matx_alloc_t alloc = matx_alloc_default();
    matx_dense_z_i8_t A = NULL;
    ASSERT_EQ(matx_dense_z_i8_create(&alloc, &A, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);
    A->data[0] = {2.0, 1.0};
    A->data[1] = {0.0, 0.0};
    A->data[2] = {0.0, 0.0};
    A->data[3] = {3.0, -2.0};
    matx_vec_z_i8_t eigenvalues = NULL;
    ASSERT_EQ(matx_vec_z_i8_create(&alloc, &eigenvalues, NULL, 2), MATX_OK);
    matx_dense_z_i8_t right = NULL;
    matx_dense_linsolve_t ls = matx_dense_linsolve_default(matx_alloc_default());
    matx_status_t st = matx_geev_z_i8(&ls, A, eigenvalues, &right, NULL);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_vec_z_i8_destroy(&alloc, eigenvalues);
        matx_dense_z_i8_destroy(&alloc, A);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(eigenvalues->data[0].real, 2.0, 1e-10);
    EXPECT_NEAR(eigenvalues->data[0].imag, 1.0, 1e-10);
    EXPECT_NEAR(eigenvalues->data[1].real, 3.0, 1e-10);
    EXPECT_NEAR(eigenvalues->data[1].imag, -2.0, 1e-10);
    ASSERT_NE(right, nullptr);
    EXPECT_EQ(right->nrows, 2);
    EXPECT_EQ(right->ncols, 2);
    matx_dense_z_i8_destroy(&ls.alloc, right);
    matx_vec_z_i8_destroy(&alloc, eigenvalues);
    matx_dense_z_i8_destroy(&alloc, A);
}

TEST(solve_extended, qr_z_reconstructs_input)
{
    matx_alloc_t alloc = matx_alloc_default();
    matx_dense_z_i8_t A = NULL, Q = NULL, R = NULL;
    ASSERT_EQ(matx_dense_z_i8_create(&alloc, &A, MATX_COL_MAJOR, 3, 2, NULL), MATX_OK);
    const matx_complex_d_t values[6] = {{1.0, 1.0}, {2.0, -1.0}, {0.0, 0.0},
                                        {2.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}};
    for (size_t i = 0; i < 6; ++i)
        A->data[i] = values[i];
    matx_dense_linsolve_t ls = matx_dense_linsolve_default(matx_alloc_default());
    matx_status_t st = matx_qr_z_i8(&ls, A, &Q, &R);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_dense_z_i8_destroy(&alloc, A);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    ASSERT_NE(Q, nullptr);
    ASSERT_NE(R, nullptr);
    ASSERT_EQ(Q->nrows, 3);
    ASSERT_EQ(Q->ncols, 2);
    ASSERT_EQ(R->nrows, 2);
    ASSERT_EQ(R->ncols, 2);
    for (matx_int64_t row = 0; row < 3; ++row) {
        for (matx_int64_t col = 0; col < 2; ++col) {
            double real = 0.0, imag = 0.0;
            for (matx_int64_t k = 0; k < 2; ++k) {
                const auto q = Q->data[dense_index(Q->layout, Q->stride, row, k)];
                const auto r = R->data[dense_index(R->layout, R->stride, k, col)];
                real += q.real * r.real - q.imag * r.imag;
                imag += q.real * r.imag + q.imag * r.real;
            }
            const auto expected = A->data[dense_index(A->layout, A->stride, row, col)];
            EXPECT_NEAR(real, expected.real, 1e-10);
            EXPECT_NEAR(imag, expected.imag, 1e-10);
        }
    }
    matx_dense_z_i8_destroy(&ls.alloc, Q);
    matx_dense_z_i8_destroy(&ls.alloc, R);
    matx_dense_z_i8_destroy(&alloc, A);
}

TEST(solve_extended, complex_determinant_and_condition_number)
{
    matx_alloc_t alloc = matx_alloc_default();
    matx_dense_z_i8_t A = NULL, diagonal = NULL;
    ASSERT_EQ(matx_dense_z_i8_create(&alloc, &A, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);
    A->data[0] = {1.0, 1.0};
    A->data[1] = {3.0, 0.0};
    A->data[2] = {2.0, 0.0};
    A->data[3] = {4.0, -1.0};
    matx_dense_linsolve_t ls = matx_dense_linsolve_default(matx_alloc_default());
    matx_complex_d_t det = {};
    matx_status_t st = matx_det_dense_z_i8(&ls, A, &det);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_dense_z_i8_destroy(&alloc, A);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(det.real, -1.0, 1e-10);
    EXPECT_NEAR(det.imag, 3.0, 1e-10);

    ASSERT_EQ(matx_dense_z_i8_create(&alloc, &diagonal, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);
    diagonal->data[0] = {1.0, 0.0};
    diagonal->data[1] = {0.0, 0.0};
    diagonal->data[2] = {0.0, 0.0};
    diagonal->data[3] = {10.0, 0.0};
    double condition = 0.0;
    st = matx_cond_dense_z_i8(&ls, diagonal, &condition);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_dense_z_i8_destroy(&alloc, diagonal);
        matx_dense_z_i8_destroy(&alloc, A);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(condition, 10.0, 1e-8);
    matx_dense_z_i8_destroy(&alloc, diagonal);
    matx_dense_z_i8_destroy(&alloc, A);
}

TEST(compute_dense_extended, complex_inverse)
{
    matx_alloc_t alloc = matx_alloc_default();
    matx_dense_z_i8_t A = NULL, inverse = NULL;
    ASSERT_EQ(matx_dense_z_i8_create(&alloc, &A, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);
    ASSERT_EQ(matx_dense_z_i8_create(&alloc, &inverse, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);
    A->data[0] = {1.0, 1.0};
    A->data[1] = {3.0, 0.0};
    A->data[2] = {2.0, 0.0};
    A->data[3] = {4.0, -1.0};
    matx_dense_backend_t blas = matx_blas_default();
    matx_status_t st = matx_inv_dense_z_i8(&blas, A, inverse);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_dense_z_i8_destroy(&alloc, inverse);
        matx_dense_z_i8_destroy(&alloc, A);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    for (matx_int64_t row = 0; row < 2; ++row) {
        for (matx_int64_t col = 0; col < 2; ++col) {
            double real = 0.0, imag = 0.0;
            for (matx_int64_t k = 0; k < 2; ++k) {
                const auto lhs = A->data[dense_index(A->layout, A->stride, row, k)];
                const auto rhs = inverse->data[dense_index(inverse->layout, inverse->stride, k, col)];
                real += lhs.real * rhs.real - lhs.imag * rhs.imag;
                imag += lhs.real * rhs.imag + lhs.imag * rhs.real;
            }
            EXPECT_NEAR(real, row == col ? 1.0 : 0.0, 1e-10);
            EXPECT_NEAR(imag, 0.0, 1e-10);
        }
    }
    matx_dense_z_i8_destroy(&alloc, inverse);
    matx_dense_z_i8_destroy(&alloc, A);
}

TEST(compute_dense_extended, exponential_diagonal)
{
    matx_alloc_t alloc = matx_alloc_default();
    matx_dense_d_i8_t A = NULL, expA = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&alloc, &A, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);
    ASSERT_EQ(matx_dense_d_i8_create(&alloc, &expA, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);
    A->data[0] = 1.0;
    A->data[1] = 0.0;
    A->data[2] = 0.0;
    A->data[3] = 2.0;
    matx_dense_backend_t blas = matx_blas_default();
    matx_status_t st = matx_expm_dense_d_i8(&blas, A, expA);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_dense_d_i8_destroy(&alloc, expA);
        matx_dense_d_i8_destroy(&alloc, A);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(expA->data[0], std::exp(1.0), 1e-9);
    EXPECT_NEAR(expA->data[1], 0.0, 1e-10);
    EXPECT_NEAR(expA->data[2], 0.0, 1e-10);
    EXPECT_NEAR(expA->data[3], std::exp(2.0), 2e-8);
    matx_dense_d_i8_destroy(&alloc, expA);
    matx_dense_d_i8_destroy(&alloc, A);
}
