#include "matx_test_harness.h"
#include <math.h>

extern "C" {
#include "matx/matx_dense_compute.h"
#include "matx/matx_func.h"
#include "matx/matx_types.h"
#include "matx/matx_types_internal.h"
#include "matx/matx_vec_compute.h"
}
static void fill_dense_d_i8_4x4(matx_dense_d_i8_t M, double base)
{
    for (size_t j = 0; j < 4; ++j)
        for (size_t i = 0; i < 4; ++i)
            M->data[i + j * M->stride] = base + (double) (i + 4 * j);
}

static void fill_dense_d_i8_mxn(matx_int64_t rows,
                                matx_int64_t columns,
                                matx_dense_d_i8_t M,
                                double base)
{
    for (matx_int64_t r = 0; r < rows; ++r) {
        for (matx_int64_t c = 0; c < columns; ++c) {
            size_t idx;

            if (M->layout == MATX_COL_MAJOR)
                idx = r + c * M->stride;
            else
                idx = c + r * M->stride;

            M->data[idx] = base + r + rows * c;
        }
    }
}

TEST(compute_dense, geadd_d_i8_4x4)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t A = NULL, B = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_COL_MAJOR, 4, 4, NULL), MATX_OK);
    ASSERT_EQ(matx_dense_d_i8_create(&a, &B, MATX_COL_MAJOR, 4, 4, NULL), MATX_OK);
    fill_dense_d_i8_4x4(A, 1.0);
    fill_dense_d_i8_4x4(B, 2.0);
    matx_dense_backend_t blas = matx_blas_default();
    matx_status_t st = matx_geadd_d_i8(&blas, 3.0, A, 0.0, B);
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(B->data[0], 3.0 * 1.0, 1e-12);
    EXPECT_NEAR(B->data[5], 3.0 * 6.0, 1e-12);

    matx_dense_d_i8_destroy(&a, A);
    matx_dense_d_i8_destroy(&a, B);
}

TEST(compute_dense, geadd_z_i8_4x4)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_z_i8_t A = NULL, B = NULL;
    ASSERT_EQ(matx_dense_z_i8_create(&a, &A, MATX_COL_MAJOR, 4, 4, NULL), MATX_OK);
    ASSERT_EQ(matx_dense_z_i8_create(&a, &B, MATX_COL_MAJOR, 4, 4, NULL), MATX_OK);
    for (size_t i = 0; i < 16; ++i) {
        A->data[i].real = (double) i;
        A->data[i].imag = 0.0;
        B->data[i].real = 1.0;
        B->data[i].imag = 0.0;
    }

    matx_dense_backend_t blas = matx_blas_default();
    matx_complex_d_t alpha = {2.0, 0.0};
    matx_complex_d_t beta = {0.0, 0.0};
    matx_status_t st = matx_geadd_z_i8(&blas, alpha, A, beta, B);
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(B->data[0].real, 0.0, 1e-12);
    EXPECT_NEAR(B->data[1].real, 2.0, 1e-12);

    matx_dense_z_i8_destroy(&a, A);
    matx_dense_z_i8_destroy(&a, B);
}

TEST(compute_dense, gemv_d_i8_4x4)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t A = NULL;
    matx_vec_d_i8_t x = NULL, y = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_COL_MAJOR, 4, 4, NULL), MATX_OK);
    ASSERT_EQ(matx_vec_d_i8_create(&a, &x, NULL, 4), MATX_OK);
    ASSERT_EQ(matx_vec_d_i8_create(&a, &y, NULL, 4), MATX_OK);
    fill_dense_d_i8_4x4(A, 1.0);
    x->data[0] = 1.0;
    x->data[1] = 0.0;
    x->data[2] = 0.0;
    x->data[3] = 0.0;
    y->data[0] = y->data[1] = y->data[2] = y->data[3] = 0.0;

    matx_dense_backend_t blas = matx_blas_default();
    matx_status_t st = matx_gemv_d_i8(&blas, MATX_NO_TRANS, 1.0, A, x, 0.0, y);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_dense_d_i8_destroy(&a, A);
        matx_vec_d_i8_destroy(&a, x);
        matx_vec_d_i8_destroy(&a, y);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(y->data[0], A->data[0], 1e-12);
    EXPECT_NEAR(y->data[1], A->data[1], 1e-12);

    matx_dense_d_i8_destroy(&a, A);
    matx_vec_d_i8_destroy(&a, x);
    matx_vec_d_i8_destroy(&a, y);
}

TEST(compute_dense, gemm_d_i8_4x4)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t A = NULL, B = NULL, C = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_COL_MAJOR, 4, 4, NULL), MATX_OK);
    ASSERT_EQ(matx_dense_d_i8_create(&a, &B, MATX_COL_MAJOR, 4, 4, NULL), MATX_OK);
    ASSERT_EQ(matx_dense_d_i8_create(&a, &C, MATX_COL_MAJOR, 4, 4, NULL), MATX_OK);
    fill_dense_d_i8_4x4(A, 1.0);
    fill_dense_d_i8_4x4(B, 0.5);
    for (size_t i = 0; i < 16; ++i)
        C->data[i] = 0.0;

    matx_dense_backend_t blas = matx_blas_default();
    matx_status_t st = matx_gemm_d_i8(&blas, MATX_NO_TRANS, MATX_NO_TRANS, 1.0, A, B, 0.0, C);
    ASSERT_EQ(st, MATX_OK);
    double c00 = 0.0;
    for (size_t k = 0; k < 4; ++k)
        c00 += A->data[0 + k * 4] * B->data[k + 0 * 4];
    EXPECT_NEAR(C->data[0], c00, 1e-10);

    matx_dense_d_i8_destroy(&a, A);
    matx_dense_d_i8_destroy(&a, B);
    matx_dense_d_i8_destroy(&a, C);
}

TEST(compute_dense, gemm_z_i8_4x4)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_z_i8_t A = NULL, B = NULL, C = NULL;
    ASSERT_EQ(matx_dense_z_i8_create(&a, &A, MATX_COL_MAJOR, 4, 4, NULL), MATX_OK);
    ASSERT_EQ(matx_dense_z_i8_create(&a, &B, MATX_COL_MAJOR, 4, 4, NULL), MATX_OK);
    ASSERT_EQ(matx_dense_z_i8_create(&a, &C, MATX_COL_MAJOR, 4, 4, NULL), MATX_OK);
    for (size_t i = 0; i < 16; ++i) {
        A->data[i].real = (double) i;
        A->data[i].imag = 0.0;
        B->data[i].real = (i == 0 ? 1.0 : 0.0);
        B->data[i].imag = 0.0;
        C->data[i].real = C->data[i].imag = 0.0;
    }

    matx_dense_backend_t blas = matx_blas_default();
    matx_complex_d_t alpha = {1.0, 0.0};
    matx_complex_d_t beta = {0.0, 0.0};
    matx_status_t st = matx_gemm_z_i8(&blas, MATX_NO_TRANS, MATX_NO_TRANS, alpha, A, B, beta, C);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_dense_z_i8_destroy(&a, A);
        matx_dense_z_i8_destroy(&a, B);
        matx_dense_z_i8_destroy(&a, C);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(C->data[0].real, A->data[0].real, 1e-10);

    matx_dense_z_i8_destroy(&a, A);
    matx_dense_z_i8_destroy(&a, B);
    matx_dense_z_i8_destroy(&a, C);
}

// ---- Level 1 tests ----

TEST(compute_dense, scal_d_i8)
{
    matx_alloc_t a = matx_alloc_default();
    matx_vec_d_i8_t x = NULL;
    double data[4] = {1.0, 2.0, 3.0, 4.0};
    ASSERT_EQ(matx_vec_d_i8_create(&a, &x, data, 4), MATX_OK);
    matx_vec_backend_t vblas = matx_vec_default();
    ASSERT_EQ(matx_vec_scal_d_i8(&vblas, 2.0, x), MATX_OK);
    EXPECT_NEAR(x->data[0], 2.0, 1e-12);
    EXPECT_NEAR(x->data[3], 8.0, 1e-12);
    matx_vec_d_i8_destroy(&a, x);
}

TEST(compute_dense, copy_d_i8)
{
    matx_alloc_t a = matx_alloc_default();
    matx_vec_d_i8_t x = NULL, y = NULL;
    double data[3] = {1.0, 2.0, 3.0};
    ASSERT_EQ(matx_vec_d_i8_create(&a, &x, data, 3), MATX_OK);
    ASSERT_EQ(matx_vec_d_i8_create(&a, &y, NULL, 3), MATX_OK);
    matx_vec_backend_t vblas = matx_vec_default();
    ASSERT_EQ(matx_vec_copy_d_i8(&vblas, x, y), MATX_OK);
    EXPECT_NEAR(y->data[0], 1.0, 1e-12);
    EXPECT_NEAR(y->data[2], 3.0, 1e-12);
    matx_vec_d_i8_destroy(&a, x);
    matx_vec_d_i8_destroy(&a, y);
}

TEST(compute_dense, dot_d_i8)
{
    matx_alloc_t a = matx_alloc_default();
    matx_vec_d_i8_t x = NULL, y = NULL;
    double xd[3] = {1.0, 2.0, 3.0}, yd[3] = {4.0, 5.0, 6.0};
    ASSERT_EQ(matx_vec_d_i8_create(&a, &x, xd, 3), MATX_OK);
    ASSERT_EQ(matx_vec_d_i8_create(&a, &y, yd, 3), MATX_OK);
    matx_vec_backend_t vblas = matx_vec_default();
    double result = 0.0;
    ASSERT_EQ(matx_vec_dot_d_i8(&vblas, x, y, &result), MATX_OK);
    EXPECT_NEAR(result, 32.0, 1e-12); // 1*4+2*5+3*6=32
    matx_vec_d_i8_destroy(&a, x);
    matx_vec_d_i8_destroy(&a, y);
}

TEST(compute_dense, nrm2_d_i8)
{
    matx_alloc_t a = matx_alloc_default();
    matx_vec_d_i8_t x = NULL;
    double data[3] = {3.0, 4.0, 0.0};
    ASSERT_EQ(matx_vec_d_i8_create(&a, &x, data, 3), MATX_OK);
    matx_vec_backend_t vblas = matx_vec_default();
    double result = 0.0;
    ASSERT_EQ(matx_vec_nrm2_d_i8(&vblas, x, &result), MATX_OK);
    EXPECT_NEAR(result, 5.0, 1e-12);
    matx_vec_d_i8_destroy(&a, x);
}

TEST(compute_dense, iamax_d_i8)
{
    matx_alloc_t a = matx_alloc_default();
    matx_vec_d_i8_t x = NULL;
    double data[4] = {1.0, -9.0, 3.0, 2.0};
    ASSERT_EQ(matx_vec_d_i8_create(&a, &x, data, 4), MATX_OK);
    matx_vec_backend_t vblas = matx_vec_default();
    matx_int64_t idx = -1;
    ASSERT_EQ(matx_vec_iamax_d_i8(&vblas, x, &idx), MATX_OK);
    EXPECT_EQ(idx, 1); // index of max abs value (-9)
    matx_vec_d_i8_destroy(&a, x);
}

TEST(compute_dense, ger_d_i8)
{
    matx_alloc_t a = matx_alloc_default();
    matx_vec_d_i8_t x = NULL, y = NULL;
    matx_dense_d_i8_t A = NULL;
    double xd[2] = {1.0, 2.0}, yd[3] = {1.0, 2.0, 3.0};
    ASSERT_EQ(matx_vec_d_i8_create(&a, &x, xd, 2), MATX_OK);
    ASSERT_EQ(matx_vec_d_i8_create(&a, &y, yd, 3), MATX_OK);
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_COL_MAJOR, 2, 3, NULL), MATX_OK);
    matx_dense_d_i8_zeros(A);
    matx_dense_backend_t blas = matx_blas_default();
    ASSERT_EQ(matx_ger_d_i8(&blas, 1.0, x, y, A), MATX_OK);
    EXPECT_NEAR(A->data[0], 1.0, 1e-12);
    EXPECT_NEAR(A->data[1], 2.0, 1e-12);
    matx_vec_d_i8_destroy(&a, x);
    matx_vec_d_i8_destroy(&a, y);
    matx_dense_d_i8_destroy(&a, A);
}

// ---- Transpose tests ----

TEST(compute_dense, transpose_d_i8)
{
    matx_alloc_t a = matx_alloc_default();
    double data[6] = {1, 2, 3, 4, 5, 6}; // 2x3 col-major
    matx_dense_d_i8_t A = NULL, T = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_COL_MAJOR, 2, 3, data), MATX_OK);
    ASSERT_EQ(matx_dense_d_i8_create(&a, &T, MATX_COL_MAJOR, 3, 2, NULL), MATX_OK);
    matx_dense_backend_t blas = matx_blas_default();
    matx_status_t st = matx_transpose_d_i8(&blas, A, T);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_dense_d_i8_destroy(&a, A);
        matx_dense_d_i8_destroy(&a, T);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    EXPECT_EQ(T->nrows, 3u);
    EXPECT_EQ(T->ncols, 2u);
    // A(0,0)=1 => T(0,0)=1; A(1,0)=2 => T(0,1)=2
    EXPECT_NEAR(T->data[0], 1.0, 1e-12);
    EXPECT_NEAR(T->data[1], 3.0, 1e-12);
    matx_dense_d_i8_destroy(&a, A);
    matx_dense_d_i8_destroy(&a, T);
}

TEST(compute_dense, transpose_z_i8)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_z_i8_t A = NULL, T = NULL;
    ASSERT_EQ(matx_dense_z_i8_create(&a, &A, MATX_COL_MAJOR, 2, 3, NULL), MATX_OK);
    ASSERT_EQ(matx_dense_z_i8_create(&a, &T, MATX_COL_MAJOR, 3, 2, NULL), MATX_OK);
    for (matx_int64_t i = 0; i < 6; ++i) {
        A->data[i].real = (double) (i + 1);
        A->data[i].imag = (double) (i + 1) * 2.0;
    }
    matx_dense_backend_t blas = matx_blas_default();
    matx_status_t st = matx_transpose_z_i8(&blas, A, T);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_dense_z_i8_destroy(&a, A);
        matx_dense_z_i8_destroy(&a, T);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    EXPECT_EQ(T->nrows, 3u);
    EXPECT_EQ(T->ncols, 2u);
    matx_dense_z_i8_destroy(&a, A);
    matx_dense_z_i8_destroy(&a, T);
}

TEST(compute_dense, conj_transpose_z_i8)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_z_i8_t A = NULL, T = NULL;
    ASSERT_EQ(matx_dense_z_i8_create(&a, &A, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);
    ASSERT_EQ(matx_dense_z_i8_create(&a, &T, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);
    A->data[0].real = 1.0;
    A->data[0].imag = 2.0;
    A->data[1].real = 3.0;
    A->data[1].imag = 4.0;
    A->data[2].real = 5.0;
    A->data[2].imag = 6.0;
    A->data[3].real = 7.0;
    A->data[3].imag = 8.0;
    matx_dense_backend_t blas = matx_blas_default();
    matx_status_t st = matx_conj_transpose_z_i8(&blas, A, T);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_dense_z_i8_destroy(&a, A);
        matx_dense_z_i8_destroy(&a, T);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    // T(0,0) = conj(A(0,0)) = (1, -2)
    EXPECT_NEAR(T->data[0].real, 1.0, 1e-12);
    EXPECT_NEAR(T->data[0].imag, -2.0, 1e-12);
    matx_dense_z_i8_destroy(&a, A);
    matx_dense_z_i8_destroy(&a, T);
}

// ---- Norm tests ----

TEST(compute_dense, norm1_norminf_normfro_d_i8)
{
    matx_alloc_t a = matx_alloc_default();
    double data[4] = {1, 2, 3, 4}; // 2x2 col-major: col0={1,2}, col1={3,4}
    matx_dense_d_i8_t A = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_COL_MAJOR, 2, 2, data), MATX_OK);
    matx_dense_backend_t blas = matx_blas_default();
    double n1 = 0, ni = 0, nf = 0;
    ASSERT_EQ(matx_mat_norm1_d_i8(&blas, A, &n1), MATX_OK);
    EXPECT_NEAR(n1, 7.0, 1e-12); // max col sum: col0=3, col1=7
    ASSERT_EQ(matx_mat_norminf_d_i8(&blas, A, &ni), MATX_OK);
    EXPECT_NEAR(ni, 6.0, 1e-12); // max row sum: row0=4, row1=6
    ASSERT_EQ(matx_mat_normfro_d_i8(&blas, A, &nf), MATX_OK);
    EXPECT_NEAR(nf, sqrt(1 + 4 + 9 + 16), 1e-10);
    matx_dense_d_i8_destroy(&a, A);
}

TEST(compute_dense, norm1_norminf_normfro_z_i8)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_z_i8_t A = NULL;
    ASSERT_EQ(matx_dense_z_i8_create(&a, &A, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);
    // col0: (1+1j, 2), col1: (3+4j, 4); complex magnitudes use hypot.
    A->data[0].real = 1.0;
    A->data[0].imag = 1.0;
    A->data[1].real = 2.0;
    A->data[1].imag = 0.0;
    A->data[2].real = 3.0;
    A->data[2].imag = 4.0;
    A->data[3].real = 4.0;
    A->data[3].imag = 0.0;
    matx_dense_backend_t blas = matx_blas_default();
    double n1 = 0, ni = 0, nf = 0;
    ASSERT_EQ(matx_mat_norm1_z_i8(&blas, A, &n1), MATX_OK);
    EXPECT_NEAR(n1, 9.0, 1e-12);
    ASSERT_EQ(matx_mat_norminf_z_i8(&blas, A, &ni), MATX_OK);
    EXPECT_NEAR(ni, std::sqrt(2.0) + 5.0, 1e-12);
    ASSERT_EQ(matx_mat_normfro_z_i8(&blas, A, &nf), MATX_OK);
    EXPECT_NEAR(nf, std::sqrt(47.0), 1e-10);
    matx_dense_z_i8_destroy(&a, A);
}

// ---- gerc (conjugate rank-1) tests ----

TEST(compute_dense, gerc_z_i8_3x2)
{
    // A := alpha * x * y^H + A
    // x = [(1,0),(2,0),(3,0)], y = [(4,1),(5,-1)]
    // alpha = (1,0)
    // A(0,0) += alpha*x0*conj(y0) = 1*(4,1) = (4,1)
    // A(1,0) += alpha*x1*conj(y0) = 2*(4,1) = (8,2)
    // A(2,0) += alpha*x2*conj(y0) = 3*(4,1) = (12,3)
    // A(0,1) += alpha*x0*conj(y1) = 1*(5,1) = (5,1)
    // A(1,1) += alpha*x1*conj(y1) = 2*(5,1) = (10,2)
    // A(2,1) += alpha*x2*conj(y1) = 3*(5,1) = (15,3)
    matx_alloc_t a = matx_alloc_default();
    matx_vec_z_i8_t x = NULL, y = NULL;
    matx_dense_z_i8_t A = NULL;
    ASSERT_EQ(matx_vec_z_i8_create(&a, &x, NULL, 3), MATX_OK);
    ASSERT_EQ(matx_vec_z_i8_create(&a, &y, NULL, 2), MATX_OK);
    ASSERT_EQ(matx_dense_z_i8_create(&a, &A, MATX_COL_MAJOR, 3, 2, NULL), MATX_OK);
    for (int i = 0; i < 6; ++i) {
        A->data[i].real = 0.0;
        A->data[i].imag = 0.0;
    }
    x->data[0].real = 1.0;
    x->data[0].imag = 0.0;
    x->data[1].real = 2.0;
    x->data[1].imag = 0.0;
    x->data[2].real = 3.0;
    x->data[2].imag = 0.0;
    y->data[0].real = 4.0;
    y->data[0].imag = 1.0;
    y->data[1].real = 5.0;
    y->data[1].imag = -1.0;

    matx_complex_d_t alpha = {1.0, 0.0};
    matx_dense_backend_t blas = matx_blas_default();
    matx_status_t st = matx_gerc_z_i8(&blas, alpha, x, y, A);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_vec_z_i8_destroy(&a, x);
        matx_vec_z_i8_destroy(&a, y);
        matx_dense_z_i8_destroy(&a, A);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(A->data[0].real, 4.0, 1e-12);
    EXPECT_NEAR(A->data[0].imag, -1.0, 1e-12);

    EXPECT_NEAR(A->data[1].real, 8.0, 1e-12);
    EXPECT_NEAR(A->data[1].imag, -2.0, 1e-12);

    EXPECT_NEAR(A->data[2].real, 12.0, 1e-12);
    EXPECT_NEAR(A->data[2].imag, -3.0, 1e-12);

    EXPECT_NEAR(A->data[3].real, 5.0, 1e-12);
    EXPECT_NEAR(A->data[3].imag, 1.0, 1e-12);

    EXPECT_NEAR(A->data[4].real, 10.0, 1e-12);
    EXPECT_NEAR(A->data[4].imag, 2.0, 1e-12);

    EXPECT_NEAR(A->data[5].real, 15.0, 1e-12);
    EXPECT_NEAR(A->data[5].imag, 3.0, 1e-12);

    matx_vec_z_i8_destroy(&a, x);
    matx_vec_z_i8_destroy(&a, y);
    matx_dense_z_i8_destroy(&a, A);
}

// ---- syr2k / her2k tests ----

TEST(compute_dense, syr2k_d_i8_3x2)
{
    // C = A*B^T + B*A^T, C symmetric 3x3
    // Simple case with small known matrices
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t A = NULL, B = NULL, C = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_COL_MAJOR, 3, 2, NULL), MATX_OK);
    ASSERT_EQ(matx_dense_d_i8_create(&a, &B, MATX_COL_MAJOR, 3, 2, NULL), MATX_OK);
    ASSERT_EQ(matx_dense_d_i8_create(&a, &C, MATX_COL_MAJOR, 3, 3, NULL), MATX_OK);
    fill_dense_d_i8_mxn(3, 2, A, 1.0); // use first 6 values of 4x4 pattern
    fill_dense_d_i8_mxn(3, 2, B, 0.5);
    for (int i = 0; i < 9; ++i)
        C->data[i] = 0.0;

    matx_dense_backend_t blas = matx_blas_default();
    matx_status_t st = matx_syr2k_d_i8(&blas, MATX_LOWER, MATX_NO_TRANS, 1.0, A, B, 0.0, C);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_dense_d_i8_destroy(&a, A);
        matx_dense_d_i8_destroy(&a, B);
        matx_dense_d_i8_destroy(&a, C);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    // C should be symmetric and non-zero
    EXPECT_NEAR(C->data[0], 29.0, 1e-12);
    matx_dense_d_i8_destroy(&a, A);
    matx_dense_d_i8_destroy(&a, B);
    matx_dense_d_i8_destroy(&a, C);
}

TEST(compute_dense, her2k_z_i8_3x2)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_z_i8_t A = NULL, B = NULL, C = NULL;
    ASSERT_EQ(matx_dense_z_i8_create(&a, &A, MATX_COL_MAJOR, 3, 2, NULL), MATX_OK);
    ASSERT_EQ(matx_dense_z_i8_create(&a, &B, MATX_COL_MAJOR, 3, 2, NULL), MATX_OK);
    ASSERT_EQ(matx_dense_z_i8_create(&a, &C, MATX_COL_MAJOR, 3, 3, NULL), MATX_OK);
    for (int i = 0; i < 6; ++i) {
        A->data[i].real = (double) i;
        A->data[i].imag = 0.0;
        B->data[i].real = (double) (i + 1);
        B->data[i].imag = 0.0;
    }
    for (int i = 0; i < 9; ++i) {
        C->data[i].real = 0.0;
        C->data[i].imag = 0.0;
    }

    matx_complex_d_t alpha = {1.0, 0.0};
    matx_dense_backend_t blas = matx_blas_default();
    matx_status_t st = matx_her2k_z_i8(&blas, MATX_LOWER, MATX_NO_TRANS, alpha, A, B, 0.0, C);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_dense_z_i8_destroy(&a, A);
        matx_dense_z_i8_destroy(&a, B);
        matx_dense_z_i8_destroy(&a, C);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    // Hermitian result: diagonal should be real
    EXPECT_NEAR(C->data[0].imag, 0.0, 1e-12);
    EXPECT_NEAR(C->data[4].imag, 0.0, 1e-12);
    EXPECT_NEAR(C->data[8].imag, 0.0, 1e-12);
    matx_dense_z_i8_destroy(&a, A);
    matx_dense_z_i8_destroy(&a, B);
    matx_dense_z_i8_destroy(&a, C);
}

// ---- Hadamard (element-wise) tests ----

TEST(compute_dense, hadamard_d_i8_2x2)
{
    matx_alloc_t a = matx_alloc_default();
    double data_a[4] = {1, 2, 3, 4}; // col-major 2x2
    double data_b[4] = {10, 20, 30, 40};
    matx_dense_d_i8_t A = NULL, B = NULL, C = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_COL_MAJOR, 2, 2, data_a), MATX_OK);
    ASSERT_EQ(matx_dense_d_i8_create(&a, &B, MATX_COL_MAJOR, 2, 2, data_b), MATX_OK);
    ASSERT_EQ(matx_dense_d_i8_create(&a, &C, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);
    matx_dense_backend_t blas = matx_blas_default();
    matx_status_t st = matx_hadamard_d_i8(&blas, A, B, C);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_dense_d_i8_destroy(&a, A);
        matx_dense_d_i8_destroy(&a, B);
        matx_dense_d_i8_destroy(&a, C);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(C->data[0], 10.0, 1e-12);
    EXPECT_NEAR(C->data[1], 40.0, 1e-12);
    EXPECT_NEAR(C->data[2], 90.0, 1e-12);
    EXPECT_NEAR(C->data[3], 160.0, 1e-12);
    matx_dense_d_i8_destroy(&a, A);
    matx_dense_d_i8_destroy(&a, B);
    matx_dense_d_i8_destroy(&a, C);
}

TEST(compute_dense, hadamard_z_i8_2x2)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_z_i8_t A = NULL, B = NULL, C = NULL;
    ASSERT_EQ(matx_dense_z_i8_create(&a, &A, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);
    ASSERT_EQ(matx_dense_z_i8_create(&a, &B, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);
    ASSERT_EQ(matx_dense_z_i8_create(&a, &C, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);
    A->data[0] = {1, 0};
    A->data[1] = {2, 1};
    A->data[2] = {3, -1};
    A->data[3] = {4, 0};
    B->data[0] = {2, 0};
    B->data[1] = {1, 0};
    B->data[2] = {1, 0};
    B->data[3] = {2, 0};

    matx_dense_backend_t blas = matx_blas_default();
    matx_status_t st = matx_hadamard_z_i8(&blas, A, B, C);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_dense_z_i8_destroy(&a, A);
        matx_dense_z_i8_destroy(&a, B);
        matx_dense_z_i8_destroy(&a, C);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    // (0,0)
    EXPECT_NEAR(C->data[0].real, 2.0, 1e-12);
    EXPECT_NEAR(C->data[0].imag, 0.0, 1e-12);

    // (1,0)
    EXPECT_NEAR(C->data[1].real, 2.0, 1e-12);
    EXPECT_NEAR(C->data[1].imag, 1.0, 1e-12);

    // (0,1)
    EXPECT_NEAR(C->data[2].real, 3.0, 1e-12);
    EXPECT_NEAR(C->data[2].imag, -1.0, 1e-12);

    // (1,1)
    EXPECT_NEAR(C->data[3].real, 8.0, 1e-12);
    EXPECT_NEAR(C->data[3].imag, 0.0, 1e-12);

    matx_dense_z_i8_destroy(&a, A);
    matx_dense_z_i8_destroy(&a, B);
    matx_dense_z_i8_destroy(&a, C);
}

TEST(compute_dense, gemv_z_i8_4x4)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_z_i8_t A = NULL;
    matx_vec_z_i8_t x = NULL, y = NULL;
    ASSERT_EQ(matx_dense_z_i8_create(&a, &A, MATX_COL_MAJOR, 4, 4, NULL), MATX_OK);
    ASSERT_EQ(matx_vec_z_i8_create(&a, &x, NULL, 4), MATX_OK);
    ASSERT_EQ(matx_vec_z_i8_create(&a, &y, NULL, 4), MATX_OK);
    matx_complex_d_t alpha = {1.0, 0.0};
    matx_complex_d_t beta = {0.0, 0.0};
    // A = diag(1,2,3,4)
    A->data[0] = {1, 0};
    A->data[5] = {2, 0};
    A->data[10] = {3, 0};
    A->data[15] = {4, 0};
    x->data[0] = {1, 0};
    x->data[1] = {0, 0};
    x->data[2] = {0, 0};
    x->data[3] = {0, 0};

    matx_dense_backend_t blas = matx_blas_default();
    matx_status_t st = matx_gemv_z_i8(&blas, MATX_NO_TRANS, alpha, A, x, beta, y);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_dense_z_i8_destroy(&a, A);
        matx_vec_z_i8_destroy(&a, x);
        matx_vec_z_i8_destroy(&a, y);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(y->data[0].real, 1.0, 1e-12);
    EXPECT_NEAR(y->data[0].imag, 0.0, 1e-12);

    matx_dense_z_i8_destroy(&a, A);
    matx_vec_z_i8_destroy(&a, x);
    matx_vec_z_i8_destroy(&a, y);
}

TEST(compute_dense, geru_z_i8_3x2)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_z_i8_t A = NULL;
    matx_vec_z_i8_t x = NULL, y = NULL;
    ASSERT_EQ(matx_dense_z_i8_create(&a, &A, MATX_COL_MAJOR, 3, 2, NULL), MATX_OK);
    ASSERT_EQ(matx_vec_z_i8_create(&a, &x, NULL, 3), MATX_OK);
    ASSERT_EQ(matx_vec_z_i8_create(&a, &y, NULL, 2), MATX_OK);
    matx_complex_d_t alpha = {1.0, 0.0};
    for (size_t i = 0; i < 6; ++i)
        A->data[i].real = A->data[i].imag = 0.0;
    x->data[0] = {1, 1};
    x->data[1] = {2, 0};
    x->data[2] = {3, 0};
    y->data[0] = {4, 0};
    y->data[1] = {5, 2};

    matx_dense_backend_t blas = matx_blas_default();
    matx_status_t st = matx_geru_z_i8(&blas, alpha, x, y, A);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_dense_z_i8_destroy(&a, A);
        matx_vec_z_i8_destroy(&a, x);
        matx_vec_z_i8_destroy(&a, y);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    // A(0,0) = alpha*x[0]*y[0] = 4+4i
    EXPECT_NEAR(A->data[0].real, 4.0, 1e-10);
    EXPECT_NEAR(A->data[0].imag, 4.0, 1e-10);

    matx_dense_z_i8_destroy(&a, A);
    matx_vec_z_i8_destroy(&a, x);
    matx_vec_z_i8_destroy(&a, y);
}

TEST(compute_dense, trsv_d_i8_4x4)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t A = NULL;
    matx_vec_d_i8_t x = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_COL_MAJOR, 4, 4, NULL), MATX_OK);
    ASSERT_EQ(matx_vec_d_i8_create(&a, &x, NULL, 4), MATX_OK);
    // A upper tri, diag=2, upper=1
    matx_dense_d_i8_fill(A, 0.0);
    A->data[0] = 2;
    A->data[4] = 0;
    A->data[5] = 2;
    A->data[8] = 0;
    A->data[9] = 0;
    A->data[10] = 2;
    A->data[12] = 0;
    A->data[13] = 0;
    A->data[14] = 0;
    A->data[15] = 2;
    // connect upper triangle
    A->data[1] = 1;
    A->data[2] = 1;
    A->data[3] = 1;
    A->data[6] = 1;
    A->data[7] = 1;
    A->data[11] = 1;
    x->data[0] = 0;
    x->data[1] = 1;
    x->data[2] = 0;
    x->data[3] = 3;

    matx_dense_backend_t blas = matx_blas_default();
    matx_status_t st = matx_trsv_d_i8(&blas, MATX_UPPER, MATX_NO_TRANS, MATX_NON_UNIT_DIAG, A, x);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_dense_d_i8_destroy(&a, A);
        matx_vec_d_i8_destroy(&a, x);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(x->data[0], 0.0, 1e-10);
    EXPECT_NEAR(x->data[1], 0.5, 1e-10);
    EXPECT_NEAR(x->data[2], 0.0, 1e-10);
    EXPECT_NEAR(x->data[3], 1.5, 1e-10);

    matx_dense_d_i8_destroy(&a, A);
    matx_vec_d_i8_destroy(&a, x);
}

TEST(compute_dense, trsv_z_i8_4x4)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_z_i8_t A = NULL;
    matx_vec_z_i8_t x = NULL;
    ASSERT_EQ(matx_dense_z_i8_create(&a, &A, MATX_COL_MAJOR, 4, 4, NULL), MATX_OK);
    ASSERT_EQ(matx_vec_z_i8_create(&a, &x, NULL, 4), MATX_OK);
    // A upper tri, diag=(2,0), upper=(1,0)
    matx_dense_z_i8_fill(A, {0, 0});
    A->data[0] = {2, 0};
    A->data[4] = {0, 0};
    A->data[5] = {2, 0};
    A->data[8] = {0, 0};
    A->data[9] = {0, 0};
    A->data[10] = {2, 0};
    A->data[12] = {0, 0};
    A->data[13] = {0, 0};
    A->data[14] = {0, 0};
    A->data[15] = {2, 0};
    A->data[1] = {1, 0};
    A->data[2] = {1, 0};
    A->data[3] = {1, 0};
    A->data[6] = {1, 0};
    A->data[7] = {1, 0};
    A->data[11] = {1, 0};
    x->data[0] = {0, 0};
    x->data[1] = {1, 0};
    x->data[2] = {0, 0};
    x->data[3] = {3, 0};

    matx_dense_backend_t blas = matx_blas_default();
    matx_status_t st = matx_trsv_z_i8(&blas, MATX_UPPER, MATX_NO_TRANS, MATX_NON_UNIT_DIAG, A, x);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_dense_z_i8_destroy(&a, A);
        matx_vec_z_i8_destroy(&a, x);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(x->data[0].real, 0.0, 1e-10);
    EXPECT_NEAR(x->data[1].real, 0.5, 1e-10);
    EXPECT_NEAR(x->data[2].real, 0.0, 1e-10);
    EXPECT_NEAR(x->data[3].real, 1.5, 1e-10);

    matx_dense_z_i8_destroy(&a, A);
    matx_vec_z_i8_destroy(&a, x);
}

TEST(compute_dense, trsm_d_i8_4x4)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t A = NULL, B = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_COL_MAJOR, 4, 4, NULL), MATX_OK);
    ASSERT_EQ(matx_dense_d_i8_create(&a, &B, MATX_COL_MAJOR, 4, 3, NULL), MATX_OK);
    // A lower tri, diag=2
    matx_dense_d_i8_fill(A, 0.0);
    A->data[0] = 2;
    A->data[1] = 1;
    A->data[2] = 1;
    A->data[3] = 1;
    A->data[5] = 2;
    A->data[6] = 1;
    A->data[7] = 1;
    A->data[10] = 2;
    A->data[11] = 1;
    A->data[15] = 2;
    matx_dense_d_i8_fill(B, 4.0);

    matx_dense_backend_t blas = matx_blas_default();
    matx_status_t st
        = matx_trsm_d_i8(&blas, MATX_LEFT, MATX_LOWER, MATX_NO_TRANS, MATX_NON_UNIT_DIAG, 1.0, A, B);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_dense_d_i8_destroy(&a, A);
        matx_dense_d_i8_destroy(&a, B);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    // B_col0 = A \ [4,4,4,4] = [2,1,0.5,0.25]
    EXPECT_NEAR(B->data[0], 2.0, 1e-10);
    EXPECT_NEAR(B->data[1], 1.0, 1e-10);
    EXPECT_NEAR(B->data[2], 0.5, 1e-10);
    EXPECT_NEAR(B->data[3], 0.25, 1e-10);

    matx_dense_d_i8_destroy(&a, A);
    matx_dense_d_i8_destroy(&a, B);
}

TEST(compute_dense, trsm_z_i8_4x4)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_z_i8_t A = NULL, B = NULL;
    ASSERT_EQ(matx_dense_z_i8_create(&a, &A, MATX_COL_MAJOR, 4, 4, NULL), MATX_OK);
    ASSERT_EQ(matx_dense_z_i8_create(&a, &B, MATX_COL_MAJOR, 4, 3, NULL), MATX_OK);
    matx_complex_d_t alpha = {1.0, 0.0};
    // A lower tri, diag=(2,0)
    matx_dense_z_i8_fill(A, {0, 0});
    A->data[0] = {2, 0};
    A->data[1] = {1, 0};
    A->data[2] = {1, 0};
    A->data[3] = {1, 0};
    A->data[5] = {2, 0};
    A->data[6] = {1, 0};
    A->data[7] = {1, 0};
    A->data[10] = {2, 0};
    A->data[11] = {1, 0};
    A->data[15] = {2, 0};
    matx_dense_z_i8_fill(B, {4, 0});

    matx_dense_backend_t blas = matx_blas_default();
    matx_status_t st = matx_trsm_z_i8(&blas,
                                      MATX_LEFT,
                                      MATX_LOWER,
                                      MATX_NO_TRANS,
                                      MATX_NON_UNIT_DIAG,
                                      alpha,
                                      A,
                                      B);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_dense_z_i8_destroy(&a, A);
        matx_dense_z_i8_destroy(&a, B);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    EXPECT_NEAR(B->data[0].real, 2.0, 1e-10);
    EXPECT_NEAR(B->data[1].real, 1.0, 1e-10);
    EXPECT_NEAR(B->data[2].real, 0.5, 1e-10);
    EXPECT_NEAR(B->data[3].real, 0.25, 1e-10);

    matx_dense_z_i8_destroy(&a, A);
    matx_dense_z_i8_destroy(&a, B);
}

TEST(compute_dense, syrk_d_i8_3x2)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t A = NULL, C = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_COL_MAJOR, 3, 2, NULL), MATX_OK);
    ASSERT_EQ(matx_dense_d_i8_create(&a, &C, MATX_COL_MAJOR, 3, 3, NULL), MATX_OK);
    // A = [1 4; 2 5; 3 6] (col-major)
    A->data[0] = 1;
    A->data[1] = 2;
    A->data[2] = 3;
    A->data[3] = 4;
    A->data[4] = 5;
    A->data[5] = 6;
    matx_dense_d_i8_zeros(C);

    matx_dense_backend_t blas = matx_blas_default();
    matx_status_t st = matx_syrk_d_i8(&blas, MATX_LOWER, MATX_NO_TRANS, 1.0, A, 0.0, C);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_dense_d_i8_destroy(&a, A);
        matx_dense_d_i8_destroy(&a, C);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    // C = A*A^T, C(0,0) = 1+16=17
    EXPECT_NEAR(C->data[0], 17.0, 1e-10);

    matx_dense_d_i8_destroy(&a, A);
    matx_dense_d_i8_destroy(&a, C);
}

TEST(compute_dense, herk_z_i8_3x2)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_z_i8_t A = NULL, C = NULL;
    ASSERT_EQ(matx_dense_z_i8_create(&a, &A, MATX_COL_MAJOR, 3, 2, NULL), MATX_OK);
    ASSERT_EQ(matx_dense_z_i8_create(&a, &C, MATX_COL_MAJOR, 3, 3, NULL), MATX_OK);
    // A col 0: (1,0),(2,0),(3,0); col 1: (4,0),(5,0),(6,0)
    A->data[0] = {1, 0};
    A->data[1] = {2, 0};
    A->data[2] = {3, 0};
    A->data[3] = {4, 0};
    A->data[4] = {5, 0};
    A->data[5] = {6, 0};
    matx_dense_z_i8_zeros(C);

    matx_dense_backend_t blas = matx_blas_default();
    matx_status_t st = matx_herk_z_i8(&blas, MATX_LOWER, MATX_NO_TRANS, 1.0, A, 0.0, C);
    if (st == MATX_ERR_NOT_SUPPORTED) {
        matx_dense_z_i8_destroy(&a, A);
        matx_dense_z_i8_destroy(&a, C);
        return;
    }
    ASSERT_EQ(st, MATX_OK);
    // C = A*A^H, C(0,0) = 1+16=17, diag imaginary = 0
    EXPECT_NEAR(C->data[0].real, 17.0, 1e-10);
    EXPECT_NEAR(C->data[0].imag, 0.0, 1e-10);

    matx_dense_z_i8_destroy(&a, A);
    matx_dense_z_i8_destroy(&a, C);
}

// ---- Phase 1: New BLAS operator tests ----

TEST(compute_dense, trmv_d_i8_4x4)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t A = NULL;
    matx_vec_d_i8_t x = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_COL_MAJOR, 4, 4, NULL), MATX_OK);
    ASSERT_EQ(matx_vec_d_i8_create(&a, &x, NULL, 4), MATX_OK);
    // Upper triangular: fill upper triangle only
    for (size_t i = 0; i < 16; ++i) A->data[i] = 0.0;
    A->data[0 + 0 * A->stride] = 2.0;  // (0,0)
    A->data[0 + 1 * A->stride] = 1.0;  // (0,1) upper
    A->data[1 + 1 * A->stride] = 3.0;  // (1,1)
    A->data[0 + 2 * A->stride] = 2.0;  // (0,2) upper
    A->data[1 + 2 * A->stride] = 4.0;  // (1,2) upper
    A->data[2 + 2 * A->stride] = 5.0;  // (2,2)
    A->data[0 + 3 * A->stride] = 1.0;  // (0,3) upper
    A->data[1 + 3 * A->stride] = 2.0;  // (1,3) upper
    A->data[2 + 3 * A->stride] = 3.0;  // (2,3) upper
    A->data[3 + 3 * A->stride] = 4.0;  // (3,3)
    x->data[0] = 1.0; x->data[1] = 1.0; x->data[2] = 1.0; x->data[3] = 1.0;

    matx_dense_backend_t blas = matx_blas_default();
    matx_status_t st = matx_trmv_d_i8(&blas, MATX_UPPER, MATX_NO_TRANS, MATX_NON_UNIT_DIAG, A, x);
    if (st == MATX_ERR_NOT_SUPPORTED) { matx_dense_d_i8_destroy(&a, A); matx_vec_d_i8_destroy(&a, x); return; }
    ASSERT_EQ(st, MATX_OK);
    // x = [2*1+1*1+2*1+1*1, 3*1+4*1+2*1, 5*1+3*1, 4*1] = [6, 9, 8, 4]
    EXPECT_NEAR(x->data[0], 6.0, 1e-12);
    EXPECT_NEAR(x->data[1], 9.0, 1e-12);
    EXPECT_NEAR(x->data[2], 8.0, 1e-12);
    EXPECT_NEAR(x->data[3], 4.0, 1e-12);

    matx_dense_d_i8_destroy(&a, A);
    matx_vec_d_i8_destroy(&a, x);
}

TEST(compute_dense, trmv_z_i8_4x4)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_z_i8_t A = NULL;
    matx_vec_z_i8_t x = NULL;
    ASSERT_EQ(matx_dense_z_i8_create(&a, &A, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);
    ASSERT_EQ(matx_vec_z_i8_create(&a, &x, NULL, 2), MATX_OK);
    for (size_t i = 0; i < 4; ++i) { A->data[i].real = 0.0; A->data[i].imag = 0.0; }
    // Upper triangular complex 2x2: [[1+i, 2+0i], [0, 3-i]]
    A->data[0 + 0 * A->stride].real = 1.0; A->data[0 + 0 * A->stride].imag = 1.0;
    A->data[0 + 1 * A->stride].real = 2.0; A->data[0 + 1 * A->stride].imag = 0.0;
    A->data[1 + 1 * A->stride].real = 3.0; A->data[1 + 1 * A->stride].imag = -1.0;
    x->data[0].real = 1.0; x->data[0].imag = 0.0;
    x->data[1].real = 0.0; x->data[1].imag = 1.0;

    matx_dense_backend_t blas = matx_blas_default();
    matx_status_t st = matx_trmv_z_i8(&blas, MATX_UPPER, MATX_NO_TRANS, MATX_NON_UNIT_DIAG, A, x);
    if (st == MATX_ERR_NOT_SUPPORTED) { matx_dense_z_i8_destroy(&a, A); matx_vec_z_i8_destroy(&a, x); return; }
    ASSERT_EQ(st, MATX_OK);
    // x[0] = (1+i)*1 + (2+0i)*(i) = 1+i + 2i = 1+3i
    // x[1] = (3-i)*i = 3i+1 = 1+3i
    EXPECT_NEAR(x->data[0].real, 1.0, 1e-12);
    EXPECT_NEAR(x->data[0].imag, 3.0, 1e-12);
    EXPECT_NEAR(x->data[1].real, 1.0, 1e-12);
    EXPECT_NEAR(x->data[1].imag, 3.0, 1e-12);

    matx_dense_z_i8_destroy(&a, A);
    matx_vec_z_i8_destroy(&a, x);
}

TEST(compute_dense, symv_d_i8_3x3)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t A = NULL;
    matx_vec_d_i8_t x = NULL, y = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_COL_MAJOR, 3, 3, NULL), MATX_OK);
    ASSERT_EQ(matx_vec_d_i8_create(&a, &x, NULL, 3), MATX_OK);
    ASSERT_EQ(matx_vec_d_i8_create(&a, &y, NULL, 3), MATX_OK);
    for (size_t i = 0; i < 9; ++i) A->data[i] = 0.0;
    // Symmetric: fill lower triangle, upper is implied
    A->data[0 + 0 * A->stride] = 2.0;
    A->data[1 + 0 * A->stride] = 1.0; A->data[1 + 1 * A->stride] = 2.0;
    A->data[2 + 0 * A->stride] = 0.0; A->data[2 + 1 * A->stride] = 1.0; A->data[2 + 2 * A->stride] = 2.0;
    x->data[0] = 1.0; x->data[1] = 2.0; x->data[2] = 3.0;
    y->data[0] = y->data[1] = y->data[2] = 1.0;

    matx_dense_backend_t blas = matx_blas_default();
    matx_status_t st = matx_symv_d_i8(&blas, MATX_LOWER, 2.0, A, x, 0.5, y);
    if (st == MATX_ERR_NOT_SUPPORTED) { matx_dense_d_i8_destroy(&a, A); matx_vec_d_i8_destroy(&a, x); matx_vec_d_i8_destroy(&a, y); return; }
    ASSERT_EQ(st, MATX_OK);
    // A*x = [2*1+1*2+0*3, 1*1+2*2+1*3, 0*1+1*2+2*3] = [4, 8, 8]
    // y = 2*[4,8,8] + 0.5*[1,1,1] = [8.5, 16.5, 16.5]
    EXPECT_NEAR(y->data[0], 8.5, 1e-12);
    EXPECT_NEAR(y->data[1], 16.5, 1e-12);
    EXPECT_NEAR(y->data[2], 16.5, 1e-12);

    matx_dense_d_i8_destroy(&a, A);
    matx_vec_d_i8_destroy(&a, x);
    matx_vec_d_i8_destroy(&a, y);
}

TEST(compute_dense, hemv_z_i8_2x2)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_z_i8_t A = NULL;
    matx_vec_z_i8_t x = NULL, y = NULL;
    ASSERT_EQ(matx_dense_z_i8_create(&a, &A, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);
    ASSERT_EQ(matx_vec_z_i8_create(&a, &x, NULL, 2), MATX_OK);
    ASSERT_EQ(matx_vec_z_i8_create(&a, &y, NULL, 2), MATX_OK);
    for (size_t i = 0; i < 4; ++i) { A->data[i].real = 0.0; A->data[i].imag = 0.0; }
    // Hermitian: [[4, 1+i], [1-i, 3]]
    A->data[0 + 0 * A->stride].real = 4.0; A->data[0 + 0 * A->stride].imag = 0.0;
    A->data[1 + 0 * A->stride].real = 1.0; A->data[1 + 0 * A->stride].imag = -1.0;
    A->data[1 + 1 * A->stride].real = 3.0; A->data[1 + 1 * A->stride].imag = 0.0;
    x->data[0].real = 1.0; x->data[0].imag = 0.0;
    x->data[1].real = 0.0; x->data[1].imag = 1.0;
    y->data[0].real = 0.0; y->data[0].imag = 0.0;
    y->data[1].real = 0.0; y->data[1].imag = 0.0;

    matx_dense_backend_t blas = matx_blas_default();
    matx_complex_d_t alpha = {1.0, 0.0};
    matx_complex_d_t beta = {0.0, 0.0};
    matx_status_t st = matx_hemv_z_i8(&blas, MATX_LOWER, alpha, A, x, beta, y);
    if (st == MATX_ERR_NOT_SUPPORTED) { matx_dense_z_i8_destroy(&a, A); matx_vec_z_i8_destroy(&a, x); matx_vec_z_i8_destroy(&a, y); return; }
    ASSERT_EQ(st, MATX_OK);
    // y[0] = A[0,0]*x[0] + A[0,1]*x[1] = 4*1 + (1+i)*i = 4 + i-1 = 3+i
    // y[1] = A[1,0]*x[0] + A[1,1]*x[1] = (1-i)*1 + 3*i = 1-i+3i = 1+2i
    EXPECT_NEAR(y->data[0].real, 3.0, 1e-12);
    EXPECT_NEAR(y->data[0].imag, 1.0, 1e-12);
    EXPECT_NEAR(y->data[1].real, 1.0, 1e-12);
    EXPECT_NEAR(y->data[1].imag, 2.0, 1e-12);

    matx_dense_z_i8_destroy(&a, A);
    matx_vec_z_i8_destroy(&a, x);
    matx_vec_z_i8_destroy(&a, y);
}

TEST(compute_dense, trmm_d_i8_3x3)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t A = NULL, B = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_COL_MAJOR, 3, 3, NULL), MATX_OK);
    ASSERT_EQ(matx_dense_d_i8_create(&a, &B, MATX_COL_MAJOR, 3, 3, NULL), MATX_OK);
    // Upper triangular A: [[2,1,1],[0,3,2],[0,0,4]]
    for (size_t i = 0; i < 9; ++i) A->data[i] = 0.0;
    A->data[0 + 0 * A->stride] = 2.0; A->data[0 + 1 * A->stride] = 1.0; A->data[0 + 2 * A->stride] = 1.0;
    A->data[1 + 1 * A->stride] = 3.0; A->data[1 + 2 * A->stride] = 2.0;
    A->data[2 + 2 * A->stride] = 4.0;
    // B = identity
    for (size_t i = 0; i < 9; ++i) B->data[i] = 0.0;
    B->data[0 + 0 * B->stride] = 1.0; B->data[1 + 1 * B->stride] = 1.0; B->data[2 + 2 * B->stride] = 1.0;

    matx_dense_backend_t blas = matx_blas_default();
    matx_status_t st = matx_trmm_d_i8(&blas, MATX_LEFT, MATX_UPPER, MATX_NO_TRANS, MATX_NON_UNIT_DIAG, 1.0, A, B);
    if (st == MATX_ERR_NOT_SUPPORTED) { matx_dense_d_i8_destroy(&a, A); matx_dense_d_i8_destroy(&a, B); return; }
    ASSERT_EQ(st, MATX_OK);
    // B = A*I = A (upper triangular)
    EXPECT_NEAR(B->data[0 + 0 * B->stride], 2.0, 1e-12);
    EXPECT_NEAR(B->data[0 + 1 * B->stride], 1.0, 1e-12);
    EXPECT_NEAR(B->data[1 + 1 * B->stride], 3.0, 1e-12);
    EXPECT_NEAR(B->data[2 + 2 * B->stride], 4.0, 1e-12);

    matx_dense_d_i8_destroy(&a, A);
    matx_dense_d_i8_destroy(&a, B);
}

TEST(compute_dense, trmm_z_i8_2x2)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_z_i8_t A = NULL, B = NULL;
    ASSERT_EQ(matx_dense_z_i8_create(&a, &A, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);
    ASSERT_EQ(matx_dense_z_i8_create(&a, &B, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);
    for (size_t i = 0; i < 4; ++i) { A->data[i].real = 0.0; A->data[i].imag = 0.0; B->data[i].real = 0.0; B->data[i].imag = 0.0; }
    // Upper triangular: [[1+i, 2], [0, 3]]
    A->data[0 + 0 * A->stride].real = 1.0; A->data[0 + 0 * A->stride].imag = 1.0;
    A->data[0 + 1 * A->stride].real = 2.0; A->data[0 + 1 * A->stride].imag = 0.0;
    A->data[1 + 1 * A->stride].real = 3.0; A->data[1 + 1 * A->stride].imag = 0.0;
    // B = I
    B->data[0 + 0 * B->stride].real = 1.0; B->data[1 + 1 * B->stride].real = 1.0;

    matx_dense_backend_t blas = matx_blas_default();
    matx_complex_d_t alpha = {1.0, 0.0};
    matx_status_t st = matx_trmm_z_i8(&blas, MATX_LEFT, MATX_UPPER, MATX_NO_TRANS, MATX_NON_UNIT_DIAG, alpha, A, B);
    if (st == MATX_ERR_NOT_SUPPORTED) { matx_dense_z_i8_destroy(&a, A); matx_dense_z_i8_destroy(&a, B); return; }
    ASSERT_EQ(st, MATX_OK);
    // B = A*I = A
    EXPECT_NEAR(B->data[0 + 0 * B->stride].real, 1.0, 1e-12);
    EXPECT_NEAR(B->data[0 + 0 * B->stride].imag, 1.0, 1e-12);
    EXPECT_NEAR(B->data[0 + 1 * B->stride].real, 2.0, 1e-12);
    EXPECT_NEAR(B->data[1 + 1 * B->stride].real, 3.0, 1e-12);

    matx_dense_z_i8_destroy(&a, A);
    matx_dense_z_i8_destroy(&a, B);
}

TEST(compute_dense, symm_d_i8_3x3)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t A = NULL, B = NULL, C = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_COL_MAJOR, 3, 3, NULL), MATX_OK);
    ASSERT_EQ(matx_dense_d_i8_create(&a, &B, MATX_COL_MAJOR, 3, 2, NULL), MATX_OK);
    ASSERT_EQ(matx_dense_d_i8_create(&a, &C, MATX_COL_MAJOR, 3, 2, NULL), MATX_OK);
    // Symmetric A: [[2,1,0],[1,3,1],[0,1,4]]
    for (size_t i = 0; i < 9; ++i) A->data[i] = 0.0;
    A->data[0 + 0 * A->stride] = 2.0;
    A->data[1 + 0 * A->stride] = 1.0; A->data[1 + 1 * A->stride] = 3.0;
    A->data[2 + 1 * A->stride] = 1.0; A->data[2 + 2 * A->stride] = 4.0;
    // B: [[1,0],[0,1],[0,0]]
    for (size_t i = 0; i < 6; ++i) B->data[i] = 0.0;
    B->data[0 + 0 * B->stride] = 1.0; B->data[1 + 1 * B->stride] = 1.0;
    for (size_t i = 0; i < 6; ++i) C->data[i] = 1.0;

    matx_dense_backend_t blas = matx_blas_default();
    matx_status_t st = matx_symm_d_i8(&blas, MATX_LEFT, MATX_LOWER, 2.0, A, B, 0.5, C);
    if (st == MATX_ERR_NOT_SUPPORTED) { matx_dense_d_i8_destroy(&a, A); matx_dense_d_i8_destroy(&a, B); matx_dense_d_i8_destroy(&a, C); return; }
    ASSERT_EQ(st, MATX_OK);
    // A*B[:,0] = [2,1,0]  A*B[:,1] = [1,3,1]
    // C = 2*(A*B) + 0.5*ones
    // C[0,0] = 2*2 + 0.5 = 4.5; C[1,0] = 2*1 + 0.5 = 2.5; C[2,0] = 2*0 + 0.5 = 0.5
    // C[0,1] = 2*1 + 0.5 = 2.5; C[1,1] = 2*3 + 0.5 = 6.5; C[2,1] = 2*1 + 0.5 = 2.5
    EXPECT_NEAR(C->data[0 + 0 * C->stride], 4.5, 1e-12);
    EXPECT_NEAR(C->data[1 + 0 * C->stride], 2.5, 1e-12);
    EXPECT_NEAR(C->data[2 + 0 * C->stride], 0.5, 1e-12);
    EXPECT_NEAR(C->data[0 + 1 * C->stride], 2.5, 1e-12);
    EXPECT_NEAR(C->data[1 + 1 * C->stride], 6.5, 1e-12);
    EXPECT_NEAR(C->data[2 + 1 * C->stride], 2.5, 1e-12);

    matx_dense_d_i8_destroy(&a, A);
    matx_dense_d_i8_destroy(&a, B);
    matx_dense_d_i8_destroy(&a, C);
}

TEST(compute_dense, hemm_z_i8_2x2)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_z_i8_t A = NULL, B = NULL, C = NULL;
    ASSERT_EQ(matx_dense_z_i8_create(&a, &A, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);
    ASSERT_EQ(matx_dense_z_i8_create(&a, &B, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);
    ASSERT_EQ(matx_dense_z_i8_create(&a, &C, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);
    for (size_t i = 0; i < 4; ++i) { A->data[i].real = 0.0; A->data[i].imag = 0.0; B->data[i].real = 0.0; B->data[i].imag = 0.0; C->data[i].real = 0.0; C->data[i].imag = 0.0; }
    // Hermitian: [[4, 1+i],[1-i, 3]]
    A->data[0 + 0 * A->stride].real = 4.0;
    A->data[1 + 0 * A->stride].real = 1.0; A->data[1 + 0 * A->stride].imag = -1.0;
    A->data[1 + 1 * A->stride].real = 3.0;
    // B = I
    B->data[0 + 0 * B->stride].real = 1.0; B->data[1 + 1 * B->stride].real = 1.0;

    matx_dense_backend_t blas = matx_blas_default();
    matx_complex_d_t alpha = {1.0, 0.0};
    matx_complex_d_t beta = {0.0, 0.0};
    matx_status_t st = matx_hemm_z_i8(&blas, MATX_LEFT, MATX_LOWER, alpha, A, B, beta, C);
    if (st == MATX_ERR_NOT_SUPPORTED) { matx_dense_z_i8_destroy(&a, A); matx_dense_z_i8_destroy(&a, B); matx_dense_z_i8_destroy(&a, C); return; }
    ASSERT_EQ(st, MATX_OK);
    // C = A*I = A
    EXPECT_NEAR(C->data[0 + 0 * C->stride].real, 4.0, 1e-12);
    EXPECT_NEAR(C->data[1 + 0 * C->stride].real, 1.0, 1e-12);
    EXPECT_NEAR(C->data[1 + 0 * C->stride].imag, -1.0, 1e-12);
    EXPECT_NEAR(C->data[1 + 1 * C->stride].real, 3.0, 1e-12);

    matx_dense_z_i8_destroy(&a, A);
    matx_dense_z_i8_destroy(&a, B);
    matx_dense_z_i8_destroy(&a, C);
}

TEST(compute_dense, syr_d_i8_3x3)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t A = NULL;
    matx_vec_d_i8_t x = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_COL_MAJOR, 3, 3, NULL), MATX_OK);
    ASSERT_EQ(matx_vec_d_i8_create(&a, &x, NULL, 3), MATX_OK);
    // A initial: [[1,0,0],[0,1,0],[0,0,1]]
    for (size_t i = 0; i < 9; ++i) A->data[i] = 0.0;
    A->data[0 + 0 * A->stride] = 1.0; A->data[1 + 1 * A->stride] = 1.0; A->data[2 + 2 * A->stride] = 1.0;
    x->data[0] = 1.0; x->data[1] = 2.0; x->data[2] = 3.0;

    matx_dense_backend_t blas = matx_blas_default();
    matx_status_t st = matx_syr_d_i8(&blas, MATX_LOWER, 2.0, x, A);
    if (st == MATX_ERR_NOT_SUPPORTED) { matx_dense_d_i8_destroy(&a, A); matx_vec_d_i8_destroy(&a, x); return; }
    ASSERT_EQ(st, MATX_OK);
    // A := 2*x*x^T + A
    // Lower triangle only: (0,0)=2*1*1+1=3, (1,0)=2*2*1+0=4, (2,0)=2*3*1+0=6
    // (1,1)=2*2*2+1=9, (2,1)=2*3*2+0=12, (2,2)=2*3*3+1=19
    EXPECT_NEAR(A->data[0 + 0 * A->stride], 3.0, 1e-12);
    EXPECT_NEAR(A->data[1 + 0 * A->stride], 4.0, 1e-12);
    EXPECT_NEAR(A->data[2 + 0 * A->stride], 6.0, 1e-12);
    EXPECT_NEAR(A->data[1 + 1 * A->stride], 9.0, 1e-12);
    EXPECT_NEAR(A->data[2 + 1 * A->stride], 12.0, 1e-12);
    EXPECT_NEAR(A->data[2 + 2 * A->stride], 19.0, 1e-12);

    matx_dense_d_i8_destroy(&a, A);
    matx_vec_d_i8_destroy(&a, x);
}

TEST(compute_dense, her_z_i8_2x2)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_z_i8_t A = NULL;
    matx_vec_z_i8_t x = NULL;
    ASSERT_EQ(matx_dense_z_i8_create(&a, &A, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);
    ASSERT_EQ(matx_vec_z_i8_create(&a, &x, NULL, 2), MATX_OK);
    // A = I
    for (size_t i = 0; i < 4; ++i) { A->data[i].real = 0.0; A->data[i].imag = 0.0; }
    A->data[0 + 0 * A->stride].real = 1.0; A->data[1 + 1 * A->stride].real = 1.0;
    x->data[0].real = 1.0; x->data[0].imag = 1.0;
    x->data[1].real = 2.0; x->data[1].imag = 0.0;

    matx_dense_backend_t blas = matx_blas_default();
    matx_status_t st = matx_her_z_i8(&blas, MATX_LOWER, 2.0, x, A);
    if (st == MATX_ERR_NOT_SUPPORTED) { matx_dense_z_i8_destroy(&a, A); matx_vec_z_i8_destroy(&a, x); return; }
    ASSERT_EQ(st, MATX_OK);
    // A := 2*x*x^H + A
    // x*x^H: |1+i|^2=2, (1+i)*2=2+2i, (1-i)*2=2-2i, 4
    // A(0,0) = 2*2 + 1 = 5 (real)
    // A(1,0) = 2*(2-2i) + 0 = 4-4i
    // A(1,1) = 2*4 + 1 = 9 (real)
    EXPECT_NEAR(A->data[0 + 0 * A->stride].real, 5.0, 1e-12);
    EXPECT_NEAR(A->data[0 + 0 * A->stride].imag, 0.0, 1e-12);
    EXPECT_NEAR(A->data[1 + 0 * A->stride].real, 4.0, 1e-12);
    EXPECT_NEAR(A->data[1 + 0 * A->stride].imag, -4.0, 1e-12);
    EXPECT_NEAR(A->data[1 + 1 * A->stride].real, 9.0, 1e-12);

    matx_dense_z_i8_destroy(&a, A);
    matx_vec_z_i8_destroy(&a, x);
}

TEST(compute_dense, syr2_d_i8_3x3)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_d_i8_t A = NULL;
    matx_vec_d_i8_t x = NULL, y = NULL;
    ASSERT_EQ(matx_dense_d_i8_create(&a, &A, MATX_COL_MAJOR, 3, 3, NULL), MATX_OK);
    ASSERT_EQ(matx_vec_d_i8_create(&a, &x, NULL, 3), MATX_OK);
    ASSERT_EQ(matx_vec_d_i8_create(&a, &y, NULL, 3), MATX_OK);
    // A = I
    for (size_t i = 0; i < 9; ++i) A->data[i] = 0.0;
    A->data[0 + 0 * A->stride] = 1.0; A->data[1 + 1 * A->stride] = 1.0; A->data[2 + 2 * A->stride] = 1.0;
    x->data[0] = 1.0; x->data[1] = 0.0; x->data[2] = 0.0;
    y->data[0] = 0.0; y->data[1] = 1.0; y->data[2] = 0.0;

    matx_dense_backend_t blas = matx_blas_default();
    matx_status_t st = matx_syr2_d_i8(&blas, MATX_LOWER, 1.0, x, y, A);
    if (st == MATX_ERR_NOT_SUPPORTED) { matx_dense_d_i8_destroy(&a, A); matx_vec_d_i8_destroy(&a, x); matx_vec_d_i8_destroy(&a, y); return; }
    ASSERT_EQ(st, MATX_OK);
    // A := x*y^T + y*x^T + A
    // x*y^T: only (0,1)=1 non-zero; y*x^T: only (1,0)=1 non-zero
    // Lower only updated: (1,0)=1+0+0=1, diagonal unchanged
    EXPECT_NEAR(A->data[0 + 0 * A->stride], 1.0, 1e-12);
    EXPECT_NEAR(A->data[1 + 0 * A->stride], 1.0, 1e-12);
    EXPECT_NEAR(A->data[1 + 1 * A->stride], 1.0, 1e-12);
    EXPECT_NEAR(A->data[2 + 2 * A->stride], 1.0, 1e-12);

    matx_dense_d_i8_destroy(&a, A);
    matx_vec_d_i8_destroy(&a, x);
    matx_vec_d_i8_destroy(&a, y);
}

TEST(compute_dense, her2_z_i8_2x2)
{
    matx_alloc_t a = matx_alloc_default();
    matx_dense_z_i8_t A = NULL;
    matx_vec_z_i8_t x = NULL, y = NULL;
    ASSERT_EQ(matx_dense_z_i8_create(&a, &A, MATX_COL_MAJOR, 2, 2, NULL), MATX_OK);
    ASSERT_EQ(matx_vec_z_i8_create(&a, &x, NULL, 2), MATX_OK);
    ASSERT_EQ(matx_vec_z_i8_create(&a, &y, NULL, 2), MATX_OK);
    for (size_t i = 0; i < 4; ++i) { A->data[i].real = 0.0; A->data[i].imag = 0.0; }
    A->data[0 + 0 * A->stride].real = 1.0; A->data[1 + 1 * A->stride].real = 1.0;
    x->data[0].real = 1.0; x->data[0].imag = 0.0;
    x->data[1].real = 0.0; x->data[1].imag = 0.0;
    y->data[0].real = 0.0; y->data[0].imag = 0.0;
    y->data[1].real = 1.0; y->data[1].imag = 0.0;

    matx_dense_backend_t blas = matx_blas_default();
    matx_complex_d_t alpha = {1.0, 0.0};
    matx_status_t st = matx_her2_z_i8(&blas, MATX_LOWER, alpha, x, y, A);
    if (st == MATX_ERR_NOT_SUPPORTED) { matx_dense_z_i8_destroy(&a, A); matx_vec_z_i8_destroy(&a, x); matx_vec_z_i8_destroy(&a, y); return; }
    ASSERT_EQ(st, MATX_OK);
    // A := x*y^H + conj(1)*y*x^H + A = x*y^H + y*x^H + A
    // x*y^H: (0,1)=1, y*x^H: (1,0)=1
    // Lower: A(1,0) = 1+0=1, A(1,1) unchanged
    EXPECT_NEAR(A->data[0 + 0 * A->stride].real, 1.0, 1e-12);
    EXPECT_NEAR(A->data[1 + 0 * A->stride].real, 1.0, 1e-12);
    EXPECT_NEAR(A->data[1 + 1 * A->stride].real, 1.0, 1e-12);

    matx_dense_z_i8_destroy(&a, A);
    matx_vec_z_i8_destroy(&a, x);
    matx_vec_z_i8_destroy(&a, y);
}
