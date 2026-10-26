#include "matx_test_harness.h"

#include <cmath>
#include <cstdio>

extern "C" {
#include "matx/matx_func.h"
#include "matx/matx_sparse_iterative.h"
#include "matx/matx_types.h"
#include "matx/matx_types_internal.h"
}

namespace {

struct IterativeTestResources {
    matx_alloc_t alloc = matx_alloc_default();
    matx_sparse_iterative_solver_t solver = nullptr;
    matx_coo_d_i8_t real_matrix = nullptr;
    matx_coo_z_i8_t complex_matrix = nullptr;
    matx_vec_d_i8_t real_b = nullptr;
    matx_vec_d_i8_t real_x = nullptr;
    matx_vec_z_i8_t complex_b = nullptr;
    matx_vec_z_i8_t complex_x = nullptr;

    ~IterativeTestResources()
    {
        matx_sparse_iterative_solver_destroy(solver);
        if (real_x) matx_vec_d_i8_destroy(&alloc, real_x);
        if (real_b) matx_vec_d_i8_destroy(&alloc, real_b);
        if (complex_x) matx_vec_z_i8_destroy(&alloc, complex_x);
        if (complex_b) matx_vec_z_i8_destroy(&alloc, complex_b);
        if (real_matrix) matx_coo_sparse_d_i8_destroy(&alloc, real_matrix);
        if (complex_matrix) matx_coo_sparse_z_i8_destroy(&alloc, complex_matrix);
    }
};

matx_sparse_iterative_options_t options_for(
    matx_sparse_iterative_method_t method,
    matx_sparse_iterative_preconditioner_t preconditioner)
{
    auto options = matx_sparse_iterative_options_default();
    options.method = method;
    options.preconditioner = preconditioner;
    options.relative_tolerance = 1e-10;
    options.absolute_tolerance = 1e-12;
    options.max_iterations = 200;
    options.gmres_restart = 20;
    return options;
}

void test_real_system(matx_sparse_iterative_backend_kind_t backend,
                      matx_sparse_iterative_method_t method,
                      const matx_int64_t* rows,
                      const matx_int64_t* columns,
                      const matx_double* values,
                      matx_int64_t nnz,
                      const matx_double* rhs,
                      const matx_double* expected,
                      const char* label,
                      matx_sparse_iterative_preconditioner_t preconditioner =
                          MATX_ITERATIVE_PRECOND_JACOBI,
                      bool require_support = true)
{
    constexpr matx_int64_t n = 3;
    IterativeTestResources resources;
    const auto options = options_for(method, preconditioner);
    const auto create_status = matx_sparse_iterative_solver_create(
        &resources.alloc, backend, &options, &resources.solver);
    if (create_status == MATX_ERR_NOT_SUPPORTED) {
        std::printf("[  SKIPPED ] %s / %s: backend is not enabled\n",
                    matx_sparse_iterative_backend_name(backend), label);
        return;
    }
    ASSERT_EQ(create_status, MATX_OK);
    ASSERT_EQ(matx_coo_sparse_d_i8_create(&resources.alloc, &resources.real_matrix,
        n, n, nnz, const_cast<matx_int64_t*>(rows),
        const_cast<matx_int64_t*>(columns), const_cast<matx_double*>(values)), MATX_OK);
    const auto setup_status = matx_sparse_iterative_setup_d_i8(
        resources.solver, resources.real_matrix);
    if (setup_status == MATX_ERR_NOT_SUPPORTED) {
        if (require_support) EXPECT_EQ(setup_status, MATX_OK);
        std::printf("[  SKIPPED ] %s / %s / %s: combination is not supported\n",
                    matx_sparse_iterative_backend_name(backend), label,
                    preconditioner == MATX_ITERATIVE_PRECOND_NONE ? "none" :
                    preconditioner == MATX_ITERATIVE_PRECOND_JACOBI ? "Jacobi" :
                    preconditioner == MATX_ITERATIVE_PRECOND_ILU ? "ILU" : "AMG");
        return;
    }
    ASSERT_EQ(setup_status, MATX_OK);
    ASSERT_EQ(matx_vec_d_i8_create(&resources.alloc, &resources.real_b, nullptr, n), MATX_OK);
    ASSERT_EQ(matx_vec_d_i8_create(&resources.alloc, &resources.real_x, nullptr, n), MATX_OK);
    for (matx_int64_t i = 0; i < n; ++i) {
        resources.real_b->data[i] = rhs[i];
        resources.real_x->data[i] = 0.0;
    }

    matx_sparse_iterative_result_t result{};
    ASSERT_EQ(matx_sparse_iterative_solve_d_i8(resources.solver,
        resources.real_b, resources.real_x, &result), MATX_OK);
    if (result.reason != MATX_ITERATIVE_REASON_CONVERGED ||
        result.residual_norm > 1e-9) {
        std::printf("[ RESULT   ] %s / %s / preconditioner=%d: reason=%d iterations=%d residual=%.3e x=[%.9g, %.9g, %.9g]\n",
                    matx_sparse_iterative_backend_name(backend), label,
                    static_cast<int>(preconditioner), static_cast<int>(result.reason),
                    result.iterations, result.residual_norm,
                    resources.real_x->data[0], resources.real_x->data[1],
                    resources.real_x->data[2]);
    }
    EXPECT_EQ(result.reason, MATX_ITERATIVE_REASON_CONVERGED);
    EXPECT_TRUE(result.residual_norm <= 1e-9);
    for (matx_int64_t i = 0; i < n; ++i) {
        EXPECT_TRUE(std::abs(resources.real_x->data[i] - expected[i]) <= 1e-8);
    }
}

void test_complex_system(matx_sparse_iterative_backend_kind_t backend)
{
    constexpr matx_int64_t n = 2;
    const matx_int64_t rows[] = {0, 0, 1, 1};
    const matx_int64_t columns[] = {0, 1, 0, 1};
    const matx_complex_d_t values[] = {{4.0, 0.0}, {1.0, 1.0},
                                       {1.0, -1.0}, {3.0, 0.0}};
    const matx_complex_d_t rhs[] = {{2.5, 7.5}, {0.0, 2.5}};
    const matx_complex_d_t expected[] = {{1.0, 2.0}, {-1.0, 0.5}};
    IterativeTestResources resources;
    const auto options = options_for(MATX_ITERATIVE_CG,
                                     MATX_ITERATIVE_PRECOND_JACOBI);
    const auto create_status = matx_sparse_iterative_solver_create(
        &resources.alloc, backend, &options, &resources.solver);
    if (create_status == MATX_ERR_NOT_SUPPORTED) {
        std::printf("[  SKIPPED ] %s / complex Hermitian SPD: backend is not enabled\n",
                    matx_sparse_iterative_backend_name(backend));
        return;
    }
    ASSERT_EQ(create_status, MATX_OK);
    ASSERT_EQ(matx_coo_sparse_z_i8_create(&resources.alloc, &resources.complex_matrix,
        n, n, 4, const_cast<matx_int64_t*>(rows),
        const_cast<matx_int64_t*>(columns), const_cast<matx_complex_d_t*>(values)), MATX_OK);
    const auto setup_status = matx_sparse_iterative_setup_z_i8(
        resources.solver, resources.complex_matrix);
    if (setup_status == MATX_ERR_NOT_SUPPORTED) {
        std::printf("[  SKIPPED ] %s / complex Hermitian SPD: precision unsupported\n",
                    matx_sparse_iterative_backend_name(backend));
        return;
    }
    ASSERT_EQ(setup_status, MATX_OK);
    ASSERT_EQ(matx_vec_z_i8_create(&resources.alloc, &resources.complex_b, nullptr, n), MATX_OK);
    ASSERT_EQ(matx_vec_z_i8_create(&resources.alloc, &resources.complex_x, nullptr, n), MATX_OK);
    for (matx_int64_t i = 0; i < n; ++i) {
        resources.complex_b->data[i] = rhs[i];
        resources.complex_x->data[i] = {0.0, 0.0};
    }

    matx_sparse_iterative_result_t result{};
    ASSERT_EQ(matx_sparse_iterative_solve_z_i8(resources.solver,
        resources.complex_b, resources.complex_x, &result), MATX_OK);
    EXPECT_EQ(result.reason, MATX_ITERATIVE_REASON_CONVERGED);
    EXPECT_TRUE(result.residual_norm <= 1e-9);
    for (matx_int64_t i = 0; i < n; ++i) {
        EXPECT_TRUE(std::hypot(resources.complex_x->data[i].real - expected[i].real,
                               resources.complex_x->data[i].imag - expected[i].imag) <= 1e-8);
    }
}

} // namespace

TEST(sparse_iterative, real_spd_cg_and_nonsymmetric_gmres)
{
    const matx_int64_t spd_rows[] = {0, 0, 1, 1, 1, 2, 2};
    const matx_int64_t spd_columns[] = {0, 1, 0, 1, 2, 1, 2};
    const matx_double spd_values[] = {4, -1, -1, 4, -1, -1, 3};
    const matx_double spd_rhs[] = {2, 8, -5};
    const matx_double spd_expected[] = {1, 2, -1};

    const matx_int64_t nonsymmetric_rows[] = {0, 0, 1, 1, 1, 2, 2};
    const matx_int64_t nonsymmetric_columns[] = {0, 1, 0, 1, 2, 1, 2};
    const matx_double nonsymmetric_values[] = {4, 1, -1, 3, 1, -1, 2};
    const matx_double nonsymmetric_rhs[] = {2, -4, 8};
    const matx_double nonsymmetric_expected[] = {1, -2, 3};

    const matx_sparse_iterative_backend_kind_t backends[] = {
        MATX_SPARSE_ITERATIVE_BACKEND_PETSC,
        MATX_SPARSE_ITERATIVE_BACKEND_GINKGO,
        MATX_SPARSE_ITERATIVE_BACKEND_HYPRE};
    for (const auto backend : backends) {
        test_real_system(backend, MATX_ITERATIVE_CG, spd_rows, spd_columns,
                         spd_values, 7, spd_rhs, spd_expected, "SPD CG");
        test_real_system(backend, MATX_ITERATIVE_MINRES, spd_rows, spd_columns,
                         spd_values, 7, spd_rhs, spd_expected, "SPD MINRES",
                         MATX_ITERATIVE_PRECOND_JACOBI,
                         backend != MATX_SPARSE_ITERATIVE_BACKEND_HYPRE);
        test_real_system(backend, MATX_ITERATIVE_BICGSTAB, nonsymmetric_rows,
                         nonsymmetric_columns, nonsymmetric_values, 7,
                         nonsymmetric_rhs, nonsymmetric_expected,
                         "nonsymmetric BiCGSTAB");
        test_real_system(backend, MATX_ITERATIVE_GMRES, nonsymmetric_rows,
                         nonsymmetric_columns, nonsymmetric_values, 7,
                         nonsymmetric_rhs, nonsymmetric_expected,
                         "nonsymmetric GMRES");
        test_real_system(backend, MATX_ITERATIVE_GMRES, nonsymmetric_rows,
                         nonsymmetric_columns, nonsymmetric_values, 7,
                         nonsymmetric_rhs, nonsymmetric_expected,
                         "nonsymmetric GMRES with ILU",
                         MATX_ITERATIVE_PRECOND_ILU,
                         backend != MATX_SPARSE_ITERATIVE_BACKEND_HYPRE);
        test_real_system(backend, MATX_ITERATIVE_CG, spd_rows, spd_columns,
                         spd_values, 7, spd_rhs, spd_expected,
                         "SPD CG with AMG", MATX_ITERATIVE_PRECOND_AMG,
                         backend == MATX_SPARSE_ITERATIVE_BACKEND_HYPRE);
    }
}

TEST(sparse_iterative, complex_hermitian_spd_cg)
{
    const matx_sparse_iterative_backend_kind_t backends[] = {
        MATX_SPARSE_ITERATIVE_BACKEND_PETSC,
        MATX_SPARSE_ITERATIVE_BACKEND_GINKGO,
        MATX_SPARSE_ITERATIVE_BACKEND_HYPRE};
    for (const auto backend : backends) test_complex_system(backend);
}
