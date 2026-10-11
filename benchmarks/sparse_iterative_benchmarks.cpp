#include "matx/matx_func.h"
#include "matx/matx_sparse_iterative.h"
#include "matx/matx_sparse_solve.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Workload {
    const char* name = nullptr;
    matx_sparse_iterative_method_t method{};
    matx_int64_t n = 0;
    matx_int64_t grid_columns = 0;
    std::vector<matx_int64_t> rows;
    std::vector<matx_int64_t> columns;
    std::vector<matx_double> values;
};

struct RhsSample {
    std::vector<matx_double> exact;
    std::vector<matx_double> rhs;
};

struct Resources {
    matx_alloc_t alloc = matx_alloc_default();
    matx_sparse_iterative_solver_t solver = nullptr;
    matx_coo_d_i8_t matrix = nullptr;
    matx_vec_d_i8_t rhs = nullptr;
    matx_vec_d_i8_t solution = nullptr;

    ~Resources()
    {
        matx_sparse_iterative_solver_destroy(solver);
        if (solution) matx_vec_d_i8_destroy(&alloc, solution);
        if (rhs) matx_vec_d_i8_destroy(&alloc, rhs);
        if (matrix) matx_coo_sparse_d_i8_destroy(&alloc, matrix);
    }
};

double milliseconds(Clock::duration duration)
{
    return std::chrono::duration<double, std::milli>(duration).count();
}

Workload make_workload(matx_int64_t n, bool nonsymmetric)
{
    Workload workload;
    workload.name = nonsymmetric ? "convection_diffusion" : "poisson_2d";
    workload.method = nonsymmetric ? MATX_ITERATIVE_GMRES : MATX_ITERATIVE_CG;
    workload.n = n;
    workload.grid_columns = std::max<matx_int64_t>(
        1, static_cast<matx_int64_t>(std::sqrt(static_cast<double>(n))));
    workload.rows.reserve(static_cast<std::size_t>(5 * n));
    workload.columns.reserve(static_cast<std::size_t>(5 * n));
    workload.values.reserve(static_cast<std::size_t>(5 * n));

    for (matx_int64_t index = 0; index < n; ++index) {
        const auto row = index / workload.grid_columns;
        const auto col = index % workload.grid_columns;
        auto add = [&](matx_int64_t column, matx_double value) {
            workload.rows.push_back(index);
            workload.columns.push_back(column);
            workload.values.push_back(value);
        };

        add(index, nonsymmetric ? 4.2 : 4.0);
        if (col > 0) add(index - 1, nonsymmetric ? -1.15 : -1.0);
        if (col + 1 < workload.grid_columns && index + 1 < n)
            add(index + 1, nonsymmetric ? -0.85 : -1.0);
        if (row > 0) add(index - workload.grid_columns,
                         nonsymmetric ? -1.1 : -1.0);
        if (index + workload.grid_columns < n)
            add(index + workload.grid_columns, nonsymmetric ? -0.9 : -1.0);
    }
    return workload;
}

RhsSample make_rhs_sample(const Workload& workload, int sample_index)
{
    RhsSample sample;
    sample.exact.resize(static_cast<std::size_t>(workload.n));
    sample.rhs.assign(static_cast<std::size_t>(workload.n), 0.0);
    for (matx_int64_t index = 0; index < workload.n; ++index) {
        const auto first_phase = (index * 17 + sample_index * 29) % 101;
        const auto second_phase = (index * 13 + sample_index * 23) % 89;
        sample.exact[static_cast<std::size_t>(index)] =
            0.5 + std::sin(static_cast<double>(first_phase) * 0.0622) +
            0.2 * std::cos(static_cast<double>(second_phase) * 0.071);
    }
    for (std::size_t k = 0; k < workload.values.size(); ++k) {
        sample.rhs[static_cast<std::size_t>(workload.rows[k])] +=
            workload.values[k] *
            sample.exact[static_cast<std::size_t>(workload.columns[k])];
    }
    return sample;
}

double relative_solution_error(const std::vector<matx_double>& exact,
                               const std::vector<matx_double>& solution)
{
    long double error_squared = 0.0;
    long double exact_squared = 0.0;
    for (std::size_t i = 0; i < solution.size(); ++i) {
        const long double error = solution[i] - exact[i];
        error_squared += error * error;
        exact_squared += static_cast<long double>(exact[i]) * exact[i];
    }
    return std::sqrt(static_cast<double>(error_squared / exact_squared));
}

const char* method_name(matx_sparse_iterative_method_t method)
{
    switch (method) {
    case MATX_ITERATIVE_CG: return "CG";
    case MATX_ITERATIVE_BICGSTAB: return "BiCGSTAB";
    case MATX_ITERATIVE_GMRES: return "GMRES";
    case MATX_ITERATIVE_MINRES: return "MINRES";
    default: return "unknown";
    }
}

const char* preconditioner_name(matx_sparse_iterative_preconditioner_t preconditioner)
{
    switch (preconditioner) {
    case MATX_ITERATIVE_PRECOND_NONE: return "none";
    case MATX_ITERATIVE_PRECOND_JACOBI: return "Jacobi";
    case MATX_ITERATIVE_PRECOND_ILU: return "ILU";
    case MATX_ITERATIVE_PRECOND_AMG: return "AMG";
    default: return "unknown";
    }
}

bool run_case(matx_sparse_iterative_backend_kind_t backend,
              const Workload& workload,
              matx_sparse_iterative_preconditioner_t preconditioner,
              int repeats)
{
    const auto n = workload.n;
    Resources resources;
    auto options = matx_sparse_iterative_options_default();
    options.method = workload.method;
    options.preconditioner = preconditioner;
    options.relative_tolerance = 1e-8;
    options.absolute_tolerance = 1e-12;
    options.max_iterations = 3000;
    options.gmres_restart = 50;

    const auto setup_start = Clock::now();
    auto status = matx_sparse_iterative_solver_create(
        &resources.alloc, backend, &options, &resources.solver);
    if (status == MATX_OK) {
        status = matx_coo_sparse_d_i8_create(&resources.alloc, &resources.matrix,
            n, n, static_cast<matx_int64_t>(workload.values.size()),
            const_cast<matx_int64_t*>(workload.rows.data()),
            const_cast<matx_int64_t*>(workload.columns.data()),
            const_cast<matx_double*>(workload.values.data()));
    }
    if (status == MATX_OK) {
        status = matx_sparse_iterative_setup_d_i8(resources.solver, resources.matrix);
    }
    const double setup_ms = milliseconds(Clock::now() - setup_start);
    if (status == MATX_ERR_NOT_SUPPORTED) {
        std::cout << matx_sparse_iterative_backend_name(backend) << ','
                  << workload.name << ',' << method_name(workload.method)
                  << ',' << preconditioner_name(preconditioner) << ','
                  << n << ',' << workload.values.size() << ','
                  << setup_ms << ",0,0,0,0,0,skipped\n";
        return true;
    }
    if (status != MATX_OK) {
        std::cerr << matx_sparse_iterative_backend_name(backend) << ' '
                  << workload.name << " setup failed: " << matx_status_string(status) << '\n';
        return false;
    }

    auto warmup_rhs = make_rhs_sample(workload, 0);
    std::vector<matx_double> b_data = std::move(warmup_rhs.rhs);
    std::vector<matx_double> exact_data = std::move(warmup_rhs.exact);
    std::vector<matx_double> x_data(static_cast<std::size_t>(n), 0.0);
    if (matx_vec_d_i8_wrap(&resources.alloc, &resources.rhs, n, 1, b_data.data()) != MATX_OK ||
        matx_vec_d_i8_wrap(&resources.alloc, &resources.solution, n, 1, x_data.data()) != MATX_OK) {
        std::cerr << "failed to create benchmark vectors\n";
        return false;
    }

    matx_sparse_iterative_result_t result{};
    status = matx_sparse_iterative_solve_d_i8(
        resources.solver, resources.rhs, resources.solution, &result);
    if (status != MATX_OK || result.reason != MATX_ITERATIVE_REASON_CONVERGED) {
        std::cerr << matx_sparse_iterative_backend_name(backend) << ' '
                  << workload.name << " warmup did not converge (status="
                  << matx_status_string(status) << ", reason=" << result.reason
                  << ", residual=" << result.residual_norm << ")\n";
        return false;
    }
    constexpr double solution_error_limit = 1e-5;
    double max_relative_solution_error = relative_solution_error(exact_data, x_data);
    double max_residual_norm = result.residual_norm;
    double max_relative_residual = result.relative_residual;
    bool solution_is_accurate = std::isfinite(max_relative_solution_error) &&
                                max_relative_solution_error <= solution_error_limit;

    std::vector<double> samples;
    std::vector<matx_int64_t> iteration_samples;
    samples.reserve(static_cast<std::size_t>(repeats));
    iteration_samples.reserve(static_cast<std::size_t>(repeats));
    for (int sample = 0; sample < repeats; ++sample) {
        auto rhs_sample = make_rhs_sample(workload, sample + 1);
        std::copy(rhs_sample.rhs.begin(), rhs_sample.rhs.end(), b_data.begin());
        exact_data.swap(rhs_sample.exact);
        std::fill(x_data.begin(), x_data.end(), 0.0);
        const auto start = Clock::now();
        status = matx_sparse_iterative_solve_d_i8(
            resources.solver, resources.rhs, resources.solution, &result);
        const auto elapsed = milliseconds(Clock::now() - start);
        if (status != MATX_OK || result.reason != MATX_ITERATIVE_REASON_CONVERGED) {
            std::cerr << matx_sparse_iterative_backend_name(backend) << ' '
                      << workload.name << " failed during timed solve (status="
                      << matx_status_string(status) << ", reason=" << result.reason
                      << ", residual=" << result.residual_norm << ")\n";
            return false;
        }
        samples.push_back(elapsed);
        iteration_samples.push_back(result.iterations);
        max_residual_norm = std::max(max_residual_norm, result.residual_norm);
        max_relative_residual = std::max(max_relative_residual,
                                         result.relative_residual);
        const double sample_solution_error = relative_solution_error(exact_data, x_data);
        max_relative_solution_error = std::max(max_relative_solution_error,
                                               sample_solution_error);
        solution_is_accurate = solution_is_accurate &&
            std::isfinite(sample_solution_error) &&
            sample_solution_error <= solution_error_limit;
    }
    std::sort(samples.begin(), samples.end());
    const double median_ms = samples[samples.size() / 2];
    std::sort(iteration_samples.begin(), iteration_samples.end());
    const auto median_iterations = iteration_samples[iteration_samples.size() / 2];

    std::cout << matx_sparse_iterative_backend_name(backend) << ','
              << workload.name << ',' << method_name(workload.method)
              << ',' << preconditioner_name(preconditioner) << ','
              << n << ',' << workload.values.size() << ','
              << setup_ms << ',' << median_ms << ',' << median_iterations << ','
              << max_residual_norm << ',' << max_relative_residual << ','
              << max_relative_solution_error << ','
              << (solution_is_accurate ? "ok" : "inaccurate_solution") << '\n';
    if (!solution_is_accurate) {
        std::cerr << matx_sparse_iterative_backend_name(backend) << ' '
                  << workload.name << " relative solution error "
                  << max_relative_solution_error << " exceeds "
                  << solution_error_limit << '\n';
    }
    return solution_is_accurate;
}

struct DirectBackend {
    const char* name;
    matx_sparse_linsolve_backend_kind_t kind;
    bool cholesky;
};

struct DirectResources {
    matx_alloc_t alloc = matx_alloc_default();
    matx_sparse_linsolve_t solver{};
    matx_coo_d_i8_t matrix = nullptr;
    matx_factor_sparse_d_i8_t factor{};
    bool cholesky = false;

    ~DirectResources()
    {
        if (factor.reserved) {
            if (cholesky)
                matx_factor_chol_coo_d_i8_destroy(&solver, &factor);
            else
                matx_factor_csc_d_i8_destroy(&solver, &factor);
        }
        if (matrix) matx_coo_sparse_d_i8_destroy(&alloc, matrix);
    }
};

struct Accuracy {
    double residual_norm = 0.0;
    double relative_residual = 0.0;
    double relative_solution_error = 0.0;
    bool ok = false;
};

Accuracy check_solution(const Workload& workload,
                        const std::vector<double>& rhs,
                        const std::vector<double>& exact,
                        const std::vector<double>& solution)
{
    std::vector<long double> product(rhs.size(), 0.0L);
    for (std::size_t k = 0; k < workload.values.size(); ++k) {
        product[static_cast<std::size_t>(workload.rows[k])] +=
            static_cast<long double>(workload.values[k]) *
            solution[static_cast<std::size_t>(workload.columns[k])];
    }

    long double residual_squared = 0.0L;
    long double rhs_squared = 0.0L;
    long double error_squared = 0.0L;
    long double exact_squared = 0.0L;
    for (std::size_t i = 0; i < solution.size(); ++i) {
        const long double residual = rhs[i] - product[i];
        const long double error = solution[i] - exact[i];
        residual_squared += residual * residual;
        rhs_squared += static_cast<long double>(rhs[i]) * rhs[i];
        error_squared += error * error;
        exact_squared += static_cast<long double>(exact[i]) * exact[i];
    }

    Accuracy accuracy;
    accuracy.residual_norm = std::sqrt(static_cast<double>(residual_squared));
    accuracy.relative_residual = std::sqrt(static_cast<double>(
        residual_squared / std::max(rhs_squared, 1e-300L)));
    accuracy.relative_solution_error = std::sqrt(static_cast<double>(
        error_squared / std::max(exact_squared, 1e-300L)));
    accuracy.ok = std::isfinite(accuracy.relative_residual) &&
                  std::isfinite(accuracy.relative_solution_error) &&
                  accuracy.relative_residual <= 1e-7 &&
                  accuracy.relative_solution_error <= 1e-6;
    return accuracy;
}

bool run_direct_case(const DirectBackend& backend,
                     const Workload& workload,
                     int repeats)
{
    const auto n = workload.n;
    if (backend.cholesky && workload.method != MATX_ITERATIVE_CG) {
        std::cout << backend.name << ',' << workload.name
                  << ",Cholesky,none," << n << ',' << workload.values.size()
                  << ",0,0,0,0,0,0,skipped\n";
        return true;
    }

    auto warmup_rhs = make_rhs_sample(workload, 0);
    std::vector<matx_double> rhs_data = std::move(warmup_rhs.rhs);
    std::vector<matx_double> exact_data = std::move(warmup_rhs.exact);
    DirectResources resources;
    resources.solver = matx_sparse_linsolve_by_type(backend.kind, resources.alloc);
    resources.cholesky = backend.cholesky;
    auto status = matx_coo_sparse_d_i8_create(
        &resources.alloc, &resources.matrix, n, n,
        static_cast<matx_int64_t>(workload.values.size()),
        const_cast<matx_int64_t*>(workload.rows.data()),
        const_cast<matx_int64_t*>(workload.columns.data()),
        const_cast<matx_double*>(workload.values.data()));
    double setup_ms = 0.0;
    if (status == MATX_OK) {
        const auto factor_start = Clock::now();
        if (backend.cholesky) {
            status = matx_factor_chol_coo_d_i8(
                &resources.solver, resources.matrix, &resources.factor);
        } else {
            status = matx_factor_csc_d_i8(
                &resources.solver, resources.matrix, &resources.factor);
        }
        setup_ms = milliseconds(Clock::now() - factor_start);
    }
    if (status == MATX_ERR_NOT_SUPPORTED) {
        std::cout << backend.name << ',' << workload.name << ','
                  << (backend.cholesky ? "Cholesky" : "LU")
                  << ",none," << n << ',' << workload.values.size() << ','
                  << setup_ms << ",0,0,0,0,0,skipped\n";
        return true;
    }
    if (status != MATX_OK) {
        std::cerr << backend.name << ' ' << workload.name
                  << " factorization failed: " << matx_status_string(status) << '\n';
        return false;
    }

    std::vector<double> solution(static_cast<std::size_t>(n), 0.0);
    const auto solve = [&]() {
        if (backend.cholesky) {
            return matx_solve_chol_coo_d_i8_factor(
                &resources.solver, &resources.factor,
                rhs_data.data(), solution.data());
        }
        return matx_solve_csc_d_i8_factor(
            &resources.solver, &resources.factor,
            rhs_data.data(), solution.data());
    };

    status = solve();
    if (status != MATX_OK) {
        std::cerr << backend.name << ' ' << workload.name
                  << " first solve failed: " << matx_status_string(status) << '\n';
        return false;
    }

    Accuracy accuracy = check_solution(workload, rhs_data, exact_data, solution);
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeats));
    for (int sample = 0; sample < repeats; ++sample) {
        auto rhs_sample = make_rhs_sample(workload, sample + 1);
        std::copy(rhs_sample.rhs.begin(), rhs_sample.rhs.end(), rhs_data.begin());
        exact_data.swap(rhs_sample.exact);
        std::fill(solution.begin(), solution.end(), 0.0);
        const auto start = Clock::now();
        status = solve();
        const double elapsed = milliseconds(Clock::now() - start);
        if (status != MATX_OK) {
            std::cerr << backend.name << ' ' << workload.name
                      << " timed solve failed: " << matx_status_string(status) << '\n';
            return false;
        }
        samples.push_back(elapsed);
        const auto sample_accuracy = check_solution(
            workload, rhs_data, exact_data, solution);
        accuracy.residual_norm = std::max(accuracy.residual_norm,
                                           sample_accuracy.residual_norm);
        accuracy.relative_residual = std::max(accuracy.relative_residual,
                                               sample_accuracy.relative_residual);
        accuracy.relative_solution_error = std::max(
            accuracy.relative_solution_error,
            sample_accuracy.relative_solution_error);
        accuracy.ok = accuracy.ok && sample_accuracy.ok;
    }
    std::sort(samples.begin(), samples.end());
    const double median_ms = samples[samples.size() / 2];

    std::cout << backend.name << ',' << workload.name << ','
              << (backend.cholesky ? "Cholesky" : "LU")
              << ",none," << n << ',' << workload.values.size() << ','
              << setup_ms << ',' << median_ms << ",0," << accuracy.residual_norm << ','
              << accuracy.relative_residual << ','
              << accuracy.relative_solution_error << ','
              << (accuracy.ok ? "ok" : "inaccurate_solution") << '\n';
    if (!accuracy.ok) {
        std::cerr << backend.name << ' ' << workload.name
                  << " residual/solution accuracy check failed (relative residual="
                  << accuracy.relative_residual << ", relative solution error="
                  << accuracy.relative_solution_error << ")\n";
    }
    return accuracy.ok;
}

bool parse_positive(const char* text, int* value)
{
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (!end || *end != '\0' || parsed <= 0 || parsed > 1048576) return false;
    *value = static_cast<int>(parsed);
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    int dimension = 65536;
    int repeats = 7;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--side" && i + 1 < argc) {
            int side = 0;
            if (!parse_positive(argv[++i], &side) || side > 1024) {
                std::cerr << "--side must be an integer in [1, 1024]\n";
                return 2;
            }
            dimension = side * side;
        } else if (std::string(argv[i]) == "--size" && i + 1 < argc) {
            if (!parse_positive(argv[++i], &dimension)) {
                std::cerr << "--size must be an integer in [1, 1048576]\n";
                return 2;
            }
        } else if (std::string(argv[i]) == "--repeats" && i + 1 < argc) {
            if (!parse_positive(argv[++i], &repeats) || repeats > 101) {
                std::cerr << "--repeats must be an integer in [1, 101]\n";
                return 2;
            }
        } else {
            std::cerr << "usage: " << argv[0]
                      << " [--size unknowns | --side grid_width]"
                         " [--repeats count]\n";
            return 2;
        }
    }

    std::cout << "backend,problem,method,preconditioner,n,nnz,setup_ms,"
                 "median_solve_ms,iterations,"
                 "residual_norm,relative_residual,"
                 "relative_solution_error,status\n";
    bool succeeded = true;
    const matx_sparse_iterative_backend_kind_t backends[] = {
        MATX_SPARSE_ITERATIVE_BACKEND_GINKGO,
        MATX_SPARSE_ITERATIVE_BACKEND_PETSC,
        MATX_SPARSE_ITERATIVE_BACKEND_HYPRE};
    const auto spd = make_workload(dimension, false);
    const auto nonsymmetric = make_workload(dimension, true);
    for (const auto backend : backends) {
        succeeded = run_case(backend, spd, MATX_ITERATIVE_PRECOND_NONE,
                             repeats) && succeeded;
        succeeded = run_case(backend, spd, MATX_ITERATIVE_PRECOND_JACOBI,
                             repeats) && succeeded;
        succeeded = run_case(backend, spd, MATX_ITERATIVE_PRECOND_AMG,
                             repeats) && succeeded;
        succeeded = run_case(backend, nonsymmetric, MATX_ITERATIVE_PRECOND_NONE,
                             repeats) && succeeded;
        succeeded = run_case(backend, nonsymmetric, MATX_ITERATIVE_PRECOND_JACOBI,
                             repeats) && succeeded;
        succeeded = run_case(backend, nonsymmetric, MATX_ITERATIVE_PRECOND_ILU,
                             repeats) && succeeded;
    }
    const DirectBackend direct_backends[] = {
        {"SuiteSparse KLU", MATX_LINSOLVE_BACKEND_SUITESPARSE_KLU, false},
        {"UMFPACK", MATX_LINSOLVE_BACKEND_UMFPACK, false},
        {"CXSparse", MATX_LINSOLVE_BACKEND_CXSPARSE, false},
        {"SuperLU", MATX_LINSOLVE_BACKEND_SUPERLU, false},
        {"MUMPS", MATX_LINSOLVE_BACKEND_MUMPS, false},
        {"SuiteSparse CHOLMOD", MATX_LINSOLVE_BACKEND_SUITESPARSE_KLU, true}};
    for (const auto& backend : direct_backends) {
        succeeded = run_direct_case(backend, spd, repeats) && succeeded;
        succeeded = run_direct_case(backend, nonsymmetric, repeats) && succeeded;
    }
    const auto finalize_status = matx_sparse_iterative_runtime_finalize();
    if (finalize_status != MATX_OK) {
        std::cerr << "iterative runtime finalization failed: "
                  << matx_status_string(finalize_status) << '\n';
        if (finalize_status != MATX_ERR_NOT_SUPPORTED) succeeded = false;
    }
    return succeeded ? 0 : 1;
}
