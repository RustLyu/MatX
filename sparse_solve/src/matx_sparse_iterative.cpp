#include "matx_sparse_iterative_internal.hpp"

#include "matx/matx_func.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <type_traits>

#ifndef MATX_HAVE_GINKGO
#define MATX_HAVE_GINKGO 0
#endif
#ifndef MATX_HAVE_PETSC
#define MATX_HAVE_PETSC 0
#endif
#ifndef MATX_HAVE_HYPRE
#define MATX_HAVE_HYPRE 0
#endif

using matx::sparse_iterative::Backend;
using matx::sparse_iterative::BackendResult;
using matx::sparse_iterative::CooComplex;
using matx::sparse_iterative::CooReal;

namespace {

std::mutex runtime_mutex;
std::size_t active_iterative_handles = 0;
bool runtime_finalization_started = false;
matx_status_t runtime_finalization_status = MATX_OK;

struct SolverState {
    matx_alloc_t alloc{};
    matx_sparse_iterative_backend_kind_t backend_kind{};
    matx_sparse_iterative_options_t options{};
    std::unique_ptr<Backend> backend;
    bool is_complex = false;
    bool is_setup = false;
    CooReal real_matrix;
    CooComplex complex_matrix;
};

matx_alloc_t valid_allocator(const matx_alloc_t* alloc)
{
    return alloc ? *alloc : matx_alloc_default();
}

bool valid_options(const matx_sparse_iterative_options_t& o)
{
    return o.method >= MATX_ITERATIVE_CG && o.method <= MATX_ITERATIVE_MINRES &&
           o.preconditioner >= MATX_ITERATIVE_PRECOND_NONE &&
           o.preconditioner <= MATX_ITERATIVE_PRECOND_AMG &&
           o.executor >= MATX_ITERATIVE_EXECUTOR_AUTO &&
           o.executor <= MATX_ITERATIVE_EXECUTOR_OMP &&
           std::isfinite(o.relative_tolerance) && o.relative_tolerance > 0.0 &&
           std::isfinite(o.absolute_tolerance) && o.absolute_tolerance >= 0.0 &&
           o.max_iterations > 0 && o.gmres_restart > 0;
}

std::unique_ptr<Backend> create_backend(
    matx_sparse_iterative_backend_kind_t backend,
    const matx_sparse_iterative_options_t& options)
{
    switch (backend) {
    case MATX_SPARSE_ITERATIVE_BACKEND_GINKGO:
#if MATX_HAVE_GINKGO
        return matx::sparse_iterative::make_ginkgo_backend(options);
#else
        return {};
#endif
    case MATX_SPARSE_ITERATIVE_BACKEND_PETSC:
#if MATX_HAVE_PETSC
        return matx::sparse_iterative::make_petsc_backend(options);
#else
        return {};
#endif
    case MATX_SPARSE_ITERATIVE_BACKEND_HYPRE:
#if MATX_HAVE_HYPRE
        return matx::sparse_iterative::make_hypre_backend(options);
#else
        return {};
#endif
    default:
        return {};
    }
}

template <typename T>
double stable_norm(const std::vector<T>& v)
{
    double scale = 0.0;
    double sumsq = 1.0;
    for (const auto& item : v) {
        double components[2] = {};
        if constexpr (std::is_same_v<T, std::complex<double>>) {
            components[0] = std::abs(item.real());
            components[1] = std::abs(item.imag());
        } else {
            components[0] = std::abs(item);
        }
        for (double a : components) {
            if (a == 0.0) continue;
            if (scale < a) {
                const double ratio = scale / a;
                sumsq = 1.0 + sumsq * ratio * ratio;
                scale = a;
            } else {
                const double ratio = a / scale;
                sumsq += ratio * ratio;
            }
        }
    }
    return scale == 0.0 ? 0.0 : scale * std::sqrt(sumsq);
}

template <typename T, typename Matrix>
void compute_residual(const Matrix& matrix,
                      const std::vector<T>& b,
                      const std::vector<T>& x,
                      std::vector<T>& residual)
{
    std::fill(residual.begin(), residual.end(), T{});
    for (std::size_t k = 0; k < matrix.value.size(); ++k) {
        residual[static_cast<std::size_t>(matrix.row[k])] +=
            matrix.value[k] * x[static_cast<std::size_t>(matrix.col[k])];
    }
    for (std::size_t i = 0; i < residual.size(); ++i) {
        residual[i] -= b[i];
    }
}

template <typename Matrix, typename Scalar>
bool valid_coo(const Matrix* A)
{
    if (!A || A->nrows <= 0 || A->ncols <= 0 || A->nrows != A->ncols ||
        A->nnz < 0 || (A->nnz > 0 && (!A->rows || !A->columns || !A->values))) {
        return false;
    }
    for (matx_int64_t k = 0; k < A->nnz; ++k) {
        if (A->rows[k] < 0 || A->rows[k] >= A->nrows ||
            A->columns[k] < 0 || A->columns[k] >= A->ncols) {
            return false;
        }
        if constexpr (std::is_same_v<Scalar, double>) {
            if (!std::isfinite(A->values[k])) return false;
        } else {
            if (!std::isfinite(A->values[k].real) ||
                !std::isfinite(A->values[k].imag)) return false;
        }
    }
    (void)sizeof(Scalar);
    return true;
}

template <typename T, typename VectorHandle, typename Matrix>
matx_status_t solve(SolverState* state,
                    VectorHandle b_handle,
                    VectorHandle x_handle,
                    matx_sparse_iterative_result_t* out,
                    bool complex)
{
  try {
    if (!state || !state->is_setup || state->is_complex != complex ||
        !b_handle || !x_handle || !b_handle->data || !x_handle->data ||
        b_handle->n != x_handle->n || b_handle->n <= 0 ||
        b_handle->stride <= 0 || x_handle->stride <= 0) {
        return MATX_ERR_INVALID_ARG;
    }
    const Matrix& matrix = [&]() -> const Matrix& {
        if constexpr (std::is_same_v<Matrix, CooComplex>) {
            return state->complex_matrix;
        } else {
            return state->real_matrix;
        }
    }();
    if (b_handle->n != matrix.rows) return MATX_ERR_INVALID_ARG;

    std::vector<T> b(static_cast<std::size_t>(b_handle->n));
    std::vector<T> x(static_cast<std::size_t>(x_handle->n));
    for (matx_int64_t i = 0; i < b_handle->n; ++i) {
        if constexpr (std::is_same_v<T, double>) {
            b[static_cast<std::size_t>(i)] = b_handle->data[i * b_handle->stride];
            x[static_cast<std::size_t>(i)] = x_handle->data[i * x_handle->stride];
        } else {
            const auto bv = b_handle->data[i * b_handle->stride];
            const auto xv = x_handle->data[i * x_handle->stride];
            if (!std::isfinite(bv.real) || !std::isfinite(bv.imag) ||
                (state->options.use_initial_guess &&
                 (!std::isfinite(xv.real) || !std::isfinite(xv.imag)))) {
                return MATX_ERR_INVALID_ARG;
            }
            b[static_cast<std::size_t>(i)] = T(bv.real, bv.imag);
            x[static_cast<std::size_t>(i)] = T(xv.real, xv.imag);
        }
        if constexpr (std::is_same_v<T, double>) {
            if (!std::isfinite(b[static_cast<std::size_t>(i)]) ||
                (state->options.use_initial_guess &&
                 !std::isfinite(x[static_cast<std::size_t>(i)]))) {
                return MATX_ERR_INVALID_ARG;
            }
        }
    }

    BackendResult backend_result;
    matx_status_t status;
    if constexpr (std::is_same_v<T, double>) {
        status = state->backend->solve_real(b, x, state->options.use_initial_guess,
                                            backend_result);
    } else {
        status = state->backend->solve_complex(b, x, state->options.use_initial_guess,
                                               backend_result);
    }
    if (status != MATX_OK) return status;

    /* ══ MatX-side residual verification ──────────────────────────────────
     * Backends (Ginkgo, PETSc, hypre) already compute convergence residuals
     * internally.  This second SpMV is useful for debugging but wastes a
     * full COO SpMV in production.  Enable with:
     *   -DMATX_ITERATIVE_VERIFY_RESIDUAL
     */
    matx_sparse_iterative_reason_t reason;
    double rnorm, rel;
#if MATX_ITERATIVE_VERIFY_RESIDUAL
    std::vector<T> residual(b.size());
    compute_residual(matrix, b, x, residual);
    rnorm = stable_norm(residual);
    const double bnorm = stable_norm(b);
    rel = bnorm == 0.0 ? rnorm : rnorm / bnorm;
    const double threshold = std::max(state->options.absolute_tolerance,
                                      state->options.relative_tolerance * bnorm);
    reason = backend_result.reason;
    if (rnorm <= threshold) {
        reason = MATX_ITERATIVE_REASON_CONVERGED;
    } else if (reason != MATX_ITERATIVE_REASON_BREAKDOWN) {
        reason = backend_result.iterations >= state->options.max_iterations
            ? MATX_ITERATIVE_REASON_MAX_ITERATIONS
            : MATX_ITERATIVE_REASON_BREAKDOWN;
    }
#else
    reason = backend_result.reason;
    rnorm = 0.0; /* not computed — trust backend */
    rel = 0.0;
#endif
    for (matx_int64_t i = 0; i < x_handle->n; ++i) {
        if constexpr (std::is_same_v<T, double>) {
            x_handle->data[i * x_handle->stride] = x[static_cast<std::size_t>(i)];
        } else {
            const T value = x[static_cast<std::size_t>(i)];
            x_handle->data[i * x_handle->stride] = {value.real(), value.imag()};
        }
    }
    if (out) {
        out->iterations = backend_result.iterations;
        out->residual_norm = rnorm;
        out->relative_residual = rel;
        out->reason = reason;
    }
    return MATX_OK;
  } catch (const std::bad_alloc&) {
    return MATX_ERR_OUT_OF_MEMORY;
  } catch (...) {
    return MATX_ERR_INTERNAL;
  }
}

} // namespace

struct matx_sparse_iterative_solver_opaque_t {
    SolverState* state;
};

extern "C" {

matx_sparse_iterative_options_t matx_sparse_iterative_options_default(void)
{
    matx_sparse_iterative_options_t options{};
    options.method = MATX_ITERATIVE_GMRES;
    options.preconditioner = MATX_ITERATIVE_PRECOND_JACOBI;
    options.executor = MATX_ITERATIVE_EXECUTOR_AUTO;
    options.relative_tolerance = 1e-8;
    options.absolute_tolerance = 0.0;
    options.max_iterations = 1000;
    options.gmres_restart = 50;
    options.use_initial_guess = false;
    return options;
}

const char* matx_sparse_iterative_backend_name(
    matx_sparse_iterative_backend_kind_t backend)
{
    switch (backend) {
    case MATX_SPARSE_ITERATIVE_BACKEND_GINKGO: return "Ginkgo";
    case MATX_SPARSE_ITERATIVE_BACKEND_PETSC: return "PETSc KSP";
    case MATX_SPARSE_ITERATIVE_BACKEND_HYPRE: return "hypre";
    default: return "unknown";
    }
}

matx_status_t matx_sparse_iterative_solver_create(
    const matx_alloc_t* alloc,
    matx_sparse_iterative_backend_kind_t backend,
    const matx_sparse_iterative_options_t* options,
    matx_sparse_iterative_solver_t* out)
{
    std::lock_guard<std::mutex> runtime_lock(runtime_mutex);
    if (!out || backend < MATX_SPARSE_ITERATIVE_BACKEND_GINKGO ||
        backend > MATX_SPARSE_ITERATIVE_BACKEND_HYPRE) {
        return MATX_ERR_INVALID_ARG;
    }
    *out = nullptr;
    if (runtime_finalization_started) return MATX_ERR_NOT_SUPPORTED;
    const auto selected = options ? *options : matx_sparse_iterative_options_default();
    if (!valid_options(selected)) return MATX_ERR_INVALID_ARG;
    const auto allocator = valid_allocator(alloc);
    if (!allocator.malloc_fn || !allocator.free_fn) return MATX_ERR_INVALID_ARG;

    void* storage = matx_malloc(&allocator, sizeof(matx_sparse_iterative_solver_opaque_t));
    if (!storage) return MATX_ERR_OUT_OF_MEMORY;
    auto* handle = new (storage) matx_sparse_iterative_solver_opaque_t{};
    void* state_storage = matx_malloc(&allocator, sizeof(SolverState));
    if (!state_storage) {
        handle->~matx_sparse_iterative_solver_opaque_t();
        matx_free(&allocator, handle);
        return MATX_ERR_OUT_OF_MEMORY;
    }
    auto* state = new (state_storage) SolverState{};
    state->alloc = allocator;
    state->backend_kind = backend;
    state->options = selected;
    bool backend_allocation_failed = false;
    try {
        state->backend = create_backend(backend, selected);
    } catch (const std::bad_alloc&) {
        backend_allocation_failed = true;
    } catch (...) {
        state->backend.reset();
    }
    if (!state->backend) {
        if (backend_allocation_failed) {
            state->~SolverState();
            matx_free(&allocator, state);
            handle->~matx_sparse_iterative_solver_opaque_t();
            matx_free(&allocator, handle);
            return MATX_ERR_OUT_OF_MEMORY;
        }
        const auto status =
            (backend == MATX_SPARSE_ITERATIVE_BACKEND_GINKGO && !MATX_HAVE_GINKGO) ||
            (backend == MATX_SPARSE_ITERATIVE_BACKEND_PETSC && !MATX_HAVE_PETSC) ||
            (backend == MATX_SPARSE_ITERATIVE_BACKEND_HYPRE && !MATX_HAVE_HYPRE)
                ? MATX_ERR_NOT_SUPPORTED : MATX_ERR_INTERNAL;
        state->~SolverState();
        matx_free(&allocator, state);
        handle->~matx_sparse_iterative_solver_opaque_t();
        matx_free(&allocator, handle);
        return status;
    }
    handle->state = state;
    *out = handle;
    ++active_iterative_handles;
    return MATX_OK;
}

matx_status_t matx_sparse_iterative_setup_d_i8(
    matx_sparse_iterative_solver_t solver, matx_coo_d_i8_t A)
{
    if (!solver || !solver->state || !valid_coo<matx_coo_d_i8_opaque_t, double>(A)) {
        return MATX_ERR_INVALID_ARG;
    }
    CooReal matrix;
    matrix.rows = A->nrows;
    matrix.cols = A->ncols;
    try {
        if (A->nnz > 0) {
            matrix.row.assign(A->rows, A->rows + A->nnz);
            matrix.col.assign(A->columns, A->columns + A->nnz);
            matrix.value.assign(A->values, A->values + A->nnz);
        }
    } catch (...) {
        return MATX_ERR_OUT_OF_MEMORY;
    }
    auto* state = solver->state;
    state->is_setup = false;
    matx_status_t status;
    try {
        status = state->backend->setup_real(matrix);
    } catch (const std::bad_alloc&) {
        return MATX_ERR_OUT_OF_MEMORY;
    } catch (...) {
        return MATX_ERR_INTERNAL;
    }
    if (status == MATX_OK) {
        state->real_matrix = std::move(matrix);
        state->complex_matrix = {};
        state->is_complex = false;
        state->is_setup = true;
    }
    return status;
}

matx_status_t matx_sparse_iterative_setup_z_i8(
    matx_sparse_iterative_solver_t solver, matx_coo_z_i8_t A)
{
    if (!solver || !solver->state || !valid_coo<matx_coo_z_i8_opaque_t, matx_complex_d_t>(A)) {
        return MATX_ERR_INVALID_ARG;
    }
    CooComplex matrix;
    matrix.rows = A->nrows;
    matrix.cols = A->ncols;
    try {
        if (A->nnz > 0) {
            matrix.row.assign(A->rows, A->rows + A->nnz);
            matrix.col.assign(A->columns, A->columns + A->nnz);
        }
        matrix.value.reserve(static_cast<std::size_t>(A->nnz));
        for (matx_int64_t k = 0; k < A->nnz; ++k) {
            matrix.value.emplace_back(A->values[k].real, A->values[k].imag);
        }
    } catch (...) {
        return MATX_ERR_OUT_OF_MEMORY;
    }
    auto* state = solver->state;
    state->is_setup = false;
    matx_status_t status;
    try {
        status = state->backend->setup_complex(matrix);
    } catch (const std::bad_alloc&) {
        return MATX_ERR_OUT_OF_MEMORY;
    } catch (...) {
        return MATX_ERR_INTERNAL;
    }
    if (status == MATX_OK) {
        state->complex_matrix = std::move(matrix);
        state->real_matrix = {};
        state->is_complex = true;
        state->is_setup = true;
    }
    return status;
}

matx_status_t matx_sparse_iterative_solve_d_i8(
    matx_sparse_iterative_solver_t solver, const matx_vec_d_i8_t b,
    matx_vec_d_i8_t x, matx_sparse_iterative_result_t* result)
{
    if (!solver || !solver->state) return MATX_ERR_INVALID_ARG;
    return solve<double, matx_vec_d_i8_t, CooReal>(solver->state, b, x,
                                                          result, false);
}

matx_status_t matx_sparse_iterative_solve_z_i8(
    matx_sparse_iterative_solver_t solver, const matx_vec_z_i8_t b,
    matx_vec_z_i8_t x, matx_sparse_iterative_result_t* result)
{
    if (!solver || !solver->state) return MATX_ERR_INVALID_ARG;
    return solve<std::complex<double>, matx_vec_z_i8_t, CooComplex>(
        solver->state, b, x, result, true);
}

void matx_sparse_iterative_solver_destroy(matx_sparse_iterative_solver_t solver)
{
    if (!solver) return;
    std::lock_guard<std::mutex> runtime_lock(runtime_mutex);
    matx_alloc_t alloc = solver->state ? solver->state->alloc : matx_alloc_default();
    if (solver->state) {
        solver->state->~SolverState();
        matx_free(&alloc, solver->state);
    }
    solver->~matx_sparse_iterative_solver_opaque_t();
    matx_free(&alloc, solver);
    if (active_iterative_handles > 0) --active_iterative_handles;
}

matx_status_t matx_sparse_iterative_runtime_finalize(void)
{
    std::lock_guard<std::mutex> runtime_lock(runtime_mutex);
    if (active_iterative_handles != 0) return MATX_ERR_INVALID_ARG;
    if (runtime_finalization_started) return runtime_finalization_status;
    runtime_finalization_started = true;

    runtime_finalization_status = matx::sparse_iterative::finalize_hypre_runtime();
    if (runtime_finalization_status == MATX_OK) {
        runtime_finalization_status = matx::sparse_iterative::finalize_petsc_runtime();
    }
    if (runtime_finalization_status == MATX_OK) {
        runtime_finalization_status = matx::sparse_iterative::finalize_hypre_mpi_runtime();
    }
    return runtime_finalization_status;
}

} // extern "C"
