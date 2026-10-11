#include "matx_sparse_iterative_internal.hpp"

#ifndef MATX_HAVE_HYPRE
#define MATX_HAVE_HYPRE 0
#endif

#if MATX_HAVE_HYPRE
#include <HYPRE.h>
#include <HYPRE_IJ_mv.h>
#include <HYPRE_parcsr_ls.h>
#include <HYPRE_parcsr_mv.h>
#if !defined(MATX_HYPRE_SEQUENTIAL_BUILD)
#include <mpi.h>
#endif

#include <limits>
#include <map>
#include <mutex>
#include <type_traits>

namespace matx::sparse_iterative {
namespace {

std::mutex hypre_mutex;
bool hypre_initialization_attempted = false;
bool hypre_initialization_succeeded = false;
bool hypre_owned = false;
#if !defined(MATX_HYPRE_SEQUENTIAL_BUILD)
bool mpi_owned = false;
#endif

MPI_Comm hypre_comm_self()
{
#if defined(MATX_HYPRE_SEQUENTIAL_BUILD)
    return static_cast<MPI_Comm>(0);
#else
    return MPI_COMM_SELF;
#endif
}

bool acquire_hypre()
{
    std::lock_guard<std::mutex> lock(hypre_mutex);
    if (!hypre_initialization_attempted) {
        hypre_initialization_attempted = true;
#if !defined(MATX_HYPRE_SEQUENTIAL_BUILD)
        int initialized = 0;
        int finalized = 0;
        if (MPI_Initialized(&initialized) != MPI_SUCCESS ||
            MPI_Finalized(&finalized) != MPI_SUCCESS || finalized) return false;
        if (!initialized) {
            if (MPI_Init(nullptr, nullptr) != MPI_SUCCESS) return false;
            mpi_owned = true;
        }
#endif
#if defined(HYPRE_RELEASE_NUMBER) && HYPRE_RELEASE_NUMBER >= 22900
        if (!HYPRE_Initialized()) {
            if (HYPRE_Initialize() != 0) return false;
            hypre_owned = true;
        }
#else
        if (HYPRE_Init() != 0) return false;
        hypre_owned = true;
#endif
        hypre_initialization_succeeded = true;
    }
    return hypre_initialization_succeeded;
}

class HypreBackend final : public Backend {
public:
    explicit HypreBackend(const matx_sparse_iterative_options_t& options)
        : options_(options), runtime_ok_(acquire_hypre()) {}

    ~HypreBackend() override
    {
        clear();
    }

    matx_status_t setup_real(const CooReal& matrix) override
    {
        if (!runtime_ok_) return MATX_ERR_INTERNAL;
        if constexpr (!std::is_same_v<HYPRE_Real, double>) {
            (void)matrix;
            return MATX_ERR_NOT_SUPPORTED;
        }
        if (options_.method == MATX_ITERATIVE_MINRES ||
            options_.preconditioner == MATX_ITERATIVE_PRECOND_ILU) {
            return MATX_ERR_NOT_SUPPORTED;
        }
        if (matrix.rows <= 0 || matrix.rows > std::numeric_limits<HYPRE_BigInt>::max() ||
            matrix.rows > std::numeric_limits<HYPRE_Int>::max() ||
            matrix.value.size() > static_cast<std::size_t>(std::numeric_limits<HYPRE_Int>::max()) ||
            options_.max_iterations > std::numeric_limits<HYPRE_Int>::max() ||
            options_.gmres_restart > std::numeric_limits<HYPRE_Int>::max()) {
            return MATX_ERR_NOT_SUPPORTED;
        }
        clear_objects();
        const auto lo = static_cast<HYPRE_BigInt>(0);
        const auto hi = static_cast<HYPRE_BigInt>(matrix.rows - 1);
        if (HYPRE_IJMatrixCreate(hypre_comm_self(), lo, hi, lo, hi, &ij_matrix_) != 0 ||
            HYPRE_IJMatrixSetObjectType(ij_matrix_, HYPRE_PARCSR) != 0 ||
            HYPRE_IJMatrixInitialize(ij_matrix_) != 0) {
            clear_objects();
            return MATX_ERR_INTERNAL;
        }

        std::vector<std::map<HYPRE_BigInt, HYPRE_Complex>> rows(
            static_cast<std::size_t>(matrix.rows));
        for (std::size_t k = 0; k < matrix.value.size(); ++k) {
            rows[static_cast<std::size_t>(matrix.row[k])]
                [static_cast<HYPRE_BigInt>(matrix.col[k])] += matrix.value[k];
        }
        for (std::int64_t r = 0; r < matrix.rows; ++r) {
            auto& entries = rows[static_cast<std::size_t>(r)];
            if (entries.empty()) continue;
            if (entries.size() > static_cast<std::size_t>(std::numeric_limits<HYPRE_Int>::max())) {
                clear_objects();
                return MATX_ERR_NOT_SUPPORTED;
            }
            HYPRE_Int count = static_cast<HYPRE_Int>(entries.size());
            HYPRE_BigInt row = static_cast<HYPRE_BigInt>(r);
            std::vector<HYPRE_BigInt> columns;
            std::vector<HYPRE_Complex> values;
            columns.reserve(entries.size());
            values.reserve(entries.size());
            for (const auto& entry : entries) {
                columns.push_back(entry.first);
                values.push_back(entry.second);
            }
            if (HYPRE_IJMatrixSetValues(ij_matrix_, 1, &count, &row,
                                        columns.data(), values.data()) != 0) {
                clear_objects();
                return MATX_ERR_INTERNAL;
            }
        }
        if (HYPRE_IJMatrixAssemble(ij_matrix_) != 0 ||
            HYPRE_IJMatrixGetObject(ij_matrix_, reinterpret_cast<void**>(&par_matrix_)) != 0) {
            clear_objects();
            return MATX_ERR_INTERNAL;
        }
        rows_ = matrix.rows;
        return create_solver();
    }

    matx_status_t setup_complex(const CooComplex& matrix) override
    {
        (void)matrix;
        return MATX_ERR_NOT_SUPPORTED;
    }

    matx_status_t solve_real(const std::vector<double>& b, std::vector<double>& x,
                             bool use_initial_guess, BackendResult& result) override
    {
        if (!solver_ || rows_ != static_cast<std::int64_t>(b.size())) {
            return MATX_ERR_INVALID_ARG;
        }
        HYPRE_IJVector ij_b = nullptr;
        HYPRE_IJVector ij_x = nullptr;
        HYPRE_ParVector par_b = nullptr;
        HYPRE_ParVector par_x = nullptr;
        const auto lower = static_cast<HYPRE_BigInt>(0);
        const auto upper = static_cast<HYPRE_BigInt>(rows_ - 1);
        std::vector<HYPRE_BigInt> indices(b.size());
        std::vector<HYPRE_Complex> rhs_values(b.size());
        std::vector<HYPRE_Complex> x_values(b.size());
        std::vector<HYPRE_Complex> solved(b.size());
        for (std::size_t i = 0; i < b.size(); ++i) {
            indices[i] = static_cast<HYPRE_BigInt>(i);
            rhs_values[i] = b[i];
            x_values[i] = use_initial_guess ? x[i] : 0.0;
        }
        if (HYPRE_IJVectorCreate(hypre_comm_self(), lower, upper, &ij_b) != 0 ||
            HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR) != 0 ||
            HYPRE_IJVectorInitialize(ij_b) != 0 ||
            HYPRE_IJVectorCreate(hypre_comm_self(), lower, upper, &ij_x) != 0 ||
            HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR) != 0 ||
            HYPRE_IJVectorInitialize(ij_x) != 0) {
            destroy_vector(&ij_x);
            destroy_vector(&ij_b);
            return MATX_ERR_INTERNAL;
        }
        const auto count = static_cast<HYPRE_Int>(b.size());
        if (HYPRE_IJVectorSetValues(ij_b, count, indices.data(), rhs_values.data()) != 0 ||
            HYPRE_IJVectorSetValues(ij_x, count, indices.data(), x_values.data()) != 0 ||
            HYPRE_IJVectorAssemble(ij_b) != 0 || HYPRE_IJVectorAssemble(ij_x) != 0 ||
            HYPRE_IJVectorGetObject(ij_b, reinterpret_cast<void**>(&par_b)) != 0 ||
            HYPRE_IJVectorGetObject(ij_x, reinterpret_cast<void**>(&par_x)) != 0) {
            destroy_vector(&ij_x);
            destroy_vector(&ij_b);
            return MATX_ERR_INTERNAL;
        }
        if (!solver_setup_) {
            if (setup_solver(par_b, par_x) != 0) {
                destroy_vector(&ij_x);
                destroy_vector(&ij_b);
                return MATX_ERR_INTERNAL;
            }
            solver_setup_ = true;
        }
        const HYPRE_Int solve_error = solve_solver(par_b, par_x);
        const bool did_not_converge = solve_error != 0 &&
            HYPRE_CheckError(solve_error, HYPRE_ERROR_CONV) != 0;
        if (solve_error != 0 && !did_not_converge) {
            destroy_vector(&ij_x);
            destroy_vector(&ij_b);
            return MATX_ERR_INTERNAL;
        }
        if (did_not_converge) HYPRE_ClearError(HYPRE_ERROR_CONV);
        if (HYPRE_IJVectorGetValues(ij_x, count, indices.data(), solved.data()) != 0) {
            destroy_vector(&ij_x);
            destroy_vector(&ij_b);
            return MATX_ERR_INTERNAL;
        }
        for (std::size_t i = 0; i < x.size(); ++i) x[i] = static_cast<double>(solved[i]);
        HYPRE_Int iterations = 0;
        if (get_iterations(&iterations) != 0) {
            destroy_vector(&ij_x);
            destroy_vector(&ij_b);
            return MATX_ERR_INTERNAL;
        }
        result.iterations = iterations;
        result.reason = did_not_converge
            ? (iterations >= options_.max_iterations
                   ? MATX_ITERATIVE_REASON_MAX_ITERATIONS
                   : MATX_ITERATIVE_REASON_BREAKDOWN)
            : (iterations >= options_.max_iterations
                   ? MATX_ITERATIVE_REASON_MAX_ITERATIONS
                   : MATX_ITERATIVE_REASON_CONVERGED);
        destroy_vector(&ij_x);
        destroy_vector(&ij_b);
        return MATX_OK;
    }

    matx_status_t solve_complex(const std::vector<std::complex<double>>&,
                                std::vector<std::complex<double>>&,
                                bool, BackendResult&) override
    {
        return MATX_ERR_NOT_SUPPORTED;
    }

private:
    void destroy_vector(HYPRE_IJVector* vector)
    {
        if (*vector) HYPRE_IJVectorDestroy(*vector);
        *vector = nullptr;
    }

    void clear_objects()
    {
        solver_setup_ = false;
        if (solver_) {
            switch (options_.method) {
            case MATX_ITERATIVE_CG: HYPRE_ParCSRPCGDestroy(solver_); break;
            case MATX_ITERATIVE_BICGSTAB: HYPRE_ParCSRBiCGSTABDestroy(solver_); break;
            case MATX_ITERATIVE_GMRES: HYPRE_ParCSRGMRESDestroy(solver_); break;
            default: break;
            }
            solver_ = nullptr;
        }
        if (amg_) HYPRE_BoomerAMGDestroy(amg_);
        amg_ = nullptr;
        if (ij_matrix_) HYPRE_IJMatrixDestroy(ij_matrix_);
        ij_matrix_ = nullptr;
        par_matrix_ = nullptr;
        rows_ = 0;
    }

    void clear()
    {
        clear_objects();
    }

    matx_status_t create_solver()
    {
        HYPRE_Int code = 0;
        switch (options_.method) {
        case MATX_ITERATIVE_CG: code = HYPRE_ParCSRPCGCreate(hypre_comm_self(), &solver_); break;
        case MATX_ITERATIVE_BICGSTAB: code = HYPRE_ParCSRBiCGSTABCreate(hypre_comm_self(), &solver_); break;
        case MATX_ITERATIVE_GMRES: code = HYPRE_ParCSRGMRESCreate(hypre_comm_self(), &solver_); break;
        default: return MATX_ERR_NOT_SUPPORTED;
        }
        if (code != 0) return MATX_ERR_INTERNAL;
        code = set_tolerances();
        if (code == 0) code = set_preconditioner();
        if (code != 0) {
            clear_objects();
            return MATX_ERR_INTERNAL;
        }
        return MATX_OK;
    }

    HYPRE_Int set_tolerances()
    {
        const HYPRE_Real rel = options_.relative_tolerance;
        const HYPRE_Real abs = options_.absolute_tolerance;
        const HYPRE_Int maxit = static_cast<HYPRE_Int>(options_.max_iterations);
        switch (options_.method) {
        case MATX_ITERATIVE_CG:
            if (HYPRE_ParCSRPCGSetTol(solver_, rel) != 0 ||
                HYPRE_ParCSRPCGSetAbsoluteTol(solver_, abs) != 0 ||
                HYPRE_ParCSRPCGSetMaxIter(solver_, maxit) != 0 ||
                HYPRE_ParCSRPCGSetLogging(solver_, 1) != 0) return -1;
            return HYPRE_ParCSRPCGSetPrintLevel(solver_, 0);
        case MATX_ITERATIVE_BICGSTAB:
            if (HYPRE_ParCSRBiCGSTABSetTol(solver_, rel) != 0 ||
                HYPRE_ParCSRBiCGSTABSetAbsoluteTol(solver_, abs) != 0 ||
                HYPRE_ParCSRBiCGSTABSetMaxIter(solver_, maxit) != 0 ||
                HYPRE_ParCSRBiCGSTABSetLogging(solver_, 1) != 0) return -1;
            return HYPRE_ParCSRBiCGSTABSetPrintLevel(solver_, 0);
        case MATX_ITERATIVE_GMRES:
            if (HYPRE_ParCSRGMRESSetTol(solver_, rel) != 0 ||
                HYPRE_ParCSRGMRESSetAbsoluteTol(solver_, abs) != 0 ||
                HYPRE_ParCSRGMRESSetMaxIter(solver_, maxit) != 0 ||
                HYPRE_ParCSRGMRESSetKDim(solver_,
                    static_cast<HYPRE_Int>(options_.gmres_restart)) != 0 ||
                HYPRE_ParCSRGMRESSetLogging(solver_, 1) != 0) return -1;
            return HYPRE_ParCSRGMRESSetPrintLevel(solver_, 0);
        default: return -1;
        }
    }

    HYPRE_Int set_preconditioner()
    {
        if (options_.preconditioner == MATX_ITERATIVE_PRECOND_NONE) return 0;
        HYPRE_PtrToParSolverFcn solve = nullptr;
        HYPRE_PtrToParSolverFcn setup = nullptr;
        HYPRE_Solver context = nullptr;
        if (options_.preconditioner == MATX_ITERATIVE_PRECOND_JACOBI) {
            solve = reinterpret_cast<HYPRE_PtrToParSolverFcn>(HYPRE_ParCSRDiagScale);
            setup = reinterpret_cast<HYPRE_PtrToParSolverFcn>(HYPRE_ParCSRDiagScaleSetup);
        } else if (options_.preconditioner == MATX_ITERATIVE_PRECOND_AMG) {
            if (HYPRE_BoomerAMGCreate(&amg_) != 0) return -1;
            if (HYPRE_BoomerAMGSetMaxIter(amg_, 1) != 0 ||
                HYPRE_BoomerAMGSetTol(amg_, 0.0) != 0 ||
                HYPRE_BoomerAMGSetPrintLevel(amg_, 0) != 0) return -1;
            solve = reinterpret_cast<HYPRE_PtrToParSolverFcn>(HYPRE_BoomerAMGSolve);
            setup = reinterpret_cast<HYPRE_PtrToParSolverFcn>(HYPRE_BoomerAMGSetup);
            context = amg_;
        } else {
            return -1;
        }
        switch (options_.method) {
        case MATX_ITERATIVE_CG: return HYPRE_ParCSRPCGSetPrecond(solver_, solve, setup, context);
        case MATX_ITERATIVE_BICGSTAB: return HYPRE_ParCSRBiCGSTABSetPrecond(solver_, solve, setup, context);
        case MATX_ITERATIVE_GMRES: return HYPRE_ParCSRGMRESSetPrecond(solver_, solve, setup, context);
        default: return -1;
        }
    }

    HYPRE_Int setup_solver(HYPRE_ParVector b, HYPRE_ParVector x)
    {
        switch (options_.method) {
        case MATX_ITERATIVE_CG: return HYPRE_ParCSRPCGSetup(solver_, par_matrix_, b, x);
        case MATX_ITERATIVE_BICGSTAB: return HYPRE_ParCSRBiCGSTABSetup(solver_, par_matrix_, b, x);
        case MATX_ITERATIVE_GMRES: return HYPRE_ParCSRGMRESSetup(solver_, par_matrix_, b, x);
        default: return -1;
        }
    }

    HYPRE_Int solve_solver(HYPRE_ParVector b, HYPRE_ParVector x)
    {
        switch (options_.method) {
        case MATX_ITERATIVE_CG: return HYPRE_ParCSRPCGSolve(solver_, par_matrix_, b, x);
        case MATX_ITERATIVE_BICGSTAB: return HYPRE_ParCSRBiCGSTABSolve(solver_, par_matrix_, b, x);
        case MATX_ITERATIVE_GMRES: return HYPRE_ParCSRGMRESSolve(solver_, par_matrix_, b, x);
        default: return -1;
        }
    }

    HYPRE_Int get_iterations(HYPRE_Int* iterations)
    {
        switch (options_.method) {
        case MATX_ITERATIVE_CG: return HYPRE_ParCSRPCGGetNumIterations(solver_, iterations);
        case MATX_ITERATIVE_BICGSTAB: return HYPRE_ParCSRBiCGSTABGetNumIterations(solver_, iterations);
        case MATX_ITERATIVE_GMRES: return HYPRE_ParCSRGMRESGetNumIterations(solver_, iterations);
        default: return -1;
        }
    }

    matx_sparse_iterative_options_t options_{};
    bool runtime_ok_ = false;
    HYPRE_IJMatrix ij_matrix_ = nullptr;
    HYPRE_ParCSRMatrix par_matrix_ = nullptr;
    HYPRE_Solver solver_ = nullptr;
    HYPRE_Solver amg_ = nullptr;
    std::int64_t rows_ = 0;
    bool solver_setup_ = false;
};

} // namespace

std::unique_ptr<Backend> make_hypre_backend(
    const matx_sparse_iterative_options_t& options)
{
    return std::make_unique<HypreBackend>(options);
}

matx_status_t finalize_hypre_runtime()
{
    std::lock_guard<std::mutex> lock(hypre_mutex);
    if (!hypre_owned) return MATX_OK;
#if defined(HYPRE_RELEASE_NUMBER) && HYPRE_RELEASE_NUMBER >= 22900
    if (!HYPRE_Finalized() && HYPRE_Finalize() != 0) return MATX_ERR_INTERNAL;
    hypre_owned = false;
    hypre_initialization_succeeded = false;
    return MATX_OK;
#else
    // Older releases do not expose a supported public finalize lifecycle.
    return MATX_ERR_NOT_SUPPORTED;
#endif
}

matx_status_t finalize_hypre_mpi_runtime()
{
#if !defined(MATX_HYPRE_SEQUENTIAL_BUILD)
    std::lock_guard<std::mutex> lock(hypre_mutex);
    if (!mpi_owned) return MATX_OK;
    int finalized = 0;
    if (MPI_Finalized(&finalized) != MPI_SUCCESS) return MATX_ERR_INTERNAL;
    if (!finalized && MPI_Finalize() != MPI_SUCCESS) return MATX_ERR_INTERNAL;
    mpi_owned = false;
    return MATX_OK;
#else
    return MATX_OK;
#endif
}

} // namespace matx::sparse_iterative
#else
namespace matx::sparse_iterative {
std::unique_ptr<Backend> make_hypre_backend(
    const matx_sparse_iterative_options_t&) { return {}; }
matx_status_t finalize_hypre_runtime() { return MATX_OK; }
matx_status_t finalize_hypre_mpi_runtime() { return MATX_OK; }
}
#endif
