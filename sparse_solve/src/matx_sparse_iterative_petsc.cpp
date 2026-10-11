#include "matx_sparse_iterative_internal.hpp"

#ifndef MATX_HAVE_PETSC
#define MATX_HAVE_PETSC 0
#endif
#ifndef MATX_HAVE_PETSC_HYPRE
#define MATX_HAVE_PETSC_HYPRE 0
#endif

#if MATX_HAVE_PETSC
#include <petscksp.h>
#if MATX_HAVE_PETSC_HYPRE
#include <petscpchypre.h>
#endif

#include <limits>
#include <mutex>

namespace matx::sparse_iterative {
namespace {

std::mutex petsc_mutex;
bool petsc_initialization_attempted = false;
bool petsc_initialization_succeeded = false;
bool petsc_owned = false;

bool acquire_petsc()
{
    std::lock_guard<std::mutex> lock(petsc_mutex);
    if (!petsc_initialization_attempted) {
        petsc_initialization_attempted = true;
        PetscBool initialized = PETSC_FALSE;
        if (PetscInitialized(&initialized) != 0) return false;
        if (initialized) {
            PetscBool finalized = PETSC_FALSE;
            if (PetscFinalized(&finalized) != 0 || finalized) return false;
            petsc_initialization_succeeded = true;
        } else {
            petsc_initialization_succeeded = PetscInitializeNoArguments() == 0;
            petsc_owned = petsc_initialization_succeeded;
        }
    }
    return petsc_initialization_succeeded;
}

matx_status_t finalize_petsc_runtime_locked()
{
    if (!petsc_owned) return MATX_OK;
    PetscBool finalized = PETSC_FALSE;
    PetscErrorCode ierr = PetscFinalized(&finalized);
    if (!ierr && !finalized) ierr = PetscFinalize();
    if (ierr) return MATX_ERR_INTERNAL;
    petsc_owned = false;
    petsc_initialization_succeeded = false;
    return MATX_OK;
}

PetscErrorCode to_petsc_scalar(double real, double imag, PetscScalar* out)
{
#if defined(PETSC_USE_COMPLEX)
    *out = PetscCMPLX(real, imag);
    return 0;
#else
    if (imag != 0.0) return PETSC_ERR_SUP;
    *out = real;
    return 0;
#endif
}

class PetscBackend final : public Backend {
public:
    explicit PetscBackend(const matx_sparse_iterative_options_t& options)
        : options_(options), runtime_ok_(acquire_petsc()) {}

    ~PetscBackend() override
    {
        clear();
    }

    matx_status_t setup_real(const CooReal& matrix) override
    {
        if (!runtime_ok_) return MATX_ERR_INTERNAL;
        if (!valid_dimensions(matrix.rows, matrix.value.size())) {
            return MATX_ERR_NOT_SUPPORTED;
        }
        if (options_.preconditioner == MATX_ITERATIVE_PRECOND_AMG &&
            !MATX_HAVE_PETSC_HYPRE) return MATX_ERR_NOT_SUPPORTED;
        if ((options_.method == MATX_ITERATIVE_CG ||
             options_.method == MATX_ITERATIVE_MINRES) &&
            options_.preconditioner == MATX_ITERATIVE_PRECOND_ILU) {
            return MATX_ERR_NOT_SUPPORTED;
        }
        clear_objects();
        std::vector<PetscInt> row_nnz(static_cast<std::size_t>(matrix.rows), 0);
        for (const auto row : matrix.row) {
            ++row_nnz[static_cast<std::size_t>(row)];
        }
        PetscErrorCode ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,
            static_cast<PetscInt>(matrix.rows), static_cast<PetscInt>(matrix.cols),
            0, row_nnz.data(), &matrix_);
        if (ierr) return MATX_ERR_INTERNAL;
        for (std::size_t k = 0; k < matrix.value.size(); ++k) {
            ierr = MatSetValue(matrix_, static_cast<PetscInt>(matrix.row[k]),
                               static_cast<PetscInt>(matrix.col[k]),
                               static_cast<PetscScalar>(matrix.value[k]), ADD_VALUES);
            if (ierr) return MATX_ERR_INTERNAL;
        }
        return finish_setup(false);
    }

    matx_status_t setup_complex(const CooComplex& matrix) override
    {
#if !defined(PETSC_USE_COMPLEX)
        (void)matrix;
        return MATX_ERR_NOT_SUPPORTED;
#else
        if (!runtime_ok_) return MATX_ERR_INTERNAL;
        if (!valid_dimensions(matrix.rows, matrix.value.size())) {
            return MATX_ERR_NOT_SUPPORTED;
        }
        if (options_.preconditioner == MATX_ITERATIVE_PRECOND_AMG &&
            !MATX_HAVE_PETSC_HYPRE) return MATX_ERR_NOT_SUPPORTED;
        if ((options_.method == MATX_ITERATIVE_CG ||
             options_.method == MATX_ITERATIVE_MINRES) &&
            options_.preconditioner == MATX_ITERATIVE_PRECOND_ILU) {
            return MATX_ERR_NOT_SUPPORTED;
        }
        clear_objects();
        std::vector<PetscInt> row_nnz(static_cast<std::size_t>(matrix.rows), 0);
        for (const auto row : matrix.row) {
            ++row_nnz[static_cast<std::size_t>(row)];
        }
        PetscErrorCode ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,
            static_cast<PetscInt>(matrix.rows), static_cast<PetscInt>(matrix.cols),
            0, row_nnz.data(), &matrix_);
        if (ierr) return MATX_ERR_INTERNAL;
        for (std::size_t k = 0; k < matrix.value.size(); ++k) {
            PetscScalar value;
            if (to_petsc_scalar(matrix.value[k].real(), matrix.value[k].imag(), &value)) {
                return MATX_ERR_NOT_SUPPORTED;
            }
            ierr = MatSetValue(matrix_, static_cast<PetscInt>(matrix.row[k]),
                               static_cast<PetscInt>(matrix.col[k]), value, ADD_VALUES);
            if (ierr) return MATX_ERR_INTERNAL;
        }
        return finish_setup(true);
#endif
    }

    matx_status_t solve_real(const std::vector<double>& b, std::vector<double>& x,
                             bool use_initial_guess, BackendResult& result) override
    {
        return solve<double>(b, x, use_initial_guess, result);
    }

    matx_status_t solve_complex(const std::vector<std::complex<double>>& b,
                                std::vector<std::complex<double>>& x,
                                bool use_initial_guess, BackendResult& result) override
    {
#if !defined(PETSC_USE_COMPLEX)
        (void)b; (void)x; (void)use_initial_guess; (void)result;
        return MATX_ERR_NOT_SUPPORTED;
#else
        return solve<std::complex<double>>(b, x, use_initial_guess, result);
#endif
    }

private:
    bool valid_dimensions(std::int64_t rows, std::size_t nnz) const
    {
        return sizeof(PetscReal) == sizeof(double) && rows > 0 &&
               rows <= std::numeric_limits<PetscInt>::max() &&
               nnz <= static_cast<std::size_t>(std::numeric_limits<PetscInt>::max()) &&
               options_.max_iterations <= std::numeric_limits<PetscInt>::max() &&
               options_.gmres_restart <= std::numeric_limits<PetscInt>::max();
    }

    void clear_objects()
    {
        if (ksp_) KSPDestroy(&ksp_);
        if (matrix_) MatDestroy(&matrix_);
    }

    void clear()
    {
        clear_objects();
        is_complex_ = false;
        rows_ = 0;
    }

    matx_status_t finish_setup(bool complex)
    {
        PetscErrorCode ierr = MatAssemblyBegin(matrix_, MAT_FINAL_ASSEMBLY);
        if (!ierr) ierr = MatAssemblyEnd(matrix_, MAT_FINAL_ASSEMBLY);
        if (ierr) return MATX_ERR_INTERNAL;
        ierr = KSPCreate(PETSC_COMM_SELF, &ksp_);
        if (ierr) return MATX_ERR_INTERNAL;
        ierr = KSPSetOperators(ksp_, matrix_, matrix_);
        if (!ierr) ierr = KSPSetType(ksp_, ksp_type());
        if (!ierr) ierr = KSPSetNormType(ksp_, KSP_NORM_UNPRECONDITIONED);
        if (!ierr) ierr = KSPSetTolerances(ksp_, options_.relative_tolerance,
            options_.absolute_tolerance, PETSC_DEFAULT,
            static_cast<PetscInt>(options_.max_iterations));
        if (!ierr && options_.method == MATX_ITERATIVE_GMRES) {
            ierr = KSPGMRESSetRestart(ksp_, static_cast<PetscInt>(options_.gmres_restart));
        }
        PC pc = nullptr;
        if (!ierr) ierr = KSPGetPC(ksp_, &pc);
        if (!ierr) ierr = PCSetType(pc, pc_type());
#if MATX_HAVE_PETSC_HYPRE
        if (!ierr && options_.preconditioner == MATX_ITERATIVE_PRECOND_AMG) {
            ierr = PCHYPRESetType(pc, "boomeramg");
        }
#endif
        if (!ierr) ierr = KSPSetUp(ksp_);
        if (ierr) {
            clear_objects();
            return options_.preconditioner == MATX_ITERATIVE_PRECOND_AMG
                ? MATX_ERR_NOT_SUPPORTED : MATX_ERR_INTERNAL;
        }
        PetscInt n = 0;
        ierr = MatGetSize(matrix_, &n, nullptr);
        if (ierr) return MATX_ERR_INTERNAL;
        rows_ = static_cast<std::int64_t>(n);
        is_complex_ = complex;
        return MATX_OK;
    }

    KSPType ksp_type() const
    {
        switch (options_.method) {
        case MATX_ITERATIVE_CG: return KSPCG;
        case MATX_ITERATIVE_BICGSTAB: return KSPBCGS;
        case MATX_ITERATIVE_GMRES: return KSPGMRES;
        case MATX_ITERATIVE_MINRES: return KSPMINRES;
        default: return KSPGMRES;
        }
    }

    PCType pc_type() const
    {
        switch (options_.preconditioner) {
        case MATX_ITERATIVE_PRECOND_NONE: return PCNONE;
        case MATX_ITERATIVE_PRECOND_JACOBI: return PCJACOBI;
        case MATX_ITERATIVE_PRECOND_ILU: return PCILU;
        case MATX_ITERATIVE_PRECOND_AMG:
#if MATX_HAVE_PETSC_HYPRE
            return PCHYPRE;
#else
            return PCNONE;
#endif
        default: return PCNONE;
        }
    }

    template <typename Value>
    matx_status_t solve(const std::vector<Value>& b, std::vector<Value>& x,
                        bool use_initial_guess, BackendResult& result)
    {
        if (!ksp_ || is_complex_ !=
                std::is_same_v<Value, std::complex<double>> ||
            b.size() != static_cast<std::size_t>(rows_)) {
            return MATX_ERR_INVALID_ARG;
        }
        Vec rhs = nullptr;
        Vec solution = nullptr;
        PetscErrorCode ierr = VecCreateSeq(PETSC_COMM_SELF,
            static_cast<PetscInt>(rows_), &rhs);
        if (!ierr) ierr = VecDuplicate(rhs, &solution);
        if (ierr) {
            if (rhs) VecDestroy(&rhs);
            if (solution) VecDestroy(&solution);
            return MATX_ERR_INTERNAL;
        }
        for (PetscInt i = 0; i < static_cast<PetscInt>(rows_); ++i) {
            PetscScalar bv, xv;
            if constexpr (std::is_same_v<Value, std::complex<double>>) {
                if (to_petsc_scalar(b[static_cast<std::size_t>(i)].real(),
                                    b[static_cast<std::size_t>(i)].imag(), &bv) ||
                    to_petsc_scalar(use_initial_guess ? x[static_cast<std::size_t>(i)].real() : 0.0,
                                    use_initial_guess ? x[static_cast<std::size_t>(i)].imag() : 0.0,
                                    &xv)) {
                    VecDestroy(&solution); VecDestroy(&rhs);
                    return MATX_ERR_NOT_SUPPORTED;
                }
            } else {
                bv = b[static_cast<std::size_t>(i)];
                xv = use_initial_guess ? x[static_cast<std::size_t>(i)] : 0.0;
            }
            ierr = VecSetValue(rhs, i, bv, INSERT_VALUES);
            if (!ierr) ierr = VecSetValue(solution, i, xv, INSERT_VALUES);
            if (ierr) break;
        }
        if (!ierr) ierr = VecAssemblyBegin(rhs);
        if (!ierr) ierr = VecAssemblyEnd(rhs);
        if (!ierr) ierr = VecAssemblyBegin(solution);
        if (!ierr) ierr = VecAssemblyEnd(solution);
        if (!ierr) ierr = KSPSetInitialGuessNonzero(ksp_, use_initial_guess ? PETSC_TRUE : PETSC_FALSE);
        if (!ierr) ierr = KSPSolve(ksp_, rhs, solution);
        if (!ierr) {
            const PetscScalar* values = nullptr;
            ierr = VecGetArrayRead(solution, &values);
            if (!ierr) {
                for (PetscInt i = 0; i < static_cast<PetscInt>(rows_); ++i) {
                    if constexpr (std::is_same_v<Value, std::complex<double>>) {
                        x[static_cast<std::size_t>(i)] = {
                            static_cast<double>(PetscRealPart(values[i])),
                            static_cast<double>(PetscImaginaryPart(values[i]))};
                    } else {
                        x[static_cast<std::size_t>(i)] =
                            static_cast<double>(PetscRealPart(values[i]));
                    }
                }
                VecRestoreArrayRead(solution, &values);
            }
        }
        if (!ierr) {
            PetscInt iterations = 0;
            KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
            ierr = KSPGetIterationNumber(ksp_, &iterations);
            if (!ierr) ierr = KSPGetConvergedReason(ksp_, &reason);
            result.iterations = iterations;
            result.reason = reason > 0 ? MATX_ITERATIVE_REASON_CONVERGED :
                (reason == KSP_DIVERGED_ITS ? MATX_ITERATIVE_REASON_MAX_ITERATIONS :
                 (reason < 0 ? MATX_ITERATIVE_REASON_BREAKDOWN :
                               MATX_ITERATIVE_REASON_MAX_ITERATIONS));
        }
        VecDestroy(&solution);
        VecDestroy(&rhs);
        return ierr ? MATX_ERR_INTERNAL : MATX_OK;
    }

    matx_sparse_iterative_options_t options_{};
    bool runtime_ok_ = false;
    Mat matrix_ = nullptr;
    KSP ksp_ = nullptr;
    std::int64_t rows_ = 0;
    bool is_complex_ = false;
};

} // namespace

std::unique_ptr<Backend> make_petsc_backend(
    const matx_sparse_iterative_options_t& options)
{
    auto backend = std::make_unique<PetscBackend>(options);
    return backend;
}

matx_status_t finalize_petsc_runtime()
{
    std::lock_guard<std::mutex> lock(petsc_mutex);
    return finalize_petsc_runtime_locked();
}

} // namespace matx::sparse_iterative
#else
namespace matx::sparse_iterative {
std::unique_ptr<Backend> make_petsc_backend(
    const matx_sparse_iterative_options_t&) { return {}; }
matx_status_t finalize_petsc_runtime() { return MATX_OK; }
}
#endif
