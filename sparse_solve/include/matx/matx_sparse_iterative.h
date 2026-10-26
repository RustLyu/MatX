#ifndef MATX_SPARSE_ITERATIVE_H
#define MATX_SPARSE_ITERATIVE_H

#include "matx/matx_func.h"
#include "matx/matx_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum matx_sparse_iterative_backend_kind_t {
    MATX_SPARSE_ITERATIVE_BACKEND_GINKGO = 0,
    MATX_SPARSE_ITERATIVE_BACKEND_PETSC = 1,
    MATX_SPARSE_ITERATIVE_BACKEND_HYPRE = 2
} matx_sparse_iterative_backend_kind_t;

typedef enum matx_sparse_iterative_method_t {
    MATX_ITERATIVE_CG = 0,
    MATX_ITERATIVE_BICGSTAB = 1,
    MATX_ITERATIVE_GMRES = 2,
    MATX_ITERATIVE_MINRES = 3
} matx_sparse_iterative_method_t;

typedef enum matx_sparse_iterative_preconditioner_t {
    MATX_ITERATIVE_PRECOND_NONE = 0,
    MATX_ITERATIVE_PRECOND_JACOBI = 1,
    MATX_ITERATIVE_PRECOND_ILU = 2,
    MATX_ITERATIVE_PRECOND_AMG = 3
} matx_sparse_iterative_preconditioner_t;

typedef enum matx_sparse_iterative_executor_t {
    MATX_ITERATIVE_EXECUTOR_AUTO = 0,
    MATX_ITERATIVE_EXECUTOR_REFERENCE = 1,
    MATX_ITERATIVE_EXECUTOR_OMP = 2
} matx_sparse_iterative_executor_t;

typedef enum matx_sparse_iterative_reason_t {
    MATX_ITERATIVE_REASON_NOT_RUN = 0,
    MATX_ITERATIVE_REASON_CONVERGED = 1,
    MATX_ITERATIVE_REASON_MAX_ITERATIONS = 2,
    MATX_ITERATIVE_REASON_BREAKDOWN = 3
} matx_sparse_iterative_reason_t;

typedef struct matx_sparse_iterative_options_t {
    matx_sparse_iterative_method_t method;
    matx_sparse_iterative_preconditioner_t preconditioner;
    matx_sparse_iterative_executor_t executor;
    matx_double relative_tolerance;
    matx_double absolute_tolerance;
    matx_int64_t max_iterations;
    matx_int64_t gmres_restart;
    matx_bool use_initial_guess;
} matx_sparse_iterative_options_t;

typedef struct matx_sparse_iterative_result_t {
    matx_int64_t iterations;
    matx_double residual_norm;
    matx_double relative_residual;
    matx_sparse_iterative_reason_t reason;
} matx_sparse_iterative_result_t;

typedef struct matx_sparse_iterative_solver_opaque_t* matx_sparse_iterative_solver_t;

/* General-purpose starting point: restarted GMRES with Jacobi preconditioning,
 * relative tolerance 1e-8, and up to 1000 iterations. Inspect `reason` after
 * every solve; MATX_OK reports that the operation ran, while `reason` reports
 * whether the requested residual target was reached. */
MATX_SPARSE_SOLVE_API matx_sparse_iterative_options_t
matx_sparse_iterative_options_default(void);

MATX_SPARSE_SOLVE_API const char* matx_sparse_iterative_backend_name(
    matx_sparse_iterative_backend_kind_t backend);

MATX_SPARSE_SOLVE_API matx_status_t matx_sparse_iterative_solver_create(
    const matx_alloc_t* alloc,
    matx_sparse_iterative_backend_kind_t backend,
    const matx_sparse_iterative_options_t* options,
    matx_sparse_iterative_solver_t* out);

/* Setup snapshots the square COO matrix into the selected backend. The solver
 * may be reused for multiple right-hand sides until setup is called again.
 * Unsupported backend/method/precision combinations return NOT_SUPPORTED. */
MATX_SPARSE_SOLVE_API matx_status_t matx_sparse_iterative_setup_d_i8(
    matx_sparse_iterative_solver_t solver,
    matx_coo_d_i8_t A);
MATX_SPARSE_SOLVE_API matx_status_t matx_sparse_iterative_setup_z_i8(
    matx_sparse_iterative_solver_t solver,
    matx_coo_z_i8_t A);

MATX_SPARSE_SOLVE_API matx_status_t matx_sparse_iterative_solve_d_i8(
    matx_sparse_iterative_solver_t solver,
    const matx_vec_d_i8_t b,
    matx_vec_d_i8_t x,
    matx_sparse_iterative_result_t* result);
MATX_SPARSE_SOLVE_API matx_status_t matx_sparse_iterative_solve_z_i8(
    matx_sparse_iterative_solver_t solver,
    const matx_vec_z_i8_t b,
    matx_vec_z_i8_t x,
    matx_sparse_iterative_result_t* result);

MATX_SPARSE_SOLVE_API void matx_sparse_iterative_solver_destroy(
    matx_sparse_iterative_solver_t solver);

/* Call once during process shutdown, after destroying every iterative solver
 * handle. Finalizes only PETSc/HYPRE/MPI runtimes initialized by MatX. */
MATX_SPARSE_SOLVE_API matx_status_t matx_sparse_iterative_runtime_finalize(void);

#ifdef __cplusplus
}
#endif

#endif
