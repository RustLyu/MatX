#ifndef MATX_SPARSE_ITERATIVE_INTERNAL_HPP
#define MATX_SPARSE_ITERATIVE_INTERNAL_HPP

#include "matx/matx_sparse_iterative.h"
#include "matx/matx_types_internal.h"

#include <complex>
#include <cstdint>
#include <memory>
#include <vector>

namespace matx::sparse_iterative {

struct CooReal {
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    std::vector<std::int64_t> row;
    std::vector<std::int64_t> col;
    std::vector<double> value;
};

struct CooComplex {
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    std::vector<std::int64_t> row;
    std::vector<std::int64_t> col;
    std::vector<std::complex<double>> value;
};

struct BackendResult {
    std::int64_t iterations = 0;
    matx_sparse_iterative_reason_t reason = MATX_ITERATIVE_REASON_MAX_ITERATIONS;
};

class Backend {
public:
    virtual ~Backend() = default;
    virtual matx_status_t setup_real(const CooReal&) = 0;
    virtual matx_status_t setup_complex(const CooComplex&) = 0;
    virtual matx_status_t solve_real(const std::vector<double>& b,
                                     std::vector<double>& x,
                                     bool use_initial_guess,
                                     BackendResult& result) = 0;
    virtual matx_status_t solve_complex(const std::vector<std::complex<double>>& b,
                                        std::vector<std::complex<double>>& x,
                                        bool use_initial_guess,
                                        BackendResult& result) = 0;
};

std::unique_ptr<Backend> make_ginkgo_backend(
    const matx_sparse_iterative_options_t& options);
std::unique_ptr<Backend> make_petsc_backend(
    const matx_sparse_iterative_options_t& options);
std::unique_ptr<Backend> make_hypre_backend(
    const matx_sparse_iterative_options_t& options);

matx_status_t finalize_petsc_runtime();
matx_status_t finalize_hypre_runtime();
matx_status_t finalize_hypre_mpi_runtime();

} // namespace matx::sparse_iterative

#endif
