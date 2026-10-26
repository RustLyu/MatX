#include "matx_sparse_iterative_internal.hpp"

#ifndef MATX_HAVE_GINKGO
#define MATX_HAVE_GINKGO 0
#endif

#if MATX_HAVE_GINKGO
#include <ginkgo/ginkgo.hpp>

#include <limits>
#include <type_traits>

namespace matx::sparse_iterative {
namespace {

using Index = std::int32_t;

class GinkgoBackend final : public Backend {
public:
    explicit GinkgoBackend(const matx_sparse_iterative_options_t& options)
        : options_(options)
    {
        auto reference = gko::ReferenceExecutor::create();
        if (options.executor == MATX_ITERATIVE_EXECUTOR_REFERENCE) {
            executor_ = reference;
        } else {
            try {
                executor_ = gko::OmpExecutor::create();
            } catch (...) {
                if (options.executor != MATX_ITERATIVE_EXECUTOR_AUTO) throw;
                executor_ = reference;
            }
        }
    }

    matx_status_t setup_real(const CooReal& matrix) override
    {
        return setup<double>(matrix.rows, matrix.cols, matrix.row, matrix.col,
                             matrix.value);
    }

    matx_status_t setup_complex(const CooComplex& matrix) override
    {
        return setup<std::complex<double>>(matrix.rows, matrix.cols, matrix.row,
                                           matrix.col, matrix.value);
    }

    matx_status_t solve_real(const std::vector<double>& b,
                             std::vector<double>& x,
                             bool use_initial_guess,
                             BackendResult& result) override
    {
        return solve<double>(b, x, use_initial_guess, result);
    }

    matx_status_t solve_complex(const std::vector<std::complex<double>>& b,
                                std::vector<std::complex<double>>& x,
                                bool use_initial_guess,
                                BackendResult& result) override
    {
        return solve<std::complex<double>>(b, x, use_initial_guess, result);
    }

private:
    template <typename Value>
    matx_status_t setup(std::int64_t rows, std::int64_t cols,
                        const std::vector<std::int64_t>& row,
                        const std::vector<std::int64_t>& col,
                        const std::vector<Value>& values)
    {
        if (rows > std::numeric_limits<Index>::max() ||
            cols > std::numeric_limits<Index>::max() ||
            values.size() > static_cast<std::size_t>(std::numeric_limits<Index>::max()) ||
            options_.preconditioner == MATX_ITERATIVE_PRECOND_AMG) {
            return MATX_ERR_NOT_SUPPORTED;
        }
        if (options_.preconditioner == MATX_ITERATIVE_PRECOND_ILU &&
            options_.method != MATX_ITERATIVE_GMRES &&
            options_.method != MATX_ITERATIVE_BICGSTAB) {
            return MATX_ERR_NOT_SUPPORTED;
        }
        try {
            using Csr = gko::matrix::Csr<Value, Index>;
            gko::matrix_data<Value, Index> data(
                gko::dim<2>{static_cast<gko::size_type>(rows),
                            static_cast<gko::size_type>(cols)});
            data.nonzeros.reserve(values.size());
            for (std::size_t k = 0; k < values.size(); ++k) {
                data.nonzeros.push_back({static_cast<Index>(row[k]),
                                         static_cast<Index>(col[k]), values[k]});
            }
            auto csr = gko::share(Csr::create(executor_));
            csr->read(std::move(data));
            matrix_ = csr;

            const auto max_iters = static_cast<gko::size_type>(options_.max_iterations);
            switch (options_.method) {
            case MATX_ITERATIVE_CG:
                return install_solver<Value>(
                    gko::solver::Cg<Value>::build()
                        .with_criteria(
                            gko::stop::Iteration::build().with_max_iters(max_iters),
                            gko::stop::ResidualNorm<Value>::build()
                                .with_baseline(gko::stop::mode::rhs_norm)
                                .with_reduction_factor(options_.relative_tolerance),
                            gko::stop::ResidualNorm<Value>::build()
                                .with_baseline(gko::stop::mode::absolute)
                                .with_reduction_factor(options_.absolute_tolerance))
                        .on(executor_), csr, rows);
            case MATX_ITERATIVE_BICGSTAB:
                return install_solver<Value>(
                    gko::solver::Bicgstab<Value>::build()
                        .with_criteria(
                            gko::stop::Iteration::build().with_max_iters(max_iters),
                            gko::stop::ResidualNorm<Value>::build()
                                .with_baseline(gko::stop::mode::rhs_norm)
                                .with_reduction_factor(options_.relative_tolerance),
                            gko::stop::ResidualNorm<Value>::build()
                                .with_baseline(gko::stop::mode::absolute)
                                .with_reduction_factor(options_.absolute_tolerance))
                        .on(executor_), csr, rows);
            case MATX_ITERATIVE_GMRES:
                return install_solver<Value>(
                    gko::solver::Gmres<Value>::build()
                        .with_krylov_dim(static_cast<gko::size_type>(options_.gmres_restart))
                        .with_criteria(
                            gko::stop::Iteration::build().with_max_iters(max_iters),
                            gko::stop::ResidualNorm<Value>::build()
                                .with_baseline(gko::stop::mode::rhs_norm)
                                .with_reduction_factor(options_.relative_tolerance),
                            gko::stop::ResidualNorm<Value>::build()
                                .with_baseline(gko::stop::mode::absolute)
                                .with_reduction_factor(options_.absolute_tolerance))
                        .on(executor_), csr, rows);
            case MATX_ITERATIVE_MINRES:
                return install_solver<Value>(
                    gko::solver::Minres<Value>::build()
                        .with_criteria(
                            gko::stop::Iteration::build().with_max_iters(max_iters),
                            gko::stop::ResidualNorm<Value>::build()
                                .with_baseline(gko::stop::mode::rhs_norm)
                                .with_reduction_factor(options_.relative_tolerance),
                            gko::stop::ResidualNorm<Value>::build()
                                .with_baseline(gko::stop::mode::absolute)
                                .with_reduction_factor(options_.absolute_tolerance))
                        .on(executor_), csr, rows);
            default:
                return MATX_ERR_NOT_SUPPORTED;
            }
        } catch (...) {
            return MATX_ERR_INTERNAL;
        }
    }

    template <typename Value, typename Factory>
    matx_status_t install_solver(
        std::unique_ptr<Factory> factory,
        const std::shared_ptr<gko::matrix::Csr<Value, Index>>& csr,
        std::int64_t rows)
    {
        auto generated = factory->generate(csr);
        if (options_.preconditioner == MATX_ITERATIVE_PRECOND_JACOBI) {
            auto preconditioner_factory =
                gko::preconditioner::Jacobi<Value, Index>::build().on(executor_);
            auto preconditioner = preconditioner_factory->generate(csr);
            generated->set_preconditioner(gko::share(std::move(preconditioner)));
        } else if (options_.preconditioner == MATX_ITERATIVE_PRECOND_ILU) {
            using Ilu = gko::preconditioner::Ilu<Value, Value, false, Index>;
            auto preconditioner_factory = Ilu::build().on(executor_);
            auto preconditioner = preconditioner_factory->generate(csr);
            generated->set_preconditioner(gko::share(std::move(preconditioner)));
        } else if (options_.preconditioner != MATX_ITERATIVE_PRECOND_NONE) {
            return MATX_ERR_NOT_SUPPORTED;
        }
        auto logger = gko::share(gko::log::Convergence<Value>::create());
        generated->add_logger(logger);
        solver_ = gko::share(std::move(generated));
        if constexpr (std::is_same_v<Value, double>) {
            real_logger_ = logger;
        } else {
            complex_logger_ = logger;
        }
        is_complex_ = !std::is_same_v<Value, double>;
        rows_ = rows;
        return MATX_OK;
    }

    template <typename Value>
    matx_status_t solve(const std::vector<Value>& b, std::vector<Value>& x,
                        bool use_initial_guess, BackendResult& result)
    {
        if (!solver_ || is_complex_ != !std::is_same_v<Value, double> ||
            b.size() != static_cast<std::size_t>(rows_)) {
            return MATX_ERR_INVALID_ARG;
        }
        try {
            using Dense = gko::matrix::Dense<Value>;
            auto rhs = Dense::create(executor_,
                                     gko::dim<2>{static_cast<gko::size_type>(rows_), 1});
            auto solution = Dense::create(executor_,
                                          gko::dim<2>{static_cast<gko::size_type>(rows_), 1});
            for (std::size_t i = 0; i < b.size(); ++i) {
                rhs->get_values()[i] = b[i];
                solution->get_values()[i] = use_initial_guess ? x[i] : Value{};
            }
            solver_->apply(rhs.get(), solution.get());
            for (std::size_t i = 0; i < x.size(); ++i) {
                x[i] = solution->get_values()[i];
            }
            if constexpr (std::is_same_v<Value, double>) {
                result.iterations = real_logger_ ? real_logger_->get_num_iterations() : 0;
            } else {
                result.iterations = complex_logger_ ? complex_logger_->get_num_iterations() : 0;
            }
            result.reason = result.iterations >= options_.max_iterations
                ? MATX_ITERATIVE_REASON_MAX_ITERATIONS
                : MATX_ITERATIVE_REASON_NOT_RUN;
            return MATX_OK;
        } catch (...) {
            return MATX_ERR_INTERNAL;
        }
    }

    matx_sparse_iterative_options_t options_{};
    std::shared_ptr<const gko::Executor> executor_;
    std::shared_ptr<gko::LinOp> matrix_;
    std::shared_ptr<gko::LinOp> solver_;
    std::shared_ptr<gko::log::Convergence<double>> real_logger_;
    std::shared_ptr<gko::log::Convergence<std::complex<double>>> complex_logger_;
    std::int64_t rows_ = 0;
    bool is_complex_ = false;
};

} // namespace

std::unique_ptr<Backend> make_ginkgo_backend(
    const matx_sparse_iterative_options_t& options)
{
    return std::make_unique<GinkgoBackend>(options);
}

} // namespace matx::sparse_iterative
#else
namespace matx::sparse_iterative {
std::unique_ptr<Backend> make_ginkgo_backend(
    const matx_sparse_iterative_options_t&) { return {}; }
}
#endif
