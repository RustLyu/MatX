#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <omp.h>

#if defined(__linux__)
#include <cerrno>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

extern "C" {
#include "matx/matx_dense_compute.h"
#include "matx/matx_dense_solve.h"
#include "matx/matx_func.h"
#include "matx/matx_log.h"
#include "matx/matx_print.h"
#include "matx/matx_read.h"
#include "matx/matx_sparse_compute.h"
#include "matx/matx_sparse_solve.h"
#include "matx/matx_types.h"
#include "matx/matx_types_internal.h"
#include "matx/matx_tm.h"
#include "matx/matx_vec_compute.h"
}

typedef struct {
    matx_int64_t nrows;
    matx_int64_t ncols;
    matx_int64_t nnz;
    matx_int64_t* row_ptr;
    matx_int64_t* col_ind;
    matx_double* val;
} matx_benchmark_csr_matrix_t;

extern "C" int coo_to_csr_optimized(const matx_alloc_t* alloc,
                                     matx_int64_t nrows,
                                     matx_int64_t ncols,
                                     matx_int64_t nnz,
                                     const matx_int64_t* coo_row,
                                     const matx_int64_t* coo_col,
                                     const matx_double* coo_val,
                                     matx_benchmark_csr_matrix_t* csr);

namespace {

using Clock = std::chrono::steady_clock;

bool perf_counters_enabled = false;
uint64_t perf_counter_failures = 0;

struct PerfCounts
{
    double cycles = std::numeric_limits<double>::quiet_NaN();
    double instructions = std::numeric_limits<double>::quiet_NaN();
    double cache_misses = std::numeric_limits<double>::quiet_NaN();
};

class PerfEventGroup
{
public:
    PerfEventGroup()
    {
#if defined(__linux__)
        if (!perf_counters_enabled) return;
        leader_ = open_event(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, -1);
        if (leader_ < 0) {
            ++perf_counter_failures;
            return;
        }
        instructions_ = open_event(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, leader_);
        cache_misses_ = open_event(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES, leader_);
        if (instructions_ < 0 || cache_misses_ < 0) {
            close_all();
            ++perf_counter_failures;
            return;
        }
        available_ = true;
#endif
    }

    ~PerfEventGroup() { close_all(); }
    PerfEventGroup(const PerfEventGroup&) = delete;
    PerfEventGroup& operator=(const PerfEventGroup&) = delete;

    bool available() const { return available_; }

    const char* scope() const
    {
#if defined(__linux__)
        if (!perf_counters_enabled) return "disabled";
        if (!available_) return "unavailable";
        return "calling_thread";
#else
        return perf_counters_enabled ? "unsupported" : "disabled";
#endif
    }

    bool start()
    {
#if defined(__linux__)
        if (!available_) return false;
        if (ioctl(leader_, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0
            || ioctl(leader_, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) != 0) {
            available_ = false;
            ++perf_counter_failures;
            return false;
        }
        running_ = true;
        return true;
#else
        return false;
#endif
    }

    PerfCounts stop(uint64_t measured_calls)
    {
        PerfCounts counts;
#if defined(__linux__)
        if (!running_) return counts;
        running_ = false;
        if (ioctl(leader_, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) != 0) {
            available_ = false;
            ++perf_counter_failures;
            return counts;
        }
        uint64_t values[6] = {};
        const ssize_t bytes = read(leader_, values, sizeof(values));
        if (bytes != (ssize_t) sizeof(values) || values[0] != 3 || values[2] == 0) {
            available_ = false;
            ++perf_counter_failures;
            return counts;
        }
        const double scale = (double) values[1] / (double) values[2];
        const double calls = (double) std::max<uint64_t>(measured_calls, 1);
        counts.cycles = (double) values[3] * scale / calls;
        counts.instructions = (double) values[4] * scale / calls;
        counts.cache_misses = (double) values[5] * scale / calls;
#else
        (void) measured_calls;
#endif
        return counts;
    }

private:
#if defined(__linux__)
    static int open_event(uint32_t type, uint64_t config, int group_fd)
    {
        struct perf_event_attr attr {};
        attr.size = sizeof(attr);
        attr.type = type;
        attr.config = config;
        attr.disabled = group_fd < 0;
        attr.exclude_kernel = 1;
        attr.exclude_hv = 1;
        attr.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_TOTAL_TIME_ENABLED
                           | PERF_FORMAT_TOTAL_TIME_RUNNING;
        return (int) syscall(SYS_perf_event_open, &attr, 0, -1, group_fd, 0);
    }

    int leader_ = -1;
    int instructions_ = -1;
    int cache_misses_ = -1;
#endif

    void close_all()
    {
#if defined(__linux__)
        if (cache_misses_ >= 0) close(cache_misses_);
        if (instructions_ >= 0) close(instructions_);
        if (leader_ >= 0) close(leader_);
        cache_misses_ = instructions_ = leader_ = -1;
#endif
    }

    bool available_ = false;
    bool running_ = false;
};

void check_status(matx_status_t status, const char* operation)
{
    if (status != MATX_OK) {
        throw std::runtime_error(std::string(operation) + " failed with status "
                                 + std::to_string(status));
    }
}

struct Timing
{
    double first_us;
    double steady_us;
    int calls_per_sample;
    PerfCounts perf;
    const char* perf_scope = "unavailable";
};

template <typename Function>
Timing measure(Function&& function)
{
    auto start = Clock::now();
    function();
    const double first_us
        = std::chrono::duration<double, std::micro>(Clock::now() - start).count();

    function();
    function();

    const int calls_per_sample = std::clamp(
        static_cast<int>(10000.0 / std::max(first_us, 0.1)), 1, 100000);
    std::vector<double> samples;
    samples.reserve(7);
    PerfEventGroup counters;
    bool count_perf = counters.available();
    double perf_cycles = 0.0;
    double perf_instructions = 0.0;
    double perf_cache_misses = 0.0;
    for (int sample = 0; sample < 7; ++sample) {
        const bool sample_counted = count_perf && counters.start();
        start = Clock::now();
        for (int call = 0; call < calls_per_sample; ++call) {
            function();
        }
        const double elapsed_us
            = std::chrono::duration<double, std::micro>(Clock::now() - start).count();
        samples.push_back(elapsed_us / calls_per_sample);
        if (sample_counted) {
            const PerfCounts sample_perf = counters.stop(calls_per_sample);
            count_perf = std::isfinite(sample_perf.cycles);
            perf_cycles += sample_perf.cycles;
            perf_instructions += sample_perf.instructions;
            perf_cache_misses += sample_perf.cache_misses;
        }
        else if (count_perf) count_perf = false;
    }
    PerfCounts perf;
    if (count_perf) {
        perf.cycles = perf_cycles / samples.size();
        perf.instructions = perf_instructions / samples.size();
        perf.cache_misses = perf_cache_misses / samples.size();
    }
    std::sort(samples.begin(), samples.end());
    return {first_us, samples[samples.size() / 2], calls_per_sample, perf,
            count_perf ? counters.scope() : "unavailable"};
}

template <typename Reset, typename Function>
Timing measure_reinitialized(Reset&& reset, Function&& function)
{
    reset();
    auto start = Clock::now();
    function();
    const double first_us
        = std::chrono::duration<double, std::micro>(Clock::now() - start).count();
    std::vector<double> samples;
    samples.reserve(7);
    PerfEventGroup counters;
    bool count_perf = counters.available();
    double perf_cycles = 0.0;
    double perf_instructions = 0.0;
    double perf_cache_misses = 0.0;
    for (int sample = 0; sample < 7; ++sample) {
        reset();
        const bool sample_counted = count_perf && counters.start();
        start = Clock::now();
        function();
        const double elapsed
            = std::chrono::duration<double, std::micro>(Clock::now() - start).count();
        samples.push_back(elapsed);
        if (sample_counted) {
            const PerfCounts sample_perf = counters.stop(1);
            count_perf = std::isfinite(sample_perf.cycles);
            perf_cycles += sample_perf.cycles;
            perf_instructions += sample_perf.instructions;
            perf_cache_misses += sample_perf.cache_misses;
        }
        else if (count_perf) count_perf = false;
    }
    PerfCounts perf;
    if (count_perf) {
        perf.cycles = perf_cycles / samples.size();
        perf.instructions = perf_instructions / samples.size();
        perf.cache_misses = perf_cache_misses / samples.size();
    }
    std::sort(samples.begin(), samples.end());
    return {first_us, samples[samples.size() / 2], 1, perf,
            count_perf ? counters.scope() : "unavailable"};
}

double elapsed_us(Clock::time_point start)
{
    return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}

void csv_row(const char* operation,
             const char* type,
             const char* backend,
             matx_int64_t m,
             matx_int64_t n,
             matx_int64_t k,
             matx_int64_t nnz,
             double density,
             matx_int64_t rhs,
             double setup_us,
             const Timing& timing,
             double flops,
             double max_error,
             double bytes_moved = 0.0,
             const char* layout = "N/A")
{
    const double gflops = flops / (timing.steady_us * 1000.0);
    const double gbytes = bytes_moved / (timing.steady_us * 1000.0);
    std::cout << operation << ',' << type << ',' << backend << ',' << m << ',' << n << ',' << k << ','
              << nnz << ',' << density << ',' << rhs << ',' << setup_us << ','
              << timing.first_us << ',' << timing.steady_us << ',' << gflops << ','
              << timing.calls_per_sample << ',' << max_error << ',' << gbytes << ','
              << layout << ',' << timing.perf.cycles << ',' << timing.perf.instructions
              << ',' << timing.perf.cache_misses << ',' << timing.perf_scope << '\n';
}

double seeded_value(uint64_t index, uint64_t salt)
{
    const int64_t centered = static_cast<int64_t>((index * 48271 + salt * 69621) % 2001) - 1000;
    return static_cast<double>(centered) / 1000.0;
}

template <typename T>
struct Ops;

template <>
struct Ops<double>
{
    using Dense = matx_dense_d_i8_t;
    using Sparse = matx_coo_d_i8_t;
    using Vector = matx_vec_d_i8_t;
    using Factor = matx_factor_dense_d_i8_t*;
    using SparseFactor = matx_factor_sparse_d_i8_t;

    static constexpr const char* name = "f64";
    static double make(double real, double = 0.0) { return real; }
    static double add(double a, double b) { return a + b; }
    static double multiply(double a, double b) { return a * b; }
    static double magnitude(double a) { return std::abs(a); }
    static double difference(double a, double b) { return std::abs(a - b); }
    static double one() { return 1.0; }
    static double zero() { return 0.0; }
    static double flop_factor() { return 2.0; }

    static matx_status_t dense_create(const matx_alloc_t* alloc,
                                      Dense* out,
                                      matx_int64_t rows,
                                      matx_int64_t cols,
                                      matx_layout_t layout = MATX_COL_MAJOR)
    {
        return matx_dense_d_i8_create(alloc, out, layout, rows, cols, nullptr);
    }
    static matx_status_t dense_fill(Dense value, double fill_value)
    {
        return matx_dense_d_i8_fill(value, fill_value);
    }
    static void dense_destroy(const matx_alloc_t* alloc, Dense value)
    {
        matx_dense_d_i8_destroy(alloc, value);
    }
    static void dense_write(Dense value, const char* path)
    {
        matx_print_dense_mtx_d_i8(value, path);
    }
    static matx_status_t dense_read(const matx_alloc_t* alloc, Dense* out, const char* path)
    {
        return matx_read_dense_mtx_d_i8(alloc, out, path);
    }
    static matx_status_t gemm(const matx_dense_backend_t* backend,
                              Dense a,
                              Dense b,
                              Dense c)
    {
        return matx_gemm_d_i8(backend, MATX_NO_TRANS, MATX_NO_TRANS, 1.0, a, b, 0.0, c);
    }
    static matx_status_t sparse_create(const matx_alloc_t* alloc,
                                       Sparse* out,
                                       matx_int64_t n,
                                       matx_int64_t nnz,
                                       matx_int64_t* rows,
                                       matx_int64_t* cols,
                                       double* values)
    {
        return matx_coo_sparse_d_i8_create(alloc, out, n, n, nnz, rows, cols, values);
    }
    static void sparse_destroy(const matx_alloc_t* alloc, Sparse value)
    {
        matx_coo_sparse_d_i8_destroy(alloc, value);
    }
    static void sparse_write(Sparse value, const char* path)
    {
        matx_print_sparse_mtx_d_i8(value, path);
    }
    static matx_status_t sparse_read(const matx_alloc_t* alloc, Sparse* out, const char* path)
    {
        return matx_read_sparse_mtx_d_i8(alloc, out, path);
    }
    static matx_status_t vector_create(const matx_alloc_t* alloc, Vector* out, matx_int64_t n)
    {
        return matx_vec_d_i8_create(alloc, out, nullptr, n);
    }
    static matx_status_t vector_fill(Vector value, double fill_value)
    {
        return matx_vec_d_i8_fill(value, fill_value);
    }
    static void vector_destroy(const matx_alloc_t* alloc, Vector value)
    {
        matx_vec_d_i8_destroy(alloc, value);
    }
    static void vector_write(Vector value, const char* path)
    {
        matx_print_vec_d_i8(value, path);
    }
    static matx_status_t vector_read(const matx_alloc_t* alloc, Vector* out, const char* path)
    {
        return matx_read_vec_d_i8(alloc, out, path);
    }
    static matx_status_t factor(const matx_dense_linsolve_t* solver,
                                Dense a,
                                Factor* out)
    {
        return matx_factor_dense_d_i8(solver, a, out);
    }
    static void factor_destroy(const matx_dense_linsolve_t* solver, Factor value)
    {
        matx_factor_dense_d_i8_destroy(solver, value);
    }
    static matx_status_t solve_factor(const matx_dense_linsolve_t* solver,
                                      Factor factor,
                                      Vector b,
                                      Vector x)
    {
        return matx_solve_dense_d_i8_factor(solver, factor, b->data, x->data);
    }
    static matx_status_t solve_oneshot(const matx_dense_linsolve_t* solver,
                                        Dense a,
                                        Vector b,
                                        Vector x)
    {
        return matx_solve_dense_d_i8(solver, a, b->data, x->data);
    }
    static matx_status_t sparse_factor(const matx_sparse_linsolve_t* solver,
                                       Sparse a,
                                       SparseFactor* out)
    {
        return matx_factor_csc_d_i8(solver, a, out);
    }
    static void sparse_factor_destroy(const matx_sparse_linsolve_t* solver,
                                      SparseFactor* value)
    {
        matx_factor_csc_d_i8_destroy(solver, value);
    }
    static matx_status_t sparse_solve_factor(const matx_sparse_linsolve_t* solver,
                                             SparseFactor* factor,
                                             Vector b,
                                             Vector x)
    {
        return matx_solve_csc_d_i8_factor(solver, factor, b->data, x->data);
    }
    static matx_status_t sparse_solve_oneshot(const matx_sparse_linsolve_t* solver,
                                              Sparse a,
                                              Vector b,
                                              Vector x)
    {
        return matx_solve_csc_d_i8(solver, a, b->data, x->data);
    }
    static matx_status_t spmv(const matx_sparse_backend_t* backend,
                              Sparse a,
                              Vector x,
                              Vector y)
    {
        return matx_spmv_coo_d_i8(backend, 1.0, a, x, 0.0, y);
    }
    static matx_status_t spmm(const matx_sparse_backend_t* backend,
                              Sparse a,
                              Dense b,
                              Dense c)
    {
        return matx_spmm_coo_d_i8(backend, 1.0, a, b, 0.0, c);
    }
};

template <>
struct Ops<matx_complex_d_t>
{
    using Dense = matx_dense_z_i8_t;
    using Sparse = matx_coo_z_i8_t;
    using Vector = matx_vec_z_i8_t;
    using Factor = matx_factor_dense_z_i8_t*;
    using SparseFactor = matx_factor_sparse_z_i8_t;

    static constexpr const char* name = "c64";
    static matx_complex_d_t make(double real, double imag = 0.0) { return {real, imag}; }
    static matx_complex_d_t add(matx_complex_d_t a, matx_complex_d_t b)
    {
        return {a.real + b.real, a.imag + b.imag};
    }
    static matx_complex_d_t multiply(matx_complex_d_t a, matx_complex_d_t b)
    {
        return {a.real * b.real - a.imag * b.imag,
                a.real * b.imag + a.imag * b.real};
    }
    static double magnitude(matx_complex_d_t a) { return std::hypot(a.real, a.imag); }
    static double difference(matx_complex_d_t a, matx_complex_d_t b)
    {
        return std::hypot(a.real - b.real, a.imag - b.imag);
    }
    static matx_complex_d_t one() { return {1.0, 0.0}; }
    static matx_complex_d_t zero() { return {0.0, 0.0}; }
    static double flop_factor() { return 8.0; }

    static matx_status_t dense_create(const matx_alloc_t* alloc,
                                      Dense* out,
                                      matx_int64_t rows,
                                      matx_int64_t cols,
                                      matx_layout_t layout = MATX_COL_MAJOR)
    {
        return matx_dense_z_i8_create(alloc, out, layout, rows, cols, nullptr);
    }
    static matx_status_t dense_fill(Dense value, matx_complex_d_t fill_value)
    {
        return matx_dense_z_i8_fill(value, fill_value);
    }
    static void dense_destroy(const matx_alloc_t* alloc, Dense value)
    {
        matx_dense_z_i8_destroy(alloc, value);
    }
    static void dense_write(Dense value, const char* path)
    {
        matx_print_dense_mtx_z_i8(value, path);
    }
    static matx_status_t dense_read(const matx_alloc_t* alloc, Dense* out, const char* path)
    {
        return matx_read_dense_mtx_z_i8(alloc, out, path);
    }
    static matx_status_t gemm(const matx_dense_backend_t* backend,
                              Dense a,
                              Dense b,
                              Dense c)
    {
        return matx_gemm_z_i8(backend, MATX_NO_TRANS, MATX_NO_TRANS,
                              one(), a, b, zero(), c);
    }
    static matx_status_t sparse_create(const matx_alloc_t* alloc,
                                       Sparse* out,
                                       matx_int64_t n,
                                       matx_int64_t nnz,
                                       matx_int64_t* rows,
                                       matx_int64_t* cols,
                                       matx_complex_d_t* values)
    {
        return matx_coo_sparse_z_i8_create(alloc, out, n, n, nnz, rows, cols, values);
    }
    static void sparse_destroy(const matx_alloc_t* alloc, Sparse value)
    {
        matx_coo_sparse_z_i8_destroy(alloc, value);
    }
    static void sparse_write(Sparse value, const char* path)
    {
        matx_print_sparse_mtx_z_i8(value, path);
    }
    static matx_status_t sparse_read(const matx_alloc_t* alloc, Sparse* out, const char* path)
    {
        return matx_read_sparse_mtx_z_i8(alloc, out, path);
    }
    static matx_status_t vector_create(const matx_alloc_t* alloc, Vector* out, matx_int64_t n)
    {
        return matx_vec_z_i8_create(alloc, out, nullptr, n);
    }
    static matx_status_t vector_fill(Vector value, matx_complex_d_t fill_value)
    {
        return matx_vec_z_i8_fill(value, fill_value);
    }
    static void vector_destroy(const matx_alloc_t* alloc, Vector value)
    {
        matx_vec_z_i8_destroy(alloc, value);
    }
    static void vector_write(Vector value, const char* path)
    {
        matx_print_vec_z_i8(value, path);
    }
    static matx_status_t vector_read(const matx_alloc_t* alloc, Vector* out, const char* path)
    {
        return matx_read_vec_z_i8(alloc, out, path);
    }
    static matx_status_t factor(const matx_dense_linsolve_t* solver,
                                Dense a,
                                Factor* out)
    {
        return matx_factor_dense_z_i8(solver, a, out);
    }
    static void factor_destroy(const matx_dense_linsolve_t* solver, Factor value)
    {
        matx_factor_dense_z_i8_destroy(solver, value);
    }
    static matx_status_t solve_factor(const matx_dense_linsolve_t* solver,
                                      Factor factor,
                                      Vector b,
                                      Vector x)
    {
        return matx_solve_dense_z_i8_factor(solver, factor, b, x);
    }
    static matx_status_t solve_oneshot(const matx_dense_linsolve_t* solver,
                                        Dense a,
                                        Vector b,
                                        Vector x)
    {
        return matx_solve_dense_z_i8(solver, a, b, x);
    }
    static matx_status_t sparse_factor(const matx_sparse_linsolve_t* solver,
                                       Sparse a,
                                       SparseFactor* out)
    {
        return matx_factor_csc_z_i8(solver, a, out);
    }
    static void sparse_factor_destroy(const matx_sparse_linsolve_t* solver,
                                      SparseFactor* value)
    {
        matx_factor_csc_z_i8_destroy(solver, value);
    }
    static matx_status_t sparse_solve_factor(const matx_sparse_linsolve_t* solver,
                                             SparseFactor* factor,
                                             Vector b,
                                             Vector x)
    {
        return matx_solve_csc_z_i8_factor(solver, factor, b, x);
    }
    static matx_status_t sparse_solve_oneshot(const matx_sparse_linsolve_t* solver,
                                              Sparse a,
                                              Vector b,
                                              Vector x)
    {
        return matx_solve_csc_z_i8(solver, a, b, x);
    }
    static matx_status_t spmv(const matx_sparse_backend_t* backend,
                              Sparse a,
                              Vector x,
                              Vector y)
    {
        return matx_spmv_coo_z_i8(backend, one(), a, x, zero(), y);
    }
    static matx_status_t spmm(const matx_sparse_backend_t* backend,
                              Sparse a,
                              Dense b,
                              Dense c)
    {
        return matx_spmm_coo_z_i8(backend, one(), a, b, zero(), c);
    }
};

template <typename T>
T generated_value(uint64_t index, uint64_t salt)
{
    return Ops<T>::make(seeded_value(index, salt), seeded_value(index, salt + 17) * 0.25);
}

template <typename T>
void run_dense_lu_case(const matx_dense_linsolve_t& solver,
                       matx_int64_t n,
                       bool solve_only = false)
{
    using O = Ops<T>;
    using Dense = typename O::Dense;
    using Vector = typename O::Vector;
    using Factor = typename O::Factor;
    matx_alloc_t alloc = matx_alloc_default();
    Dense a = nullptr;
    Vector b = nullptr, x = nullptr;

    const auto setup_start = Clock::now();
    check_status(O::dense_create(&alloc, &a, n, n), "create LU matrix");
    check_status(O::vector_create(&alloc, &b, n), "create LU right-hand side");
    check_status(O::vector_create(&alloc, &x, n), "create LU solution");
    for (matx_int64_t col = 0; col < n; ++col) {
        for (matx_int64_t row = 0; row < n; ++row) {
            const matx_int64_t source_row = n > 1 && row < 2 ? 1 - row : row;
            const uint64_t index = (uint64_t) source_row + (uint64_t) col * n;
            const double value = seeded_value(index, 91) * 0.01;
            const T diagonal = O::make(n + 1.0 + value, 0.0);
            const T off_diagonal = O::make(value, seeded_value(index, 109) * 0.002);
            a->data[row + col * a->stride]
                = source_row == col ? diagonal : off_diagonal;
        }
    }
    for (matx_int64_t row = 0; row < n; ++row)
        b->data[row * b->stride] = generated_value<T>((uint64_t) row, 117);
    const double setup = elapsed_us(setup_start);

    auto factor_once = [&]() {
        Factor temporary = nullptr;
        check_status(O::factor(&solver, a, &temporary), "LU factorization");
        O::factor_destroy(&solver, temporary);
    };
    const Timing factor_timing = solve_only ? Timing{} : measure(factor_once);

    Factor factor = nullptr;
    check_status(O::factor(&solver, a, &factor), "LU factorization for reused solve");
    auto reused_solve = [&]() {
        check_status(O::solve_factor(&solver, factor, b, x), "LU reused solve");
    };
    const Timing reused_timing = measure(reused_solve);

    auto residual_error = [&]() {
        double max_rhs = 0.0;
        double max_residual = 0.0;
        for (matx_int64_t row = 0; row < n; ++row) {
            T product = O::zero();
            for (matx_int64_t col = 0; col < n; ++col) {
                product = O::add(product,
                                 O::multiply(a->data[row + col * a->stride],
                                             x->data[col * x->stride]));
            }
            const T rhs = b->data[row * b->stride];
            max_rhs = std::max(max_rhs, O::magnitude(rhs));
            max_residual = std::max(max_residual, O::difference(product, rhs));
        }
        return max_residual / (1.0 + max_rhs);
    };
    const double reused_residual = residual_error();
    if (!std::isfinite(reused_residual) || reused_residual > 1e-8)
        throw std::runtime_error("dense reused LU solve residual check failed");

    Timing oneshot_timing{};
    double oneshot_residual = 0.0;
    if (!solve_only) {
        auto oneshot_solve = [&]() {
            check_status(O::solve_oneshot(&solver, a, b, x), "LU one-shot solve");
        };
        oneshot_timing = measure(oneshot_solve);
        oneshot_residual = residual_error();
        if (!std::isfinite(oneshot_residual) || oneshot_residual > 1e-8)
            throw std::runtime_error("dense one-shot LU solve residual check failed");
    }

    const double factor_flops = O::flop_factor() * static_cast<double>(n) * n * n / 3.0;
    const double solve_flops = O::flop_factor() * static_cast<double>(n) * n;
    const double matrix_bytes = sizeof(T) * static_cast<double>(n) * n;
    const char* backend = matx_dense_linsolve_backend_name(solver.kind);
    if (!solve_only)
        csv_row("lu_factor", O::name, backend, n, n, n, 0, 0.0, 1,
                setup, factor_timing, factor_flops, 0.0, 2.0 * matrix_bytes);
    csv_row("lu_solve_reused", O::name, backend, n, 1, n, 0, 0.0, 1,
            setup, reused_timing, solve_flops, reused_residual, 2.0 * n * sizeof(T));
    if (!solve_only)
        csv_row("lu_solve_oneshot", O::name, backend, n, 1, n, 0, 0.0, 1,
                setup, oneshot_timing, factor_flops + solve_flops, oneshot_residual,
                2.0 * matrix_bytes);

    O::factor_destroy(&solver, factor);
    O::vector_destroy(&alloc, x);
    O::vector_destroy(&alloc, b);
    O::dense_destroy(&alloc, a);
}

template <typename T>
void run_sparse_lu_case(const matx_sparse_linsolve_t& solver,
                        matx_int64_t n,
                        bool solve_only = false)
{
    using O = Ops<T>;
    using Sparse = typename O::Sparse;
    using Vector = typename O::Vector;
    using Factor = typename O::SparseFactor;
    matx_alloc_t alloc = matx_alloc_default();
    std::vector<matx_int64_t> rows;
    std::vector<matx_int64_t> columns;
    std::vector<T> values;
    rows.reserve((size_t) n * 3);
    columns.reserve((size_t) n * 3);
    values.reserve((size_t) n * 3);
    for (matx_int64_t row = 0; row < n; ++row) {
        rows.push_back(row);
        columns.push_back(row);
        values.push_back(O::make(4.0, 0.0));
        if (row > 0) {
            rows.push_back(row);
            columns.push_back(row - 1);
            values.push_back(O::make(-0.5, 0.025));
        }
        if (row + 1 < n) {
            rows.push_back(row);
            columns.push_back(row + 1);
            values.push_back(O::make(-0.5, -0.025));
        }
    }

    Sparse a_factor = nullptr, a_oneshot = nullptr, a_probe = nullptr;
    Vector b = nullptr, x = nullptr;
    const auto setup_start = Clock::now();
    const matx_int64_t nnz = (matx_int64_t) values.size();
    check_status(O::sparse_create(&alloc, &a_factor, n, nnz,
                                  rows.data(), columns.data(), values.data()),
                 "create sparse LU matrix");
    if (!solve_only)
        check_status(O::sparse_create(&alloc, &a_oneshot, n, nnz,
                                      rows.data(), columns.data(), values.data()),
                     "create one-shot sparse LU matrix");
    check_status(O::vector_create(&alloc, &b, n), "create sparse LU right-hand side");
    check_status(O::vector_create(&alloc, &x, n), "create sparse LU solution");
    for (matx_int64_t row = 0; row < n; ++row)
        b->data[row * b->stride] = generated_value<T>((uint64_t) row, 129);
    const double setup = elapsed_us(setup_start);

    if (!solve_only) {
        check_status(O::sparse_create(&alloc, &a_probe, n, nnz,
                                      rows.data(), columns.data(), values.data()),
                     "create sparse LU support probe");
        Factor support_probe{};
        matx_status_t support_status = O::sparse_factor(&solver, a_probe, &support_probe);
        if (support_status == MATX_ERR_NOT_SUPPORTED) {
            O::sparse_factor_destroy(&solver, &support_probe);
            O::sparse_destroy(&alloc, a_probe);
            O::sparse_destroy(&alloc, a_oneshot);
            O::sparse_destroy(&alloc, a_factor);
            O::vector_destroy(&alloc, x);
            O::vector_destroy(&alloc, b);
            return;
        }
        check_status(support_status, "sparse LU support probe");
        O::sparse_factor_destroy(&solver, &support_probe);
        O::sparse_destroy(&alloc, a_probe);
    }

    auto factor_once = [&]() {
        Factor temporary{};
        check_status(O::sparse_factor(&solver, a_factor, &temporary),
                     "sparse LU factorization");
        O::sparse_factor_destroy(&solver, &temporary);
    };
    const Timing factor_timing = solve_only ? Timing{} : measure(factor_once);

    Factor factor{};
    matx_status_t factor_status = O::sparse_factor(&solver, a_factor, &factor);
    if (factor_status == MATX_ERR_NOT_SUPPORTED) {
        O::sparse_factor_destroy(&solver, &factor);
        O::sparse_destroy(&alloc, a_oneshot);
        O::sparse_destroy(&alloc, a_factor);
        O::vector_destroy(&alloc, x);
        O::vector_destroy(&alloc, b);
        return;
    }
    check_status(factor_status, "sparse LU factorization for reused solve");
    auto reused_solve = [&]() {
        check_status(O::sparse_solve_factor(&solver, &factor, b, x),
                     "sparse LU reused solve");
    };
    const Timing reused_timing = measure(reused_solve);

    std::vector<T> product((size_t) n, O::zero());
    auto residual_error = [&]() {
        std::fill(product.begin(), product.end(), O::zero());
        for (matx_int64_t k = 0; k < nnz; ++k) {
            const matx_int64_t row = rows[(size_t) k];
            const matx_int64_t col = columns[(size_t) k];
            product[(size_t) row]
                = O::add(product[(size_t) row],
                         O::multiply(values[(size_t) k], x->data[col * x->stride]));
        }
        double max_rhs = 0.0;
        double max_residual = 0.0;
        for (matx_int64_t row = 0; row < n; ++row) {
            const T rhs = b->data[row * b->stride];
            max_rhs = std::max(max_rhs, O::magnitude(rhs));
            max_residual = std::max(max_residual,
                                    O::difference(product[(size_t) row], rhs));
        }
        return max_residual / (1.0 + max_rhs);
    };
    const double reused_residual = residual_error();
    if (!std::isfinite(reused_residual) || reused_residual > 1e-8)
        throw std::runtime_error("sparse reused LU solve residual check failed");

    Timing oneshot_timing{};
    double oneshot_residual = 0.0;
    if (!solve_only) {
        auto oneshot_solve = [&]() {
            check_status(O::sparse_solve_oneshot(&solver, a_oneshot, b, x),
                         "sparse LU one-shot solve");
        };
        oneshot_timing = measure(oneshot_solve);
        oneshot_residual = residual_error();
        if (!std::isfinite(oneshot_residual) || oneshot_residual > 1e-8)
            throw std::runtime_error("sparse one-shot LU solve residual check failed");
    }

    const double density = (double) nnz / ((double) n * (double) n);
    const char* backend = matx_sparse_linsolve_backend_name(solver.kind);
    if (!solve_only)
        csv_row("sparse_lu_factor", O::name, backend, n, n, 1, nnz, density, 1,
                setup, factor_timing, 0.0, 0.0);
    csv_row("sparse_lu_solve_reused", O::name, backend, n, 1, 1, nnz, density, 1,
            setup, reused_timing, 0.0, reused_residual);
    if (!solve_only)
        csv_row("sparse_lu_solve_oneshot", O::name, backend, n, 1, 1, nnz, density, 1,
                setup, oneshot_timing, 0.0, oneshot_residual);

    O::sparse_factor_destroy(&solver, &factor);
    O::sparse_destroy(&alloc, a_oneshot);
    O::sparse_destroy(&alloc, a_factor);
    O::vector_destroy(&alloc, x);
    O::vector_destroy(&alloc, b);
}

std::string benchmark_file_path(const std::string& category)
{
    const auto stamp = Clock::now().time_since_epoch().count();
    return (std::filesystem::temp_directory_path()
            / (std::string("matx_bench_") + category + "_" + std::to_string(stamp) + ".mtx"))
        .string();
}

template <typename T>
void run_core_case(matx_int64_t n)
{
    using O = Ops<T>;
    using Dense = typename O::Dense;
    using Sparse = typename O::Sparse;
    using Vector = typename O::Vector;
    matx_alloc_t alloc = matx_alloc_default();
    Vector vector = nullptr;
    Dense dense = nullptr;
    Sparse sparse = nullptr;
    const matx_int64_t nnz = n * 3 - 2;
    std::vector<matx_int64_t> rows, columns;
    std::vector<T> values;
    rows.reserve((size_t) nnz);
    columns.reserve((size_t) nnz);
    values.reserve((size_t) nnz);
    for (matx_int64_t row = 0; row < n; ++row) {
        rows.push_back(row);
        columns.push_back(row);
        values.push_back(O::one());
        if (row > 0) {
            rows.push_back(row);
            columns.push_back(row - 1);
            values.push_back(O::one());
        }
        if (row + 1 < n) {
            rows.push_back(row);
            columns.push_back(row + 1);
            values.push_back(O::one());
        }
    }

    const auto setup_start = Clock::now();
    check_status(O::vector_create(&alloc, &vector, n), "create core vector");
    check_status(O::dense_create(&alloc, &dense, n, n), "create core dense matrix");
    check_status(O::sparse_create(&alloc, &sparse, n, nnz,
                                  rows.data(), columns.data(), values.data()),
                 "create core sparse matrix");
    const double setup = elapsed_us(setup_start);

    auto vector_alloc = [&]() {
        Vector value = nullptr;
        check_status(O::vector_create(&alloc, &value, n), "allocate core vector");
        O::vector_destroy(&alloc, value);
    };
    auto dense_alloc = [&]() {
        Dense value = nullptr;
        check_status(O::dense_create(&alloc, &value, n, n), "allocate core dense matrix");
        O::dense_destroy(&alloc, value);
    };
    auto sparse_alloc = [&]() {
        Sparse value = nullptr;
        check_status(O::sparse_create(&alloc, &value, n, nnz,
                                      rows.data(), columns.data(), values.data()),
                     "allocate core sparse matrix");
        O::sparse_destroy(&alloc, value);
    };
    const Timing vector_alloc_timing = measure(vector_alloc);
    const Timing dense_alloc_timing = measure(dense_alloc);
    const Timing sparse_alloc_timing = measure(sparse_alloc);

    const T fill_value = generated_value<T>(0, 151);
    auto vector_fill = [&]() {
        check_status(O::vector_fill(vector, fill_value), "fill core vector");
    };
    auto dense_fill = [&]() {
        check_status(O::dense_fill(dense, fill_value), "fill core dense matrix");
    };
    const Timing vector_fill_timing = measure(vector_fill);
    const Timing dense_fill_timing = measure(dense_fill);
    if (O::difference(vector->data[(n - 1) * vector->stride], fill_value) != 0.0
        || O::difference(dense->data[(n - 1) + (n - 1) * dense->stride], fill_value) != 0.0) {
        throw std::runtime_error("core fill reference check failed");
    }

    const double density = (double) nnz / ((double) n * (double) n);
    csv_row("core_vector_alloc_free", O::name, "ALLOCATOR", n, 1, 0, 0, 0.0, 1,
            setup, vector_alloc_timing, 0.0, 0.0, (double) n * sizeof(T));
    csv_row("core_dense_alloc_free", O::name, "ALLOCATOR", n, n, 0, 0, 0.0, 1,
            setup, dense_alloc_timing, 0.0, 0.0, (double) n * n * sizeof(T));
    csv_row("core_sparse_alloc_free", O::name, "ALLOCATOR", n, n, 0, nnz, density, 1,
            setup, sparse_alloc_timing, 0.0, 0.0,
            (double) nnz * (2 * sizeof(matx_int64_t) + sizeof(T)));
    csv_row("core_vector_fill", O::name, "CORE", n, 1, 0, 0, 0.0, 1,
            setup, vector_fill_timing, 0.0, 0.0, 2.0 * n * sizeof(T));
    csv_row("core_dense_fill", O::name, "CORE", n, n, 0, 0, 0.0, 1,
            setup, dense_fill_timing, 0.0, 0.0, 2.0 * n * n * sizeof(T));

    O::sparse_destroy(&alloc, sparse);
    O::dense_destroy(&alloc, dense);
    O::vector_destroy(&alloc, vector);
}

template <typename T>
void run_dense_scalar_case(matx_int64_t n, matx_layout_t layout)
{
    using O = Ops<T>;
    using Dense = typename O::Dense;
    matx_alloc_t alloc = matx_alloc_default();
    Dense matrix = nullptr, other = nullptr, output = nullptr;
    check_status(O::dense_create(&alloc, &matrix, n, n, layout),
                 "create scalar-operation matrix");
    check_status(O::dense_create(&alloc, &other, n, n, layout),
                 "create elementwise second matrix");
    check_status(O::dense_fill(other, O::make(1.25, 0.75)),
                 "fill elementwise second matrix");

    const T input_value = O::make(0.25, -0.5);
    const T add_value = O::make(0.125, 0.25);
    const T mul_value = O::make(0.5, -0.25);
    std::vector<T> initial((size_t) n * (size_t) n, input_value);
    auto reset = [&]() {
        std::copy(initial.begin(), initial.end(), matrix->data);
    };
    auto scalar_add = [&]() {
        if constexpr (std::is_same_v<T, double>)
            check_status(matx_dense_d_i8_add_scalar(matrix, add_value),
                         "dense scalar add");
        else
            check_status(matx_dense_z_i8_add_scalar(matrix, add_value),
                         "complex dense scalar add");
    };
    auto scalar_mul = [&]() {
        if constexpr (std::is_same_v<T, double>)
            check_status(matx_dense_d_i8_mul_scalar(matrix, mul_value),
                         "dense scalar multiply");
        else
            check_status(matx_dense_z_i8_mul_scalar(matrix, mul_value),
                         "complex dense scalar multiply");
    };
    auto add_reset = [&]() { reset(); };
    const Timing add_timing = measure_reinitialized(add_reset, scalar_add);
    const T expected_add = O::add(input_value, add_value);
    if (O::difference(matrix->data[0], expected_add) > 1e-14)
        throw std::runtime_error("dense scalar add reference check failed");

    auto mul_reset = [&]() { reset(); };
    const Timing mul_timing = measure_reinitialized(mul_reset, scalar_mul);
    const T expected_mul = O::multiply(input_value, mul_value);
    if (O::difference(matrix->data[0], expected_mul) > 1e-14)
        throw std::runtime_error("dense scalar multiply reference check failed");
    reset();

    auto dense_add = [&]() {
        if constexpr (std::is_same_v<T, double>)
            check_status(matx_dense_d_i8_add(&alloc, matrix, other, &output),
                         "dense elementwise add");
        else
            check_status(matx_dense_z_i8_add(&alloc, matrix, other, &output),
                         "complex dense elementwise add");
    };
    auto dense_sub = [&]() {
        if constexpr (std::is_same_v<T, double>)
            check_status(matx_dense_d_i8_sub(&alloc, matrix, other, &output),
                         "dense elementwise subtract");
        else
            check_status(matx_dense_z_i8_sub(&alloc, matrix, other, &output),
                         "complex dense elementwise subtract");
    };
    auto dense_mul = [&]() {
        if constexpr (std::is_same_v<T, double>)
            check_status(matx_dense_d_i8_mul(&alloc, matrix, other, &output),
                         "dense elementwise multiply");
        else
            check_status(matx_dense_z_i8_mul(&alloc, matrix, other, &output),
                         "complex dense elementwise multiply");
    };
    auto dense_div = [&]() {
        if constexpr (std::is_same_v<T, double>)
            check_status(matx_dense_d_i8_div(&alloc, matrix, other, &output),
                         "dense elementwise divide");
        else
            check_status(matx_dense_z_i8_div(&alloc, matrix, other, &output),
                         "complex dense elementwise divide");
    };
    const Timing add_api_timing = measure(dense_add);
    const Timing sub_api_timing = measure(dense_sub);
    const Timing mul_api_timing = measure(dense_mul);
    const Timing div_api_timing = measure(dense_div);
    const T expected_add_api = O::add(input_value, O::make(1.25, 0.75));
    const T expected_sub_api = O::add(input_value, O::make(-1.25, -0.75));
    const T expected_mul_api = O::multiply(input_value, O::make(1.25, 0.75));
    dense_mul();
    if (!output || O::difference(output->data[0], expected_mul_api) > 1e-12)
        throw std::runtime_error("dense elementwise reference check failed");
    dense_add();
    if (O::difference(output->data[0], expected_add_api) > 1e-12)
        throw std::runtime_error("dense elementwise add reference check failed");
    dense_sub();
    if (O::difference(output->data[0], expected_sub_api) > 1e-12)
        throw std::runtime_error("dense elementwise subtract reference check failed");
    dense_div();
    const T expected_div = O::make(std::is_same_v<T, double> ? 0.2 : -1.0 / 34.0,
                                   std::is_same_v<T, double> ? 0.0 : -13.0 / 34.0);
    if (O::difference(output->data[0], expected_div) > 1e-12)
        throw std::runtime_error("dense elementwise divide reference check failed");

    Timing random_timing{};
    Timing zeros_timing{};
    Timing ones_timing{};
    if constexpr (std::is_same_v<T, double>) {
        auto random_fill = [&]() {
            check_status(matx_dense_d_i8_rand_uniform(matrix, -1.0, 1.0, 991),
                         "dense uniform random fill");
        };
        random_timing = measure(random_fill);
        for (matx_int64_t i = 0; i < n * n; ++i) {
            if (matrix->data[i] < -1.0 || matrix->data[i] > 1.0)
                throw std::runtime_error("dense random fill range check failed");
        }
        auto zeros = [&]() {
            check_status(matx_dense_d_i8_zeros(matrix), "real dense zeros");
        };
        auto ones = [&]() {
            check_status(matx_dense_d_i8_ones(matrix), "real dense ones");
        };
        zeros_timing = measure(zeros);
        ones_timing = measure(ones);
        if (matrix->data[0] != 1.0)
            throw std::runtime_error("real dense one fill check failed");
    } else {
        auto zeros = [&]() {
            check_status(matx_dense_z_i8_zeros(matrix), "complex dense zeros");
        };
        zeros_timing = measure(zeros);
        if (matrix->data[0].real != 0.0 || matrix->data[0].imag != 0.0)
            throw std::runtime_error("complex dense zero fill check failed");
    }

    const char* layout_name = layout == MATX_COL_MAJOR ? "COL_MAJOR" : "ROW_MAJOR";
    const double bytes = 2.0 * (double) n * (double) n * sizeof(T);
    csv_row("core_dense_scalar_add", O::name, "CORE", n, n, 0, 0, 0.0, 1,
            0.0, add_timing, 0.0, 0.0, bytes, layout_name);
    csv_row("core_dense_scalar_mul", O::name, "CORE", n, n, 0, 0, 0.0, 1,
            0.0, mul_timing, 0.0, 0.0, bytes, layout_name);
    csv_row("core_dense_elementwise_add", O::name, "CORE", n, n, 0, 0, 0.0, 1,
            0.0, add_api_timing, 0.0, 0.0, bytes, layout_name);
    csv_row("core_dense_elementwise_sub", O::name, "CORE", n, n, 0, 0, 0.0, 1,
            0.0, sub_api_timing, 0.0, 0.0, bytes, layout_name);
    csv_row("core_dense_elementwise_mul", O::name, "CORE", n, n, 0, 0, 0.0, 1,
            0.0, mul_api_timing, 0.0, 0.0, bytes, layout_name);
    csv_row("core_dense_elementwise_div", O::name, "CORE", n, n, 0, 0, 0.0, 1,
            0.0, div_api_timing, 0.0, 0.0, bytes, layout_name);
    if constexpr (std::is_same_v<T, double>) {
        csv_row("core_dense_rand_uniform", O::name, "CORE", n, n, 0, 0, 0.0, 1,
                0.0, random_timing, 0.0, 0.0, bytes, layout_name);
        csv_row("core_dense_zeros", O::name, "CORE", n, n, 0, 0, 0.0, 1,
                0.0, zeros_timing, 0.0, 0.0, bytes, layout_name);
        csv_row("core_dense_ones", O::name, "CORE", n, n, 0, 0, 0.0, 1,
                0.0, ones_timing, 0.0, 0.0, bytes, layout_name);
    } else {
        csv_row("core_dense_zeros", O::name, "CORE", n, n, 0, 0, 0.0, 1,
                0.0, zeros_timing, 0.0, 0.0, bytes, layout_name);
    }
    O::dense_destroy(&alloc, output);
    O::dense_destroy(&alloc, other);
    O::dense_destroy(&alloc, matrix);
}

void run_core_unary_case(matx_int64_t n, matx_layout_t layout)
{
    matx_alloc_t alloc = matx_alloc_default();
    matx_vec_d_i8_t vector_d = nullptr;
    matx_vec_z_i8_t vector_z = nullptr;
    matx_dense_d_i8_t dense_d = nullptr;
    matx_dense_z_i8_t dense_z = nullptr;
    check_status(matx_vec_d_i8_create(&alloc, &vector_d, nullptr, n), "create unary real vector");
    check_status(matx_vec_z_i8_create(&alloc, &vector_z, nullptr, n), "create unary complex vector");
    check_status(matx_dense_d_i8_create(&alloc, &dense_d, layout, n, n, nullptr),
                 "create unary real matrix");
    check_status(matx_dense_z_i8_create(&alloc, &dense_z, layout, n, n, nullptr),
                 "create unary complex matrix");
    auto dense_index = [=](matx_int64_t row, matx_int64_t col, matx_int64_t stride) {
        return layout == MATX_COL_MAJOR ? row + col * stride : row * stride + col;
    };
    for (matx_int64_t i = 0; i < n; ++i) {
        const double real = 0.2 + 0.001 * (i % 37);
        vector_d->data[i] = real;
        vector_z->data[i] = {real, 0.05 + 0.001 * (i % 13)};
        for (matx_int64_t j = 0; j < n; ++j) {
            const double matrix_real = 0.2 + 0.001 * ((i + j) % 37);
            dense_d->data[dense_index(i, j, dense_d->stride)] = matrix_real;
            dense_z->data[dense_index(i, j, dense_z->stride)]
                = {matrix_real, 0.05 + 0.001 * ((i + j) % 13)};
        }
    }
    const char* layout_name = layout == MATX_COL_MAJOR ? "COL_MAJOR" : "ROW_MAJOR";
    const matx_int64_t samples[3] = {0, n / 2, n - 1};
    const double vector_bytes = 2.0 * (double) n * sizeof(double);
    const double matrix_bytes = 2.0 * (double) n * (double) n * sizeof(double);

    auto run_vec_d = [&](const char* operation, auto function, auto reference) {
        matx_vec_d_i8_t output = nullptr;
        auto invoke = [&]() { check_status(function(&alloc, vector_d, &output), operation); };
        const Timing timing = measure(invoke);
        double error = 0.0;
        for (matx_int64_t i : samples)
            error = std::max(error, std::abs(output->data[i] - reference(vector_d->data[i])));
        if (!std::isfinite(error) || error > 1e-11)
            throw std::runtime_error(std::string(operation) + " reference check failed");
        csv_row(operation, "f64", "CORE", n, 1, 0, 0, 0.0, 1,
                0.0, timing, (double) n, error, vector_bytes);
        matx_vec_d_i8_destroy(&alloc, output);
    };
    auto run_vec_z = [&](const char* operation, auto function, auto reference) {
        matx_vec_z_i8_t output = nullptr;
        auto invoke = [&]() { check_status(function(&alloc, vector_z, &output), operation); };
        const Timing timing = measure(invoke);
        double error = 0.0;
        for (matx_int64_t i : samples) {
            const auto input = vector_z->data[i * vector_z->stride];
            const auto actual = output->data[i * output->stride];
            const std::complex<double> expected = reference({input.real, input.imag});
            error = std::max(error, std::hypot(actual.real - expected.real(),
                                               actual.imag - expected.imag()));
        }
        if (!std::isfinite(error) || error > 1e-11)
            throw std::runtime_error(std::string(operation) + " reference check failed");
        csv_row(operation, "c64", "CORE", n, 1, 0, 0, 0.0, 1,
                0.0, timing, 8.0 * n, error, 2.0 * vector_bytes);
        matx_vec_z_i8_destroy(&alloc, output);
    };
    auto run_vec_z_abs = [&](const char* operation, auto function, auto reference) {
        matx_vec_d_i8_t output = nullptr;
        auto invoke = [&]() { check_status(function(&alloc, vector_z, &output), operation); };
        const Timing timing = measure(invoke);
        double error = 0.0;
        for (matx_int64_t i : samples) {
            const auto input = vector_z->data[i * vector_z->stride];
            error = std::max(error, std::abs(output->data[i * output->stride]
                                             - reference({input.real, input.imag})));
        }
        if (!std::isfinite(error) || error > 1e-11)
            throw std::runtime_error(std::string(operation) + " reference check failed");
        csv_row(operation, "c64", "CORE", n, 1, 0, 0, 0.0, 1,
                0.0, timing, 8.0 * n, error, 2.0 * vector_bytes);
        matx_vec_d_i8_destroy(&alloc, output);
    };
    auto run_dense_d = [&](const char* operation, auto function, auto reference) {
        matx_dense_d_i8_t output = nullptr;
        auto invoke = [&]() { check_status(function(&alloc, dense_d, &output), operation); };
        const Timing timing = measure(invoke);
        double error = 0.0;
        const matx_int64_t coords[3][2] = {{0, 0}, {n / 2, n / 2}, {n - 1, n - 1}};
        for (const auto& coordinate : coords) {
            const matx_int64_t row = coordinate[0], col = coordinate[1];
            const auto input = dense_d->data[dense_index(row, col, dense_d->stride)];
            const auto actual = output->data[dense_index(row, col, output->stride)];
            error = std::max(error, std::abs(actual - reference(input)));
        }
        if (!std::isfinite(error) || error > 1e-11)
            throw std::runtime_error(std::string(operation) + " reference check failed");
        csv_row(operation, "f64", "CORE", n, n, 0, 0, 0.0, 1,
                0.0, timing, (double) n * n, error, matrix_bytes, layout_name);
        matx_dense_d_i8_destroy(&alloc, output);
    };
    auto run_dense_z = [&](const char* operation, auto function, auto reference) {
        matx_dense_z_i8_t output = nullptr;
        auto invoke = [&]() { check_status(function(&alloc, dense_z, &output), operation); };
        const Timing timing = measure(invoke);
        double error = 0.0;
        const matx_int64_t coords[3][2] = {{0, 0}, {n / 2, n / 2}, {n - 1, n - 1}};
        for (const auto& coordinate : coords) {
            const matx_int64_t row = coordinate[0], col = coordinate[1];
            const auto input = dense_z->data[dense_index(row, col, dense_z->stride)];
            const auto actual = output->data[dense_index(row, col, output->stride)];
            const std::complex<double> expected = reference({input.real, input.imag});
            error = std::max(error, std::hypot(actual.real - expected.real(),
                                               actual.imag - expected.imag()));
        }
        if (!std::isfinite(error) || error > 1e-11)
            throw std::runtime_error(std::string(operation) + " reference check failed");
        csv_row(operation, "c64", "CORE", n, n, 0, 0, 0.0, 1,
                0.0, timing, 8.0 * (double) n * n, error, 2.0 * matrix_bytes, layout_name);
        matx_dense_z_i8_destroy(&alloc, output);
    };
    auto run_dense_z_abs = [&](const char* operation, auto function, auto reference) {
        matx_dense_d_i8_t output = nullptr;
        auto invoke = [&]() { check_status(function(&alloc, dense_z, &output), operation); };
        const Timing timing = measure(invoke);
        double error = 0.0;
        const matx_int64_t coords[3][2] = {{0, 0}, {n / 2, n / 2}, {n - 1, n - 1}};
        for (const auto& coordinate : coords) {
            const matx_int64_t row = coordinate[0], col = coordinate[1];
            const auto input = dense_z->data[dense_index(row, col, dense_z->stride)];
            const auto actual = output->data[dense_index(row, col, output->stride)];
            error = std::max(error, std::abs(actual - reference({input.real, input.imag})));
        }
        if (!std::isfinite(error) || error > 1e-11)
            throw std::runtime_error(std::string(operation) + " reference check failed");
        csv_row(operation, "c64", "CORE", n, n, 0, 0, 0.0, 1,
                0.0, timing, 8.0 * (double) n * n, error, 2.0 * matrix_bytes, layout_name);
        matx_dense_d_i8_destroy(&alloc, output);
    };

    auto real_exp = [](double x) { return std::exp(x); };
    auto real_log = [](double x) { return std::log(x); };
    auto real_sqrt = [](double x) { return std::sqrt(x); };
    auto real_sin = [](double x) { return std::sin(x); };
    auto real_cos = [](double x) { return std::cos(x); };
    auto real_abs = [](double x) { return std::abs(x); };
    auto real_pow = [](double x) { return x * x; };
    auto complex_exp = [](std::complex<double> x) { return std::exp(x); };
    auto complex_log = [](std::complex<double> x) { return std::log(x); };
    auto complex_sqrt = [](std::complex<double> x) { return std::sqrt(x); };
    auto complex_sin = [](std::complex<double> x) { return std::sin(x); };
    auto complex_cos = [](std::complex<double> x) { return std::cos(x); };
    auto complex_abs = [](std::complex<double> x) { return std::abs(x); };
    auto complex_pow = [](std::complex<double> x) { return std::pow(x, std::complex<double>(2.0, 1.0)); };

    run_vec_d("core_vec_exp_f64", [](const auto* a, auto x, auto* y) { return matx_vec_d_i8_exp(a, x, y); }, real_exp);
    run_vec_d("core_vec_log_f64", [](const auto* a, auto x, auto* y) { return matx_vec_d_i8_log(a, x, y); }, real_log);
    run_vec_d("core_vec_sqrt_f64", [](const auto* a, auto x, auto* y) { return matx_vec_d_i8_sqrt(a, x, y); }, real_sqrt);
    run_vec_d("core_vec_sin_f64", [](const auto* a, auto x, auto* y) { return matx_vec_d_i8_sin(a, x, y); }, real_sin);
    run_vec_d("core_vec_cos_f64", [](const auto* a, auto x, auto* y) { return matx_vec_d_i8_cos(a, x, y); }, real_cos);
    run_vec_d("core_vec_abs_f64", [](const auto* a, auto x, auto* y) { return matx_vec_d_i8_abs(a, x, y); }, real_abs);
    run_vec_d("core_vec_pow_f64", [](const auto* a, auto x, auto* y) { return matx_vec_d_i8_pow(a, x, 2.0, y); }, real_pow);
    run_vec_z("core_vec_exp_c64", [](const auto* a, auto x, auto* y) { return matx_vec_z_i8_exp(a, x, y); }, complex_exp);
    run_vec_z("core_vec_log_c64", [](const auto* a, auto x, auto* y) { return matx_vec_z_i8_log(a, x, y); }, complex_log);
    run_vec_z("core_vec_sqrt_c64", [](const auto* a, auto x, auto* y) { return matx_vec_z_i8_sqrt(a, x, y); }, complex_sqrt);
    run_vec_z("core_vec_sin_c64", [](const auto* a, auto x, auto* y) { return matx_vec_z_i8_sin(a, x, y); }, complex_sin);
    run_vec_z("core_vec_cos_c64", [](const auto* a, auto x, auto* y) { return matx_vec_z_i8_cos(a, x, y); }, complex_cos);
    run_vec_z_abs("core_vec_abs_c64", [](const auto* a, auto x, auto* y) { return matx_vec_z_i8_abs(a, x, y); }, complex_abs);
    run_vec_z("core_vec_pow_c64", [](const auto* a, auto x, auto* y) { return matx_vec_z_i8_pow(a, x, {2.0, 1.0}, y); }, complex_pow);
    run_dense_d("core_dense_exp_f64", [](const auto* a, auto x, auto* y) { return matx_dense_d_i8_exp(a, x, y); }, real_exp);
    run_dense_d("core_dense_log_f64", [](const auto* a, auto x, auto* y) { return matx_dense_d_i8_log(a, x, y); }, real_log);
    run_dense_d("core_dense_sqrt_f64", [](const auto* a, auto x, auto* y) { return matx_dense_d_i8_sqrt(a, x, y); }, real_sqrt);
    run_dense_d("core_dense_sin_f64", [](const auto* a, auto x, auto* y) { return matx_dense_d_i8_sin(a, x, y); }, real_sin);
    run_dense_d("core_dense_cos_f64", [](const auto* a, auto x, auto* y) { return matx_dense_d_i8_cos(a, x, y); }, real_cos);
    run_dense_d("core_dense_abs_f64", [](const auto* a, auto x, auto* y) { return matx_dense_d_i8_abs(a, x, y); }, real_abs);
    run_dense_d("core_dense_pow_f64", [](const auto* a, auto x, auto* y) { return matx_dense_d_i8_pow(a, x, 2.0, y); }, real_pow);
    run_dense_z("core_dense_exp_c64", [](const auto* a, auto x, auto* y) { return matx_dense_z_i8_exp(a, x, y); }, complex_exp);
    run_dense_z("core_dense_log_c64", [](const auto* a, auto x, auto* y) { return matx_dense_z_i8_log(a, x, y); }, complex_log);
    run_dense_z("core_dense_sqrt_c64", [](const auto* a, auto x, auto* y) { return matx_dense_z_i8_sqrt(a, x, y); }, complex_sqrt);
    run_dense_z("core_dense_sin_c64", [](const auto* a, auto x, auto* y) { return matx_dense_z_i8_sin(a, x, y); }, complex_sin);
    run_dense_z("core_dense_cos_c64", [](const auto* a, auto x, auto* y) { return matx_dense_z_i8_cos(a, x, y); }, complex_cos);
    run_dense_z_abs("core_dense_abs_c64", [](const auto* a, auto x, auto* y) { return matx_dense_z_i8_abs(a, x, y); }, complex_abs);
    run_dense_z("core_dense_pow_c64", [](const auto* a, auto x, auto* y) { return matx_dense_z_i8_pow(a, x, {2.0, 1.0}, y); }, complex_pow);

    matx_dense_z_i8_destroy(&alloc, dense_z);
    matx_dense_d_i8_destroy(&alloc, dense_d);
    matx_vec_z_i8_destroy(&alloc, vector_z);
    matx_vec_d_i8_destroy(&alloc, vector_d);
}

void run_core_vector_api_case(matx_int64_t n)
{
    matx_alloc_t alloc = matx_alloc_default();
    matx_vec_d_i8_t a = nullptr, b = nullptr, out_d = nullptr;
    matx_vec_z_i8_t za = nullptr, zb = nullptr, out_z = nullptr, converted = nullptr;
    check_status(matx_vec_d_i8_create(&alloc, &a, nullptr, n), "create core API real vector A");
    check_status(matx_vec_d_i8_create(&alloc, &b, nullptr, n), "create core API real vector B");
    check_status(matx_vec_z_i8_create(&alloc, &za, nullptr, n), "create core API complex vector A");
    check_status(matx_vec_z_i8_create(&alloc, &zb, nullptr, n), "create core API complex vector B");
    for (matx_int64_t i = 0; i < n; ++i) {
        a->data[i] = 1.0 + 0.001 * i;
        b->data[i] = 2.0 + 0.001 * i;
        za->data[i] = {a->data[i], 0.25};
        zb->data[i] = {b->data[i], -0.5};
    }
    const matx_int64_t samples[3] = {0, n / 2, n - 1};
    auto run_d_binary = [&](const char* operation, auto function, auto reference) {
        auto invoke = [&]() { check_status(function(&alloc, a, b, &out_d), operation); };
        const Timing timing = measure(invoke);
        double error = 0.0;
        for (matx_int64_t i : samples)
            error = std::max(error, std::abs(out_d->data[i] - reference(a->data[i], b->data[i])));
        if (!std::isfinite(error) || error > 1e-11)
            throw std::runtime_error(std::string(operation) + " reference check failed");
        csv_row(operation, "f64", "CORE", n, 1, 0, 0, 0.0, 1,
                0.0, timing, (double) n, error, 3.0 * n * sizeof(double));
        matx_vec_d_i8_destroy(&alloc, out_d);
        out_d = nullptr;
    };
    auto run_z_binary = [&](const char* operation, auto function, auto reference) {
        auto invoke = [&]() { check_status(function(&alloc, za, zb, &out_z), operation); };
        const Timing timing = measure(invoke);
        double error = 0.0;
        for (matx_int64_t i : samples) {
            const auto actual = out_z->data[i * out_z->stride];
            const auto expected = reference(za->data[i * za->stride], zb->data[i * zb->stride]);
            error = std::max(error, std::hypot(actual.real - expected.real,
                                               actual.imag - expected.imag));
        }
        if (!std::isfinite(error) || error > 1e-11)
            throw std::runtime_error(std::string(operation) + " reference check failed");
        csv_row(operation, "c64", "CORE", n, 1, 0, 0, 0.0, 1,
                0.0, timing, 8.0 * n, error, 3.0 * n * sizeof(matx_complex_d_t));
        matx_vec_z_i8_destroy(&alloc, out_z);
        out_z = nullptr;
    };
    run_d_binary("core_vec_add_f64", [](const auto* al, auto x, auto y, auto* z) { return matx_vec_d_i8_add(al, x, y, z); },
                 [](double x, double y) { return x + y; });
    run_d_binary("core_vec_sub_f64", [](const auto* al, auto x, auto y, auto* z) { return matx_vec_d_i8_sub(al, x, y, z); },
                 [](double x, double y) { return x - y; });
    run_d_binary("core_vec_mul_f64", [](const auto* al, auto x, auto y, auto* z) { return matx_vec_d_i8_mul(al, x, y, z); },
                 [](double x, double y) { return x * y; });
    run_d_binary("core_vec_div_f64", [](const auto* al, auto x, auto y, auto* z) { return matx_vec_d_i8_div(al, x, y, z); },
                 [](double x, double y) { return x / y; });
    auto z_add = [](auto x, auto y) { return matx_complex_d_t{x.real + y.real, x.imag + y.imag}; };
    auto z_sub = [](auto x, auto y) { return matx_complex_d_t{x.real - y.real, x.imag - y.imag}; };
    auto z_mul = [](auto x, auto y) { return matx_complex_d_t{x.real * y.real - x.imag * y.imag,
                                                               x.real * y.imag + x.imag * y.real}; };
    auto z_div = [](auto x, auto y) {
        const double den = y.real * y.real + y.imag * y.imag;
        return matx_complex_d_t{(x.real * y.real + x.imag * y.imag) / den,
                                (x.imag * y.real - x.real * y.imag) / den};
    };
    run_z_binary("core_vec_add_c64", [](const auto* al, auto x, auto y, auto* z) { return matx_vec_z_i8_add(al, x, y, z); }, z_add);
    run_z_binary("core_vec_sub_c64", [](const auto* al, auto x, auto y, auto* z) { return matx_vec_z_i8_sub(al, x, y, z); }, z_sub);
    run_z_binary("core_vec_mul_c64", [](const auto* al, auto x, auto y, auto* z) { return matx_vec_z_i8_mul(al, x, y, z); }, z_mul);
    run_z_binary("core_vec_div_c64", [](const auto* al, auto x, auto y, auto* z) { return matx_vec_z_i8_div(al, x, y, z); }, z_div);

    std::vector<double> original_d((size_t) n);
    std::vector<matx_complex_d_t> original_z((size_t) n);
    for (matx_int64_t i = 0; i < n; ++i) {
        original_d[(size_t) i] = a->data[i];
        original_z[(size_t) i] = za->data[i];
    }
    auto reset_d = [&]() { std::copy(original_d.begin(), original_d.end(), a->data); };
    auto reset_z = [&]() { std::copy(original_z.begin(), original_z.end(), za->data); };
    auto scalar_d_add = [&]() { check_status(matx_vec_d_i8_add_scalar(a, 0.5), "real vector scalar add"); };
    auto scalar_d_mul = [&]() { check_status(matx_vec_d_i8_mul_scalar(a, 0.5), "real vector scalar multiply"); };
    auto scalar_z_add = [&]() { check_status(matx_vec_z_i8_add_scalar(za, {0.5, -0.25}), "complex vector scalar add"); };
    auto scalar_z_mul = [&]() { check_status(matx_vec_z_i8_mul_scalar(za, {0.5, -0.25}), "complex vector scalar multiply"); };
    const Timing scalar_d_add_timing = measure_reinitialized(reset_d, scalar_d_add);
    const Timing scalar_d_mul_timing = measure_reinitialized(reset_d, scalar_d_mul);
    const Timing scalar_z_add_timing = measure_reinitialized(reset_z, scalar_z_add);
    const Timing scalar_z_mul_timing = measure_reinitialized(reset_z, scalar_z_mul);
    if (std::abs(a->data[0] - original_d[0] * 0.5) > 1e-12)
        throw std::runtime_error("real vector scalar multiply reference check failed");
    const auto z0 = za->data[0];
    const auto z0_expected = matx_complex_d_t{original_z[0].real * 0.5 - original_z[0].imag * -0.25,
                                               original_z[0].real * -0.25 + original_z[0].imag * 0.5};
    if (std::hypot(z0.real - z0_expected.real, z0.imag - z0_expected.imag) > 1e-12)
        throw std::runtime_error("complex vector scalar multiply reference check failed");
    csv_row("core_vec_add_scalar_f64", "f64", "CORE", n, 1, 0, 0, 0.0, 1,
            0.0, scalar_d_add_timing, 0.0, 0.0, 2.0 * n * sizeof(double));
    csv_row("core_vec_mul_scalar_f64", "f64", "CORE", n, 1, 0, 0, 0.0, 1,
            0.0, scalar_d_mul_timing, 0.0, 0.0, 2.0 * n * sizeof(double));
    csv_row("core_vec_add_scalar_c64", "c64", "CORE", n, 1, 0, 0, 0.0, 1,
            0.0, scalar_z_add_timing, 0.0, 0.0, 2.0 * n * sizeof(matx_complex_d_t));
    csv_row("core_vec_mul_scalar_c64", "c64", "CORE", n, 1, 0, 0, 0.0, 1,
            0.0, scalar_z_mul_timing, 0.0, 0.0, 2.0 * n * sizeof(matx_complex_d_t));

    auto random_uniform = [&]() {
        check_status(matx_vec_d_i8_rand_uniform(a, -1.0, 1.0, 277),
                     "vector uniform random fill");
    };
    auto random_normal = [&]() {
        check_status(matx_vec_d_i8_rand_normal(a, 0.5, 0.25, 281),
                     "vector normal random fill");
    };
    const Timing random_uniform_timing = measure(random_uniform);
    for (matx_int64_t i = 0; i < n; ++i) {
        if (a->data[i] < -1.0 || a->data[i] > 1.0)
            throw std::runtime_error("vector uniform random range check failed");
    }
    const Timing random_normal_timing = measure(random_normal);
    for (matx_int64_t i = 0; i < n; ++i) {
        if (!std::isfinite(a->data[i]))
            throw std::runtime_error("vector normal random finite check failed");
    }
    csv_row("core_vec_rand_uniform", "f64", "CORE", n, 1, 0, 0, 0.0, 1,
            0.0, random_uniform_timing, 0.0, 0.0, 2.0 * n * sizeof(double));
    csv_row("core_vec_rand_normal", "f64", "CORE", n, 1, 0, 0, 0.0, 1,
            0.0, random_normal_timing, 0.0, 0.0, 2.0 * n * sizeof(double));

    auto zeros_d = [&]() { check_status(matx_vec_d_i8_zeros(a), "real vector zeros"); };
    auto zeros_z = [&]() { check_status(matx_vec_z_i8_zeros(za), "complex vector zeros"); };
    auto ones_d = [&]() { check_status(matx_vec_d_i8_ones(a), "real vector ones"); };
    const Timing zeros_d_timing = measure(zeros_d);
    const Timing zeros_z_timing = measure(zeros_z);
    const Timing ones_d_timing = measure(ones_d);
    if (a->data[0] != 1.0 || za->data[0].real != 0.0 || za->data[0].imag != 0.0)
        throw std::runtime_error("vector zero/one fill reference check failed");
    csv_row("core_vec_zeros_f64", "f64", "CORE", n, 1, 0, 0, 0.0, 1,
            0.0, zeros_d_timing, 0.0, 0.0, 2.0 * n * sizeof(double));
    csv_row("core_vec_zeros_c64", "c64", "CORE", n, 1, 0, 0, 0.0, 1,
            0.0, zeros_z_timing, 0.0, 0.0, 2.0 * n * sizeof(matx_complex_d_t));
    csv_row("core_vec_ones_f64", "f64", "CORE", n, 1, 0, 0, 0.0, 1,
            0.0, ones_d_timing, 0.0, 0.0, 2.0 * n * sizeof(double));

    auto cumsum = [&]() { check_status(matx_vec_d_i8_cumsum(&alloc, a, &out_d), "vector cumsum"); };
    const Timing cumsum_timing = measure(cumsum);
    double expected_sum = 0.0;
    for (matx_int64_t i = 0; i < n; ++i) expected_sum += a->data[i];
    if (std::abs(out_d->data[n - 1] - expected_sum) > 1e-10)
        throw std::runtime_error("vector cumsum reference check failed");
    csv_row("core_vec_cumsum_f64", "f64", "CORE", n, 1, 0, 0, 0.0, 1,
            0.0, cumsum_timing, (double) n, 0.0, 2.0 * n * sizeof(double));

    auto convert = [&]() { check_status(matx_vec_d_i8_to_z_i8(&alloc, b, &converted), "real to complex vector"); };
    const Timing convert_timing = measure(convert);
    if (converted->data[n - 1].real != b->data[n - 1] || converted->data[n - 1].imag != 0.0)
        throw std::runtime_error("real to complex vector reference check failed");
    csv_row("core_vec_real_to_complex", "f64_to_c64", "CORE", n, 1, 0, 0, 0.0, 1,
            0.0, convert_timing, (double) n, 0.0,
            (double) n * (sizeof(double) + sizeof(matx_complex_d_t)));

    matx_vec_z_i8_destroy(&alloc, converted);
    matx_vec_d_i8_destroy(&alloc, out_d);
    matx_vec_z_i8_destroy(&alloc, zb);
    matx_vec_z_i8_destroy(&alloc, za);
    matx_vec_d_i8_destroy(&alloc, b);
    matx_vec_d_i8_destroy(&alloc, a);
}

void run_core_dense_api_case(matx_int64_t n, matx_layout_t layout)
{
    matx_alloc_t alloc = matx_alloc_default();
    matx_dense_d_i8_t a = nullptr, duplicate = nullptr, block = nullptr, diagonal_matrix = nullptr;
    matx_dense_z_i8_t za = nullptr, zduplicate = nullptr, zblock = nullptr, zdiagonal_matrix = nullptr;
    matx_dense_z_i8_t converted = nullptr;
    matx_vec_d_i8_t diagonal = nullptr, extracted_diagonal = nullptr;
    matx_vec_z_i8_t zdiagonal = nullptr, zextracted_diagonal = nullptr;
    check_status(matx_dense_d_i8_create(&alloc, &a, layout, n, n, nullptr), "create core API dense real input");
    check_status(matx_dense_z_i8_create(&alloc, &za, layout, n, n, nullptr), "create core API dense complex input");
    check_status(matx_vec_d_i8_create(&alloc, &diagonal, nullptr, n), "create core API real diagonal vector");
    check_status(matx_vec_z_i8_create(&alloc, &zdiagonal, nullptr, n), "create core API complex diagonal vector");
    auto index = [=](matx_int64_t row, matx_int64_t col, matx_int64_t stride) {
        return layout == MATX_COL_MAJOR ? row + col * stride : row * stride + col;
    };
    double expected_trace = 0.0;
    matx_complex_d_t expected_ztrace{0.0, 0.0};
    for (matx_int64_t row = 0; row < n; ++row) {
        diagonal->data[row] = 1.0 + row;
        zdiagonal->data[row] = {1.0 + row, -0.25 * row};
        for (matx_int64_t col = 0; col < n; ++col) {
            const double real = 1.0 + row + col * 0.01;
            a->data[index(row, col, a->stride)] = real;
            za->data[index(row, col, za->stride)] = {real, -0.5 * real};
            if (row == col) {
                expected_trace += real;
                expected_ztrace.real += real;
                expected_ztrace.imag += -0.5 * real;
            }
        }
    }
    const char* layout_name = layout == MATX_COL_MAJOR ? "COL_MAJOR" : "ROW_MAJOR";
    const double setup = 0.0;
    const matx_int64_t half = std::max<matx_int64_t>(1, n / 2);
    auto duplicate_d = [&]() { check_status(matx_dense_d_i8_dup(&alloc, a, &duplicate), "dense real duplicate"); };
    auto duplicate_z = [&]() { check_status(matx_dense_z_i8_dup(&alloc, za, &zduplicate), "dense complex duplicate"); };
    const Timing duplicate_d_timing = measure(duplicate_d);
    const Timing duplicate_z_timing = measure(duplicate_z);
    if (duplicate->data[index(n - 1, n - 1, duplicate->stride)]
            != a->data[index(n - 1, n - 1, a->stride)]
        || std::hypot(zduplicate->data[index(n - 1, n - 1, zduplicate->stride)].real
                          - za->data[index(n - 1, n - 1, za->stride)].real,
                      zduplicate->data[index(n - 1, n - 1, zduplicate->stride)].imag
                          - za->data[index(n - 1, n - 1, za->stride)].imag) > 1e-12)
        throw std::runtime_error("dense duplicate reference check failed");
    csv_row("core_dense_dup", "f64", "CORE", n, n, 0, 0, 0.0, 1,
            setup, duplicate_d_timing, 0.0, 0.0, 2.0 * n * n * sizeof(double), layout_name);
    csv_row("core_dense_dup", "c64", "CORE", n, n, 0, 0, 0.0, 1,
            setup, duplicate_z_timing, 0.0, 0.0, 2.0 * n * n * sizeof(matx_complex_d_t), layout_name);

    double trace = 0.0;
    matx_complex_d_t ztrace{0.0, 0.0};
    auto trace_d = [&]() { check_status(matx_dense_d_i8_trace(a, &trace), "real dense trace"); };
    auto trace_z = [&]() { check_status(matx_dense_z_i8_trace(za, &ztrace), "complex dense trace"); };
    const Timing trace_d_timing = measure(trace_d);
    const Timing trace_z_timing = measure(trace_z);
    if (std::abs(trace - expected_trace) > 1e-10
        || std::hypot(ztrace.real - expected_ztrace.real, ztrace.imag - expected_ztrace.imag) > 1e-10)
        throw std::runtime_error("dense trace reference check failed");
    csv_row("core_dense_trace", "f64", "CORE", n, n, 0, 0, 0.0, 1,
            setup, trace_d_timing, (double) n, 0.0, (double) n * sizeof(double), layout_name);
    csv_row("core_dense_trace", "c64", "CORE", n, n, 0, 0, 0.0, 1,
            setup, trace_z_timing, 2.0 * n, 0.0, (double) n * sizeof(matx_complex_d_t), layout_name);

    auto convert = [&]() { check_status(matx_dense_d_i8_to_z_i8(&alloc, a, &converted), "dense real to complex"); };
    const Timing convert_timing = measure(convert);
    const auto converted_value = converted->data[index(n - 1, n - 1, converted->stride)];
    if (converted_value.real != a->data[index(n - 1, n - 1, a->stride)] || converted_value.imag != 0.0)
        throw std::runtime_error("dense real to complex reference check failed");
    csv_row("core_dense_real_to_complex", "f64_to_c64", "CORE", n, n, 0, 0, 0.0, 1,
            setup, convert_timing, (double) n * n, 0.0,
            (double) n * n * (sizeof(double) + sizeof(matx_complex_d_t)), layout_name);

    auto create_diag_d = [&]() { check_status(matx_diag_d_i8_create(&alloc, diagonal, &diagonal_matrix), "real diagonal matrix create"); };
    auto create_diag_z = [&]() { check_status(matx_diag_z_i8_create(&alloc, zdiagonal, &zdiagonal_matrix), "complex diagonal matrix create"); };
    auto get_diag_d = [&]() { check_status(matx_dense_d_i8_get_diag(&alloc, a, &extracted_diagonal), "real dense get diagonal"); };
    auto get_diag_z = [&]() { check_status(matx_dense_z_i8_get_diag(&alloc, za, &zextracted_diagonal), "complex dense get diagonal"); };
    const Timing create_diag_d_timing = measure(create_diag_d);
    const Timing create_diag_z_timing = measure(create_diag_z);
    const Timing get_diag_d_timing = measure(get_diag_d);
    const Timing get_diag_z_timing = measure(get_diag_z);
    if (diagonal_matrix->data[n - 1 + (n - 1) * diagonal_matrix->stride] != diagonal->data[n - 1]
        || zdiagonal_matrix->data[n - 1 + (n - 1) * zdiagonal_matrix->stride].imag != zdiagonal->data[n - 1].imag
        || extracted_diagonal->data[n - 1] != a->data[index(n - 1, n - 1, a->stride)]
        || std::hypot(zextracted_diagonal->data[n - 1].real - za->data[index(n - 1, n - 1, za->stride)].real,
                      zextracted_diagonal->data[n - 1].imag - za->data[index(n - 1, n - 1, za->stride)].imag) > 1e-12)
        throw std::runtime_error("dense diagonal API reference check failed");
    csv_row("core_diag_create", "f64", "CORE", n, n, 0, 0, 0.0, 1,
            setup, create_diag_d_timing, (double) n, 0.0, (double) n * n * sizeof(double), layout_name);
    csv_row("core_diag_create", "c64", "CORE", n, n, 0, 0, 0.0, 1,
            setup, create_diag_z_timing, 2.0 * n, 0.0, (double) n * n * sizeof(matx_complex_d_t), layout_name);
    csv_row("core_dense_get_diag", "f64", "CORE", n, n, 0, 0, 0.0, 1,
            setup, get_diag_d_timing, (double) n, 0.0, (double) n * sizeof(double), layout_name);
    csv_row("core_dense_get_diag", "c64", "CORE", n, n, 0, 0, 0.0, 1,
            setup, get_diag_z_timing, 2.0 * n, 0.0, (double) n * sizeof(matx_complex_d_t), layout_name);

    auto get_block_d = [&]() { check_status(matx_dense_d_i8_get_block(&alloc, a, 0, half, 0, half, &block), "real dense get block"); };
    auto get_block_z = [&]() { check_status(matx_dense_z_i8_get_block(&alloc, za, 0, half, 0, half, &zblock), "complex dense get block"); };
    const Timing get_block_d_timing = measure(get_block_d);
    const Timing get_block_z_timing = measure(get_block_z);
    const double expected_block = a->data[index(half - 1, half - 1, a->stride)];
    if (block->data[half - 1 + (half - 1) * block->stride] != expected_block
        || std::hypot(zblock->data[half - 1 + (half - 1) * zblock->stride].real
                          - za->data[index(half - 1, half - 1, za->stride)].real,
                      zblock->data[half - 1 + (half - 1) * zblock->stride].imag
                          - za->data[index(half - 1, half - 1, za->stride)].imag) > 1e-12)
        throw std::runtime_error("dense get block reference check failed");
    csv_row("core_dense_get_block", "f64", "CORE", half, half, 0, 0, 0.0, 1,
            setup, get_block_d_timing, 0.0, 0.0, 2.0 * half * half * sizeof(double), layout_name);
    csv_row("core_dense_get_block", "c64", "CORE", half, half, 0, 0, 0.0, 1,
            setup, get_block_z_timing, 0.0, 0.0, 2.0 * half * half * sizeof(matx_complex_d_t), layout_name);

    matx_dense_d_i8_t destination = nullptr;
    matx_dense_z_i8_t zdestination = nullptr;
    check_status(matx_dense_d_i8_create(&alloc, &destination, layout, n, n, nullptr), "create real block destination");
    check_status(matx_dense_z_i8_create(&alloc, &zdestination, layout, n, n, nullptr), "create complex block destination");
    auto reset_destination = [&]() { check_status(matx_dense_d_i8_zeros(destination), "reset real block destination"); };
    auto reset_zdestination = [&]() { check_status(matx_dense_z_i8_zeros(zdestination), "reset complex block destination"); };
    auto set_block_d = [&]() { check_status(matx_dense_d_i8_set_block(a, 0, half, 0, half, destination, 1, 1), "real dense set block"); };
    auto set_block_z = [&]() { check_status(matx_dense_z_i8_set_block(za, 0, half, 0, half, zdestination, 1, 1), "complex dense set block"); };
    const Timing set_block_d_timing = measure_reinitialized(reset_destination, set_block_d);
    const Timing set_block_z_timing = measure_reinitialized(reset_zdestination, set_block_z);
    if (destination->data[index(half, half, destination->stride)] != a->data[index(half - 1, half - 1, a->stride)]
        || std::hypot(zdestination->data[index(half, half, zdestination->stride)].real
                          - za->data[index(half - 1, half - 1, za->stride)].real,
                      zdestination->data[index(half, half, zdestination->stride)].imag
                          - za->data[index(half - 1, half - 1, za->stride)].imag) > 1e-12)
        throw std::runtime_error("dense set block reference check failed");
    csv_row("core_dense_set_block", "f64", "CORE", half, half, 0, 0, 0.0, 1,
            setup, set_block_d_timing, 0.0, 0.0, 2.0 * half * half * sizeof(double), layout_name);
    csv_row("core_dense_set_block", "c64", "CORE", half, half, 0, 0, 0.0, 1,
            setup, set_block_z_timing, 0.0, 0.0, 2.0 * half * half * sizeof(matx_complex_d_t), layout_name);

    matx_dense_z_i8_destroy(&alloc, zdestination);
    matx_dense_d_i8_destroy(&alloc, destination);
    matx_dense_z_i8_destroy(&alloc, zblock);
    matx_dense_d_i8_destroy(&alloc, block);
    matx_dense_z_i8_destroy(&alloc, zdiagonal_matrix);
    matx_dense_d_i8_destroy(&alloc, diagonal_matrix);
    matx_vec_z_i8_destroy(&alloc, zextracted_diagonal);
    matx_vec_d_i8_destroy(&alloc, extracted_diagonal);
    matx_vec_z_i8_destroy(&alloc, zdiagonal);
    matx_vec_d_i8_destroy(&alloc, diagonal);
    matx_dense_z_i8_destroy(&alloc, converted);
    matx_dense_z_i8_destroy(&alloc, zduplicate);
    matx_dense_d_i8_destroy(&alloc, duplicate);
    matx_dense_z_i8_destroy(&alloc, za);
    matx_dense_d_i8_destroy(&alloc, a);
}

void run_core_lifecycle_api_case(matx_int64_t n)
{
    matx_alloc_t alloc = matx_alloc_default();
    matx_vec_d_i8_t vd = nullptr, vd_dup = nullptr, vd_wrap = nullptr;
    matx_vec_z_i8_t vz = nullptr, vz_dup = nullptr, vz_wrap = nullptr;
    matx_dense_d_i8_t md = nullptr, md_dup = nullptr, md_wrap = nullptr;
    matx_dense_z_i8_t mz = nullptr, mz_dup = nullptr, mz_wrap = nullptr;
    matx_coo_d_i8_t sd = nullptr, sd_dup = nullptr;
    matx_coo_z_i8_t sz = nullptr, sz_dup = nullptr, sz_wrap = nullptr;
    check_status(matx_vec_d_i8_create(&alloc, &vd, nullptr, n), "create duplicate real vector input");
    check_status(matx_vec_z_i8_create(&alloc, &vz, nullptr, n), "create duplicate complex vector input");
    check_status(matx_dense_d_i8_create(&alloc, &md, MATX_COL_MAJOR, n, n, nullptr), "create duplicate real dense input");
    check_status(matx_dense_z_i8_create(&alloc, &mz, MATX_COL_MAJOR, n, n, nullptr), "create duplicate complex dense input");
    matx_int64_t rows[3] = {0, 1, 2};
    matx_int64_t cols[3] = {0, 1, 2};
    double vals_d[3] = {1.0, 2.0, 3.0};
    matx_complex_d_t vals_z[3] = {{1.0, 0.5}, {2.0, -0.5}, {3.0, 1.0}};
    check_status(matx_coo_sparse_d_i8_create(&alloc, &sd, 3, 3, 3, rows, cols, vals_d),
                 "create duplicate real COO input");
    check_status(matx_coo_sparse_z_i8_create(&alloc, &sz, 3, 3, 3, rows, cols, vals_z),
                 "create duplicate complex COO input");
    std::vector<double> vd_data((size_t) n, 1.25);
    std::vector<matx_complex_d_t> vz_data((size_t) n, {1.25, -0.5});
    std::vector<double> md_data((size_t) n * (size_t) n, 2.5);
    std::vector<matx_complex_d_t> mz_data((size_t) n * (size_t) n, {2.5, 0.75});

    auto duplicate_vd = [&]() { check_status(matx_vec_d_i8_dup(&alloc, vd, &vd_dup), "real vector duplicate"); };
    auto duplicate_vz = [&]() { check_status(matx_vec_z_i8_dup(&alloc, vz, &vz_dup), "complex vector duplicate"); };
    auto duplicate_md = [&]() { check_status(matx_dense_d_i8_dup(&alloc, md, &md_dup), "real dense duplicate"); };
    auto duplicate_mz = [&]() { check_status(matx_dense_z_i8_dup(&alloc, mz, &mz_dup), "complex dense duplicate"); };
    auto duplicate_sd = [&]() { check_status(matx_coo_d_i8_dup(&alloc, sd, &sd_dup), "real COO duplicate"); };
    auto duplicate_sz = [&]() { check_status(matx_coo_z_i8_dup(&alloc, sz, &sz_dup), "complex COO duplicate"); };
    const Timing duplicate_vd_timing = measure(duplicate_vd);
    const Timing duplicate_vz_timing = measure(duplicate_vz);
    const Timing duplicate_md_timing = measure(duplicate_md);
    const Timing duplicate_mz_timing = measure(duplicate_mz);
    const Timing duplicate_sd_timing = measure(duplicate_sd);
    const Timing duplicate_sz_timing = measure(duplicate_sz);
    if (vd_dup->data[n - 1] != vd->data[n - 1]
        || std::hypot(vz_dup->data[n - 1].real - vz->data[n - 1].real,
                      vz_dup->data[n - 1].imag - vz->data[n - 1].imag) > 1e-12
        || md_dup->data[n * n - 1] != md->data[n * n - 1]
        || sd_dup->nnz != sd->nnz || sz_dup->nnz != sz->nnz)
        throw std::runtime_error("container duplicate reference check failed");
    csv_row("core_vector_dup", "f64", "ALLOCATOR", n, 1, 0, 0, 0.0, 1,
            0.0, duplicate_vd_timing, 0.0, 0.0, 2.0 * n * sizeof(double));
    csv_row("core_vector_dup", "c64", "ALLOCATOR", n, 1, 0, 0, 0.0, 1,
            0.0, duplicate_vz_timing, 0.0, 0.0, 2.0 * n * sizeof(matx_complex_d_t));
    csv_row("core_dense_dup", "f64", "ALLOCATOR", n, n, 0, 0, 0.0, 1,
            0.0, duplicate_md_timing, 0.0, 0.0, 2.0 * n * n * sizeof(double));
    csv_row("core_dense_dup", "c64", "ALLOCATOR", n, n, 0, 0, 0.0, 1,
            0.0, duplicate_mz_timing, 0.0, 0.0, 2.0 * n * n * sizeof(matx_complex_d_t));
    csv_row("core_sparse_dup", "f64", "ALLOCATOR", 3, 3, 0, 3, 1.0 / 3.0, 1,
            0.0, duplicate_sd_timing, 0.0, 0.0, 2.0 * 3 * (2 * sizeof(matx_int64_t) + sizeof(double)));
    csv_row("core_sparse_dup", "c64", "ALLOCATOR", 3, 3, 0, 3, 1.0 / 3.0, 1,
            0.0, duplicate_sz_timing, 0.0, 0.0, 2.0 * 3 * (2 * sizeof(matx_int64_t) + sizeof(matx_complex_d_t)));

    auto wrap_vd = [&]() {
        vd_wrap = nullptr;
        check_status(matx_vec_d_i8_wrap(&alloc, &vd_wrap, n, 1, vd_data.data()), "real vector wrap");
        matx_vec_d_i8_destroy(&alloc, vd_wrap);
    };
    auto wrap_vz = [&]() {
        vz_wrap = nullptr;
        check_status(matx_vec_z_i8_wrap(&alloc, &vz_wrap, n, 1, vz_data.data()), "complex vector wrap");
        matx_vec_z_i8_destroy(&alloc, vz_wrap);
    };
    auto wrap_md = [&]() {
        md_wrap = nullptr;
        check_status(matx_dense_d_i8_wrap(&alloc, &md_wrap, n, n, n, MATX_COL_MAJOR, md_data.data()),
                     "real dense wrap");
        matx_dense_d_i8_destroy(&alloc, md_wrap);
    };
    auto wrap_mz = [&]() {
        mz_wrap = nullptr;
        check_status(matx_dense_z_i8_wrap(&alloc, &mz_wrap, n, n, n, MATX_COL_MAJOR, mz_data.data()),
                     "complex dense wrap");
        matx_dense_z_i8_destroy(&alloc, mz_wrap);
    };
    auto wrap_sz = [&]() {
        sz_wrap = nullptr;
        check_status(matx_coo_sparse_z_i8_wrap(&alloc, &sz_wrap, 3, 3, 3, rows, cols, vals_z),
                     "complex COO wrap");
        matx_coo_sparse_z_i8_destroy(&alloc, sz_wrap);
    };
    const Timing wrap_vd_timing = measure(wrap_vd);
    const Timing wrap_vz_timing = measure(wrap_vz);
    const Timing wrap_md_timing = measure(wrap_md);
    const Timing wrap_mz_timing = measure(wrap_mz);
    const Timing wrap_sz_timing = measure(wrap_sz);
    csv_row("core_vector_wrap", "f64", "ALLOCATOR", n, 1, 0, 0, 0.0, 1,
            0.0, wrap_vd_timing, 0.0, 0.0, 0.0);
    csv_row("core_vector_wrap", "c64", "ALLOCATOR", n, 1, 0, 0, 0.0, 1,
            0.0, wrap_vz_timing, 0.0, 0.0, 0.0);
    csv_row("core_dense_wrap", "f64", "ALLOCATOR", n, n, 0, 0, 0.0, 1,
            0.0, wrap_md_timing, 0.0, 0.0, 0.0, "COL_MAJOR");
    csv_row("core_dense_wrap", "c64", "ALLOCATOR", n, n, 0, 0, 0.0, 1,
            0.0, wrap_mz_timing, 0.0, 0.0, 0.0, "COL_MAJOR");
    csv_row("core_sparse_wrap", "c64", "ALLOCATOR", 3, 3, 0, 3, 1.0 / 3.0, 1,
            0.0, wrap_sz_timing, 0.0, 0.0, 0.0);

    matx_csc_d_i8_t csc_d = nullptr, csc_d_wrap = nullptr;
    matx_csc_z_i8_t csc_z = nullptr;
    std::vector<matx_int64_t> col_ptr((size_t) n + 1, 3);
    col_ptr[0] = 0;
    if (n > 0) col_ptr[1] = 1;
    if (n > 1) col_ptr[2] = 2;
    for (matx_int64_t i = 3; i <= n; ++i) col_ptr[(size_t) i] = 3;
    matx_int64_t csc_rows[3] = {0, 1, 2};
    const double csc_values[3] = {1.0, 2.0, 3.0};
    auto create_csc_d = [&]() {
        csc_d = nullptr;
        check_status(matx_csc_sparse_d_i8_create(&alloc, &csc_d, n, n, 3), "real CSC create");
        matx_csc_sparse_d_i8_destroy(&alloc, csc_d);
    };
    auto create_csc_z = [&]() {
        csc_z = nullptr;
        check_status(matx_csc_sparse_z_i8_create(&alloc, &csc_z, n, n, 3), "complex CSC create");
        matx_csc_sparse_z_i8_destroy(&alloc, csc_z);
    };
    auto wrap_csc_d = [&]() {
        csc_d_wrap = nullptr;
        check_status(matx_csc_sparse_d_i8_wrap(&alloc, &csc_d_wrap, n, n, 3,
                                                col_ptr.data(), csc_rows, csc_values),
                     "real CSC wrap");
        matx_csc_sparse_d_i8_destroy(&alloc, csc_d_wrap);
    };
    const Timing csc_d_timing = measure(create_csc_d);
    const Timing csc_z_timing = measure(create_csc_z);
    const Timing csc_wrap_timing = measure(wrap_csc_d);
    csv_row("core_csc_create_destroy", "f64", "ALLOCATOR", n, n, 0, 3, 3.0 / (n * n), 1,
            0.0, csc_d_timing, 0.0, 0.0, 3.0 * (2 * sizeof(matx_int64_t) + sizeof(double)));
    csv_row("core_csc_create_destroy", "c64", "ALLOCATOR", n, n, 0, 3, 3.0 / (n * n), 1,
            0.0, csc_z_timing, 0.0, 0.0, 3.0 * (2 * sizeof(matx_int64_t) + sizeof(matx_complex_d_t)));
    csv_row("core_csc_wrap_destroy", "f64", "ALLOCATOR", n, n, 0, 3, 3.0 / (n * n), 1,
            0.0, csc_wrap_timing, 0.0, 0.0, 0.0);

    auto allocate_free = [&]() {
        void* memory = matx_malloc(&alloc, (size_t) n * sizeof(double));
        if (!memory) throw std::runtime_error("MatX allocator returned null");
        matx_free(&alloc, memory);
    };
    const Timing allocator_timing = measure(allocate_free);
    csv_row("core_malloc_free", "f64", "ALLOCATOR", n, 1, 0, 0, 0.0, 1,
            0.0, allocator_timing, 0.0, 0.0, 2.0 * n * sizeof(double));

    matx_vec_z_i8_destroy(&alloc, vz_dup);
    matx_vec_d_i8_destroy(&alloc, vd_dup);
    matx_coo_sparse_z_i8_destroy(&alloc, sz_dup);
    matx_coo_sparse_d_i8_destroy(&alloc, sd_dup);
    matx_dense_z_i8_destroy(&alloc, mz_dup);
    matx_dense_d_i8_destroy(&alloc, md_dup);
    matx_coo_sparse_z_i8_destroy(&alloc, sz);
    matx_coo_sparse_d_i8_destroy(&alloc, sd);
    matx_dense_z_i8_destroy(&alloc, mz);
    matx_dense_d_i8_destroy(&alloc, md);
    matx_vec_z_i8_destroy(&alloc, vz);
    matx_vec_d_i8_destroy(&alloc, vd);
}

void run_system_api_case(const std::string& log_directory)
{
    const char* status = nullptr;
    auto status_string = [&]() { status = matx_status_string(MATX_OK); };
    const Timing status_timing = measure(status_string);
    const char* version = nullptr;
    auto version_string = [&]() { version = matx_version_string(); };
    const Timing version_timing = measure(version_string);
    if (!status || !version)
        throw std::runtime_error("MatX diagnostic API returned null");
    csv_row("system_status_string", "N/A", "CORE", 0, 0, 0, 0, 0.0, 1,
            0.0, status_timing, 0.0, 0.0);
    csv_row("system_version_string", "N/A", "TYPES", 0, 0, 0, 0, 0.0, 1,
            0.0, version_timing, 0.0, 0.0);

    matx_sparse_backend_t sparse_backend{};
    auto sparse_select = [&]() {
        sparse_backend = matx_sparse_by_type(MATX_SPARSE_BACKEND_GRAPHBLAS);
    };
    const Timing sparse_select_timing = measure(sparse_select);
    if (sparse_backend.kind != MATX_SPARSE_BACKEND_GRAPHBLAS)
        throw std::runtime_error("GraphBLAS backend selection returned the wrong backend");
    csv_row("system_sparse_backend_select", "N/A", "GRAPHBLAS", 0, 0, 0, 0,
            0.0, 1, 0.0, sparse_select_timing, 0.0, 0.0);

    struct TimeCase { matx_tm_unit unit; const char* name; };
    const TimeCase time_cases[] = {
        {MATX_TM_SECOND, "system_time_seconds"},
        {MATX_TM_MILLISECOND, "system_time_milliseconds"},
        {MATX_TM_MICROSECOND, "system_time_microseconds"},
        {MATX_TM_NANOSECOND, "system_time_nanoseconds"},
    };
    for (const auto& time_case : time_cases) {
        matx_int64_t value = 0;
        auto read_clock = [&]() { value = matx_tm_now(time_case.unit); };
        const Timing timing = measure(read_clock);
        if (value <= 0) throw std::runtime_error("MatX clock API returned a non-positive time");
        csv_row(time_case.name, "N/A", "TOOLS", 0, 0, 0, 0, 0.0, 1,
                0.0, timing, 0.0, 0.0);
    }

    PerfEventGroup log_init_counters;
    const bool count_log_init = log_init_counters.start();
    const auto log_start = Clock::now();
    matx_log_init(log_directory.c_str());
    const double log_init_us = elapsed_us(log_start);
    Timing log_init_timing{log_init_us, log_init_us, 1};
    if (count_log_init) {
        log_init_timing.perf = log_init_counters.stop(1);
        log_init_timing.perf_scope = log_init_counters.scope();
    }
    csv_row("system_log_init", "N/A", "TOOLS", 0, 0, 0, 0, 0.0, 1,
            0.0, log_init_timing, 0.0, 0.0);

    auto suppress_logs = []() { matx_log_set_level(static_cast<matx_log_level>(6)); };
    const Timing set_level_timing = measure(suppress_logs);
    csv_row("system_log_set_level", "N/A", "TOOLS", 0, 0, 0, 0, 0.0, 1,
            0.0, set_level_timing, 0.0, 0.0);

    auto log_trace = []() { matx_log_trace(__FILE__, __LINE__, "MatX API benchmark"); };
    auto log_debug = []() { matx_log_debug(__FILE__, __LINE__, "MatX API benchmark"); };
    auto log_info = []() { matx_log_info(__FILE__, __LINE__, "MatX API benchmark"); };
    auto log_warn = []() { matx_log_warn(__FILE__, __LINE__, "MatX API benchmark"); };
    auto log_error = []() { matx_log_error(__FILE__, __LINE__, "MatX API benchmark"); };
    auto log_fatal = []() { matx_log_fatal(__FILE__, __LINE__, "MatX API benchmark"); };
    const std::pair<const char*, Timing> log_cases[] = {
        {"system_log_trace", measure(log_trace)},
        {"system_log_debug", measure(log_debug)},
        {"system_log_info", measure(log_info)},
        {"system_log_warn", measure(log_warn)},
        {"system_log_error", measure(log_error)},
        {"system_log_fatal", measure(log_fatal)},
    };
    for (const auto& log_case : log_cases)
        csv_row(log_case.first, "N/A", "TOOLS", 0, 0, 0, 0, 0.0, 1,
                0.0, log_case.second, 0.0, 0.0);
}

template <typename T>
void run_io_dense_case(matx_int64_t n)
{
    using O = Ops<T>;
    using Dense = typename O::Dense;
    matx_alloc_t alloc = matx_alloc_default();
    Dense matrix = nullptr;
    const auto setup_start = Clock::now();
    check_status(O::dense_create(&alloc, &matrix, n, n), "create Matrix Market dense input");
    for (matx_int64_t row = 0; row < n; ++row) {
        for (matx_int64_t col = 0; col < n; ++col) {
            matrix->data[row + col * matrix->stride]
                = generated_value<T>((uint64_t) row + (uint64_t) col * n, 173);
        }
    }
    const double setup = elapsed_us(setup_start);
    const std::string path = benchmark_file_path("dense_" + std::string(O::name));
    auto write = [&]() { O::dense_write(matrix, path.c_str()); };
    const Timing write_timing = measure(write);
    const double bytes = (double) std::filesystem::file_size(path);

    auto read = [&]() {
        Dense loaded = nullptr;
        check_status(O::dense_read(&alloc, &loaded, path.c_str()), "read Matrix Market dense input");
        O::dense_destroy(&alloc, loaded);
    };
    const Timing read_timing = measure(read);

    Dense loaded = nullptr;
    check_status(O::dense_read(&alloc, &loaded, path.c_str()), "verify Matrix Market dense input");
    double max_error = 0.0;
    for (matx_int64_t row = 0; row < n; ++row) {
        for (matx_int64_t col = 0; col < n; ++col) {
            const T expected = matrix->data[row + col * matrix->stride];
            max_error = std::max(max_error,
                                 O::difference(loaded->data[row + col * loaded->stride], expected));
        }
    }
    if (max_error > 1e-12) throw std::runtime_error("dense Matrix Market roundtrip failed");

    csv_row("io_dense_write", O::name, "MTX", n, n, 0, 0, 0.0, 1,
            setup, write_timing, 0.0, max_error, bytes);
    csv_row("io_dense_read", O::name, "MTX", n, n, 0, 0, 0.0, 1,
            setup, read_timing, 0.0, max_error, bytes);
    O::dense_destroy(&alloc, loaded);
    O::dense_destroy(&alloc, matrix);
    std::filesystem::remove(path);
}

template <typename T>
void run_io_vector_case(matx_int64_t n)
{
    using O = Ops<T>;
    using Vector = typename O::Vector;
    matx_alloc_t alloc = matx_alloc_default();
    Vector vector = nullptr;
    const auto setup_start = Clock::now();
    check_status(O::vector_create(&alloc, &vector, n), "create Matrix Market vector input");
    for (matx_int64_t i = 0; i < n; ++i)
        vector->data[i * vector->stride] = generated_value<T>((uint64_t) i, 167);
    const double setup = elapsed_us(setup_start);
    const std::string path = benchmark_file_path("vector_" + std::string(O::name));
    auto write = [&]() { O::vector_write(vector, path.c_str()); };
    const Timing write_timing = measure(write);
    const double bytes = (double) std::filesystem::file_size(path);

    auto read = [&]() {
        Vector loaded = nullptr;
        check_status(O::vector_read(&alloc, &loaded, path.c_str()),
                     "read Matrix Market vector input");
        O::vector_destroy(&alloc, loaded);
    };
    const Timing read_timing = measure(read);

    Vector loaded = nullptr;
    check_status(O::vector_read(&alloc, &loaded, path.c_str()),
                 "verify Matrix Market vector input");
    if (loaded->n != n)
        throw std::runtime_error("Matrix Market vector length changed");
    double max_error = 0.0;
    for (matx_int64_t i = 0; i < n; ++i)
        max_error = std::max(max_error,
                             O::difference(loaded->data[i * loaded->stride],
                                           vector->data[i * vector->stride]));
    if (max_error > 1e-12)
        throw std::runtime_error("Matrix Market vector roundtrip failed");

    csv_row("io_vector_write", O::name, "MTX", n, 1, 0, 0, 0.0, 1,
            setup, write_timing, 0.0, max_error, bytes);
    csv_row("io_vector_read", O::name, "MTX", n, 1, 0, 0, 0.0, 1,
            setup, read_timing, 0.0, max_error, bytes);
    O::vector_destroy(&alloc, loaded);
    O::vector_destroy(&alloc, vector);
    std::filesystem::remove(path);
}

template <typename T>
void run_io_sparse_case(matx_int64_t n)
{
    using O = Ops<T>;
    using Sparse = typename O::Sparse;
    matx_alloc_t alloc = matx_alloc_default();
    std::vector<matx_int64_t> rows, columns;
    std::vector<T> values;
    rows.reserve((size_t) n * 3);
    columns.reserve((size_t) n * 3);
    values.reserve((size_t) n * 3);
    for (matx_int64_t row = 0; row < n; ++row) {
        rows.push_back(row);
        columns.push_back(row);
        values.push_back(generated_value<T>((uint64_t) row, 181));
        if (row > 0) {
            rows.push_back(row);
            columns.push_back(row - 1);
            values.push_back(generated_value<T>((uint64_t) row, 191));
        }
        if (row + 1 < n) {
            rows.push_back(row);
            columns.push_back(row + 1);
            values.push_back(generated_value<T>((uint64_t) row, 199));
        }
    }
    const matx_int64_t nnz = (matx_int64_t) values.size();
    Sparse matrix = nullptr;
    const auto setup_start = Clock::now();
    check_status(O::sparse_create(&alloc, &matrix, n, nnz,
                                  rows.data(), columns.data(), values.data()),
                 "create Matrix Market sparse input");
    const double setup = elapsed_us(setup_start);
    const std::string path = benchmark_file_path("sparse_" + std::string(O::name));
    auto write = [&]() { O::sparse_write(matrix, path.c_str()); };
    const Timing write_timing = measure(write);
    const double bytes = (double) std::filesystem::file_size(path);

    auto read = [&]() {
        Sparse loaded = nullptr;
        check_status(O::sparse_read(&alloc, &loaded, path.c_str()), "read Matrix Market sparse input");
        O::sparse_destroy(&alloc, loaded);
    };
    const Timing read_timing = measure(read);

    Sparse loaded = nullptr;
    check_status(O::sparse_read(&alloc, &loaded, path.c_str()), "verify Matrix Market sparse input");
    double max_error = 0.0;
    if (loaded->nrows != n || loaded->ncols != n || loaded->nnz != nnz)
        throw std::runtime_error("sparse Matrix Market dimensions changed");
    for (matx_int64_t i = 0; i < nnz; ++i) {
        const size_t index = (size_t) i;
        if (loaded->rows[index] != rows[index] || loaded->columns[index] != columns[index])
            throw std::runtime_error("sparse Matrix Market coordinates changed");
        max_error = std::max(max_error, O::difference(loaded->values[index], values[index]));
    }
    if (max_error > 1e-12) throw std::runtime_error("sparse Matrix Market values changed");

    const double density = (double) nnz / ((double) n * (double) n);
    csv_row("io_sparse_write", O::name, "MTX", n, n, 0, nnz, density, 1,
            setup, write_timing, 0.0, max_error, bytes);
    csv_row("io_sparse_read", O::name, "MTX", n, n, 0, nnz, density, 1,
            setup, read_timing, 0.0, max_error, bytes);
    O::sparse_destroy(&alloc, loaded);
    O::sparse_destroy(&alloc, matrix);
    std::filesystem::remove(path);
}

template <typename T>
void run_dense_extended_case(const matx_dense_backend_t& backend, matx_int64_t n)
{
    using O = Ops<T>;
    using Dense = typename O::Dense;
    using Vector = typename O::Vector;
    matx_alloc_t alloc = matx_alloc_default();
    Dense a = nullptr, b = nullptr, c = nullptr, inverse = nullptr, exponential = nullptr;
    Dense triangular = nullptr;
    Vector tri_x = nullptr, ger_x = nullptr, ger_y = nullptr;
    const matx_int64_t k = std::max<matx_int64_t>(2, n / 2);
    auto setup_start = Clock::now();
    check_status(O::dense_create(&alloc, &a, n, n), "create extended dense A");
    check_status(O::dense_create(&alloc, &b, n, k), "create extended dense B");
    check_status(O::dense_create(&alloc, &c, n, n), "create extended dense C");
    check_status(O::dense_create(&alloc, &inverse, n, n), "create extended dense inverse");
    check_status(O::dense_create(&alloc, &triangular, n, n), "create triangular dense matrix");
    check_status(O::vector_create(&alloc, &tri_x, n), "create triangular solve vector");
    check_status(O::vector_create(&alloc, &ger_x, n), "create GER x vector");
    check_status(O::vector_create(&alloc, &ger_y, n), "create GER y vector");
    if constexpr (std::is_same_v<T, double>)
        check_status(O::dense_create(&alloc, &exponential, n, n), "create dense exponential");
    for (matx_int64_t col = 0; col < n; ++col) {
        for (matx_int64_t row = 0; row < n; ++row) {
            const double perturbation
                = seeded_value((uint64_t) row + (uint64_t) col * n, 227) * 1e-3;
            a->data[row + col * a->stride]
                = O::make(row == col ? 2.0 + perturbation : perturbation,
                          row == col ? 0.0 : perturbation * 0.25);
            if constexpr (std::is_same_v<T, double>) {
                if (row == col) a->data[row + col * a->stride] = 0.001 + perturbation * 0.01;
                exponential->data[row + col * exponential->stride] = 0.0;
            }
            const double tri_value = row == col ? 2.0 + perturbation : perturbation * 0.01;
            triangular->data[row + col * triangular->stride]
                = O::make(tri_value, row == col ? 0.0 : tri_value * 0.1);
        }
    }
    for (matx_int64_t col = 0; col < k; ++col) {
        for (matx_int64_t row = 0; row < n; ++row) {
            const uint64_t index = (uint64_t) row + (uint64_t) col * n;
            b->data[row + col * b->stride]
                = O::make(seeded_value(index, 233) * 0.1,
                          seeded_value(index, 250) * 0.025);
        }
    }
    std::vector<T> a_backup((size_t) n * (size_t) n);
    for (matx_int64_t col = 0; col < n; ++col)
        for (matx_int64_t row = 0; row < n; ++row)
            a_backup[(size_t) row + (size_t) col * n] = a->data[row + col * a->stride];
    for (matx_int64_t row = 0; row < n; ++row)
        ger_x->data[row * ger_x->stride] = generated_value<T>((uint64_t) row, 241);
    for (matx_int64_t col = 0; col < n; ++col)
        ger_y->data[col * ger_y->stride] = generated_value<T>((uint64_t) col, 251);
    const double setup = elapsed_us(setup_start);

    auto rankk = [&]() {
        if constexpr (std::is_same_v<T, double>)
            check_status(matx_syrk_d_i8(&backend, MATX_LOWER, MATX_NO_TRANS,
                                         1.0, b, 0.0, c), "SYRK");
        else
            check_status(matx_herk_z_i8(&backend, MATX_LOWER, MATX_NO_TRANS,
                                         1.0, b, 0.0, c), "HERK");
    };
    const Timing rankk_timing = measure(rankk);
    auto rank2k = [&]() {
        if constexpr (std::is_same_v<T, double>)
            check_status(matx_syr2k_d_i8(&backend, MATX_LOWER, MATX_NO_TRANS,
                                          1.0, b, b, 0.0, c), "SYR2K");
        else
            check_status(matx_her2k_z_i8(&backend, MATX_LOWER, MATX_NO_TRANS,
                                          O::one(), b, b, 0.0, c), "HER2K");
    };
    const Timing rank2k_timing = measure(rank2k);

    auto invert = [&]() {
        if constexpr (std::is_same_v<T, double>)
            check_status(matx_inv_dense_d_i8(&backend, a, inverse), "dense inverse");
        else
            check_status(matx_inv_dense_z_i8(&backend, a, inverse), "complex dense inverse");
    };
    const Timing inverse_timing = measure(invert);
    double inverse_error = 0.0;
    for (matx_int64_t col = 0; col < n; ++col) {
        for (matx_int64_t row = 0; row < n; ++row) {
            T value = O::zero();
            for (matx_int64_t inner = 0; inner < n; ++inner)
                value = O::add(value, O::multiply(a->data[row + inner * a->stride],
                                                  inverse->data[inner + col * inverse->stride]));
            const T expected = row == col ? O::one() : O::zero();
            inverse_error = std::max(inverse_error, O::difference(value, expected));
        }
    }
    if (!std::isfinite(inverse_error) || inverse_error > 1e-8)
        throw std::runtime_error("dense inverse reference check failed");

    std::vector<T> trsv_rhs((size_t) n, O::zero());
    for (matx_int64_t row = 0; row < n; ++row)
        for (matx_int64_t col = 0; col <= row; ++col)
            trsv_rhs[(size_t) row] = O::add(trsv_rhs[(size_t) row],
                                             triangular->data[row + col * triangular->stride]);
    auto reset_trsv = [&]() {
        for (matx_int64_t row = 0; row < n; ++row)
            tri_x->data[row * tri_x->stride] = trsv_rhs[(size_t) row];
    };
    auto trsv = [&]() {
        if constexpr (std::is_same_v<T, double>)
            check_status(matx_trsv_d_i8(&backend, MATX_LOWER, MATX_NO_TRANS,
                                         MATX_NON_UNIT_DIAG, triangular, tri_x), "TRSV");
        else
            check_status(matx_trsv_z_i8(&backend, MATX_LOWER, MATX_NO_TRANS,
                                         MATX_NON_UNIT_DIAG, triangular, tri_x), "complex TRSV");
    };
    const Timing trsv_timing = measure_reinitialized(reset_trsv, trsv);
    double trsv_error = 0.0;
    for (matx_int64_t row = 0; row < n; ++row)
        trsv_error = std::max(trsv_error,
                              O::difference(tri_x->data[row * tri_x->stride], O::one()));
    if (trsv_error > 1e-8) throw std::runtime_error("TRSV reference check failed");

    std::vector<T> trsm_rhs((size_t) n * (size_t) k, O::zero());
    for (matx_int64_t col = 0; col < k; ++col) {
        const T expected = O::make((double) (col + 1));
        for (matx_int64_t row = 0; row < n; ++row) {
            T value = O::zero();
            for (matx_int64_t inner = 0; inner <= row; ++inner)
                value = O::add(value, O::multiply(triangular->data[row + inner * triangular->stride], expected));
            trsm_rhs[(size_t) row + (size_t) col * n] = value;
        }
    }
    auto reset_trsm = [&]() {
        for (matx_int64_t col = 0; col < k; ++col)
            for (matx_int64_t row = 0; row < n; ++row)
                b->data[row + col * b->stride] = trsm_rhs[(size_t) row + (size_t) col * n];
    };
    auto trsm = [&]() {
        if constexpr (std::is_same_v<T, double>)
            check_status(matx_trsm_d_i8(&backend, MATX_LEFT, MATX_LOWER, MATX_NO_TRANS,
                                         MATX_NON_UNIT_DIAG, 1.0, triangular, b), "TRSM");
        else
            check_status(matx_trsm_z_i8(&backend, MATX_LEFT, MATX_LOWER, MATX_NO_TRANS,
                                         MATX_NON_UNIT_DIAG, O::one(), triangular, b), "complex TRSM");
    };
    const Timing trsm_timing = measure_reinitialized(reset_trsm, trsm);
    double trsm_error = 0.0;
    for (matx_int64_t col = 0; col < k; ++col)
        for (matx_int64_t row = 0; row < n; ++row)
            trsm_error = std::max(trsm_error,
                                  O::difference(b->data[row + col * b->stride],
                                                O::make((double) (col + 1))));
    if (trsm_error > 1e-8) throw std::runtime_error("TRSM reference check failed");

    auto restore_a = [&]() {
        for (matx_int64_t col = 0; col < n; ++col)
            for (matx_int64_t row = 0; row < n; ++row)
                a->data[row + col * a->stride] = a_backup[(size_t) row + (size_t) col * n];
    };
    auto ger = [&]() {
        if constexpr (std::is_same_v<T, double>)
            check_status(matx_ger_d_i8(&backend, 0.5, ger_x, ger_y, a), "GER");
        else
            check_status(matx_geru_z_i8(&backend, O::make(0.5), ger_x, ger_y, a), "GERU");
    };
    const Timing ger_timing = measure_reinitialized(restore_a, ger);
    double ger_error = 0.0;
    if constexpr (std::is_same_v<T, double>) {
        for (matx_int64_t col = 0; col < n; ++col) {
            for (matx_int64_t row = 0; row < n; ++row) {
                const T update = O::multiply(O::make(0.5),
                                             O::multiply(ger_x->data[row * ger_x->stride],
                                                         ger_y->data[col * ger_y->stride]));
                const T expected = O::add(a_backup[(size_t) row + (size_t) col * n], update);
                ger_error = std::max(ger_error,
                                     O::difference(a->data[row + col * a->stride], expected));
            }
        }
    } else {
        auto gerc = [&]() {
            check_status(matx_gerc_z_i8(&backend, O::make(0.5), ger_x, ger_y, a), "GERC");
        };
        const Timing gerc_timing = measure_reinitialized(restore_a, gerc);
        for (matx_int64_t col = 0; col < n; ++col)
            for (matx_int64_t row = 0; row < n; ++row)
                if (!std::isfinite(O::magnitude(a->data[row + col * a->stride]))) ger_error = 1.0;
        csv_row("dense_gerc", O::name, matx_blas_backend_name(backend.kind), n, n, 1, 0,
                0.0, 1, setup, gerc_timing, 8.0 * n * n, ger_error,
                2.0 * n * n * sizeof(T));
    }
    if (ger_error > 1e-8) throw std::runtime_error("GER reference check failed");

    if constexpr (std::is_same_v<T, double>) {
        auto matrix_exp = [&]() {
            check_status(matx_expm_dense_d_i8(&backend, a, exponential), "matrix exponential");
        };
        const Timing exp_timing = measure(matrix_exp);
        if (!std::isfinite(exponential->data[0]))
            throw std::runtime_error("matrix exponential returned a non-finite result");
        csv_row("dense_expm", O::name, matx_blas_backend_name(backend.kind), n, n, 0, 0,
                0.0, 1, setup, exp_timing, 0.0, 0.0,
                2.0 * n * n * sizeof(T));
    }

    const double rank_flops = O::flop_factor() * (double) n * n * k;
    const double matrix_bytes = (double) n * n * sizeof(T);
    csv_row("dense_rankk", O::name, matx_blas_backend_name(backend.kind), n, n, k, 0,
            0.0, 1, setup, rankk_timing, rank_flops, 0.0, 3.0 * matrix_bytes);
    csv_row("dense_rank2k", O::name, matx_blas_backend_name(backend.kind), n, n, k, 0,
            0.0, 1, setup, rank2k_timing, 2.0 * rank_flops, 0.0,
            4.0 * matrix_bytes);
    csv_row("dense_inverse", O::name, matx_blas_backend_name(backend.kind), n, n, n, 0,
            0.0, 1, setup, inverse_timing,
            O::flop_factor() * (double) n * n * n / 3.0, inverse_error,
            2.0 * matrix_bytes);
    csv_row("dense_trsv", O::name, matx_blas_backend_name(backend.kind), n, 1, 1, 0,
            0.0, 1, setup, trsv_timing, (double) n * n, trsv_error,
            (double) n * n * sizeof(T));
    csv_row("dense_trsm", O::name, matx_blas_backend_name(backend.kind), n, k, n, 0,
            0.0, k, setup, trsm_timing, 2.0 * n * n * k, trsm_error,
            2.0 * n * k * sizeof(T));
    csv_row(std::is_same_v<T, double> ? "dense_ger" : "dense_geru", O::name,
            matx_blas_backend_name(backend.kind), n, n, 1, 0, 0.0, 1,
            setup, ger_timing, 2.0 * n * n, ger_error, 2.0 * n * n * sizeof(T));

    if constexpr (std::is_same_v<T, double>) O::dense_destroy(&alloc, exponential);
    O::vector_destroy(&alloc, ger_y);
    O::vector_destroy(&alloc, ger_x);
    O::vector_destroy(&alloc, tri_x);
    O::dense_destroy(&alloc, inverse);
    O::dense_destroy(&alloc, triangular);
    O::dense_destroy(&alloc, c);
    O::dense_destroy(&alloc, b);
    O::dense_destroy(&alloc, a);
}

template <typename T>
void run_sparse_utility_case(const matx_sparse_backend_t& backend, matx_int64_t n)
{
    using O = Ops<T>;
    using Sparse = typename O::Sparse;
    using Dense = typename O::Dense;
    using Vector = typename O::Vector;
    matx_alloc_t alloc = matx_alloc_default();
    const matx_int64_t nnz = n * 3 - 2;
    std::vector<matx_int64_t> rows, columns;
    std::vector<T> values;
    rows.reserve((size_t) nnz);
    columns.reserve((size_t) nnz);
    values.reserve((size_t) nnz);
    for (matx_int64_t row = 0; row < n; ++row) {
        rows.push_back(row);
        columns.push_back(row);
        values.push_back(O::make(2.0 + row * 0.01, row * 0.001));
        if (row > 0) {
            rows.push_back(row);
            columns.push_back(row - 1);
            values.push_back(O::make(0.25, 0.1));
        }
        if (row + 1 < n) {
            rows.push_back(row);
            columns.push_back(row + 1);
            values.push_back(O::make(-0.125, 0.05));
        }
    }
    Sparse a = nullptr, b = nullptr, transposed = nullptr, conjugated = nullptr, sum = nullptr;
    Dense dense = nullptr;
    Vector row_counts = nullptr, col_counts = nullptr;
    Vector row_sums = nullptr, col_sums = nullptr, diagonal = nullptr, row = nullptr, col = nullptr;
    Vector scale = nullptr;
    auto setup_start = Clock::now();
    check_status(O::sparse_create(&alloc, &a, n, nnz, rows.data(), columns.data(), values.data()),
                 "create sparse utility input");
    check_status(O::sparse_create(&alloc, &b, n, nnz, rows.data(), columns.data(), values.data()),
                 "create sparse utility add input");
    check_status(O::sparse_create(&alloc, &transposed, n, nnz, nullptr, nullptr, nullptr),
                 "create sparse transpose output");
    check_status(O::sparse_create(&alloc, &conjugated, n, nnz, nullptr, nullptr, nullptr),
                 "create sparse conjugate transpose output");
    check_status(O::sparse_create(&alloc, &sum, n, nnz, nullptr, nullptr, nullptr),
                 "create sparse addition output");
    check_status(O::dense_create(&alloc, &dense, n, n), "create sparse to dense output");
    check_status(O::vector_create(&alloc, &row_counts, n), "create sparse row counts");
    check_status(O::vector_create(&alloc, &col_counts, n), "create sparse column counts");
    check_status(O::vector_create(&alloc, &row_sums, n), "create sparse row sums");
    check_status(O::vector_create(&alloc, &col_sums, n), "create sparse column sums");
    check_status(O::vector_create(&alloc, &diagonal, n), "create sparse diagonal output");
    check_status(O::vector_create(&alloc, &row, n), "create sparse row output");
    check_status(O::vector_create(&alloc, &col, n), "create sparse column output");
    check_status(O::vector_create(&alloc, &scale, n), "create sparse scaling vector");
    check_status(O::vector_fill(scale, O::one()), "initialize sparse scaling vector");
    const double setup = elapsed_us(setup_start);

    auto transpose = [&]() {
        if constexpr (std::is_same_v<T, double>)
            check_status(matx_transpose_coo_d_i8(&backend, a, transposed), "sparse transpose");
        else
            check_status(matx_transpose_coo_z_i8(&backend, a, transposed), "complex sparse transpose");
    };
    auto reset_transpose_output = [&]() {
        O::sparse_destroy(&alloc, transposed);
        transposed = nullptr;
        check_status(O::sparse_create(&alloc, &transposed, n, nnz, nullptr, nullptr, nullptr),
                     "reset sparse transpose output");
    };
    const Timing transpose_timing = measure_reinitialized(reset_transpose_output, transpose);
    Timing conjugate_timing{};
    if constexpr (std::is_same_v<T, matx_complex_d_t>) {
        auto conjugate_transpose = [&]() {
            check_status(matx_conj_coo_z_i8(&backend, a, conjugated),
                         "sparse conjugate transpose");
        };
        auto reset_conjugate_output = [&]() {
            O::sparse_destroy(&alloc, conjugated);
            conjugated = nullptr;
            check_status(O::sparse_create(&alloc, &conjugated, n, nnz, nullptr, nullptr, nullptr),
                         "reset sparse conjugate transpose output");
        };
        conjugate_timing = measure_reinitialized(reset_conjugate_output, conjugate_transpose);
    }
    auto add = [&]() {
        if constexpr (std::is_same_v<T, double>)
            check_status(matx_spadd_coo_d_i8(&backend, 1.0, a, 1.0, b, sum), "sparse addition");
        else
            check_status(matx_spadd_coo_z_i8(&backend, O::one(), a, O::one(), b, sum),
                         "complex sparse addition");
    };
    auto reset_sum_output = [&]() {
        O::sparse_destroy(&alloc, sum);
        sum = nullptr;
        check_status(O::sparse_create(&alloc, &sum, n, nnz, nullptr, nullptr, nullptr),
                     "reset sparse addition output");
    };
    const Timing add_timing = measure_reinitialized(reset_sum_output, add);
    auto counts = [&]() {
        if constexpr (std::is_same_v<T, double>) {
            check_status(matx_spnnz_rows_coo_d_i8(&backend, a, row_counts), "sparse row counts");
            check_status(matx_spnnz_cols_coo_d_i8(&backend, a, col_counts), "sparse column counts");
        } else {
            check_status(matx_spnnz_rows_coo_z_i8(&backend, a, row_counts), "complex sparse row counts");
            check_status(matx_spnnz_cols_coo_z_i8(&backend, a, col_counts), "complex sparse column counts");
        }
    };
    const Timing count_timing = measure(counts);
    auto sums = [&]() {
        if constexpr (std::is_same_v<T, double>) {
            check_status(matx_sprowsums_coo_d_i8(&backend, a, row_sums), "sparse row sums");
            check_status(matx_spcolsums_coo_d_i8(&backend, a, col_sums), "sparse column sums");
        } else {
            check_status(matx_sprowsums_coo_z_i8(&backend, a, row_sums), "complex sparse row sums");
            check_status(matx_spcolsums_coo_z_i8(&backend, a, col_sums), "complex sparse column sums");
        }
    };
    const Timing sums_timing = measure(sums);
    auto extract_diagonal = [&]() {
        if constexpr (std::is_same_v<T, double>)
            check_status(matx_spdiag_coo_d_i8(&backend, a, 0, diagonal), "sparse diagonal extraction");
        else
            check_status(matx_spdiag_coo_z_i8(&backend, a, 0, diagonal), "complex sparse diagonal extraction");
    };
    const Timing diagonal_timing = measure(extract_diagonal);
    auto extract_row = [&]() {
        if constexpr (std::is_same_v<T, double>)
            check_status(matx_coo_get_row_d_i8(a, n / 2, row), "sparse row extraction");
        else
            check_status(matx_coo_get_row_z_i8(a, n / 2, row), "complex sparse row extraction");
    };
    const Timing row_timing = measure(extract_row);
    auto extract_col = [&]() {
        if constexpr (std::is_same_v<T, double>)
            check_status(matx_coo_get_col_d_i8(a, n / 2, col), "sparse column extraction");
        else
            check_status(matx_coo_get_col_z_i8(a, n / 2, col), "complex sparse column extraction");
    };
    const Timing col_timing = measure(extract_col);
    auto convert_dense = [&]() {
        if constexpr (std::is_same_v<T, double>)
            check_status(matx_coo_to_dense_d_i8(a, dense), "COO to dense conversion");
        else
            check_status(matx_coo_to_dense_z_i8(a, dense), "complex COO to dense conversion");
    };
    const Timing dense_timing = measure(convert_dense);
    auto scale_rows = [&]() {
        if constexpr (std::is_same_v<T, double>)
            check_status(matx_scale_rows_coo_d_i8(&backend, a, scale), "sparse row scaling");
        else
            check_status(matx_scale_rows_coo_z_i8(&backend, a, scale), "complex sparse row scaling");
    };
    const Timing scale_rows_timing = measure(scale_rows);
    auto scale_cols = [&]() {
        if constexpr (std::is_same_v<T, double>)
            check_status(matx_scale_cols_coo_d_i8(&backend, a, scale), "sparse column scaling");
        else
            check_status(matx_scale_cols_coo_z_i8(&backend, a, scale), "complex sparse column scaling");
    };
    const Timing scale_cols_timing = measure(scale_cols);
    auto coo_to_csc = [&]() {
        if constexpr (std::is_same_v<T, double>)
            check_status(coo_to_csc_d_i8(a), "real COO to CSC conversion");
        else
            check_status(coo_to_csc_z_i8(a), "complex COO to CSC conversion");
    };
    const Timing coo_to_csc_timing = measure(coo_to_csc);

    double max_error = 0.0;
    if (sum->nnz != nnz || transposed->nnz != nnz)
        throw std::runtime_error("sparse utility output nonzero count changed: n="
                                 + std::to_string(n) + " expected=" + std::to_string(nnz)
                                 + " sum=" + std::to_string(sum->nnz)
                                 + " transpose=" + std::to_string(transposed->nnz));
    std::unordered_map<uint64_t, T> sum_values;
    std::unordered_map<uint64_t, T> transpose_values;
    sum_values.reserve((size_t) nnz);
    transpose_values.reserve((size_t) nnz);
    for (matx_int64_t i = 0; i < nnz; ++i) {
        const size_t index = (size_t) i;
        sum_values[(uint64_t) sum->rows[index] * (uint64_t) n + (uint64_t) sum->columns[index]]
            = sum->values[index];
        transpose_values[(uint64_t) transposed->rows[index] * (uint64_t) n
                         + (uint64_t) transposed->columns[index]] = transposed->values[index];
    }
    for (matx_int64_t i = 0; i < nnz; ++i) {
        const uint64_t key = (uint64_t) rows[(size_t) i] * (uint64_t) n
                             + (uint64_t) columns[(size_t) i];
        const uint64_t transposed_key = (uint64_t) columns[(size_t) i] * (uint64_t) n
                                        + (uint64_t) rows[(size_t) i];
        max_error = std::max(max_error,
                             O::difference(sum_values.at(key), O::multiply(values[(size_t) i], O::make(2.0))));
        if (O::difference(transpose_values.at(transposed_key), values[(size_t) i]) > 1e-12)
            throw std::runtime_error("sparse transpose reference check failed");
    }
    if constexpr (std::is_same_v<T, matx_complex_d_t>) {
        if (conjugated->nnz != nnz)
            throw std::runtime_error("sparse conjugate transpose nonzero count changed");
        std::unordered_map<uint64_t, T> conjugate_values;
        conjugate_values.reserve((size_t) nnz);
        for (matx_int64_t i = 0; i < nnz; ++i) {
            const size_t index = (size_t) i;
            const uint64_t key = (uint64_t) conjugated->rows[index] * (uint64_t) n
                                 + (uint64_t) conjugated->columns[index];
            conjugate_values[key] = conjugated->values[index];
        }
        for (matx_int64_t i = 0; i < nnz; ++i) {
            const uint64_t key = (uint64_t) columns[(size_t) i] * (uint64_t) n
                                 + (uint64_t) rows[(size_t) i];
            T expected = values[(size_t) i];
            expected.imag = -expected.imag;
            if (O::difference(conjugate_values.at(key), expected) > 1e-12)
                throw std::runtime_error("sparse conjugate transpose reference check failed");
        }
    }
    const double expected_center_count = 3.0;
    const T expected_center_value = O::make(2.0 + (n / 2) * 0.01, (n / 2) * 0.001);
    if (O::difference(row_counts->data[(n / 2) * row_counts->stride], O::make(expected_center_count)) > 0.0
        || O::difference(dense->data[(n / 2) + (n / 2) * dense->stride], expected_center_value) > 1e-12)
        throw std::runtime_error("sparse utility reference check failed");
    if (max_error > 1e-12) throw std::runtime_error("sparse addition reference check failed");

    const double density = (double) nnz / ((double) n * (double) n);
    const double sparse_bytes = (double) nnz * (2 * sizeof(matx_int64_t) + sizeof(T));
    const char* backend_name = matx_sparse_backend_name(backend.kind);
    csv_row("sparse_transpose", O::name, backend_name, n, n, 0, nnz, density, 1,
            setup, transpose_timing, 0.0, 0.0, sparse_bytes);
    if constexpr (std::is_same_v<T, matx_complex_d_t>) {
        csv_row("sparse_conjugate_transpose", O::name, backend_name, n, n, 0, nnz,
                density, 1, setup, conjugate_timing, 0.0, 0.0, sparse_bytes);
    }
    csv_row("sparse_add", O::name, backend_name, n, n, 0, nnz, density, 1,
            setup, add_timing, 0.0, max_error, 3.0 * sparse_bytes);
    csv_row("sparse_nnz_rows_cols", O::name, backend_name, n, n, 0, nnz, density, 1,
            setup, count_timing, 0.0, 0.0, sparse_bytes);
    csv_row("sparse_abs_sums_rows_cols", O::name, backend_name, n, n, 0, nnz, density, 1,
            setup, sums_timing, 0.0, 0.0, sparse_bytes);
    csv_row("sparse_diagonal", O::name, backend_name, n, n, 0, nnz, density, 1,
            setup, diagonal_timing, 0.0, 0.0, sparse_bytes);
    csv_row("sparse_extract_row", O::name, "COO", n, n, 0, nnz, density, 1,
            setup, row_timing, 0.0, 0.0, sparse_bytes);
    csv_row("sparse_extract_column", O::name, "COO", n, n, 0, nnz, density, 1,
            setup, col_timing, 0.0, 0.0, sparse_bytes);
    csv_row("sparse_coo_to_dense", O::name, "COO", n, n, 0, nnz, density, 1,
            setup, dense_timing, 0.0, 0.0, sparse_bytes + (double) n * n * sizeof(T));
    csv_row("sparse_scale_rows", O::name, backend_name, n, n, 0, nnz, density, 1,
            setup, scale_rows_timing, 0.0, 0.0, 2.0 * sparse_bytes);
    csv_row("sparse_scale_columns", O::name, backend_name, n, n, 0, nnz, density, 1,
            setup, scale_cols_timing, 0.0, 0.0, 2.0 * sparse_bytes);
    csv_row("sparse_coo_to_csc", O::name, "CSC", n, n, 0, nnz, density, 1,
            setup, coo_to_csc_timing, 0.0, 0.0, sparse_bytes);

    O::vector_destroy(&alloc, scale);
    O::vector_destroy(&alloc, col);
    O::vector_destroy(&alloc, row);
    O::vector_destroy(&alloc, diagonal);
    O::vector_destroy(&alloc, col_sums);
    O::vector_destroy(&alloc, row_sums);
    O::vector_destroy(&alloc, col_counts);
    O::vector_destroy(&alloc, row_counts);
    O::dense_destroy(&alloc, dense);
    O::sparse_destroy(&alloc, sum);
    O::sparse_destroy(&alloc, conjugated);
    O::sparse_destroy(&alloc, transposed);
    O::sparse_destroy(&alloc, b);
    O::sparse_destroy(&alloc, a);
}

template <typename T>
void run_coo_to_csc_case(matx_int64_t n, matx_int64_t nnz, bool sorted)
{
    using O = Ops<T>;
    using Sparse = typename O::Sparse;
    matx_alloc_t alloc = matx_alloc_default();
    std::vector<std::pair<matx_int64_t, matx_int64_t>> coordinates((size_t) nnz);
    std::vector<matx_int64_t> rows((size_t) nnz), columns((size_t) nnz);
    std::vector<matx_int64_t> column_counts((size_t) n, 0);
    std::vector<T> values((size_t) nnz, O::make(1.0, 0.0));

    uint64_t state = 0x9e3779b97f4a7c15ULL ^ (uint64_t) n ^ (uint64_t) nnz;
    for (matx_int64_t i = 0; i < nnz; ++i) {
        state ^= state << 7;
        state ^= state >> 9;
        const matx_int64_t col = (matx_int64_t) (state % (uint64_t) n);
        state ^= state << 8;
        state ^= state >> 11;
        const matx_int64_t row = (matx_int64_t) (state % (uint64_t) n);
        coordinates[(size_t) i] = {col, row};
        ++column_counts[(size_t) col];
    }
    if (sorted) {
        std::sort(coordinates.begin(), coordinates.end(),
                  [](const auto& left, const auto& right) {
                      return left.first != right.first ? left.first < right.first
                                                       : left.second < right.second;
                  });
    }
    for (matx_int64_t i = 0; i < nnz; ++i) {
        columns[(size_t) i] = coordinates[(size_t) i].first;
        rows[(size_t) i] = coordinates[(size_t) i].second;
    }

    Sparse matrix = nullptr;
    const auto setup_start = Clock::now();
    check_status(O::sparse_create(&alloc, &matrix, n, nnz, rows.data(),
                                  columns.data(), values.data()),
                 "create COO to CSC input");
    const double setup = elapsed_us(setup_start);
    auto convert = [&]() {
        if constexpr (std::is_same_v<T, double>)
            check_status(coo_to_csc_d_i8(matrix), "real COO to CSC conversion");
        else
            check_status(coo_to_csc_z_i8(matrix), "complex COO to CSC conversion");
    };
    const Timing timing = measure(convert);

    const auto csc = matrix->handle_csc;
    if (!csc || csc->nnz <= 0 || csc->nnz > nnz) {
        throw std::runtime_error("COO to CSC returned an invalid nonzero count");
    }
    for (matx_int64_t col = 0; col < n; ++col) {
        if (csc->col_ptr[col] > csc->col_ptr[col + 1])
            throw std::runtime_error("COO to CSC returned invalid column pointers");
        for (matx_int64_t p = csc->col_ptr[col]; p < csc->col_ptr[col + 1]; ++p) {
            if (csc->row_ind[p] < 0 || csc->row_ind[p] >= n
                || (p > csc->col_ptr[col] && csc->row_ind[p - 1] >= csc->row_ind[p])) {
                throw std::runtime_error("COO to CSC row indices are not strictly sorted");
            }
        }
    }
    for (matx_int64_t i = 0; i < nnz; ++i) {
        const matx_int64_t dst = csc->coo_csc_index_map[i];
        if (dst < csc->col_ptr[columns[(size_t) i]]
            || dst >= csc->col_ptr[columns[(size_t) i] + 1]
            || csc->row_ind[dst] != rows[(size_t) i]) {
            throw std::runtime_error("COO to CSC input map failed validation");
        }
    }

    const double density = (double) nnz / ((double) n * (double) n);
    const double input_and_output_bytes
        = 2.0 * (double) nnz * (2 * sizeof(matx_int64_t) + sizeof(T))
          + (double) (n + 1) * sizeof(matx_int64_t);
    const matx_int64_t max_column_count
        = *std::max_element(column_counts.begin(), column_counts.end());
    const size_t scratch_entry_bytes
        = max_column_count >= 256 ? sizeof(matx_int64_t) : 2 * sizeof(matx_int64_t);
    const double scratch_bytes
        = sorted ? 0.0 : (double) nnz * scratch_entry_bytes;
    const char* operation = sorted ? "sparse_coo_to_csc_sorted" : "sparse_coo_to_csc_unsorted";
    csv_row(operation, O::name, "CSC", n, n, 0, nnz, density, 1,
            setup, timing, 0.0, 0.0, input_and_output_bytes + 2.0 * scratch_bytes);
    std::cerr << "COO-to-CSC scratch estimate: n=" << n << ", nnz=" << nnz
              << ", ordered=" << (sorted ? "yes" : "no")
              << ", max_column_count=" << max_column_count
              << ", temporary_bytes=" << (uint64_t) scratch_bytes << '\n';

    O::sparse_destroy(&alloc, matrix);
}

void run_coo_to_csr_case(matx_int64_t n, matx_int64_t nnz)
{
    matx_alloc_t alloc = matx_alloc_default();
    std::vector<matx_int64_t> rows((size_t) nnz), columns((size_t) nnz);
    std::vector<matx_double> values((size_t) nnz, 1.0);
    uint64_t state = 0xd1b54a32d192ed03ULL ^ (uint64_t) n ^ (uint64_t) nnz;
    const auto setup_start = Clock::now();
    for (matx_int64_t i = 0; i < nnz; ++i) {
        state ^= state << 7;
        state ^= state >> 9;
        rows[(size_t) i] = (matx_int64_t) (state % (uint64_t) n);
        state ^= state << 8;
        state ^= state >> 11;
        columns[(size_t) i] = (matx_int64_t) (state % (uint64_t) n);
    }
    const double setup = elapsed_us(setup_start);

    matx_benchmark_csr_matrix_t csr{};
    auto release_output = [&]() {
        matx_free(&alloc, csr.row_ptr);
        matx_free(&alloc, csr.col_ind);
        matx_free(&alloc, csr.val);
        csr = {};
    };
    auto convert = [&]() {
        release_output();
        if (coo_to_csr_optimized(&alloc, n, n, nnz,
                                  rows.data(), columns.data(), values.data(), &csr) != 0) {
            throw std::runtime_error("COO to CSR conversion failed");
        }
    };
    const Timing timing = measure(convert);

    if (!csr.row_ptr || !csr.col_ind || !csr.val || csr.nnz <= 0 || csr.nnz > nnz
        || csr.row_ptr[0] != 0 || csr.row_ptr[n] != csr.nnz) {
        throw std::runtime_error("COO to CSR returned invalid dimensions or pointers");
    }
    matx_int64_t found = 0;
    double output_sum = 0.0;
    for (matx_int64_t row = 0; row < n; ++row) {
        if (csr.row_ptr[row] > csr.row_ptr[row + 1])
            throw std::runtime_error("COO to CSR returned decreasing row pointers");
        for (matx_int64_t p = csr.row_ptr[row]; p < csr.row_ptr[row + 1]; ++p) {
            if (csr.col_ind[p] < 0 || csr.col_ind[p] >= n
                || (p > csr.row_ptr[row] && csr.col_ind[p - 1] >= csr.col_ind[p])) {
                throw std::runtime_error("COO to CSR columns are not strictly sorted");
            }
            output_sum += csr.val[p];
        }
    }
    for (matx_int64_t i = 0; i < nnz; ++i) {
        const matx_int64_t row = rows[(size_t) i];
        const matx_int64_t col = columns[(size_t) i];
        const matx_int64_t* begin = csr.col_ind + csr.row_ptr[row];
        const matx_int64_t* end = csr.col_ind + csr.row_ptr[row + 1];
        if (std::binary_search(begin, end, col)) ++found;
    }
    if (found != nnz || std::abs(output_sum - (double) nnz) > 1e-9 * nnz) {
        throw std::runtime_error("COO to CSR numerical validation failed");
    }

    const double density = (double) nnz / ((double) n * (double) n);
    const double bytes_moved
        = (double) nnz * (3 * sizeof(matx_int64_t) + 2 * sizeof(matx_double))
          + (double) (n + 1) * sizeof(matx_int64_t);
    csv_row("sparse_coo_to_csr", "f64", "CSR", n, n, 0, nnz, density, 1,
            setup, timing, 0.0, std::abs(output_sum - (double) nnz), bytes_moved);
    std::cerr << "COO-to-CSR temporary_bytes: n=" << n << ", nnz=" << nnz
              << ", temporary_bytes=0 (row-pointer cursors reused)\n";
    release_output();
}

template <typename T>
void run_coo_to_csc_suite()
{
    run_coo_to_csc_case<T>(64, 8192, false);
    run_coo_to_csc_case<T>(2000, 60000, false);
    run_coo_to_csc_case<T>(2048, 750000, false);
    run_coo_to_csc_case<T>(100000, 750000, false);
    run_coo_to_csc_case<T>(100000, 750000, true);
    run_coo_to_csc_case<T>(131072, 1000000, false);
}

void run_coo_to_csr_suite()
{
    run_coo_to_csr_case(64, 8192);
    run_coo_to_csr_case(2000, 60000);
    run_coo_to_csr_case(2048, 750000);
    run_coo_to_csr_case(100000, 300000);
    run_coo_to_csr_case(131072, 1000000);
}

template <typename T>
void run_sparse_product_case(const matx_sparse_backend_t& backend, matx_int64_t n)
{
    using O = Ops<T>;
    using Sparse = typename O::Sparse;
    using Dense = typename O::Dense;
    matx_alloc_t alloc = matx_alloc_default();
    std::vector<matx_int64_t> rows, columns;
    std::vector<T> values;
    std::vector<T> reference((size_t) n * (size_t) n, O::zero());
    rows.reserve((size_t) n * 3);
    columns.reserve((size_t) n * 3);
    values.reserve((size_t) n * 3);
    for (matx_int64_t row = 0; row < n; ++row) {
        const T diagonal = O::make(2.0 + row * 0.01, row * 0.001);
        rows.push_back(row);
        columns.push_back(row);
        values.push_back(diagonal);
        reference[(size_t) row * (size_t) n + (size_t) row] = diagonal;
        if (row > 0) {
            const T lower = O::make(0.25, 0.1);
            rows.push_back(row);
            columns.push_back(row - 1);
            values.push_back(lower);
            reference[(size_t) row * (size_t) n + (size_t) (row - 1)] = lower;
        }
        if (row + 1 < n) {
            const T upper = O::make(-0.125, 0.05);
            rows.push_back(row);
            columns.push_back(row + 1);
            values.push_back(upper);
            reference[(size_t) row * (size_t) n + (size_t) (row + 1)] = upper;
        }
    }
    const matx_int64_t nnz = (matx_int64_t) values.size();
    Sparse a = nullptr;
    Dense c = nullptr;
    const auto setup_start = Clock::now();
    check_status(O::sparse_create(&alloc, &a, n, nnz,
                                  rows.data(), columns.data(), values.data()),
                 "create sparse-sparse product input");
    check_status(O::dense_create(&alloc, &c, n, n),
                 "create sparse-sparse product output");
    const double setup = elapsed_us(setup_start);
    auto product = [&]() {
        if constexpr (std::is_same_v<T, double>)
            check_status(matx_dsp2md_coo_d_i8(&backend, 1.0, a, a, 0.0, c),
                         "real sparse-sparse product to dense");
        else
            check_status(matx_zsp2md_coo_z_i8(&backend, O::one(), a, a, O::zero(), c),
                         "complex sparse-sparse product to dense");
    };
    const Timing timing = measure(product);
    double max_error = 0.0;
    for (matx_int64_t row = 0; row < n; ++row) {
        for (matx_int64_t col = 0; col < n; ++col) {
            T expected = O::zero();
            const matx_int64_t begin = std::max<matx_int64_t>(0, std::max(row - 1, col - 1));
            const matx_int64_t end = std::min<matx_int64_t>(n - 1, std::min(row + 1, col + 1));
            for (matx_int64_t inner = begin; inner <= end; ++inner)
                expected = O::add(expected,
                    O::multiply(reference[(size_t) row * (size_t) n + (size_t) inner],
                                reference[(size_t) inner * (size_t) n + (size_t) col]));
            const T actual = c->data[row + col * c->stride];
            max_error = std::max(max_error, O::difference(actual, expected));
        }
    }
    if (!std::isfinite(max_error) || max_error > 1e-10)
        throw std::runtime_error("sparse-sparse product reference check failed");

    const double density = (double) nnz / ((double) n * (double) n);
    const double sparse_bytes = (double) nnz * (2 * sizeof(matx_int64_t) + sizeof(T));
    csv_row("sparse_sparse_product_dense", O::name,
            matx_sparse_backend_name(backend.kind), n, n, n, nnz, density, 1,
            setup, timing, 10.0 * n, max_error,
            2.0 * sparse_bytes + (double) n * n * sizeof(T));
    O::dense_destroy(&alloc, c);
    O::sparse_destroy(&alloc, a);
}

void run_sparse_cholesky_case(const matx_sparse_linsolve_t& solver,
                              matx_int64_t n,
                              bool solve_only = false)
{
    matx_alloc_t alloc = matx_alloc_default();
    const matx_int64_t nnz = n * 3 - 2;
    std::vector<matx_int64_t> rows, columns;
    std::vector<double> values;
    std::vector<double> rhs((size_t) n, 2.0);
    std::vector<double> solution((size_t) n, 0.0);
    rows.reserve((size_t) nnz);
    columns.reserve((size_t) nnz);
    values.reserve((size_t) nnz);
    for (matx_int64_t row = 0; row < n; ++row) {
        rows.push_back(row);
        columns.push_back(row);
        values.push_back(4.0);
        if (row > 0) {
            rows.push_back(row);
            columns.push_back(row - 1);
            values.push_back(-1.0);
        }
        if (row + 1 < n) {
            rows.push_back(row);
            columns.push_back(row + 1);
            values.push_back(-1.0);
        }
    }
    if (n == 1) rhs[0] = 4.0;
    else {
        rhs.front() = 3.0;
        rhs.back() = 3.0;
    }
    matx_coo_d_i8_t matrix = nullptr;
    const auto setup_start = Clock::now();
    check_status(matx_coo_sparse_d_i8_create(&alloc, &matrix, n, n, nnz,
                                              rows.data(), columns.data(), values.data()),
                 "create sparse Cholesky matrix");
    const double setup = elapsed_us(setup_start);

    auto factor_once = [&]() {
        matx_factor_sparse_d_i8_t temporary{};
        const matx_status_t status = matx_factor_chol_coo_d_i8(&solver, matrix, &temporary);
        if (status == MATX_ERR_NOT_SUPPORTED)
            throw std::runtime_error("sparse Cholesky is not supported by selected solver");
        check_status(status, "sparse Cholesky factorization");
        matx_factor_chol_coo_d_i8_destroy(&solver, &temporary);
    };
    const Timing factor_timing = solve_only ? Timing{} : measure(factor_once);

    matx_factor_sparse_d_i8_t factor{};
    check_status(matx_factor_chol_coo_d_i8(&solver, matrix, &factor),
                 "sparse Cholesky factorization for reused solve");
    auto reused_solve = [&]() {
        check_status(matx_solve_chol_coo_d_i8_factor(&solver, &factor,
                                                       rhs.data(), solution.data()),
                     "sparse Cholesky reused solve");
    };
    const Timing reused_timing = measure(reused_solve);
    Timing oneshot_timing{};
    if (!solve_only) {
        auto oneshot_solve = [&]() {
            check_status(matx_solve_chol_coo_d_i8(&solver, matrix, rhs.data(), solution.data()),
                         "sparse Cholesky one-shot solve");
        };
        oneshot_timing = measure(oneshot_solve);
    }
    auto residual = [&]() {
        std::vector<double> product((size_t) n, 0.0);
        for (matx_int64_t k = 0; k < nnz; ++k)
            product[(size_t) rows[(size_t) k]]
                += values[(size_t) k] * solution[(size_t) columns[(size_t) k]];
        double max_residual = 0.0;
        for (matx_int64_t row = 0; row < n; ++row)
            max_residual = std::max(max_residual,
                                    std::abs(product[(size_t) row] - rhs[(size_t) row]));
        return max_residual / (1.0 + *std::max_element(rhs.begin(), rhs.end()));
    };
    const double solve_error = residual();
    if (!std::isfinite(solve_error) || solve_error > 1e-10)
        throw std::runtime_error("sparse Cholesky solve residual check failed");

    const double density = (double) nnz / ((double) n * (double) n);
    const double bytes = (double) nnz * (2 * sizeof(matx_int64_t) + sizeof(double));
    const char* backend = matx_sparse_linsolve_backend_name(solver.kind);
    if (!solve_only)
        csv_row("sparse_chol_factor", "f64", backend, n, n, 1, nnz, density, 1,
                setup, factor_timing, (double) n * n * n / 3.0, 0.0, 2.0 * bytes);
    csv_row("sparse_chol_solve_reused", "f64", backend, n, 1, 1, nnz, density, 1,
            setup, reused_timing, 2.0 * nnz, solve_error, 2.0 * n * sizeof(double));
    if (!solve_only)
        csv_row("sparse_chol_solve_oneshot", "f64", backend, n, 1, 1, nnz, density, 1,
                setup, oneshot_timing, (double) n * n * n / 3.0 + 2.0 * nnz,
                solve_error, 2.0 * bytes);
    matx_factor_chol_coo_d_i8_destroy(&solver, &factor);
    matx_coo_sparse_d_i8_destroy(&alloc, matrix);
}

template <typename T>
void run_dense_cholesky_solve_case(const matx_dense_linsolve_t& solver, matx_int64_t n)
{
    using O = Ops<T>;
    using Dense = typename O::Dense;
    using Vector = typename O::Vector;
    using Factor = typename O::Factor;
    matx_alloc_t alloc = matx_alloc_default();
    Dense matrix = nullptr;
    Vector rhs = nullptr, solution = nullptr;
    const auto setup_start = Clock::now();
    check_status(O::dense_create(&alloc, &matrix, n, n), "create dense Cholesky matrix");
    check_status(O::vector_create(&alloc, &rhs, n), "create dense Cholesky RHS");
    check_status(O::vector_create(&alloc, &solution, n), "create dense Cholesky solution");
    for (matx_int64_t col = 0; col < n; ++col) {
        for (matx_int64_t row = 0; row < n; ++row) {
            if (row == col) {
                matrix->data[row + col * matrix->stride] = O::make(4.0, 0.0);
            } else {
                const matx_int64_t low = std::min(row, col);
                const matx_int64_t high = std::max(row, col);
                const double real = seeded_value((uint64_t) low + (uint64_t) high * n, 337) * 0.001;
                const double imag = row < col ? 0.0002 : -0.0002;
                matrix->data[row + col * matrix->stride] = O::make(real, imag);
            }
        }
    }
    for (matx_int64_t row = 0; row < n; ++row)
        rhs->data[row * rhs->stride] = generated_value<T>((uint64_t) row, 347);
    const double setup = elapsed_us(setup_start);

    Factor factor = nullptr;
    matx_status_t status;
    if constexpr (std::is_same_v<T, double>)
        status = matx_factor_chol_d_i8(&solver, matrix, MATX_LOWER, &factor);
    else
        status = matx_factor_chol_z_i8(&solver, matrix, MATX_LOWER, &factor);
    check_status(status, "dense Cholesky factorization for reused solve");
    auto solve = [&]() {
        if constexpr (std::is_same_v<T, double>)
            check_status(matx_solve_chol_d_i8(&solver, factor, rhs->data, solution->data),
                         "dense Cholesky reused solve");
        else
            check_status(matx_solve_chol_z_i8(&solver, factor, rhs, solution),
                         "complex dense Cholesky reused solve");
    };
    const Timing timing = measure(solve);

    double max_rhs = 0.0;
    double max_residual = 0.0;
    for (matx_int64_t row = 0; row < n; ++row) {
        T product = O::zero();
        for (matx_int64_t col = 0; col < n; ++col)
            product = O::add(product,
                             O::multiply(matrix->data[row + col * matrix->stride],
                                         solution->data[col * solution->stride]));
        const T b = rhs->data[row * rhs->stride];
        max_rhs = std::max(max_rhs, O::magnitude(b));
        max_residual = std::max(max_residual, O::difference(product, b));
    }
    const double residual = max_residual / (1.0 + max_rhs);
    if (!std::isfinite(residual) || residual > 1e-8)
        throw std::runtime_error("dense Cholesky solve residual check failed");

    csv_row("chol_solve_reused", O::name, matx_dense_linsolve_backend_name(solver.kind),
            n, 1, n, 0, 1.0, 1, setup, timing,
            O::flop_factor() * (double) n * n, residual, 2.0 * n * sizeof(T));
    O::factor_destroy(&solver, factor);
    O::vector_destroy(&alloc, solution);
    O::vector_destroy(&alloc, rhs);
    O::dense_destroy(&alloc, matrix);
}

template <typename T>
void run_dense_cholesky_solve_suite(const matx_dense_linsolve_t& solver)
{
    for (matx_int64_t size : {1, 16, 31, 32, 33, 64, 99, 100, 512, 1999, 2000})
        run_dense_cholesky_solve_case<T>(solver, size);
}

void run_sparse_cholesky_suite(const matx_sparse_linsolve_t& solver,
                               bool full,
                               bool quick,
                               bool solve_only = false)
{
    if (!solver.vt.factor_chol_csc_d_i8 || !solver.vt.solve_chol_csc_d_i8) {
        std::cerr << "Sparse Cholesky benchmark skipped: selected SuiteSparse solver has no Cholesky backend\n";
        return;
    }
    const std::vector<matx_int64_t> sizes
        = solve_only ? std::vector<matx_int64_t>{1, 16, 64, 99, 100, 512, 1999, 2000,
                                                 16384, 100000}
                     : quick ? std::vector<matx_int64_t>{16, 64}
                             : full ? std::vector<matx_int64_t>{128, 512, 2048}
                                    : std::vector<matx_int64_t>{128, 512};
    for (matx_int64_t size : sizes)
        run_sparse_cholesky_case(solver, size, solve_only);
}

template <typename T>
void run_dense_solve_extended_case(const matx_dense_linsolve_t& solver, matx_int64_t n)
{
    using O = Ops<T>;
    using Dense = typename O::Dense;
    using Vector = typename O::Vector;
    using Factor = typename O::Factor;
    const matx_int64_t m = n * 2;
    matx_alloc_t alloc = matx_alloc_default();
    Dense square = nullptr, rectangular = nullptr, q = nullptr, r = nullptr;
    Vector rhs = nullptr, solution = nullptr, square_rhs = nullptr;
    matx_vec_d_i8_t eigenvalues = nullptr, singular_values = nullptr;
    matx_vec_z_i8_t general_eigenvalues = nullptr;
    const auto setup_start = Clock::now();
    check_status(O::dense_create(&alloc, &square, n, n), "create dense solver square matrix");
    check_status(O::dense_create(&alloc, &rectangular, m, n), "create dense solver rectangular matrix");
    check_status(O::vector_create(&alloc, &rhs, m), "create dense solver RHS");
    check_status(O::vector_create(&alloc, &solution, n), "create dense solver solution");
    check_status(O::vector_create(&alloc, &square_rhs, n), "create dense Cholesky RHS");
    check_status(matx_vec_d_i8_create(&alloc, &eigenvalues, nullptr, n),
                 "create dense eigenvalue output");
    check_status(matx_vec_d_i8_create(&alloc, &singular_values, nullptr, n),
                 "create singular value output");
    check_status(matx_vec_z_i8_create(&alloc, &general_eigenvalues, nullptr, n),
                 "create general eigenvalue output");

    for (matx_int64_t col = 0; col < n; ++col) {
        for (matx_int64_t row = 0; row < n; ++row) {
            const matx_int64_t low = std::min(row, col);
            const matx_int64_t high = std::max(row, col);
            const uint64_t index = (uint64_t) low + (uint64_t) high * n;
            const double real = row == col ? 4.0 + seeded_value(index, 281) * 0.01
                                           : seeded_value(index, 283) * 0.01;
            const double imag = row == col ? 0.0
                                           : seeded_value(index, 293) * (row > col ? 0.002 : -0.002);
            square->data[row + col * square->stride] = O::make(real, imag);
        }
    }
    std::vector<T> square_backup((size_t) n * (size_t) n);
    for (matx_int64_t col = 0; col < n; ++col)
        for (matx_int64_t row = 0; row < n; ++row)
            square_backup[(size_t) row + (size_t) col * n]
                = square->data[row + col * square->stride];

    for (matx_int64_t col = 0; col < n; ++col) {
        for (matx_int64_t row = 0; row < m; ++row) {
            const uint64_t index = (uint64_t) row + (uint64_t) col * m;
            const double real = (row == col ? 1.5 : 0.0) + seeded_value(index, 307) * 0.002;
            const double imag = seeded_value(index, 311) * 0.001;
            rectangular->data[row + col * rectangular->stride] = O::make(real, imag);
        }
    }
    std::vector<T> rectangular_backup((size_t) m * (size_t) n);
    for (matx_int64_t col = 0; col < n; ++col)
        for (matx_int64_t row = 0; row < m; ++row)
            rectangular_backup[(size_t) row + (size_t) col * m]
                = rectangular->data[row + col * rectangular->stride];

    for (matx_int64_t row = 0; row < n; ++row) {
        T value = O::zero();
        for (matx_int64_t col = 0; col < n; ++col)
            value = O::add(value, square->data[row + col * square->stride]);
        square_rhs->data[row * square_rhs->stride] = value;
        solution->data[row * solution->stride] = O::zero();
    }
    for (matx_int64_t row = 0; row < m; ++row) {
        T value = O::zero();
        for (matx_int64_t col = 0; col < n; ++col)
            value = O::add(value, rectangular->data[row + col * rectangular->stride]);
        rhs->data[row * rhs->stride] = value;
    }
    const double setup = elapsed_us(setup_start);
    auto restore_square = [&]() {
        for (matx_int64_t col = 0; col < n; ++col)
            for (matx_int64_t row = 0; row < n; ++row)
                square->data[row + col * square->stride]
                    = square_backup[(size_t) row + (size_t) col * n];
    };
    auto restore_rectangular = [&]() {
        for (matx_int64_t col = 0; col < n; ++col)
            for (matx_int64_t row = 0; row < m; ++row)
                rectangular->data[row + col * rectangular->stride]
                    = rectangular_backup[(size_t) row + (size_t) col * m];
    };

    Factor chol_factor = nullptr;
    auto chol_factor_once = [&]() {
        Factor temporary = nullptr;
        matx_status_t status;
        if constexpr (std::is_same_v<T, double>)
            status = matx_factor_chol_d_i8(&solver, square, MATX_LOWER, &temporary);
        else
            status = matx_factor_chol_z_i8(&solver, square, MATX_LOWER, &temporary);
        if (status == MATX_ERR_NOT_SUPPORTED)
            throw std::runtime_error("dense Cholesky is not supported by selected solver");
        check_status(status, "dense Cholesky factorization");
        O::factor_destroy(&solver, temporary);
    };
    const Timing chol_factor_timing = measure(chol_factor_once);
    matx_status_t chol_status;
    if constexpr (std::is_same_v<T, double>)
        chol_status = matx_factor_chol_d_i8(&solver, square, MATX_LOWER, &chol_factor);
    else
        chol_status = matx_factor_chol_z_i8(&solver, square, MATX_LOWER, &chol_factor);
    check_status(chol_status, "dense Cholesky factorization for reused solve");
    auto chol_solve = [&]() {
        if constexpr (std::is_same_v<T, double>)
            check_status(matx_solve_chol_d_i8(&solver, chol_factor,
                                               square_rhs->data, solution->data), "dense Cholesky solve");
        else
            check_status(matx_solve_chol_z_i8(&solver, chol_factor, square_rhs, solution),
                         "complex dense Cholesky solve");
    };
    const Timing chol_solve_timing = measure(chol_solve);
    double solve_error = 0.0;
    for (matx_int64_t row = 0; row < n; ++row) {
        T product = O::zero();
        for (matx_int64_t col = 0; col < n; ++col)
            product = O::add(product,
                             O::multiply(square->data[row + col * square->stride],
                                         solution->data[col * solution->stride]));
        solve_error = std::max(solve_error,
                               O::difference(product, square_rhs->data[row * square_rhs->stride]));
    }
    if (!std::isfinite(solve_error) || solve_error > 1e-8)
        throw std::runtime_error("dense Cholesky solve residual check failed");

    if constexpr (std::is_same_v<T, double>) {
        auto chol_oneshot = [&]() {
            check_status(matx_solve_chol_d_i8_oneshot(
                             &solver, square, MATX_LOWER,
                             square_rhs->data, solution->data),
                         "dense Cholesky one-shot solve");
        };
        const Timing oneshot_timing = measure_reinitialized(restore_square, chol_oneshot);
        restore_square();
        double oneshot_error = 0.0;
        for (matx_int64_t row = 0; row < n; ++row) {
            double product = 0.0;
            for (matx_int64_t col = 0; col < n; ++col)
                product += square->data[row + col * square->stride]
                           * solution->data[col * solution->stride];
            oneshot_error = std::max(oneshot_error,
                std::abs(product - square_rhs->data[row * square_rhs->stride]));
        }
        if (!std::isfinite(oneshot_error) || oneshot_error > 1e-8)
            throw std::runtime_error("dense Cholesky one-shot residual check failed");
        csv_row("chol_solve_oneshot", O::name,
                matx_dense_linsolve_backend_name(solver.kind), n, 1, n, 0, 1.0, 1,
                setup, oneshot_timing, (double) n * n, oneshot_error,
                (double) n * n * sizeof(double));
    }

    auto gels = [&]() {
        if constexpr (std::is_same_v<T, double>)
            check_status(matx_gels_d_i8(&solver, rectangular, rhs->data, solution->data),
                         "real least-squares solve");
        else
            check_status(matx_gels_z_i8(&solver, rectangular, rhs, solution),
                         "complex least-squares solve");
    };
    const Timing gels_timing = measure_reinitialized(restore_rectangular, gels);
    double gels_error = 0.0;
    for (matx_int64_t row = 0; row < m; ++row) {
        T product = O::zero();
        for (matx_int64_t col = 0; col < n; ++col)
            product = O::add(product,
                             O::multiply(rectangular_backup[(size_t) row + (size_t) col * m],
                                         solution->data[col * solution->stride]));
        gels_error = std::max(gels_error,
                              O::difference(product, rhs->data[row * rhs->stride]));
    }
    if (!std::isfinite(gels_error) || gels_error > 1e-7)
        throw std::runtime_error("least-squares residual check failed");

    auto qr = [&]() {
        if constexpr (std::is_same_v<T, double>)
            check_status(matx_qr_d_i8(&solver, rectangular, &q, &r), "real QR factorization");
        else
            check_status(matx_qr_z_i8(&solver, rectangular, &q, &r), "complex QR factorization");
    };
    auto reset_qr_outputs = [&]() {
        O::dense_destroy(&alloc, q);
        O::dense_destroy(&alloc, r);
        q = nullptr;
        r = nullptr;
    };
    const Timing qr_timing = measure_reinitialized(reset_qr_outputs, qr);
    const double qr_error = q && r && q->nrows == m && r->ncols == n ? 0.0 : 1.0;
    if (qr_error != 0.0) throw std::runtime_error("QR output shape check failed");

    auto syev = [&]() {
        if constexpr (std::is_same_v<T, double>)
            check_status(matx_syev_d_i8(&solver, square, eigenvalues, nullptr),
                         "real symmetric eigenvalue solve");
        else
            check_status(matx_syev_z_i8(&solver, square, eigenvalues, nullptr),
                         "Hermitian eigenvalue solve");
    };
    const Timing syev_timing = measure_reinitialized(restore_square, syev);
    double eigen_error = 0.0;
    for (matx_int64_t i = 0; i < n; ++i)
        if (!std::isfinite(eigenvalues->data[i * eigenvalues->stride])) eigen_error = 1.0;
    if (eigen_error != 0.0) throw std::runtime_error("eigenvalue output check failed");

    auto svd = [&]() {
        if constexpr (std::is_same_v<T, double>)
            check_status(matx_gesvd_d_i8(&solver, rectangular, singular_values, nullptr, nullptr),
                         "real SVD");
        else
            check_status(matx_gesvd_z_i8(&solver, rectangular, singular_values, nullptr, nullptr),
                         "complex SVD");
    };
    const Timing svd_timing = measure_reinitialized(restore_rectangular, svd);
    double svd_error = 0.0;
    for (matx_int64_t i = 0; i < n; ++i) {
        const double value = singular_values->data[i * singular_values->stride];
        if (!std::isfinite(value) || (i > 0 && value > singular_values->data[(i - 1) * singular_values->stride]))
            svd_error = 1.0;
    }
    if (svd_error != 0.0) throw std::runtime_error("SVD singular value check failed");

    auto geev = [&]() {
        if constexpr (std::is_same_v<T, double>)
            check_status(matx_geev_d_i8(&solver, square, general_eigenvalues, nullptr, nullptr),
                         "real general eigenvalue solve");
        else
            check_status(matx_geev_z_i8(&solver, square, general_eigenvalues, nullptr, nullptr),
                         "complex general eigenvalue solve");
    };
    const Timing geev_timing = measure_reinitialized(restore_square, geev);
    for (matx_int64_t i = 0; i < n; ++i)
        if (!std::isfinite(general_eigenvalues->data[i * general_eigenvalues->stride].real)
            || !std::isfinite(general_eigenvalues->data[i * general_eigenvalues->stride].imag))
            throw std::runtime_error("general eigenvalue output check failed");

    double determinant = 0.0, condition = 0.0;
    auto det = [&]() {
        if constexpr (std::is_same_v<T, double>)
            check_status(matx_det_dense_d_i8(&solver, square, &determinant), "dense determinant");
        else {
            matx_complex_d_t value{};
            check_status(matx_det_dense_z_i8(&solver, square, &value), "complex dense determinant");
            determinant = std::hypot(value.real, value.imag);
        }
    };
    const Timing det_timing = measure(det);
    auto cond = [&]() {
        if constexpr (std::is_same_v<T, double>)
            check_status(matx_cond_dense_d_i8(&solver, square, &condition), "dense condition number");
        else
            check_status(matx_cond_dense_z_i8(&solver, square, &condition), "complex dense condition number");
    };
    const Timing cond_timing = measure(cond);
    if (!std::isfinite(determinant) || !std::isfinite(condition))
        throw std::runtime_error("dense determinant or condition number is non-finite");

    const char* backend = matx_dense_linsolve_backend_name(solver.kind);
    const double density = (double) (n * n) / ((double) n * n);
    csv_row("chol_factor", O::name, backend, n, n, n, 0, density, 1,
            setup, chol_factor_timing, O::flop_factor() * (double) n * n * n / 3.0, 0.0);
    csv_row("chol_solve_reused", O::name, backend, n, 1, n, 0, 1.0, 1,
            setup, chol_solve_timing, O::flop_factor() * (double) n * n, solve_error);
    csv_row("gels", O::name, backend, m, n, n, 0, 1.0, 1,
            setup, gels_timing, 2.0 * O::flop_factor() * (double) m * n * n, gels_error);
    csv_row("qr", O::name, backend, m, n, n, 0, 1.0, 1,
            setup, qr_timing, O::flop_factor() * (2.0 * m * n * n - 2.0 * n * n * n / 3.0), qr_error);
    csv_row("syev", O::name, backend, n, n, n, 0, density, 1,
            setup, syev_timing, O::flop_factor() * (double) n * n * n, eigen_error);
    csv_row("svd", O::name, backend, m, n, n, 0, 1.0, 1,
            setup, svd_timing, 4.0 * O::flop_factor() * (double) m * n * n, svd_error);
    csv_row("geev", O::name, backend, n, n, n, 0, density, 1,
            setup, geev_timing, 10.0 * O::flop_factor() * (double) n * n * n, 0.0);
    csv_row("determinant", O::name, backend, n, n, n, 0, density, 1,
            setup, det_timing, O::flop_factor() * (double) n * n * n / 3.0, 0.0);
    csv_row("condition_number", O::name, backend, n, n, n, 0, density, 1,
            setup, cond_timing, O::flop_factor() * 2.0 * (double) n * n * n, 0.0);

    O::factor_destroy(&solver, chol_factor);
    reset_qr_outputs();
    matx_vec_z_i8_destroy(&alloc, general_eigenvalues);
    matx_vec_d_i8_destroy(&alloc, singular_values);
    matx_vec_d_i8_destroy(&alloc, eigenvalues);
    O::vector_destroy(&alloc, square_rhs);
    O::vector_destroy(&alloc, solution);
    O::vector_destroy(&alloc, rhs);
    O::dense_destroy(&alloc, rectangular);
    O::dense_destroy(&alloc, square);
}

template <typename T>
void run_dense_solve_extended_suite(const matx_dense_linsolve_t& solver,
                                    bool full,
                                    bool quick)
{
    const std::vector<matx_int64_t> sizes
        = quick ? std::vector<matx_int64_t>{4, 8}
                : full ? std::vector<matx_int64_t>{32, 64, 128}
                       : std::vector<matx_int64_t>{32, 64};
    for (matx_int64_t size : sizes) run_dense_solve_extended_case<T>(solver, size);
}

template <typename T>
void run_core_suite(bool full, bool quick)
{
    std::vector<matx_int64_t> sizes
        = quick ? std::vector<matx_int64_t>{16, 64}
                : std::vector<matx_int64_t>{16, 128, 512};
    if (full) sizes.push_back(1024);
    for (matx_int64_t size : sizes) {
        run_core_case<T>(size);
    }

    std::vector<matx_int64_t> scalar_sizes
        = quick ? std::vector<matx_int64_t>{64}
                : std::vector<matx_int64_t>{64, 512, 2048};
    for (matx_int64_t size : scalar_sizes) {
        run_dense_scalar_case<T>(size, MATX_COL_MAJOR);
        run_dense_scalar_case<T>(size, MATX_ROW_MAJOR);
    }
}

template <typename T>
void run_io_suite(bool full, bool quick)
{
    const std::vector<matx_int64_t> dense_sizes
        = quick ? std::vector<matx_int64_t>{16, 64}
                : full ? std::vector<matx_int64_t>{128, 512, 1024}
                       : std::vector<matx_int64_t>{128, 512};
    const std::vector<matx_int64_t> sparse_sizes
        = quick ? std::vector<matx_int64_t>{16, 64}
                : full ? std::vector<matx_int64_t>{128, 2048, 8192}
                       : std::vector<matx_int64_t>{128, 2048};
    for (matx_int64_t size : dense_sizes) run_io_dense_case<T>(size);
    for (matx_int64_t size : dense_sizes) run_io_vector_case<T>(size);
    for (matx_int64_t size : sparse_sizes) run_io_sparse_case<T>(size);
}

template <typename T>
void run_dense_extended_suite(const matx_dense_backend_t& backend, bool full, bool quick)
{
    const std::vector<matx_int64_t> sizes
        = quick ? std::vector<matx_int64_t>{16, 32}
                : full ? std::vector<matx_int64_t>{64, 128, 256}
                       : std::vector<matx_int64_t>{64, 128};
    for (matx_int64_t size : sizes) run_dense_extended_case<T>(backend, size);
}

template <typename T>
void run_sparse_utility_suite(const matx_sparse_backend_t& backend, bool full, bool quick)
{
    const std::vector<matx_int64_t> sizes
        = quick ? std::vector<matx_int64_t>{16, 64}
                : full ? std::vector<matx_int64_t>{512, 2048, 8192}
                       : std::vector<matx_int64_t>{128, 512};
    for (matx_int64_t size : sizes) run_sparse_utility_case<T>(backend, size);
}

template <typename T>
void run_sparse_product_suite(const matx_sparse_backend_t& backend, bool full, bool quick)
{
    const std::vector<matx_int64_t> sizes
        = quick ? std::vector<matx_int64_t>{16, 64}
                : full ? std::vector<matx_int64_t>{64, 256, 512}
                       : std::vector<matx_int64_t>{64, 256};
    for (matx_int64_t size : sizes) run_sparse_product_case<T>(backend, size);
}

template <typename T>
void run_gemm_case(const matx_dense_backend_t& backend,
                   matx_int64_t m,
                   matx_int64_t k,
                   matx_int64_t n)
{
    using O = Ops<T>;
    using Dense = typename O::Dense;
    matx_alloc_t alloc = matx_alloc_default();
    Dense a = nullptr, b = nullptr, c = nullptr;

    auto setup_start = Clock::now();
    check_status(O::dense_create(&alloc, &a, m, k), "create A");
    check_status(O::dense_create(&alloc, &b, k, n), "create B");
    check_status(O::dense_create(&alloc, &c, m, n), "create C");
    for (matx_int64_t col = 0; col < k; ++col)
        for (matx_int64_t row = 0; row < m; ++row)
            a->data[row + col * a->stride] = generated_value<T>(row + col * m, 1);
    for (matx_int64_t col = 0; col < n; ++col)
        for (matx_int64_t row = 0; row < k; ++row)
            b->data[row + col * b->stride] = generated_value<T>(row + col * k, 2);
    const double setup = elapsed_us(setup_start);

    auto invoke = [&]() { check_status(O::gemm(&backend, a, b, c), "GEMM"); };
    const Timing timing = measure(invoke);
    double max_error = 0.0;
    for (int sample = 0; sample < 9; ++sample) {
        const matx_int64_t row = (m - 1) * sample / 8;
        const matx_int64_t col = (n - 1) * sample / 8;
        T expected = O::zero();
        for (matx_int64_t inner = 0; inner < k; ++inner) {
            expected = O::add(expected,
                              O::multiply(a->data[row + inner * a->stride],
                                          b->data[inner + col * b->stride]));
        }
        const double error = O::difference(c->data[row + col * c->stride], expected);
        max_error = std::max(max_error, error);
        if (error > 1e-9 * (1.0 + O::magnitude(expected))) {
            throw std::runtime_error("GEMM result check failed");
        }
    }
    const double flops = O::flop_factor() * static_cast<double>(m) * n * k;
    csv_row("gemm", O::name, matx_blas_backend_name(backend.kind),
            m, n, k, 0, 0.0, 1, setup, timing, flops, max_error,
            sizeof(T) * static_cast<double>(m * k + k * n + m * n));

    O::dense_destroy(&alloc, a);
    O::dense_destroy(&alloc, b);
    O::dense_destroy(&alloc, c);
}

matx_int64_t dense_offset(matx_layout_t layout,
                          matx_int64_t row,
                          matx_int64_t col,
                          matx_int64_t stride)
{
    return layout == MATX_COL_MAJOR ? row + col * stride : row * stride + col;
}

template <typename T, typename Function, typename Verify>
void run_vector_measurement(const char* operation,
                            const char* backend,
                            matx_int64_t n,
                            double setup_us,
                            double flops,
                            double bytes_moved,
                            Function&& function,
                            Verify&& verify,
                            const char* layout = "N/A");

template <typename T>
void run_dense_stream_case(const matx_dense_backend_t& backend,
                           matx_int64_t n,
                           matx_layout_t layout)
{
    using O = Ops<T>;
    using Dense = typename O::Dense;
    using Vector = typename O::Vector;
    matx_alloc_t alloc = matx_alloc_default();
    Dense a = nullptr, b = nullptr, c = nullptr, transposed = nullptr;
    Vector x = nullptr, y = nullptr;
    const auto setup_start = Clock::now();
    check_status(O::dense_create(&alloc, &a, n, n, layout), "create dense A");
    check_status(O::dense_create(&alloc, &b, n, n, layout), "create dense B");
    check_status(O::dense_create(&alloc, &c, n, n, layout), "create dense C");
    check_status(O::dense_create(&alloc, &transposed, n, n, layout), "create transpose output");
    check_status(O::vector_create(&alloc, &x, n), "create dense vector x");
    check_status(O::vector_create(&alloc, &y, n), "create dense vector y");
    std::vector<T> host_a(static_cast<size_t>(n * n));
    std::vector<T> host_b(static_cast<size_t>(n * n));
    std::vector<T> host_x(static_cast<size_t>(n));
    for (matx_int64_t col = 0; col < n; ++col) {
        for (matx_int64_t row = 0; row < n; ++row) {
            const T av = generated_value<T>(static_cast<uint64_t>(row + col * n), 53);
            const T bv = generated_value<T>(static_cast<uint64_t>(row + col * n), 59);
            const size_t index = static_cast<size_t>(row * n + col);
            host_a[index] = av;
            host_b[index] = bv;
            a->data[dense_offset(layout, row, col, a->stride)] = av;
            b->data[dense_offset(layout, row, col, b->stride)] = bv;
        }
    }
    for (matx_int64_t i = 0; i < n; ++i) {
        host_x[static_cast<size_t>(i)] = generated_value<T>(static_cast<uint64_t>(i), 61);
        x->data[i * x->stride] = host_x[static_cast<size_t>(i)];
    }
    const double setup = elapsed_us(setup_start);
    const char* backend_name = matx_blas_backend_name(backend.kind);
    const char* layout_name = layout == MATX_COL_MAJOR ? "COL_MAJOR" : "ROW_MAJOR";
    const double matrix_bytes = static_cast<double>(n) * n * sizeof(T);
    const double scalar_flops = std::is_same_v<T, double> ? 1.0 : 6.0;

    auto geadd = [&]() {
        if constexpr (std::is_same_v<T, double>)
            check_status(matx_geadd_d_i8(&backend, 1.0, a, 0.0, b), "GEADD");
        else
            check_status(matx_geadd_z_i8(&backend, O::one(), a, O::zero(), b), "GEADD");
    };
    run_vector_measurement<T>("geadd", backend_name, n, setup,
                              scalar_flops * n * n, 2.0 * matrix_bytes, geadd, [&]() {
        double error = 0.0;
        for (matx_int64_t col = 0; col < n; ++col)
            for (matx_int64_t row = 0; row < n; ++row)
                error = std::max(error, O::difference(
                    b->data[dense_offset(layout, row, col, b->stride)],
                    host_a[static_cast<size_t>(row * n + col)]));
        return error;
    }, layout_name);

    for (matx_int64_t col = 0; col < n; ++col)
            for (matx_int64_t row = 0; row < n; ++row)
                b->data[dense_offset(layout, row, col, b->stride)] = host_b[static_cast<size_t>(row * n + col)];

    auto hadamard = [&]() {
        if constexpr (std::is_same_v<T, double>)
            check_status(matx_hadamard_d_i8(&backend, a, b, c), "Hadamard");
        else
            check_status(matx_hadamard_z_i8(&backend, a, b, c), "Hadamard");
    };
    run_vector_measurement<T>("hadamard", backend_name, n, setup,
                              (std::is_same_v<T, double> ? 1.0 : 6.0) * n * n,
                              3.0 * matrix_bytes, hadamard, [&]() {
        double error = 0.0;
        for (matx_int64_t col = 0; col < n; ++col)
            for (matx_int64_t row = 0; row < n; ++row) {
                const size_t index = static_cast<size_t>(row * n + col);
                error = std::max(error, O::difference(c->data[dense_offset(layout, row, col, c->stride)],
                    O::multiply(host_a[index], host_b[index])));
            }
        return error;
    }, layout_name);

    auto transpose = [&]() {
        if constexpr (std::is_same_v<T, double>)
            check_status(matx_transpose_d_i8(&backend, a, transposed), "transpose");
        else
            check_status(matx_transpose_z_i8(&backend, a, transposed), "transpose");
    };
    run_vector_measurement<T>("transpose", backend_name, n, setup, 0.0,
                              2.0 * matrix_bytes, transpose, [&]() {
        double error = 0.0;
        for (matx_int64_t col = 0; col < n; ++col)
            for (matx_int64_t row = 0; row < n; ++row)
                error = std::max(error, O::difference(
                    transposed->data[dense_offset(layout, row, col, transposed->stride)],
                    host_a[static_cast<size_t>(col * n + row)]));
        return error;
    }, layout_name);

    if constexpr (std::is_same_v<T, matx_complex_d_t>) {
        auto conjugate_transpose = [&]() {
            check_status(matx_conj_transpose_z_i8(&backend, a, transposed),
                         "conjugate transpose");
        };
        run_vector_measurement<T>("conjugate_transpose", backend_name, n, setup, 0.0,
                                  2.0 * matrix_bytes, conjugate_transpose, [&]() {
            double error = 0.0;
            for (matx_int64_t col = 0; col < n; ++col) {
                for (matx_int64_t row = 0; row < n; ++row) {
                    const T source = host_a[static_cast<size_t>(col * n + row)];
                    const T expected = O::make(source.real, -source.imag);
                    error = std::max(error, O::difference(
                        transposed->data[dense_offset(layout, row, col, transposed->stride)],
                        expected));
                }
            }
            return error;
        }, layout_name);
    }

    auto gemv = [&]() {
        if constexpr (std::is_same_v<T, double>)
            check_status(matx_gemv_d_i8(&backend, MATX_NO_TRANS, 1.0, a, x, 0.0, y), "GEMV");
        else
            check_status(matx_gemv_z_i8(&backend, MATX_NO_TRANS, O::one(), a, x, O::zero(), y), "GEMV");
    };
    run_vector_measurement<T>("gemv", backend_name, n, setup,
                              (std::is_same_v<T, double> ? 2.0 : 8.0) * n * n,
                              matrix_bytes + 2.0 * n * sizeof(T), gemv, [&]() {
        double error = 0.0;
        for (matx_int64_t row = 0; row < n; ++row) {
            T expected = O::zero();
            for (matx_int64_t col = 0; col < n; ++col)
                expected = O::add(expected, O::multiply(host_a[static_cast<size_t>(row * n + col)], host_x[static_cast<size_t>(col)]));
            error = std::max(error, O::difference(y->data[row * y->stride], expected));
        }
        return error;
    }, layout_name);

    double norm1 = 0.0, norminf = 0.0, normfro = 0.0;
    double expected_norm1 = 0.0, expected_norminf = 0.0, sum_squares = 0.0;
    for (matx_int64_t col = 0; col < n; ++col) {
        double col_sum = 0.0;
        for (matx_int64_t row = 0; row < n; ++row) {
            const double value = O::magnitude(host_a[static_cast<size_t>(row * n + col)]);
            col_sum += value;
            sum_squares += value * value;
        }
        expected_norm1 = std::max(expected_norm1, col_sum);
    }
    for (matx_int64_t row = 0; row < n; ++row) {
        double row_sum = 0.0;
        for (matx_int64_t col = 0; col < n; ++col)
            row_sum += O::magnitude(host_a[static_cast<size_t>(row * n + col)]);
        expected_norminf = std::max(expected_norminf, row_sum);
    }
    const double expected_normfro = std::sqrt(sum_squares);
    auto run_norm = [&](const char* op, auto&& invoke, double& result, double expected) {
        run_vector_measurement<T>(op, backend_name, n, setup, n * n, matrix_bytes,
                                  std::forward<decltype(invoke)>(invoke), [&]() {
            return std::abs(result - expected) / (1.0 + expected);
        }, layout_name);
    };
    if constexpr (std::is_same_v<T, double>) {
        run_norm("mat_norm1", [&]() { check_status(matx_mat_norm1_d_i8(&backend, a, &norm1), "matrix norm1"); }, norm1, expected_norm1);
        run_norm("mat_norminf", [&]() { check_status(matx_mat_norminf_d_i8(&backend, a, &norminf), "matrix norminf"); }, norminf, expected_norminf);
        run_norm("mat_normfro", [&]() { check_status(matx_mat_normfro_d_i8(&backend, a, &normfro), "matrix normfro"); }, normfro, expected_normfro);
    } else {
        run_norm("mat_norm1", [&]() { check_status(matx_mat_norm1_z_i8(&backend, a, &norm1), "matrix norm1"); }, norm1, expected_norm1);
        run_norm("mat_norminf", [&]() { check_status(matx_mat_norminf_z_i8(&backend, a, &norminf), "matrix norminf"); }, norminf, expected_norminf);
        run_norm("mat_normfro", [&]() { check_status(matx_mat_normfro_z_i8(&backend, a, &normfro), "matrix normfro"); }, normfro, expected_normfro);
    }

    O::dense_destroy(&alloc, a);
    O::dense_destroy(&alloc, b);
    O::dense_destroy(&alloc, c);
    O::dense_destroy(&alloc, transposed);
    O::vector_destroy(&alloc, x);
    O::vector_destroy(&alloc, y);
}

template <typename T>
double vector_reference_nrm2(const std::vector<T>& values)
{
    double sum = 0.0;
    for (const T& value : values) {
        const double magnitude = Ops<T>::magnitude(value);
        sum += magnitude * magnitude;
    }
    return std::sqrt(sum);
}

template <typename T, typename Function, typename Verify>
void run_vector_measurement(const char* operation,
                            const char* backend,
                            matx_int64_t n,
                            double setup_us,
                            double flops,
                            double bytes_moved,
                            Function&& function,
                            Verify&& verify,
                            const char* layout)
{
    const Timing timing = measure(std::forward<Function>(function));
    const double error = verify();
    if (!std::isfinite(error) || error > 1e-8) {
        throw std::runtime_error(std::string(operation) + " result check failed: "
                                 + std::to_string(error));
    }
    csv_row(operation, Ops<T>::name, backend, n, 1, 1, 0, 0.0, 1,
            setup_us, timing, flops, error, bytes_moved, layout);
}

template <typename T>
void run_vector_case(const matx_vec_backend_t& backend, matx_int64_t n)
{
    using O = Ops<T>;
    using Vector = typename O::Vector;
    matx_alloc_t alloc = matx_alloc_default();
    Vector x = nullptr, y = nullptr, z = nullptr;
    const auto setup_start = Clock::now();
    check_status(O::vector_create(&alloc, &x, n), "create vector x");
    check_status(O::vector_create(&alloc, &y, n), "create vector y");
    check_status(O::vector_create(&alloc, &z, n), "create vector z");
    std::vector<T> host_x(static_cast<size_t>(n));
    std::vector<T> host_y(static_cast<size_t>(n));
    for (matx_int64_t i = 0; i < n; ++i) {
        const T xv = generated_value<T>(static_cast<uint64_t>(i), 31);
        const T yv = generated_value<T>(static_cast<uint64_t>(i), 37);
        host_x[static_cast<size_t>(i)] = xv;
        host_y[static_cast<size_t>(i)] = yv;
        x->data[i * x->stride] = xv;
        y->data[i * y->stride] = yv;
    }
    const double setup = elapsed_us(setup_start);
    const double element_bytes = static_cast<double>(sizeof(T));
    const double bytes = element_bytes * static_cast<double>(n);
    const char* backend_name = matx_vec_backend_name(backend.kind);

    if constexpr (std::is_same_v<T, double>) {
        auto scale = [&]() {
            check_status(matx_vec_scal_d_i8(&backend, 0.5, x), "vector scale");
            check_status(matx_vec_scal_d_i8(&backend, 2.0, x), "vector scale");
        };
        run_vector_measurement<T>("vec_scal_pair", backend_name, n, setup,
                                  2.0 * n, 4.0 * bytes, scale, [&]() {
            double error = 0.0;
            for (matx_int64_t i = 0; i < n; ++i)
                error = std::max(error, std::abs(x->data[i * x->stride] - host_x[static_cast<size_t>(i)]));
            return error;
        });
        run_vector_measurement<T>("vec_copy", backend_name, n, setup, 0.0, 2.0 * bytes,
                                  [&]() { check_status(matx_vec_copy_d_i8(&backend, x, z), "vector copy"); }, [&]() {
            double error = 0.0;
            for (matx_int64_t i = 0; i < n; ++i)
                error = std::max(error, std::abs(z->data[i * z->stride] - host_x[static_cast<size_t>(i)]));
            return error;
        });
        auto swap_pair = [&]() {
            check_status(matx_vec_swap_d_i8(&backend, x, y), "vector swap");
            check_status(matx_vec_swap_d_i8(&backend, x, y), "vector swap");
        };
        run_vector_measurement<T>("vec_swap_pair", backend_name, n, setup, 0.0, 4.0 * bytes,
                                  swap_pair, [&]() {
            double error = 0.0;
            for (matx_int64_t i = 0; i < n; ++i) {
                const size_t index = static_cast<size_t>(i);
                error = std::max(error, std::abs(x->data[i * x->stride] - host_x[index]));
                error = std::max(error, std::abs(y->data[i * y->stride] - host_y[index]));
            }
            return error;
        });
        double dot = 0.0;
        double expected_dot = 0.0;
        for (matx_int64_t i = 0; i < n; ++i)
            expected_dot += host_x[static_cast<size_t>(i)] * host_y[static_cast<size_t>(i)];
        run_vector_measurement<T>("vec_dot", backend_name, n, setup, 2.0 * n, 2.0 * bytes,
                                  [&]() { check_status(matx_vec_dot_d_i8(&backend, x, y, &dot), "vector dot"); }, [&]() {
            return std::abs(dot - expected_dot) / (1.0 + std::abs(expected_dot));
        });
        double norm = 0.0;
        run_vector_measurement<T>("vec_nrm2", backend_name, n, setup, 2.0 * n, bytes,
                                  [&]() { check_status(matx_vec_nrm2_d_i8(&backend, x, &norm), "vector nrm2"); }, [&]() {
            return std::abs(norm - vector_reference_nrm2(host_x)) / (1.0 + norm);
        });
        double asum = 0.0;
        double expected_asum = 0.0;
        for (double value : host_x) expected_asum += std::abs(value);
        run_vector_measurement<T>("vec_asum", backend_name, n, setup, n, bytes,
                                  [&]() { check_status(matx_vec_asum_d_i8(&backend, x, &asum), "vector asum"); }, [&]() {
            return std::abs(asum - expected_asum) / (1.0 + expected_asum);
        });
        matx_int64_t index = 0;
        matx_int64_t expected_index = 0;
        for (matx_int64_t i = 1; i < n; ++i)
            if (std::abs(host_x[static_cast<size_t>(i)]) > std::abs(host_x[static_cast<size_t>(expected_index)])) expected_index = i;
        run_vector_measurement<T>("vec_iamax", backend_name, n, setup, n, bytes,
                                  [&]() { check_status(matx_vec_iamax_d_i8(&backend, x, &index), "vector iamax"); }, [&]() {
            return index == expected_index ? 0.0 : 1.0;
        });
        auto axpy_pair = [&]() {
            check_status(matx_vec_axpy_d_i8(&backend, 1.0, x, y), "vector axpy");
            check_status(matx_vec_axpy_d_i8(&backend, -1.0, x, y), "vector axpy");
        };
        run_vector_measurement<T>("vec_axpy_pair", backend_name, n, setup, 4.0 * n, 4.0 * bytes,
                                  axpy_pair, [&]() {
            double error = 0.0;
            for (matx_int64_t i = 0; i < n; ++i)
                error = std::max(error, std::abs(y->data[i * y->stride] - host_y[static_cast<size_t>(i)]));
            return error;
        });
        double norm1 = 0.0, norm2 = 0.0, norminf = 0.0;
        auto verify_norm1 = [&]() {
            double expected = 0.0;
            for (double value : host_x) expected += std::abs(value);
            return std::abs(norm1 - expected) / (1.0 + expected);
        };
        auto verify_norm2 = [&]() {
            return std::abs(norm2 - vector_reference_nrm2(host_x)) / (1.0 + norm2);
        };
        double expected_inf = 0.0;
        for (double value : host_x) expected_inf = std::max(expected_inf, std::abs(value));
        run_vector_measurement<T>("vec_norm1", backend_name, n, setup, n, bytes,
                                  [&]() { check_status(matx_vec_norm1_d_i8(&backend, x, &norm1), "vector norm1"); }, verify_norm1);
        run_vector_measurement<T>("vec_norm2", backend_name, n, setup, 2.0 * n, bytes,
                                  [&]() { check_status(matx_vec_norm2_d_i8(&backend, x, &norm2), "vector norm2"); }, verify_norm2);
        run_vector_measurement<T>("vec_norminf", backend_name, n, setup, n, bytes,
                                  [&]() { check_status(matx_vec_norminf_d_i8(&backend, x, &norminf), "vector norminf"); }, [&]() {
            return std::abs(norminf - expected_inf) / (1.0 + expected_inf);
        });
    } else {
        const T one = O::one();
        const T half = O::make(0.5);
        const T two = O::make(2.0);
        const T minus_one = O::make(-1.0);
        auto scale = [&]() {
            check_status(matx_vec_scal_z_i8(&backend, half, x), "vector scale");
            check_status(matx_vec_scal_z_i8(&backend, two, x), "vector scale");
        };
        run_vector_measurement<T>("vec_scal_pair", backend_name, n, setup,
                                  12.0 * n, 4.0 * bytes, scale, [&]() {
            double error = 0.0;
            for (matx_int64_t i = 0; i < n; ++i)
                error = std::max(error, O::difference(x->data[i * x->stride], host_x[static_cast<size_t>(i)]));
            return error;
        });
        run_vector_measurement<T>("vec_copy", backend_name, n, setup, 0.0, 2.0 * bytes,
                                  [&]() { check_status(matx_vec_copy_z_i8(&backend, x, z), "vector copy"); }, [&]() {
            double error = 0.0;
            for (matx_int64_t i = 0; i < n; ++i)
                error = std::max(error, O::difference(z->data[i * z->stride], host_x[static_cast<size_t>(i)]));
            return error;
        });
        auto swap_pair = [&]() {
            check_status(matx_vec_swap_z_i8(&backend, x, y), "vector swap");
            check_status(matx_vec_swap_z_i8(&backend, x, y), "vector swap");
        };
        run_vector_measurement<T>("vec_swap_pair", backend_name, n, setup, 0.0, 4.0 * bytes,
                                  swap_pair, [&]() {
            double error = 0.0;
            for (matx_int64_t i = 0; i < n; ++i) {
                const size_t index = static_cast<size_t>(i);
                error = std::max(error, O::difference(x->data[i * x->stride], host_x[index]));
                error = std::max(error, O::difference(y->data[i * y->stride], host_y[index]));
            }
            return error;
        });
        T dotu = O::zero(), dotc = O::zero(), expected_dotu = O::zero(), expected_dotc = O::zero();
        for (matx_int64_t i = 0; i < n; ++i) {
            const T xv = host_x[static_cast<size_t>(i)], yv = host_y[static_cast<size_t>(i)];
            expected_dotu = O::add(expected_dotu, O::multiply(xv, yv));
            const T conjugated_x = O::make(xv.real, -xv.imag);
            expected_dotc = O::add(expected_dotc, O::multiply(conjugated_x, yv));
        }
        run_vector_measurement<T>("vec_dotu", backend_name, n, setup, 8.0 * n, 2.0 * bytes,
                                  [&]() { check_status(matx_vec_dotu_z_i8(&backend, x, y, &dotu), "vector dotu"); }, [&]() {
            return O::difference(dotu, expected_dotu) / (1.0 + O::magnitude(expected_dotu));
        });
        run_vector_measurement<T>("vec_dotc", backend_name, n, setup, 8.0 * n, 2.0 * bytes,
                                  [&]() { check_status(matx_vec_dotc_z_i8(&backend, x, y, &dotc), "vector dotc"); }, [&]() {
            return O::difference(dotc, expected_dotc) / (1.0 + O::magnitude(expected_dotc));
        });
        double norm = 0.0;
        run_vector_measurement<T>("vec_nrm2", backend_name, n, setup, 6.0 * n, bytes,
                                  [&]() { check_status(matx_vec_nrm2_z_i8(&backend, x, &norm), "vector nrm2"); }, [&]() {
            return std::abs(norm - vector_reference_nrm2(host_x)) / (1.0 + norm);
        });
        double asum = 0.0, expected_asum = 0.0;
        for (const T& value : host_x) expected_asum += std::abs(value.real) + std::abs(value.imag);
        run_vector_measurement<T>("vec_asum", backend_name, n, setup, 2.0 * n, bytes,
                                  [&]() { check_status(matx_vec_asum_z_i8(&backend, x, &asum), "vector asum"); }, [&]() {
            return std::abs(asum - expected_asum) / (1.0 + expected_asum);
        });
        matx_int64_t index = 0, expected_index = 0;
        auto abs1 = [](T value) { return std::abs(value.real) + std::abs(value.imag); };
        for (matx_int64_t i = 1; i < n; ++i)
            if (abs1(host_x[static_cast<size_t>(i)]) > abs1(host_x[static_cast<size_t>(expected_index)])) expected_index = i;
        run_vector_measurement<T>("vec_iamax", backend_name, n, setup, n, bytes,
                                  [&]() { check_status(matx_vec_iamax_z_i8(&backend, x, &index), "vector iamax"); }, [&]() {
            return index == expected_index ? 0.0 : 1.0;
        });
        auto axpy_pair = [&]() {
            check_status(matx_vec_axpy_z_i8(&backend, one, x, y), "vector axpy");
            check_status(matx_vec_axpy_z_i8(&backend, minus_one, x, y), "vector axpy");
        };
        run_vector_measurement<T>("vec_axpy_pair", backend_name, n, setup, 24.0 * n, 4.0 * bytes,
                                  axpy_pair, [&]() {
            double error = 0.0;
            for (matx_int64_t i = 0; i < n; ++i)
                error = std::max(error, O::difference(y->data[i * y->stride], host_y[static_cast<size_t>(i)]));
            return error;
        });
        double norm1 = 0.0, norm2 = 0.0, norminf = 0.0;
        double expected1 = 0.0, expected_inf = 0.0;
        for (const T& value : host_x) {
            expected1 += O::magnitude(value);
            expected_inf = std::max(expected_inf, O::magnitude(value));
        }
        run_vector_measurement<T>("vec_norm1", backend_name, n, setup, 2.0 * n, bytes,
                                  [&]() { check_status(matx_vec_norm1_z_i8(&backend, x, &norm1), "vector norm1"); }, [&]() {
            return std::abs(norm1 - expected1) / (1.0 + expected1);
        });
        run_vector_measurement<T>("vec_norm2", backend_name, n, setup, 6.0 * n, bytes,
                                  [&]() { check_status(matx_vec_norm2_z_i8(&backend, x, &norm2), "vector norm2"); }, [&]() {
            return std::abs(norm2 - vector_reference_nrm2(host_x)) / (1.0 + norm2);
        });
        run_vector_measurement<T>("vec_norminf", backend_name, n, setup, 2.0 * n, bytes,
                                  [&]() { check_status(matx_vec_norminf_z_i8(&backend, x, &norminf), "vector norminf"); }, [&]() {
            return std::abs(norminf - expected_inf) / (1.0 + expected_inf);
        });
    }

    O::vector_destroy(&alloc, x);
    O::vector_destroy(&alloc, y);
    O::vector_destroy(&alloc, z);
}

template <typename T>
void run_vector_cross(const matx_vec_backend_t& backend)
{
    using O = Ops<T>;
    using Vector = typename O::Vector;
    matx_alloc_t alloc = matx_alloc_default();
    Vector x = nullptr, y = nullptr, out = nullptr;
    const auto setup_start = Clock::now();
    check_status(O::vector_create(&alloc, &x, 3), "create cross x");
    check_status(O::vector_create(&alloc, &y, 3), "create cross y");
    check_status(O::vector_create(&alloc, &out, 3), "create cross output");
    std::vector<T> xv(3), yv(3), expected(3);
    for (matx_int64_t i = 0; i < 3; ++i) {
        xv[static_cast<size_t>(i)] = generated_value<T>(static_cast<uint64_t>(i), 43);
        yv[static_cast<size_t>(i)] = generated_value<T>(static_cast<uint64_t>(i), 47);
        x->data[i * x->stride] = xv[static_cast<size_t>(i)];
        y->data[i * y->stride] = yv[static_cast<size_t>(i)];
    }
    auto subtract = [](T a, T b) {
        if constexpr (std::is_same_v<T, double>)
            return a - b;
        else
            return O::add(a, O::make(-b.real, -b.imag));
    };
    expected[0] = subtract(O::multiply(xv[1], yv[2]), O::multiply(xv[2], yv[1]));
    expected[1] = subtract(O::multiply(xv[2], yv[0]), O::multiply(xv[0], yv[2]));
    expected[2] = subtract(O::multiply(xv[0], yv[1]), O::multiply(xv[1], yv[0]));
    const double setup = elapsed_us(setup_start);
    auto invoke = [&]() {
        if constexpr (std::is_same_v<T, double>)
            check_status(matx_vec_cross_d_i8(&backend, x, y, out), "vector cross");
        else
            check_status(matx_vec_cross_z_i8(&backend, x, y, out), "vector cross");
    };
    run_vector_measurement<T>("vec_cross", matx_vec_backend_name(backend.kind), 3,
                              setup, 9.0, 6.0 * sizeof(T), invoke, [&]() {
        double error = 0.0;
        for (matx_int64_t i = 0; i < 3; ++i)
            error = std::max(error, O::difference(out->data[i * out->stride], expected[static_cast<size_t>(i)]));
        return error;
    });
    O::vector_destroy(&alloc, x);
    O::vector_destroy(&alloc, y);
    O::vector_destroy(&alloc, out);
}

template <typename T>
void run_sparse_case(const matx_sparse_backend_t& backend,
                     matx_int64_t n,
                     double density,
                     matx_int64_t rhs_columns)
{
    using O = Ops<T>;
    using Sparse = typename O::Sparse;
    using Dense = typename O::Dense;
    using Vector = typename O::Vector;
    const matx_int64_t nnz = std::max<matx_int64_t>(
        1, static_cast<matx_int64_t>(static_cast<double>(n) * n * density));
    std::vector<matx_int64_t> rows(static_cast<size_t>(nnz));
    std::vector<matx_int64_t> cols(static_cast<size_t>(nnz));
    std::vector<T> values(static_cast<size_t>(nnz));
    const uint64_t cells = static_cast<uint64_t>(n) * static_cast<uint64_t>(n);
    for (matx_int64_t i = 0; i < nnz; ++i) {
        const uint64_t position
            = (static_cast<uint64_t>(i) * 104729ULL + 17ULL) % cells;
        rows[static_cast<size_t>(i)] = static_cast<matx_int64_t>(position / n);
        cols[static_cast<size_t>(i)] = static_cast<matx_int64_t>(position % n);
        values[static_cast<size_t>(i)] = generated_value<T>(static_cast<uint64_t>(i), 3);
    }

    matx_alloc_t alloc = matx_alloc_default();
    Sparse a = nullptr;
    Vector x = nullptr;
    Dense b = nullptr, c = nullptr;
    auto setup_start = Clock::now();
    check_status(O::sparse_create(&alloc, &a, n, nnz, rows.data(), cols.data(), values.data()),
                 "create COO");
    check_status(O::vector_create(&alloc, &x, n), "create x");
    Vector sparse_y = nullptr;
    check_status(O::vector_create(&alloc, &sparse_y, n), "create SpMV output");
    check_status(O::dense_create(&alloc, &b, n, rhs_columns), "create B");
    check_status(O::dense_create(&alloc, &c, n, rhs_columns), "create C");
    for (matx_int64_t i = 0; i < n; ++i) {
        x->data[i * x->stride] = generated_value<T>(static_cast<uint64_t>(i), 5);
    }
    for (matx_int64_t col = 0; col < rhs_columns; ++col)
        for (matx_int64_t row = 0; row < n; ++row)
            b->data[row + col * b->stride] = generated_value<T>(
                static_cast<uint64_t>(row + col * n), 7);
    const double setup = elapsed_us(setup_start);
    std::vector<T> expected_y(static_cast<size_t>(n), O::zero());
    for (matx_int64_t i = 0; i < nnz; ++i) {
        const size_t row = static_cast<size_t>(rows[static_cast<size_t>(i)]);
        const size_t col = static_cast<size_t>(cols[static_cast<size_t>(i)]);
        expected_y[row] = O::add(
            expected_y[row],
            O::multiply(values[static_cast<size_t>(i)], x->data[col * x->stride]));
    }
    std::vector<T> direct_y(static_cast<size_t>(n), O::zero());
    auto invoke_direct_spmv = [&]() {
        std::fill(direct_y.begin(), direct_y.end(), O::zero());
        for (matx_int64_t i = 0; i < nnz; ++i) {
            const size_t row = static_cast<size_t>(rows[static_cast<size_t>(i)]);
            const size_t col = static_cast<size_t>(cols[static_cast<size_t>(i)]);
            direct_y[row] = O::add(
                direct_y[row],
                O::multiply(values[static_cast<size_t>(i)], x->data[col * x->stride]));
        }
    };
    const Timing direct_spmv_timing = measure(invoke_direct_spmv);
    double direct_spmv_error = 0.0;
    for (matx_int64_t i = 0; i < n; ++i) {
        const double error = O::difference(direct_y[static_cast<size_t>(i)],
                                           expected_y[static_cast<size_t>(i)]);
        direct_spmv_error = std::max(direct_spmv_error, error);
        if (error > 1e-9 * (1.0 + O::magnitude(expected_y[static_cast<size_t>(i)]))) {
            throw std::runtime_error("direct COO SpMV result check failed");
        }
    }
    const double spmv_flops = O::flop_factor() * static_cast<double>(nnz);
    const double sparse_bytes = static_cast<double>(nnz)
        * (2.0 * sizeof(matx_int64_t) + sizeof(T)) + 2.0 * n * sizeof(T);
    csv_row("spmv_coo", O::name, "COO_LOOP", n, n, 1, nnz, density, 1,
            setup, direct_spmv_timing, spmv_flops, direct_spmv_error, sparse_bytes);

    int spmv_calls = 0;
    auto invoke_spmv = [&]() {
        ++spmv_calls;
        const matx_status_t status = O::spmv(&backend, a, x, sparse_y);
        if (status != MATX_OK) {
            std::cerr << "SpMV call " << spmv_calls << " failed: A=" << a->nrows << 'x'
                      << a->ncols << ", x=" << x->n << ", y=" << sparse_y->n
                      << ", data=" << static_cast<const void*>(a->values) << '/'
                      << static_cast<const void*>(x->data) << '/'
                      << static_cast<const void*>(sparse_y->data) << '\n';
            check_status(status, "SpMV");
        }
    };
    const Timing spmv_timing = measure(invoke_spmv);
    double spmv_error = 0.0;
    for (matx_int64_t i = 0; i < n; ++i) {
        const double error = O::difference(sparse_y->data[i * sparse_y->stride],
                                           expected_y[static_cast<size_t>(i)]);
        spmv_error = std::max(spmv_error, error);
        if (error > 1e-9 * (1.0 + O::magnitude(expected_y[static_cast<size_t>(i)]))) {
            throw std::runtime_error("SpMV result check failed");
        }
    }
    csv_row("spmv", O::name, "GRAPHBLAS+COO",
            n, n, 1, nnz, density, 1, setup, spmv_timing,
            spmv_flops, spmv_error, sparse_bytes);

    std::vector<T> expected_c(static_cast<size_t>(n * rhs_columns), O::zero());
    for (matx_int64_t i = 0; i < nnz; ++i) {
        const matx_int64_t row = rows[static_cast<size_t>(i)];
        const matx_int64_t col = cols[static_cast<size_t>(i)];
        for (matx_int64_t rhs = 0; rhs < rhs_columns; ++rhs) {
            const size_t out_index = static_cast<size_t>(row + rhs * n);
            const T product = O::multiply(values[static_cast<size_t>(i)],
                                          b->data[col + rhs * b->stride]);
            expected_c[out_index] = O::add(expected_c[out_index], product);
        }
    }
    std::vector<T> direct_c(static_cast<size_t>(n * rhs_columns), O::zero());
    auto invoke_direct_spmm = [&]() {
        std::fill(direct_c.begin(), direct_c.end(), O::zero());
        for (matx_int64_t i = 0; i < nnz; ++i) {
            const matx_int64_t row = rows[static_cast<size_t>(i)];
            const matx_int64_t col = cols[static_cast<size_t>(i)];
            for (matx_int64_t rhs = 0; rhs < rhs_columns; ++rhs) {
                const size_t out_index = static_cast<size_t>(row + rhs * n);
                direct_c[out_index] = O::add(
                    direct_c[out_index],
                    O::multiply(values[static_cast<size_t>(i)],
                                b->data[col + rhs * b->stride]));
            }
        }
    };
    const Timing direct_spmm_timing = measure(invoke_direct_spmm);
    double direct_spmm_error = 0.0;
    for (size_t i = 0; i < expected_c.size(); ++i) {
        const double error = O::difference(direct_c[i], expected_c[i]);
        direct_spmm_error = std::max(direct_spmm_error, error);
        if (error > 1e-9 * (1.0 + O::magnitude(expected_c[i]))) {
            throw std::runtime_error("direct COO SpMM result check failed");
        }
    }
    const double spmm_flops
        = O::flop_factor() * static_cast<double>(nnz) * rhs_columns;
    csv_row("spmm_coo", O::name, "COO_LOOP", n, n, rhs_columns,
            nnz, density, rhs_columns, setup, direct_spmm_timing,
            spmm_flops, direct_spmm_error,
            static_cast<double>(nnz) * (2.0 * sizeof(matx_int64_t) + sizeof(T))
                + 2.0 * n * rhs_columns * sizeof(T));

    auto invoke_spmm = [&]() { check_status(O::spmm(&backend, a, b, c), "SpMM"); };
    const Timing spmm_timing = measure(invoke_spmm);
    double spmm_error = 0.0;
    for (matx_int64_t col = 0; col < rhs_columns; ++col) {
        for (matx_int64_t row = 0; row < n; ++row) {
            const T expected = expected_c[static_cast<size_t>(row + col * n)];
            const double error
                = O::difference(c->data[row + col * c->stride], expected);
            spmm_error = std::max(spmm_error, error);
            if (error > 1e-9 * (1.0 + O::magnitude(expected))) {
                throw std::runtime_error("SpMM result check failed");
            }
        }
    }
    csv_row("spmm", O::name, "GRAPHBLAS+COO",
            n, n, rhs_columns, nnz, density, rhs_columns,
            setup, spmm_timing, spmm_flops, spmm_error,
            static_cast<double>(nnz) * (2.0 * sizeof(matx_int64_t) + sizeof(T))
                + 2.0 * n * rhs_columns * sizeof(T));

    std::vector<double> row_abs(static_cast<size_t>(n), 0.0);
    std::vector<double> col_abs(static_cast<size_t>(n), 0.0);
    double sum_squares = 0.0;
    for (matx_int64_t i = 0; i < nnz; ++i) {
        const size_t index = static_cast<size_t>(i);
        const double magnitude = O::magnitude(values[index]);
        row_abs[static_cast<size_t>(rows[index])] += magnitude;
        col_abs[static_cast<size_t>(cols[index])] += magnitude;
        sum_squares += magnitude * magnitude;
    }
    const double expected_norm1 = *std::max_element(col_abs.begin(), col_abs.end());
    const double expected_norminf = *std::max_element(row_abs.begin(), row_abs.end());
    const double expected_normfro = std::sqrt(sum_squares);
    auto run_sparse_norm = [&](const char* operation, auto&& invoke, double& result, double expected) {
        const Timing timing = measure(std::forward<decltype(invoke)>(invoke));
        const double error = std::abs(result - expected);
        if (!std::isfinite(error) || error > 1e-8 * (1.0 + expected))
            throw std::runtime_error(std::string(operation) + " result check failed: got "
                                     + std::to_string(result) + ", expected "
                                     + std::to_string(expected));
        csv_row(operation, O::name, matx_sparse_backend_name(backend.kind),
                n, n, 0, nnz, density, 1, setup, timing,
                static_cast<double>(nnz), error,
                static_cast<double>(nnz) * (2.0 * sizeof(matx_int64_t) + sizeof(T)));
    };
    double norm1 = 0.0, norminf = 0.0, normfro = 0.0;
    if constexpr (std::is_same_v<T, double>) {
        run_sparse_norm("sparse_norm1", [&]() {
            check_status(matx_norm1_mat_coo_d_i8(&backend, a, &norm1), "sparse norm1");
        }, norm1, expected_norm1);
        run_sparse_norm("sparse_norminf", [&]() {
            check_status(matx_norminf_mat_coo_d_i8(&backend, a, &norminf), "sparse norminf");
        }, norminf, expected_norminf);
        run_sparse_norm("sparse_normfro", [&]() {
            check_status(matx_normfro_mat_coo_d_i8(&backend, a, &normfro), "sparse normfro");
        }, normfro, expected_normfro);
    } else {
        run_sparse_norm("sparse_norm1", [&]() {
            check_status(matx_norm1_mat_coo_z_i8(&backend, a, &norm1), "sparse norm1");
        }, norm1, expected_norm1);
        run_sparse_norm("sparse_norminf", [&]() {
            check_status(matx_norminf_mat_coo_z_i8(&backend, a, &norminf), "sparse norminf");
        }, norminf, expected_norminf);
        run_sparse_norm("sparse_normfro", [&]() {
            check_status(matx_normfro_mat_coo_z_i8(&backend, a, &normfro), "sparse normfro");
        }, normfro, expected_normfro);
    }

    O::sparse_destroy(&alloc, a);
    O::vector_destroy(&alloc, x);
    O::vector_destroy(&alloc, sparse_y);
    O::dense_destroy(&alloc, b);
    O::dense_destroy(&alloc, c);
}

template <typename T>
void run_gemm_suite(const matx_dense_backend_t& backend, bool full, bool quick)
{
    std::vector<matx_int64_t> sizes
        = quick ? std::vector<matx_int64_t>{16, 128, 512}
                : std::vector<matx_int64_t>{16, 32, 64, 128, 256, 512, 1024, 2048};
    if (full) sizes.push_back(4096);
    for (matx_int64_t size : sizes) {
        run_gemm_case<T>(backend, size, size, size);
    }
    if (!quick) {
        run_gemm_case<T>(backend, 16, 256, 64);
        run_gemm_case<T>(backend, 64, 512, 32);
        run_gemm_case<T>(backend, 512, 128, 1024);
        run_gemm_case<T>(backend, 1024, 64, 512);
    }
}

template <typename T>
void run_sparse_suite(const matx_sparse_backend_t& backend, bool full, bool quick)
{
    std::vector<matx_int64_t> sizes
        = quick ? std::vector<matx_int64_t>{128, 1024, 4096}
                : std::vector<matx_int64_t>{64, 256, 1024, 4096, 8192};
    if (full) sizes.push_back(16384);
    const std::vector<double> densities
        = quick ? std::vector<double>{0.001}
                : std::vector<double>{0.0001, 0.001, 0.01};
    for (matx_int64_t size : sizes) {
        for (double density : densities) {
            run_sparse_case<T>(backend, size, density, 16);
        }
    }
}

template <typename T>
void run_vector_suite(const matx_vec_backend_t& backend, bool full, bool quick)
{
    std::vector<matx_int64_t> sizes
        = quick ? std::vector<matx_int64_t>{128, 4096}
                : std::vector<matx_int64_t>{32, 1024, 65536, 1048576};
    if (full) sizes.push_back(4194304);
    for (matx_int64_t size : sizes) run_vector_case<T>(backend, size);
    run_vector_cross<T>(backend);
}

template <typename T>
void run_dense_stream_suite(const matx_dense_backend_t& backend, bool full, bool quick)
{
    std::vector<matx_int64_t> sizes
        = quick ? std::vector<matx_int64_t>{32, 128}
                : std::vector<matx_int64_t>{32, 128, 512, 2048};
    if (full) sizes.push_back(4096);
    for (matx_layout_t layout : {MATX_COL_MAJOR, MATX_ROW_MAJOR})
        for (matx_int64_t size : sizes) run_dense_stream_case<T>(backend, size, layout);
}

template <typename T>
void run_dense_lu_suite(const matx_dense_linsolve_t& solver,
                        bool full,
                        bool quick,
                        bool solve_only = false)
{
    std::vector<matx_int64_t> sizes
        = solve_only ? std::vector<matx_int64_t>{1, 16, 31, 32, 33, 64, 99, 100, 512, 1999, 2000}
        : quick ? std::vector<matx_int64_t>{16, 64}
                : std::vector<matx_int64_t>{16, 64, 256, 1024};
    if (full && !solve_only) sizes.push_back(2048);
    for (matx_int64_t size : sizes) run_dense_lu_case<T>(solver, size, solve_only);
}

template <typename T>
void run_sparse_lu_suite(const matx_sparse_linsolve_t& solver,
                         bool full,
                         bool quick,
                         bool solve_only = false)
{
    std::vector<matx_int64_t> sizes
        = solve_only ? std::vector<matx_int64_t>{1, 16, 64, 99, 100, 512, 1999, 2000,
                                                 16384, 100000}
        : quick ? std::vector<matx_int64_t>{16, 64}
                : std::vector<matx_int64_t>{16, 64, 512, 4096};
    if (full && !solve_only) sizes.push_back(16384);
    for (matx_int64_t size : sizes) run_sparse_lu_case<T>(solver, size, solve_only);
}

} // namespace

int main(int argc, char** argv)
{
    bool quick = false;
    bool full = false;
    bool solve_only = false;
    bool coo_conversion_only = false;
    for (int i = 1; i < argc; ++i) {
        const std::string argument(argv[i]);
        if (argument == "--quick") quick = true;
        else if (argument == "--full") full = true;
        else if (argument == "--solve-only") solve_only = true;
        else if (argument == "--coo-conversion-only") coo_conversion_only = true;
        else if (argument == "--perf-counters") perf_counters_enabled = true;
        else if (argument == "--help") {
            std::cout << "Usage: matx_benchmarks [--quick] [--full] [--solve-only] [--coo-conversion-only] [--perf-counters]\n"
                      << "  --quick  run a small scale sweep for smoke checks\n"
                      << "  --full   include 4096 dense and 16384 sparse matrices\n"
                      << "  --solve-only  time reused LU/Cholesky solves across small, medium, and large cases\n"
                      << "  --coo-conversion-only  benchmark MatX COO-to-CSR and COO-to-CSC conversions\n"
                      << "  --perf-counters  append per-operation Linux perf counters to the CSV\n";
            return 0;
        } else {
            std::cerr << "Unknown argument: " << argument << '\n';
            return 2;
        }
    }

    std::cout << "operation,type,backend,m,n,k,nnz,density,rhs,setup_us,first_us,"
                 "steady_us,gflops,calls_per_sample,max_abs_error,gbytes_per_sec,layout,"
                 "perf_cycles_per_call,perf_instructions_per_call,perf_cache_misses_per_call,"
                 "perf_scope\n"
              << std::setprecision(8);
    try {
        if (coo_conversion_only) {
            run_coo_to_csr_suite();
            run_coo_to_csc_suite<double>();
            run_coo_to_csc_suite<matx_complex_d_t>();
            return 0;
        }
        const matx_dense_backend_t dense_backend = matx_blas_default();
        const matx_dense_linsolve_t dense_solver
            = matx_dense_linsolve_default(matx_alloc_default());
        const matx_sparse_linsolve_t sparse_solver
            = matx_sparse_linsolve_default(matx_alloc_default());
        const matx_sparse_linsolve_t sparse_cholesky_solver
            = matx_sparse_linsolve_by_type(MATX_LINSOLVE_BACKEND_SUITESPARSE_KLU,
                                           matx_alloc_default());
        const matx_sparse_backend_t sparse_backend = matx_sparse_default();
        const matx_vec_backend_t vector_backend = matx_vec_default();
        std::cerr << "Dense backend: "
                  << matx_blas_backend_name(dense_backend.kind) << '\n';
        std::cerr << "Sparse backend: "
                  << matx_sparse_backend_name(sparse_backend.kind) << '\n';
        std::cerr << "Vector backend: "
                  << matx_vec_backend_name(vector_backend.kind) << '\n';
        std::cerr << "Dense solver backend: "
                  << matx_dense_linsolve_backend_name(dense_solver.kind) << '\n';
        std::cerr << "Sparse solver backend: "
                  << matx_sparse_linsolve_backend_name(sparse_solver.kind) << '\n';
        std::cerr << "Compiler: " << MATX_COMPILER_ID << ' ' << MATX_COMPILER_VERSION
                  << ", OpenMP threads: " << omp_get_max_threads() << '\n';

        if (solve_only) {
            run_dense_lu_suite<double>(dense_solver, false, false, true);
            run_dense_lu_suite<matx_complex_d_t>(dense_solver, false, false, true);
            run_dense_cholesky_solve_suite<double>(dense_solver);
            run_dense_cholesky_solve_suite<matx_complex_d_t>(dense_solver);
            for (matx_sparse_linsolve_backend_kind_t kind : {
                     MATX_LINSOLVE_BACKEND_UMFPACK,
                     MATX_LINSOLVE_BACKEND_CXSPARSE,
                     MATX_LINSOLVE_BACKEND_MUMPS,
                     MATX_LINSOLVE_BACKEND_SUITESPARSE_KLU}) {
                const matx_sparse_linsolve_t solver
                    = matx_sparse_linsolve_by_type(kind, matx_alloc_default());
                run_sparse_lu_suite<double>(solver, false, false, true);
                run_sparse_lu_suite<matx_complex_d_t>(solver, false, false, true);
            }
            std::cerr << "SuperLU solve-only scan skipped: its LP64 BLAS calls are incompatible "
                         "with this OpenBLAS64 build (DTRSV reports invalid LDA)\n";
            run_sparse_cholesky_suite(sparse_cholesky_solver, false, false, true);
            check_status(matx_finalize(&sparse_backend), "finalize sparse backend");
            return 0;
        }

        run_gemm_suite<double>(dense_backend, full, quick);
        run_gemm_suite<matx_complex_d_t>(dense_backend, full, quick);
        run_dense_extended_suite<double>(dense_backend, full, quick);
        run_dense_extended_suite<matx_complex_d_t>(dense_backend, full, quick);
        run_core_suite<double>(full, quick);
        run_core_suite<matx_complex_d_t>(full, quick);
        const matx_int64_t core_math_size = quick ? 64 : full ? 512 : 128;
        run_core_unary_case(core_math_size, MATX_COL_MAJOR);
        run_core_unary_case(core_math_size, MATX_ROW_MAJOR);
        run_core_vector_api_case(core_math_size);
        run_core_dense_api_case(core_math_size, MATX_COL_MAJOR);
        run_core_dense_api_case(core_math_size, MATX_ROW_MAJOR);
        run_core_lifecycle_api_case(core_math_size);
        run_system_api_case("build-verify/benchmark-logs");
        run_io_suite<double>(full, quick);
        run_io_suite<matx_complex_d_t>(full, quick);
        run_dense_stream_suite<double>(dense_backend, full, quick);
        run_dense_stream_suite<matx_complex_d_t>(dense_backend, full, quick);
        run_dense_lu_suite<double>(dense_solver, full, quick);
        run_dense_lu_suite<matx_complex_d_t>(dense_solver, full, quick);
        run_dense_solve_extended_suite<double>(dense_solver, full, quick);
        run_dense_solve_extended_suite<matx_complex_d_t>(dense_solver, full, quick);
        run_sparse_lu_suite<double>(sparse_solver, full, quick);
        run_sparse_lu_suite<matx_complex_d_t>(sparse_solver, full, quick);
        run_vector_suite<double>(vector_backend, full, quick);
        run_vector_suite<matx_complex_d_t>(vector_backend, full, quick);
        run_sparse_utility_suite<double>(sparse_backend, full, quick);
        run_sparse_utility_suite<matx_complex_d_t>(sparse_backend, full, quick);
        run_sparse_product_suite<double>(sparse_backend, full, quick);
        run_sparse_product_suite<matx_complex_d_t>(sparse_backend, full, quick);
        run_sparse_cholesky_suite(sparse_cholesky_solver, full, quick);
        run_sparse_suite<double>(sparse_backend, full, quick);
        run_sparse_suite<matx_complex_d_t>(sparse_backend, full, quick);

        check_status(matx_finalize(&sparse_backend), "finalize sparse backend");
    } catch (const std::exception& error) {
        std::cerr << "Benchmark failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
