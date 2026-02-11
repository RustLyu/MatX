/*******************************************************************************
* Copyright 2019-2022 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

#ifndef _SPBLAS_HPP_
#define _SPBLAS_HPP_

#include <CL/sycl.hpp>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "oneapi/mkl/export.hpp"
#include "oneapi/mkl/types.hpp"
#include "oneapi/mkl/exceptions.hpp"

namespace oneapi {
namespace mkl {
namespace sparse {

enum class property : char {
    symmetric = 0,
    sorted    = 1,
};

struct matrix_handle;
typedef struct matrix_handle *matrix_handle_t;

DLL_EXPORT void init_matrix_handle(matrix_handle_t *handle);


// non-blocking version of release_matrix_handle, will schedule clean up of handle
// asychronously pending dependencies and return an event to track it
DLL_EXPORT sycl::event
release_matrix_handle(sycl::queue &queue,
                      matrix_handle_t *handle,
                      const std::vector<sycl::event> &dependencies = {});

[[deprecated("Use oneapi::mkl::sparse::release_matrix_handle(queue, handle, dependencies) instead.")]]
DLL_EXPORT void
release_matrix_handle(matrix_handle_t *handle,
                      const std::vector<sycl::event> &dependencies = {});

DLL_EXPORT void set_matrix_property(matrix_handle_t handle, property property_value);

DLL_EXPORT void set_csr_data(sycl::queue &queue,
                             matrix_handle_t handle,
                             const std::int32_t num_rows,
                             const std::int32_t num_cols,
                             index_base index,
                             sycl::buffer<std::int32_t, 1> &row_ptr,
                             sycl::buffer<std::int32_t, 1> &col_ind,
                             sycl::buffer<float, 1> &val);

DLL_EXPORT void set_csr_data(sycl::queue &queue,
                             matrix_handle_t handle,
                             const std::int64_t num_rows,
                             const std::int64_t num_cols,
                             index_base index,
                             sycl::buffer<std::int64_t, 1> &row_ptr,
                             sycl::buffer<std::int64_t, 1> &col_ind,
                             sycl::buffer<float, 1> &val);

DLL_EXPORT void set_csr_data(sycl::queue &queue,
                             matrix_handle_t handle,
                             const std::int32_t num_rows,
                             const std::int32_t num_cols,
                             index_base index,
                             sycl::buffer<std::int32_t, 1> &row_ptr,
                             sycl::buffer<std::int32_t, 1> &col_ind,
                             sycl::buffer<double, 1> &val);

DLL_EXPORT void set_csr_data(sycl::queue &queue,
                             matrix_handle_t handle,
                             const std::int64_t num_rows,
                             const std::int64_t num_cols,
                             index_base index,
                             sycl::buffer<std::int64_t, 1> &row_ptr,
                             sycl::buffer<std::int64_t, 1> &col_ind,
                             sycl::buffer<double, 1> &val);

DLL_EXPORT void set_csr_data(sycl::queue &queue,
                             matrix_handle_t handle,
                             const std::int32_t num_rows,
                             const std::int32_t num_cols,
                             index_base index,
                             sycl::buffer<std::int32_t, 1> &row_ptr,
                             sycl::buffer<std::int32_t, 1> &col_ind,
                             sycl::buffer<std::complex<float>, 1> &val);

DLL_EXPORT void set_csr_data(sycl::queue &queue,
                             matrix_handle_t handle,
                             const std::int64_t num_rows,
                             const std::int64_t num_cols,
                             index_base index,
                             sycl::buffer<std::int64_t, 1> &row_ptr,
                             sycl::buffer<std::int64_t, 1> &col_ind,
                             sycl::buffer<std::complex<float>, 1> &val);

DLL_EXPORT void set_csr_data(sycl::queue &queue,
                             matrix_handle_t handle,
                             const std::int32_t num_rows,
                             const std::int32_t num_cols,
                             index_base index,
                             sycl::buffer<std::int32_t, 1> &row_ptr,
                             sycl::buffer<std::int32_t, 1> &col_ind,
                             sycl::buffer<std::complex<double>, 1> &val);

DLL_EXPORT void set_csr_data(sycl::queue &queue,
                             matrix_handle_t handle,
                             const std::int64_t num_rows,
                             const std::int64_t num_cols,
                             index_base index,
                             sycl::buffer<std::int64_t, 1> &row_ptr,
                             sycl::buffer<std::int64_t, 1> &col_ind,
                             sycl::buffer<std::complex<double>, 1> &val);

DLL_EXPORT sycl::event set_csr_data(sycl::queue &queue,
                                    matrix_handle_t handle,
                                    const std::int32_t num_rows,
                                    const std::int32_t num_cols,
                                    index_base index,
                                    std::int32_t *row_ptr,
                                    std::int32_t *col_ind,
                                    float *val,
                                    const std::vector<sycl::event> &dependencies = {});

DLL_EXPORT sycl::event set_csr_data(sycl::queue &queue,
                                    matrix_handle_t handle,
                                    const std::int64_t num_rows,
                                    const std::int64_t num_cols,
                                    index_base index,
                                    std::int64_t *row_ptr,
                                    std::int64_t *col_ind,
                                    float *val,
                                    const std::vector<sycl::event> &dependencies = {});

DLL_EXPORT sycl::event set_csr_data(sycl::queue &queue,
                                    matrix_handle_t handle,
                                    const std::int32_t num_rows,
                                    const std::int32_t num_cols,
                                    index_base index,
                                    std::int32_t *row_ptr,
                                    std::int32_t *col_ind,
                                    double *val,
                                    const std::vector<sycl::event> &dependencies = {});

DLL_EXPORT sycl::event set_csr_data(sycl::queue &queue,
                                    matrix_handle_t handle,
                                    const std::int64_t num_rows,
                                    const std::int64_t num_cols,
                                    index_base index,
                                    std::int64_t *row_ptr,
                                    std::int64_t *col_ind,
                                    double *val,
                                    const std::vector<sycl::event> &dependencies = {});

DLL_EXPORT sycl::event set_csr_data(sycl::queue &queue,
                                    matrix_handle_t handle,
                                    const std::int32_t num_rows,
                                    const std::int32_t num_cols,
                                    index_base index,
                                    std::int32_t *row_ptr,
                                    std::int32_t *col_ind,
                                    std::complex<float> *val,
                                    const std::vector<sycl::event> &dependencies = {});

DLL_EXPORT sycl::event set_csr_data(sycl::queue &queue,
                                    matrix_handle_t handle,
                                    const std::int64_t num_rows,
                                    const std::int64_t num_cols,
                                    index_base index,
                                    std::int64_t *row_ptr,
                                    std::int64_t *col_ind,
                                    std::complex<float> *val,
                                    const std::vector<sycl::event> &dependencies = {});

DLL_EXPORT sycl::event set_csr_data(sycl::queue &queue,
                                    matrix_handle_t handle,
                                    const std::int32_t num_rows,
                                    const std::int32_t num_cols,
                                    index_base index,
                                    std::int32_t *row_ptr,
                                    std::int32_t *col_ind,
                                    std::complex<double> *val,
                                    const std::vector<sycl::event> &dependencies = {});

DLL_EXPORT sycl::event set_csr_data(sycl::queue &queue,
                                    matrix_handle_t handle,
                                    const std::int64_t num_rows,
                                    const std::int64_t num_cols,
                                    index_base index,
                                    std::int64_t *row_ptr,
                                    std::int64_t *col_ind,
                                    std::complex<double> *val,
                                    const std::vector<sycl::event> &dependencies = {});

[[deprecated("Use oneapi::mkl::sparse::set_csr_data(queue, handle, ...) instead.")]]
DLL_EXPORT void set_csr_data(matrix_handle_t handle,
                             const std::int32_t num_rows,
                             const std::int32_t num_cols,
                             index_base index,
                             sycl::buffer<std::int32_t, 1> &row_ptr,
                             sycl::buffer<std::int32_t, 1> &col_ind,
                             sycl::buffer<float, 1> &val);

[[deprecated("Use oneapi::mkl::sparse::set_csr_data(queue, handle, ...) instead.")]]
DLL_EXPORT void set_csr_data(matrix_handle_t handle,
                             const std::int64_t num_rows,
                             const std::int64_t num_cols,
                             index_base index,
                             sycl::buffer<std::int64_t, 1> &row_ptr,
                             sycl::buffer<std::int64_t, 1> &col_ind,
                             sycl::buffer<float, 1> &val);

[[deprecated("Use oneapi::mkl::sparse::set_csr_data(queue, handle, ...) instead.")]]
DLL_EXPORT void set_csr_data(matrix_handle_t handle,
                             const std::int32_t num_rows,
                             const std::int32_t num_cols,
                             index_base index,
                             sycl::buffer<std::int32_t, 1> &row_ptr,
                             sycl::buffer<std::int32_t, 1> &col_ind,
                             sycl::buffer<double, 1> &val);

[[deprecated("Use oneapi::mkl::sparse::set_csr_data(queue, handle, ...) instead.")]]
DLL_EXPORT void set_csr_data(matrix_handle_t handle,
                             const std::int64_t num_rows,
                             const std::int64_t num_cols,
                             index_base index,
                             sycl::buffer<std::int64_t, 1> &row_ptr,
                             sycl::buffer<std::int64_t, 1> &col_ind,
                             sycl::buffer<double, 1> &val);

[[deprecated("Use oneapi::mkl::sparse::set_csr_data(queue, handle, ...) instead.")]]
DLL_EXPORT void set_csr_data(matrix_handle_t handle,
                             const std::int32_t num_rows,
                             const std::int32_t num_cols,
                             index_base index,
                             sycl::buffer<std::int32_t, 1> &row_ptr,
                             sycl::buffer<std::int32_t, 1> &col_ind,
                             sycl::buffer<std::complex<float>, 1> &val);

[[deprecated("Use oneapi::mkl::sparse::set_csr_data(queue, handle, ...) instead.")]]
DLL_EXPORT void set_csr_data(matrix_handle_t handle,
                             const std::int64_t num_rows,
                             const std::int64_t num_cols,
                             index_base index,
                             sycl::buffer<std::int64_t, 1> &row_ptr,
                             sycl::buffer<std::int64_t, 1> &col_ind,
                             sycl::buffer<std::complex<float>, 1> &val);

[[deprecated("Use oneapi::mkl::sparse::set_csr_data(queue, handle, ...) instead.")]]
DLL_EXPORT void set_csr_data(matrix_handle_t handle,
                             const std::int32_t num_rows,
                             const std::int32_t num_cols,
                             index_base index,
                             sycl::buffer<std::int32_t, 1> &row_ptr,
                             sycl::buffer<std::int32_t, 1> &col_ind,
                             sycl::buffer<std::complex<double>, 1> &val);

[[deprecated("Use oneapi::mkl::sparse::set_csr_data(queue, handle, ...) instead.")]]
DLL_EXPORT void set_csr_data(matrix_handle_t handle,
                             const std::int64_t num_rows,
                             const std::int64_t num_cols,
                             index_base index,
                             sycl::buffer<std::int64_t, 1> &row_ptr,
                             sycl::buffer<std::int64_t, 1> &col_ind,
                             sycl::buffer<std::complex<double>, 1> &val);

[[deprecated("Use oneapi::mkl::sparse::set_csr_data(queue, handle, ...) instead.")]]
DLL_EXPORT void set_csr_data(matrix_handle_t handle,
                             const std::int32_t num_rows,
                             const std::int32_t num_cols,
                             index_base index,
                             std::int32_t *row_ptr,
                             std::int32_t *col_ind,
                             float *val);

[[deprecated("Use oneapi::mkl::sparse::set_csr_data(queue, handle, ..., dependencies) instead.")]]
DLL_EXPORT void set_csr_data(matrix_handle_t handle,
                             const std::int64_t num_rows,
                             const std::int64_t num_cols,
                             index_base index,
                             std::int64_t *row_ptr,
                             std::int64_t *col_ind,
                             float *val);

[[deprecated("Use oneapi::mkl::sparse::set_csr_data(queue, handle, ..., dependencies) instead.")]]
DLL_EXPORT void set_csr_data(matrix_handle_t handle,
                             const std::int32_t num_rows,
                             const std::int32_t num_cols,
                             index_base index,
                             std::int32_t *row_ptr,
                             std::int32_t *col_ind,
                             double *val);

[[deprecated("Use oneapi::mkl::sparse::set_csr_data(queue, handle, ..., dependencies) instead.")]]
DLL_EXPORT void set_csr_data(matrix_handle_t handle,
                             const std::int64_t num_rows,
                             const std::int64_t num_cols,
                             index_base index,
                             std::int64_t *row_ptr,
                             std::int64_t *col_ind,
                             double *val);

[[deprecated("Use oneapi::mkl::sparse::set_csr_data(queue, handle, ..., dependencies) instead.")]]
DLL_EXPORT void set_csr_data(matrix_handle_t handle,
                             const std::int32_t num_rows,
                             const std::int32_t num_cols,
                             index_base index,
                             std::int32_t *row_ptr,
                             std::int32_t *col_ind,
                             std::complex<float> *val);

[[deprecated("Use oneapi::mkl::sparse::set_csr_data(queue, handle, ..., dependencies) instead.")]]
DLL_EXPORT void set_csr_data(matrix_handle_t handle,
                             const std::int64_t num_rows,
                             const std::int64_t num_cols,
                             index_base index,
                             std::int64_t *row_ptr,
                             std::int64_t *col_ind,
                             std::complex<float> *val);

[[deprecated("Use oneapi::mkl::sparse::set_csr_data(queue, handle, ..., dependencies) instead.")]]
DLL_EXPORT void set_csr_data(matrix_handle_t handle,
                             const std::int32_t num_rows,
                             const std::int32_t num_cols,
                             index_base index,
                             std::int32_t *row_ptr,
                             std::int32_t *col_ind,
                             std::complex<double> *val);

[[deprecated("Use oneapi::mkl::sparse::set_csr_data(queue, handle, ..., dependencies) instead.")]]
DLL_EXPORT void set_csr_data(matrix_handle_t handle,
                             const std::int64_t num_rows,
                             const std::int64_t num_cols,
                             index_base index,
                             std::int64_t *row_ptr,
                             std::int64_t *col_ind,
                             std::complex<double> *val);

DLL_EXPORT sycl::event omatcopy(sycl::queue &queue,
                                transpose transpose_flag,
                                matrix_handle_t from_handle,
                                matrix_handle_t to_handle,
                                const std::vector<sycl::event> &dependencies = {});

[[deprecated("Use oneapi::mkl::sparse::omatcopy(queue, transpose_flag, handle, "
             "trans_handle, dependencies) instead.")]] DLL_EXPORT void
make_transpose(sycl::queue &queue,
               transpose transpose_flag,
               matrix_handle_t handle,
               matrix_handle_t trans_handle);

[[deprecated("This undocumented API was meant for internal optimizations and is not necessary for "
             "common sparse workloads. Please reach out on https://community.intel.com/ to provide "
             "feedback if you need this API.")]] DLL_EXPORT void
make_symmetric(sycl::queue &queue,
               uplo uplo_flag,
               matrix_handle_t handle,
               matrix_handle_t sym_handle);

DLL_EXPORT sycl::event
optimize_gemv(sycl::queue &queue,
              transpose transpose_flag,
              matrix_handle_t handle,
              const std::vector<sycl::event> &dependencies = {});

DLL_EXPORT sycl::event
optimize_trmv(sycl::queue &queue,
              uplo uplo_flag,
              transpose transpose_flag,
              diag diag_val,
              matrix_handle_t handle,
              const std::vector<sycl::event> &dependencies = {});

DLL_EXPORT sycl::event
optimize_trsv(sycl::queue &queue,
              uplo uplo_flag,
              transpose transpose_flag,
              diag diag_val,
              matrix_handle_t handle,
              const std::vector<sycl::event> &dependencies = {});

DLL_EXPORT void gemv(sycl::queue &queue,
                     transpose transpose_flag,
                     const float alpha,
                     matrix_handle_t handle,
                     sycl::buffer<float, 1> &x,
                     const float beta,
                     sycl::buffer<float, 1> &y);

DLL_EXPORT void gemv(sycl::queue &queue,
                     transpose transpose_flag,
                     const double alpha,
                     matrix_handle_t handle,
                     sycl::buffer<double, 1> &x,
                     const double beta,
                     sycl::buffer<double, 1> &y);

DLL_EXPORT void gemv(sycl::queue &queue,
                     transpose transpose_flag,
                     const std::complex<float> alpha,
                     matrix_handle_t handle,
                     sycl::buffer<std::complex<float>, 1> &x,
                     const std::complex<float> beta,
                     sycl::buffer<std::complex<float>, 1> &y);

DLL_EXPORT void gemv(sycl::queue &queue,
                     transpose transpose_flag,
                     const std::complex<double> alpha,
                     matrix_handle_t handle,
                     sycl::buffer<std::complex<double>, 1> &x,
                     const std::complex<double> beta,
                     sycl::buffer<std::complex<double>, 1> &y);

DLL_EXPORT sycl::event gemv(sycl::queue &queue,
                                transpose transpose_flag,
                                const float alpha,
                                matrix_handle_t handle,
                                const float *x,
                                const float beta,
                                float *y,
                                const std::vector<sycl::event> &dependencies = {});

DLL_EXPORT sycl::event gemv(sycl::queue &queue,
                                transpose transpose_flag,
                                const double alpha,
                                matrix_handle_t handle,
                                const double *x,
                                const double beta,
                                double *y,
                                const std::vector<sycl::event> &dependencies = {});

DLL_EXPORT sycl::event gemv(sycl::queue &queue,
                                transpose transpose_flag,
                                const std::complex<float> alpha,
                                matrix_handle_t handle,
                                const std::complex<float> *x,
                                const std::complex<float> beta,
                                std::complex<float> *y,
                                const std::vector<sycl::event> &dependencies = {});

DLL_EXPORT sycl::event gemv(sycl::queue &queue,
                                transpose transpose_flag,
                                const std::complex<double> alpha,
                                matrix_handle_t handle,
                                const std::complex<double> *x,
                                const std::complex<double> beta,
                                std::complex<double> *y,
                                const std::vector<sycl::event> &dependencies = {});

DLL_EXPORT void gemvdot(sycl::queue &queue,
                        transpose transpose_flag,
                        const float alpha,
                        matrix_handle_t handle,
                        sycl::buffer<float, 1> &x,
                        const float beta,
                        sycl::buffer<float, 1> &y,
                        sycl::buffer<float, 1> &d);

DLL_EXPORT void gemvdot(sycl::queue &queue,
                        transpose transpose_flag,
                        const double alpha,
                        matrix_handle_t handle,
                        sycl::buffer<double, 1> &x,
                        const double beta,
                        sycl::buffer<double, 1> &y,
                        sycl::buffer<double, 1> &d);

DLL_EXPORT void gemvdot(sycl::queue &queue,
                        transpose transpose_flag,
                        const std::complex<float> alpha,
                        matrix_handle_t handle,
                        sycl::buffer<std::complex<float>, 1> &x,
                        const std::complex<float> beta,
                        sycl::buffer<std::complex<float>, 1> &y,
                        sycl::buffer<std::complex<float>, 1> &d);

DLL_EXPORT void gemvdot(sycl::queue &queue,
                        transpose transpose_flag,
                        const std::complex<double> alpha,
                        matrix_handle_t handle,
                        sycl::buffer<std::complex<double>, 1> &x,
                        const std::complex<double> beta,
                        sycl::buffer<std::complex<double>, 1> &y,
                        sycl::buffer<std::complex<double>, 1> &d);

DLL_EXPORT sycl::event
gemvdot(sycl::queue &queue,
        transpose transpose_flag,
        const float alpha,
        matrix_handle_t handle,
        float *x,
        const float beta,
        float *y,
        float *d,
        const std::vector<sycl::event> &dependencies = {});

DLL_EXPORT sycl::event
gemvdot(sycl::queue &queue,
        transpose transpose_flag,
        const double alpha,
        matrix_handle_t handle,
        double *x,
        const double beta,
        double *y,
        double *d,
        const std::vector<sycl::event> &dependencies = {});

DLL_EXPORT sycl::event
gemvdot(sycl::queue &queue,
        transpose transpose_flag,
        const std::complex<float> alpha,
        matrix_handle_t handle,
        std::complex<float> *x,
        const std::complex<float> beta,
        std::complex<float> *y,
        std::complex<float> *d,
        const std::vector<sycl::event> &dependencies = {});

DLL_EXPORT sycl::event
gemvdot(sycl::queue &queue,
        transpose transpose_flag,
        const std::complex<double> alpha,
        matrix_handle_t handle,
        std::complex<double> *x,
        const std::complex<double> beta,
        std::complex<double> *y,
        std::complex<double> *d,
        const std::vector<sycl::event> &dependencies = {});

DLL_EXPORT void symv(sycl::queue &queue,
                     uplo uplo_flag,
                     const float alpha,
                     matrix_handle_t handle,
                     sycl::buffer<float, 1> &x,
                     const float beta,
                     sycl::buffer<float, 1> &y);

DLL_EXPORT void symv(sycl::queue &queue,
                     uplo uplo_flag,
                     const double alpha,
                     matrix_handle_t handle,
                     sycl::buffer<double, 1> &x,
                     const double beta,
                     sycl::buffer<double, 1> &y);

DLL_EXPORT void symv(sycl::queue &queue,
                     uplo uplo_flag,
                     const std::complex<float> alpha,
                     matrix_handle_t handle,
                     sycl::buffer<std::complex<float>, 1> &x,
                     const std::complex<float> beta,
                     sycl::buffer<std::complex<float>, 1> &y);

DLL_EXPORT void symv(sycl::queue &queue,
                     uplo uplo_flag,
                     const std::complex<double> alpha,
                     matrix_handle_t handle,
                     sycl::buffer<std::complex<double>, 1> &x,
                     const std::complex<double> beta,
                     sycl::buffer<std::complex<double>, 1> &y);

DLL_EXPORT sycl::event symv(sycl::queue &queue,
                                uplo uplo_flag,
                                const float alpha,
                                matrix_handle_t handle,
                                float *x,
                                const float beta,
                                float *y,
                                const std::vector<sycl::event> &dependencies = {});

DLL_EXPORT sycl::event symv(sycl::queue &queue,
                                uplo uplo_flag,
                                const double alpha,
                                matrix_handle_t handle,
                                double *x,
                                const double beta,
                                double *y,
                                const std::vector<sycl::event> &dependencies = {});

DLL_EXPORT sycl::event symv(sycl::queue &queue,
                                uplo uplo_flag,
                                const std::complex<float> alpha,
                                matrix_handle_t handle,
                                std::complex<float> *x,
                                const std::complex<float> beta,
                                std::complex<float> *y,
                                const std::vector<sycl::event> &dependencies = {});

DLL_EXPORT sycl::event symv(sycl::queue &queue,
                                uplo uplo_flag,
                                const std::complex<double> alpha,
                                matrix_handle_t handle,
                                std::complex<double> *x,
                                const std::complex<double> beta,
                                std::complex<double> *y,
                                const std::vector<sycl::event> &dependencies = {});

DLL_EXPORT void trmv(sycl::queue &queue,
                     uplo uplo_flag,
                     transpose transpose_flag,
                     diag diag_val,
                     const float alpha,
                     matrix_handle_t handle,
                     sycl::buffer<float, 1> &x,
                     const float beta,
                     sycl::buffer<float, 1> &y);

DLL_EXPORT void trmv(sycl::queue &queue,
                     uplo uplo_flag,
                     transpose transpose_flag,
                     diag diag_val,
                     const double alpha,
                     matrix_handle_t handle,
                     sycl::buffer<double, 1> &x,
                     const double beta,
                     sycl::buffer<double, 1> &y);

DLL_EXPORT void trmv(sycl::queue &queue,
                     uplo uplo_flag,
                     transpose transpose_flag,
                     diag diag_val,
                     const std::complex<float> alpha,
                     matrix_handle_t handle,
                     sycl::buffer<std::complex<float>, 1> &x,
                     const std::complex<float> beta,
                     sycl::buffer<std::complex<float>, 1> &y);

DLL_EXPORT void trmv(sycl::queue &queue,
                     uplo uplo_flag,
                     transpose transpose_flag,
                     diag diag_val,
                     const std::complex<double> alpha,
                     matrix_handle_t handle,
                     sycl::buffer<std::complex<double>, 1> &x,
                     const std::complex<double> beta,
                     sycl::buffer<std::complex<double>, 1> &y);

DLL_EXPORT sycl::event trmv(sycl::queue &queue,
                                uplo uplo_flag,
                                transpose transpose_flag,
                                diag diag_flag,
                                const float alpha,
                                matrix_handle_t handle,
                                float *x,
                                const float beta,
                                float *y,
                                const std::vector<sycl::event> &dependencies = {});

DLL_EXPORT sycl::event trmv(sycl::queue &queue,
                                uplo uplo_flag,
                                transpose transpose_flag,
                                diag diag_flag,
                                const double alpha,
                                matrix_handle_t handle,
                                double *x,
                                const double beta,
                                double *y,
                                const std::vector<sycl::event> &dependencies = {});

DLL_EXPORT sycl::event trmv(sycl::queue &queue,
                                uplo uplo_flag,
                                transpose transpose_flag,
                                diag diag_flag,
                                const std::complex<float> alpha,
                                matrix_handle_t handle,
                                std::complex<float> *x,
                                const std::complex<float> beta,
                                std::complex<float> *y,
                                const std::vector<sycl::event> &dependencies = {});

DLL_EXPORT sycl::event trmv(sycl::queue &queue,
                                uplo uplo_flag,
                                transpose transpose_flag,
                                diag diag_flag,
                                const std::complex<double> alpha,
                                matrix_handle_t handle,
                                std::complex<double> *x,
                                const std::complex<double> beta,
                                std::complex<double> *y,
                                const std::vector<sycl::event> &dependencies = {});

DLL_EXPORT void trsv(sycl::queue &queue,
                     uplo uplo_flag,
                     transpose transpose_flag,
                     diag diag_val,
                     matrix_handle_t handle,
                     sycl::buffer<float, 1> &x,
                     sycl::buffer<float, 1> &y);

DLL_EXPORT void trsv(sycl::queue &queue,
                     uplo uplo_flag,
                     transpose transpose_flag,
                     diag diag_val,
                     matrix_handle_t handle,
                     sycl::buffer<double, 1> &x,
                     sycl::buffer<double, 1> &y);

DLL_EXPORT void trsv(sycl::queue &queue,
                     uplo uplo_flag,
                     transpose transpose_flag,
                     diag diag_val,
                     matrix_handle_t handle,
                     sycl::buffer<std::complex<float>, 1> &x,
                     sycl::buffer<std::complex<float>, 1> &y);

DLL_EXPORT void trsv(sycl::queue &queue,
                     uplo uplo_flag,
                     transpose transpose_flag,
                     diag diag_val,
                     matrix_handle_t handle,
                     sycl::buffer<std::complex<double>, 1> &x,
                     sycl::buffer<std::complex<double>, 1> &y);

DLL_EXPORT sycl::event trsv(sycl::queue &queue,
                                uplo uplo_flag,
                                transpose transpose_flag,
                                diag diag_flag,
                                matrix_handle_t handle,
                                float *x,
                                float *y,
                                const std::vector<sycl::event> &dependencies = {});

DLL_EXPORT sycl::event trsv(sycl::queue &queue,
                                uplo uplo_flag,
                                transpose transpose_flag,
                                diag diag_flag,
                                matrix_handle_t handle,
                                double *x,
                                double *y,
                                const std::vector<sycl::event> &dependencies = {});

DLL_EXPORT sycl::event trsv(sycl::queue &queue,
                                uplo uplo_flag,
                                transpose transpose_flag,
                                diag diag_flag,
                                matrix_handle_t handle,
                                std::complex<float> *x,
                                std::complex<float> *y,
                                const std::vector<sycl::event> &dependencies = {});

DLL_EXPORT sycl::event trsv(sycl::queue &queue,
                                uplo uplo_flag,
                                transpose transpose_flag,
                                diag diag_flag,
                                matrix_handle_t handle,
                                std::complex<double> *x,
                                std::complex<double> *y,
                                const std::vector<sycl::event> &dependencies = {});

[[deprecated("Use oneapi::mkl::sparse::gemm(queue, oneapi::mkl::layout::row_major, transpose_A, "
             "oneapi::mkl::transpose:nontrans, alpha, ... ) instead.")]] DLL_EXPORT void
gemm(sycl::queue &queue,
     transpose transpose_flag,
     const float alpha,
     matrix_handle_t handle,
     sycl::buffer<float, 1> &b,
     const std::int64_t columns,
     const std::int64_t ldb,
     const float beta,
     sycl::buffer<float, 1> &c,
     const std::int64_t ldc);

[[deprecated("Use oneapi::mkl::sparse::gemm(queue, oneapi::mkl::layout::row_major, transpose_A, "
             "oneapi::mkl::transpose:nontrans, alpha, ... ) instead.")]] DLL_EXPORT void
gemm(sycl::queue &queue,
     transpose transpose_flag,
     const double alpha,
     matrix_handle_t handle,
     sycl::buffer<double, 1> &b,
     const std::int64_t columns,
     const std::int64_t ldb,
     const double beta,
     sycl::buffer<double, 1> &c,
     const std::int64_t ldc);

[[deprecated("Use oneapi::mkl::sparse::gemm(queue, oneapi::mkl::layout::row_major, transpose_A, "
             "oneapi::mkl::transpose:nontrans, alpha, ... ) instead.")]] DLL_EXPORT void
gemm(sycl::queue &queue,
     transpose transpose_flag,
     const std::complex<float> alpha,
     matrix_handle_t handle,
     sycl::buffer<std::complex<float>, 1> &b,
     const std::int64_t columns,
     const std::int64_t ldb,
     const std::complex<float> beta,
     sycl::buffer<std::complex<float>, 1> &c,
     const std::int64_t ldc);

[[deprecated("Use oneapi::mkl::sparse::gemm(queue, oneapi::mkl::layout::row_major, transpose_A, "
             "oneapi::mkl::transpose:nontrans, alpha, ... ) instead.")]] DLL_EXPORT void
gemm(sycl::queue &queue,
     transpose transpose_flag,
     const std::complex<double> alpha,
     matrix_handle_t handle,
     sycl::buffer<std::complex<double>, 1> &b,
     const std::int64_t columns,
     const std::int64_t ldb,
     const std::complex<double> beta,
     sycl::buffer<std::complex<double>, 1> &c,
     const std::int64_t ldc);

[[deprecated("Use oneapi::mkl::sparse::gemm(queue, oneapi::mkl::layout::row_major, transpose_A, "
             "oneapi::mkl::transpose:nontrans, alpha, ... ) instead.")]] DLL_EXPORT sycl::event
gemm(sycl::queue &queue,
     transpose transpose_flag,
     const float alpha,
     matrix_handle_t handle,
     float *b,
     const std::int64_t columns,
     const std::int64_t ldb,
     const float beta,
     float *c,
     const std::int64_t ldc,
     const std::vector<sycl::event> &dependencies = {});

[[deprecated("Use oneapi::mkl::sparse::gemm(queue, oneapi::mkl::layout::row_major, transpose_A, "
             "oneapi::mkl::transpose:nontrans, alpha, ... ) instead.")]] DLL_EXPORT sycl::event
gemm(sycl::queue &queue,
     transpose transpose_flag,
     const double alpha,
     matrix_handle_t handle,
     double *b,
     const std::int64_t columns,
     const std::int64_t ldb,
     const double beta,
     double *c,
     const std::int64_t ldc,
     const std::vector<sycl::event> &dependencies = {});

[[deprecated("Use oneapi::mkl::sparse::gemm(queue, oneapi::mkl::layout::row_major, transpose_A, "
             "oneapi::mkl::transpose:nontrans, alpha, ... ) instead.")]] DLL_EXPORT sycl::event
gemm(sycl::queue &queue,
     transpose transpose_flag,
     const std::complex<float> alpha,
     matrix_handle_t handle,
     std::complex<float> *b,
     const std::int64_t columns,
     const std::int64_t ldb,
     const std::complex<float> beta,
     std::complex<float> *c,
     const std::int64_t ldc,
     const std::vector<sycl::event> &dependencies = {});

[[deprecated("Use oneapi::mkl::sparse::gemm(queue, oneapi::mkl::layout::row_major, transpose_A, "
             "oneapi::mkl::transpose:nontrans, alpha, ... ) instead.")]] DLL_EXPORT sycl::event
gemm(sycl::queue &queue,
     transpose transpose_flag,
     const std::complex<double> alpha,
     matrix_handle_t handle,
     std::complex<double> *b,
     const std::int64_t columns,
     const std::int64_t ldb,
     const std::complex<double> beta,
     std::complex<double> *c,
     const std::int64_t ldc,
     const std::vector<sycl::event> &dependencies = {});

DLL_EXPORT void gemm(sycl::queue &queue,
                     layout dense_matrix_layout,
                     transpose opA,
                     transpose opB,
                     const float alpha,
                     matrix_handle_t handle,
                     sycl::buffer<float, 1> &b,
                     const std::int64_t columns,
                     const std::int64_t ldb,
                     const float beta,
                     sycl::buffer<float, 1> &c,
                     const std::int64_t ldc);

DLL_EXPORT void gemm(sycl::queue &queue,
                     layout dense_matrix_layout,
                     transpose opA,
                     transpose opB,
                     const double alpha,
                     matrix_handle_t handle,
                     sycl::buffer<double, 1> &b,
                     const std::int64_t columns,
                     const std::int64_t ldb,
                     const double beta,
                     sycl::buffer<double, 1> &c,
                     const std::int64_t ldc);

DLL_EXPORT void gemm(sycl::queue &queue,
                     layout dense_matrix_layout,
                     transpose opA,
                     transpose opB,
                     const std::complex<float> alpha,
                     matrix_handle_t handle,
                     sycl::buffer<std::complex<float>, 1> &b,
                     const std::int64_t columns,
                     const std::int64_t ldb,
                     const std::complex<float> beta,
                     sycl::buffer<std::complex<float>, 1> &c,
                     const std::int64_t ldc);

DLL_EXPORT void gemm(sycl::queue &queue,
                     layout dense_matrix_layout,
                     transpose opA,
                     transpose opB,
                     const std::complex<double> alpha,
                     matrix_handle_t handle,
                     sycl::buffer<std::complex<double>, 1> &b,
                     const std::int64_t columns,
                     const std::int64_t ldb,
                     const std::complex<double> beta,
                     sycl::buffer<std::complex<double>, 1> &c,
                     const std::int64_t ldc);

DLL_EXPORT sycl::event gemm(sycl::queue &queue,
                                layout dense_matrix_layout,
                                transpose opA,
                                transpose opB,
                                const float alpha,
                                matrix_handle_t handle,
                                float *b,
                                const std::int64_t columns,
                                const std::int64_t ldb,
                                const float beta,
                                float *c,
                                const std::int64_t ldc,
                                const std::vector<sycl::event> &dependencies = {});

DLL_EXPORT sycl::event gemm(sycl::queue &queue,
                                layout dense_matrix_layout,
                                transpose opA,
                                transpose opB,
                                const double alpha,
                                matrix_handle_t handle,
                                double *b,
                                const std::int64_t columns,
                                const std::int64_t ldb,
                                const double beta,
                                double *c,
                                const std::int64_t ldc,
                                const std::vector<sycl::event> &dependencies = {});

DLL_EXPORT sycl::event gemm(sycl::queue &queue,
                                layout dense_matrix_layout,
                                transpose opA,
                                transpose opB,
                                const std::complex<float> alpha,
                                matrix_handle_t handle,
                                std::complex<float> *b,
                                const std::int64_t columns,
                                const std::int64_t ldb,
                                const std::complex<float> beta,
                                std::complex<float> *c,
                                const std::int64_t ldc,
                                const std::vector<sycl::event> &dependencies = {});

DLL_EXPORT sycl::event gemm(sycl::queue &queue,
                                layout dense_matrix_layout,
                                transpose opA,
                                transpose opB,
                                const std::complex<double> alpha,
                                matrix_handle_t handle,
                                std::complex<double> *b,
                                const std::int64_t columns,
                                const std::int64_t ldb,
                                const std::complex<double> beta,
                                std::complex<double> *c,
                                const std::int64_t ldc,
                                const std::vector<sycl::event> &dependencies = {});

//
// Types, Objects and APIs for Sparse Matrix * Sparse Matrix (matmat) multiplication
//
enum class matrix_view_descr : std::int32_t { general = 1 };

enum class matmat_request : std::int32_t {
    get_work_estimation_buf_size = 1,
    work_estimation = 2,

    get_compute_structure_buf_size = 3,
    compute_structure = 4,
    finalize_structure = 5,

    get_compute_buf_size = 6,
    compute = 7,
    get_nnz = 8,
    finalize = 9,
};

struct matmat_descr;
typedef matmat_descr *matmat_descr_t;

DLL_EXPORT void init_matmat_descr(matmat_descr_t *desc);
DLL_EXPORT void release_matmat_descr(matmat_descr_t *desc);

DLL_EXPORT void set_matmat_data(matmat_descr_t descr,
                                matrix_view_descr viewA,
                                transpose opA,
                                matrix_view_descr viewB,
                                transpose opB,
                                matrix_view_descr viewC);

DLL_EXPORT void get_matmat_data(matmat_descr_t descr,
                                matrix_view_descr &viewA,
                                transpose &opA,
                                matrix_view_descr &viewB,
                                transpose &opB,
                                matrix_view_descr &viewC);

DLL_EXPORT sycl::event matmat(sycl::queue &queue,
                              matrix_handle_t A,
                              matrix_handle_t B,
                              matrix_handle_t C,
                              sparse::matmat_request req,
                              sparse::matmat_descr_t descr,
                              std::int64_t *sizeTempBuffer,
                              void *tempBuffer,
                              const std::vector<sycl::event> &dependencies);

DLL_EXPORT void matmat(sycl::queue &queue,
                       matrix_handle_t A,
                       matrix_handle_t B,
                       matrix_handle_t C,
                       sparse::matmat_request req,
                       sparse::matmat_descr_t descr,
                       sycl::buffer<std::int64_t, 1> *sizeTempBuffer,
                       sycl::buffer<std::uint8_t, 1> *tempBuffer);

DLL_EXPORT sycl::event sort_matrix(sycl::queue &queue,
                                   matrix_handle_t hMatrix,
                                   const std::vector<cl::sycl::event> &dependencies = {});

} /* namespace oneapi::mkl::sparse */
} /* namespace mkl */
} // namespace oneapi

#endif /* _SPBLAS_HPP_ */
