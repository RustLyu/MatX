/*******************************************************************************
* Copyright (C) 2024 Intel Corporation
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

#ifndef _DFT_HPP_
#define _DFT_HPP_

#include <cinttypes>    // std::int64_t
#include <vector>       // std::vector
#include <sycl/sycl.hpp>  // sycl::
#include "oneapi/mkl/dft/spec.hpp"
#include "mkl_dfti.h"   // DFTI_DESCRIPTOR_HANDLE

typedef struct SYCL_DFTI_DESCRIPTOR* SYCL_DFTI_DESCRIPTOR_HANDLE;

namespace oneapi {
namespace mkl {
namespace dft {

enum class precision {
    SINGLE = DFTI_SINGLE,
    DOUBLE = DFTI_DOUBLE
};

enum class domain {
    REAL = DFTI_REAL,
    COMPLEX = DFTI_COMPLEX
};

enum class config_param {
    FORWARD_DOMAIN                          = DFTI_FORWARD_DOMAIN,
    DIMENSION                               = DFTI_DIMENSION,
    LENGTHS                                 = DFTI_LENGTHS,
    PRECISION                               = DFTI_PRECISION,
    FORWARD_SCALE                           = DFTI_FORWARD_SCALE,
    BACKWARD_SCALE                          = DFTI_BACKWARD_SCALE,
    NUMBER_OF_TRANSFORMS                    = DFTI_NUMBER_OF_TRANSFORMS,
    COMPLEX_STORAGE                         = DFTI_COMPLEX_STORAGE,
    CONJUGATE_EVEN_STORAGE  [[deprecated]]  = DFTI_CONJUGATE_EVEN_STORAGE,
    PLACEMENT                               = DFTI_PLACEMENT,
    INPUT_STRIDES           [[deprecated("Use FWD/BWD strides instead.")]]
                                            = DFTI_INPUT_STRIDES,
    OUTPUT_STRIDES          [[deprecated("Use FWD/BWD strides instead.")]]
                                            = DFTI_OUTPUT_STRIDES,
    FWD_DISTANCE                            = DFTI_FWD_DISTANCE,
    BWD_DISTANCE                            = DFTI_BWD_DISTANCE,
    WORKSPACE                               = DFTI_WORKSPACE,
    PACKED_FORMAT           [[deprecated]]  = DFTI_PACKED_FORMAT,
    COMMIT_STATUS                           = DFTI_COMMIT_STATUS,
    VERSION                 [[deprecated]]  = DFTI_VERSION,
    THREAD_LIMIT                            = DFTI_THREAD_LIMIT,
    DESTROY_INPUT                           = DFTI_DESTROY_INPUT,
    WORKSPACE_ESTIMATE_BYTES,
    WORKSPACE_BYTES,
    FWD_STRIDES,
    BWD_STRIDES,
    WORKSPACE_PLACEMENT,        // alias for WORKSPACE
    WORKSPACE_EXTERNAL_BYTES    // alias for WORKSPACE_BYTES
};

enum class config_value {
    COMMITTED                               = DFTI_COMMITTED,
    UNCOMMITTED                             = DFTI_UNCOMMITTED,
    COMPLEX_COMPLEX                         = DFTI_COMPLEX_COMPLEX,
    REAL_REAL                               = DFTI_REAL_REAL,
    INPLACE                                 = DFTI_INPLACE,
    NOT_INPLACE                             = DFTI_NOT_INPLACE,
    WORKSPACE_AUTOMATIC,            // alias for WORKSPACE_INTERNAL
    COMPLEX_REAL            [[deprecated]]  = DFTI_COMPLEX_REAL,
    ALLOW                                   = DFTI_ALLOW,
    AVOID                                   = DFTI_AVOID,
    CCE_FORMAT              [[deprecated]]  = DFTI_CCE_FORMAT,
    PERM_FORMAT             [[deprecated]]  = DFTI_PERM_FORMAT,
    PACK_FORMAT             [[deprecated]]  = DFTI_PACK_FORMAT,
    CCS_FORMAT              [[deprecated]]  = DFTI_CCS_FORMAT,
    WORKSPACE_INTERNAL,
    WORKSPACE_EXTERNAL
};

// Compute functions which will be friends with the descriptor class
template<typename descriptor_type, typename data_type>
void compute_forward(
    descriptor_type &desc,
    sycl::buffer<data_type, 1> &inout);

template<typename descriptor_type, typename input_type, typename output_type>
void compute_forward(
    descriptor_type &desc,
    sycl::buffer<input_type, 1> &in,
    sycl::buffer<output_type, 1> &out);

template<typename descriptor_type, typename data_type>
void compute_backward(
    descriptor_type &desc,
    sycl::buffer<data_type, 1> &inout);

template<typename descriptor_type, typename input_type, typename output_type>
void compute_backward(
    descriptor_type &desc,
    sycl::buffer<input_type, 1> &in,
    sycl::buffer<output_type, 1> &out);

template <typename descriptor_type, typename data_type>
sycl::event compute_forward(
    descriptor_type &desc,
    data_type *inout,
    const std::vector<sycl::event> &dependencies = {});

template<typename descriptor_type, typename input_type, typename output_type>
sycl::event compute_forward(
    descriptor_type &desc,
    input_type *in,
    output_type *out,
    const std::vector<sycl::event> &dependencies = {});

template <typename descriptor_type, typename data_type>
sycl::event compute_backward(
    descriptor_type &desc,
    data_type *inout,
    const std::vector<sycl::event> &dependencies = {});

template<typename descriptor_type, typename input_type, typename output_type>
sycl::event compute_backward(
    descriptor_type &desc,
    input_type *in,
    output_type *out,
    const std::vector<sycl::event> &dependencies = {});

template <precision prec, domain dom>
class descriptor {
    using real_scalar_t = std::conditional_t<prec == precision::DOUBLE, double, float>;
 public:
    // initializes the DFT descriptor for a multi-dimensional DFT
    descriptor(std::vector<std::int64_t> dimensions);
    // initializes the DFT descriptor for a one-dimensional DFT
    descriptor(std::int64_t length);
    ~descriptor();
    descriptor(const descriptor&) = delete;
    descriptor& operator=(const descriptor&) = delete;
    descriptor(descriptor&&);
    descriptor& operator=(descriptor&&);

    void commit(sycl::queue &in);

    // in place forward computation; buffer API
    template <typename descriptor_type, typename data_type>
    friend void compute_forward(
        descriptor_type &desc,
        sycl::buffer<data_type, 1> &inout);
    // out of place forward computation; buffer API
    template<typename descriptor_type, typename input_type, typename output_type>
    friend void compute_forward(
        descriptor_type &desc,
        sycl::buffer<input_type, 1> &in,
        sycl::buffer<output_type, 1> &out);
    // in place backward computation; buffer API
    template <typename descriptor_type, typename data_type>
    friend void compute_backward(
        descriptor_type &desc,
        sycl::buffer<data_type, 1> &inout);
    // out of place backward computation; buffer API
    template<typename descriptor_type, typename input_type, typename output_type>
    friend void compute_backward(
        descriptor_type &desc,
        sycl::buffer<input_type, 1> &in,
        sycl::buffer<output_type, 1> &out);
    // in place forward computation; USM API
    template <typename descriptor_type, typename data_type>
    friend sycl::event compute_forward(
        descriptor_type &desc,
        data_type *inout,
        const std::vector<sycl::event> &dependencies);
    // out of place forward computation; USM API
    template<typename descriptor_type, typename input_type, typename output_type>
    friend sycl::event compute_forward(
        descriptor_type &desc,
        input_type *in,
        output_type *out,
        const std::vector<sycl::event> &dependencies);
    // in place backward computation; USM API
    template <typename descriptor_type, typename data_type>
    friend sycl::event compute_backward(
        descriptor_type &desc,
        data_type *inout,
        const std::vector<sycl::event> &dependencies);
    // out of place backward computation; USM API
    template<typename descriptor_type, typename input_type, typename output_type>
    friend sycl::event compute_backward(
        descriptor_type &desc,
        input_type *in,
        output_type *out,
        const std::vector<sycl::event> &dependencies);

    // configuration-setting member functions:
    [[deprecated("Use set_value(config_param, config_value) instead.")]]
    void set_value(config_param, DFTI_CONFIG_VALUE);
    void set_value(config_param, config_value);
    void set_value(config_param, std::int64_t);
    void set_value(config_param, real_scalar_t);
    [[deprecated("Use set_value(config_param, const std::vector<std::int64_t>&) instead.")]]
    void set_value(config_param, const std::int64_t*);
    void set_value(config_param, const std::vector<std::int64_t>&);
    template <typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
    void set_value(config_param param, T value) {
        set_value(param, static_cast<std::int64_t>(value));
    }
    template <typename T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
    void set_value(config_param param, T value) {
        set_value(param, static_cast<real_scalar_t>(value));
    }
    [[deprecated("This set_value member function is deprecated.")]]
    void set_value(config_param, ...);
    // configuration-querying member functions:
    [[deprecated("Use MKL_Get_Version_String(char*, int) instead.")]]
    void get_value(config_param, char*) const;
    [[deprecated("Use get_value(config_param, config_value*), "
                 "get_value(config_param, domain*) or "
                 "get_value(config_param, precision*) instead.")]]
    void get_value(config_param, DFTI_CONFIG_VALUE*) const;
    void get_value(config_param, config_value*) const;
    void get_value(config_param, domain*) const;
    void get_value(config_param, precision*) const;
    [[deprecated("Use get_value(config_param, std::int64_t*) instead.")]]
    void get_value(config_param, size_t*) const;
    void get_value(config_param, std::int64_t*) const;
    void get_value(config_param, real_scalar_t*) const;
    void get_value(config_param, std::vector<std::int64_t>*) const;
    [[deprecated("This get_value member function is deprecated.")]]
    void get_value(config_param, ...) const;

    template<typename data_type>
    void set_workspace(sycl::buffer<data_type, 1> &workspace);
    template<typename data_type>
    void set_workspace(data_type *workspace);

 private:
    DFTI_DESCRIPTOR_HANDLE handle;
    SYCL_DFTI_DESCRIPTOR_HANDLE device_handle;
    sycl::buffer<SYCL_DFTI_DESCRIPTOR_HANDLE, 1> handle_buffer;
};

}  // namespace dft
}  // namespace mkl
} // namespace oneapi

#endif  /* _DFT_HPP_ */
