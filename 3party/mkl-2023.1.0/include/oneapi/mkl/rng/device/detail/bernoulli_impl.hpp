/*******************************************************************************
* Copyright 2021-2022 Intel Corporation.
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

#ifndef _MKL_RNG_DEVICE_BERNOULLI_IMPL_HPP_
#define _MKL_RNG_DEVICE_BERNOULLI_IMPL_HPP_

namespace oneapi {
namespace mkl {
namespace rng {
namespace device {
namespace detail {

template <typename IntType, typename Method>
class distribution_base<oneapi::mkl::rng::device::bernoulli<IntType, Method>> {
public:
    struct param_type {
        param_type(float p) : p_(p) {}
        float p_;
    };

    distribution_base(float p) : p_(p) {
#ifndef __SYCL_DEVICE_ONLY__
        if ((p > 1.0f) || (p < 0.0f)) {
            throw oneapi::mkl::invalid_argument("rng", "bernoulli", "p < 0 || p > 1");
        }
#endif
    }

    float p() const {
        return p_;
    }

    param_type param() const {
        return param_type(p_);
    }

    void param(const param_type& pt) {
#ifndef __SYCL_DEVICE_ONLY__
        if ((pt.p_ > 1.0f) || (pt.p_ < 0.0f)) {
            throw oneapi::mkl::invalid_argument("rng", "bernoulli", "p < 0 || p > 1");
        }
#endif
        p_ = pt.p_;
    }

protected:
    template <typename EngineType>
    auto generate(EngineType& engine) ->
        typename std::conditional<EngineType::vec_size == 1, IntType,
                                  sycl::vec<IntType, EngineType::vec_size>>::type {
        // Usage of bit arithmetic doesn't improve the performance
        auto uni_res = uniform_.generate(engine);
        if constexpr (EngineType::vec_size == 1) {
            return IntType{uni_res < p_};
        }
        else {
            sycl::vec<IntType, EngineType::vec_size> vec_out(IntType{ 0 });
            for (int i = 0; i < EngineType::vec_size; ++i) {
                if (uni_res[i] < p_) {
                    vec_out[i] = IntType{ 1 };
                }
            }
            return vec_out;
        }
    }

    template <typename EngineType>
    IntType generate_single(EngineType& engine) {
        auto uni_res = uniform_.generate_single(engine);
        return IntType{uni_res < p_};
    }

    distribution_base<oneapi::mkl::rng::device::uniform<float>> uniform_ = { 0.0f, 1.0f };
    float p_;
};

} // namespace detail
} // namespace device
} // namespace rng
} // namespace mkl
} // namespace oneapi

#endif // _MKL_RNG_DEVICE_BERNOULLI_IMPL_HPP_
