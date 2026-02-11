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

#ifndef _MKL_RNG_DEVICE_EXPONENTIAL_IMPL_HPP_
#define _MKL_RNG_DEVICE_EXPONENTIAL_IMPL_HPP_

#include "vm_wrappers.hpp"

namespace oneapi {
namespace mkl {
namespace rng {
namespace device {
namespace detail {

template <typename RealType, typename Method>
class distribution_base<oneapi::mkl::rng::device::exponential<RealType, Method>> {
public:
    struct param_type {
        param_type(RealType a, RealType beta) : a_(a), beta_(beta) {}
        RealType a_;
        RealType beta_;
    };

    distribution_base(RealType a, RealType beta) : a_(a), beta_(beta) {
#ifndef __SYCL_DEVICE_ONLY__
        if (beta <= static_cast<RealType>(0.0)) {
            throw oneapi::mkl::invalid_argument("rng", "exponential", "beta <= 0");
        }
#endif
    }

    RealType a() const {
        return a_;
    }

    RealType beta() const {
        return beta_;
    }

    param_type param() const {
        return param_type(a_, beta_);
    }

    void param(const param_type& pt) {
#ifndef __SYCL_DEVICE_ONLY__
        if (pt.beta_ <= static_cast<RealType>(0.0)) {
            throw oneapi::mkl::invalid_argument("rng", "exponential", "beta <= 0");
        }
#endif
        a_ = pt.a_;
        beta_ = pt.beta_;
    }

protected:
    template <typename EngineType>
    auto generate(EngineType& engine) ->
        typename std::conditional<EngineType::vec_size == 1, RealType,
                                  sycl::vec<RealType, EngineType::vec_size>>::type {
        auto res = uniform_.generate(engine);
        if constexpr (EngineType::vec_size == 1) {
            res = ln_wrapper(res);
        }
        else {
            for (int i = 0; i < EngineType::vec_size; ++i) {
                res[i] = ln_wrapper(res[i]);
            }
        }
        res = a_ - res * beta_;
        if constexpr (std::is_same<Method, exponential_method::icdf_accurate>::value) {
            res = sycl::fmax(res, a_);
        }
        return res;
    }

    template <typename EngineType>
    RealType generate_single(EngineType& engine) {
        RealType res = uniform_.generate_single(engine);
        res = ln_wrapper(res);
        res = a_ - res * beta_;
        if constexpr (std::is_same<Method, exponential_method::icdf_accurate>::value) {
            res = sycl::fmax(res, a_);
        }
        return res;
    }

    distribution_base<oneapi::mkl::rng::device::uniform<RealType>> uniform_ = {
        static_cast<RealType>(0.0), static_cast<RealType>(1.0)
    };
    RealType a_;
    RealType beta_;

    friend class distribution_base<
        oneapi::mkl::rng::device::poisson<std::int32_t, poisson_method::devroye>>;
    friend class distribution_base<
        oneapi::mkl::rng::device::poisson<std::uint32_t, poisson_method::devroye>>;
};

} // namespace detail
} // namespace device
} // namespace rng
} // namespace mkl
} // namespace oneapi

#endif // _MKL_RNG_DEVICE_EXPONENTIAL_IMPL_HPP_
