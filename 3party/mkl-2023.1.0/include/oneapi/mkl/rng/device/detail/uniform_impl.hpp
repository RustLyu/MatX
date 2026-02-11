/*******************************************************************************
* Copyright 2020-2022 Intel Corporation.
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

#ifndef _MKL_RNG_DEVICE_UNIFORM_IMPL_HPP_
#define _MKL_RNG_DEVICE_UNIFORM_IMPL_HPP_

namespace oneapi {
namespace mkl {
namespace rng {
namespace device {
namespace detail {

template <typename Type, typename Method>
class distribution_base<oneapi::mkl::rng::device::uniform<Type, Method>> {
public:
    struct param_type {
        param_type(Type a, Type b) : a_(a), b_(b) {}
        Type a_;
        Type b_;
    };

    distribution_base(Type a, Type b) : a_(a), b_(b) {
#ifndef __SYCL_DEVICE_ONLY__
        if (a >= b) {
            throw oneapi::mkl::invalid_argument("rng", "uniform", "a >= b");
        }
#endif
    }

    Type a() const {
        return a_;
    }

    Type b() const {
        return b_;
    }

    param_type param() const {
        return param_type(a_, b_);
    }

    void param(const param_type& pt) {
#ifndef __SYCL_DEVICE_ONLY__
        if (pt.a_ >= pt.b_) {
            throw oneapi::mkl::invalid_argument("rng", "uniform", "a >= b");
        }
#endif
        a_ = pt.a_;
        b_ = pt.b_;
    }

protected:
    template <typename EngineType>
    auto generate(EngineType& engine) ->
        typename std::conditional<EngineType::vec_size == 1, Type,
                                  sycl::vec<Type, EngineType::vec_size>>::type {
        using OutType = typename std::conditional<EngineType::vec_size == 1, Type,
                                                  sycl::vec<Type, EngineType::vec_size>>::type;
        using FpType =
            typename std::conditional<std::is_same<Method, uniform_method::accurate>::value, double,
                                      float>::type;
        OutType res;
        if constexpr (std::is_integral<Type>::value) {
            if constexpr (EngineType::vec_size == 1) {
                FpType res_fp = engine.generate(static_cast<FpType>(a_), static_cast<FpType>(b_));
                res_fp = sycl::floor(res_fp);
                res = static_cast<Type>(res_fp);
                return res;
            }
            else {
                sycl::vec<FpType, EngineType::vec_size> res_fp;
                res_fp = engine.generate(static_cast<FpType>(a_), static_cast<FpType>(b_));
                res_fp = sycl::floor(res_fp);
                res = res_fp.template convert<Type>();
                return res;
            }
        }
        else {
            res = engine.generate(a_, b_);
            if constexpr (std::is_same<Method, uniform_method::accurate>::value) {
                res = sycl::fmax(res, a_);
                res = sycl::fmin(res, b_);
            }
        }

        return res;
    }

    template <typename EngineType>
    Type generate_single(EngineType& engine) {
        using FpType =
            typename std::conditional<std::is_same<Method, uniform_method::accurate>::value, double,
                                      float>::type;
        Type res;
        if constexpr (std::is_integral<Type>::value) {
            FpType res_fp =
                engine.generate_single(static_cast<FpType>(a_), static_cast<FpType>(b_));
            res_fp = sycl::floor(res_fp);
            res = static_cast<Type>(res_fp);
            return res;
        }
        else {
            res = engine.generate_single(a_, b_);
            if constexpr (std::is_same<Method, uniform_method::accurate>::value) {
                res = sycl::fmax(res, a_);
                res = sycl::fmin(res, b_);
            }
        }

        return res;
    }

    Type a_;
    Type b_;

    friend class distribution_base<
        oneapi::mkl::rng::device::gaussian<Type, gaussian_method::box_muller2>>;
#if MKL_RNG_USE_BINARY_CODE
    friend class distribution_base<
        oneapi::mkl::rng::device::gaussian<Type, gaussian_method::icdf>>;
#endif
    friend class distribution_base<
        oneapi::mkl::rng::device::exponential<Type, exponential_method::icdf>>;
    friend class distribution_base<
        oneapi::mkl::rng::device::exponential<Type, exponential_method::icdf_accurate>>;
    friend class distribution_base<
        oneapi::mkl::rng::device::poisson<std::int32_t, poisson_method::devroye>>;
    friend class distribution_base<
        oneapi::mkl::rng::device::poisson<std::uint32_t, poisson_method::devroye>>;
    friend class distribution_base<
        oneapi::mkl::rng::device::bernoulli<std::int32_t, bernoulli_method::icdf>>;
    friend class distribution_base<
        oneapi::mkl::rng::device::bernoulli<std::uint32_t, bernoulli_method::icdf>>;
};

} // namespace detail
} // namespace device
} // namespace rng
} // namespace mkl
} // namespace oneapi

#endif // _MKL_RNG_DEVICE_UNIFORM_IMPL_HPP_
