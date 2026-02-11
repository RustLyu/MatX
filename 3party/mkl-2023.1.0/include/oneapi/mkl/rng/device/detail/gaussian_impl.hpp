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

#ifndef _MKL_RNG_DEVICE_GAUSSIAN_IMPL_HPP_
#define _MKL_RNG_DEVICE_GAUSSIAN_IMPL_HPP_

#include "vm_wrappers.hpp"

namespace oneapi {
namespace mkl {
namespace rng {
namespace device {
namespace detail {

template <typename RealType>
class distribution_base<
    oneapi::mkl::rng::device::gaussian<RealType, gaussian_method::box_muller2>> {
public:
    struct param_type {
        param_type(RealType mean, RealType stddev) : mean_(mean), stddev_(stddev) {}
        RealType mean_;
        RealType stddev_;
    };

    distribution_base(RealType mean, RealType stddev) : mean_(mean), stddev_(stddev) {
        flag_ = false;
#ifndef __SYCL_DEVICE_ONLY__
        if (stddev <= RealType(0)) {
            throw oneapi::mkl::invalid_argument("rng", "gaussian", "stddev <= 0");
        }
#endif
    }

    RealType mean() const {
        return mean_;
    }

    RealType stddev() const {
        return stddev_;
    }

    param_type param() const {
        return param_type(mean_, stddev_);
    }

    void param(const param_type& pt) {
#ifndef __SYCL_DEVICE_ONLY__
        if (pt.stddev_ <= RealType(0)) {
            throw oneapi::mkl::invalid_argument("rng", "gaussian", "stddev <= 0");
        }
#endif
        mean_ = pt.mean_;
        stddev_ = pt.stddev_;
    }

protected:

    template <typename EngineType>
    __attribute__((always_inline))
    inline auto generate(EngineType& engine) ->
        typename std::conditional<EngineType::vec_size == 1, RealType,
                                  sycl::vec<RealType, EngineType::vec_size>>::type {
        RealType u1, u2, u1_transformed;

        if constexpr (EngineType::vec_size == 1) {
            RealType res;
            if (!flag_) {
                u1 = engine.generate(RealType(0), RealType(1));
                u2 = engine.generate(RealType(0), RealType(1));
                u1_transformed = ln_wrapper(u1);
                u1_transformed = sqrt_wrapper(static_cast<RealType>(-2.0) * u1_transformed);
                res = u1_transformed * sinpi_wrapper(RealType(2) * u2) * stddev_ + mean_;
                u1_transformed_ = u1_transformed;
                u2_ = u2;
                flag_ = true;
                return res;
            }
            res = u1_transformed_ * cospi_wrapper(RealType(2) * u2_) * stddev_ + mean_;
            flag_ = false;
            return res;
        }
        else {
            RealType sin, cos;
            sycl::vec<RealType, EngineType::vec_size> res;
            if (!flag_) {
                constexpr std::int32_t tail = EngineType::vec_size % 2;
                auto uniform_res = engine.generate(RealType(0), RealType(1));
#pragma unroll
                for (std::int32_t i = 0; i < EngineType::vec_size - tail; i += 2) {
                    u1 = uniform_res[i];
                    u2 = uniform_res[i + 1];
                    u1_transformed = ln_wrapper(u1);
                    u1_transformed = sqrt_wrapper(static_cast<RealType>(-2.0) * u1_transformed);
                    sin = sincospi_wrapper(RealType(2.0) * u2, cos);
                    res[i] = (u1_transformed * sin) * stddev_ + mean_;
                    res[i + 1] = (u1_transformed * cos) * stddev_ + mean_;
                }
                if constexpr (tail) {
                    u1 = uniform_res[EngineType::vec_size - 1];
                    u2 = engine.generate_single(RealType(0), RealType(1));
                    u1_transformed = ln_wrapper(u1);
                    u1_transformed = sqrt_wrapper(static_cast<RealType>(-2.0) * u1_transformed);
                    res[EngineType::vec_size - 1] =
                        u1_transformed * sinpi_wrapper(RealType(2) * u2) * stddev_ + mean_;
                    u1_transformed_ = u1_transformed;
                    u2_ = u2;
                    flag_ = true;
                }
                return res;
            }

            res[0] = u1_transformed_ * cospi_wrapper(RealType(2) * u2_) * stddev_ + mean_;
            flag_ = false;
            constexpr std::int32_t tail = (EngineType::vec_size - 1) % 2;
#pragma unroll
            for (std::int32_t i = 1; i < EngineType::vec_size - tail; i += 2) {
                u1 = engine.generate_single(RealType(0), RealType(1));
                u2 = engine.generate_single(RealType(0), RealType(1));
                u1_transformed = ln_wrapper(u1);
                u1_transformed = sqrt_wrapper(static_cast<RealType>(-2.0) * u1_transformed);
                sin = sincospi_wrapper(RealType(2.0) * u2, cos);
                res[i] = (u1_transformed * sin) * stddev_ + mean_;
                res[i + 1] = (u1_transformed * cos) * stddev_ + mean_;
            }
            if constexpr (tail) {
                u1 = engine.generate_single(RealType(0), RealType(1));
                u2 = engine.generate_single(RealType(0), RealType(1));
                u1_transformed = ln_wrapper(u1);
                u1_transformed = sqrt_wrapper(static_cast<RealType>(-2.0) * u1_transformed);
                res[EngineType::vec_size - 1] =
                    u1_transformed * sinpi_wrapper(RealType(2) * u2) * stddev_ + mean_;
                u1_transformed_ = u1_transformed;
                u2_ = u2;
                flag_ = true;
            }
            return res;
        }
    }

    template <typename EngineType>
    __attribute__((always_inline))
    inline RealType generate_single(EngineType& engine) {
        RealType u1, u2, u1_transformed;
        RealType res;
        if (!flag_) {
            u1 = engine.generate_single(RealType(0), RealType(1));
            u2 = engine.generate_single(RealType(0), RealType(1));
            u1_transformed = ln_wrapper(u1);
            u1_transformed = sqrt_wrapper(static_cast<RealType>(-2.0) * u1_transformed);
            res = u1_transformed * sinpi_wrapper(RealType(2) * u2) * stddev_ + mean_;
            u1_transformed_ = u1_transformed;
            u2_ = u2;
            flag_ = true;
            return res;
        }
        res = u1_transformed_ * cospi_wrapper(RealType(2) * u2_) * stddev_ + mean_;
        flag_ = false;
        return res;
    }

    distribution_base<oneapi::mkl::rng::device::uniform<RealType>> uniform_ = {
        RealType(0), static_cast<RealType>(1.0)
    };
    RealType mean_;
    RealType stddev_;
    bool flag_ = false;
    RealType u1_transformed_;
    RealType u2_;

    friend class distribution_base<
        oneapi::mkl::rng::device::lognormal<RealType, lognormal_method::box_muller2>>;
    friend class distribution_base<
        oneapi::mkl::rng::device::poisson<std::int32_t, poisson_method::devroye>>;
    friend class distribution_base<
        oneapi::mkl::rng::device::poisson<std::uint32_t, poisson_method::devroye>>;
};

#if MKL_RNG_USE_BINARY_CODE

template <typename RealType>
class distribution_base<
    oneapi::mkl::rng::device::gaussian<RealType, gaussian_method::icdf>> {
public:
    struct param_type {
        param_type(RealType mean, RealType stddev) : mean_(mean), stddev_(stddev) {}
        RealType mean_;
        RealType stddev_;
    };

    distribution_base(RealType mean, RealType stddev) : mean_(mean), stddev_(stddev) {
#ifndef __SYCL_DEVICE_ONLY__
        if (stddev <= RealType(0)) {
            throw oneapi::mkl::invalid_argument("rng", "gaussian", "stddev <= 0");
        }
#endif
    }

    RealType mean() const {
        return mean_;
    }

    RealType stddev() const {
        return stddev_;
    }

    param_type param() const {
        return param_type(mean_, stddev_);
    }

    void param(const param_type& pt) {
#ifndef __SYCL_DEVICE_ONLY__
        if (pt.stddev_ <= RealType(0)) {
            throw oneapi::mkl::invalid_argument("rng", "gaussian", "stddev <= 0");
        }
#endif
        mean_ = pt.mean_;
        stddev_ = pt.stddev_;
    }

protected:

    template <typename EngineType>
    __attribute__((always_inline))
    inline auto generate(EngineType& engine) ->
        typename std::conditional<EngineType::vec_size == 1, RealType,
                                  sycl::vec<RealType, EngineType::vec_size>>::type {
        if constexpr (EngineType::vec_size == 1) {
            return generate_single(engine);
        }
        else {
            RealType stddev = stddev_ * distr_common::sqrt2<RealType>();
            sycl::vec<RealType, EngineType::vec_size> res;
            sycl::vec<RealType, EngineType::vec_size> u =
                engine.generate(RealType(-1), RealType(1));
            for (std::int32_t i = 0; i < EngineType::vec_size; i++){
                res[i] = erf_inv_wrapper(u[i]);
            }
            return res * stddev + mean_;
        }
    }

    template <typename EngineType>
    __attribute__((always_inline))
    inline RealType generate_single(EngineType& engine) {
        RealType stddev = stddev_ * distr_common::sqrt2<RealType>();
        RealType u = engine.generate_single(RealType(-1), RealType(1));
        return sycl::fma(erf_inv_wrapper(u), stddev, mean_);
    }

    RealType mean_;
    RealType stddev_;
};
#endif

} // namespace detail
} // namespace device
} // namespace rng
} // namespace mkl
} // namespace oneapi

#endif // _MKL_RNG_DEVICE_GAUSSIAN_IMPL_HPP_
