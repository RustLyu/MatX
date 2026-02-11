/*******************************************************************************
* Copyright 2023 Intel Corporation.
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

#ifndef _MKL_RNG_DEVICE_UNIFORM_BITS_IMPL_HPP_
#define _MKL_RNG_DEVICE_UNIFORM_BITS_IMPL_HPP_

#include "engine_base.hpp"

namespace oneapi {
namespace mkl {
namespace rng {
namespace device {
namespace detail {

template <typename UIntType>
class distribution_base<oneapi::mkl::rng::device::uniform_bits<UIntType>> {
protected:
    template <typename EngineType>
    auto generate(EngineType& engine) -> 
        typename std::conditional<EngineType::vec_size == 1, UIntType, sycl::vec<UIntType, EngineType::vec_size>>::type
    {
        static_assert(std::is_same<EngineType, philox4x32x10<EngineType::vec_size>>::value ||
                  std::is_same<EngineType, mcg59<EngineType::vec_size>>::value, 
                  "oneMKL: uniform_bits works only with philox4x32x10/mcg59 engines");
        return engine.template generate_uniform_bits<UIntType>();
    }

    template <typename EngineType>
    UIntType generate_single(EngineType& engine) 
    {
        static_assert(std::is_same<EngineType, philox4x32x10<EngineType::vec_size>>::value ||
                  std::is_same<EngineType, mcg59<EngineType::vec_size>>::value, 
                  "oneMKL: uniform_bits works only with philox4x32x10/mcg59 engines");
        return engine.template generate_single_uniform_bits<UIntType>();
    }
};

} // namespace detail
} // namespace device
} // namespace rng
} // namespace mkl
} // namespace oneapi

#endif // _MKL_RNG_DEVICE_UNIFORM_BITS_IMPL_HPP_
