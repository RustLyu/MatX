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

#ifndef _MKL_RNG_DEVICE_BITS_IMPL_HPP_
#define _MKL_RNG_DEVICE_BITS_IMPL_HPP_

#include "engine_base.hpp"

namespace oneapi {
namespace mkl {
namespace rng {
namespace device {
namespace detail {

template <typename UIntType>
class distribution_base<oneapi::mkl::rng::device::bits<UIntType>> {
protected:
    template <typename EngineType>
    auto generate(EngineType& engine) -> 
        typename std::enable_if<!std::is_same<EngineType, mcg59<EngineType::vec_size>>::value, 
        typename std::conditional<EngineType::vec_size == 1, UIntType, sycl::vec<UIntType, EngineType::vec_size>>::type>::type 
    {
        static_assert(std::is_same<UIntType, uint32_t>::value,
                  "oneMKL: bits works only with uint32_t");
        return engine.generate();
    }

    template <typename EngineType>
    auto generate(EngineType& engine) ->
        typename std::enable_if<std::is_same<EngineType, mcg59<EngineType::vec_size>>::value, 
        typename std::conditional<EngineType::vec_size == 1, UIntType,
                                  sycl::vec<UIntType, EngineType::vec_size>>::type>::type {
        static_assert(std::is_same<UIntType, uint64_t>::value,
                  "oneMKL: bits with mcg59 works only with uint64_t");
        return engine.generate_bits();
    }

    template <typename EngineType>
    typename std::enable_if<!std::is_same<EngineType, mcg59<EngineType::vec_size>>::value, UIntType>::type
    generate_single(EngineType& engine) {
        static_assert(std::is_same<UIntType, uint32_t>::value,
                  "oneMKL: bits works only with uint32_t");
        return engine.generate_single();
    }

    template <typename EngineType>
    typename std::enable_if<std::is_same<EngineType, mcg59<EngineType::vec_size>>::value, UIntType>::type
    generate_single(EngineType& engine) {
        static_assert(std::is_same<UIntType, uint64_t>::value,
                  "oneMKL: bits with mcg59 works only with uint64_t");
        return engine.generate_single();
    }
};

} // namespace detail
} // namespace device
} // namespace rng
} // namespace mkl
} // namespace oneapi

#endif // _MKL_RNG_DEVICE_BITS_IMPL_HPP_
