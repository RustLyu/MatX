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

#ifndef _MKL_RNG_DISTRIBUTION_BASE_HPP_
#define _MKL_RNG_DISTRIBUTION_BASE_HPP_

#include <CL/sycl.hpp>

#include "oneapi/mkl/exceptions.hpp"
#include "oneapi/mkl/rng/device/types.hpp"
#include "oneapi/mkl/rng/device/detail/types.hpp"

namespace oneapi {
namespace mkl {
namespace rng {
namespace device {
namespace detail {

template <typename DistrType>
class distribution_base {};

namespace distr_common {

// sqrt(2)
template <typename RealType = float>
inline RealType sqrt2() {
    return 0x1.6A09E6P+0f; // 1.414213562
}

template <>
inline double sqrt2<double>() {
    return 0x1.6A09E667F3BCDP+0; // 1.414213562
}

} // namespace distr_common
} // namespace detail

// declarations of distribution classes
template <typename Type = float, typename Method = uniform_method::by_default>
class uniform;

template <typename RealType = float, typename Method = gaussian_method::by_default>
class gaussian;

template <typename RealType = float, typename Method = lognormal_method::by_default>
class lognormal;

template <typename UIntType = std::uint32_t>
class uniform_bits;

template <typename UIntType = std::uint32_t>
class bits;

template <typename RealType = float, typename Method = exponential_method::by_default>
class exponential;

template <typename IntType = std::int32_t, typename Method = poisson_method::by_default>
class poisson;

template <typename IntType = std::uint32_t, typename Method = bernoulli_method::by_default>
class bernoulli;

} // namespace device
} // namespace rng
} // namespace mkl
} // namespace oneapi

#include "oneapi/mkl/rng/device/detail/uniform_impl.hpp"
#include "oneapi/mkl/rng/device/detail/gaussian_impl.hpp"
#include "oneapi/mkl/rng/device/detail/lognormal_impl.hpp"
#include "oneapi/mkl/rng/device/detail/bits_impl.hpp"
#include "oneapi/mkl/rng/device/detail/uniform_bits_impl.hpp"
#include "oneapi/mkl/rng/device/detail/exponential_impl.hpp"
#include "oneapi/mkl/rng/device/detail/poisson_impl.hpp"
#include "oneapi/mkl/rng/device/detail/bernoulli_impl.hpp"

#endif // _MKL_RNG_DISTRIBUTION_BASE_HPP_
