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

#ifndef _MKL_RNG_DEVICE_VM_WRAPPERS_HPP_
#define _MKL_RNG_DEVICE_VM_WRAPPERS_HPP_

#include <cmath>

#if MKL_RNG_USE_BINARY_CODE
#include "oneapi/mkl/vm/device/vm.hpp"
#endif

namespace oneapi::mkl::rng::device::detail {

#if MKL_RNG_USE_BINARY_CODE
namespace vm_d = oneapi::mkl::vm::device;
#endif

template<typename DataType>
static inline DataType sqrt_wrapper(DataType a) {
#if MKL_RNG_USE_BINARY_CODE
    DataType t;
    if constexpr(std::is_same_v<DataType, double>)
        vm_d::sqrt(&a, &t, vm_d::mode::ep);
    else
        vm_d::sqrt(&a, &t, vm_d::mode::la);
    return t;
#else
    return sycl::sqrt(a);
#endif // MKL_RNG_USE_BINARY_CODE
}

template<typename DataType>
static inline DataType sinpi_wrapper(DataType a) {
#if MKL_RNG_USE_BINARY_CODE
    DataType t;
    if constexpr(std::is_same_v<DataType, double>)
        vm_d::sinpi(&a, &t, vm_d::mode::ep);
    else
        vm_d::sinpi(&a, &t, vm_d::mode::la);
    return t;
#else
    return sycl::sinpi(a);
#endif // MKL_RNG_USE_BINARY_CODE
}

template<typename DataType>
static inline DataType cospi_wrapper(DataType a) {
#if MKL_RNG_USE_BINARY_CODE
    DataType t;
    if constexpr(std::is_same_v<DataType, double>)
        vm_d::cospi(&a, &t, vm_d::mode::ep);
    else
        vm_d::cospi(&a, &t, vm_d::mode::la);
    return t;
#else
    return sycl::cospi(a);
#endif // MKL_RNG_USE_BINARY_CODE
}

template<typename DataType>
static inline DataType sincospi_wrapper(DataType a, DataType& b) {
#if MKL_RNG_USE_BINARY_CODE
    DataType t;
    if constexpr(std::is_same_v<DataType, double>) {
        vm_d::sincospi(&a, &t, &b, vm_d::mode::ep);
    }
    else {
        vm_d::sincospi(&a, &t, &b, vm_d::mode::la);
    }
    return t;
#else
    b = sycl::cospi(a);
    return sycl::sinpi(a);
#endif // MKL_RNG_USE_BINARY_CODE
}

template<typename DataType>
static inline DataType ln_wrapper(DataType a) {
    if(a == DataType(0)){
        if constexpr(std::is_same_v<DataType, double>)
            return -0x1.74385446D71C3P+9; // ln(0.494065e-323) = -744.440072
        else
            return -0x1.9D1DA0P+6f; // ln(0.14012984e-44) = -103.278929
    }

#if MKL_RNG_USE_BINARY_CODE
    DataType t;
    if constexpr(std::is_same_v<DataType, double>)
        vm_d::ln(&a, &t, vm_d::mode::ep);
    else
        vm_d::ln(&a, &t, vm_d::mode::la);
    return t;
#else
    return sycl::log(a);
#endif // MKL_RNG_USE_BINARY_CODE
}

#if MKL_RNG_USE_BINARY_CODE
template <typename RealType>
inline RealType erf_inv_wrapper(RealType x) {
    if (std::fabs(x) == RealType(1)) {
        if constexpr (std::is_same_v<RealType, float>) {
            return std::copysign(0x1.EA8F96P+1f, x); // ErfInv(0.99999994039) = 3.83250689506
        }
        else {
            return std::copysign(0x1.7744F8F74E94AP+2, x); // ErfInv(0.999999999) = 5.86358474875
        }
    }
    else {
        RealType res{};
        if constexpr(std::is_same_v<RealType, double>) {
            vm_d::erfinv(&x, &res, vm_d::mode::ep);
        }
        else {
            vm_d::erfinv(&x, &res, vm_d::mode::la);
        }
        return res;
    }
}
#endif

} // namespace oneapi::mkl::rng::device::detail

#endif // _MKL_RNG_DEVICE_VM_WRAPPERS_HPP_
