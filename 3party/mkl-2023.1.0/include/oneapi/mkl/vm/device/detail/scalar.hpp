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

#ifndef __ONEAPI_MKL_VM_DEVICE_DETAIL_SCALAR_HPP__
#define __ONEAPI_MKL_VM_DEVICE_DETAIL_SCALAR_HPP__ 1

#include <complex>
#include <type_traits>
#include <tuple>
#include <sycl/sycl.hpp>
#include "oneapi/mkl/vm/device/detail/decls.hpp"

namespace oneapi::mkl::vm::detail::gpu::intel::scalar {
SYCL_EXTERNAL int abs_s_ep_gen(const float* a, float* y);
SYCL_EXTERNAL int abs_s_la_gen(const float* a, float* y);
SYCL_EXTERNAL int abs_s_ha_gen(const float* a, float* y);
SYCL_EXTERNAL int abs_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int abs_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int abs_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int abs_c_ep_gen(const std::complex<float>* a, float* y);
SYCL_EXTERNAL int abs_c_la_gen(const std::complex<float>* a, float* y);
SYCL_EXTERNAL int abs_c_ha_gen(const std::complex<float>* a, float* y);
SYCL_EXTERNAL int abs_z_ep_gen(const std::complex<double>* a, double* y);
SYCL_EXTERNAL int abs_z_la_gen(const std::complex<double>* a, double* y);
SYCL_EXTERNAL int abs_z_ha_gen(const std::complex<double>* a, double* y);
SYCL_EXTERNAL int acos_s_ep_gen(const float* a, float* y);
SYCL_EXTERNAL int acos_s_la_gen(const float* a, float* y);
SYCL_EXTERNAL int acos_s_ha_gen(const float* a, float* y);
SYCL_EXTERNAL int acos_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int acos_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int acos_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int acosh_s_ep_gen(const float* a, float* y);
SYCL_EXTERNAL int acosh_s_la_gen(const float* a, float* y);
SYCL_EXTERNAL int acosh_s_ha_gen(const float* a, float* y);
SYCL_EXTERNAL int acosh_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int acosh_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int acosh_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int acospi_s_ep_gen(const float* a, float* y);
SYCL_EXTERNAL int acospi_s_la_gen(const float* a, float* y);
SYCL_EXTERNAL int acospi_s_ha_gen(const float* a, float* y);
SYCL_EXTERNAL int acospi_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int acospi_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int acospi_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int add_s_ep_gen(const float* a, const float* b, float* y);
SYCL_EXTERNAL int add_s_la_gen(const float* a, const float* b, float* y);
SYCL_EXTERNAL int add_s_ha_gen(const float* a, const float* b, float* y);
SYCL_EXTERNAL int add_d_ep_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int add_d_la_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int add_d_ha_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int add_c_ep_gen(const std::complex<float>* a, const std::complex<float>* b, std::complex<float>* y);
SYCL_EXTERNAL int add_c_la_gen(const std::complex<float>* a, const std::complex<float>* b, std::complex<float>* y);
SYCL_EXTERNAL int add_c_ha_gen(const std::complex<float>* a, const std::complex<float>* b, std::complex<float>* y);
SYCL_EXTERNAL int add_z_ep_gen(const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* y);
SYCL_EXTERNAL int add_z_la_gen(const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* y);
SYCL_EXTERNAL int add_z_ha_gen(const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* y);
SYCL_EXTERNAL int arg_c_ep_gen(const std::complex<float>* a, float* y);
SYCL_EXTERNAL int arg_c_la_gen(const std::complex<float>* a, float* y);
SYCL_EXTERNAL int arg_c_ha_gen(const std::complex<float>* a, float* y);
SYCL_EXTERNAL int arg_z_ep_gen(const std::complex<double>* a, double* y);
SYCL_EXTERNAL int arg_z_la_gen(const std::complex<double>* a, double* y);
SYCL_EXTERNAL int arg_z_ha_gen(const std::complex<double>* a, double* y);
SYCL_EXTERNAL int asin_s_ep_gen(const float* a, float* y);
SYCL_EXTERNAL int asin_s_la_gen(const float* a, float* y);
SYCL_EXTERNAL int asin_s_ha_gen(const float* a, float* y);
SYCL_EXTERNAL int asin_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int asin_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int asin_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int asinh_s_ep_gen(const float* a, float* y);
SYCL_EXTERNAL int asinh_s_la_gen(const float* a, float* y);
SYCL_EXTERNAL int asinh_s_ha_fp64only(const float* a, float* y);
SYCL_EXTERNAL int asinh_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int asinh_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int asinh_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int asinpi_s_ep_gen(const float* a, float* y);
SYCL_EXTERNAL int asinpi_s_la_gen(const float* a, float* y);
SYCL_EXTERNAL int asinpi_s_ha_gen(const float* a, float* y);
SYCL_EXTERNAL int asinpi_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int asinpi_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int asinpi_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int atan_s_ep_gen(const float* a, float* y);
SYCL_EXTERNAL int atan_s_la_gen(const float* a, float* y);
SYCL_EXTERNAL int atan_s_ha_gen(const float* a, float* y);
SYCL_EXTERNAL int atan_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int atan_d_la_nolut(const double* a, double* y);
SYCL_EXTERNAL int atan_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int atan2_s_ep_gen(const float* a, const float* b, float* y);
SYCL_EXTERNAL int atan2_s_la_gen(const float* a, const float* b, float* y);
SYCL_EXTERNAL int atan2_s_ha_gen(const float* a, const float* b, float* y);
SYCL_EXTERNAL int atan2_d_ep_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int atan2_d_la_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int atan2_d_ha_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int atan2pi_s_ep_fp64only(const float* a, const float* b, float* y);
SYCL_EXTERNAL int atan2pi_s_la_gen(const float* a, const float* b, float* y);
SYCL_EXTERNAL int atan2pi_s_ha_fp64only(const float* a, const float* b, float* y);
SYCL_EXTERNAL int atan2pi_d_ep_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int atan2pi_d_la_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int atan2pi_d_ha_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int atanh_s_ep_gen(const float* a, float* y);
SYCL_EXTERNAL int atanh_s_la_gen(const float* a, float* y);
SYCL_EXTERNAL int atanh_s_ha_gen(const float* a, float* y);
SYCL_EXTERNAL int atanh_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int atanh_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int atanh_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int atanpi_s_ep_gen(const float* a, float* y);
SYCL_EXTERNAL int atanpi_s_la_gen(const float* a, float* y);
SYCL_EXTERNAL int atanpi_s_ha_gen(const float* a, float* y);
SYCL_EXTERNAL int atanpi_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int atanpi_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int atanpi_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int cbrt_s_ep_gen(const float* a, float* y);
SYCL_EXTERNAL int cbrt_s_la_gen(const float* a, float* y);
SYCL_EXTERNAL int cbrt_s_ha_gen(const float* a, float* y);
SYCL_EXTERNAL int cbrt_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int cbrt_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int cbrt_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int cdfnorm_s_ep_fp64only(const float* a, float* y);
SYCL_EXTERNAL int cdfnorm_s_la_fp64only(const float* a, float* y);
SYCL_EXTERNAL int cdfnorm_s_ha_fp64only(const float* a, float* y);
SYCL_EXTERNAL int cdfnorm_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int cdfnorm_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int cdfnorm_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int cdfnorminv_s_ep_fp64only(const float* a, float* y);
SYCL_EXTERNAL int cdfnorminv_s_la_fp64only(const float* a, float* y);
SYCL_EXTERNAL int cdfnorminv_s_ha_fp64only(const float* a, float* y);
SYCL_EXTERNAL int cdfnorminv_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int cdfnorminv_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int cdfnorminv_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int ceil_s_ep_gen(const float* a, float* y);
SYCL_EXTERNAL int ceil_s_la_gen(const float* a, float* y);
SYCL_EXTERNAL int ceil_s_ha_gen(const float* a, float* y);
SYCL_EXTERNAL int ceil_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int ceil_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int ceil_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int cis_c_ep_gen(const float* a, std::complex<float>* y);
SYCL_EXTERNAL int cis_c_la_gen(const float* a, std::complex<float>* y);
SYCL_EXTERNAL int cis_c_ha_gen(const float* a, std::complex<float>* y);
SYCL_EXTERNAL int cis_z_ep_gen(const double* a, std::complex<double>* y);
SYCL_EXTERNAL int cis_z_la_gen(const double* a, std::complex<double>* y);
SYCL_EXTERNAL int cis_z_ha_gen(const double* a, std::complex<double>* y);
SYCL_EXTERNAL int conj_c_ep_gen(const std::complex<float>* a, std::complex<float>* y);
SYCL_EXTERNAL int conj_c_la_gen(const std::complex<float>* a, std::complex<float>* y);
SYCL_EXTERNAL int conj_c_ha_gen(const std::complex<float>* a, std::complex<float>* y);
SYCL_EXTERNAL int conj_z_ep_gen(const std::complex<double>* a, std::complex<double>* y);
SYCL_EXTERNAL int conj_z_la_gen(const std::complex<double>* a, std::complex<double>* y);
SYCL_EXTERNAL int conj_z_ha_gen(const std::complex<double>* a, std::complex<double>* y);
SYCL_EXTERNAL int copysign_s_ep_gen(const float* a, const float* b, float* y);
SYCL_EXTERNAL int copysign_s_la_gen(const float* a, const float* b, float* y);
SYCL_EXTERNAL int copysign_s_ha_gen(const float* a, const float* b, float* y);
SYCL_EXTERNAL int copysign_d_ep_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int copysign_d_la_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int copysign_d_ha_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int cos_s_ep_gen(const float* a, float* y);
SYCL_EXTERNAL int cos_s_la_gen(const float* a, float* y);
SYCL_EXTERNAL int cos_s_ha_fp64only(const float* a, float* y);
SYCL_EXTERNAL int cos_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int cos_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int cos_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int cosd_s_ep_fp64only(const float* a, float* y);
SYCL_EXTERNAL int cosd_s_la_fp64only(const float* a, float* y);
SYCL_EXTERNAL int cosd_s_ha_fp64only(const float* a, float* y);
SYCL_EXTERNAL int cosd_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int cosd_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int cosd_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int cosh_s_ep_gen(const float* a, float* y);
SYCL_EXTERNAL int cosh_s_la_gen(const float* a, float* y);
SYCL_EXTERNAL int cosh_s_ha_fp64only(const float* a, float* y);
SYCL_EXTERNAL int cosh_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int cosh_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int cosh_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int cospi_s_ep_gen(const float* a, float* y);
SYCL_EXTERNAL int cospi_s_la_gen(const float* a, float* y);
SYCL_EXTERNAL int cospi_s_ha_fp64only(const float* a, float* y);
SYCL_EXTERNAL int cospi_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int cospi_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int cospi_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int div_s_ep_gen(const float* a, const float* b, float* y);
SYCL_EXTERNAL int div_s_la_gen(const float* a, const float* b, float* y);
SYCL_EXTERNAL int div_s_ha_fp64only(const float* a, const float* b, float* y);
SYCL_EXTERNAL int div_d_ep_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int div_d_la_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int div_d_ha_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int div_c_ep_gen(const std::complex<float>* a, const std::complex<float>* b, std::complex<float>* y);
SYCL_EXTERNAL int div_c_la_gen(const std::complex<float>* a, const std::complex<float>* b, std::complex<float>* y);
SYCL_EXTERNAL int div_c_ha_gen(const std::complex<float>* a, const std::complex<float>* b, std::complex<float>* y);
SYCL_EXTERNAL int div_z_ep_gen(const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* y);
SYCL_EXTERNAL int div_z_la_gen(const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* y);
SYCL_EXTERNAL int div_z_ha_gen(const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* y);
SYCL_EXTERNAL int erf_s_ep_gen(const float* a, float* y);
SYCL_EXTERNAL int erf_s_la_gen(const float* a, float* y);
SYCL_EXTERNAL int erf_s_ha_gen(const float* a, float* y);
SYCL_EXTERNAL int erf_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int erf_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int erf_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int erfc_s_ep_fp64only(const float* a, float* y);
SYCL_EXTERNAL int erfc_s_la_fp64only(const float* a, float* y);
SYCL_EXTERNAL int erfc_s_ha_fp64only(const float* a, float* y);
SYCL_EXTERNAL int erfc_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int erfc_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int erfc_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int erfcx_s_ep_gen(const float* a, float* y);
SYCL_EXTERNAL int erfcx_s_la_gen(const float* a, float* y);
SYCL_EXTERNAL int erfcx_s_ha_gen(const float* a, float* y);
SYCL_EXTERNAL int erfcx_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int erfcx_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int erfcx_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int erfcinv_s_ep_fp64only(const float* a, float* y);
SYCL_EXTERNAL int erfcinv_s_la_fp64only(const float* a, float* y);
SYCL_EXTERNAL int erfcinv_s_ha_fp64only(const float* a, float* y);
SYCL_EXTERNAL int erfcinv_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int erfcinv_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int erfcinv_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int erfinv_s_ep_nolut(const float* a, float* y);
SYCL_EXTERNAL int erfinv_s_la_nolut(const float* a, float* y);
SYCL_EXTERNAL int erfinv_s_ha_nolut(const float* a, float* y);
SYCL_EXTERNAL int erfinv_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int erfinv_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int erfinv_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int exp_s_ep_gen(const float* a, float* y);
SYCL_EXTERNAL int exp_s_la_gen(const float* a, float* y);
SYCL_EXTERNAL int exp_s_ha_gen(const float* a, float* y);
SYCL_EXTERNAL int exp_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int exp_d_la_nolut(const double* a, double* y);
SYCL_EXTERNAL int exp_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int exp_c_ep_gen(const std::complex<float>* a, std::complex<float>* y);
SYCL_EXTERNAL int exp_c_la_gen(const std::complex<float>* a, std::complex<float>* y);
SYCL_EXTERNAL int exp_c_ha_gen(const std::complex<float>* a, std::complex<float>* y);
SYCL_EXTERNAL int exp_z_ep_gen(const std::complex<double>* a, std::complex<double>* y);
SYCL_EXTERNAL int exp_z_la_gen(const std::complex<double>* a, std::complex<double>* y);
SYCL_EXTERNAL int exp_z_ha_nolut(const std::complex<double>* a, std::complex<double>* y);
SYCL_EXTERNAL int exp10_s_ep_gen(const float* a, float* y);
SYCL_EXTERNAL int exp10_s_la_gen(const float* a, float* y);
SYCL_EXTERNAL int exp10_s_ha_gen(const float* a, float* y);
SYCL_EXTERNAL int exp10_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int exp10_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int exp10_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int exp2_s_ep_gen(const float* a, float* y);
SYCL_EXTERNAL int exp2_s_la_gen(const float* a, float* y);
SYCL_EXTERNAL int exp2_s_ha_gen(const float* a, float* y);
SYCL_EXTERNAL int exp2_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int exp2_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int exp2_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int expm1_s_ep_gen(const float* a, float* y);
SYCL_EXTERNAL int expm1_s_la_gen(const float* a, float* y);
SYCL_EXTERNAL int expm1_s_ha_gen(const float* a, float* y);
SYCL_EXTERNAL int expm1_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int expm1_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int expm1_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int fdim_s_ep_gen(const float* a, const float* b, float* y);
SYCL_EXTERNAL int fdim_s_la_gen(const float* a, const float* b, float* y);
SYCL_EXTERNAL int fdim_s_ha_gen(const float* a, const float* b, float* y);
SYCL_EXTERNAL int fdim_d_ep_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int fdim_d_la_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int fdim_d_ha_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int floor_s_ep_gen(const float* a, float* y);
SYCL_EXTERNAL int floor_s_la_gen(const float* a, float* y);
SYCL_EXTERNAL int floor_s_ha_gen(const float* a, float* y);
SYCL_EXTERNAL int floor_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int floor_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int floor_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int fmax_s_ep_gen(const float* a, const float* b, float* y);
SYCL_EXTERNAL int fmax_s_la_gen(const float* a, const float* b, float* y);
SYCL_EXTERNAL int fmax_s_ha_gen(const float* a, const float* b, float* y);
SYCL_EXTERNAL int fmax_d_ep_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int fmax_d_la_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int fmax_d_ha_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int fmin_s_ep_gen(const float* a, const float* b, float* y);
SYCL_EXTERNAL int fmin_s_la_gen(const float* a, const float* b, float* y);
SYCL_EXTERNAL int fmin_s_ha_gen(const float* a, const float* b, float* y);
SYCL_EXTERNAL int fmin_d_ep_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int fmin_d_la_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int fmin_d_ha_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int fmod_s_ep_gen(const float* a, const float* b, float* y);
SYCL_EXTERNAL int fmod_s_la_gen(const float* a, const float* b, float* y);
SYCL_EXTERNAL int fmod_s_ha_gen(const float* a, const float* b, float* y);
SYCL_EXTERNAL int fmod_d_ep_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int fmod_d_la_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int fmod_d_ha_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int frac_s_ep_fp64only(const float* a, float* y);
SYCL_EXTERNAL int frac_s_la_fp64only(const float* a, float* y);
SYCL_EXTERNAL int frac_s_ha_fp64only(const float* a, float* y);
SYCL_EXTERNAL int frac_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int frac_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int frac_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int hypot_s_ep_fp64only(const float* a, const float* b, float* y);
SYCL_EXTERNAL int hypot_s_la_fp64only(const float* a, const float* b, float* y);
SYCL_EXTERNAL int hypot_s_ha_fp64only(const float* a, const float* b, float* y);
SYCL_EXTERNAL int hypot_d_ep_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int hypot_d_la_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int hypot_d_ha_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int inv_s_ep_gen(const float* a, float* y);
SYCL_EXTERNAL int inv_s_la_gen(const float* a, float* y);
SYCL_EXTERNAL int inv_s_ha_gen(const float* a, float* y);
SYCL_EXTERNAL int inv_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int inv_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int inv_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int invcbrt_s_ep_fp64only(const float* a, float* y);
SYCL_EXTERNAL int invcbrt_s_la_fp64only(const float* a, float* y);
SYCL_EXTERNAL int invcbrt_s_ha_fp64only(const float* a, float* y);
SYCL_EXTERNAL int invcbrt_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int invcbrt_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int invcbrt_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int invsqrt_s_ep_fp64only(const float* a, float* y);
SYCL_EXTERNAL int invsqrt_s_la_fp64only(const float* a, float* y);
SYCL_EXTERNAL int invsqrt_s_ha_fp64only(const float* a, float* y);
SYCL_EXTERNAL int invsqrt_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int invsqrt_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int invsqrt_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int lgamma_s_ep_fp64only(const float* a, float* y);
SYCL_EXTERNAL int lgamma_s_la_fp64only(const float* a, float* y);
SYCL_EXTERNAL int lgamma_s_ha_fp64only(const float* a, float* y);
SYCL_EXTERNAL int ln_s_ep_gen(const float* a, float* y);
SYCL_EXTERNAL int ln_s_la_gen(const float* a, float* y);
SYCL_EXTERNAL int ln_s_ha_gen(const float* a, float* y);
SYCL_EXTERNAL int ln_d_ep_nolut(const double* a, double* y);
SYCL_EXTERNAL int ln_d_la_nolut(const double* a, double* y);
SYCL_EXTERNAL int ln_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int ln_c_ep_gen(const std::complex<float>* a, std::complex<float>* y);
SYCL_EXTERNAL int ln_c_la_gen(const std::complex<float>* a, std::complex<float>* y);
SYCL_EXTERNAL int ln_c_ha_gen(const std::complex<float>* a, std::complex<float>* y);
SYCL_EXTERNAL int ln_z_ep_gen(const std::complex<double>* a, std::complex<double>* y);
SYCL_EXTERNAL int ln_z_la_gen(const std::complex<double>* a, std::complex<double>* y);
SYCL_EXTERNAL int ln_z_ha_nolut(const std::complex<double>* a, std::complex<double>* y);
SYCL_EXTERNAL int log10_s_ep_gen(const float* a, float* y);
SYCL_EXTERNAL int log10_s_la_gen(const float* a, float* y);
SYCL_EXTERNAL int log10_s_ha_fp64only(const float* a, float* y);
SYCL_EXTERNAL int log10_d_ep_nolut(const double* a, double* y);
SYCL_EXTERNAL int log10_d_la_nolut(const double* a, double* y);
SYCL_EXTERNAL int log10_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int log1p_s_ep_gen(const float* a, float* y);
SYCL_EXTERNAL int log1p_s_la_gen(const float* a, float* y);
SYCL_EXTERNAL int log1p_s_ha_gen(const float* a, float* y);
SYCL_EXTERNAL int log1p_d_ep_nolut(const double* a, double* y);
SYCL_EXTERNAL int log1p_d_la_nolut(const double* a, double* y);
SYCL_EXTERNAL int log1p_d_ha_nolut(const double* a, double* y);
SYCL_EXTERNAL int log2_s_ep_gen(const float* a, float* y);
SYCL_EXTERNAL int log2_s_la_gen(const float* a, float* y);
SYCL_EXTERNAL int log2_s_ha_fp64only(const float* a, float* y);
SYCL_EXTERNAL int log2_d_ep_nolut(const double* a, double* y);
SYCL_EXTERNAL int log2_d_la_nolut(const double* a, double* y);
SYCL_EXTERNAL int log2_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int logb_s_ep_gen(const float* a, float* y);
SYCL_EXTERNAL int logb_s_la_gen(const float* a, float* y);
SYCL_EXTERNAL int logb_s_ha_gen(const float* a, float* y);
SYCL_EXTERNAL int logb_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int logb_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int logb_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int maxmag_s_ep_gen(const float* a, const float* b, float* y);
SYCL_EXTERNAL int maxmag_s_la_gen(const float* a, const float* b, float* y);
SYCL_EXTERNAL int maxmag_s_ha_gen(const float* a, const float* b, float* y);
SYCL_EXTERNAL int maxmag_d_ep_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int maxmag_d_la_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int maxmag_d_ha_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int minmag_s_ep_gen(const float* a, const float* b, float* y);
SYCL_EXTERNAL int minmag_s_la_gen(const float* a, const float* b, float* y);
SYCL_EXTERNAL int minmag_s_ha_gen(const float* a, const float* b, float* y);
SYCL_EXTERNAL int minmag_d_ep_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int minmag_d_la_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int minmag_d_ha_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int modf_s_ep_gen(const float* a, float* y, float* z);
SYCL_EXTERNAL int modf_s_la_gen(const float* a, float* y, float* z);
SYCL_EXTERNAL int modf_s_ha_gen(const float* a, float* y, float* z);
SYCL_EXTERNAL int modf_d_ep_gen(const double* a, double* y, double* z);
SYCL_EXTERNAL int modf_d_la_gen(const double* a, double* y, double* z);
SYCL_EXTERNAL int modf_d_ha_gen(const double* a, double* y, double* z);
SYCL_EXTERNAL int mul_s_ep_gen(const float* a, const float* b, float* y);
SYCL_EXTERNAL int mul_s_la_gen(const float* a, const float* b, float* y);
SYCL_EXTERNAL int mul_s_ha_gen(const float* a, const float* b, float* y);
SYCL_EXTERNAL int mul_d_ep_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int mul_d_la_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int mul_d_ha_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int mul_c_ep_gen(const std::complex<float>* a, const std::complex<float>* b, std::complex<float>* y);
SYCL_EXTERNAL int mul_c_la_gen(const std::complex<float>* a, const std::complex<float>* b, std::complex<float>* y);
SYCL_EXTERNAL int mul_c_ha_nolut(const std::complex<float>* a, const std::complex<float>* b, std::complex<float>* y);
SYCL_EXTERNAL int mul_z_ep_gen(const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* y);
SYCL_EXTERNAL int mul_z_la_gen(const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* y);
SYCL_EXTERNAL int mul_z_ha_nolut(const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* y);
SYCL_EXTERNAL int mulbyconj_c_ep_gen(const std::complex<float>* a, const std::complex<float>* b, std::complex<float>* y);
SYCL_EXTERNAL int mulbyconj_c_la_gen(const std::complex<float>* a, const std::complex<float>* b, std::complex<float>* y);
SYCL_EXTERNAL int mulbyconj_c_ha_gen(const std::complex<float>* a, const std::complex<float>* b, std::complex<float>* y);
SYCL_EXTERNAL int mulbyconj_z_ep_gen(const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* y);
SYCL_EXTERNAL int mulbyconj_z_la_gen(const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* y);
SYCL_EXTERNAL int mulbyconj_z_ha_gen(const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* y);
SYCL_EXTERNAL int nearbyint_s_ep_gen(const float* a, float* y);
SYCL_EXTERNAL int nearbyint_s_la_gen(const float* a, float* y);
SYCL_EXTERNAL int nearbyint_s_ha_gen(const float* a, float* y);
SYCL_EXTERNAL int nearbyint_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int nearbyint_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int nearbyint_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int nextafter_s_ep_gen(const float* a, const float* b, float* y);
SYCL_EXTERNAL int nextafter_s_la_gen(const float* a, const float* b, float* y);
SYCL_EXTERNAL int nextafter_s_ha_gen(const float* a, const float* b, float* y);
SYCL_EXTERNAL int nextafter_d_ep_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int nextafter_d_la_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int nextafter_d_ha_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int pow_s_ep_gen(const float* a, const float* b, float* y);
SYCL_EXTERNAL int pow_s_la_nolut(const float* a, const float* b, float* y);
SYCL_EXTERNAL int pow_s_ha_nolut(const float* a, const float* b, float* y);
SYCL_EXTERNAL int pow_d_ep_nolut(const double* a, const double* b, double* y);
SYCL_EXTERNAL int pow_d_la_nolut(const double* a, const double* b, double* y);
SYCL_EXTERNAL int pow_d_ha_nolut(const double* a, const double* b, double* y);
SYCL_EXTERNAL int pow2o3_s_ep_gen(const float* a, float* y);
SYCL_EXTERNAL int pow2o3_s_la_gen(const float* a, float* y);
SYCL_EXTERNAL int pow2o3_s_ha_gen(const float* a, float* y);
SYCL_EXTERNAL int pow2o3_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int pow2o3_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int pow2o3_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int pow3o2_s_ep_gen(const float* a, float* y);
SYCL_EXTERNAL int pow3o2_s_la_gen(const float* a, float* y);
SYCL_EXTERNAL int pow3o2_s_ha_gen(const float* a, float* y);
SYCL_EXTERNAL int pow3o2_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int pow3o2_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int pow3o2_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int powr_s_ep_gen(const float* a, const float* b, float* y);
SYCL_EXTERNAL int powr_s_la_nolut(const float* a, const float* b, float* y);
SYCL_EXTERNAL int powr_s_ha_nolut(const float* a, const float* b, float* y);
SYCL_EXTERNAL int powr_d_ep_nolut(const double* a, const double* b, double* y);
SYCL_EXTERNAL int powr_d_la_nolut(const double* a, const double* b, double* y);
SYCL_EXTERNAL int powr_d_ha_nolut(const double* a, const double* b, double* y);
SYCL_EXTERNAL int powx_s_ep_nolut(const float* a, float b, float* y);
SYCL_EXTERNAL int powx_s_la_nolut(const float* a, float b, float* y);
SYCL_EXTERNAL int powx_s_ha_nolut(const float* a, float b, float* y);
SYCL_EXTERNAL int powx_d_ep_nolut(const double* a, double b, double* y);
SYCL_EXTERNAL int powx_d_la_nolut(const double* a, double b, double* y);
SYCL_EXTERNAL int powx_d_ha_nolut(const double* a, double b, double* y);
SYCL_EXTERNAL int remainder_s_ep_gen(const float* a, const float* b, float* y);
SYCL_EXTERNAL int remainder_s_la_gen(const float* a, const float* b, float* y);
SYCL_EXTERNAL int remainder_s_ha_gen(const float* a, const float* b, float* y);
SYCL_EXTERNAL int remainder_d_ep_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int remainder_d_la_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int remainder_d_ha_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int rint_s_ep_gen(const float* a, float* y);
SYCL_EXTERNAL int rint_s_la_gen(const float* a, float* y);
SYCL_EXTERNAL int rint_s_ha_gen(const float* a, float* y);
SYCL_EXTERNAL int rint_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int rint_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int rint_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int round_s_ep_gen(const float* a, float* y);
SYCL_EXTERNAL int round_s_la_gen(const float* a, float* y);
SYCL_EXTERNAL int round_s_ha_gen(const float* a, float* y);
SYCL_EXTERNAL int round_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int round_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int round_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int sin_s_ep_gen(const float* a, float* y);
SYCL_EXTERNAL int sin_s_la_gen(const float* a, float* y);
SYCL_EXTERNAL int sin_s_ha_fp64only(const float* a, float* y);
SYCL_EXTERNAL int sin_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int sin_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int sin_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int sincos_s_ep_gen(const float* a, float* y, float* z);
SYCL_EXTERNAL int sincos_s_la_gen(const float* a, float* y, float* z);
SYCL_EXTERNAL int sincos_s_ha_fp64only(const float* a, float* y, float* z);
SYCL_EXTERNAL int sincos_d_ep_gen(const double* a, double* y, double* z);
SYCL_EXTERNAL int sincos_d_la_gen(const double* a, double* y, double* z);
SYCL_EXTERNAL int sincos_d_ha_gen(const double* a, double* y, double* z);
SYCL_EXTERNAL int sincospi_s_ep_gen(const float* a, float* y, float* z);
SYCL_EXTERNAL int sincospi_s_la_gen(const float* a, float* y, float* z);
SYCL_EXTERNAL int sincospi_s_ha_gen(const float* a, float* y, float* z);
SYCL_EXTERNAL int sincospi_d_ep_gen(const double* a, double* y, double* z);
SYCL_EXTERNAL int sincospi_d_la_gen(const double* a, double* y, double* z);
SYCL_EXTERNAL int sincospi_d_ha_gen(const double* a, double* y, double* z);
SYCL_EXTERNAL int sind_s_ep_fp64only(const float* a, float* y);
SYCL_EXTERNAL int sind_s_la_fp64only(const float* a, float* y);
SYCL_EXTERNAL int sind_s_ha_fp64only(const float* a, float* y);
SYCL_EXTERNAL int sind_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int sind_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int sind_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int sinh_s_ep_gen(const float* a, float* y);
SYCL_EXTERNAL int sinh_s_la_gen(const float* a, float* y);
SYCL_EXTERNAL int sinh_s_ha_fp64only(const float* a, float* y);
SYCL_EXTERNAL int sinh_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int sinh_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int sinh_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int sinpi_s_ep_gen(const float* a, float* y);
SYCL_EXTERNAL int sinpi_s_la_gen(const float* a, float* y);
SYCL_EXTERNAL int sinpi_s_ha_fp64only(const float* a, float* y);
SYCL_EXTERNAL int sinpi_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int sinpi_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int sinpi_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int sqr_s_ep_gen(const float* a, float* y);
SYCL_EXTERNAL int sqr_s_la_gen(const float* a, float* y);
SYCL_EXTERNAL int sqr_s_ha_gen(const float* a, float* y);
SYCL_EXTERNAL int sqr_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int sqr_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int sqr_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int sqrt_s_ep_gen(const float* a, float* y);
SYCL_EXTERNAL int sqrt_s_la_gen(const float* a, float* y);
SYCL_EXTERNAL int sqrt_s_ha_gen(const float* a, float* y);
SYCL_EXTERNAL int sqrt_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int sqrt_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int sqrt_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int sqrt_c_ep_gen(const std::complex<float>* a, std::complex<float>* y);
SYCL_EXTERNAL int sqrt_c_la_gen(const std::complex<float>* a, std::complex<float>* y);
SYCL_EXTERNAL int sqrt_c_ha_gen(const std::complex<float>* a, std::complex<float>* y);
SYCL_EXTERNAL int sqrt_z_ep_gen(const std::complex<double>* a, std::complex<double>* y);
SYCL_EXTERNAL int sqrt_z_la_gen(const std::complex<double>* a, std::complex<double>* y);
SYCL_EXTERNAL int sqrt_z_ha_gen(const std::complex<double>* a, std::complex<double>* y);
SYCL_EXTERNAL int sub_s_ep_gen(const float* a, const float* b, float* y);
SYCL_EXTERNAL int sub_s_la_gen(const float* a, const float* b, float* y);
SYCL_EXTERNAL int sub_s_ha_gen(const float* a, const float* b, float* y);
SYCL_EXTERNAL int sub_d_ep_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int sub_d_la_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int sub_d_ha_gen(const double* a, const double* b, double* y);
SYCL_EXTERNAL int sub_c_ep_gen(const std::complex<float>* a, const std::complex<float>* b, std::complex<float>* y);
SYCL_EXTERNAL int sub_c_la_gen(const std::complex<float>* a, const std::complex<float>* b, std::complex<float>* y);
SYCL_EXTERNAL int sub_c_ha_gen(const std::complex<float>* a, const std::complex<float>* b, std::complex<float>* y);
SYCL_EXTERNAL int sub_z_ep_gen(const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* y);
SYCL_EXTERNAL int sub_z_la_gen(const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* y);
SYCL_EXTERNAL int sub_z_ha_gen(const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* y);
SYCL_EXTERNAL int tan_s_ep_gen(const float* a, float* y);
SYCL_EXTERNAL int tan_s_la_gen(const float* a, float* y);
SYCL_EXTERNAL int tan_s_ha_fp64only(const float* a, float* y);
SYCL_EXTERNAL int tan_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int tan_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int tan_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int tand_s_ep_fp64only(const float* a, float* y);
SYCL_EXTERNAL int tand_s_la_fp64only(const float* a, float* y);
SYCL_EXTERNAL int tand_s_ha_fp64only(const float* a, float* y);
SYCL_EXTERNAL int tand_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int tand_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int tand_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int tanh_s_ep_gen(const float* a, float* y);
SYCL_EXTERNAL int tanh_s_la_gen(const float* a, float* y);
SYCL_EXTERNAL int tanh_s_ha_gen(const float* a, float* y);
SYCL_EXTERNAL int tanh_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int tanh_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int tanh_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int tanpi_s_ep_gen(const float* a, float* y);
SYCL_EXTERNAL int tanpi_s_la_gen(const float* a, float* y);
SYCL_EXTERNAL int tanpi_s_ha_gen(const float* a, float* y);
SYCL_EXTERNAL int tanpi_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int tanpi_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int tanpi_d_ha_gen(const double* a, double* y);
SYCL_EXTERNAL int tgamma_s_ep_fp64only(const float* a, float* y);
SYCL_EXTERNAL int tgamma_s_la_fp64only(const float* a, float* y);
SYCL_EXTERNAL int tgamma_s_ha_fp64only(const float* a, float* y);
SYCL_EXTERNAL int tgamma_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int tgamma_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int trunc_s_ep_gen(const float* a, float* y);
SYCL_EXTERNAL int trunc_s_la_gen(const float* a, float* y);
SYCL_EXTERNAL int trunc_s_ha_gen(const float* a, float* y);
SYCL_EXTERNAL int trunc_d_ep_gen(const double* a, double* y);
SYCL_EXTERNAL int trunc_d_la_gen(const double* a, double* y);
SYCL_EXTERNAL int trunc_d_ha_gen(const double* a, double* y);
}

namespace oneapi::mkl::vm::device::detail {
enum class Function : int {
Abs = 0,
Acos = 1,
Acosh = 2,
Acospi = 3,
Add = 4,
Arg = 5,
Asin = 6,
Asinh = 7,
Asinpi = 8,
Atan = 9,
Atan2 = 10,
Atan2pi = 11,
Atanh = 12,
Atanpi = 13,
Cbrt = 14,
Cdfnorm = 15,
Cdfnorminv = 16,
Ceil = 17,
Cis = 18,
Conj = 19,
Copysign = 20,
Cos = 21,
Cosd = 22,
Cosh = 23,
Cospi = 24,
Div = 25,
Erf = 26,
Erfc = 27,
Erfcx = 28,
Erfcinv = 29,
Erfinv = 30,
Exp = 31,
Exp10 = 32,
Exp2 = 33,
Expm1 = 34,
Fdim = 35,
Floor = 36,
Fmax = 37,
Fmin = 38,
Fmod = 39,
Frac = 40,
Hypot = 41,
Inv = 42,
Invcbrt = 43,
Invsqrt = 44,
Lgamma = 45,
Ln = 46,
Log10 = 47,
Log1p = 48,
Log2 = 49,
Logb = 50,
Maxmag = 51,
Minmag = 52,
Modf = 53,
Mul = 54,
Mulbyconj = 55,
Nearbyint = 56,
Nextafter = 57,
Pow = 58,
Pow2o3 = 59,
Pow3o2 = 60,
Powr = 61,
Powx = 62,
Remainder = 63,
Rint = 64,
Round = 65,
Sin = 66,
Sincos = 67,
Sincospi = 68,
Sind = 69,
Sinh = 70,
Sinpi = 71,
Sqr = 72,
Sqrt = 73,
Sub = 74,
Tan = 75,
Tand = 76,
Tanh = 77,
Tanpi = 78,
Tgamma = 79,
Trunc = 80,
};

template <Function func, typename Tin, typename Tout, Accuracy acc = Accuracy::NS, Feature fea = Feature::NS>
struct Evaluator {};

template <>
struct Evaluator<Function::Abs, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::abs_s_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Abs, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::abs_s_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Abs, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::abs_s_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Abs, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::abs_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Abs, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::abs_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Abs, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::abs_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Abs, std::complex<float>, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<float>* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::abs_c_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Abs, std::complex<float>, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<float>* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::abs_c_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Abs, std::complex<float>, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<float>* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::abs_c_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Abs, std::complex<double>, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<double>* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::abs_z_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Abs, std::complex<double>, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<double>* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::abs_z_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Abs, std::complex<double>, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<double>* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::abs_z_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Acos, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::acos_s_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Acos, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::acos_s_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Acos, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::acos_s_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Acos, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::acos_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Acos, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::acos_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Acos, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::acos_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Acosh, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::acosh_s_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Acosh, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::acosh_s_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Acosh, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::acosh_s_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Acosh, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::acosh_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Acosh, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::acosh_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Acosh, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::acosh_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Acospi, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::acospi_s_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Acospi, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::acospi_s_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Acospi, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::acospi_s_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Acospi, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::acospi_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Acospi, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::acospi_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Acospi, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::acospi_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Add, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::add_s_ep_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Add, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::add_s_la_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Add, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::add_s_ha_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Add, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::add_d_ep_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Add, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::add_d_la_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Add, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::add_d_ha_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Add, std::complex<float>, std::complex<float>, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<float>* a, const std::complex<float>* b, std::complex<float>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::add_c_ep_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Add, std::complex<float>, std::complex<float>, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<float>* a, const std::complex<float>* b, std::complex<float>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::add_c_la_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Add, std::complex<float>, std::complex<float>, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<float>* a, const std::complex<float>* b, std::complex<float>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::add_c_ha_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Add, std::complex<double>, std::complex<double>, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::add_z_ep_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Add, std::complex<double>, std::complex<double>, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::add_z_la_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Add, std::complex<double>, std::complex<double>, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::add_z_ha_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Arg, std::complex<float>, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<float>* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::arg_c_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Arg, std::complex<float>, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<float>* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::arg_c_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Arg, std::complex<float>, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<float>* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::arg_c_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Arg, std::complex<double>, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<double>* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::arg_z_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Arg, std::complex<double>, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<double>* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::arg_z_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Arg, std::complex<double>, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<double>* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::arg_z_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Asin, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::asin_s_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Asin, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::asin_s_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Asin, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::asin_s_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Asin, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::asin_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Asin, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::asin_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Asin, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::asin_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Asinh, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::asinh_s_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Asinh, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::asinh_s_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Asinh, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::asinh_s_ha_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Asinh, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::asinh_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Asinh, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::asinh_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Asinh, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::asinh_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Asinpi, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::asinpi_s_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Asinpi, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::asinpi_s_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Asinpi, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::asinpi_s_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Asinpi, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::asinpi_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Asinpi, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::asinpi_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Asinpi, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::asinpi_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Atan, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::atan_s_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Atan, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::atan_s_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Atan, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::atan_s_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Atan, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::atan_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Atan, double, double, Accuracy::LA, Feature::TA> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::atan_d_la_nolut(a, y); }
};

template <>
struct Evaluator<Function::Atan, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::atan_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Atan2, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::atan2_s_ep_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Atan2, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::atan2_s_la_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Atan2, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::atan2_s_ha_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Atan2, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::atan2_d_ep_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Atan2, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::atan2_d_la_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Atan2, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::atan2_d_ha_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Atan2pi, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::atan2pi_s_ep_fp64only(a, b, y); }
};

template <>
struct Evaluator<Function::Atan2pi, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::atan2pi_s_la_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Atan2pi, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::atan2pi_s_ha_fp64only(a, b, y); }
};

template <>
struct Evaluator<Function::Atan2pi, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::atan2pi_d_ep_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Atan2pi, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::atan2pi_d_la_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Atan2pi, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::atan2pi_d_ha_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Atanh, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::atanh_s_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Atanh, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::atanh_s_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Atanh, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::atanh_s_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Atanh, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::atanh_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Atanh, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::atanh_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Atanh, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::atanh_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Atanpi, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::atanpi_s_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Atanpi, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::atanpi_s_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Atanpi, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::atanpi_s_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Atanpi, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::atanpi_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Atanpi, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::atanpi_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Atanpi, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::atanpi_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Cbrt, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cbrt_s_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Cbrt, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cbrt_s_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Cbrt, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cbrt_s_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Cbrt, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cbrt_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Cbrt, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cbrt_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Cbrt, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cbrt_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Cdfnorm, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cdfnorm_s_ep_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Cdfnorm, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cdfnorm_s_la_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Cdfnorm, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cdfnorm_s_ha_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Cdfnorm, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cdfnorm_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Cdfnorm, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cdfnorm_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Cdfnorm, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cdfnorm_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Cdfnorminv, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cdfnorminv_s_ep_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Cdfnorminv, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cdfnorminv_s_la_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Cdfnorminv, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cdfnorminv_s_ha_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Cdfnorminv, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cdfnorminv_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Cdfnorminv, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cdfnorminv_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Cdfnorminv, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cdfnorminv_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Ceil, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::ceil_s_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Ceil, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::ceil_s_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Ceil, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::ceil_s_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Ceil, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::ceil_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Ceil, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::ceil_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Ceil, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::ceil_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Cis, float, std::complex<float>, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, std::complex<float>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cis_c_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Cis, float, std::complex<float>, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, std::complex<float>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cis_c_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Cis, float, std::complex<float>, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, std::complex<float>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cis_c_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Cis, double, std::complex<double>, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, std::complex<double>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cis_z_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Cis, double, std::complex<double>, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, std::complex<double>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cis_z_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Cis, double, std::complex<double>, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, std::complex<double>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cis_z_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Conj, std::complex<float>, std::complex<float>, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<float>* a, std::complex<float>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::conj_c_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Conj, std::complex<float>, std::complex<float>, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<float>* a, std::complex<float>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::conj_c_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Conj, std::complex<float>, std::complex<float>, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<float>* a, std::complex<float>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::conj_c_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Conj, std::complex<double>, std::complex<double>, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<double>* a, std::complex<double>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::conj_z_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Conj, std::complex<double>, std::complex<double>, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<double>* a, std::complex<double>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::conj_z_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Conj, std::complex<double>, std::complex<double>, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<double>* a, std::complex<double>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::conj_z_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Copysign, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::copysign_s_ep_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Copysign, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::copysign_s_la_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Copysign, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::copysign_s_ha_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Copysign, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::copysign_d_ep_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Copysign, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::copysign_d_la_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Copysign, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::copysign_d_ha_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Cos, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cos_s_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Cos, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cos_s_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Cos, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cos_s_ha_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Cos, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cos_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Cos, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cos_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Cos, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cos_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Cosd, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cosd_s_ep_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Cosd, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cosd_s_la_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Cosd, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cosd_s_ha_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Cosd, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cosd_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Cosd, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cosd_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Cosd, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cosd_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Cosh, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cosh_s_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Cosh, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cosh_s_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Cosh, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cosh_s_ha_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Cosh, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cosh_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Cosh, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cosh_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Cosh, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cosh_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Cospi, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cospi_s_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Cospi, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cospi_s_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Cospi, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cospi_s_ha_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Cospi, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cospi_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Cospi, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cospi_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Cospi, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::cospi_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Div, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::div_s_ep_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Div, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::div_s_la_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Div, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::div_s_ha_fp64only(a, b, y); }
};

template <>
struct Evaluator<Function::Div, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::div_d_ep_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Div, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::div_d_la_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Div, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::div_d_ha_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Div, std::complex<float>, std::complex<float>, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<float>* a, const std::complex<float>* b, std::complex<float>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::div_c_ep_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Div, std::complex<float>, std::complex<float>, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<float>* a, const std::complex<float>* b, std::complex<float>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::div_c_la_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Div, std::complex<float>, std::complex<float>, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<float>* a, const std::complex<float>* b, std::complex<float>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::div_c_ha_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Div, std::complex<double>, std::complex<double>, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::div_z_ep_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Div, std::complex<double>, std::complex<double>, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::div_z_la_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Div, std::complex<double>, std::complex<double>, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::div_z_ha_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Erf, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::erf_s_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Erf, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::erf_s_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Erf, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::erf_s_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Erf, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::erf_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Erf, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::erf_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Erf, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::erf_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Erfc, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::erfc_s_ep_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Erfc, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::erfc_s_la_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Erfc, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::erfc_s_ha_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Erfc, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::erfc_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Erfc, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::erfc_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Erfc, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::erfc_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Erfcx, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::erfcx_s_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Erfcx, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::erfcx_s_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Erfcx, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::erfcx_s_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Erfcx, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::erfcx_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Erfcx, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::erfcx_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Erfcx, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::erfcx_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Erfcinv, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::erfcinv_s_ep_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Erfcinv, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::erfcinv_s_la_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Erfcinv, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::erfcinv_s_ha_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Erfcinv, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::erfcinv_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Erfcinv, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::erfcinv_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Erfcinv, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::erfcinv_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Erfinv, float, float, Accuracy::EP, Feature::TA> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::erfinv_s_ep_nolut(a, y); }
};

template <>
struct Evaluator<Function::Erfinv, float, float, Accuracy::LA, Feature::TA> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::erfinv_s_la_nolut(a, y); }
};

template <>
struct Evaluator<Function::Erfinv, float, float, Accuracy::HA, Feature::TA> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::erfinv_s_ha_nolut(a, y); }
};

template <>
struct Evaluator<Function::Erfinv, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::erfinv_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Erfinv, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::erfinv_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Erfinv, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::erfinv_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Exp, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::exp_s_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Exp, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::exp_s_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Exp, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::exp_s_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Exp, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::exp_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Exp, double, double, Accuracy::LA, Feature::TA> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::exp_d_la_nolut(a, y); }
};

template <>
struct Evaluator<Function::Exp, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::exp_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Exp, std::complex<float>, std::complex<float>, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<float>* a, std::complex<float>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::exp_c_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Exp, std::complex<float>, std::complex<float>, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<float>* a, std::complex<float>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::exp_c_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Exp, std::complex<float>, std::complex<float>, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<float>* a, std::complex<float>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::exp_c_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Exp, std::complex<double>, std::complex<double>, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<double>* a, std::complex<double>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::exp_z_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Exp, std::complex<double>, std::complex<double>, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<double>* a, std::complex<double>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::exp_z_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Exp, std::complex<double>, std::complex<double>, Accuracy::HA, Feature::TA> {
    void operator()() { }
    int  operator()(const std::complex<double>* a, std::complex<double>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::exp_z_ha_nolut(a, y); }
};

template <>
struct Evaluator<Function::Exp10, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::exp10_s_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Exp10, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::exp10_s_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Exp10, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::exp10_s_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Exp10, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::exp10_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Exp10, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::exp10_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Exp10, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::exp10_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Exp2, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::exp2_s_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Exp2, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::exp2_s_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Exp2, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::exp2_s_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Exp2, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::exp2_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Exp2, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::exp2_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Exp2, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::exp2_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Expm1, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::expm1_s_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Expm1, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::expm1_s_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Expm1, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::expm1_s_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Expm1, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::expm1_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Expm1, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::expm1_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Expm1, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::expm1_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Fdim, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::fdim_s_ep_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Fdim, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::fdim_s_la_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Fdim, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::fdim_s_ha_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Fdim, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::fdim_d_ep_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Fdim, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::fdim_d_la_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Fdim, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::fdim_d_ha_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Floor, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::floor_s_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Floor, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::floor_s_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Floor, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::floor_s_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Floor, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::floor_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Floor, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::floor_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Floor, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::floor_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Fmax, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::fmax_s_ep_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Fmax, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::fmax_s_la_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Fmax, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::fmax_s_ha_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Fmax, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::fmax_d_ep_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Fmax, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::fmax_d_la_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Fmax, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::fmax_d_ha_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Fmin, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::fmin_s_ep_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Fmin, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::fmin_s_la_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Fmin, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::fmin_s_ha_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Fmin, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::fmin_d_ep_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Fmin, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::fmin_d_la_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Fmin, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::fmin_d_ha_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Fmod, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::fmod_s_ep_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Fmod, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::fmod_s_la_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Fmod, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::fmod_s_ha_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Fmod, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::fmod_d_ep_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Fmod, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::fmod_d_la_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Fmod, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::fmod_d_ha_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Frac, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::frac_s_ep_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Frac, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::frac_s_la_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Frac, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::frac_s_ha_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Frac, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::frac_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Frac, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::frac_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Frac, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::frac_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Hypot, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::hypot_s_ep_fp64only(a, b, y); }
};

template <>
struct Evaluator<Function::Hypot, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::hypot_s_la_fp64only(a, b, y); }
};

template <>
struct Evaluator<Function::Hypot, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::hypot_s_ha_fp64only(a, b, y); }
};

template <>
struct Evaluator<Function::Hypot, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::hypot_d_ep_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Hypot, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::hypot_d_la_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Hypot, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::hypot_d_ha_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Inv, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::inv_s_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Inv, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::inv_s_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Inv, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::inv_s_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Inv, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::inv_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Inv, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::inv_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Inv, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::inv_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Invcbrt, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::invcbrt_s_ep_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Invcbrt, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::invcbrt_s_la_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Invcbrt, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::invcbrt_s_ha_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Invcbrt, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::invcbrt_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Invcbrt, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::invcbrt_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Invcbrt, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::invcbrt_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Invsqrt, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::invsqrt_s_ep_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Invsqrt, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::invsqrt_s_la_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Invsqrt, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::invsqrt_s_ha_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Invsqrt, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::invsqrt_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Invsqrt, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::invsqrt_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Invsqrt, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::invsqrt_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Lgamma, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::lgamma_s_ep_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Lgamma, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::lgamma_s_la_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Lgamma, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::lgamma_s_ha_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Ln, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::ln_s_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Ln, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::ln_s_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Ln, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::ln_s_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Ln, double, double, Accuracy::EP, Feature::TA> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::ln_d_ep_nolut(a, y); }
};

template <>
struct Evaluator<Function::Ln, double, double, Accuracy::LA, Feature::TA> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::ln_d_la_nolut(a, y); }
};

template <>
struct Evaluator<Function::Ln, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::ln_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Ln, std::complex<float>, std::complex<float>, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<float>* a, std::complex<float>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::ln_c_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Ln, std::complex<float>, std::complex<float>, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<float>* a, std::complex<float>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::ln_c_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Ln, std::complex<float>, std::complex<float>, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<float>* a, std::complex<float>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::ln_c_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Ln, std::complex<double>, std::complex<double>, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<double>* a, std::complex<double>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::ln_z_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Ln, std::complex<double>, std::complex<double>, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<double>* a, std::complex<double>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::ln_z_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Ln, std::complex<double>, std::complex<double>, Accuracy::HA, Feature::TA> {
    void operator()() { }
    int  operator()(const std::complex<double>* a, std::complex<double>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::ln_z_ha_nolut(a, y); }
};

template <>
struct Evaluator<Function::Log10, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::log10_s_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Log10, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::log10_s_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Log10, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::log10_s_ha_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Log10, double, double, Accuracy::EP, Feature::TA> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::log10_d_ep_nolut(a, y); }
};

template <>
struct Evaluator<Function::Log10, double, double, Accuracy::LA, Feature::TA> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::log10_d_la_nolut(a, y); }
};

template <>
struct Evaluator<Function::Log10, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::log10_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Log1p, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::log1p_s_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Log1p, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::log1p_s_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Log1p, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::log1p_s_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Log1p, double, double, Accuracy::EP, Feature::TA> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::log1p_d_ep_nolut(a, y); }
};

template <>
struct Evaluator<Function::Log1p, double, double, Accuracy::LA, Feature::TA> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::log1p_d_la_nolut(a, y); }
};

template <>
struct Evaluator<Function::Log1p, double, double, Accuracy::HA, Feature::TA> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::log1p_d_ha_nolut(a, y); }
};

template <>
struct Evaluator<Function::Log2, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::log2_s_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Log2, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::log2_s_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Log2, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::log2_s_ha_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Log2, double, double, Accuracy::EP, Feature::TA> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::log2_d_ep_nolut(a, y); }
};

template <>
struct Evaluator<Function::Log2, double, double, Accuracy::LA, Feature::TA> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::log2_d_la_nolut(a, y); }
};

template <>
struct Evaluator<Function::Log2, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::log2_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Logb, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::logb_s_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Logb, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::logb_s_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Logb, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::logb_s_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Logb, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::logb_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Logb, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::logb_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Logb, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::logb_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Maxmag, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::maxmag_s_ep_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Maxmag, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::maxmag_s_la_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Maxmag, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::maxmag_s_ha_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Maxmag, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::maxmag_d_ep_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Maxmag, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::maxmag_d_la_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Maxmag, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::maxmag_d_ha_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Minmag, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::minmag_s_ep_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Minmag, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::minmag_s_la_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Minmag, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::minmag_s_ha_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Minmag, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::minmag_d_ep_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Minmag, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::minmag_d_la_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Minmag, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::minmag_d_ha_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Modf, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y, float* z) { return oneapi::mkl::vm::detail::gpu::intel::scalar::modf_s_ep_gen(a, y, z); }
};

template <>
struct Evaluator<Function::Modf, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y, float* z) { return oneapi::mkl::vm::detail::gpu::intel::scalar::modf_s_la_gen(a, y, z); }
};

template <>
struct Evaluator<Function::Modf, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y, float* z) { return oneapi::mkl::vm::detail::gpu::intel::scalar::modf_s_ha_gen(a, y, z); }
};

template <>
struct Evaluator<Function::Modf, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y, double* z) { return oneapi::mkl::vm::detail::gpu::intel::scalar::modf_d_ep_gen(a, y, z); }
};

template <>
struct Evaluator<Function::Modf, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y, double* z) { return oneapi::mkl::vm::detail::gpu::intel::scalar::modf_d_la_gen(a, y, z); }
};

template <>
struct Evaluator<Function::Modf, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y, double* z) { return oneapi::mkl::vm::detail::gpu::intel::scalar::modf_d_ha_gen(a, y, z); }
};

template <>
struct Evaluator<Function::Mul, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::mul_s_ep_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Mul, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::mul_s_la_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Mul, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::mul_s_ha_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Mul, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::mul_d_ep_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Mul, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::mul_d_la_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Mul, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::mul_d_ha_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Mul, std::complex<float>, std::complex<float>, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<float>* a, const std::complex<float>* b, std::complex<float>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::mul_c_ep_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Mul, std::complex<float>, std::complex<float>, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<float>* a, const std::complex<float>* b, std::complex<float>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::mul_c_la_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Mul, std::complex<float>, std::complex<float>, Accuracy::HA, Feature::TA> {
    void operator()() { }
    int  operator()(const std::complex<float>* a, const std::complex<float>* b, std::complex<float>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::mul_c_ha_nolut(a, b, y); }
};

template <>
struct Evaluator<Function::Mul, std::complex<double>, std::complex<double>, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::mul_z_ep_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Mul, std::complex<double>, std::complex<double>, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::mul_z_la_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Mul, std::complex<double>, std::complex<double>, Accuracy::HA, Feature::TA> {
    void operator()() { }
    int  operator()(const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::mul_z_ha_nolut(a, b, y); }
};

template <>
struct Evaluator<Function::Mulbyconj, std::complex<float>, std::complex<float>, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<float>* a, const std::complex<float>* b, std::complex<float>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::mulbyconj_c_ep_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Mulbyconj, std::complex<float>, std::complex<float>, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<float>* a, const std::complex<float>* b, std::complex<float>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::mulbyconj_c_la_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Mulbyconj, std::complex<float>, std::complex<float>, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<float>* a, const std::complex<float>* b, std::complex<float>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::mulbyconj_c_ha_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Mulbyconj, std::complex<double>, std::complex<double>, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::mulbyconj_z_ep_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Mulbyconj, std::complex<double>, std::complex<double>, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::mulbyconj_z_la_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Mulbyconj, std::complex<double>, std::complex<double>, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::mulbyconj_z_ha_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Nearbyint, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::nearbyint_s_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Nearbyint, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::nearbyint_s_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Nearbyint, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::nearbyint_s_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Nearbyint, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::nearbyint_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Nearbyint, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::nearbyint_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Nearbyint, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::nearbyint_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Nextafter, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::nextafter_s_ep_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Nextafter, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::nextafter_s_la_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Nextafter, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::nextafter_s_ha_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Nextafter, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::nextafter_d_ep_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Nextafter, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::nextafter_d_la_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Nextafter, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::nextafter_d_ha_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Pow, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::pow_s_ep_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Pow, float, float, Accuracy::LA, Feature::TA> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::pow_s_la_nolut(a, b, y); }
};

template <>
struct Evaluator<Function::Pow, float, float, Accuracy::HA, Feature::TA> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::pow_s_ha_nolut(a, b, y); }
};

template <>
struct Evaluator<Function::Pow, double, double, Accuracy::EP, Feature::TA> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::pow_d_ep_nolut(a, b, y); }
};

template <>
struct Evaluator<Function::Pow, double, double, Accuracy::LA, Feature::TA> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::pow_d_la_nolut(a, b, y); }
};

template <>
struct Evaluator<Function::Pow, double, double, Accuracy::HA, Feature::TA> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::pow_d_ha_nolut(a, b, y); }
};

template <>
struct Evaluator<Function::Pow2o3, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::pow2o3_s_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Pow2o3, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::pow2o3_s_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Pow2o3, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::pow2o3_s_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Pow2o3, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::pow2o3_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Pow2o3, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::pow2o3_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Pow2o3, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::pow2o3_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Pow3o2, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::pow3o2_s_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Pow3o2, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::pow3o2_s_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Pow3o2, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::pow3o2_s_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Pow3o2, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::pow3o2_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Pow3o2, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::pow3o2_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Pow3o2, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::pow3o2_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Powr, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::powr_s_ep_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Powr, float, float, Accuracy::LA, Feature::TA> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::powr_s_la_nolut(a, b, y); }
};

template <>
struct Evaluator<Function::Powr, float, float, Accuracy::HA, Feature::TA> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::powr_s_ha_nolut(a, b, y); }
};

template <>
struct Evaluator<Function::Powr, double, double, Accuracy::EP, Feature::TA> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::powr_d_ep_nolut(a, b, y); }
};

template <>
struct Evaluator<Function::Powr, double, double, Accuracy::LA, Feature::TA> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::powr_d_la_nolut(a, b, y); }
};

template <>
struct Evaluator<Function::Powr, double, double, Accuracy::HA, Feature::TA> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::powr_d_ha_nolut(a, b, y); }
};

template <>
struct Evaluator<Function::Powx, float, float, Accuracy::EP, Feature::TA> {
    void operator()() { }
    int  operator()(const float* a, float b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::powx_s_ep_nolut(a, b, y); }
};

template <>
struct Evaluator<Function::Powx, float, float, Accuracy::LA, Feature::TA> {
    void operator()() { }
    int  operator()(const float* a, float b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::powx_s_la_nolut(a, b, y); }
};

template <>
struct Evaluator<Function::Powx, float, float, Accuracy::HA, Feature::TA> {
    void operator()() { }
    int  operator()(const float* a, float b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::powx_s_ha_nolut(a, b, y); }
};

template <>
struct Evaluator<Function::Powx, double, double, Accuracy::EP, Feature::TA> {
    void operator()() { }
    int  operator()(const double* a, double b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::powx_d_ep_nolut(a, b, y); }
};

template <>
struct Evaluator<Function::Powx, double, double, Accuracy::LA, Feature::TA> {
    void operator()() { }
    int  operator()(const double* a, double b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::powx_d_la_nolut(a, b, y); }
};

template <>
struct Evaluator<Function::Powx, double, double, Accuracy::HA, Feature::TA> {
    void operator()() { }
    int  operator()(const double* a, double b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::powx_d_ha_nolut(a, b, y); }
};

template <>
struct Evaluator<Function::Remainder, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::remainder_s_ep_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Remainder, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::remainder_s_la_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Remainder, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::remainder_s_ha_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Remainder, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::remainder_d_ep_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Remainder, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::remainder_d_la_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Remainder, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::remainder_d_ha_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Rint, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::rint_s_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Rint, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::rint_s_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Rint, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::rint_s_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Rint, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::rint_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Rint, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::rint_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Rint, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::rint_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Round, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::round_s_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Round, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::round_s_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Round, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::round_s_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Round, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::round_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Round, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::round_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Round, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::round_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Sin, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sin_s_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Sin, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sin_s_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Sin, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sin_s_ha_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Sin, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sin_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Sin, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sin_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Sin, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sin_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Sincos, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y, float* z) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sincos_s_ep_gen(a, y, z); }
};

template <>
struct Evaluator<Function::Sincos, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y, float* z) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sincos_s_la_gen(a, y, z); }
};

template <>
struct Evaluator<Function::Sincos, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y, float* z) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sincos_s_ha_fp64only(a, y, z); }
};

template <>
struct Evaluator<Function::Sincos, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y, double* z) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sincos_d_ep_gen(a, y, z); }
};

template <>
struct Evaluator<Function::Sincos, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y, double* z) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sincos_d_la_gen(a, y, z); }
};

template <>
struct Evaluator<Function::Sincos, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y, double* z) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sincos_d_ha_gen(a, y, z); }
};

template <>
struct Evaluator<Function::Sincospi, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y, float* z) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sincospi_s_ep_gen(a, y, z); }
};

template <>
struct Evaluator<Function::Sincospi, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y, float* z) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sincospi_s_la_gen(a, y, z); }
};

template <>
struct Evaluator<Function::Sincospi, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y, float* z) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sincospi_s_ha_gen(a, y, z); }
};

template <>
struct Evaluator<Function::Sincospi, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y, double* z) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sincospi_d_ep_gen(a, y, z); }
};

template <>
struct Evaluator<Function::Sincospi, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y, double* z) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sincospi_d_la_gen(a, y, z); }
};

template <>
struct Evaluator<Function::Sincospi, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y, double* z) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sincospi_d_ha_gen(a, y, z); }
};

template <>
struct Evaluator<Function::Sind, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sind_s_ep_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Sind, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sind_s_la_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Sind, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sind_s_ha_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Sind, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sind_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Sind, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sind_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Sind, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sind_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Sinh, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sinh_s_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Sinh, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sinh_s_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Sinh, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sinh_s_ha_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Sinh, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sinh_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Sinh, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sinh_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Sinh, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sinh_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Sinpi, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sinpi_s_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Sinpi, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sinpi_s_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Sinpi, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sinpi_s_ha_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Sinpi, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sinpi_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Sinpi, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sinpi_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Sinpi, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sinpi_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Sqr, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sqr_s_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Sqr, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sqr_s_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Sqr, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sqr_s_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Sqr, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sqr_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Sqr, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sqr_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Sqr, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sqr_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Sqrt, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sqrt_s_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Sqrt, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sqrt_s_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Sqrt, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sqrt_s_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Sqrt, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sqrt_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Sqrt, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sqrt_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Sqrt, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sqrt_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Sqrt, std::complex<float>, std::complex<float>, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<float>* a, std::complex<float>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sqrt_c_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Sqrt, std::complex<float>, std::complex<float>, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<float>* a, std::complex<float>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sqrt_c_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Sqrt, std::complex<float>, std::complex<float>, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<float>* a, std::complex<float>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sqrt_c_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Sqrt, std::complex<double>, std::complex<double>, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<double>* a, std::complex<double>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sqrt_z_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Sqrt, std::complex<double>, std::complex<double>, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<double>* a, std::complex<double>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sqrt_z_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Sqrt, std::complex<double>, std::complex<double>, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<double>* a, std::complex<double>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sqrt_z_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Sub, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sub_s_ep_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Sub, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sub_s_la_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Sub, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, const float* b, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sub_s_ha_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Sub, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sub_d_ep_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Sub, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sub_d_la_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Sub, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, const double* b, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sub_d_ha_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Sub, std::complex<float>, std::complex<float>, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<float>* a, const std::complex<float>* b, std::complex<float>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sub_c_ep_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Sub, std::complex<float>, std::complex<float>, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<float>* a, const std::complex<float>* b, std::complex<float>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sub_c_la_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Sub, std::complex<float>, std::complex<float>, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<float>* a, const std::complex<float>* b, std::complex<float>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sub_c_ha_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Sub, std::complex<double>, std::complex<double>, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sub_z_ep_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Sub, std::complex<double>, std::complex<double>, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sub_z_la_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Sub, std::complex<double>, std::complex<double>, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::sub_z_ha_gen(a, b, y); }
};

template <>
struct Evaluator<Function::Tan, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::tan_s_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Tan, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::tan_s_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Tan, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::tan_s_ha_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Tan, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::tan_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Tan, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::tan_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Tan, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::tan_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Tand, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::tand_s_ep_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Tand, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::tand_s_la_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Tand, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::tand_s_ha_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Tand, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::tand_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Tand, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::tand_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Tand, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::tand_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Tanh, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::tanh_s_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Tanh, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::tanh_s_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Tanh, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::tanh_s_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Tanh, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::tanh_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Tanh, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::tanh_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Tanh, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::tanh_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Tanpi, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::tanpi_s_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Tanpi, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::tanpi_s_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Tanpi, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::tanpi_s_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Tanpi, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::tanpi_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Tanpi, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::tanpi_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Tanpi, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::tanpi_d_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Tgamma, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::tgamma_s_ep_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Tgamma, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::tgamma_s_la_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Tgamma, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::tgamma_s_ha_fp64only(a, y); }
};

template <>
struct Evaluator<Function::Tgamma, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::tgamma_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Tgamma, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::tgamma_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Trunc, float, float, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::trunc_s_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Trunc, float, float, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::trunc_s_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Trunc, float, float, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const float* a, float* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::trunc_s_ha_gen(a, y); }
};

template <>
struct Evaluator<Function::Trunc, double, double, Accuracy::EP, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::trunc_d_ep_gen(a, y); }
};

template <>
struct Evaluator<Function::Trunc, double, double, Accuracy::LA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::trunc_d_la_gen(a, y); }
};

template <>
struct Evaluator<Function::Trunc, double, double, Accuracy::HA, Feature::GE> {
    void operator()() { }
    int  operator()(const double* a, double* y) { return oneapi::mkl::vm::detail::gpu::intel::scalar::trunc_d_ha_gen(a, y); }
};

}

#endif

