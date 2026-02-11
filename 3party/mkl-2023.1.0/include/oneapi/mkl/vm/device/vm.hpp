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

#ifndef __ONEAPI_MKL_VM_DEVICE_VM_HPP__
#define __ONEAPI_MKL_VM_DEVICE_VM_HPP__ 1

#include "oneapi/mkl/vm/device/detail/scalar.hpp"
#include "oneapi/mkl/vm/device/detail/decls.hpp"

namespace oneapi::mkl::vm::device {
namespace detail {

template <Function f, typename Tin, typename Tout, Accuracy acc, Feature fea, typename = void>
constexpr bool Exists {false};

template <Function f, typename Tin, typename Tout, Accuracy acc, Feature fea>
constexpr bool Exists<f, Tin, Tout, acc, fea,
    std::void_t<
        decltype(std::declval<Evaluator<f, Tin, Tout, acc, fea>>().operator()())
    >
> = true;

using AccFeaT = std::pair<Accuracy, Feature>;

constexpr bool operator< (enum Accuracy lhs, enum Accuracy rhs) {
    if (Accuracy::High == lhs && Accuracy::CorrectlyRounded == rhs) { return true; }
    if (Accuracy::Lower == lhs && (Accuracy::High == rhs || Accuracy::CorrectlyRounded == rhs)) { return true; }
    if (Accuracy::EnhancedPerformance == lhs && (Accuracy::Lower == rhs || Accuracy::High == rhs || Accuracy::CorrectlyRounded == rhs)) { return true; }
    return false;
}

using AccFeaT = std::pair<Accuracy, Feature>;

template <Function Func, typename Tin, typename Tout, Accuracy Acc = Accuracy::NS, Feature Fea = Feature::NS>
struct CompileTimeSelector {
    static constexpr auto result = []() -> AccFeaT {
        if constexpr (Exists<Func, Tin, Tout, Acc, Fea>) { return AccFeaT{Acc, Fea}; }

        if constexpr (Acc >= Accuracy::CR && Exists<Func, Tin, Tout, Accuracy::CR, Feature::TA>) { return AccFeaT{Accuracy::CR, Feature::TA}; }
        if constexpr (Acc >= Accuracy::CR && Exists<Func, Tin, Tout, Accuracy::CR, Feature::GE>) { return AccFeaT{Accuracy::CR, Feature::GE}; }

        if constexpr (Acc >= Accuracy::HA && Exists<Func, Tin, Tout, Accuracy::HA, Feature::TA>) { return AccFeaT{Accuracy::HA, Feature::TA}; }
        if constexpr (Acc >= Accuracy::HA && Exists<Func, Tin, Tout, Accuracy::HA, Feature::GE>) { return AccFeaT{Accuracy::HA, Feature::GE}; }
        if constexpr (Acc >= Accuracy::HA && Exists<Func, Tin, Tout, Accuracy::CR, Feature::TA>) { return AccFeaT{Accuracy::CR, Feature::TA}; }
        if constexpr (Acc >= Accuracy::HA && Exists<Func, Tin, Tout, Accuracy::CR, Feature::GE>) { return AccFeaT{Accuracy::CR, Feature::GE}; }

        if constexpr (Acc >= Accuracy::LA && Exists<Func, Tin, Tout, Accuracy::LA, Feature::TA>) { return AccFeaT{Accuracy::LA, Feature::TA}; }
        if constexpr (Acc >= Accuracy::LA && Exists<Func, Tin, Tout, Accuracy::LA, Feature::GE>) { return AccFeaT{Accuracy::LA, Feature::GE}; }
        if constexpr (Acc >= Accuracy::LA && Exists<Func, Tin, Tout, Accuracy::HA, Feature::TA>) { return AccFeaT{Accuracy::HA, Feature::TA}; }
        if constexpr (Acc >= Accuracy::LA && Exists<Func, Tin, Tout, Accuracy::HA, Feature::GE>) { return AccFeaT{Accuracy::HA, Feature::GE}; }
        if constexpr (Acc >= Accuracy::LA && Exists<Func, Tin, Tout, Accuracy::CR, Feature::TA>) { return AccFeaT{Accuracy::CR, Feature::TA}; }
        if constexpr (Acc >= Accuracy::LA && Exists<Func, Tin, Tout, Accuracy::CR, Feature::GE>) { return AccFeaT{Accuracy::CR, Feature::GE}; }

        if constexpr (Acc >= Accuracy::EP && Exists<Func, Tin, Tout, Accuracy::EP, Feature::TA>) { return AccFeaT{Accuracy::EP, Feature::TA}; }
        if constexpr (Acc >= Accuracy::EP && Exists<Func, Tin, Tout, Accuracy::EP, Feature::GE>) { return AccFeaT{Accuracy::EP, Feature::GE}; }
        if constexpr (Acc >= Accuracy::EP && Exists<Func, Tin, Tout, Accuracy::LA, Feature::TA>) { return AccFeaT{Accuracy::LA, Feature::TA}; }
        if constexpr (Acc >= Accuracy::EP && Exists<Func, Tin, Tout, Accuracy::LA, Feature::GE>) { return AccFeaT{Accuracy::LA, Feature::GE}; }
        if constexpr (Acc >= Accuracy::EP && Exists<Func, Tin, Tout, Accuracy::HA, Feature::TA>) { return AccFeaT{Accuracy::HA, Feature::TA}; }
        if constexpr (Acc >= Accuracy::EP && Exists<Func, Tin, Tout, Accuracy::HA, Feature::GE>) { return AccFeaT{Accuracy::HA, Feature::GE}; }
        if constexpr (Acc >= Accuracy::EP && Exists<Func, Tin, Tout, Accuracy::CR, Feature::TA>) { return AccFeaT{Accuracy::CR, Feature::TA}; }
        if constexpr (Acc >= Accuracy::EP && Exists<Func, Tin, Tout, Accuracy::CR, Feature::GE>) { return AccFeaT{Accuracy::CR, Feature::GE}; }

        return AccFeaT{Accuracy::NA, Feature::NA};
    }();
};


template <Function Func, typename Tin, typename Tout, Accuracy Acc = Accuracy::NS, Feature Fea = Feature::NS>
constexpr bool ExistsAtAll = CompileTimeSelector<Func, Tin, Tout, Acc, Fea>::result.first != Accuracy::NA &&  CompileTimeSelector<Func, Tin, Tout, Acc, Fea>::result.second != Feature::NA;

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Abs, Tin, Tout, Acc::a>, int>::type
abs(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Abs, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Abs, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Acos, Tin, Tout, Acc::a>, int>::type
acos(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Acos, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Acos, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Acosh, Tin, Tout, Acc::a>, int>::type
acosh(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Acosh, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Acosh, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Acospi, Tin, Tout, Acc::a>, int>::type
acospi(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Acospi, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Acospi, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Add, Tin, Tout, Acc::a>, int>::type
add(const Tin* a, const Tin* b, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Add, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Add, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, b, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Arg, Tin, Tout, Acc::a>, int>::type
arg(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Arg, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Arg, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Asin, Tin, Tout, Acc::a>, int>::type
asin(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Asin, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Asin, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Asinh, Tin, Tout, Acc::a>, int>::type
asinh(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Asinh, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Asinh, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Asinpi, Tin, Tout, Acc::a>, int>::type
asinpi(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Asinpi, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Asinpi, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Atan, Tin, Tout, Acc::a>, int>::type
atan(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Atan, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Atan, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Atan2, Tin, Tout, Acc::a>, int>::type
atan2(const Tin* a, const Tin* b, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Atan2, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Atan2, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, b, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Atan2pi, Tin, Tout, Acc::a>, int>::type
atan2pi(const Tin* a, const Tin* b, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Atan2pi, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Atan2pi, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, b, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Atanh, Tin, Tout, Acc::a>, int>::type
atanh(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Atanh, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Atanh, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Atanpi, Tin, Tout, Acc::a>, int>::type
atanpi(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Atanpi, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Atanpi, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Cbrt, Tin, Tout, Acc::a>, int>::type
cbrt(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Cbrt, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Cbrt, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Cdfnorm, Tin, Tout, Acc::a>, int>::type
cdfnorm(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Cdfnorm, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Cdfnorm, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Cdfnorminv, Tin, Tout, Acc::a>, int>::type
cdfnorminv(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Cdfnorminv, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Cdfnorminv, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Ceil, Tin, Tout, Acc::a>, int>::type
ceil(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Ceil, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Ceil, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Cis, Tin, Tout, Acc::a>, int>::type
cis(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Cis, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Cis, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Conj, Tin, Tout, Acc::a>, int>::type
conj(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Conj, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Conj, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Copysign, Tin, Tout, Acc::a>, int>::type
copysign(const Tin* a, const Tin* b, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Copysign, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Copysign, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, b, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Cos, Tin, Tout, Acc::a>, int>::type
cos(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Cos, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Cos, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Cosd, Tin, Tout, Acc::a>, int>::type
cosd(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Cosd, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Cosd, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Cosh, Tin, Tout, Acc::a>, int>::type
cosh(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Cosh, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Cosh, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Cospi, Tin, Tout, Acc::a>, int>::type
cospi(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Cospi, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Cospi, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Div, Tin, Tout, Acc::a>, int>::type
div(const Tin* a, const Tin* b, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Div, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Div, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, b, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Erf, Tin, Tout, Acc::a>, int>::type
erf(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Erf, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Erf, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Erfc, Tin, Tout, Acc::a>, int>::type
erfc(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Erfc, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Erfc, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Erfc, Tin, Tout, Acc::a>, int>::type
erfcx(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Erfc, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Erfc, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Erfcinv, Tin, Tout, Acc::a>, int>::type
erfcinv(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Erfcinv, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Erfcinv, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Erfinv, Tin, Tout, Acc::a>, int>::type
erfinv(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Erfinv, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Erfinv, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Exp, Tin, Tout, Acc::a>, int>::type
exp(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Exp, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Exp, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Exp10, Tin, Tout, Acc::a>, int>::type
exp10(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Exp10, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Exp10, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Exp2, Tin, Tout, Acc::a>, int>::type
exp2(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Exp2, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Exp2, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Expm1, Tin, Tout, Acc::a>, int>::type
expm1(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Expm1, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Expm1, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Fdim, Tin, Tout, Acc::a>, int>::type
fdim(const Tin* a, const Tin* b, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Fdim, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Fdim, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, b, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Floor, Tin, Tout, Acc::a>, int>::type
floor(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Floor, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Floor, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Fmax, Tin, Tout, Acc::a>, int>::type
fmax(const Tin* a, const Tin* b, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Fmax, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Fmax, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, b, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Fmin, Tin, Tout, Acc::a>, int>::type
fmin(const Tin* a, const Tin* b, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Fmin, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Fmin, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, b, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Fmod, Tin, Tout, Acc::a>, int>::type
fmod(const Tin* a, const Tin* b, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Fmod, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Fmod, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, b, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Frac, Tin, Tout, Acc::a>, int>::type
frac(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Frac, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Frac, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Hypot, Tin, Tout, Acc::a>, int>::type
hypot(const Tin* a, const Tin* b, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Hypot, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Hypot, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, b, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Inv, Tin, Tout, Acc::a>, int>::type
inv(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Inv, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Inv, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Invcbrt, Tin, Tout, Acc::a>, int>::type
invcbrt(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Invcbrt, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Invcbrt, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Invsqrt, Tin, Tout, Acc::a>, int>::type
invsqrt(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Invsqrt, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Invsqrt, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Lgamma, Tin, Tout, Acc::a>, int>::type
lgamma(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Lgamma, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Lgamma, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Ln, Tin, Tout, Acc::a>, int>::type
ln(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Ln, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Ln, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Log10, Tin, Tout, Acc::a>, int>::type
log10(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Log10, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Log10, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Log1p, Tin, Tout, Acc::a>, int>::type
log1p(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Log1p, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Log1p, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Log2, Tin, Tout, Acc::a>, int>::type
log2(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Log2, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Log2, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Logb, Tin, Tout, Acc::a>, int>::type
logb(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Logb, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Logb, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Maxmag, Tin, Tout, Acc::a>, int>::type
maxmag(const Tin* a, const Tin* b, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Maxmag, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Maxmag, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, b, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Minmag, Tin, Tout, Acc::a>, int>::type
minmag(const Tin* a, const Tin* b, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Minmag, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Minmag, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, b, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Modf, Tin, Tout, Acc::a>, int>::type
modf(const Tin* a, Tout* y, Tout* z, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Modf, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Modf, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y, z);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Mul, Tin, Tout, Acc::a>, int>::type
mul(const Tin* a, const Tin* b, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Mul, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Mul, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, b, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Mulbyconj, Tin, Tout, Acc::a>, int>::type
mulbyconj(const Tin* a, const Tin* b, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Mulbyconj, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Mulbyconj, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, b, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Nearbyint, Tin, Tout, Acc::a>, int>::type
nearbyint(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Nearbyint, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Nearbyint, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Nextafter, Tin, Tout, Acc::a>, int>::type
nextafter(const Tin* a, const Tin* b, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Nextafter, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Nextafter, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, b, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Pow, Tin, Tout, Acc::a>, int>::type
pow(const Tin* a, const Tin* b, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Pow, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Pow, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, b, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Pow2o3, Tin, Tout, Acc::a>, int>::type
pow2o3(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Pow2o3, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Pow2o3, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Pow3o2, Tin, Tout, Acc::a>, int>::type
pow3o2(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Pow3o2, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Pow3o2, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Powr, Tin, Tout, Acc::a>, int>::type
powr(const Tin* a, const Tin* b, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Powr, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Powr, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, b, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Powx, Tin, Tout, Acc::a>, int>::type
powx(const Tin* a, const Tin b, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Powx, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Powx, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, b, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Remainder, Tin, Tout, Acc::a>, int>::type
remainder(const Tin* a, const Tin* b, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Remainder, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Remainder, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, b, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Rint, Tin, Tout, Acc::a>, int>::type
rint(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Rint, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Rint, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Round, Tin, Tout, Acc::a>, int>::type
round(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Round, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Round, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Sin, Tin, Tout, Acc::a>, int>::type
sin(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Sin, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Sin, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Sincos, Tin, Tout, Acc::a>, int>::type
sincos(const Tin* a, Tout* y, Tout* z, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Sincos, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Sincos, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y, z);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Sincospi, Tin, Tout, Acc::a>, int>::type
sincospi(const Tin* a, Tout* y, Tout* z, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Sincospi, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Sincospi, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y, z);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Sind, Tin, Tout, Acc::a>, int>::type
sind(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Sind, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Sind, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Sinh, Tin, Tout, Acc::a>, int>::type
sinh(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Sinh, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Sinh, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Sinpi, Tin, Tout, Acc::a>, int>::type
sinpi(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Sinpi, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Sinpi, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Sqr, Tin, Tout, Acc::a>, int>::type
sqr(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Sqr, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Sqr, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Sqrt, Tin, Tout, Acc::a>, int>::type
sqrt(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Sqrt, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Sqrt, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Sub, Tin, Tout, Acc::a>, int>::type
sub(const Tin* a, const Tin* b, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Sub, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Sub, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, b, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Tan, Tin, Tout, Acc::a>, int>::type
tan(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Tan, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Tan, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Tand, Tin, Tout, Acc::a>, int>::type
tand(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Tand, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Tand, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Tanh, Tin, Tout, Acc::a>, int>::type
tanh(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Tanh, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Tanh, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Tanpi, Tin, Tout, Acc::a>, int>::type
tanpi(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Tanpi, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Tanpi, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Tgamma, Tin, Tout, Acc::a>, int>::type
tgamma(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Tgamma, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Tgamma, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}

template <typename Tin, typename Tout, typename Acc = decltype(DE)>
typename std::enable_if<ExistsAtAll<Function::Trunc, Tin, Tout, Acc::a>, int>::type
trunc(const Tin* a, Tout* y, Acc const& acc =  {}) {
    constexpr auto cts = CompileTimeSelector<Function::Trunc, Tin, Tout, Acc::a>::result;
    Evaluator<Function::Trunc, Tin, Tout, cts.first, cts.second> EV;
    return EV(a, y);
}




} // detail

using oneapi::mkl::vm::device::detail::abs;
using oneapi::mkl::vm::device::detail::acos;
using oneapi::mkl::vm::device::detail::acosh;
using oneapi::mkl::vm::device::detail::acospi;
using oneapi::mkl::vm::device::detail::add;
using oneapi::mkl::vm::device::detail::arg;
using oneapi::mkl::vm::device::detail::asin;
using oneapi::mkl::vm::device::detail::asinh;
using oneapi::mkl::vm::device::detail::asinpi;
using oneapi::mkl::vm::device::detail::atan;
using oneapi::mkl::vm::device::detail::atan2;
using oneapi::mkl::vm::device::detail::atan2pi;
using oneapi::mkl::vm::device::detail::atanh;
using oneapi::mkl::vm::device::detail::atanpi;
using oneapi::mkl::vm::device::detail::cbrt;
using oneapi::mkl::vm::device::detail::cdfnorm;
using oneapi::mkl::vm::device::detail::cdfnorminv;
using oneapi::mkl::vm::device::detail::ceil;
using oneapi::mkl::vm::device::detail::cis;
using oneapi::mkl::vm::device::detail::conj;
using oneapi::mkl::vm::device::detail::copysign;
using oneapi::mkl::vm::device::detail::cos;
using oneapi::mkl::vm::device::detail::cosd;
using oneapi::mkl::vm::device::detail::cosh;
using oneapi::mkl::vm::device::detail::cospi;
using oneapi::mkl::vm::device::detail::div;
using oneapi::mkl::vm::device::detail::erf;
using oneapi::mkl::vm::device::detail::erfc;
using oneapi::mkl::vm::device::detail::erfcx;
using oneapi::mkl::vm::device::detail::erfcinv;
using oneapi::mkl::vm::device::detail::erfinv;
using oneapi::mkl::vm::device::detail::exp;
using oneapi::mkl::vm::device::detail::exp10;
using oneapi::mkl::vm::device::detail::exp2;
using oneapi::mkl::vm::device::detail::expm1;
using oneapi::mkl::vm::device::detail::fdim;
using oneapi::mkl::vm::device::detail::floor;
using oneapi::mkl::vm::device::detail::fmax;
using oneapi::mkl::vm::device::detail::fmin;
using oneapi::mkl::vm::device::detail::fmod;
using oneapi::mkl::vm::device::detail::frac;
using oneapi::mkl::vm::device::detail::hypot;
using oneapi::mkl::vm::device::detail::inv;
using oneapi::mkl::vm::device::detail::invcbrt;
using oneapi::mkl::vm::device::detail::invsqrt;
using oneapi::mkl::vm::device::detail::lgamma;
using oneapi::mkl::vm::device::detail::ln;
using oneapi::mkl::vm::device::detail::log10;
using oneapi::mkl::vm::device::detail::log1p;
using oneapi::mkl::vm::device::detail::log2;
using oneapi::mkl::vm::device::detail::logb;
using oneapi::mkl::vm::device::detail::maxmag;
using oneapi::mkl::vm::device::detail::minmag;
using oneapi::mkl::vm::device::detail::modf;
using oneapi::mkl::vm::device::detail::mul;
using oneapi::mkl::vm::device::detail::mulbyconj;
using oneapi::mkl::vm::device::detail::nearbyint;
using oneapi::mkl::vm::device::detail::nextafter;
using oneapi::mkl::vm::device::detail::pow;
using oneapi::mkl::vm::device::detail::pow2o3;
using oneapi::mkl::vm::device::detail::pow3o2;
using oneapi::mkl::vm::device::detail::powr;
using oneapi::mkl::vm::device::detail::powx;
using oneapi::mkl::vm::device::detail::remainder;
using oneapi::mkl::vm::device::detail::rint;
using oneapi::mkl::vm::device::detail::round;
using oneapi::mkl::vm::device::detail::sin;
using oneapi::mkl::vm::device::detail::sincos;
using oneapi::mkl::vm::device::detail::sincospi;
using oneapi::mkl::vm::device::detail::sind;
using oneapi::mkl::vm::device::detail::sinh;
using oneapi::mkl::vm::device::detail::sinpi;
using oneapi::mkl::vm::device::detail::sqr;
using oneapi::mkl::vm::device::detail::sqrt;
using oneapi::mkl::vm::device::detail::sub;
using oneapi::mkl::vm::device::detail::tan;
using oneapi::mkl::vm::device::detail::tand;
using oneapi::mkl::vm::device::detail::tanh;
using oneapi::mkl::vm::device::detail::tanpi;
using oneapi::mkl::vm::device::detail::tgamma;
using oneapi::mkl::vm::device::detail::trunc;

namespace mode { using detail::ha; using detail::la; using detail::ep; using detail::not_defined; }

}

#endif

