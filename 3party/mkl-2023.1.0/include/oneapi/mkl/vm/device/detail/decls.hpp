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

#ifndef __ONEAPI_MKL_VM_DEVICE_DETAIL_DECLS_HPP__
#define __ONEAPI_MKL_VM_DEVICE_DETAIL_DECLS_HPP__


namespace oneapi::mkl::vm::device {
namespace detail {

enum class Accuracy : int {
    NotAvailable = -1,
    NA = NotAvailable,

    NotSpecified = 0,
    NS = NotSpecified,

    CorrectlyRounded = 1,
    CR = CorrectlyRounded,

    High = 4,
    HA = High,

    Lower = 8,
    LA = Lower,

    EnhancedPerformance = 15,
    EP = EnhancedPerformance,
};

struct _HA { static constexpr Accuracy a = Accuracy::High; };
struct _LA { static constexpr Accuracy a = Accuracy::Lower; };
struct _EP { static constexpr Accuracy a = Accuracy::EnhancedPerformance; };

constexpr _HA HA;
constexpr _LA LA;
constexpr _EP EP;

constexpr _HA DE;

constexpr _HA ha;
constexpr _LA la;
constexpr _EP ep;

constexpr _HA not_defined;

enum class Feature : int {
    NotAvailable = -1,
    NA = NotAvailable,

    NotSpecified = 0,
    NS = NotSpecified,

    Generic = 1,
    GE = Generic,

    Tailored = 2,
    TA = Tailored,
};

}


}

#endif


