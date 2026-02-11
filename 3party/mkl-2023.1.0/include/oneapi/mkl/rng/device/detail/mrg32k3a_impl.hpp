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

#ifndef _MKL_RNG_DEVICE_MRG32K3A_IMPL_HPP_
#define _MKL_RNG_DEVICE_MRG32K3A_IMPL_HPP_

#include "oneapi/mkl/rng/device/detail/mrg32k3a_skip_ahead_matrix.hpp"

namespace oneapi {
namespace mkl {
namespace rng {
namespace device {

template <std::int32_t VecSize = 1>
class mrg32k3a;

namespace detail {

// specialization of engine_state_device for mrg32k3a engine for generic DPC++ version
template <std::int32_t VecSize>
struct engine_state_device<oneapi::mkl::rng::device::mrg32k3a<VecSize>, device_type::generic> {
    std::uint32_t s[6];
};

// specialization of engine_state for mrg32k3a
template <std::int32_t VecSize>
union engine_state<oneapi::mkl::rng::device::mrg32k3a<VecSize>> {
    engine_state_device<oneapi::mkl::rng::device::mrg32k3a<VecSize>, device_type::generic>
        generic_state;
};

namespace mrg32k3a_impl {

struct mrg32k3a_params {
    static constexpr std::uint32_t m1 = 4294967087;
    static constexpr std::uint32_t m2 = 4294944443;
    static constexpr std::uint32_t a12 = 1403580;
    static constexpr std::uint32_t a13 = 4294156359;
    static constexpr std::uint32_t a21 = 527612;
    static constexpr std::uint32_t a23 = 4293573854;
    static constexpr std::uint32_t a13n = 810728;
    static constexpr std::uint32_t a23n = 1370589;
};

template <std::uint32_t M>
struct two_pow_32_minus_m {};

template <>
struct two_pow_32_minus_m<mrg32k3a_params::m1> {
    static constexpr std::int64_t val = 209;
};

template <>
struct two_pow_32_minus_m<mrg32k3a_params::m2> {
    static constexpr std::int64_t val = 22853;
};

template <std::int64_t M>
static inline std::uint64_t bit_shift_and_mask(std::int64_t in) {
    return ((in >> 32) * two_pow_32_minus_m<M>::val + (in & 0x00000000ffffffff));
}

// overload not to cast uint64_t input argument to int64_t and avoid warnings
template <std::int64_t M>
static inline std::uint64_t bit_shift_and_mask(std::uint64_t in) {
    return ((in >> 32) * two_pow_32_minus_m<M>::val + (in & 0x00000000ffffffffu));
}

// y = (x * (a ^ n)) % M
template <std::uint32_t M>
static inline void mv_mod(std::uint32_t a[3][3], std::uint32_t x[3], std::uint32_t y[3]) {
    std::uint64_t temp[3];
    for (int i = 0; i < 3; i++) {
        temp[i] = 0;
        for (int k = 0; k < 3; k++) {
            std::uint64_t tmp =
                static_cast<std::uint64_t>(a[i][k]) * static_cast<std::uint64_t>(x[k]);
            tmp = bit_shift_and_mask<M>(tmp);
            tmp = bit_shift_and_mask<M>(tmp);
            if (tmp >= M)
                tmp -= M;
            temp[i] += tmp;
        }
        temp[i] = bit_shift_and_mask<M>(temp[i]);
        if (temp[i] >= M)
            temp[i] -= M;
    }

    y[0] = static_cast<std::uint32_t>(temp[0]);
    y[1] = static_cast<std::uint32_t>(temp[1]);
    y[2] = static_cast<std::uint32_t>(temp[2]);
    return;
}

// c = (a * b) % M
template <std::uint32_t M>
static inline void mm_mod(const std::uint32_t a[3][3], std::uint32_t b[3][3],
                          std::uint32_t c[3][3]) {
    std::uint64_t temp[3][3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            temp[i][j] = 0;
            for (int k = 0; k < 3; k++) {
                std::uint64_t tmp =
                    static_cast<std::uint64_t>(a[i][k]) * static_cast<std::uint64_t>(b[k][j]);
                tmp = bit_shift_and_mask<M>(tmp);
                tmp = bit_shift_and_mask<M>(tmp);
                if (tmp >= M)
                    tmp -= M;
                temp[i][j] += tmp;
            }
            temp[i][j] = bit_shift_and_mask<M>(temp[i][j]);
            if (temp[i][j] >= M)
                temp[i][j] -= M;
        }
    }
    c[0][0] = static_cast<std::uint32_t>(temp[0][0]);
    c[0][1] = static_cast<std::uint32_t>(temp[0][1]);
    c[0][2] = static_cast<std::uint32_t>(temp[0][2]);
    c[1][0] = static_cast<std::uint32_t>(temp[1][0]);
    c[1][1] = static_cast<std::uint32_t>(temp[1][1]);
    c[1][2] = static_cast<std::uint32_t>(temp[1][2]);
    c[2][0] = static_cast<std::uint32_t>(temp[2][0]);
    c[2][1] = static_cast<std::uint32_t>(temp[2][1]);
    c[2][2] = static_cast<std::uint32_t>(temp[2][2]);
}

template <std::uint32_t M>
static inline void mm_pow_mod_precomp(std::uint32_t a[3][3], std::uint64_t n,
                                      const std::uint64_t* skip_params,
                                      const std::uint32_t skip_ahead_matrix[193][3][3]) {
    std::uint32_t tmp;
    const std::uint32_t* skip_params_32 = reinterpret_cast<const std::uint32_t*>(skip_params);
    int i;
    for (int j = 0; j < 2 * n; j++) {
        tmp = skip_params_32[j];
        i = 0;
        while (tmp) {
            if (tmp & (1 << i)) {
                mm_mod<M>(skip_ahead_matrix[j * 32 + i], a, a);
                tmp &= ~(1 << i);
            }
            i++;
        }
    }
}

template <std::uint32_t M>
static inline void vec_pow_mod_precomp(std::uint32_t x[3], std::uint64_t n,
                                       const std::uint64_t* skip_params,
                                       const std::uint32_t skip_ahead_matrix[193][3][3]) {
    std::uint32_t temp[3][3] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    mm_pow_mod_precomp<M>(temp, n, skip_params, skip_ahead_matrix);
    mv_mod<M>(temp, x, x);
}

template <std::int32_t VecSize>
static inline void skip_ahead(
    engine_state_device<oneapi::mkl::rng::device::mrg32k3a<VecSize>, device_type::generic>& state,
    std::uint64_t n, const std::uint64_t* num_to_skip_ptr) {
    if (n > 3) {
        n = 3;
#ifndef __SYCL_DEVICE_ONLY__
        throw oneapi::mkl::invalid_argument("rng", "mrg32k3a",
                                            "period is 2 ^ 191, skip on more than 2^192");
#endif
    }
    vec_pow_mod_precomp<mrg32k3a_params::m1>(state.s, n, num_to_skip_ptr, skip_ahead_table[0]);
    vec_pow_mod_precomp<mrg32k3a_params::m2>(state.s + 3, n, num_to_skip_ptr, skip_ahead_table[1]);
}

template <std::int32_t VecSize>
static inline void validate_seed(
    engine_state_device<oneapi::mkl::rng::device::mrg32k3a<VecSize>, device_type::generic>& state) {
    int i;
    for (i = 0; i < 3; i++) {
        if (state.s[i] >= mrg32k3a_params::m1) {
            state.s[i] -= mrg32k3a_params::m1;
        }
    }
    for (; i < 6; i++) {
        if (state.s[i] >= mrg32k3a_params::m2) {
            state.s[i] -= mrg32k3a_params::m2;
        }
    }

    if ((state.s[0]) == 0 && (state.s[1]) == 0 && (state.s[2]) == 0) {
        state.s[0] = 1;
    }
    if ((state.s[3]) == 0 && (state.s[4]) == 0 && (state.s[5]) == 0) {
        state.s[3] = 1;
    }
}

template <std::int32_t VecSize>
static inline void init(
    engine_state_device<oneapi::mkl::rng::device::mrg32k3a<VecSize>, device_type::generic>& state,
    std::uint64_t n, const std::uint32_t* seed_ptr, std::uint64_t n_offset,
    const std::uint64_t* offset_ptr) {
    int i;
    if (n > 6) {
        n = 6;
    }
    for (i = 0; i < n; i++) {
        state.s[i] = seed_ptr[i];
    }
    for (; i < 6; i++) {
        state.s[i] = 1;
    }
    validate_seed(state);
    mrg32k3a_impl::skip_ahead(state, n_offset, offset_ptr);
}

template <std::int32_t VecSize>
static inline void init(
    engine_state_device<oneapi::mkl::rng::device::mrg32k3a<VecSize>, device_type::generic>& state,
    size_t id, const sycl::accessor<std::uint32_t, 1, sycl::access::mode::read_write>& accessor) {
    size_t num_elements_acc =
        sizeof(engine_state<oneapi::mkl::rng::device::mrg32k3a<VecSize>>) / sizeof(std::uint32_t);
    state.s[0] = accessor[id * num_elements_acc];
    state.s[1] = accessor[id * num_elements_acc + 1];
    state.s[2] = accessor[id * num_elements_acc + 2];
    state.s[3] = accessor[id * num_elements_acc + 3];
    state.s[4] = accessor[id * num_elements_acc + 4];
    state.s[5] = accessor[id * num_elements_acc + 5];
}

template <std::int32_t VecSize>
static inline void store(
    engine_state_device<oneapi::mkl::rng::device::mrg32k3a<VecSize>, device_type::generic>& state,
    size_t id, const sycl::accessor<std::uint32_t, 1, sycl::access::mode::read_write>& accessor) {
    size_t num_elements_acc =
        sizeof(engine_state<oneapi::mkl::rng::device::mrg32k3a<VecSize>>) / sizeof(std::uint32_t);
    accessor[id * num_elements_acc] = state.s[0];
    accessor[id * num_elements_acc + 1] = state.s[1];
    accessor[id * num_elements_acc + 2] = state.s[2];
    accessor[id * num_elements_acc + 3] = state.s[3];
    accessor[id * num_elements_acc + 4] = state.s[4];
    accessor[id * num_elements_acc + 5] = state.s[5];
}

template <std::int32_t VecSize>
static inline sycl::vec<std::uint32_t, VecSize> generate(
    engine_state_device<oneapi::mkl::rng::device::mrg32k3a<VecSize>, device_type::generic>& state) {
    const std::int32_t num_elements = VecSize;
    sycl::vec<std::uint32_t, VecSize> res;
    std::int64_t x, y;
    std::int32_t i = 0;
    for (i = 0; i < num_elements; i++) {
        x = mrg32k3a_params::a12 * static_cast<std::int64_t>(state.s[1]) -
            mrg32k3a_params::a13n * static_cast<std::int64_t>(state.s[0]);
        // perform modulus
        x = bit_shift_and_mask<mrg32k3a_params::m1>(x);
        if (x >= mrg32k3a_params::m1)
            x -= mrg32k3a_params::m1;
        x += ((x & 0x8000000000000000) >> 63) * mrg32k3a_params::m1;
        y = mrg32k3a_params::a21 * static_cast<std::int64_t>(state.s[5]) -
            mrg32k3a_params::a23n * static_cast<std::int64_t>(state.s[3]);
        // perform modulus
        y = bit_shift_and_mask<mrg32k3a_params::m2>(y);
        y = bit_shift_and_mask<mrg32k3a_params::m2>(y);
        if (y >= mrg32k3a_params::m2)
            y -= mrg32k3a_params::m2;
        y += ((y & 0x8000000000000000) >> 63) * mrg32k3a_params::m2;
        state.s[0] = state.s[1];
        state.s[1] = state.s[2];
        state.s[2] = x;
        state.s[3] = state.s[4];
        state.s[4] = state.s[5];
        state.s[5] = y;
        if (x <= y) {
            res[i] = x + (mrg32k3a_params::m1 - y);
        }
        else {
            res[i] = x - y;
        }
    }
    return res;
}

template <std::int32_t VecSize>
static inline std::uint32_t generate_single(
    engine_state_device<oneapi::mkl::rng::device::mrg32k3a<VecSize>, device_type::generic>& state) {
    std::uint32_t res;
    std::int64_t x, y;
    x = mrg32k3a_params::a12 * static_cast<std::int64_t>(state.s[1]) -
        mrg32k3a_params::a13n * static_cast<std::int64_t>(state.s[0]);
    // perform modulus
    x = bit_shift_and_mask<mrg32k3a_params::m1>(x);
    if (x >= mrg32k3a_params::m1)
        x -= mrg32k3a_params::m1;
    x += ((x & 0x8000000000000000) >> 63) * mrg32k3a_params::m1;
    y = mrg32k3a_params::a21 * static_cast<std::int64_t>(state.s[5]) -
        mrg32k3a_params::a23n * static_cast<std::int64_t>(state.s[3]);
    // perform modulus
    y = bit_shift_and_mask<mrg32k3a_params::m2>(y);
    y = bit_shift_and_mask<mrg32k3a_params::m2>(y);
    if (y >= mrg32k3a_params::m2)
        y -= mrg32k3a_params::m2;
    y += ((y & 0x8000000000000000) >> 63) * mrg32k3a_params::m2;
    state.s[0] = state.s[1];
    state.s[1] = state.s[2];
    state.s[2] = x;
    state.s[3] = state.s[4];
    state.s[4] = state.s[5];
    state.s[5] = y;
    if (x <= y) {
        res = x + (mrg32k3a_params::m1 - y);
    }
    else {
        res = x - y;
    }

    return res;
}

} // namespace mrg32k3a_impl

// specialization for engine_accessor_base for mrg32k3a
template <std::int32_t VecSize>
class engine_accessor_base<oneapi::mkl::rng::device::mrg32k3a<VecSize>> {
public:
    engine_accessor_base(sycl::buffer<std::uint32_t, 1>& state_buf, sycl::handler& cgh)
            : states_accessor_(state_buf, cgh) {}

    oneapi::mkl::rng::device::mrg32k3a<VecSize> load(size_t id) const {
        oneapi::mkl::rng::device::mrg32k3a<VecSize> engine;
        mrg32k3a_impl::init(engine.state_.generic_state, id, states_accessor_);
        return engine;
    }

    void store(oneapi::mkl::rng::device::mrg32k3a<VecSize>& engine, size_t id) const {
        mrg32k3a_impl::store(engine.state_.generic_state, id, states_accessor_);
    }

protected:
    sycl::accessor<std::uint32_t, 1, sycl::access::mode::read_write> states_accessor_;
};

// specialization for engine_descriptor_base for mrg32k3a
template <std::int32_t VecSize>
class engine_descriptor_base<oneapi::mkl::rng::device::mrg32k3a<VecSize>> {
public:
    using engine_type = oneapi::mkl::rng::device::mrg32k3a<VecSize>;

    using accessor_type =
        oneapi::mkl::rng::device::engine_accessor<oneapi::mkl::rng::device::mrg32k3a<VecSize>>;

    engine_descriptor_base(sycl::queue& queue, sycl::range<1> range, std::uint32_t seed,
                           std::uint64_t offset)
            : states_buffer_(range.get(0) * sizeof(engine_state<engine_type>) /
                             sizeof(std::uint32_t)) {
        queue.submit([&](sycl::handler& cgh) {
            accessor_type states_accessor(states_buffer_, cgh);

            cgh.parallel_for<class init_kernel<engine_type>>
                (range, [=](sycl::item<1> item) {
                size_t id = item.get_id(0);
                oneapi::mkl::rng::device::mrg32k3a<VecSize> engine(seed, offset* id);
                states_accessor.store(engine, id);
            });
        });
    }

    template <typename InitEngineFunc>
    engine_descriptor_base(sycl::queue& queue, sycl::range<1> range, InitEngineFunc func)
            : states_buffer_(range.get(0) * sizeof(engine_state<engine_type>) /
                             sizeof(std::uint32_t)) {
        queue.submit([&](sycl::handler& cgh) {
            accessor_type states_accessor(states_buffer_, cgh);

            cgh.parallel_for<class init_kernel_ex<engine_type>>
                (range, [=](sycl::item<1> item) {
                size_t id = item.get_id(0);
                states_accessor.store(func(item), id);
            });
        });
    }

    accessor_type get_access(sycl::handler& cgh) {
        return accessor_type{ states_buffer_, cgh };
    }

protected:
    sycl::buffer<std::uint32_t, 1> states_buffer_;
};

template <std::int32_t VecSize>
class engine_base<oneapi::mkl::rng::device::mrg32k3a<VecSize>> {
protected:
    engine_base(std::uint32_t seed, std::uint64_t offset = 0) {
        mrg32k3a_impl::init(this->state_.generic_state, 1, &seed, 1, &offset);
    }

    engine_base(std::uint64_t n, const std::uint32_t* seed, std::uint64_t offset = 0) {
        mrg32k3a_impl::init(this->state_.generic_state, n, seed, 1, &offset);
    }

    engine_base(std::uint32_t seed, std::uint64_t n_offset, const std::uint64_t* offset_ptr) {
        mrg32k3a_impl::init(this->state_.generic_state, 1, &seed, n_offset, offset_ptr);
    }

    engine_base(std::uint64_t n, const std::uint32_t* seed, std::uint64_t n_offset,
                const std::uint64_t* offset_ptr) {
        mrg32k3a_impl::init(this->state_.generic_state, n, seed, n_offset, offset_ptr);
    }

    template <typename RealType>
    auto generate(RealType a, RealType b) ->
        typename std::conditional<VecSize == 1, RealType, sycl::vec<RealType, VecSize>>::type {
        sycl::vec<RealType, VecSize> res;
        sycl::vec<std::uint32_t, VecSize> res_uint;
        RealType c;

        c = (b - a) / (static_cast<RealType>(mrg32k3a_impl::mrg32k3a_params::m1));

        res_uint = mrg32k3a_impl::generate(this->state_.generic_state);

        for (int i = 0; i < VecSize; i++) {
            res[i] = (RealType)(res_uint[i]) * c + a;
        }
        return res;
    }

    auto generate() -> typename std::conditional<VecSize == 1, std::uint32_t,
                                                 sycl::vec<std::uint32_t, VecSize>>::type {
        return mrg32k3a_impl::generate(this->state_.generic_state);
    }

    template <typename RealType>
    RealType generate_single(RealType a, RealType b) {
        RealType res;
        std::uint32_t res_uint;
        RealType c;

        c = (b - a) / (static_cast<RealType>(mrg32k3a_impl::mrg32k3a_params::m1));

        res_uint = mrg32k3a_impl::generate_single(this->state_.generic_state);

        res = (RealType)(res_uint)*c + a;

        return res;
    }

    std::uint32_t generate_single() {
        return mrg32k3a_impl::generate_single(this->state_.generic_state);
    }

    void skip_ahead(std::uint64_t num_to_skip) {
        detail::mrg32k3a_impl::skip_ahead(this->state_.generic_state, 1, &num_to_skip);
    }

    void skip_ahead(std::initializer_list<std::uint64_t> num_to_skip) {
        detail::mrg32k3a_impl::skip_ahead(this->state_.generic_state, num_to_skip.size(),
                                          num_to_skip.begin());
    }

    engine_state<oneapi::mkl::rng::device::mrg32k3a<VecSize>> state_;
};

} // namespace detail
} // namespace device
} // namespace rng
} // namespace mkl
} // namespace oneapi

#endif // _MKL_RNG_DEVICE_MRG32K3A_IMPL_HPP_
