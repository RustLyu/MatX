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

#ifndef _MKL_RNG_DEVICE_PHILOX4X32X10_IMPL_HPP_
#define _MKL_RNG_DEVICE_PHILOX4X32X10_IMPL_HPP_

#include <cstdint> // to use std::int32_t, std::uint64_t
#include <utility> // std::pair

namespace oneapi {
namespace mkl {
namespace rng {
namespace device {

template <std::int32_t VecSize = 1>
class philox4x32x10;

namespace detail {

// specialization of engine_state_device for philox4x32x10 engine for generic DPC++ version
template <std::int32_t VecSize>
struct engine_state_device<oneapi::mkl::rng::device::philox4x32x10<VecSize>, device_type::generic> {
    std::uint32_t key[2];
    std::uint32_t counter[4];
    std::uint32_t part;
    std::uint32_t result[4];
};

// specialization of engine_state for philox4x32x10
template <std::int32_t VecSize>
union engine_state<oneapi::mkl::rng::device::philox4x32x10<VecSize>> {
    engine_state_device<oneapi::mkl::rng::device::philox4x32x10<VecSize>, device_type::generic>
        generic_state;
};

namespace philox4x32x10_impl {

static inline void add128(std::uint32_t* a, std::uint64_t b) {
    std::uint32_t b_lo = static_cast<std::uint32_t>(b);
    std::uint32_t b_hi = static_cast<std::uint32_t>(b >> 32);

    a[0] += b_lo;
    if (a[0] < b_lo) {
        b_hi++;
    }
    a[1] += b_hi;
    if (a[1] >= b_hi) {
        return;
    }
    if (++a[2]) {
        return;
    }
    ++a[3];
}

static inline void add128_1(std::uint32_t* a) {
    if (++a[0]) {
        return;
    }
    if (++a[1]) {
        return;
    }
    if (++a[2]) {
        return;
    }
    ++a[3];
}

static inline std::pair<std::uint32_t, std::uint32_t> mul_hilo_32(std::uint32_t a,
                                                                  std::uint32_t b) {
    std::uint64_t res_64 = static_cast<std::uint64_t>(a) * static_cast<std::uint64_t>(b);
    return std::make_pair(static_cast<std::uint32_t>(res_64),
                          static_cast<std::uint32_t>(res_64 >> 32));
}

static inline void round(std::uint32_t* cnt, std::uint32_t* k) {
    auto [L0, H0] = mul_hilo_32(0xD2511F53, cnt[0]);
    auto [L1, H1] = mul_hilo_32(0xCD9E8D57, cnt[2]);

    cnt[0] = H1 ^ cnt[1] ^ k[0];
    cnt[1] = L1;
    cnt[2] = H0 ^ cnt[3] ^ k[1];
    cnt[3] = L0;
}

static inline void round_10(std::uint32_t* cnt, std::uint32_t* k) {
    round(cnt, k); // 1
    // increasing keys with philox4x32x10 constants
    k[0] += 0x9E3779B9;
    k[1] += 0xBB67AE85;
    round(cnt, k); // 2
    k[0] += 0x9E3779B9;
    k[1] += 0xBB67AE85;
    round(cnt, k); // 3
    k[0] += 0x9E3779B9;
    k[1] += 0xBB67AE85;
    round(cnt, k); // 4
    k[0] += 0x9E3779B9;
    k[1] += 0xBB67AE85;
    round(cnt, k); // 5
    k[0] += 0x9E3779B9;
    k[1] += 0xBB67AE85;
    round(cnt, k); // 6
    k[0] += 0x9E3779B9;
    k[1] += 0xBB67AE85;
    round(cnt, k); // 7
    k[0] += 0x9E3779B9;
    k[1] += 0xBB67AE85;
    round(cnt, k); // 8
    k[0] += 0x9E3779B9;
    k[1] += 0xBB67AE85;
    round(cnt, k); // 9
    k[0] += 0x9E3779B9;
    k[1] += 0xBB67AE85;
    round(cnt, k); // 10
}

template <std::int32_t VecSize>
static inline void skip_ahead(engine_state_device<oneapi::mkl::rng::device::philox4x32x10<VecSize>,
                                                  device_type::generic>& state,
                              std::uint64_t num_to_skip) {
    std::uint64_t num_to_skip_tmp = num_to_skip;
    std::uint64_t c_inc;
    std::uint32_t counter[4];
    std::uint32_t key[2];
    std::uint64_t tail;
    if (num_to_skip_tmp <= state.part) {
        state.part -= num_to_skip_tmp;
    }
    else {
        tail = num_to_skip % 4;
        if ((tail == 0) && (state.part == 0)) {
            add128(state.counter, num_to_skip / 4);
        }
        else {
            num_to_skip_tmp = num_to_skip_tmp - state.part;
            state.part = 0;
            c_inc = (num_to_skip_tmp - 1) / 4;
            state.part = (4 - num_to_skip_tmp % 4) % 4;
            add128(state.counter, c_inc);
            counter[0] = state.counter[0];
            counter[1] = state.counter[1];
            counter[2] = state.counter[2];
            counter[3] = state.counter[3];
            key[0] = state.key[0];
            key[1] = state.key[1];
            round_10(counter, key);
            state.result[0] = counter[0];
            state.result[1] = counter[1];
            state.result[2] = counter[2];
            state.result[3] = counter[3];
            add128_1(state.counter);
        }
    }
}

template <std::int32_t VecSize>
static inline void skip_ahead(engine_state_device<oneapi::mkl::rng::device::philox4x32x10<VecSize>,
                                                  device_type::generic>& state,
                              std::uint64_t n, const std::uint64_t* num_to_skip_ptr) {
    constexpr std::uint64_t uint_max = 0xFFFFFFFFFFFFFFFF;
    std::uint64_t post_buffer, pre_buffer;
    std::int32_t num_elements = 0;
    std::int32_t remained_counter;
    std::uint64_t tmp_skip_array[3] = { 0, 0, 0 };

    for (int i = 0; (i < 3) && (i < n); i++) {
        tmp_skip_array[i] = num_to_skip_ptr[i];
        if (tmp_skip_array[i]) {
            num_elements = i + 1;
        }
    }

    if (num_elements == 0) {
        return;
    }
    if ((num_elements == 1) && (tmp_skip_array[0] <= state.part)) {
        state.part -= static_cast<std::uint32_t>(tmp_skip_array[0]);
        return;
    }
    std::uint32_t counter[4];
    std::uint32_t key[2];

    if ((tmp_skip_array[0] - state.part) <= tmp_skip_array[0]) {
        tmp_skip_array[0] = tmp_skip_array[0] - state.part;
    }
    else if ((num_elements == 2) || (tmp_skip_array[1] - 1 < tmp_skip_array[1])) {
        tmp_skip_array[1] = tmp_skip_array[1] - 1;
        tmp_skip_array[0] = uint_max - state.part + tmp_skip_array[0];
    }
    else {
        tmp_skip_array[2] = tmp_skip_array[2] - 1;
        tmp_skip_array[1] = uint_max - 1;
        tmp_skip_array[0] = uint_max - state.part + tmp_skip_array[0];
    }

    state.part = 0;

    post_buffer = 0;

    remained_counter = static_cast<std::uint32_t>(tmp_skip_array[0] % 4);

    for (int i = num_elements - 1; i >= 0; i--) {
        pre_buffer = (tmp_skip_array[i] << 62);
        tmp_skip_array[i] >>= 2;
        tmp_skip_array[i] |= post_buffer;
        post_buffer = pre_buffer;
    }

    state.part = 4 - remained_counter;

    std::uint64_t counter64[] = { state.counter[1], state.counter[3] };
    counter64[0] = ((counter64[0] << 32ull) | state.counter[0]);
    counter64[1] = ((counter64[1] << 32ull) | state.counter[2]);

    counter64[0] += tmp_skip_array[0];

    if (counter64[0] < tmp_skip_array[0]) {
        counter64[1]++;
    }

    counter64[1] += tmp_skip_array[1];

    counter[0] = static_cast<std::uint32_t>(counter64[0]);
    counter[1] = static_cast<std::uint32_t>(counter64[0] >> 32);
    counter[2] = static_cast<std::uint32_t>(counter64[1]);
    counter[3] = static_cast<std::uint32_t>(counter64[1] >> 32);

    key[0] = state.key[0];
    key[1] = state.key[1];

    round_10(counter, key);

    state.result[0] = counter[0];
    state.result[1] = counter[1];
    state.result[2] = counter[2];
    state.result[3] = counter[3];

    counter64[0]++;

    if (counter64[0] < 1) {
        counter64[1]++;
    }

    state.counter[0] = static_cast<std::uint32_t>(counter64[0]);
    state.counter[1] = static_cast<std::uint32_t>(counter64[0] >> 32);
    state.counter[2] = static_cast<std::uint32_t>(counter64[1]);
    state.counter[3] = static_cast<std::uint32_t>(counter64[1] >> 32);
}

template <std::int32_t VecSize>
static inline void init(engine_state_device<oneapi::mkl::rng::device::philox4x32x10<VecSize>,
                                            device_type::generic>& state,
                        std::uint64_t n, const std::uint64_t* seed_ptr, std::uint64_t offset) {
    state.key[0] = static_cast<std::uint32_t>(seed_ptr[0]);
    state.key[1] = static_cast<std::uint32_t>(seed_ptr[0] >> 32);

    state.counter[0] = (n >= 2 ? static_cast<std::uint32_t>(seed_ptr[1]) : 0);
    state.counter[1] = (n >= 2 ? static_cast<std::uint32_t>(seed_ptr[1] >> 32) : 0);

    state.counter[2] = (n >= 3 ? static_cast<std::uint32_t>(seed_ptr[2]) : 0);
    state.counter[3] = (n >= 3 ? static_cast<std::uint32_t>(seed_ptr[2] >> 32) : 0);

    state.part = 0;
    state.result[0] = 0;
    state.result[1] = 0;
    state.result[2] = 0;
    state.result[3] = 0;
    skip_ahead(state, offset);
}

template <std::int32_t VecSize>
static inline void init(engine_state_device<oneapi::mkl::rng::device::philox4x32x10<VecSize>,
                                            device_type::generic>& state,
                        std::uint64_t n, const std::uint64_t* seed_ptr, std::uint64_t n_offset,
                        const std::uint64_t* offset_ptr) {
    state.key[0] = static_cast<std::uint32_t>(seed_ptr[0]);
    state.key[1] = static_cast<std::uint32_t>(seed_ptr[0] >> 32);

    state.counter[0] = (n >= 2 ? static_cast<std::uint32_t>(seed_ptr[1]) : 0);
    state.counter[1] = (n >= 2 ? static_cast<std::uint32_t>(seed_ptr[1] >> 32) : 0);

    state.counter[2] = (n >= 3 ? static_cast<std::uint32_t>(seed_ptr[2]) : 0);
    state.counter[3] = (n >= 3 ? static_cast<std::uint32_t>(seed_ptr[2] >> 32) : 0);

    state.part = 0;
    state.result[0] = 0;
    state.result[1] = 0;
    state.result[2] = 0;
    state.result[3] = 0;
    skip_ahead(state, n_offset, offset_ptr);
}

template <std::int32_t VecSize>
static inline void init(
    engine_state_device<oneapi::mkl::rng::device::philox4x32x10<VecSize>, device_type::generic>&
        state,
    size_t id, const sycl::accessor<std::uint32_t, 1, sycl::access::mode::read_write>& accessor) {
    size_t num_elements_acc =
        sizeof(engine_state<oneapi::mkl::rng::device::philox4x32x10<VecSize>>) /
        sizeof(std::uint32_t);
    state.key[0] = accessor[id * num_elements_acc];
    state.key[1] = accessor[id * num_elements_acc + 1];
    state.counter[0] = accessor[id * num_elements_acc + 2];
    state.counter[1] = accessor[id * num_elements_acc + 3];
    state.counter[2] = accessor[id * num_elements_acc + 4];
    state.counter[3] = accessor[id * num_elements_acc + 5];

    state.part = accessor[id * num_elements_acc + 6];

    state.result[0] = accessor[id * num_elements_acc + 7];
    state.result[1] = accessor[id * num_elements_acc + 8];
    state.result[2] = accessor[id * num_elements_acc + 9];
    state.result[3] = accessor[id * num_elements_acc + 10];
}

template <std::int32_t VecSize>
static inline void store(
    engine_state_device<oneapi::mkl::rng::device::philox4x32x10<VecSize>, device_type::generic>&
        state,
    size_t id, const sycl::accessor<std::uint32_t, 1, sycl::access::mode::read_write>& accessor) {
    size_t num_elements_acc =
        sizeof(engine_state<oneapi::mkl::rng::device::philox4x32x10<VecSize>>) /
        sizeof(std::uint32_t);
    accessor[id * num_elements_acc] = state.key[0];
    accessor[id * num_elements_acc + 1] = state.key[1];
    accessor[id * num_elements_acc + 2] = state.counter[0];
    accessor[id * num_elements_acc + 3] = state.counter[1];
    accessor[id * num_elements_acc + 4] = state.counter[2];
    accessor[id * num_elements_acc + 5] = state.counter[3];
    accessor[id * num_elements_acc + 6] = state.part;
    accessor[id * num_elements_acc + 7] = state.result[0];
    accessor[id * num_elements_acc + 8] = state.result[1];
    accessor[id * num_elements_acc + 9] = state.result[2];
    accessor[id * num_elements_acc + 10] = state.result[3];
}

// for VecSize > 4
template <std::int32_t VecSize>
__attribute__((always_inline))
static inline sycl::vec<std::uint32_t, VecSize> generate_full(
    engine_state_device<oneapi::mkl::rng::device::philox4x32x10<VecSize>, device_type::generic>&
        state) {
    const std::int32_t num_elements = VecSize;
    sycl::vec<std::uint32_t, VecSize> res;

    std::uint32_t counter[4];

    int i = 0;
    int part = (int)state.part;
    while (part && (i < num_elements)) {
        res[i++] = state.result[3 - (--part)];
    }
    if (i == num_elements) {
        skip_ahead(state, num_elements);
        return res;
    }

    counter[0] = state.counter[0];
    counter[1] = state.counter[1];
    counter[2] = state.counter[2];
    counter[3] = state.counter[3];

    std::uint32_t cntTmp[4];
    std::uint32_t keyTmp[2];
    for (; i < num_elements; i += 4) {
        cntTmp[0] = counter[0];
        cntTmp[1] = counter[1];
        cntTmp[2] = counter[2];
        cntTmp[3] = counter[3];

        keyTmp[0] = state.key[0];
        keyTmp[1] = state.key[1];

        round_10(cntTmp, keyTmp);

        if (i + 4 <= num_elements) {
            for (int j = 0; j < 4; j++) {
                res[i + j] = cntTmp[j];
            }
            add128_1(counter);
        }
        else {
            // here if last iteration
            for (int j = 0; i < num_elements; i++, j++) {
                res[i] = cntTmp[j];
            }
        }
    }
    skip_ahead(state, num_elements);
    return res;
}

// for VecSize <= 4
template <std::int32_t VecSize>
__attribute__((always_inline))
static inline sycl::vec<std::uint32_t, VecSize> generate_small(
    engine_state_device<oneapi::mkl::rng::device::philox4x32x10<VecSize>, device_type::generic>&
        state) {
    const std::int32_t num_elements = VecSize;
    sycl::vec<std::uint32_t, VecSize> res;

    std::uint32_t counter[4];
    std::uint32_t key[2];

    int i = 0;
    int part = (int)state.part;
    while (part && (i < num_elements)) {
        res[i++] = state.result[3 - (--part)];
    }
    if (i == num_elements) {
        skip_ahead(state, num_elements);
        return res;
    }

    counter[0] = state.counter[0];
    counter[1] = state.counter[1];
    counter[2] = state.counter[2];
    counter[3] = state.counter[3];
    key[0] = state.key[0];
    key[1] = state.key[1];

    round_10(counter, key);

    for (int j = 0; i < num_elements; i++, j++) {
        res[i] = counter[j];
    }

    skip_ahead(state, num_elements);
    return res;
}

template <int VecSize>
__attribute__((always_inline))
static inline std::uint32_t generate_single(
    engine_state_device<oneapi::mkl::rng::device::philox4x32x10<VecSize>, device_type::generic>&
        state) {
    std::uint32_t res;

    std::uint32_t counter[4];
    std::uint32_t key[2];

    std::int32_t part = static_cast<std::int32_t>(state.part);
    if (part != 0) {
        res = state.result[3 - (--part)];
        skip_ahead(state, 1);
        return res;
    }
    counter[0] = state.counter[0];
    counter[1] = state.counter[1];
    counter[2] = state.counter[2];
    counter[3] = state.counter[3];
    key[0] = state.key[0];
    key[1] = state.key[1];

    round_10(counter, key);

    res = counter[0];

    skip_ahead(state, 1);
    return res;
}

} // namespace philox4x32x10_impl

// specialization for engine_accessor_base for philox4x32x10
template <std::int32_t VecSize>
class engine_accessor_base<oneapi::mkl::rng::device::philox4x32x10<VecSize>> {
public:
    engine_accessor_base(sycl::buffer<std::uint32_t, 1>& state_buf, sycl::handler& cgh)
            : states_accessor_(state_buf, cgh) {}

    oneapi::mkl::rng::device::philox4x32x10<VecSize> load(size_t id) const {
        oneapi::mkl::rng::device::philox4x32x10<VecSize> engine;
        philox4x32x10_impl::init(engine.state_.generic_state, id, states_accessor_);
        return engine;
    }

    void store(oneapi::mkl::rng::device::philox4x32x10<VecSize>& engine, size_t id) const {
        philox4x32x10_impl::store(engine.state_.generic_state, id, states_accessor_);
    }

protected:
    sycl::accessor<std::uint32_t, 1, sycl::access::mode::read_write> states_accessor_;
};

// specialization for engine_descriptor_base for philox4x32x10
template <std::int32_t VecSize>
class engine_descriptor_base<oneapi::mkl::rng::device::philox4x32x10<VecSize>> {
public:
    using engine_type = oneapi::mkl::rng::device::philox4x32x10<VecSize>;

    using accessor_type =
        oneapi::mkl::rng::device::engine_accessor<oneapi::mkl::rng::device::philox4x32x10<VecSize>>;

    engine_descriptor_base(sycl::queue& queue, sycl::range<1> range, std::uint64_t seed,
                           std::uint64_t offset)
            : states_buffer_(range.get(0) * sizeof(engine_state<engine_type>) /
                             sizeof(std::uint32_t)) {
        queue.submit([&](sycl::handler& cgh) {
            accessor_type states_accessor(states_buffer_, cgh);

            cgh.parallel_for<class init_kernel<engine_type>>
                (range, [=](sycl::item<1> item) {
                size_t id = item.get_id(0);
                oneapi::mkl::rng::device::philox4x32x10<VecSize> engine(seed, offset* id);
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
class engine_base<oneapi::mkl::rng::device::philox4x32x10<VecSize>> {
protected:
    engine_base(std::uint64_t seed, std::uint64_t offset = 0) {
        philox4x32x10_impl::init(this->state_.generic_state, 1, &seed, offset);
    }

    engine_base(std::uint64_t n, const std::uint64_t* seed, std::uint64_t offset = 0) {
        philox4x32x10_impl::init(this->state_.generic_state, n, seed, offset);
    }

    engine_base(std::uint64_t seed, std::uint64_t n_offset, const std::uint64_t* offset_ptr) {
        philox4x32x10_impl::init(this->state_.generic_state, 1, &seed, n_offset, offset_ptr);
    }

    engine_base(std::uint64_t n, const std::uint64_t* seed, std::uint64_t n_offset,
                const std::uint64_t* offset_ptr) {
        philox4x32x10_impl::init(this->state_.generic_state, n, seed, n_offset, offset_ptr);
    }

    template <typename RealType>
    __attribute__((always_inline))
    inline auto generate(RealType a, RealType b) ->
        typename std::conditional<VecSize == 1, RealType, sycl::vec<RealType, VecSize>>::type {
        sycl::vec<RealType, VecSize> res;
        sycl::vec<std::uint32_t, VecSize> res_uint;
        RealType a1;
        RealType c1;

        c1 = (b - a) / (static_cast<RealType>((std::numeric_limits<std::uint32_t>::max)()) + 1);
        a1 = (b + a) / static_cast<RealType>(2.0);

        if constexpr (VecSize > 4) {
            res_uint = philox4x32x10_impl::generate_full(this->state_.generic_state);
        }
        else {
            res_uint = philox4x32x10_impl::generate_small(this->state_.generic_state);
        }
        for (int i = 0; i < VecSize; i++) {
            res[i] = static_cast<RealType>(static_cast<std::int32_t>(res_uint[i])) * c1 + a1;
        }
        return res;
    }

    __attribute__((always_inline))
    inline auto generate() -> typename std::conditional<VecSize == 1, std::uint32_t,
                                                 sycl::vec<std::uint32_t, VecSize>>::type {
        if constexpr (VecSize > 4) {
            return philox4x32x10_impl::generate_full(this->state_.generic_state);
        }
        return philox4x32x10_impl::generate_small(this->state_.generic_state);
    }

    template <typename UIntType>
    __attribute__((always_inline))
    inline auto generate_uniform_bits() ->
        typename std::conditional<VecSize == 1, UIntType,
                                                 sycl::vec<UIntType, VecSize>>::type {
        if constexpr (std::is_same<UIntType, std::uint32_t>::value) {
            return generate();
        } else {
            auto uni_res1 = generate();
            auto uni_res2 = generate();

            if constexpr (VecSize == 1) {
                return (static_cast<std::uint64_t>(uni_res2) << 32) + uni_res1;
            } else {
                sycl::vec<std::uint64_t, VecSize> vec_out;

                if constexpr (VecSize != 3) {
                    for (int i = 0; i < VecSize / 2; i++) {
                        vec_out[i] = (static_cast<std::uint64_t>(uni_res1[2 * i + 1]) << 32) + uni_res1[2 * i];
                        vec_out[i + VecSize / 2] = (static_cast<std::uint64_t>(uni_res2[2 * i + 1]) << 32) + uni_res2[2 * i];
                    }
                } else {
                    vec_out[0] = (static_cast<std::uint64_t>(uni_res1[1]) << 32) + uni_res1[0];
                    vec_out[1] = (static_cast<std::uint64_t>(uni_res2[0]) << 32) + uni_res1[2];
                    vec_out[2] = (static_cast<std::uint64_t>(uni_res2[2]) << 32) + uni_res2[1];
                }

                return vec_out;
            }
        }
    }

    template <typename RealType>
    RealType generate_single(RealType a, RealType b) {
        RealType res;
        std::uint32_t res_uint;
        RealType a1;
        RealType c1;

        c1 = (b - a) / (static_cast<RealType>((std::numeric_limits<std::uint32_t>::max)()) + 1);
        a1 = (b + a) / static_cast<RealType>(2.0);

        res_uint = philox4x32x10_impl::generate_single(this->state_.generic_state);

        res = static_cast<RealType>(static_cast<std::int32_t>(res_uint)) * c1 + a1;

        return res;
    }

    __attribute__((always_inline))
    inline std::uint32_t generate_single() {
        return philox4x32x10_impl::generate_single(this->state_.generic_state);
    }

    template <typename UIntType>
    __attribute__((always_inline))
    inline auto generate_single_uniform_bits() {
        if constexpr (std::is_same<UIntType, std::uint32_t>::value) {
            return philox4x32x10_impl::generate_single(this->state_.generic_state);
        } else {
            auto uni_res1 = philox4x32x10_impl::generate_single(this->state_.generic_state);
            auto uni_res2 = philox4x32x10_impl::generate_single(this->state_.generic_state);

            return (static_cast<std::uint64_t>(uni_res2) << 32) + uni_res1;
        }
    }

    void skip_ahead(std::uint64_t num_to_skip) {
        detail::philox4x32x10_impl::skip_ahead(this->state_.generic_state, num_to_skip);
    }

    void skip_ahead(std::initializer_list<std::uint64_t> num_to_skip) {
        detail::philox4x32x10_impl::skip_ahead(this->state_.generic_state, num_to_skip.size(),
                                               num_to_skip.begin());
    }

    engine_state<oneapi::mkl::rng::device::philox4x32x10<VecSize>> state_;
};

} // namespace detail
} // namespace device
} // namespace rng
} // namespace mkl
} // namespace oneapi

#endif // _MKL_RNG_DEVICE_PHILOX4X32X10_IMPL_HPP_
