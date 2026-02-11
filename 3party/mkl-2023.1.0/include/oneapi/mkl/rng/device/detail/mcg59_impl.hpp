/*******************************************************************************
* Copyright 2021-2022 Intel Corporation.
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

#ifndef _MKL_RNG_DEVICE_MCG59_IMPL_HPP_
#define _MKL_RNG_DEVICE_MCG59_IMPL_HPP_

#include <cstdint> // to use std::int32_t, std::uint64_t

namespace oneapi {
namespace mkl {
namespace rng {
namespace device {

template <std::int32_t VecSize = 1> class mcg59;

namespace detail {

template <std::uint32_t VecSize>
constexpr sycl::vec<uint64_t, VecSize> select_vector_a_mcg59() {
  if constexpr (VecSize == 1)
    return sycl::vec<uint64_t, 1>(UINT64_C(1));
  else if constexpr (VecSize == 2)
    return sycl::vec<uint64_t, 2>({UINT64_C(1), UINT64_C(0x113769B23C5FD)});
  else if constexpr (VecSize == 3)
    return sycl::vec<uint64_t, 3>(
        {UINT64_C(1), UINT64_C(0x113769B23C5FD), UINT64_C(0x65C69FC1A4D5C09)});
  else if constexpr (VecSize == 4)
    return sycl::vec<uint64_t, 4>(
        {UINT64_C(1), UINT64_C(0x113769B23C5FD), UINT64_C(0x65C69FC1A4D5C09), UINT64_C(0x1CE44D68E81E1E5)});
  else if constexpr (VecSize == 8)
    return sycl::vec<uint64_t, 8>({UINT64_C(1), UINT64_C(0x113769B23C5FD), UINT64_C(0x65C69FC1A4D5C09),
                                   UINT64_C(0x1CE44D68E81E1E5), UINT64_C(0x2F861CA52807851),
                                   UINT64_C(0x1CCDF2FE3A03D0D), UINT64_C(0x707AB5B7C1E56D9),
                                   UINT64_C(0x6139AE457BD175)});
  else
    return sycl::vec<uint64_t, 16>(
        {UINT64_C(1), UINT64_C(0x113769B23C5FD), UINT64_C(0x65C69FC1A4D5C09), UINT64_C(0x1CE44D68E81E1E5),
         UINT64_C(0x2F861CA52807851), UINT64_C(0x1CCDF2FE3A03D0D), UINT64_C(0x707AB5B7C1E56D9),
         UINT64_C(0x6139AE457BD175), UINT64_C(0x171CF606D8C09A1), UINT64_C(0x3764DC8D2D1691D),
         UINT64_C(0x50A1576CCF32A9), UINT64_C(0x499F3083ADC1E05), UINT64_C(0x7A30C00B05283F1),
         UINT64_C(0x4FE299EB607DA2D), UINT64_C(0x51CCFD803CE3F79), UINT64_C(0x58145D06A37D795)});
}

template <std::uint32_t VecSize> struct mcg59_vector_a {
  static constexpr sycl::vec<std::uint64_t, VecSize> vector_a =
      select_vector_a_mcg59<VecSize>(); // powers of a
};

struct mcg59_param {
  static constexpr uint64_t a = 0x113769B23C5FD;       // 13^13
  static constexpr uint64_t m_64 = 0x7FFFFFFFFFFFFFF;  // 2^59 - 1
  static constexpr float m_fl = 576460752303423488.0f; // 2^59
};

// specialization of engine_state_device for mcg59 engine for generic DPC++
// version
template <std::int32_t VecSize>
struct engine_state_device<oneapi::mkl::rng::device::mcg59<VecSize>,
                           device_type::generic> {
  std::uint64_t s;
};

// specialization of engine_state for mcg59
template <std::int32_t VecSize>
union engine_state<oneapi::mkl::rng::device::mcg59<VecSize>> {
  engine_state_device<oneapi::mkl::rng::device::mcg59<VecSize>,
                      device_type::generic>
      generic_state;
};

namespace mcg59_impl {

template <typename T> static inline T custom_mod(T x) {
  return (x & mcg59_param::m_64);
}

static inline std::uint64_t power(std::uint64_t a, std::uint64_t n) {
  // initialize result by 1 for recurrency
  std::uint64_t result = 1;
  if (n == 0) {
    // return (a^0)%m = 1
    return 1;
  }
  do {
    // For each odd n
    if (n & 1) {
      result = custom_mod(result * a);
    }
    // n := n/2
    n >>= 1;
    a = custom_mod(a * a);
  } while (n);

  return result;
}

template <std::int32_t VecSize>
static inline void
skip_ahead(engine_state_device<oneapi::mkl::rng::device::mcg59<VecSize>,
                               device_type::generic> &state,
           std::uint64_t num_to_skip) {
  std::uint64_t loc_A = power(mcg59_param::a, num_to_skip);
  state.s = custom_mod(loc_A * state.s);
}

template <std::int32_t VecSize>
static inline void
init(engine_state_device<oneapi::mkl::rng::device::mcg59<VecSize>,
                         device_type::generic> &state,
     std::uint64_t n, std::uint32_t *seed_ptr, std::uint64_t offset) {
  if (n < 1) {
    state.s = 1;
  } else if (n == 1) {
    state.s = static_cast<uint64_t>(seed_ptr[0]) & mcg59_param::m_64;
  } else {
    state.s =
        *(reinterpret_cast<std::uint64_t *>(&seed_ptr[0])) & mcg59_param::m_64;
  }
  if (state.s == 0)
    state.s = 1;

  skip_ahead(state, offset);
}

template <std::int32_t VecSize>
static inline void
init(engine_state_device<oneapi::mkl::rng::device::mcg59<VecSize>,
                         device_type::generic> &state,
     size_t id,
     const sycl::accessor<std::uint32_t, 1, sycl::access::mode::read_write>
         &accessor) {
  size_t num_elements_acc =
      sizeof(engine_state<oneapi::mkl::rng::device::mcg59<VecSize>>) /
      sizeof(std::uint32_t);
  state.s =
      *(reinterpret_cast<std::uint64_t *>(&accessor[id * num_elements_acc]));
}

template <std::int32_t VecSize>
static inline void
store(engine_state_device<oneapi::mkl::rng::device::mcg59<VecSize>,
                          device_type::generic> &state,
      size_t id,
      const sycl::accessor<std::uint32_t, 1, sycl::access::mode::read_write>
          &accessor) {
  size_t num_elements_acc =
      sizeof(engine_state<oneapi::mkl::rng::device::mcg59<VecSize>>) /
      sizeof(std::uint32_t);
   accessor[id * num_elements_acc] = reinterpret_cast<std::uint32_t*>(&state.s)[0];
   accessor[id * num_elements_acc + 1] = reinterpret_cast<std::uint32_t*>(&state.s)[1];
}

template <std::int32_t VecSize>
static inline sycl::vec<std::uint64_t, VecSize>
generate(engine_state_device<oneapi::mkl::rng::device::mcg59<VecSize>,
                             device_type::generic> &state) {
  sycl::vec<std::uint64_t, VecSize> res(state.s);
  res = custom_mod(mcg59_vector_a<VecSize>::vector_a * res);
  state.s = custom_mod(mcg59_param::a * res[VecSize - 1]);
  return res;
}

template <std::int32_t VecSize>
static inline std::uint64_t
generate_single(engine_state_device<oneapi::mkl::rng::device::mcg59<VecSize>,
                                    device_type::generic> &state) {
  std::uint64_t x = state.s;
  state.s = custom_mod(mcg59_param::a * x);
  return x;
}

} // namespace mcg59_impl

// specialization for engine_accessor_base for mcg59
template <std::int32_t VecSize>
class engine_accessor_base<oneapi::mkl::rng::device::mcg59<VecSize>> {
public:
  engine_accessor_base(sycl::buffer<std::uint32_t, 1> &state_buf,
                       sycl::handler &cgh)
      : states_accessor_(state_buf, cgh) {}

  oneapi::mkl::rng::device::mcg59<VecSize> load(size_t id) const {
    oneapi::mkl::rng::device::mcg59<VecSize> engine;
    mcg59_impl::init(engine.state_.generic_state, id, states_accessor_);
    return engine;
  }

  void store(oneapi::mkl::rng::device::mcg59<VecSize> &engine,
             size_t id) const {
    mcg59_impl::store(engine.state_.generic_state, id, states_accessor_);
  }

protected:
  sycl::accessor<std::uint32_t, 1, sycl::access::mode::read_write>
      states_accessor_;
};

// specialization for engine_descriptor_base for mcg59
template <std::int32_t VecSize>
class engine_descriptor_base<oneapi::mkl::rng::device::mcg59<VecSize>> {
public:
  using engine_type = oneapi::mkl::rng::device::mcg59<VecSize>;

  using accessor_type = oneapi::mkl::rng::device::engine_accessor<
      oneapi::mkl::rng::device::mcg59<VecSize>>;

  engine_descriptor_base(sycl::queue &queue, sycl::range<1> range,
                         std::uint64_t seed, std::uint64_t offset)
      : states_buffer_(range.get(0) * sizeof(engine_state<engine_type>) /
                       sizeof(std::uint32_t)) {
    queue.submit([&](sycl::handler &cgh) {
      accessor_type states_accessor(states_buffer_, cgh);

            cgh.parallel_for<class init_kernel<engine_type>>
                (range, [=](sycl::item<1> item) {
              size_t id = item.get_id(0);
              oneapi::mkl::rng::device::mcg59<VecSize> engine(seed, offset *id);
              states_accessor.store(engine, id);
            });
    });
  }

  template <typename InitEngineFunc>
  engine_descriptor_base(sycl::queue &queue, sycl::range<1> range,
                         InitEngineFunc func)
      : states_buffer_(range.get(0) * sizeof(engine_state<engine_type>) /
                       sizeof(std::uint32_t)) {
    queue.submit([&](sycl::handler &cgh) {
      accessor_type states_accessor(states_buffer_, cgh);

            cgh.parallel_for<class init_kernel_ex<engine_type>>
                (range, [=](sycl::item<1> item) {
              size_t id = item.get_id(0);
              states_accessor.store(func(item), id);
            });
    });
  }

  accessor_type get_access(sycl::handler &cgh) {
    return accessor_type{states_buffer_, cgh};
  }

protected:
  sycl::buffer<std::uint32_t, 1> states_buffer_;
};

template <std::int32_t VecSize>
class engine_base<oneapi::mkl::rng::device::mcg59<VecSize>> {
protected:
  engine_base(std::uint32_t seed, std::uint64_t offset = 0) {
    mcg59_impl::init(this->state_.generic_state, 1, &seed, offset);
  }

  engine_base(std::uint64_t n, const std::uint32_t *seed,
              std::uint64_t offset = 0) {
    mcg59_impl::init(this->state_.generic_state, n, seed, offset);
  }

  template <typename RealType>
  auto generate(RealType a, RealType b) ->
      typename std::conditional<VecSize == 1, RealType,
                                sycl::vec<RealType, VecSize>>::type {
    sycl::vec<RealType, VecSize> res;

    RealType c = (b - a) / static_cast<RealType>(mcg59_param::m_fl);
    sycl::vec<std::uint64_t, VecSize> res_uint =
        mcg59_impl::generate(this->state_.generic_state);

    res = res_uint.template convert<RealType>() * c + a;

    return res;
  }

  auto generate() ->
      typename std::conditional<VecSize == 1, std::uint32_t,
                                sycl::vec<std::uint32_t, VecSize>>::type {
    return mcg59_impl::generate(this->state_.generic_state);
  }

  auto generate_bits() -> 
      typename std::conditional<VecSize == 1, std::uint64_t,
                                sycl::vec<std::uint64_t, VecSize>>::type {
    return mcg59_impl::generate(this->state_.generic_state);
  }

  template <typename UIntType>
  auto generate_uniform_bits() ->
      typename std::conditional<VecSize == 1, UIntType,
                                sycl::vec<UIntType, VecSize>>::type {
    if constexpr (std::is_same<UIntType, std::uint32_t>::value) {
      auto uni_res = mcg59_impl::generate(this->state_.generic_state);
  
      if constexpr (VecSize == 1) {
        return static_cast<std::uint32_t>(uni_res[0] >> 27);
      } else {
        sycl::vec<std::uint32_t, VecSize> vec_out;

        for (std::int32_t i = 0; i < VecSize; i++) {
            vec_out[i] = static_cast<std::uint32_t>(uni_res[i] >> 27);
        }

        return vec_out;
      }
    } else {
      auto uni_res1 = mcg59_impl::generate(this->state_.generic_state);
      auto uni_res2 = mcg59_impl::generate(this->state_.generic_state);

      if constexpr (VecSize == 1) {
        uni_res1 >>= 27;
        uni_res2 >>= 27;

        return (uni_res2 << 32) + uni_res1;
      } else {
        sycl::vec<std::uint64_t, VecSize> vec_out;

        for (int i = 0; i < VecSize; i++) {
          uni_res1[i] >>= 27;
          uni_res2[i] >>= 27;
        }

        if constexpr (VecSize != 3) {
          for (int i = 0; i < VecSize / 2; i++) {
            vec_out[i] = (uni_res1[2 * i + 1] << 32) + uni_res1[2 * i];
            vec_out[i + VecSize / 2] = (uni_res2[2 * i + 1] << 32) + uni_res2[2 * i];
          }
        } else {
          vec_out[0] = (uni_res1[1] << 32) + uni_res1[0];
          vec_out[1] = (uni_res2[0] << 32) + uni_res1[2];
          vec_out[2] = (uni_res2[2] << 32) + uni_res2[1];
        }

        return vec_out;
      }
    }
  }

  template <typename RealType>
  RealType generate_single(RealType a, RealType b) {
    RealType res;
    std::uint64_t res_uint;

    RealType c = (b - a) / static_cast<RealType>(mcg59_param::m_fl);

    res_uint = mcg59_impl::generate_single(this->state_.generic_state);
    res = static_cast<RealType>(res_uint) * c + a;

    return res;
  }

  auto generate_single() {
    return mcg59_impl::generate_single(this->state_.generic_state);
  }

  template <typename UIntType>
  auto generate_single_uniform_bits() {
    if constexpr (std::is_same<UIntType, std::uint32_t>::value) {
      auto uni_res = mcg59_impl::generate_single(this->state_.generic_state) >> 27;

      return static_cast<std::uint32_t>(uni_res);
    } else {
      auto uni_res1 = mcg59_impl::generate_single(this->state_.generic_state);
      auto uni_res2 = mcg59_impl::generate_single(this->state_.generic_state);

      uni_res1 >>= 27;
      uni_res2 >>= 27;

      return (uni_res2 << 32) + uni_res1;
    }
  }

  void skip_ahead(std::uint64_t num_to_skip) {
    detail::mcg59_impl::skip_ahead(this->state_.generic_state, num_to_skip);
  }

  engine_state<oneapi::mkl::rng::device::mcg59<VecSize>> state_;
};

} // namespace detail
} // namespace device
} // namespace rng
} // namespace mkl
} // namespace oneapi

#endif // _MKL_RNG_DEVICE_MCG59_IMPL_HPP_
