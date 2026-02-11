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

#ifndef _MKL_RNG_DEVICE_MCG31M1_IMPL_HPP_
#define _MKL_RNG_DEVICE_MCG31M1_IMPL_HPP_

namespace oneapi {
namespace mkl {
namespace rng {
namespace device {

template <std::int32_t VecSize = 1> class mcg31m1;

namespace detail {

template <std::uint64_t VecSize>
constexpr sycl::vec<std::uint64_t, VecSize> select_vector_a_mcg31m1() {
  if constexpr (VecSize == 1)
    return sycl::vec<std::uint64_t, 1>(UINT64_C(1));
  else if constexpr (VecSize == 2)
    return sycl::vec<std::uint64_t, 2>({UINT64_C(1), UINT64_C(1132489760)});
  else if constexpr (VecSize == 3)
    return sycl::vec<std::uint64_t, 3>({UINT64_C(1), UINT64_C(1132489760), UINT64_C(826537482)});
  else if constexpr (VecSize == 4)
    return sycl::vec<std::uint64_t, 4>(
        {UINT64_C(1), UINT64_C(1132489760), UINT64_C(826537482), UINT64_C(289798557)});
  else if constexpr (VecSize == 8)
    return sycl::vec<std::uint64_t, 8>({UINT64_C(1), UINT64_C(1132489760), UINT64_C(826537482),
                                        UINT64_C(289798557), UINT64_C(480863449), UINT64_C(1381340036),
                                        UINT64_C(1582925527), UINT64_C(1918178478)});
  else
    return sycl::vec<std::uint64_t, 16>(
        {UINT64_C(1), UINT64_C(1132489760), UINT64_C(826537482), UINT64_C(289798557), UINT64_C(480863449), UINT64_C(1381340036),
         UINT64_C(1582925527), UINT64_C(1918178478), UINT64_C(1286028348), UINT64_C(482167044), UINT64_C(262060616),
         UINT64_C(1856662125), UINT64_C(839877947), UINT64_C(1997268203), UINT64_C(458714024), UINT64_C(650347998)});
}

template <std::uint64_t VecSize> struct mcg31m1_vector_a {
  static constexpr sycl::vec<std::uint64_t, VecSize> vector_a =
      select_vector_a_mcg31m1<VecSize>(); // powers of a
};

struct mcg31m1_param {
  static constexpr std::uint32_t a = 1132489760;
  static constexpr std::uint64_t m_64 = 0x000000007FFFFFFF; // 2^31 - 1
  static constexpr double m_fl = 2147483647.0;              // 2^31 - 1
  static constexpr std::uint64_t bits = 31;
};

// specialization of engine_state_device for mcg31m1 engine for generic DPC++
// version
template <std::int32_t VecSize>
struct engine_state_device<oneapi::mkl::rng::device::mcg31m1<VecSize>,
                           device_type::generic> {
  std::uint32_t s;
};

// specialization of engine_state for mcg31m1
template <std::int32_t VecSize>
union engine_state<oneapi::mkl::rng::device::mcg31m1<VecSize>> {
  engine_state_device<oneapi::mkl::rng::device::mcg31m1<VecSize>,
                      device_type::generic>
      generic_state;
};

namespace mcg31m1_impl {

// Improved modulus x % (2^31 - 1) operation (possible to do for divisor (2^N
// -1), but MCG31M1 needs only 2^31 - 1) if we want to do x % (2^N -1) we can
// find out that: x = A + B * 2^N, where A = x % 2^N = x & 00..01..11 (binary)
// where quantity of 1 is N, B = x / 2^N = x >> N also x = A + B * (2^N - 1 + 1)
// = (A + B) + B * (2^N - 1), but (A + B) may be greater than (2^N - 1), that's
// why we put x1 = A + B = A' + B' * 2^N = ... until new (A + B) < (2^N - 1) for
// MCG31m1 N = 31
template <typename T> static inline T custom_mod(std::uint64_t x) {
  std::uint64_t b = x >> mcg31m1_param::bits;
  std::uint64_t a = x & mcg31m1_param::m_64;
  x = a + b;
  b = x >> mcg31m1_param::bits;
  a = x & mcg31m1_param::m_64;
  return static_cast<T>(a + b);
}

template <std::int32_t VecSize>
static inline sycl::vec<std::uint32_t, VecSize>
custom_mod(const sycl::vec<std::uint64_t, VecSize> &x) {
  sycl::vec<std::uint64_t, VecSize> b = x >> mcg31m1_param::bits;
  sycl::vec<std::uint64_t, VecSize> a = x & mcg31m1_param::m_64;
  sycl::vec<std::uint64_t, VecSize> res = a + b;
  b = res >> mcg31m1_param::bits;
  a = res & mcg31m1_param::m_64;
  res = a + b;
  return res.template convert<std::uint32_t>();
}

static inline std::uint64_t power(std::uint64_t a, std::uint64_t n) {
  std::uint64_t a2;
  // initialize result by 1 for recurrence
  std::uint32_t result = 1;

  if (n == 0) {
    // return (a^0)%m = 1
    return std::uint64_t{1};
  }

  // Recurrence loop
  do {
    // For each odd n
    if (n & 1) {
      a2 = static_cast<std::uint64_t>(result) * a;
      result = custom_mod<std::uint32_t>(a2);
    }
    // n := n/2
    n >>= 1;

    a2 = a * a;
    a = custom_mod<std::uint64_t>(a2);
  } while (n);

  return static_cast<std::uint64_t>(result);
}

template <std::int32_t VecSize>
static inline void
skip_ahead(engine_state_device<oneapi::mkl::rng::device::mcg31m1<VecSize>,
                               device_type::generic> &state,
           std::uint64_t num_to_skip) {
  std::uint64_t loc_A =
      power(static_cast<std::uint64_t>(mcg31m1_param::a), num_to_skip);
  state.s =
      custom_mod<std::uint32_t>(loc_A * static_cast<std::uint64_t>(state.s));
}

template <std::int32_t VecSize>
static inline void
init(engine_state_device<oneapi::mkl::rng::device::mcg31m1<VecSize>,
                         device_type::generic> &state,
     std::uint64_t n, const std::uint64_t *seed_ptr, std::uint64_t offset) {
  if (n == 0)
    state.s = 1;
  else {
    state.s = custom_mod<std::uint32_t>(seed_ptr[0]);
    if (state.s == 0)
      state.s = 1;
  }
  skip_ahead(state, offset);
}

template <std::int32_t VecSize>
static inline void
init(engine_state_device<oneapi::mkl::rng::device::mcg31m1<VecSize>,
                         device_type::generic> &state,
     size_t id,
     const sycl::accessor<std::uint32_t, 1, sycl::access::mode::read_write>
         &accessor) {
  size_t num_elements_acc =
      sizeof(engine_state<oneapi::mkl::rng::device::mcg31m1<VecSize>>) /
      sizeof(std::uint32_t);
  state.s = accessor[id * num_elements_acc];
}

template <std::int32_t VecSize>
static inline void
store(engine_state_device<oneapi::mkl::rng::device::mcg31m1<VecSize>,
                          device_type::generic> &state,
      size_t id,
      const sycl::accessor<std::uint32_t, 1, sycl::access::mode::read_write>
          &accessor) {
  size_t num_elements_acc =
      sizeof(engine_state<oneapi::mkl::rng::device::mcg31m1<VecSize>>) /
      sizeof(std::uint32_t);
  accessor[id * num_elements_acc] = state.s;
}

template <std::int32_t VecSize>
static inline sycl::vec<std::uint32_t, VecSize>
generate(engine_state_device<oneapi::mkl::rng::device::mcg31m1<VecSize>,
                             device_type::generic> &state) {
  sycl::vec<std::uint64_t, VecSize> x(state.s);
  sycl::vec<std::uint32_t, VecSize> res =
      custom_mod(mcg31m1_vector_a<VecSize>::vector_a * x);
  state.s = custom_mod<std::uint32_t>(
      mcg31m1_param::a * static_cast<std::uint64_t>(res[VecSize - 1]));
  return res;
}

template <std::int32_t VecSize>
static inline std::uint32_t
generate_single(engine_state_device<oneapi::mkl::rng::device::mcg31m1<VecSize>,
                                    device_type::generic> &state) {
  std::uint32_t x = state.s;
  state.s = custom_mod<std::uint32_t>(mcg31m1_param::a *
                                      static_cast<std::uint64_t>(state.s));
  return x;
}

} // namespace mcg31m1_impl

// specialization for engine_accessor_base for mcg31m1
template <std::int32_t VecSize>
class engine_accessor_base<oneapi::mkl::rng::device::mcg31m1<VecSize>> {
public:
  engine_accessor_base(sycl::buffer<std::uint32_t, 1> &state_buf,
                       sycl::handler &cgh)
      : states_accessor_(state_buf, cgh) {}

  oneapi::mkl::rng::device::mcg31m1<VecSize> load(size_t id) const {
    oneapi::mkl::rng::device::mcg31m1<VecSize> engine;
    mcg31m1_impl::init(engine.state_.generic_state, id, states_accessor_);
    return engine;
  }

  void store(oneapi::mkl::rng::device::mcg31m1<VecSize> &engine,
             size_t id) const {
    mcg31m1_impl::store(engine.state_.generic_state, id, states_accessor_);
  }

protected:
  sycl::accessor<std::uint32_t, 1, sycl::access::mode::read_write>
      states_accessor_;
};

// specialization for engine_descriptor_base for mcg31m1
template <std::int32_t VecSize>
class engine_descriptor_base<oneapi::mkl::rng::device::mcg31m1<VecSize>> {
public:
  using engine_type = oneapi::mkl::rng::device::mcg31m1<VecSize>;

  using accessor_type = oneapi::mkl::rng::device::engine_accessor<
      oneapi::mkl::rng::device::mcg31m1<VecSize>>;

  engine_descriptor_base(sycl::queue &queue, sycl::range<1> range,
                         std::uint64_t seed, std::uint64_t offset)
      : states_buffer_(range.get(0) * sizeof(engine_state<engine_type>) /
                       sizeof(std::uint32_t)) {
    queue.submit([&](sycl::handler &cgh) {
      accessor_type states_accessor(states_buffer_, cgh);

            cgh.parallel_for<class init_kernel<engine_type>>
                (range, [=](sycl::item<1> item) {
              size_t id = item.get_id(0);
              oneapi::mkl::rng::device::mcg31m1<VecSize> engine(seed,
                                                                offset *id);
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
class engine_base<oneapi::mkl::rng::device::mcg31m1<VecSize>> {
protected:
  engine_base(std::uint64_t seed, std::uint64_t offset = 0) {
    mcg31m1_impl::init(this->state_.generic_state, 1, &seed, offset);
  }

  engine_base(std::uint64_t n, const std::uint64_t *seed,
              std::uint64_t offset = 0) {
    mcg31m1_impl::init(this->state_.generic_state, n, seed, offset);
  }

  template <typename RealType>
  auto generate(RealType a, RealType b) ->
      typename std::conditional<VecSize == 1, RealType,
                                sycl::vec<RealType, VecSize>>::type {
    sycl::vec<RealType, VecSize> res;
    sycl::vec<std::uint32_t, VecSize> res_uint;

    RealType c = (b - a) / static_cast<RealType>(mcg31m1_param::m_fl);

    res_uint = mcg31m1_impl::generate(this->state_.generic_state);

    res = res_uint.template convert<RealType>() * c + a;

    return res;
  }

  auto generate() ->
      typename std::conditional<VecSize == 1, std::uint32_t,
                                sycl::vec<std::uint32_t, VecSize>>::type {
    return mcg31m1_impl::generate(this->state_.generic_state);
  }

  template <typename RealType>
  RealType generate_single(RealType a, RealType b) {
    RealType res;
    std::uint32_t res_uint;

    RealType c = (b - a) / static_cast<RealType>(mcg31m1_param::m_fl);

    res_uint = mcg31m1_impl::generate_single(this->state_.generic_state);

    res = static_cast<RealType>(res_uint) * c + a;
    return res;
  }

  std::uint32_t generate_single() {
    return mcg31m1_impl::generate_single(this->state_.generic_state);
  }

  void skip_ahead(std::uint64_t num_to_skip) {
    detail::mcg31m1_impl::skip_ahead(this->state_.generic_state, num_to_skip);
  }

  engine_state<oneapi::mkl::rng::device::mcg31m1<VecSize>> state_;
};

} // namespace detail
} // namespace device
} // namespace rng
} // namespace mkl
} // namespace oneapi

#endif // _MKL_RNG_DEVICE_MCG31M1_IMPL_HPP_
