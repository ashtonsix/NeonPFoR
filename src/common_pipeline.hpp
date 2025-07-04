#pragma once
#include <arm_neon.h>
#include <array>
#include <cassert>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace CommonPipeline {
// branch‑free, non‑recursive compile‑time switch
template <std::size_t I, class... Fs>
decltype(auto) static_switch(Fs&&... fs) {
  static_assert(I < sizeof...(Fs), "static_switch: index out of bounds");

  using tuple_t = std::tuple<std::decay_t<Fs>...>;
  using ret0_t = std::invoke_result_t<std::tuple_element_t<0, tuple_t>>;
  static_assert((std::is_same_v<ret0_t, std::invoke_result_t<Fs>> && ...),
                "static_switch: all arms must return the same type");

  return std::get<I>(std::forward_as_tuple(std::forward<Fs>(fs)...))();
}

#if defined(NDEBUG)
constexpr bool kEnableCallCheck = false;
#else
constexpr bool kEnableCallCheck = false;
#endif

// Debug‑only call‑check to verify next<I>() calling behaviour
// Parametrised call‑check: N = expected number of tiles
template <int N, bool Enabled = true>
struct CallCheckImpl {
  static_assert(Enabled, "internal helper misuse");
  static_assert(N > 0 && N <= 64, "CallCheck supports 1..64 calls");
  std::size_t mask = 0;

  template <int I, bool AllowOoO = false>
  void touch() {
    static_assert(I < N, "next<I>() index out of range for this stage");
    constexpr std::size_t bit = 1ull << I;
    // 1. forbid duplicates
    assert((mask & bit) == 0 && "duplicate next<I>() call");
    // 2. enforce strict ascending order, unless AllowOoO
    if constexpr (!AllowOoO) {
      assert(mask == (bit - 1) && "out-of-order or skipped next<I>()");
    }
    mask |= bit;
  }

  void check() const {
    // 3. enforce completion
    constexpr std::size_t full = N == 64 ? ~0ull : ((1ull << N) - 1);
    assert(mask == full && "next<I>() not called exactly N times");
  }
};

template <int N>
struct CallCheckImpl<N, /*Enabled=*/false> {
  template <int I, bool AllowOoO = false>
  constexpr void touch() noexcept {} // inline, does nothing
  constexpr void check() const noexcept {}
};

template <int N>
using CallCheck = CallCheckImpl<N, kEnableCallCheck>;

struct CommonCtx {
  const uint8_t* in;
  uint8_t* out;
};

struct Port {
  int tile_c;
  int width = 1;
};

struct PortPair {
  Port in;
  Port out;
};

// Adjust scale factors for each kernel to make adjacent ports compatible
template <std::size_t N>
consteval std::array<PortPair, N> rewrite_ports(const std::array<PortPair, N>& kernel_ports) {
  std::array<int, N> scales{};
  std::fill(scales.begin(), scales.end(), 1);

  // Compute scale factors for each kernel to make adjacent ports compatible
  for (std::size_t i = 0; i < N - 1; ++i) {
    int current_out_port = kernel_ports[i].out.tile_c * scales[i];
    int next_in_port = kernel_ports[i + 1].in.tile_c * scales[i + 1];

    if (current_out_port != next_in_port) {
      int min_port = std::min(current_out_port, next_in_port);
      int max_port = std::max(current_out_port, next_in_port);

      if (max_port % min_port != 0) {
        throw "Kernel port mismatch: scale factor must be integer";
      }

      int scale_factor = max_port / min_port;

      if (current_out_port < next_in_port) {
        // Scale current kernel and backtrack to scale all previous kernels
        for (std::size_t j = 0; j <= i; ++j) {
          scales[j] *= scale_factor;
        }
      } else {
        scales[i + 1] *= scale_factor;
      }
    }
  }

  // Add computed scales to ports
  std::array<PortPair, N> scaled_ports = kernel_ports;
  for (std::size_t i = 0; i < N; ++i) {
    scaled_ports[i].in = Port{kernel_ports[i].in.tile_c, scales[i]};
    scaled_ports[i].out = Port{kernel_ports[i].out.tile_c, scales[i]};
  }
  return scaled_ports;
}

// Extract and scale a value from an array where at most one element is non-zero, so it patches port width
template <std::size_t N>
consteval int extract_one_and_scale_with_ports(const std::array<int, N>& values, const std::array<PortPair, N>& ports) {
  int result = 0;
  int non_zero_count = 0;

  for (std::size_t i = 0; i < N; ++i) {
    if (values[i] != 0) {
      ++non_zero_count;
      result = values[i] * ports[i].in.width;
    }
  }

  if (non_zero_count > 1) {
    throw "At most one element can be non-zero";
  }

  return result;
}

// Series combinator that chains kernels with automatic port rewriting
template <class... Ks>
struct SeriesBase {
  static_assert(sizeof...(Ks) > 0, "Series must have at least one kernel");

private:
  static constexpr auto original_ports = std::array<PortPair, sizeof...(Ks)>{PortPair{Ks::in_t, Ks::out_t}...};
  static constexpr auto updated_ports = rewrite_ports(original_ports);

  static constexpr auto ctx_in_incrs =
      std::array<int, sizeof...(Ks)>{(requires { Ks::ctx_in_incr; } ? Ks::ctx_in_incr : 0)...};
  static constexpr auto ctx_out_incrs =
      std::array<int, sizeof...(Ks)>{(requires { Ks::ctx_out_incr; } ? Ks::ctx_out_incr : 0)...};

  template <std::size_t I, class Src, class... RestKs>
  struct SeriesImpl;

  template <std::size_t I, class Src>
  struct SeriesImpl<I, Src> {
    using type = Src;
  };

  template <std::size_t I, class Src, class K, class... RestKs>
  struct SeriesImpl<I, Src, K, RestKs...> {
    using current_impl = typename K::template impl<updated_ports[I].out, updated_ports[I].in, Src>;
    using type = typename SeriesImpl<I + 1, current_impl, RestKs...>::type;
  };

public:
  static constexpr auto in_t = updated_ports[0].in;
  static constexpr auto out_t = updated_ports[sizeof...(Ks) - 1].out;

  static constexpr int ctx_in_incr = extract_one_and_scale_with_ports(ctx_in_incrs, updated_ports);
  static constexpr int ctx_out_incr = extract_one_and_scale_with_ports(ctx_out_incrs, updated_ports);

  template <Port OutT, Port InT, class Src>
  using impl = typename SeriesImpl<0, Src, Ks...>::type;
};

// Public interface flattens nested Series.
//
// Example:
//   Series<
//     Load,
//     Series<Narrow<32, 8>, Pack<8, 6>>,
//     Store
//   >
// becomes:
//   SeriesBase<
//     Load,
//     Narrow<32, 8>,
//     Pack<8, 6>,
//     Store
//   >
template <class... Ks>
struct Series;
template <class... Ks>
struct Series : SeriesBase<Ks...> {};
template <class... Nested, class... Rest>
struct Series<Series<Nested...>, Rest...> : Series<Nested..., Rest...> {};
template <class First, class... Nested, class... Rest>
struct Series<First, Series<Nested...>, Rest...> : Series<First, Nested..., Rest...> {};

#define NEONPFOR_STRINGIFY_PRAGMA(x) #x
#define NEONPFOR_APPLY_PRAGMA(directive) _Pragma(NEONPFOR_STRINGIFY_PRAGMA(directive))

struct Noop {
  struct State {};
  static inline void start(State&, CommonCtx) {}
  template <int I>
  static inline void next(State&) {}
  static inline void end(State&) {}
};

template <class Kernel, int UnrollCount = 1>
struct LoopDriver {
private:
  static constexpr int InIncr = requires { Kernel::ctx_in_incr; } ? Kernel::ctx_in_incr : 0;
  static constexpr int OutIncr = requires { Kernel::ctx_out_incr; } ? Kernel::ctx_out_incr : 0;

  using KernelImpl = typename Kernel::template impl<Kernel::out_t, Kernel::in_t, Noop>;

public:
  static inline void run(const uint8_t* __restrict in, uint8_t* __restrict out, std::size_t n_iters) {
    in = static_cast<const uint8_t*>(__builtin_assume_aligned(in, 16));
    out = static_cast<uint8_t*>(__builtin_assume_aligned(out, 16));

    NEONPFOR_APPLY_PRAGMA(clang loop unroll_count(UnrollCount))
    for (; n_iters; --n_iters, in += InIncr, out += OutIncr) {
      typename KernelImpl::State state;
      KernelImpl::start(state, {in, out});
      [&]<std::size_t... I>(std::index_sequence<I...>) {
        (KernelImpl::template next<I>(state), ...);
      }(std::make_index_sequence<Kernel::out_t.tile_c>{});
      KernelImpl::end(state);
    }
  }
};

#undef NEONPFOR_APPLY_PRAGMA
#undef NEONPFOR_STRINGIFY_PRAGMA

} // namespace CommonPipeline