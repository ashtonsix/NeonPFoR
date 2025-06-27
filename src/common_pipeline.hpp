#pragma once
#include "common_vec.hpp"
#include <cassert>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

using namespace VecNeon;

namespace CommonPipeline {

template <class...>
inline constexpr bool dependent_false_v = false;

// branch‑free, non‑recursive compile‑time switch (2a + 2b)
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

// -------------------------------------------------------------------------------------------------
//  Common context (I/O base pointers)
// -------------------------------------------------------------------------------------------------

struct CommonCtx {
  const uint8_t* in;
  uint8_t* out;
};

// -------------------------------------------------------------------------------------------------
//  Kernel runner – fully unrolled, leaf‑function
// -------------------------------------------------------------------------------------------------

template <class Stage>
struct Kernel {
  static inline void run(CommonCtx ctx) {
    typename Stage::State st{};
    Stage::start(st, ctx);
    if constexpr (Stage::length_v > 0)
      runSteps(st, std::make_index_sequence<Stage::length_v>{});
    Stage::end(st);
  }

private:
  template <std::size_t... Is>
  static inline void runSteps(typename Stage::State& st, std::index_sequence<Is...>) {
    (Stage::template next<Is>(st), ...);
  }
};

// -------------------------------------------------------------------------------------------------
//  Load stage – reads N × 32‑byte tiles
// -------------------------------------------------------------------------------------------------

template <int N>
struct Load {
  static constexpr int length_v = N;

  struct State {
    const uint8_t* in;
    CallCheck<N> calls;
  };

  static inline void start(State& st, CommonCtx ctx) { st.in = ctx.in; }
  template <int I>
  static inline uint8x16x2_t next(State& st) {
    st.calls.template touch<I, true>();
    return vldpq_u8<I * 32, true>(st.in);
  }
  static inline void end(State& st) { st.calls.check(); }
};

// -------------------------------------------------------------------------------------------------
//  Store stage – writes the tiles it receives
// -------------------------------------------------------------------------------------------------

template <class Src>
struct Store {
  static constexpr int length_v = 0;

  struct State {
    typename Src::State src;
    uint8_t* out;
  };

  static inline void start(State& st, CommonCtx ctx) {
    st.out = ctx.out;
    Src::start(st.src, ctx);
  }

  template <int I>
  static inline void next(State&) {
    static_assert(dependent_false_v<std::integral_constant<int, I>>, "Store::next should never be instantiated");
  }

private:
  template <std::size_t... Is>
  static inline void storeImpl(State& st, std::index_sequence<Is...>) {
    (vstpq_u8<Is * 32, false>(st.out, Src::template next<Is>(st.src)), ...);
  }

public:
  static inline void end(State& st) {
    storeImpl(st, std::make_index_sequence<Src::length_v>{});
    Src::end(st.src);
  }
};

#define NEONPFOR_STRINGIFY_PRAGMA(x) #x
#define NEONPFOR_APPLY_PRAGMA(directive) _Pragma(NEONPFOR_STRINGIFY_PRAGMA(directive))

template <class Pipeline, int InIncr, int OutIncr, int UNROLL = 1>
[[gnu::always_inline]]
inline void loop(const uint8_t* __restrict in, uint8_t* __restrict out, std::size_t n_iters) {
  in = static_cast<const uint8_t*>(__builtin_assume_aligned(in, 16));
  out = static_cast<uint8_t*>(__builtin_assume_aligned(out, 16));

  NEONPFOR_APPLY_PRAGMA(clang loop unroll_count(UNROLL))
  for (; n_iters; --n_iters, in += InIncr, out += OutIncr) {
    Kernel<Pipeline>::run({in, out});
  }
}

#undef NEONPFOR_STRINGIFY_PRAGMA
#undef NEONPFOR_APPLY_PRAGMA

} // namespace CommonPipeline