#pragma once
#include "common_pipeline.hpp"
#include "common_vec.hpp"
#include <cassert>
#include <cstddef>
#include <utility>

namespace CommonKernel {
using namespace VecNeon;
using namespace CommonPipeline;

template <Port OutT, class Src>
struct LoadImpl {
  struct State {
    const uint8_t* in;
    CallCheck<OutT.tile_c> calls;
  };

  static inline void start(State& st, CommonCtx ctx) { st.in = ctx.in; }
  template <int I>
  static inline uint8x16x2_t next(State& st) {
    st.calls.template touch<I, true>();
    return vldpq_u8<I * 32, true>(st.in);
  }
  static inline void end(State& st) { st.calls.check(); }
};

struct Load {
  static constexpr auto in_t = Port{0};
  static constexpr auto out_t = Port{1};

  static constexpr int ctx_in_incr = 32;

  template <Port OutT, Port InT, class Src>
  using impl = LoadImpl<OutT, Src>;
};

// Store

template <Port InT, class Src>
struct StoreImpl {
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
    static_assert(false, "Store::next should never be instantiated");
  }

private:
  template <std::size_t... I>
  static inline void storeImpl(State& st, std::index_sequence<I...>) {
    (vstpq_u8<I * 32, false>(st.out, Src::template next<I>(st.src)), ...);
  }

public:
  static inline void end(State& st) {
    storeImpl(st, std::make_index_sequence<InT.tile_c>{});
    Src::end(st.src);
  }
};

struct Store {
  static constexpr auto in_t = Port{1};
  static constexpr auto out_t = Port{0};

  static constexpr int ctx_out_incr = 32;

  template <Port OutT, Port InT, class Src>
  using impl = StoreImpl<InT, Src>;
};

// Narrow

template <Port OutT, Port InT, class Src, int InK, int OutK>
struct NarrowImpl {
  struct State {
    typename Src::State src;
    CallCheck<OutT.tile_c> calls;
  };

  static inline void start(State& st, CommonCtx ctx) { Src::start(st.src, ctx); }

  template <int I>
  static inline uint8x16x2_t next(State& st) {
    st.calls.template touch<I>();
    constexpr auto p = (I * (InK / OutK));

    if constexpr (InK == 16) {
      auto n0 = Src::template next<p + 0>(st.src);
      auto n1 = Src::template next<p + 1>(st.src);
      return vxtnq_u16_x2(n0, n1);

    } else if constexpr (InK == 32) {
      auto n0 = Src::template next<p + 0>(st.src);
      auto n1 = Src::template next<p + 1>(st.src);
      auto x0 = vxtnq_u32_x2(n0, n1);

      if constexpr (OutK == 16) {
        return x0;
      }

      auto n2 = Src::template next<p + 2>(st.src);
      auto n3 = Src::template next<p + 3>(st.src);
      auto x1 = vxtnq_u32_x2(n2, n3);
      return vxtnq_u16_x2(x0, x1);

    } else /* InK == 64 */ {
      auto n0 = Src::template next<p + 0>(st.src);
      auto n1 = Src::template next<p + 1>(st.src);
      auto x0 = vxtnq_u64_x2(n0, n1);

      if constexpr (OutK == 32) {
        return x0;
      }

      auto n2 = Src::template next<p + 2>(st.src);
      auto n3 = Src::template next<p + 3>(st.src);
      auto x1 = vxtnq_u64_x2(n2, n3);
      auto x01 = vxtnq_u32_x2(x0, x1);

      if constexpr (OutK == 16) {
        return x01;
      }

      auto n4 = Src::template next<p + 4>(st.src);
      auto n5 = Src::template next<p + 5>(st.src);
      auto x2 = vxtnq_u64_x2(n4, n5);

      auto n6 = Src::template next<p + 6>(st.src);
      auto n7 = Src::template next<p + 7>(st.src);
      auto x3 = vxtnq_u64_x2(n6, n7);

      auto x23 = vxtnq_u32_x2(x2, x3);
      return vxtnq_u16_x2(x01, x23);
    }
  }

  static inline void end(State& st) {
    st.calls.check();
    Src::end(st.src);
  }
};

template <int InK, int OutK>
struct Narrow {
  static_assert(OutK == 8 || OutK == 16 || OutK == 32, "OutK must be 8, 16, or 32");
  static_assert(InK == 16 || InK == 32 || InK == 64, "InK must be 16, 32, or 64");
  static_assert(InK > OutK, "InK must be greater than OutK");

  static constexpr auto in_t = Port{InK / OutK};
  static constexpr auto out_t = Port{1};

  template <Port OutT, Port InT, class Src>
  using impl = NarrowImpl<OutT, InT, Src, InK, OutK>;
};

template <Port OutT, Port InT, class Src>
struct IdentityImpl {
  struct State {
    typename Src::State src;
    CallCheck<OutT.tile_c> calls;
  };

  static inline void start(State& st, CommonCtx ctx) { Src::start(st.src, ctx); }

  template <int I>
  static inline uint8x16x2_t next(State& st) {
    st.calls.template touch<I>();
    return Src::template next<I>(st.src);
  }

  static inline void end(State& st) {
    st.calls.check();
    Src::end(st.src);
  }
};

struct Identity {
  static constexpr auto in_t = Port{1};
  static constexpr auto out_t = Port{1};

  template <Port OutT, Port InT, class Src>
  using impl = IdentityImpl<OutT, InT, Src>;
};

} // namespace CommonKernel