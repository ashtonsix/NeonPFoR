// b.hpp
#pragma once
#include "common_pipeline.hpp"
#include "common_vec.hpp"
#include <arm_neon.h>
#include <cassert>

namespace NeonPForLib {
namespace Kern {

using namespace VecNeon;
using namespace CommonPipeline;

template <class Src = Load<8>>
struct Pack_K01from08_N256 {
  static constexpr int length_v = 1;

  struct State {
    typename Src::State src;
    CallCheck<length_v> calls;
  };

  static inline void start(State& st, CommonCtx ctx) { Src::start(st.src, ctx); }

  template <int I>
  static inline uint8x16x2_t next(State& st) {
    // x0:00000000 := x0:_______0 + x1:______0_ + x2:_____0__ + x3:____0___
    //              + x4:___0____ + x5:__0_____ + x6:_0______ + x7:0_______
    st.calls.template touch<I>();

    // merge tree reduces dependency chain length (8->3)

    uint8x16x2_t x0 = Src::template next<0>(st.src);
    uint8x16x2_t x1 = Src::template next<1>(st.src);
    x0 = vsliq_n_u8_x2<1>(x0, x1); // x1->x0 (1b)

    uint8x16x2_t x2 = Src::template next<2>(st.src);
    uint8x16x2_t x3 = Src::template next<3>(st.src);
    x2 = vsliq_n_u8_x2<1>(x2, x3); // x3->x2 (1b)
    x0 = vsliq_n_u8_x2<2>(x0, x2); // x2->x0 (2b)

    uint8x16x2_t x4 = Src::template next<4>(st.src);
    uint8x16x2_t x5 = Src::template next<5>(st.src);
    x4 = vsliq_n_u8_x2<1>(x4, x5); // x5->x4 (1b)

    uint8x16x2_t x6 = Src::template next<6>(st.src);
    uint8x16x2_t x7 = Src::template next<7>(st.src);
    x6 = vsliq_n_u8_x2<1>(x6, x7); // x7->x6 (1b)
    x4 = vsliq_n_u8_x2<2>(x4, x6); // x6->x4 (2b)
    x0 = vsliq_n_u8_x2<4>(x0, x4); // x4->x0 (4b)

    return x0;
  }

  static inline void end(State& st) {
    st.calls.check();
    Src::end(st.src);
  }
};

template <class Src = Load<4>>
struct Pack_K02from08_N128 {
  static constexpr int length_v = 1;

  struct State {
    typename Src::State src;
    CallCheck<length_v> calls;
  };

  static inline void start(State& st, CommonCtx ctx) { Src::start(st.src, ctx); }

  template <int I>
  static inline uint8x16x2_t next(State& st) {
    // x0:10101010 := x0:______10 + x1:____10__ + x2:__10____ + x3:10______
    st.calls.template touch<I>();

    // merge tree reduces dependency chain length (4->2)

    uint8x16x2_t x0 = Src::template next<0>(st.src);
    uint8x16x2_t x1 = Src::template next<1>(st.src);
    x0 = vsliq_n_u8_x2<2>(x0, x1); // x1->x0 (2b)

    uint8x16x2_t x2 = Src::template next<2>(st.src);
    uint8x16x2_t x3 = Src::template next<3>(st.src);
    x2 = vsliq_n_u8_x2<2>(x2, x3); // x3->x2 (2b)

    x0 = vsliq_n_u8_x2<4>(x0, x2); // x2->x0 (4b)

    return x0;
  }

  static inline void end(State& st) {
    st.calls.check();
    Src::end(st.src);
  }
};

template <class Src = Load<8>>
struct Pack_K03from08_N256 {
  static constexpr int length_v = 3;

  struct State {
    typename Src::State src;
    uint8x16x2_t x0, x1, x2, x3, x4, x5, x6, x7;
    CallCheck<length_v> calls;
  };

  static inline void start(State& st, CommonCtx ctx) { Src::start(st.src, ctx); }

  template <int I>
  static inline uint8x16x2_t next(State& st) {
    // x0:10210210 := x0:_____210 + x1:__210___ + x2:10______
    // x3:22210210 := x3:_____210 + x4:__210___ + x2:_2______ + x5:2_______
    // x6:10210210 := x6:_____210 + x7:__210___ + x5:10______
    st.calls.template touch<I>();
    const uint8x16_t mask6 = vdupq_n_u8(0b0011'1111);
    const uint8x16_t mask7 = vdupq_n_u8(0b0111'1111);

    return static_switch<I>(
        [&] {
          st.x0 = Src::template next<0>(st.src);
          st.x1 = Src::template next<1>(st.src);
          st.x2 = Src::template next<2>(st.src);
          st.x0 = vsliq_n_u8_x2<3>(st.x0, st.x1); // x1->x0 (3b)
          st.x0 = vsliq_n_u8_x2<6>(st.x0, st.x2); // x2->x0 (2b)
          return st.x0;
        },
        [&] {
          st.x3 = Src::template next<3>(st.src);
          st.x4 = Src::template next<4>(st.src);
          st.x5 = Src::template next<5>(st.src);
          st.x2 = vbifq_n_u8_x2(vshlq_n_u8_x2<4>(st.x2), vshlq_n_u8_x2<5>(st.x5), mask7); // x5->x2 (1b)
          st.x3 = vsliq_n_u8_x2<3>(st.x3, st.x4);                                         // x4->x3 (3b)
          st.x3 = vbifq_n_u8_x2(st.x3, st.x2, mask6);                                     // x2->x3 (2b)
          return st.x3;
        },
        [&] {
          st.x6 = Src::template next<6>(st.src);
          st.x7 = Src::template next<7>(st.src);
          st.x6 = vsliq_n_u8_x2<3>(st.x6, st.x7); // x7->x6 (3b)
          st.x6 = vsliq_n_u8_x2<6>(st.x6, st.x5); // x5->x6 (2b)
          return st.x6;
        });
  }

  static inline void end(State& st) {
    st.calls.check();
    Src::end(st.src);
  }
};

template <class Src = Load<2>>
struct Pack_K04from08_N064 {
  static constexpr int length_v = 1;

  struct State {
    typename Src::State src;
    CallCheck<length_v> calls;
  };

  static inline void start(State& st, CommonCtx ctx) { Src::start(st.src, ctx); }

  template <int I>
  static inline uint8x16x2_t next(State& st) {
    // x0:32103210 := x0:____3210 + x1:3210____
    st.calls.template touch<I>();

    uint8x16x2_t x0 = Src::template next<0>(st.src);
    uint8x16x2_t x1 = Src::template next<1>(st.src);
    x0 = vsliq_n_u8_x2<4>(x0, x1); // x1->x0 (4b)
    return x0;
  }

  static inline void end(State& st) {
    st.calls.check();
    Src::end(st.src);
  }
};

template <class Src = Load<8>>
struct Pack_K05from08_N256 {
  static constexpr int length_v = 5;

  struct State {
    typename Src::State src;
    uint8x16x2_t x0, x1, x2, x3, x4, x5, x6, x7;
    CallCheck<length_v> calls;
  };

  static inline void start(State& st, CommonCtx ctx) { Src::start(st.src, ctx); }

  template <int I>
  static inline uint8x16x2_t next(State& st) {
    // x0:43243210 := x0:___43210 + x1:432_____
    // x2:43243210 := x2:___43210 + x3:432_____
    // x4:43243210 := x4:___43210 + x5:432_____
    // x6:43243210 := x6:___43210 + x7:432_____
    // x1:10101010 := x1:______10 + x3:____10__ + x5:__10____ + x7:10______
    st.calls.template touch<I>();
    const uint8x16_t mask5 = vdupq_n_u8(0b0001'1111);

    return static_switch<I>(
        [&] {
          st.x0 = Src::template next<0>(st.src);
          st.x1 = Src::template next<1>(st.src);
          st.x0 = vbifq_n_u8_x2(st.x0, vshlq_n_u8_x2<3>(st.x1), mask5); // x1->x0 (3b)
          return st.x0;
        },
        [&] {
          st.x2 = Src::template next<2>(st.src);
          st.x3 = Src::template next<3>(st.src);
          st.x2 = vbifq_n_u8_x2(st.x2, vshlq_n_u8_x2<3>(st.x3), mask5); // x3->x2 (3b)
          st.x1 = vsliq_n_u8_x2<2>(st.x1, st.x3);                       // x3->x1 (2b)
          return st.x2;
        },
        [&] {
          st.x4 = Src::template next<4>(st.src);
          st.x5 = Src::template next<5>(st.src);
          st.x4 = vbifq_n_u8_x2(st.x4, vshlq_n_u8_x2<3>(st.x5), mask5); // x5->x4 (3b)
          st.x1 = vsliq_n_u8_x2<4>(st.x1, st.x5);                       // x5->x1 (2b)
          return st.x4;
        },
        [&] {
          st.x6 = Src::template next<6>(st.src);
          st.x7 = Src::template next<7>(st.src);
          st.x6 = vbifq_n_u8_x2(st.x6, vshlq_n_u8_x2<3>(st.x7), mask5); // x7->x6 (3b)
          st.x1 = vsliq_n_u8_x2<6>(st.x1, st.x7);                       // x7->x1 (2b)
          return st.x6;
        },
        [&] {
          return st.x1;
        });
  }

  static inline void end(State& st) {
    st.calls.check();
    Src::end(st.src);
  }
};

template <class Src = Load<4>>
struct Pack_K06from08_N128 {
  static constexpr int length_v = 3;

  struct State {
    typename Src::State src;
    uint8x16x2_t x0, x1, x2, x3;
    CallCheck<length_v> calls;
  };

  static inline void start(State& st, CommonCtx ctx) { Src::start(st.src, ctx); }

  template <int I>
  static inline uint8x16x2_t next(State& st) {
    // x0:54543210 := x0:__543210 + x1:54______
    // x1:32103210 := x1:____3210 + x2:3210____
    // x3:54543210 := x3:__543210 + x2:54______
    st.calls.template touch<I>();
    const uint8x16_t mask6 = vdupq_n_u8(0b0011'1111);

    return static_switch<I>(
        [&] {
          st.x0 = Src::template next<0>(st.src);
          st.x1 = Src::template next<1>(st.src);
          st.x0 = vbifq_n_u8_x2(st.x0, vshlq_n_u8_x2<2>(st.x1), mask6); // x1->x0 (2b)
          return st.x0;
        },
        [&] {
          st.x2 = Src::template next<2>(st.src);
          st.x1 = vsliq_n_u8_x2<4>(st.x1, st.x2); // x2->x1 (4b)
          return st.x1;
        },
        [&] {
          st.x3 = Src::template next<3>(st.src);
          st.x3 = vbifq_n_u8_x2(st.x3, vshlq_n_u8_x2<2>(st.x2), mask6); // x2->x3 (2b)
          return st.x3;
        });
  }

  static inline void end(State& st) {
    st.calls.check();
    Src::end(st.src);
  }
};

template <class Src = Load<8>>
struct Pack_K07from08_N256 {
  static constexpr int length_v = 7;

  struct State {
    typename Src::State src;
    uint8x16x2_t x0, x1, x2, x3, x4, x5, x6, x7;
    CallCheck<length_v> calls;
  };

  static inline void start(State& st, CommonCtx ctx) { Src::start(st.src, ctx); }

  template <int I>
  static inline uint8x16x2_t next(State& st) {
    // x0:66543210 := x0:_6543210 + x1:6_______
    // x1:54543210 := x1:__543210 + x2:54______
    // x3:66543210 := x3:_6543210 + x2:6_______
    // x2:32103210 := x2:____3210 + x6:3210____
    // x4:66543210 := x4:_6543210 + x5:6_______
    // x5:54543210 := x5:__543210 + x6:54______
    // x7:66543210 := x7:_6543210 + x6:6_______
    uint8x16_t mask6 = vdupq_n_u8(0b0011'1111);
    uint8x16_t mask7 = vdupq_n_u8(0b0111'1111);

    return static_switch<I>(
        [&] {
          // x0:66543210 := x0:_6543210 + x1:6_______
          st.x0 = Src::template next<0>(st.src);
          st.x1 = Src::template next<1>(st.src);
          return vbifq_n_u8_x2(st.x0, vshlq_n_u8_x2<1>(st.x1), mask7); // x1->x0 (1b)
        },
        [&] {
          // x1:54543210 := x1:__543210 + x2:54______
          st.x2 = Src::template next<2>(st.src);
          return vbifq_n_u8_x2(st.x1, vshlq_n_u8_x2<2>(st.x2), mask6); // x2->x1 (2b)
        },
        [&] {
          // x3:66543210 := x3:_6543210 + x2:6_______
          st.x3 = Src::template next<3>(st.src);
          return vbifq_n_u8_x2(st.x3, vshlq_n_u8_x2<1>(st.x2), mask7); // x2->x3 (1b)
        },
        [&] {
          // x2:32103210 := x2:____3210 + x6:3210____
          st.x6 = Src::template next<6>(st.src);
          return vsliq_n_u8_x2<4>(st.x2, st.x6); // x6->x2 (4b)
        },
        [&] {
          // x4:66543210 := x4:_6543210 + x5:6_______
          st.x4 = Src::template next<4>(st.src);
          st.x5 = Src::template next<5>(st.src);
          return vbifq_n_u8_x2(st.x4, vshlq_n_u8_x2<1>(st.x5), mask7); // x5->x4 (1b)
        },
        [&] {
          // x5:54543210 := x5:__543210 + x6:54______
          return vbifq_n_u8_x2(st.x5, vshlq_n_u8_x2<2>(st.x6), mask6); // x6->x5 (2b)
        },
        [&] {
          // x7:66543210 := x7:_6543210 + x6:6_______
          st.x7 = Src::template next<7>(st.src);
          return vbifq_n_u8_x2(st.x7, vshlq_n_u8_x2<1>(st.x6), mask7); // x6->x7 (1b)
        });
  }

  static inline void end(State& st) {
    st.calls.check();
    Src::end(st.src);
  }
};

template <class Src = Load<1>>
struct Unpack_K01to08_N256 {
  static constexpr int length_v = 8;

  struct State {
    typename Src::State src;
    uint8x16x2_t x0;
    CallCheck<length_v> calls;
  };

  static inline void start(State& st, CommonCtx ctx) { Src::start(st.src, ctx); }

  template <int I>
  static inline uint8x16x2_t next(State& st) {
    // y0 := x0:_______0
    // y1 := x0:______0_
    // y2 := x0:_____0__
    // y3 := x0:____0___
    // y4 := x0:___0____
    // y5 := x0:__0_____
    // y6 := x0:_0______
    // y7 := x0:0_______
    st.calls.template touch<I>();
    const uint8x16_t mask = vdupq_n_u8(0x0000'0001);

    return static_switch<I>(
        [&] {
          st.x0 = Src::template next<0>(st.src);
          return vandq_u8_x2(st.x0, mask); // x0->y0 (1b)
        },
        [&] {
          return vandq_u8_x2(vshrq_n_u8_x2<1>(st.x0), mask); // x0->y1 (1b)
        },
        [&] {
          return vandq_u8_x2(vshrq_n_u8_x2<2>(st.x0), mask); // x0->y2 (1b)
        },
        [&] {
          return vandq_u8_x2(vshrq_n_u8_x2<3>(st.x0), mask); // x0->y3 (1b)
        },
        [&] {
          return vandq_u8_x2(vshrq_n_u8_x2<4>(st.x0), mask); // x0->y4 (1b)
        },
        [&] {
          return vandq_u8_x2(vshrq_n_u8_x2<5>(st.x0), mask); // x0->y5 (1b)
        },
        [&] {
          return vandq_u8_x2(vshrq_n_u8_x2<6>(st.x0), mask); // x0->y6 (1b)
        },
        [&] {
          return vshrq_n_u8_x2<7>(st.x0); // x0->y7 (1b)
        });
  }

  static inline void end(State& st) {
    st.calls.check();
    Src::end(st.src);
  }
};

template <class Src = Load<1>>
struct Unpack_K02to08_N128 {
  static constexpr int length_v = 4;

  struct State {
    typename Src::State src;
    uint8x16x2_t x0;
    CallCheck<length_v> calls;
  };

  static inline void start(State& st, CommonCtx ctx) { Src::start(st.src, ctx); }

  template <int I>
  static inline uint8x16x2_t next(State& st) {
    // y0 := x0:______10
    // y1 := x0:____10__
    // y2 := x0:__10____
    // y3 := x0:10______
    st.calls.template touch<I>();
    const uint8x16_t mask = vdupq_n_u8(0b0000'0011);

    return static_switch<I>(
        [&] {
          st.x0 = Src::template next<0>(st.src);
          return vandq_u8_x2(st.x0, mask); // x0->y0 (2b)
        },
        [&] {
          return vandq_u8_x2(vshrq_n_u8_x2<2>(st.x0), mask); // x0->y1 (2b)
        },
        [&] {
          return vandq_u8_x2(vshrq_n_u8_x2<4>(st.x0), mask); // x0->y2 (2b)
        },
        [&] {
          return vshrq_n_u8_x2<6>(st.x0); // x0->y3 (2b)
        });
  }

  static inline void end(State& st) {
    st.calls.check();
    Src::end(st.src);
  }
};

template <class Src = Load<3>>
struct Unpack_K03to08_N256 {
  static constexpr int length_v = 8;

  struct State {
    typename Src::State src;
    uint8x16x2_t x0, x1, x2;
    CallCheck<length_v> calls;
  };

  static inline void start(State& st, CommonCtx ctx) { Src::start(st.src, ctx); }

  template <int I>
  static inline uint8x16x2_t next(State& st) {
    // y0 := x0:_____210
    // y1 := x0:__210___
    // y2 := x0:10______ + x1:_2______
    // y3 := x1:_____210
    // y4 := x1:__210___
    // y5 := x2:10______ + x1:2_______
    // y6 := x2:_____210
    // y7 := x2:__210___
    st.calls.template touch<I>();
    const uint8x16_t mask = vdupq_n_u8(0b0000'0111);

    return static_switch<I>(
        [&] {
          st.x0 = Src::template next<0>(st.src);
          return vandq_u8_x2(st.x0, mask); // x0->y0 (3b)
        },
        [&] {
          return vandq_u8_x2(vshrq_n_u8_x2<3>(st.x0), mask); // x0->y1 (3b)
        },
        [&] {
          st.x1 = Src::template next<1>(st.src);
          auto y2 = vandq_u8_x2(vshrq_n_u8_x2<4>(st.x1), mask); // x1->y2 (1b)
          return vsriq_n_u8_x2<6>(y2, st.x0);                   // x0->y2 (2b)
        },
        [&] {
          return vandq_u8_x2(st.x1, mask); // x1->y3 (3b)
        },
        [&] {
          return vandq_u8_x2(vshrq_n_u8_x2<3>(st.x1), mask); // x1->y4 (3b)
        },
        [&] {
          st.x2 = Src::template next<2>(st.src);
          auto y5 = vshrq_n_u8_x2<5>(st.x1);  // x1->y5 (1b)
          return vsriq_n_u8_x2<6>(y5, st.x2); // x2->y5 (2b)
        },
        [&] {
          return vandq_u8_x2(st.x2, mask); // x2->y6 (3b)
        },
        [&] {
          return vandq_u8_x2(vshrq_n_u8_x2<3>(st.x2), mask); // x2->y7 (3b)
        });
  }

  static inline void end(State& st) {
    st.calls.check();
    Src::end(st.src);
  }
};

template <class Src = Load<2>>
struct Unpack_K04to08_N064 {
  static constexpr int length_v = 2;

  struct State {
    typename Src::State src;
    uint8x16x2_t x0;
    CallCheck<length_v> calls;
  };

  static inline void start(State& st, CommonCtx ctx) { Src::start(st.src, ctx); }

  template <int I>
  static inline uint8x16x2_t next(State& st) {
    // y0 := x0:____3210
    // y1 := x0:3210____
    st.calls.template touch<I>();
    const uint8x16_t mask = vdupq_n_u8(0b0000'1111);

    return static_switch<I>(
        [&] {
          st.x0 = Src::template next<0>(st.src);
          return vandq_u8_x2(st.x0, mask); // x0->y0 (4b)
        },
        [&] {
          return vshrq_n_u8_x2<4>(st.x0); // x0->y1 (4b)
        });
  }

  static inline void end(State& st) {
    st.calls.check();
    Src::end(st.src);
  }
};

template <class Src = Load<5>>
struct Unpack_K05to08_N256 {
  static constexpr int length_v = 8;

  struct State {
    typename Src::State src;
    uint8x16x2_t x0, x1, x2, x3, x4;
    CallCheck<length_v> calls;
  };

  static inline void start(State& st, CommonCtx ctx) { Src::start(st.src, ctx); }

  template <int I>
  static inline uint8x16x2_t next(State& st) {
    // y0 := x0:___43210
    // y1 := x0:432_____ + x4:______10
    // y2 := x1:___43210
    // y3 := x1:432_____ + x4:____10__
    // y4 := x2:___43210
    // y5 := x2:432_____ + x4:__10____
    // y6 := x3:___43210
    // y7 := x3:432_____ + x4:10______
    st.calls.template touch<I>();
    const uint8x16_t mask2 = vdupq_n_u8(0b0000'0011);
    const uint8x16_t mask5 = vdupq_n_u8(0b0001'1111);

    return static_switch<I>(
        [&] {
          st.x0 = Src::template next<0>(st.src);
          return vandq_u8_x2(st.x0, mask5); // x0->y0 (5b)
        },
        [&] {
          st.x4 = Src::template next<4>(st.src);
          auto y1 = vshrq_n_u8_x2<3>(st.x0);      // x0->y1 (3b)
          return vbitq_n_u8_x2(y1, st.x4, mask2); // x4->y1 (2b)
        },
        [&] {
          st.x1 = Src::template next<1>(st.src);
          return vandq_u8_x2(st.x1, mask5); // x1->y2 (5b)
        },
        [&] {
          auto y3 = vshrq_n_u8_x2<3>(st.x1);                        // x1->y3 (3b)
          return vbitq_n_u8_x2(y3, vshrq_n_u8_x2<2>(st.x4), mask2); // x4->y3 (2b)
        },
        [&] {
          st.x2 = Src::template next<2>(st.src);
          return vandq_u8_x2(st.x2, mask5); // x2->y4 (5b)
        },
        [&] {
          auto y5 = vshrq_n_u8_x2<3>(st.x2);                        // x2->y5 (3b)
          return vbitq_n_u8_x2(y5, vshrq_n_u8_x2<4>(st.x4), mask2); // x4->y5 (2b)
        },
        [&] {
          st.x3 = Src::template next<3>(st.src);
          return vandq_u8_x2(st.x3, mask5); // x3->y6 (5b)
        },
        [&] {
          auto y7 = vshrq_n_u8_x2<3>(st.x3);  // x3->y7 (3b)
          return vsriq_n_u8_x2<6>(y7, st.x4); // x4->y7 (2b)
        });
  }

  static inline void end(State& st) {
    st.calls.check();
    Src::end(st.src);
  }
};

template <class Src = Load<3>>
struct Unpack_K06to08_N128 {
  static constexpr int length_v = 4;

  struct State {
    typename Src::State src;
    uint8x16x2_t x0, x1, x2;
    CallCheck<length_v> calls;
  };

  static inline void start(State& st, CommonCtx ctx) { Src::start(st.src, ctx); }

  template <int I>
  static inline uint8x16x2_t next(State& st) {
    // y0 := x0:__543210
    // y1 := x0:54______ + x1:____3210
    // y2 := x2:54______ + x1:3210____
    // y3 := x2:__543210
    st.calls.template touch<I>();
    const uint8x16_t mask4 = vdupq_n_u8(0b0000'1111);
    const uint8x16_t mask6 = vdupq_n_u8(0b0011'1111);

    return static_switch<I>(
        [&] {
          st.x0 = Src::template next<0>(st.src);
          return vandq_u8_x2(st.x0, mask6); // x0->y0 (6b)
        },
        [&] {
          st.x1 = Src::template next<1>(st.src);
          auto y1 = vshrq_n_u8_x2<2>(st.x0);      // x0->y1 (2b)
          return vbitq_n_u8_x2(y1, st.x1, mask4); // x1->y1 (4b)
        },
        [&] {
          st.x2 = Src::template next<2>(st.src);
          auto y2 = vshrq_n_u8_x2<2>(st.x2);  // x2->y2 (2b)
          return vsriq_n_u8_x2<4>(y2, st.x1); // x1->y2 (4b)
        },
        [&] {
          return vandq_u8_x2(st.x2, mask6); // x2->y3 (6b)
        });
  }

  static inline void end(State& st) {
    st.calls.check();
    Src::end(st.src);
  }
};

template <class Src = Load<7>>
struct Unpack_K07to08_N256 {
  static constexpr int length_v = 8;

  struct State {
    typename Src::State src;
    uint8x16x2_t x0, x1, x2, x3, x4, x5, x6;
    CallCheck<length_v> calls;
  };

  static inline void start(State& st, CommonCtx ctx) { Src::start(st.src, ctx); }

  template <int I>
  static inline uint8x16x2_t next(State& st) {
    // y0 := x0:_6543210
    // y1 := x0:6_______ + x1:__543210
    // y2 := x2:6_______ + x1:54______ + x3:____3210
    // y3 := x2:_6543210
    // y4 := x4:_6543210
    // y5 := x4:6_______ + x5:__543210
    // y6 := x6:6_______ + x5:54______ + x3:3210____
    // y7 := x6:_6543210
    st.calls.template touch<I>();
    const uint8x16_t mask4 = vdupq_n_u8(0b0000'1111);
    const uint8x16_t mask6 = vdupq_n_u8(0b0011'1111);
    const uint8x16_t mask7 = vdupq_n_u8(0b0111'1111);

    // Interleave pattern order minimises register pressure (only x3 requires a long lifetime)

    return static_switch<I>(
        [&] {
          // y0 := x0:_6543210
          st.x0 = Src::template next<0>(st.src);
          return vandq_u8_x2(st.x0, mask7); // x0->y0 (7b)
        },
        [&] {
          // y1 := x0:6_______ + x1:__543210
          st.x1 = Src::template next<1>(st.src);
          uint8x16x2_t y1 = vshrq_n_u8_x2<1>(st.x0); // x0->y1 (1b)
          return vbitq_n_u8_x2(y1, st.x1, mask6);    // x1->y1 (6b)
        },
        [&] {
          // y2 := x2:6_______ + x1:54______ + x3:____3210
          st.x2 = Src::template next<2>(st.src);
          st.x3 = Src::template next<3>(st.src);
          uint8x16x2_t y2 = vshrq_n_u8_x2<1>(st.x2); // x2->y2 (1b)
          y2 = vsriq_n_u8_x2<2>(y2, st.x1);          // x1->y2 (2b)
          return vbitq_n_u8_x2(y2, st.x3, mask4);    // x3->y2 (4b)
        },
        [&] {
          // y3 := x2:_6543210
          return vandq_u8_x2(st.x2, mask7); // x2->y3 (7b)
        },
        [&] {
          // y4 := x4:_6543210
          st.x4 = Src::template next<4>(st.src);
          return vandq_u8_x2(st.x4, mask7); // x4->y4 (7b)
        },
        [&] {
          // y5 := x4:6_______ + x5:__543210
          st.x5 = Src::template next<5>(st.src);
          uint8x16x2_t y5 = vshrq_n_u8_x2<1>(st.x4); // x4->y5 (1b)
          return vbitq_n_u8_x2(y5, st.x5, mask6);    // x5->y5 (6b)
        },
        [&] {
          // y6 := x6:6_______ + x5:54______ + x3:3210____
          st.x6 = Src::template next<6>(st.src);
          uint8x16x2_t y6 = vshrq_n_u8_x2<1>(st.x6); // x6->y6 (1b)
          y6 = vsriq_n_u8_x2<2>(y6, st.x5);          // x5->y6 (2b)
          return vsriq_n_u8_x2<4>(y6, st.x3);        // x3->y6 (4b)
        },
        [&] {
          // y7 := x6:_6543210
          return vandq_u8_x2(st.x6, mask7); // x6->y7 (7b)
        });
  }

  static inline void end(State& st) {
    st.calls.check();
    Src::end(st.src);
  }
};

} // namespace Kern

// -------------------------------------------------------------------------------------------------
//  Public wrappers
// -------------------------------------------------------------------------------------------------

using namespace Kern;

void pack(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, uint32_t kU, uint32_t kP, uint32_t n) {
  __builtin_assume((kU == 8u || kU == 16u || kU == 32u || kU == 64u) && (1u <= kP && kP <= kU) && (n % 256u == 0));
  in = (const uint8_t*)__builtin_assume_aligned(in, 16);
  out = (uint8_t*)__builtin_assume_aligned(out, 16);

  // clang-format off
  switch (kU) {
  case 8:
    switch (kP) {
    case 1: loop<Store<Pack_K01from08_N256<>>, 256,  32, 1>(in, out, n / 256); break;
    case 2: loop<Store<Pack_K02from08_N128<>>, 128,  32, 2>(in, out, n / 128); break;
    case 3: loop<Store<Pack_K03from08_N256<>>, 256,  96, 1>(in, out, n / 256); break;
    case 4: loop<Store<Pack_K04from08_N064<>>,  64,  32, 4>(in, out, n /  64); break;
    case 5: loop<Store<Pack_K05from08_N256<>>, 256, 160, 1>(in, out, n / 256); break;
    case 6: loop<Store<Pack_K06from08_N128<>>, 128,  96, 2>(in, out, n / 128); break;
    case 7: loop<Store<Pack_K07from08_N256<>>, 256, 224, 1>(in, out, n / 256); break;
    case 8: loop<Store<Load<1>>, 32, 32, 8>(in, out, n / 32); break;
    default: __builtin_unreachable();
    } break;
  default: __builtin_unreachable();
  }
  // clang-format on

  // Memory barrier prevents LLVM's common subexpression elimination,
  // which interferes with other optimisation passes if applied
  asm volatile("" ::: "memory");
}

void unpack(const uint8_t* in, uint8_t* out, uint32_t kU, uint32_t kP, uint32_t n) {
  __builtin_assume((kU == 8 || kU == 16 || kU == 32 || kU == 64) && (1u <= kP && kP <= kU) && (n % 256u == 0) &&
                   "pack(): bad arguments");

  // clang-format off
  switch (kU) {
  case 8:
    switch (kP) {
    case 1: loop<Store<Unpack_K01to08_N256<>>,  32, 256, 1>(in, out, n / 256); break;
    case 2: loop<Store<Unpack_K02to08_N128<>>,  32, 128, 2>(in, out, n / 128); break;
    case 3: loop<Store<Unpack_K03to08_N256<>>,  96, 256, 1>(in, out, n / 256); break;
    case 4: loop<Store<Unpack_K04to08_N064<>>,  32,  64, 4>(in, out, n /  64); break;
    case 5: loop<Store<Unpack_K05to08_N256<>>, 160, 256, 1>(in, out, n / 256); break;
    case 6: loop<Store<Unpack_K06to08_N128<>>,  96, 128, 2>(in, out, n / 128); break;
    case 7: loop<Store<Unpack_K07to08_N256<>>, 224, 256, 1>(in, out, n / 256); break;
    case 8: loop<Store<Load<1>>, 32, 32, 8>(in, out, n / 32); break;
    default: __builtin_unreachable();
    } break;
  default: __builtin_unreachable();
  }
  // clang-format on

  asm volatile("" ::: "memory");
}

} // namespace NeonPForLib