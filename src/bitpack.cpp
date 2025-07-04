#pragma once
#include "common_kernel.hpp"
#include "common_pipeline.hpp"
#include "common_vec.hpp"
#include <arm_neon.h>
#include <cassert>

namespace NeonPForLib {
namespace Kern {

using namespace VecNeon;
using namespace CommonPipeline;
using namespace CommonKernel;

/**
 * @brief Compute tile index mapping for a packed block schema.
 *
 * Generates an index mapping used to interleave PackSplit/PackMSB iterators
 * based on a block-level schema describing the byte layout of packed values.
 *
 * Each schema entry is interpreted as follows:
 *   - A `0` indicates a block of multi-byte tails. Each such block contributes
 *     `BytesPerTail` tiles (even-numbered).
 *   - A `1` indicates a block of MSBs (most significant bytes). Each such block
 *     contributes 1 tile (odd-numbered).
 *
 * Tile indices are assigned as follows:
 *   - Tail tiles recieve even-numbered indices, starting from 0 and increasing by 2,
 *     with additional gaps left for MSB tiles.
 *   - MSB tiles recieve odd-numbered indices, starting from 1 and increasing by 2.
 *
 * Example:
 *   For k = 20, this function receives `BytesPerTail = 2` and `Schema = {0, 0, 1}`:
 *     - There are two tail blocks (containing bits 0–15) and one MSB block (containing bits 16–19).
 *     - Each tail block contributes 2 tiles, for a total of 4 tail tiles: {0, 2, 6, 8}.
 *     - The MSB block contributes 1 tile with index 1. Each byte in this tile holds two 4-bit
 *       MSB values extracted from the unpacked elements.
 *     - Final mapping: {0, 2, 6, 8, 1}
 *
 * @tparam Schema        Reference to a constexpr array of `0`s and `1`s defining the schema.
 * @tparam BytesPerTail  Number of tiles per tail block (i.e., BytesPerValue - 1).
 * @return               A `std::array<int, Len>` mapping packed byte positions to tile indices.
 */
template <auto& Schema, std::size_t BytesPerTail>
consteval auto packed_block_schema_to_tile_index() {
  constexpr auto& src = Schema;
  constexpr std::size_t N = src.size();

  constexpr std::size_t Len = [] {
    std::size_t l = 0;
    for (std::size_t i = 0; i < N; ++i)
      l += (src[i] == 0) ? BytesPerTail : 1;
    return l;
  }();

  std::array<int, Len> out{};

  std::size_t tail_cursor = 0;
  std::size_t msb_cursor = 0;
  std::size_t pos = 0;
  for (std::size_t i = 0; i < N; ++i) {
    if (src[i] == 0) {
      for (std::size_t k = 0; k < BytesPerTail; ++k)
        out[pos++] = (tail_cursor++) * 2;
      tail_cursor++;
    } else {
      out[pos++] = (msb_cursor++) * 2 + 1;
    }
  }
  return out;
}

/**
 * PackSplit kernels discard empty leading bytes, and isolate the most significant
 * bytes (MSBs) of each value in preparation for further processing.
 *
 * - Discards empty/zero leading bytes, while preserving tail bytes
 * - Every (I % K == K - 1)th next() call returns a tile of MSBs
 */

template <Port OutT, Port InT, class Src>
struct PackSplitImpl_2B {
  struct State {
    uint8x16x2_t x0, x1;
  };
  template <int I>
  static inline uint8x16x2_t next(State& st) {
    constexpr auto i = I % 2;
    constexpr auto p = I;
    return static_switch<i>(
        [&] {
          st.x0 = Src::template next<p + 0>();
          st.x1 = Src::template next<p + 1>();
          return vtrn1q_u8_x2(st.x0, st.x1); // X[0:31], X_B[0]
        },
        [&] {
          return vtrn2q_u8_x2(st.x0, st.x1); // X[0:31], X_B[1]
        });
  }
};

struct PackSplit_2B {
  static constexpr auto in_t = Port{2};
  static constexpr auto out_t = Port{2};

  template <Port OutT, Port InT, class Src>
  using impl = PackSplitImpl_2B<OutT, InT, Src>;
};

template <Port OutT, Port InT, class Src>
struct PackSplitImpl_KB_Kin34 {
  struct State {
    uint8x16x2_t x01, x23;
  };
  template <int I>
  static inline uint8x16x2_t next(State& st) {
    constexpr auto i = I % OutT.tile_c;
    constexpr auto p = (I / OutT.tile_c) * 4;
    return static_switch<i>(
        [&] {
          auto x0 = Src::template next<p + 0>();
          auto x1 = Src::template next<p + 1>();
          st.x01 = vtrn2q_u16_x2(x0, x1); // X[0:15], X_B[2:3]
          return vtrn1q_u16_x2(x0, x1);   // X[0:15], X_B[0:1]
        },
        [&] {
          auto x2 = Src::template next<p + 2>();
          auto x3 = Src::template next<p + 3>();
          st.x23 = vtrn2q_u16_x2(x2, x3); // X[16:31], X_B[2:3]
          return vtrn1q_u16_x2(x2, x3);   // X[16:31], X_B[0:1]
        },
        [&] {
          return vtrn1q_u8_x2(st.x01, st.x23); // X[0:31], X_B[2]
        },
        [&] {
          return vtrn2q_u8_x2(st.x01, st.x23); // X[0:31], X_B[3]
        });
  };
};

template <int K>
struct PackSplit_KB_Kin34 {
  static constexpr auto in_t = Port{4};
  static constexpr auto out_t = Port{K};

  template <Port OutT, Port InT, class Src>
  using impl = PackSplitImpl_KB_Kin34<OutT, InT, Src>;
};

template <Port OutT, Port InT, class Src, bool FastK6 = false /* FastK6 used for 48-bit values only */>
struct PackSplitImpl_KB_Kin5678 {
  struct State {
    uint8x16x2_t x01, x0123_B45, x0123_B67;
    uint8x16x2_t x45, x4567_B45, x4567_B67;
  };

  template <int I>
  static inline uint8x16x2_t next(State& st) {
    constexpr auto K = OutT.tile_c;
    constexpr auto i = I % K;
    constexpr auto p = (I / K) * 8;
    if constexpr (i <= 3) {
      return static_switch<i>(
          [&] {
            auto x0 = Src::template next<p + 0>();
            auto x1 = Src::template next<p + 1>();
            st.x01 = vtrn2q_u32_x2(x0, x1); // X[0:7], X_B[4:7]
            return vtrn1q_u32_x2(x0, x1);   // X[0:7], X_B[0:3]
          },
          [&] {
            auto x2 = Src::template next<p + 2>();
            auto x3 = Src::template next<p + 3>();
            auto x23 = vtrn2q_u32_x2(x2, x3);          // X[8:15], X_B[4:7]
            st.x0123_B45 = vtrn1q_u16_x2(st.x01, x23); // X[0:15], X_B[4:5]
            if constexpr (K >= 7) {
              st.x0123_B67 = vtrn2q_u16_x2(st.x01, x23); // X[0:15], X_B[6:7]
            }
            return vtrn1q_u32_x2(x2, x3); // X[8:15], X_B[0:3]
          },
          [&] {
            auto x4 = Src::template next<p + 4>();
            auto x5 = Src::template next<p + 5>();
            st.x45 = vtrn2q_u32_x2(x4, x5); // X[16:23], X_B[4:7]
            return vtrn1q_u32_x2(x4, x5);   // X[16:23], X_B[0:3]
          },
          [&] {
            auto x6 = Src::template next<p + 6>();
            auto x7 = Src::template next<p + 7>();
            auto x67 = vtrn2q_u32_x2(x6, x7);          // X[24:31], X_B[4:7]
            st.x4567_B45 = vtrn1q_u16_x2(st.x45, x67); // X[16:31], X_B[4:5]
            if constexpr (K >= 7) {
              st.x4567_B67 = vtrn2q_u16_x2(st.x45, x67); // X[16:31], X_B[6:7]
            }
            return vtrn1q_u32_x2(x6, x7); // X[24:31], X_B[0:3]
          });
    } else if (K == 5 || K == 6 && !FastK6) {
      return static_switch<i - 4>(
          [&] {
            return vtrn1q_u8_x2(st.x0123_B45, st.x4567_B45); // X[0:31], X_B[4]
          },
          [&] {
            return vtrn2q_u8_x2(st.x0123_B45, st.x4567_B45); // X[0:31], X_B[5]
          });
    } else /* FastK6 || K == 7 || K == 8 */ {
      return static_switch<i - 4>(
          [&] {
            return st.x0123_B45; // X[0:15], X_B[4:5]
          },
          [&] {
            return st.x4567_B45; // X[16:31], X_B[4:5]
          },
          [&] {
            return vtrn1q_u8_x2(st.x0123_B67, st.x4567_B67); // X[0:31], X_B[6]
          },
          [&] {
            return vtrn2q_u8_x2(st.x0123_B67, st.x4567_B67); // X[0:31], X_B[7]
          });
    }
  }
};

template <int K, bool FastK6 = false /* FastK6 used for 48-bit values only */>
struct PackSplit_KB_Kin5678 {
  static constexpr auto in_t = Port{8};
  static constexpr auto out_t = Port{K};

  template <Port OutT, Port InT, class Src>
  using impl = PackSplitImpl_KB_Kin5678<OutT, InT, Src, FastK6>;
};

template <int K, bool FastK6 = false>
struct PackSplitSelector;

// clang-format off
template <> struct PackSplitSelector<2> { using type = PackSplit_2B; };
template <> struct PackSplitSelector<3> { using type = PackSplit_KB_Kin34<3>; };
template <> struct PackSplitSelector<4> { using type = PackSplit_KB_Kin34<4>; };
template <> struct PackSplitSelector<5> { using type = PackSplit_KB_Kin5678<5>; };
template <bool FastK6> struct PackSplitSelector<6, FastK6> { using type = PackSplit_KB_Kin5678<6, FastK6>; };
template <> struct PackSplitSelector<7> { using type = PackSplit_KB_Kin5678<7>; };
template <> struct PackSplitSelector<8> { using type = PackSplit_KB_Kin5678<8>; };
// clang-format on

template <int K, bool FastK6 = false>
using PackSplit = typename PackSplitSelector<K, FastK6>::type;

/**
 * PackMSB kernels use interleave patterns for bitpacking.
 */

template <Port OutT, Port InT, class Src>
struct PackImpl_KX_Xmod8eq1_N256 {
  struct State {
    typename Src::State src;
    CallCheck<OutT.tile_c> calls;
  };

  static inline void start(State& st, CommonCtx ctx) { Src::start(st.src, ctx); }

  template <int I>
  static inline uint8x16x2_t next(State& st) {
    // x0:00000000 := x0:_______0 + x1:______0_ + x2:_____0__ + x3:____0___
    //              + x4:___0____ + x5:__0_____ + x6:_0______ + x7:0_______
    st.calls.template touch<I>();
    constexpr int i = 0;
    constexpr int p = I * 8;

    // Merge tree reduces dependency chain length (8->3)

    auto x0 = Src::template next<p + 0>(st.src);
    auto x1 = Src::template next<p + 1>(st.src);
    x0 = vsliq_n_u8_x2<1>(x0, x1); // x1->x0 (1b)

    auto x2 = Src::template next<p + 2>(st.src);
    auto x3 = Src::template next<p + 3>(st.src);
    x2 = vsliq_n_u8_x2<1>(x2, x3); // x3->x2 (1b)
    x0 = vsliq_n_u8_x2<2>(x0, x2); // x2->x0 (2b)

    auto x4 = Src::template next<p + 4>(st.src);
    auto x5 = Src::template next<p + 5>(st.src);
    x4 = vsliq_n_u8_x2<1>(x4, x5); // x5->x4 (1b)

    auto x6 = Src::template next<p + 6>(st.src);
    auto x7 = Src::template next<p + 7>(st.src);
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

struct Pack_KX_Xmod8eq1_N256 {
  static constexpr auto in_t = Port{8};
  static constexpr auto out_t = Port{1};
  static constexpr std::array block_schema{0, 0, 0, 0, 0, 0, 0, 0, 1};

  template <Port OutT, Port InT, class Src>
  using impl = PackImpl_KX_Xmod8eq1_N256<OutT, InT, Src>;
};

template <Port OutT, Port InT, class Src>
struct PackImpl_KX_Xmod8eq2_N128 {
  struct State {
    typename Src::State src;
    CallCheck<OutT.tile_c> calls;
  };

  static inline void start(State& st, CommonCtx ctx) { Src::start(st.src, ctx); }

  template <int I>
  static inline uint8x16x2_t next(State& st) {
    // x0:10101010 := x0:______10 + x1:____10__ + x2:__10____ + x3:10______
    st.calls.template touch<I>();
    constexpr int i = 0;
    constexpr int p = I * 4;

    // Merge tree reduces dependency chain length (4->2)

    auto x0 = Src::template next<p + 0>(st.src);
    auto x1 = Src::template next<p + 1>(st.src);
    x0 = vsliq_n_u8_x2<2>(x0, x1); // x1->x0 (2b)

    auto x2 = Src::template next<p + 2>(st.src);
    auto x3 = Src::template next<p + 3>(st.src);
    x2 = vsliq_n_u8_x2<2>(x2, x3); // x3->x2 (2b)

    x0 = vsliq_n_u8_x2<4>(x0, x2); // x2->x0 (4b)

    return x0;
  }

  static inline void end(State& st) {
    st.calls.check();
    Src::end(st.src);
  }
};

struct Pack_KX_Xmod8eq2_N128 {
  static constexpr auto in_t = Port{4};
  static constexpr auto out_t = Port{1};
  static constexpr std::array block_schema{0, 0, 0, 0, 1};

  template <Port OutT, Port InT, class Src>
  using impl = PackImpl_KX_Xmod8eq2_N128<OutT, InT, Src>;
};

template <Port OutT, Port InT, class Src>
struct PackImpl_KX_Xmod8eq3_N256 {
  struct State {
    typename Src::State src;
    uint8x16x2_t x0, x1, x2, x3, x4, x5, x6, x7;
    CallCheck<OutT.tile_c> calls;
  };

  static inline void start(State& st, CommonCtx ctx) { Src::start(st.src, ctx); }

  template <int I>
  static inline uint8x16x2_t next(State& st) {
    // x0:10210210 := x0:_____210 + x1:__210___ + x2:10______
    // x3:22210210 := x3:_____210 + x4:__210___ + x2:_2______ + x5:2_______
    // x6:10210210 := x6:_____210 + x7:__210___ + x5:10______
    st.calls.template touch<I>();
    constexpr int i = I % 3;
    constexpr int p = (I / 3) * 8;
    const uint8x16_t mask6 = vdupq_n_u8(0b0011'1111);
    const uint8x16_t mask7 = vdupq_n_u8(0b0111'1111);

    return static_switch<i>(
        [&] {
          st.x0 = Src::template next<p + 0>(st.src);
          st.x1 = Src::template next<p + 1>(st.src);
          st.x2 = Src::template next<p + 2>(st.src);
          st.x0 = vsliq_n_u8_x2<3>(st.x0, st.x1); // x1->x0 (3b)
          st.x0 = vsliq_n_u8_x2<6>(st.x0, st.x2); // x2->x0 (2b)
          return st.x0;
        },
        [&] {
          st.x3 = Src::template next<p + 3>(st.src);
          st.x4 = Src::template next<p + 4>(st.src);
          st.x5 = Src::template next<p + 5>(st.src);
          st.x2 = vbifq_n_u8_x2(vshlq_n_u8_x2<4>(st.x2), vshlq_n_u8_x2<5>(st.x5), mask7); // x5->x2 (1b)
          st.x3 = vsliq_n_u8_x2<3>(st.x3, st.x4);                                         // x4->x3 (3b)
          st.x3 = vbifq_n_u8_x2(st.x3, st.x2, mask6);                                     // x2->x3 (2b)
          return st.x3;
        },
        [&] {
          st.x6 = Src::template next<p + 6>(st.src);
          st.x7 = Src::template next<p + 7>(st.src);
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

struct Pack_KX_Xmod8eq3_N256 {
  static constexpr auto in_t = Port{8};
  static constexpr auto out_t = Port{3};
  static constexpr std::array block_schema{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1};

  template <Port OutT, Port InT, class Src>
  using impl = PackImpl_KX_Xmod8eq3_N256<OutT, InT, Src>;
};

template <Port OutT, Port InT, class Src>
struct PackImpl_KX_Xmod8eq4_N064 {
  struct State {
    typename Src::State src;
    CallCheck<OutT.tile_c> calls;
  };

  static inline void start(State& st, CommonCtx ctx) { Src::start(st.src, ctx); }

  template <int I>
  static inline uint8x16x2_t next(State& st) {
    // x0:32103210 := x0:____3210 + x1:3210____
    st.calls.template touch<I>();
    constexpr int i = 0;
    constexpr int p = I * 2;

    auto x0 = Src::template next<p + 0>(st.src);
    auto x1 = Src::template next<p + 1>(st.src);
    x0 = vsliq_n_u8_x2<4>(x0, x1); // x1->x0 (4b)
    return x0;
  }

  static inline void end(State& st) {
    st.calls.check();
    Src::end(st.src);
  }
};

struct Pack_KX_Xmod8eq4_N064 {
  static constexpr auto in_t = Port{2};
  static constexpr auto out_t = Port{1};
  static constexpr std::array block_schema{0, 0, 1};

  template <Port OutT, Port InT, class Src>
  using impl = PackImpl_KX_Xmod8eq4_N064<OutT, InT, Src>;
};

template <Port OutT, Port InT, class Src>
struct PackImpl_KX_Xmod8eq5_N256 {
  struct State {
    typename Src::State src;
    uint8x16x2_t x0, x1, x2, x3, x4, x5, x6, x7;
    CallCheck<OutT.tile_c> calls;
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
    constexpr int i = I % 5;
    constexpr int p = (I / 5) * 8;
    const uint8x16_t mask5 = vdupq_n_u8(0b0001'1111);

    return static_switch<i>(
        [&] {
          st.x0 = Src::template next<p + 0>(st.src);
          st.x1 = Src::template next<p + 1>(st.src);
          st.x0 = vbifq_n_u8_x2(st.x0, vshlq_n_u8_x2<3>(st.x1), mask5); // x1->x0 (3b)
          return st.x0;
        },
        [&] {
          st.x2 = Src::template next<p + 2>(st.src);
          st.x3 = Src::template next<p + 3>(st.src);
          st.x2 = vbifq_n_u8_x2(st.x2, vshlq_n_u8_x2<3>(st.x3), mask5); // x3->x2 (3b)
          st.x1 = vsliq_n_u8_x2<2>(st.x1, st.x3);                       // x3->x1 (2b)
          return st.x2;
        },
        [&] {
          st.x4 = Src::template next<p + 4>(st.src);
          st.x5 = Src::template next<p + 5>(st.src);
          st.x4 = vbifq_n_u8_x2(st.x4, vshlq_n_u8_x2<3>(st.x5), mask5); // x5->x4 (3b)
          st.x1 = vsliq_n_u8_x2<4>(st.x1, st.x5);                       // x5->x1 (2b)
          return st.x4;
        },
        [&] {
          st.x6 = Src::template next<p + 6>(st.src);
          st.x7 = Src::template next<p + 7>(st.src);
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

struct Pack_KX_Xmod8eq5_N256 {
  static constexpr auto in_t = Port{8};
  static constexpr auto out_t = Port{5};
  static constexpr std::array block_schema{0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1};

  template <Port OutT, Port InT, class Src>
  using impl = PackImpl_KX_Xmod8eq5_N256<OutT, InT, Src>;
};

template <Port OutT, Port InT, class Src>
struct PackImpl_KX_Xmod8eq6_N128 {
  struct State {
    typename Src::State src;
    uint8x16x2_t x0, x1, x2, x3;
    CallCheck<OutT.tile_c> calls;
  };

  static inline void start(State& st, CommonCtx ctx) { Src::start(st.src, ctx); }

  template <int I>
  static inline uint8x16x2_t next(State& st) {
    // x0:54543210 := x0:__543210 + x1:54______
    // x1:32103210 := x1:____3210 + x2:3210____
    // x3:54543210 := x3:__543210 + x2:54______
    st.calls.template touch<I>();
    constexpr int i = I % 3;
    constexpr int p = (I / 3) * 4;
    const uint8x16_t mask6 = vdupq_n_u8(0b0011'1111);

    return static_switch<i>(
        [&] {
          st.x0 = Src::template next<p + 0>(st.src);
          st.x1 = Src::template next<p + 1>(st.src);
          st.x0 = vbifq_n_u8_x2(st.x0, vshlq_n_u8_x2<2>(st.x1), mask6); // x1->x0 (2b)
          return st.x0;
        },
        [&] {
          st.x2 = Src::template next<p + 2>(st.src);
          st.x1 = vsliq_n_u8_x2<4>(st.x1, st.x2); // x2->x1 (4b)
          return st.x1;
        },
        [&] {
          st.x3 = Src::template next<p + 3>(st.src);
          st.x3 = vbifq_n_u8_x2(st.x3, vshlq_n_u8_x2<2>(st.x2), mask6); // x2->x3 (2b)
          return st.x3;
        });
  }

  static inline void end(State& st) {
    st.calls.check();
    Src::end(st.src);
  }
};

struct Pack_KX_Xmod8eq6_N128 {
  static constexpr auto in_t = Port{4};
  static constexpr auto out_t = Port{3};
  static constexpr std::array block_schema{0, 0, 1, 0, 1, 0, 1};

  template <Port OutT, Port InT, class Src>
  using impl = PackImpl_KX_Xmod8eq6_N128<OutT, InT, Src>;
};

template <Port OutT, Port InT, class Src>
struct PackImpl_KX_Xmod8eq7_N256 {
  struct State {
    typename Src::State src;
    uint8x16x2_t x0, x1, x2, x3, x4, x5, x6, x7;
    CallCheck<OutT.tile_c> calls;
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
    st.calls.template touch<I>();
    constexpr int i = I % 7;
    constexpr int p = (I / 7) * 8;

    uint8x16_t mask6 = vdupq_n_u8(0b0011'1111);
    uint8x16_t mask7 = vdupq_n_u8(0b0111'1111);

    return static_switch<i>(
        [&] {
          st.x0 = Src::template next<p + 0>(st.src);
          st.x1 = Src::template next<p + 1>(st.src);
          return vbifq_n_u8_x2(st.x0, vshlq_n_u8_x2<1>(st.x1), mask7); // x1->x0 (1b)
        },
        [&] {
          st.x2 = Src::template next<p + 2>(st.src);
          return vbifq_n_u8_x2(st.x1, vshlq_n_u8_x2<2>(st.x2), mask6); // x2->x1 (2b)
        },
        [&] {
          st.x3 = Src::template next<p + 3>(st.src);
          return vbifq_n_u8_x2(st.x3, vshlq_n_u8_x2<1>(st.x2), mask7); // x2->x3 (1b)
        },
        [&] {
          st.x6 = Src::template next<p + 6>(st.src);
          return vsliq_n_u8_x2<4>(st.x2, st.x6); // x6->x2 (4b)
        },
        [&] {
          st.x4 = Src::template next<p + 4>(st.src);
          st.x5 = Src::template next<p + 5>(st.src);
          return vbifq_n_u8_x2(st.x4, vshlq_n_u8_x2<1>(st.x5), mask7); // x5->x4 (1b)
        },
        [&] {
          return vbifq_n_u8_x2(st.x5, vshlq_n_u8_x2<2>(st.x6), mask6); // x6->x5 (2b)
        },
        [&] {
          st.x7 = Src::template next<p + 7>(st.src);
          return vbifq_n_u8_x2(st.x7, vshlq_n_u8_x2<1>(st.x6), mask7); // x6->x7 (1b)
        });
  }

  static inline void end(State& st) {
    st.calls.check();
    Src::end(st.src);
  }
};

struct Pack_KX_Xmod8eq7_N256 {
  static constexpr auto in_t = Port{8};
  static constexpr auto out_t = Port{7};
  static constexpr std::array block_schema{0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};

  template <Port OutT, Port InT, class Src>
  using impl = PackImpl_KX_Xmod8eq7_N256<OutT, InT, Src>;
};

template <Port OutT, Port InT, class Src, class PackSplit, class PackMsb>
struct PackFuseImpl {
  static constexpr auto tile_idx =
      packed_block_schema_to_tile_index<PackMsb::block_schema, PackSplit::out_t.tile_c - 1>();

  using PackSplitImpl =
      typename PackSplit::template impl<Port{PackSplit::out_t.tile_c, PackMsb::in_t.tile_c* OutT.width},
                                        Port{PackSplit::in_t.tile_c, PackMsb::in_t.tile_c* InT.width}, Src>;

  struct PackMsbSrc {
    struct State {};
    static inline void start(State& st, CommonCtx ctx) {}
    template <int I>
    static inline uint8x16x2_t next() {
      return PackSplitImpl::template next<(I + 1) * PackMsb::out_t.tile_c - 1>();
    }
    static inline void end(State& st, CommonCtx ctx) {}
  };

  using PackMsbImpl = typename PackMsb::template impl<Port{PackMsb::out_t.tile_c, OutT.width},
                                                      Port{PackMsb::in_t.tile_c, InT.width}, PackMsbSrc>;

  struct State {
    typename PackSplitImpl::State tail;
    typename PackMsbImpl::State msb;
    CallCheck<OutT.tile_c> calls;
  };

  static inline void start(State& st, CommonCtx ctx) {
    PackSplitImpl::start(st.tail, ctx);
    PackMsbImpl::start(st.msb, ctx);
  }

  template <int I>
  static inline void next(State& st) {
    st.calls.template touch<I>();

    constexpr int i = I % tile_idx.size();

    return static_switch<tile_idx[i] % 2>(
        [&] {
          return PackSplitImpl::template call<tile_idx[i] / 2>(st.tail);
        },
        [&] {
          return PackMsbImpl::template call<tile_idx[i] / 2>(st.msb);
        });
  }

  static inline void end(State& st) {
    st.calls.check();
    PackSplitImpl::end(st.tail);
    PackMsbImpl::end(st.msb);
  }
};

template <class PackSplit, class PackMsb>
struct PackFuse {
  static constexpr auto in_t = Port{PackMsb::in_t.tile_c * PackSplit::in_t.tile_c};
  static constexpr auto out_t =
      Port{packed_block_schema_to_tile_index<PackMsb::block_schema, PackSplit::out_t.tile_c - 1>().size()};

  template <Port OutT, Port InT, class Src>
  using impl = PackFuseImpl<OutT, InT, Src, PackSplit, PackMsb>;
};

template <int K>
struct PackSelector;

// clang-format off
template <> struct PackSelector< 1> { using type = Pack_KX_Xmod8eq1_N256; };
template <> struct PackSelector< 2> { using type = Pack_KX_Xmod8eq2_N128; };
template <> struct PackSelector< 3> { using type = Pack_KX_Xmod8eq3_N256; };
template <> struct PackSelector< 4> { using type = Pack_KX_Xmod8eq4_N064; };
template <> struct PackSelector< 5> { using type = Pack_KX_Xmod8eq5_N256; };
template <> struct PackSelector< 6> { using type = Pack_KX_Xmod8eq6_N128; };
template <> struct PackSelector< 7> { using type = Pack_KX_Xmod8eq7_N256; };
template <> struct PackSelector< 8> { using type = Identity; };
template <> struct PackSelector< 9> { using type = PackFuse<PackSplit<2>, Pack_KX_Xmod8eq1_N256>; };
template <> struct PackSelector<10> { using type = PackFuse<PackSplit<2>, Pack_KX_Xmod8eq2_N128>; };
template <> struct PackSelector<11> { using type = PackFuse<PackSplit<2>, Pack_KX_Xmod8eq3_N256>; };
template <> struct PackSelector<12> { using type = PackFuse<PackSplit<2>, Pack_KX_Xmod8eq4_N064>; };
template <> struct PackSelector<13> { using type = PackFuse<PackSplit<2>, Pack_KX_Xmod8eq5_N256>; };
template <> struct PackSelector<14> { using type = PackFuse<PackSplit<2>, Pack_KX_Xmod8eq6_N128>; };
template <> struct PackSelector<15> { using type = PackFuse<PackSplit<2>, Pack_KX_Xmod8eq7_N256>; };
template <> struct PackSelector<16> { using type = Identity; };
template <> struct PackSelector<17> { using type = PackFuse<PackSplit<3>, Pack_KX_Xmod8eq1_N256>; };
template <> struct PackSelector<18> { using type = PackFuse<PackSplit<3>, Pack_KX_Xmod8eq2_N128>; };
template <> struct PackSelector<19> { using type = PackFuse<PackSplit<3>, Pack_KX_Xmod8eq3_N256>; };
template <> struct PackSelector<20> { using type = PackFuse<PackSplit<3>, Pack_KX_Xmod8eq4_N064>; };
template <> struct PackSelector<21> { using type = PackFuse<PackSplit<3>, Pack_KX_Xmod8eq5_N256>; };
template <> struct PackSelector<22> { using type = PackFuse<PackSplit<3>, Pack_KX_Xmod8eq6_N128>; };
template <> struct PackSelector<23> { using type = PackFuse<PackSplit<3>, Pack_KX_Xmod8eq7_N256>; };
template <> struct PackSelector<24> { using type = PackSplit<3>; };
template <> struct PackSelector<25> { using type = PackFuse<PackSplit<4>, Pack_KX_Xmod8eq1_N256>; };
template <> struct PackSelector<26> { using type = PackFuse<PackSplit<4>, Pack_KX_Xmod8eq2_N128>; };
template <> struct PackSelector<27> { using type = PackFuse<PackSplit<4>, Pack_KX_Xmod8eq3_N256>; };
template <> struct PackSelector<28> { using type = PackFuse<PackSplit<4>, Pack_KX_Xmod8eq4_N064>; };
template <> struct PackSelector<29> { using type = PackFuse<PackSplit<4>, Pack_KX_Xmod8eq5_N256>; };
template <> struct PackSelector<30> { using type = PackFuse<PackSplit<4>, Pack_KX_Xmod8eq6_N128>; };
template <> struct PackSelector<31> { using type = PackFuse<PackSplit<4>, Pack_KX_Xmod8eq7_N256>; };
template <> struct PackSelector<32> { using type = Identity; };
template <> struct PackSelector<33> { using type = PackFuse<PackSplit<5>, Pack_KX_Xmod8eq1_N256>; };
template <> struct PackSelector<34> { using type = PackFuse<PackSplit<5>, Pack_KX_Xmod8eq2_N128>; };
template <> struct PackSelector<35> { using type = PackFuse<PackSplit<5>, Pack_KX_Xmod8eq3_N256>; };
template <> struct PackSelector<36> { using type = PackFuse<PackSplit<5>, Pack_KX_Xmod8eq4_N064>; };
template <> struct PackSelector<37> { using type = PackFuse<PackSplit<5>, Pack_KX_Xmod8eq5_N256>; };
template <> struct PackSelector<38> { using type = PackFuse<PackSplit<5>, Pack_KX_Xmod8eq6_N128>; };
template <> struct PackSelector<39> { using type = PackFuse<PackSplit<5>, Pack_KX_Xmod8eq7_N256>; };
template <> struct PackSelector<40> { using type = PackSplit<5>; };
template <> struct PackSelector<41> { using type = PackFuse<PackSplit<6>, Pack_KX_Xmod8eq1_N256>; };
template <> struct PackSelector<42> { using type = PackFuse<PackSplit<6>, Pack_KX_Xmod8eq2_N128>; };
template <> struct PackSelector<43> { using type = PackFuse<PackSplit<6>, Pack_KX_Xmod8eq3_N256>; };
template <> struct PackSelector<44> { using type = PackFuse<PackSplit<6>, Pack_KX_Xmod8eq4_N064>; };
template <> struct PackSelector<45> { using type = PackFuse<PackSplit<6>, Pack_KX_Xmod8eq5_N256>; };
template <> struct PackSelector<46> { using type = PackFuse<PackSplit<6>, Pack_KX_Xmod8eq6_N128>; };
template <> struct PackSelector<47> { using type = PackFuse<PackSplit<6>, Pack_KX_Xmod8eq7_N256>; };
template <> struct PackSelector<48> { using type = PackSplit<6, true>; };
template <> struct PackSelector<49> { using type = PackFuse<PackSplit<7>, Pack_KX_Xmod8eq1_N256>; };
template <> struct PackSelector<50> { using type = PackFuse<PackSplit<7>, Pack_KX_Xmod8eq2_N128>; };
template <> struct PackSelector<51> { using type = PackFuse<PackSplit<7>, Pack_KX_Xmod8eq3_N256>; };
template <> struct PackSelector<52> { using type = PackFuse<PackSplit<7>, Pack_KX_Xmod8eq4_N064>; };
template <> struct PackSelector<53> { using type = PackFuse<PackSplit<7>, Pack_KX_Xmod8eq5_N256>; };
template <> struct PackSelector<54> { using type = PackFuse<PackSplit<7>, Pack_KX_Xmod8eq6_N128>; };
template <> struct PackSelector<55> { using type = PackFuse<PackSplit<7>, Pack_KX_Xmod8eq7_N256>; };
template <> struct PackSelector<56> { using type = PackSplit<7>; };
template <> struct PackSelector<57> { using type = PackFuse<PackSplit<8>, Pack_KX_Xmod8eq1_N256>; };
template <> struct PackSelector<58> { using type = PackFuse<PackSplit<8>, Pack_KX_Xmod8eq2_N128>; };
template <> struct PackSelector<59> { using type = PackFuse<PackSplit<8>, Pack_KX_Xmod8eq3_N256>; };
template <> struct PackSelector<60> { using type = PackFuse<PackSplit<8>, Pack_KX_Xmod8eq4_N064>; };
template <> struct PackSelector<61> { using type = PackFuse<PackSplit<8>, Pack_KX_Xmod8eq5_N256>; };
template <> struct PackSelector<62> { using type = PackFuse<PackSplit<8>, Pack_KX_Xmod8eq6_N128>; };
template <> struct PackSelector<63> { using type = PackFuse<PackSplit<8>, Pack_KX_Xmod8eq7_N256>; };
template <> struct PackSelector<64> { using type = Identity; };
// clang-format on

template <int InK, int OutK>
struct PackSelector2 {
  static_assert((InK == 8 || InK == 16 || InK == 32 || InK == 64), "InK must be 8, 16, 32, or 64");
  static_assert(InK >= OutK, "InK must be greater than or equal to OutK");

  static constexpr int Pow2RoundedOutK = OutK <= 1 ? 1 : 1 << (32 - __builtin_clz(OutK - 1));

  using type = std::conditional_t<(InK > Pow2RoundedOutK),
                                  Series<Narrow<InK, Pow2RoundedOutK>, typename PackSelector<OutK>::type>,
                                  typename PackSelector<OutK>::type>;
};

template <int InK, int OutK>
using Pack = typename PackSelector2<InK, OutK>::type;

template <class Src = Load<1>>
struct Unpack_K01to08_N256 {
  static constexpr int length_v = 8;

  struct State {
    typename Src::State src;
    uint8x16x2_t x0;
    CallCheck<OutT.tile_c> calls;
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

    // Block order minimises register pressure (only x3 requires a long lifetime)

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
    case 1: loop<Store<Pack_KX_Xmod8eq1_N256<>>, 256,  32, 1>(in, out, n / 256); break;
    case 2: loop<Store<Pack_KX_Xmod8eq2_N128<>>, 128,  32, 2>(in, out, n / 128); break;
    case 3: loop<Store<Pack_KX_Xmod8eq3_N256<>>, 256,  96, 1>(in, out, n / 256); break;
    case 4: loop<Store<Pack_KX_Xmod8eq4_N064<>>,  64,  32, 4>(in, out, n /  64); break;
    case 5: loop<Store<Pack_KX_Xmod8eq5_N256<>>, 256, 160, 1>(in, out, n / 256); break;
    case 6: loop<Store<Pack_KX_Xmod8eq6_N128<>>, 128,  96, 2>(in, out, n / 128); break;
    case 7: loop<Store<Pack_KX_Xmod8eq7_N256<>>, 256, 224, 1>(in, out, n / 256); break;
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