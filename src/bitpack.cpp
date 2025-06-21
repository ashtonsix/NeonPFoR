#include "bitpack.h"
#include <arm_sve.h>
#include <cstdint>

namespace NeonPForLib {

namespace impl {

// Implement ARM NEON bitwise selection instructions not available as intrinsics
// Both modify the first operand (a) in-place and return the modified value
// BIT: a = (a & ~sel) | (b & sel)
static inline uint8x16_t vbitq_n_u8(uint8x16_t a, uint8x16_t b, uint8x16_t sel) {
  asm("bit %0.16b, %1.16b, %2.16b" : "+w"(a) : "w"(b), "w"(sel));
  return a;
}

// BIF: a = (a & sel) | (b & ~sel) - inverse of BIT
static inline uint8x16_t vbifq_n_u8(uint8x16_t a, uint8x16_t b, uint8x16_t sel) {
  asm("bif %0.16b, %1.16b, %2.16b" : "+w"(a) : "w"(b), "w"(sel));
  return a;
}

// Making two vld1q/vst1q calls instead of one call to vld1q_x2/vst1q_x2 lowers to LDP/STP
// more reliably (perf optimal). LDP/STP aren't exposed as intrinsics, and using via inline
// asm leads to poor regalloc in LLVM, so gentle encouragement like this is the best option
template <int offset>
static inline uint8x16x2_t vldpq_u8(const uint8_t* in) {
  uint8x16x2_t result;
  result.val[0] = vld1q_u8(in + offset);
  result.val[1] = vld1q_u8(in + offset + 16);
  return result;
}

template <int offset>
static inline void vstpq_u8(uint8_t* out, uint8x16x2_t data) {
  vst1q_u8(out + offset, data.val[0]);
  vst1q_u8(out + offset + 16, data.val[1]);
}

static inline uint8x16x2_t vandq_u8_x2(uint8x16x2_t src, uint8x16_t mask) {
  uint8x16x2_t result;
  result.val[0] = vandq_u8(src.val[0], mask);
  result.val[1] = vandq_u8(src.val[1], mask);
  return result;
}

template <int n>
static inline uint8x16x2_t vshrq_n_u8_x2(uint8x16x2_t src) {
  uint8x16x2_t result;
  result.val[0] = vshrq_n_u8(src.val[0], n);
  result.val[1] = vshrq_n_u8(src.val[1], n);
  return result;
}

template <int n>
static inline uint8x16x2_t vshlq_n_u8_x2(uint8x16x2_t src) {
  uint8x16x2_t result;
  result.val[0] = vshlq_n_u8(src.val[0], n);
  result.val[1] = vshlq_n_u8(src.val[1], n);
  return result;
}

template <int n>
static inline uint8x16x2_t vsliq_n_u8_x2(uint8x16x2_t a, uint8x16x2_t b) {
  a.val[0] = vsliq_n_u8(a.val[0], b.val[0], n);
  a.val[1] = vsliq_n_u8(a.val[1], b.val[1], n);
  return a;
}

template <int n>
static inline uint8x16x2_t vsriq_n_u8_x2(uint8x16x2_t a, uint8x16x2_t b) {
  a.val[0] = vsriq_n_u8(a.val[0], b.val[0], n);
  a.val[1] = vsriq_n_u8(a.val[1], b.val[1], n);
  return a;
}

static inline uint8x16x2_t vbitq_n_u8_x2(uint8x16x2_t a, uint8x16x2_t b, uint8x16_t sel) {
  a.val[0] = vbitq_n_u8(a.val[0], b.val[0], sel);
  a.val[1] = vbitq_n_u8(a.val[1], b.val[1], sel);
  return a;
}

static inline uint8x16x2_t vbifq_n_u8_x2(uint8x16x2_t a, uint8x16x2_t b, uint8x16_t sel) {
  a.val[0] = vbifq_n_u8(a.val[0], b.val[0], sel);
  a.val[1] = vbifq_n_u8(a.val[1], b.val[1], sel);
  return a;
}

static inline void unpack1_8_256(const uint8_t* __restrict__ in, uint8_t* __restrict__ out) {
  // y0 := x0:_______0
  // y1 := x0:______0_
  // y2 := x0:_____0__
  // y3 := x0:____0___
  // y4 := x0:___0____
  // y5 := x0:__0_____
  // y6 := x0:_0______
  // y7 := x0:0_______
  uint8x16_t mask = vdupq_n_u8(0x01);

  uint8x16x2_t x0 = vldpq_u8<0>(in);
  uint8x16x2_t y0 = vandq_u8_x2(x0, mask);
  vstpq_u8<0>(out, y0);

  uint8x16x2_t y1 = vandq_u8_x2(vshrq_n_u8_x2<1>(x0), mask);
  vstpq_u8<32>(out, y1);

  uint8x16x2_t y2 = vandq_u8_x2(vshrq_n_u8_x2<2>(x0), mask);
  vstpq_u8<64>(out, y2);

  uint8x16x2_t y3 = vandq_u8_x2(vshrq_n_u8_x2<3>(x0), mask);
  vstpq_u8<96>(out, y3);

  uint8x16x2_t y4 = vandq_u8_x2(vshrq_n_u8_x2<4>(x0), mask);
  vstpq_u8<128>(out, y4);

  uint8x16x2_t y5 = vandq_u8_x2(vshrq_n_u8_x2<5>(x0), mask);
  vstpq_u8<160>(out, y5);

  uint8x16x2_t y6 = vandq_u8_x2(vshrq_n_u8_x2<6>(x0), mask);
  vstpq_u8<192>(out, y6);

  uint8x16x2_t y7 = vshrq_n_u8_x2<7>(x0);
  vstpq_u8<224>(out, y7);
}

static inline void unpack2_8_128(const uint8_t* __restrict__ in, uint8_t* __restrict__ out) {
  // y0 := x0:______10
  // y1 := x0:____10__
  // y2 := x0:__10____
  // y3 := x0:10______
  uint8x16_t mask = vdupq_n_u8(0b00000011);

  uint8x16x2_t x0 = vldpq_u8<0>(in);
  uint8x16x2_t y0 = vandq_u8_x2(x0, mask);
  vstpq_u8<0>(out, y0);

  uint8x16x2_t y1 = vandq_u8_x2(vshrq_n_u8_x2<2>(x0), mask);
  vstpq_u8<32>(out, y1);

  uint8x16x2_t y2 = vandq_u8_x2(vshrq_n_u8_x2<4>(x0), mask);
  vstpq_u8<64>(out, y2);

  uint8x16x2_t y3 = vshrq_n_u8_x2<6>(x0);
  vstpq_u8<96>(out, y3);
}

static inline void unpack3_8_256(const uint8_t* __restrict__ in, uint8_t* __restrict__ out) {
  // y0 := x0:_____210
  // y1 := x0:__210___
  // y2 := x0:10______ + x2:_2______
  // y3 := x1:_____210
  // y4 := x1:__210___
  // y5 := x1:10______ + x2:2_______
  // y6 := x2:_____210
  // y7 := x2:__210___

  uint8x16_t mask = vdupq_n_u8(0b00000111);

  uint8x16x2_t x0 = vldpq_u8<0>(in);
  uint8x16x2_t y0 = vandq_u8_x2(x0, mask);                   // x0->y0 (3 bits)
  uint8x16x2_t y1 = vandq_u8_x2(vshrq_n_u8_x2<3>(x0), mask); // x0->y1 (3 bits)
  vstpq_u8<0>(out, y0);
  vstpq_u8<32>(out, y1);

  uint8x16x2_t x2 = vldpq_u8<64>(in);
  uint8x16x2_t y2 = vandq_u8_x2(vshrq_n_u8_x2<4>(x2), mask); // x2->y2 (1 bit)
  y2 = vsriq_n_u8_x2<6>(y2, x0);                             // x0->y2 (2 bits)
  vstpq_u8<64>(out, y2);

  uint8x16x2_t x1 = vldpq_u8<32>(in);
  uint8x16x2_t y3 = vandq_u8_x2(x1, mask);                   // x1->y2 (3 bits)
  uint8x16x2_t y4 = vandq_u8_x2(vshrq_n_u8_x2<3>(x1), mask); // x1->y3 (3 bits)
  vstpq_u8<96>(out, y3);
  vstpq_u8<128>(out, y4);

  uint8x16x2_t y5 = vshrq_n_u8_x2<5>(x2); // x2->y5 (1 bit)
  y5 = vsriq_n_u8_x2<6>(y5, x1);          // x1->y5 (2 bits)
  vstpq_u8<160>(out, y5);

  uint8x16x2_t y6 = vandq_u8_x2(x2, mask);                   // x2->y6 (3 bits)
  uint8x16x2_t y7 = vandq_u8_x2(vshrq_n_u8_x2<3>(x2), mask); // x2->y7 (3 bits)
  vstpq_u8<192>(out, y6);
  vstpq_u8<224>(out, y7);
}

static inline void unpack4_8_64(const uint8_t* __restrict__ in, uint8_t* __restrict__ out) {
  // y0 := x0:____3210
  // y1 := x0:3210____

  uint8x16_t mask = vdupq_n_u8(0b00001111);

  uint8x16x2_t x0 = vldpq_u8<0>(in);
  uint8x16x2_t y0 = vandq_u8_x2(x0, mask); // x0->y0 (4 bits)
  vstpq_u8<0>(out, y0);

  uint8x16x2_t y1 = vshrq_n_u8_x2<4>(x0); // x0->y1 (4 bits)
  vstpq_u8<32>(out, y1);
}

static inline void unpack5_8_256(const uint8_t* __restrict__ in, uint8_t* __restrict__ out) {
  // y0 := x0:___43210
  // y1 := x0:432_____ + x4:______10
  // y2 := x1:___43210
  // y3 := x1:432_____ + x4:____10__
  // y4 := x2:___43210
  // y5 := x2:432_____ + x4:__10____
  // y6 := x3:___43210
  // y7 := x3:432_____ + x4:10______

  uint8x16_t mask2 = vdupq_n_u8(0b00000011);
  uint8x16_t mask5 = vdupq_n_u8(0b00011111);

  uint8x16x2_t x0 = vldpq_u8<0>(in);
  uint8x16x2_t y0 = vandq_u8_x2(x0, mask5); // x0->y0 (5 bits)
  vstpq_u8<0>(out, y0);

  uint8x16x2_t x4 = vldpq_u8<128>(in);
  uint8x16x2_t y1 = vshrq_n_u8_x2<3>(x0); // x0->y1 (3 bits)
  y1 = vbitq_n_u8_x2(y1, x4, mask2);      // x4->y1 (2 bits)
  vstpq_u8<32>(out, y1);

  uint8x16x2_t x1 = vldpq_u8<32>(in);
  uint8x16x2_t y2 = vandq_u8_x2(x1, mask5); // x1->y2 (5 bits)
  vstpq_u8<64>(out, y2);

  uint8x16x2_t y3 = vshrq_n_u8_x2<3>(x1);              // x1->y3 (3 bits)
  y3 = vbitq_n_u8_x2(y3, vshrq_n_u8_x2<2>(x4), mask2); // x4->y3 (2 bits)
  vstpq_u8<96>(out, y3);

  uint8x16x2_t x2 = vldpq_u8<64>(in);
  uint8x16x2_t y4 = vandq_u8_x2(x2, mask5); // x2->y4 (5 bits)
  vstpq_u8<128>(out, y4);

  uint8x16x2_t y5 = vshrq_n_u8_x2<3>(x2);              // x2->y5 (3 bits)
  y5 = vbitq_n_u8_x2(y5, vshrq_n_u8_x2<4>(x4), mask2); // x4->y5 (2 bits)
  vstpq_u8<160>(out, y5);

  uint8x16x2_t x3 = vldpq_u8<96>(in);
  uint8x16x2_t y6 = vandq_u8_x2(x3, mask5); // x3->y6 (5 bits)
  vstpq_u8<192>(out, y6);

  uint8x16x2_t y7 = vshrq_n_u8_x2<3>(x3); // x3->y7 (3 bits)
  y7 = vsriq_n_u8_x2<6>(y7, x4);          // x4->y7 (2 bits)
  vstpq_u8<224>(out, y7);
}

static inline void unpack6_8_128(const uint8_t* __restrict__ in, uint8_t* __restrict__ out) {
  // y0 := x0:__543210
  // y1 := x0:54______ + x2:____3210
  // y2 := x1:__543210
  // y3 := x1:54______ + x2:3210____

  uint8x16_t mask4 = vdupq_n_u8(0b00001111);
  uint8x16_t mask6 = vdupq_n_u8(0b00111111);

  uint8x16x2_t x0 = vldpq_u8<0>(in);
  uint8x16x2_t y0 = vandq_u8_x2(x0, mask6); // x0->y0 (6 bits)
  vstpq_u8<0>(out, y0);

  uint8x16x2_t x2 = vldpq_u8<64>(in);
  uint8x16x2_t y1 = vshrq_n_u8_x2<2>(x0); // x0->y1 (2 bits)
  y1 = vbitq_n_u8_x2(y1, x2, mask4);      // x2->y1 (4 bits)
  vstpq_u8<32>(out, y1);

  uint8x16x2_t x1 = vldpq_u8<32>(in);
  uint8x16x2_t y2 = vandq_u8_x2(x1, mask6); // x1->y2 (6 bits)
  vstpq_u8<64>(out, y2);

  uint8x16x2_t y3 = vshrq_n_u8_x2<2>(x1); // x1->y3 (2 bits)
  y3 = vsriq_n_u8_x2<4>(y3, x2);          // x2->y3 (4 bits)
  vstpq_u8<96>(out, y3);
}

static inline void unpack7_8_256(const uint8_t* __restrict__ in, uint8_t* __restrict__ out) {
  // y0 := x0:_6543210
  // y1 := x0:6_______ + x4:__543210
  // y2 := x1:_6543210
  // y3 := x1:6_______ + x5:__543210
  // y4 := x2:_6543210
  // y5 := x2:6_______ + x4:54______ + x6:____3210
  // y6 := x3:_6543210
  // y7 := x3:6_______ + x5:54______ + x6:3210____

  uint8x16_t mask4 = vdupq_n_u8(0b00001111);
  uint8x16_t mask6 = vdupq_n_u8(0b00111111);
  uint8x16_t mask7 = vdupq_n_u8(0b01111111);

  uint8x16x2_t x0 = vldpq_u8<0>(in);
  uint8x16x2_t y0 = vandq_u8_x2(x0, mask7); // x0->y0 (7 bits)
  vstpq_u8<0>(out, y0);

  uint8x16x2_t x4 = vldpq_u8<128>(in);
  uint8x16x2_t y1 = vshrq_n_u8_x2<1>(x0); // x0->y1 (1 bit)
  y1 = vbitq_n_u8_x2(y1, x4, mask6);      // x4->y1 (6 bits)
  vstpq_u8<32>(out, y1);

  uint8x16x2_t x1 = vldpq_u8<32>(in);
  uint8x16x2_t y2 = vandq_u8_x2(x1, mask7); // x1->y2 (7 bits)
  vstpq_u8<64>(out, y2);

  uint8x16x2_t x5 = vldpq_u8<160>(in);
  uint8x16x2_t y3 = vshrq_n_u8_x2<1>(x1); // x1->y3 (1 bit)
  y3 = vbitq_n_u8_x2(y3, x5, mask6);      // x5->y3 (6 bits)
  vstpq_u8<96>(out, y3);

  uint8x16x2_t x2 = vldpq_u8<64>(in);
  uint8x16x2_t x6 = vldpq_u8<192>(in);
  uint8x16x2_t y4 = vandq_u8_x2(x2, mask7); // x2->y4 (7 bits)
  vstpq_u8<128>(out, y4);

  x2 = vsriq_n_u8_x2<1>(x2, x4);          // x4->x2 (2 bits)
  uint8x16x2_t y5 = vshrq_n_u8_x2<1>(x2); // x2->y5 (3 bits)
  y5 = vbitq_n_u8_x2(y5, x6, mask4);      // x6->y5 (4 bits)
  vstpq_u8<160>(out, y5);

  uint8x16x2_t x3 = vldpq_u8<96>(in);
  uint8x16x2_t y6 = vandq_u8_x2(x3, mask7); // x3->y6 (7 bits)
  vstpq_u8<192>(out, y6);

  x3 = vsriq_n_u8_x2<1>(x3, x5);          // x5->x3 (2 bits)
  uint8x16x2_t y7 = vshrq_n_u8_x2<1>(x3); // x3->y7 (3 bits)
  y7 = vsriq_n_u8_x2<4>(y7, x6);          // x6->y7 (4 bits)
  vstpq_u8<224>(out, y7);
}

static inline void unpack8_8_32(const uint8_t* __restrict__ in, uint8_t* __restrict__ out) {
  // y0 := x0:76543210
  vstpq_u8<0>(out, vldpq_u8<0>(in));
}

static inline void pack1_8_256(const uint8_t* __restrict__ in, uint8_t* __restrict__ out) {
  // x0:00000000 := x0:_______0 + x1:______0_ + x2:_____0__ + x3:____0___
  //              + x4:___0____ + x5:__0_____ + x6:_0______ + x7:0_______
  uint8x16x2_t x0 = vldpq_u8<0>(in);
  uint8x16x2_t x1 = vldpq_u8<32>(in);
  x0 = vsliq_n_u8_x2<1>(x0, x1); // x1->x0 (1 bit)

  uint8x16x2_t x2 = vldpq_u8<64>(in);
  uint8x16x2_t x3 = vldpq_u8<96>(in);
  x2 = vsliq_n_u8_x2<1>(x2, x3); // x3->x2 (1 bit)
  x0 = vsliq_n_u8_x2<2>(x0, x2); // x2->x0 (2 bits)

  uint8x16x2_t x4 = vldpq_u8<128>(in);
  uint8x16x2_t x5 = vldpq_u8<160>(in);
  x4 = vsliq_n_u8_x2<1>(x4, x5); // x5->x4 (1 bit)

  uint8x16x2_t x6 = vldpq_u8<192>(in);
  uint8x16x2_t x7 = vldpq_u8<224>(in);
  x6 = vsliq_n_u8_x2<1>(x6, x7); // x7->x6 (1 bit)
  x4 = vsliq_n_u8_x2<2>(x4, x6); // x6->x4 (2 bits)
  x0 = vsliq_n_u8_x2<4>(x0, x4); // x4->x0 (4 bits)

  vstpq_u8<0>(out, x0);
}

static inline void pack2_8_128(const uint8_t* __restrict__ in, uint8_t* __restrict__ out) {
  // x0:00000000 := x0:______10 + x1:____10__ + x2:__10____ + x3:10______
  uint8x16x2_t x0 = vldpq_u8<0>(in);
  uint8x16x2_t x1 = vldpq_u8<32>(in);
  x0 = vsliq_n_u8_x2<2>(x0, x1); // x1->x0 (2 bits)

  uint8x16x2_t x2 = vldpq_u8<64>(in);
  uint8x16x2_t x3 = vldpq_u8<96>(in);
  x2 = vsliq_n_u8_x2<2>(x2, x3); // x3->x2 (2 bits)

  x0 = vsliq_n_u8_x2<4>(x0, x2); // x2->x0 (4 bits)

  vstpq_u8<0>(out, x0);
}

static inline void pack3_8_256(const uint8_t* __restrict__ in, uint8_t* __restrict__ out) {
  // x0:10210210 := x0:_____210 + x1:__210___ + x2:10______
  // x3:10210210 := x3:_____210 + x4:__210___ + x5:10______
  // x6:22210210 := x6:_____210 + x7:__210___ + x2:_2______ + x5:2_______

  uint8x16_t mask6 = vdupq_n_u8(0b00111111);
  uint8x16_t mask7 = vdupq_n_u8(0b01111111);

  uint8x16x2_t x0 = vldpq_u8<0>(in);
  uint8x16x2_t x1 = vldpq_u8<32>(in);
  uint8x16x2_t x2 = vldpq_u8<64>(in);
  x0 = vsliq_n_u8_x2<3>(x0, x1); // x1->x0 (3 bits)
  x0 = vsliq_n_u8_x2<6>(x0, x2); // x2->x0 (2 bits)
  vstpq_u8<0>(out, x0);

  uint8x16x2_t x3 = vldpq_u8<96>(in);
  uint8x16x2_t x4 = vldpq_u8<128>(in);
  uint8x16x2_t x5 = vldpq_u8<160>(in);
  x3 = vsliq_n_u8_x2<3>(x3, x4); // x4->x3 (3 bits)
  x3 = vsliq_n_u8_x2<6>(x3, x5); // x5->x3 (2 bits)
  vstpq_u8<32>(out, x3);

  x2 = vbifq_n_u8_x2(vshlq_n_u8_x2<4>(x2), vshlq_n_u8_x2<5>(x5), mask7); // x5->x2 (1 bit)

  uint8x16x2_t x6 = vldpq_u8<192>(in);
  uint8x16x2_t x7 = vldpq_u8<224>(in);
  x6 = vsliq_n_u8_x2<3>(x6, x7);     // x7->x6 (3 bits)
  x6 = vbifq_n_u8_x2(x6, x2, mask6); // x2->x6 (2 bits)
  vstpq_u8<64>(out, x6);
}

static inline void pack4_8_64(const uint8_t* __restrict__ in, uint8_t* __restrict__ out) {
  // x0:43243210 := x0:____3210 + x1:3210____
  uint8x16x2_t x0 = vldpq_u8<0>(in);
  uint8x16x2_t x1 = vldpq_u8<32>(in);
  x0 = vsliq_n_u8_x2<4>(x0, x1);
  vstpq_u8<0>(out, x0);
}

static inline void pack5_8_256(const uint8_t* __restrict__ in, uint8_t* __restrict__ out) {
  // x0:43243210 := x0:___43210 + x1:432_____
  // x2:43243210 := x2:___43210 + x3:432_____
  // x4:43243210 := x4:___43210 + x5:432_____
  // x6:43243210 := x6:___43210 + x7:432_____
  // x1:10101010 := x1:______10 + x3:____10__ + x5:__10____ + x7:10______

  uint8x16_t mask5 = vdupq_n_u8(0b00011111);

  uint8x16x2_t x0 = vldpq_u8<0>(in);
  uint8x16x2_t x1 = vldpq_u8<32>(in);
  x0 = vbifq_n_u8_x2(x0, vshlq_n_u8_x2<3>(x1), mask5); // x1->x0 (3 bits)
  vstpq_u8<0>(out, x0);

  uint8x16x2_t x2 = vldpq_u8<64>(in);
  uint8x16x2_t x3 = vldpq_u8<96>(in);
  x2 = vbifq_n_u8_x2(x2, vshlq_n_u8_x2<3>(x3), mask5); // x3->x2 (3 bits)
  x1 = vsliq_n_u8_x2<2>(x1, x3);                       // x3->x1 (2 bits)
  vstpq_u8<32>(out, x2);

  uint8x16x2_t x4 = vldpq_u8<128>(in);
  uint8x16x2_t x5 = vldpq_u8<160>(in);
  x4 = vbifq_n_u8_x2(x4, vshlq_n_u8_x2<3>(x5), mask5); // x5->x4 (3 bits)
  x1 = vsliq_n_u8_x2<4>(x1, x5);                       // x5->x1 (2 bits)
  vstpq_u8<64>(out, x4);

  uint8x16x2_t x6 = vldpq_u8<192>(in);
  uint8x16x2_t x7 = vldpq_u8<224>(in);
  x6 = vbifq_n_u8_x2(x6, vshlq_n_u8_x2<3>(x7), mask5); // x7->x6 (3 bits)
  x1 = vsliq_n_u8_x2<6>(x1, x7);                       // x7->x1 (2 bits)
  vstpq_u8<96>(out, x6);
  vstpq_u8<128>(out, x1);
}

static inline void pack6_8_128(const uint8_t* __restrict__ in, uint8_t* __restrict__ out) {
  // x0:54543210 := x0:__543210 + x1:54______
  // x2:54543210 := x2:__543210 + x3:54______
  // x1:32103210 := x1:____3210 + x3:3210____

  uint8x16_t mask6 = vdupq_n_u8(0b00111111);

  uint8x16x2_t x0 = vldpq_u8<0>(in);
  uint8x16x2_t x1 = vldpq_u8<32>(in);
  x0 = vbifq_n_u8_x2(x0, vshlq_n_u8_x2<2>(x1), mask6); // x1->x0 (2 bits)
  vstpq_u8<0>(out, x0);

  uint8x16x2_t x2 = vldpq_u8<64>(in);
  uint8x16x2_t x3 = vldpq_u8<96>(in);
  x2 = vbifq_n_u8_x2(x2, vshlq_n_u8_x2<2>(x3), mask6); // x3->x2 (2 bits)
  x1 = vsliq_n_u8_x2<4>(x1, x3);                       // x3->x1 (4 bits)
  vstpq_u8<32>(out, x2);
  vstpq_u8<64>(out, x1);
}

static inline void pack7_8_256(const uint8_t* __restrict__ in, uint8_t* __restrict__ out) {
  // x0:66543210 := x0:_6543210 + x1:6_______
  // x2:66543210 := x2:_6543210 + x3:6_______
  // x4:66543210 := x4:_6543210 + x5:6_______
  // x6:66543210 := x6:_6543210 + x7:6_______
  // x1:54543210 := x1:__543210 + x5:54______
  // x3:54543210 := x3:__543210 + x7:54______
  // x5:32103210 := x5:____3210 + x7:3210____

  uint8x16_t mask6 = vdupq_n_u8(0b00111111);
  uint8x16_t mask7 = vdupq_n_u8(0b01111111);

  uint8x16x2_t x0 = vldpq_u8<0>(in);
  uint8x16x2_t x1 = vldpq_u8<32>(in);
  x0 = vbifq_n_u8_x2(x0, vshlq_n_u8_x2<1>(x1), mask7); // x1->x0 (1 bit)
  vstpq_u8<0>(out, x0);

  uint8x16x2_t x2 = vldpq_u8<64>(in);
  uint8x16x2_t x3 = vldpq_u8<96>(in);
  x2 = vbifq_n_u8_x2(x2, vshlq_n_u8_x2<1>(x3), mask7); // x3->x2 (1 bit)
  vstpq_u8<32>(out, x2);

  uint8x16x2_t x4 = vldpq_u8<128>(in);
  uint8x16x2_t x5 = vldpq_u8<160>(in);
  x4 = vbifq_n_u8_x2(x4, vshlq_n_u8_x2<1>(x5), mask7); // x5->x4 (1 bit)
  x1 = vbifq_n_u8_x2(x1, vshlq_n_u8_x2<2>(x5), mask6); // x5->x1 (2 bits)
  vstpq_u8<64>(out, x4);
  vstpq_u8<128>(out, x1);

  uint8x16x2_t x6 = vldpq_u8<192>(in);
  uint8x16x2_t x7 = vldpq_u8<224>(in);
  x6 = vbifq_n_u8_x2(x6, vshlq_n_u8_x2<1>(x7), mask7); // x7->x6 (1 bit)
  x3 = vbifq_n_u8_x2(x3, vshlq_n_u8_x2<2>(x7), mask6); // x7->x3 (2 bits)
  vstpq_u8<96>(out, x6);
  vstpq_u8<160>(out, x3);

  x5 = vsliq_n_u8_x2<4>(x5, x7); // x7->x5 (4 bits)
  vstpq_u8<192>(out, x5);
}

static inline void pack8_8_32(const uint8_t* __restrict__ in, uint8_t* __restrict__ out) {
  // x0:76543210 := x0:76543210
  vstpq_u8<0>(out, vldpq_u8<0>(in));
}

#define PFOR_PRAGMA_STRINGIFY(x) #x
#define PFOR_LOOP_PRAGMA_UNROLL(directive) _Pragma(PFOR_PRAGMA_STRINGIFY(directive))

template <int K, int OUT, int UNROLL, auto Blk, bool PACK>
inline void loop(const uint8_t* __restrict in, uint8_t* __restrict out, std::size_t n) {
  constexpr std::size_t BLOCK_OUT = OUT;
  constexpr std::size_t BLOCK_IN = (BLOCK_OUT * K) / 8;

  __builtin_assume(!(n & (BLOCK_OUT - 1)));
  in = static_cast<const uint8_t*>(__builtin_assume_aligned(in, 16));
  out = static_cast<uint8_t*>(__builtin_assume_aligned(out, 16));

  std::size_t blocks = n / BLOCK_OUT;

  if constexpr (PACK) {
    PFOR_LOOP_PRAGMA_UNROLL(clang loop unroll_count(UNROLL))
    for (; blocks; --blocks, in += BLOCK_OUT, out += BLOCK_IN)
      Blk(in, out);
  } else {
    PFOR_LOOP_PRAGMA_UNROLL(clang loop unroll_count(UNROLL))
    for (; blocks; --blocks, in += BLOCK_IN, out += BLOCK_OUT)
      Blk(in, out);
  }
}

#undef PFOR_PRAGMA_STRINGIFY
#undef PFOR_LOOP_PRAGMA_UNROLL

using fn_t = void (*)(const uint8_t*, uint8_t*, std::size_t);

struct Entry {
  std::uint32_t out_bytes;
  std::uint32_t unroll;
  fn_t pack_n;
  fn_t unpack_n;
};

constexpr Entry k_table[] = {
    /* 0 */ {0, 0, nullptr, nullptr},
    /* 1 */ {256, 1, loop<1, 256, 1, pack1_8_256, true>, loop<1, 256, 1, unpack1_8_256, false>},
    /* 2 */ {128, 2, loop<2, 128, 2, pack2_8_128, true>, loop<2, 128, 2, unpack2_8_128, false>},
    /* 3 */ {256, 1, loop<3, 256, 1, pack3_8_256, true>, loop<3, 256, 1, unpack3_8_256, false>},
    /* 4 */ {64, 4, loop<4, 64, 4, pack4_8_64, true>, loop<4, 64, 4, unpack4_8_64, false>},
    /* 5 */ {256, 1, loop<5, 256, 1, pack5_8_256, true>, loop<5, 256, 1, unpack5_8_256, false>},
    /* 6 */ {128, 2, loop<6, 128, 2, pack6_8_128, true>, loop<6, 128, 2, unpack6_8_128, false>},
    /* 7 */ {256, 1, loop<7, 256, 1, pack7_8_256, true>, loop<7, 256, 1, unpack7_8_256, false>},
    /* 8 */ {32, 8, loop<8, 32, 8, pack8_8_32, true>, loop<8, 32, 8, unpack8_8_32, false>},
};

} // namespace impl

/* ───────────────────────  public front-ends  ─────────────────────────────── */

void pack(const uint8_t* in, uint8_t* out, std::uint32_t bit, std::uint32_t n) {
  if (bit == 0 || bit > 8)
    return;
  impl::k_table[bit].pack_n(in, out, n);
  // Memory barrier prevents LLVM's common subexpression elimination,
  // which interferes with other optimisation passes if applied
  asm volatile("" ::: "memory");
}

void unpack(const uint8_t* in, uint8_t* out, std::uint32_t bit, std::uint32_t n) {
  if (bit == 0 || bit > 8)
    return;
  impl::k_table[bit].unpack_n(in, out, n);
  asm volatile("" ::: "memory");
}
} // namespace NeonPForLib
