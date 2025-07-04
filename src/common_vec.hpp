// vec.hpp
#pragma once
#include <arm_neon.h>
#include <arm_neon_sve_bridge.h>
#include <arm_sve.h>
#include <cassert>

namespace VecNeon {

// vbitq_n_u8 / vbifq_n_u8
// These bitwise ops are not exposed as intrinsics but trivially implemented,
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

// vldpq_u8 / vstpq_u8 (⚠️🤬😭🤬😭⚠️)
//
// These functions encourage load/store pair fusion (ldp/stp) for better performance.
// We use consecutive vld1q_u8/vst1q_u8 calls since ldp/stp aren't directly exposed
// as intrinsics.
//
// Challenge: LLVM's instruction reordering pass (MachineScheduler) runs before and
// disrupts AArch64LoadStoreOpt. The WithReorderBarrier param gives options here:
//   - With barriers: Guarantees pair fusion but limits latency-hiding scheduling
//   - Without barriers: Allows latency-hiding scheduling but fusion isn't reliable
//
// Inline ASM isn't viable, because it precludes AArch64AddressingModes optimisation.
// The vld1q_u8_x2 intrinsic isn't viable either, because of poor µ-arch performance.
//
// Note: on Apple chips specifically, two ld1 calls are typically preferred to ldp.
// Note: atomic_signal_fence and volatile +r hints also garuntee fusion, but disrupt
//       ALU interleaving in MachineScheduler, thus we prefer memory clobbers
// Perf: for some kernels WithReorderBarrier improves throughput by 20%, on others it
//       worsens throughput by 5%

// load
template <int Offset, bool WithReorderBarrier = true /* recommended=true */>
static inline uint8x16x2_t vldpq_u8(const uint8_t* in) {
  static_assert((Offset & 15) == 0, "must be 16-byte aligned");
  if constexpr (WithReorderBarrier)
    asm volatile("" ::: "memory");
  uint8x16x2_t result;
  result.val[0] = vld1q_u8(in + Offset);
  result.val[1] = vld1q_u8(in + Offset + 16);
  if constexpr (WithReorderBarrier)
    asm volatile("" ::: "memory");
  return result;
}

// store
template <int Offset, bool WithReorderBarrier = false /* recommended=false */>
static inline void vstpq_u8(uint8_t* out, uint8x16x2_t data) {
  static_assert((Offset & 15) == 0, "must be 16-byte aligned");
  if constexpr (WithReorderBarrier)
    asm volatile("" ::: "memory");
  vst1q_u8(out + Offset, data.val[0]);
  vst1q_u8(out + Offset + 16, data.val[1]);
  if constexpr (WithReorderBarrier)
    asm volatile("" ::: "memory");
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

static inline uint8x16x2_t vxtnq_u16_x2(uint8x16x2_t a, uint8x16x2_t b) {
  uint8x16x2_t result;
  result.val[0] = vmovn_high_u16(vmovn_u16(a.val[0]), a.val[1]);
  result.val[1] = vmovn_high_u16(vmovn_u16(b.val[0]), b.val[1]);
  return result;
}

static inline uint8x16x2_t vxtnq_u32_x2(uint8x16x2_t a, uint8x16x2_t b) {
  uint8x16x2_t result;
  result.val[0] = vmovn_high_u32(vmovn_u32(a.val[0]), a.val[1]);
  result.val[1] = vmovn_high_u32(vmovn_u32(b.val[0]), b.val[1]);
  return result;
}

static inline uint8x16x2_t vxtnq_u64_x2(uint8x16x2_t a, uint8x16x2_t b) {
  uint8x16x2_t result;
  result.val[0] = vmovn_high_u64(vmovn_u64(a.val[0]), a.val[1]);
  result.val[1] = vmovn_high_u64(vmovn_u64(b.val[0]), b.val[1]);
  return result;
}

static inline uint8x16x2_t vtrn1q_u8_x2(uint8x16x2_t a, uint8x16x2_t b) {
  uint8x16x2_t result;
  result.val[0] = vtrn1q_u8(a.val[0], b.val[0]);
  result.val[1] = vtrn1q_u8(a.val[1], b.val[1]);
  return result;
}

static inline uint8x16x2_t vtrn2q_u8_x2(uint8x16x2_t a, uint8x16x2_t b) {
  uint8x16x2_t result;
  result.val[0] = vtrn2q_u8(a.val[0], b.val[0]);
  result.val[1] = vtrn2q_u8(a.val[1], b.val[1]);
  return result;
}

static inline uint8x16x2_t vtrn1q_u16_x2(uint8x16x2_t a, uint8x16x2_t b) {
  uint8x16x2_t result;
  result.val[0] = vtrn1q_u16(a.val[0], b.val[0]);
  result.val[1] = vtrn1q_u16(a.val[1], b.val[1]);
  return result;
}

static inline uint8x16x2_t vtrn2q_u16_x2(uint8x16x2_t a, uint8x16x2_t b) {
  uint8x16x2_t result;
  result.val[0] = vtrn2q_u16(a.val[0], b.val[0]);
  result.val[1] = vtrn2q_u16(a.val[1], b.val[1]);
  return result;
}

static inline uint8x16x2_t vtrn1q_u32_x2(uint8x16x2_t a, uint8x16x2_t b) {
  uint8x16x2_t result;
  result.val[0] = vtrn1q_u32(a.val[0], b.val[0]);
  result.val[1] = vtrn1q_u32(a.val[1], b.val[1]);
  return result;
}

static inline uint8x16x2_t vtrn2q_u32_x2(uint8x16x2_t a, uint8x16x2_t b) {
  uint8x16x2_t result;
  result.val[0] = vtrn2q_u32(a.val[0], b.val[0]);
  result.val[1] = vtrn2q_u32(a.val[1], b.val[1]);
  return result;
}

static inline uint8x16x2_t vzip1q_u8_x2(uint8x16x2_t a, uint8x16x2_t b) {
  uint8x16x2_t result;
  result.val[0] = vzip1q_u8(a.val[0], b.val[0]);
  result.val[1] = vzip1q_u8(a.val[1], b.val[1]);
  return result;
}

static inline uint8x16x2_t vzip2q_u8_x2(uint8x16x2_t a, uint8x16x2_t b) {
  uint8x16x2_t result;
  result.val[0] = vzip2q_u8(a.val[0], b.val[0]);
  result.val[1] = vzip2q_u8(a.val[1], b.val[1]);
  return result;
}

} // namespace VecNeon
