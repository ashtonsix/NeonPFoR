#ifndef NEON_PFOR_DELTA_H_
#define NEON_PFOR_DELTA_H_

#include <stdint.h>
#include <stdlib.h>

#ifndef SIMDE_ENABLE_NATIVE_ALIASES
#define SIMDE_ENABLE_NATIVE_ALIASES
#endif /* SIMDE_ENABLE_NATIVE_ALIASES */
#ifdef __ARM_NEON
#include <arm_neon.h>
#else
#include <simde/arm/neon.h>
#endif

namespace NeonPForLib {

class Delta {
public:
  // Delta-1 encoding: each value becomes difference from previous value
  static void encodeDelta1_32(uint32_t* pData, const size_t TotalQty);

  // Delta-4 encoding: each value becomes difference from value 4 positions back
  static void encodeDelta4_32(uint32_t* pData, const size_t TotalQty);

  // Delta-1 decoding: reconstruct original values from delta-1 encoded data
  static void decodeDelta1_32(uint32_t* pData, const size_t TotalQty);

  // Delta-4 decoding: reconstruct original values from delta-4 encoded data
  static void decodeDelta4_32(uint32_t* pData, const size_t TotalQty);
};

} // namespace NeonPForLib

#endif /* NEON_PFOR_DELTA_H_ */