/**
 * ARM NEON-optimized delta encoding/decoding for NeonPFoR
 * Port of FastPFoR delta operations to ARM NEON
 */

#include "delta.h"

namespace NeonPForLib {

void Delta::encodeDelta1_32(uint32_t* pData, const size_t TotalQty) {
  // TODO: Implement ARM NEON version of delta-1 encoding
  // Current stub: scalar fallback
  if (TotalQty == 0)
    return;

  for (size_t i = TotalQty - 1; i > 0; --i) {
    pData[i] -= pData[i - 1];
  }
}

void Delta::encodeDelta4_32(uint32_t* pData, const size_t TotalQty) {
  // TODO: Implement ARM NEON version of delta-4 encoding
  // Current stub: scalar fallback
  if (TotalQty < 4)
    return;

  for (size_t i = TotalQty - 1; i >= 4; --i) {
    pData[i] -= pData[i - 4];
  }
}

void Delta::decodeDelta1_32(uint32_t* pData, const size_t TotalQty) {
  // TODO: Implement ARM NEON version of delta-1 decoding
  // Current stub: scalar fallback
  if (TotalQty == 0)
    return;

  for (size_t i = 1; i < TotalQty; ++i) {
    pData[i] += pData[i - 1];
  }
}

void Delta::decodeDelta4_32(uint32_t* pData, const size_t TotalQty) {
  // TODO: Implement ARM NEON version of delta-4 decoding
  // Current stub: scalar fallback
  if (TotalQty < 4)
    return;

  for (size_t i = 4; i < TotalQty; ++i) {
    pData[i] += pData[i - 4];
  }
}

} // namespace NeonPForLib