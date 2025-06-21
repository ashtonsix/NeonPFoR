/**
 * Naive scalar delta encoding/decoding implementation
 * Simple loop-based approach that surprisingly
 * outperforms FastPFoR's SIMD implementation
 */

#include "naive-delta.h"

namespace NaiveDeltaLib {

void Delta::encodeDelta1_32(uint32_t* pData, const size_t TotalQty) {
  if (TotalQty == 0)
    return;

  for (size_t i = TotalQty - 1; i > 0; --i) {
    pData[i] -= pData[i - 1];
  }
}

void Delta::encodeDelta4_32(uint32_t* pData, const size_t TotalQty) {
  if (TotalQty < 4)
    return;

  for (size_t i = TotalQty - 1; i >= 4; --i) {
    pData[i] -= pData[i - 4];
  }
}

void Delta::decodeDelta1_32(uint32_t* pData, const size_t TotalQty) {
  if (TotalQty == 0)
    return;

  for (size_t i = 1; i < TotalQty; ++i) {
    pData[i] += pData[i - 1];
  }
}

void Delta::decodeDelta4_32(uint32_t* pData, const size_t TotalQty) {
  if (TotalQty < 4)
    return;

  for (size_t i = 4; i < TotalQty; ++i) {
    pData[i] += pData[i - 4];
  }
}

} // namespace NaiveDeltaLib