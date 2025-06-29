/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * (c) Daniel Lemire and Leonid Boytsov, modified by Ashton Six
 */

#include "delta.h"

namespace FastPForLib {

void Delta::encodeDelta1_32(uint32_t* pData, const size_t TotalQty) {
  const size_t Qty4 = TotalQty / 4;
  __m128i* pCurr = reinterpret_cast<__m128i*>(pData);
  const __m128i* pEnd = pCurr + Qty4;

  __m128i last = _mm_setzero_si128();
  while (pCurr < pEnd) {
    __m128i a0 = _mm_loadu_si128(pCurr);
    __m128i a1 = _mm_sub_epi32(a0, _mm_srli_si128(last, 12));
    a1 = _mm_sub_epi32(a1, _mm_slli_si128(a0, 4));
    last = a0;

    _mm_storeu_si128(pCurr++, a1);
  }

  if (Qty4 * 4 < TotalQty) {
    uint32_t lastVal = _mm_cvtsi128_si32(_mm_srli_si128(last, 12));
    for (size_t i = Qty4 * 4; i < TotalQty; ++i) {
      uint32_t newVal = pData[i];
      pData[i] -= lastVal;
      lastVal = newVal;
    }
  }
}

void Delta::encodeDelta4_32(uint32_t* pData, const size_t TotalQty) {
  const size_t Qty4 = TotalQty / 4;
  for (size_t i = 4 * Qty4; i < TotalQty; ++i) {
    pData[i] -= pData[i - 4];
  }
  __m128i* pCurr = reinterpret_cast<__m128i*>(pData) + Qty4 - 1;
  const __m128i* pStart = reinterpret_cast<__m128i*>(pData);
  __m128i a = _mm_loadu_si128(pCurr);
  while (pCurr > pStart) {
    __m128i b = _mm_loadu_si128(pCurr - 1);
    _mm_storeu_si128(pCurr--, _mm_sub_epi32(a, b));
    a = b;
  }
}

void Delta::decodeDelta1_32(uint32_t* pData, const size_t TotalQty) {
  const size_t Qty4 = TotalQty / 4;

  __m128i runningCount = _mm_setzero_si128();
  __m128i* pCurr = reinterpret_cast<__m128i*>(pData);
  const __m128i* pEnd = pCurr + Qty4;
  while (pCurr < pEnd) {
    __m128i a0 = _mm_loadu_si128(pCurr);
    __m128i a1 = _mm_add_epi32(_mm_slli_si128(a0, 8), a0);
    __m128i a2 = _mm_add_epi32(_mm_slli_si128(a1, 4), a1);
    a0 = _mm_add_epi32(a2, runningCount);
    runningCount = _mm_shuffle_epi32(a0, 0xFF);
    _mm_storeu_si128(pCurr++, a0);
  }

  for (size_t i = Qty4 * 4; i < TotalQty; ++i) {
    pData[i] += pData[i - 1];
  }
}

void Delta::decodeDelta4_32(uint32_t* pData, const size_t TotalQty) {
  const size_t Qty4 = TotalQty / 4;

  __m128i* pCurr = reinterpret_cast<__m128i*>(pData);
  const __m128i* pEnd = pCurr + Qty4;
  __m128i a = _mm_loadu_si128(pCurr++);
  while (pCurr < pEnd) {
    __m128i b = _mm_loadu_si128(pCurr);
    a = _mm_add_epi32(a, b);
    _mm_storeu_si128(pCurr++, a);
  }

  for (size_t i = Qty4 * 4; i < TotalQty; ++i) {
    pData[i] += pData[i - 4];
  }
}

} // namespace FastPForLib
