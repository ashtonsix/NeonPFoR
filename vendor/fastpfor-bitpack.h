/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * (c) Daniel Lemire, modified by Ashton Six
 */
#ifndef FASTPFOR_BITPACKING_H_
#define FASTPFOR_BITPACKING_H_

#include <iostream>
#include <stdexcept>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// For SIMDe implementation of SSE intrinsics
#ifndef SIMDE_ENABLE_NATIVE_ALIASES
#define SIMDE_ENABLE_NATIVE_ALIASES
#endif /* SIMDE_ENABLE_NATIVE_ALIASES */
#ifdef __SSE4_1__
#include <smmintrin.h>
#else
#include <simde/x86/sse4.1.h>
#endif

namespace FastPForLib {

void simdpack(const uint32_t *__restrict__ in, __m128i *__restrict__ out,
              uint32_t bit);
void simdunpack(const __m128i *__restrict__ in, uint32_t *__restrict__ out,
                uint32_t bit);

} // namespace FastPForLib

#endif /* FASTPFOR_BITPACKING_H_ */
