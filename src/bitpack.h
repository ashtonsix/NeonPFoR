#ifndef NEON_PFOR_BITPACK_H_
#define NEON_PFOR_BITPACK_H_

#include <stdexcept>
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

void pack(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, uint32_t k_in, uint32_t k_out, uint32_t n);
void unpack(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, uint32_t k_in, uint32_t k_out, uint32_t n);

} // namespace NeonPForLib

#endif /* NEON_PFOR_BITPACK_H_ */
