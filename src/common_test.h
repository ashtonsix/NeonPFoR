#ifndef NEON_PFOR_TEST_COMMON_H_
#define NEON_PFOR_TEST_COMMON_H_

#include <chrono>
#include <string>

namespace NeonPForLib {
namespace Testing {

// ──────────────────────────────────────────────────────────────────────────────
// Shared config
// ──────────────────────────────────────────────────────────────────────────────
const std::string OUT_DIR = "data";
constexpr size_t ITERATIONS_BITPACK = 1'000'000;
constexpr size_t ITERATIONS_DELTA = 100'000;
constexpr bool MEASURE_BASELINE = true; // subtract empty-loop overhead
constexpr double NS_IN_SEC = 1e9;

// ──────────────────────────────────────────────────────────────────────────────
// Baseline empty loop timing (chrono)
// ──────────────────────────────────────────────────────────────────────────────
static inline double timeEmptyLoop(size_t iterations) {
  auto t0 = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < iterations; ++i) {
    __asm__ __volatile__("" ::: "memory");
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::nano>(t1 - t0).count();
}

} // namespace Testing
} // namespace NeonPForLib

#endif /* NEON_PFOR_TEST_COMMON_H_ */