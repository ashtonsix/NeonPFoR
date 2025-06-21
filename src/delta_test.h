#ifndef NEON_PFOR_DELTA_TEST_H_
#define NEON_PFOR_DELTA_TEST_H_

#include "../vendor/fastpfor-delta.h"
#include "../vendor/naive-delta.h"
#include "delta.h"
#include "test_common.h"

#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

namespace NeonPForLib {
namespace Testing {

// ──────────────────────────────────────────────────────────────────────────────
// Delta-specific config
// ──────────────────────────────────────────────────────────────────────────────
constexpr size_t DELTA_TEST_SIZE = 4096; // number of uint32_t values for testing

// ──────────────────────────────────────────────────────────────────────────────
// DeltaSpec – in-place delta operations
// ──────────────────────────────────────────────────────────────────────────────
struct DeltaSpec {
  std::string name; // "fastpfor" | "neonpfor"
  std::function<void(uint32_t* data, size_t n)> encodeDelta1;
  std::function<void(uint32_t* data, size_t n)> decodeDelta1;
  std::function<void(uint32_t* data, size_t n)> encodeDelta4;
  std::function<void(uint32_t* data, size_t n)> decodeDelta4;
};

// ──────────────────────────────────────────────────────────────────────────────
// Factories
// ──────────────────────────────────────────────────────────────────────────────
static inline DeltaSpec makeNeonPFoRDelta() {
  DeltaSpec s;
  s.name = "neonpfor";
  s.encodeDelta1 = [](uint32_t* data, size_t n) { NeonPForLib::Delta::encodeDelta1_32(data, n); };
  s.decodeDelta1 = [](uint32_t* data, size_t n) { NeonPForLib::Delta::decodeDelta1_32(data, n); };
  s.encodeDelta4 = [](uint32_t* data, size_t n) { NeonPForLib::Delta::encodeDelta4_32(data, n); };
  s.decodeDelta4 = [](uint32_t* data, size_t n) { NeonPForLib::Delta::decodeDelta4_32(data, n); };
  return s;
}

static inline DeltaSpec makeFastPFoRDelta() {
  DeltaSpec s;
  s.name = "fastpfor";
  s.encodeDelta1 = [](uint32_t* data, size_t n) { FastPForLib::Delta::encodeDelta1_32(data, n); };
  s.decodeDelta1 = [](uint32_t* data, size_t n) { FastPForLib::Delta::decodeDelta1_32(data, n); };
  s.encodeDelta4 = [](uint32_t* data, size_t n) { FastPForLib::Delta::encodeDelta4_32(data, n); };
  s.decodeDelta4 = [](uint32_t* data, size_t n) { FastPForLib::Delta::decodeDelta4_32(data, n); };
  return s;
}

static inline DeltaSpec makeNaiveDelta() {
  DeltaSpec s;
  s.name = "naive";
  s.encodeDelta1 = [](uint32_t* data, size_t n) { NaiveDeltaLib::Delta::encodeDelta1_32(data, n); };
  s.decodeDelta1 = [](uint32_t* data, size_t n) { NaiveDeltaLib::Delta::decodeDelta1_32(data, n); };
  s.encodeDelta4 = [](uint32_t* data, size_t n) { NaiveDeltaLib::Delta::encodeDelta4_32(data, n); };
  s.decodeDelta4 = [](uint32_t* data, size_t n) { NaiveDeltaLib::Delta::decodeDelta4_32(data, n); };
  return s;
}

// ──────────────────────────────────────────────────────────────────────────────
// Test helpers
// ──────────────────────────────────────────────────────────────────────────────
static inline std::vector<uint32_t> generateTestData(size_t n) {
  std::vector<uint32_t> data(n);
  std::mt19937 gen(42); // fixed seed for reproducibility
  std::uniform_int_distribution<uint32_t> dis(0, 1000000);

  for (size_t i = 0; i < n; ++i) {
    data[i] = dis(gen);
  }

  return data;
}

static inline bool testDeltaRoundtrip(const DeltaSpec& spec, const std::string& deltaType) {
  auto originalData = generateTestData(DELTA_TEST_SIZE);
  auto testData = originalData;

  std::cout << "check " << spec.name << " delta-" << deltaType << "...";

  try {
    if (deltaType == "1") {
      spec.encodeDelta1(testData.data(), testData.size());
      spec.decodeDelta1(testData.data(), testData.size());
    } else if (deltaType == "4") {
      spec.encodeDelta4(testData.data(), testData.size());
      spec.decodeDelta4(testData.data(), testData.size());
    } else {
      std::cout << " invalid delta type\n";
      return false;
    }

    // Check if round-trip preserved the data
    bool passed = (testData == originalData);
    if (passed) {
      std::cout << " passed\n";
    } else {
      std::cout << " failed - data corruption\n";
      // Find first difference for debugging
      for (size_t i = 0; i < originalData.size(); ++i) {
        if (originalData[i] != testData[i]) {
          std::cout << "  First difference at index " << i << ": " << originalData[i] << " -> " << testData[i] << "\n";
          break;
        }
      }
    }
    return passed;
  } catch (const std::exception& e) {
    std::cout << " exception: " << e.what() << "\n";
    return false;
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// Benchmark helpers
// ──────────────────────────────────────────────────────────────────────────────
static inline void benchmarkDeltaOperation(const DeltaSpec& spec, const std::string& deltaType,
                                           const std::string& operation,
                                           const std::function<void(uint32_t*, size_t)>& fn) {
  alignas(64) std::vector<uint32_t> data = generateTestData(DELTA_TEST_SIZE);

  // Warm-up
  for (size_t i = 0; i < 10'000; ++i) {
    fn(data.data(), data.size());
  }

  const double baselineNS = MEASURE_BASELINE ? timeEmptyLoop(ITERATIONS_DELTA) : 0.0;

  const auto t0 = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < ITERATIONS_DELTA; ++i) {
    fn(data.data(), data.size());
  }
  const auto t1 = std::chrono::high_resolution_clock::now();

  const double nsTotal = std::chrono::duration<double, std::nano>(t1 - t0).count();
  const double nsPerCall = (nsTotal - baselineNS) / ITERATIONS_DELTA;

  const double intsPerSec = (static_cast<double>(DELTA_TEST_SIZE) * NS_IN_SEC) / nsPerCall;
  const double gbpsIn = static_cast<double>(DELTA_TEST_SIZE * sizeof(uint32_t)) / nsPerCall;

  std::cout << "bench " << std::left << std::setw(8) << spec.name << " delta-" << deltaType << " " << std::setw(6)
            << operation << " : " << std::fixed << std::right << std::setw(8) << std::setprecision(1)
            << intsPerSec / 1e6 << " M int/s, " << std::setw(6) << std::setprecision(3) << gbpsIn << " GB/s\n";
}

// ──────────────────────────────────────────────────────────────────────────────
// Public interface
// ──────────────────────────────────────────────────────────────────────────────
inline bool testDelta(const std::vector<std::string>& impls) {
  bool allPassed = true;

  for (const auto& impl : impls) {
    // Skip unsupported implementations
    if (impl != "fastpfor" && impl != "naive" && impl != "neonpfor") {
      continue;
    }

    auto makeSpec = (impl == "fastpfor") ? makeFastPFoRDelta : (impl == "naive") ? makeNaiveDelta : makeNeonPFoRDelta;
    auto spec = makeSpec();

    if (!testDeltaRoundtrip(spec, "1")) {
      allPassed = false;
    }
    if (!testDeltaRoundtrip(spec, "4")) {
      allPassed = false;
    }
  }

  return allPassed;
}

inline void benchmarkDelta(const std::vector<std::string>& impls) {
  for (const auto& impl : impls) {
    // Skip unsupported implementations
    if (impl != "fastpfor" && impl != "naive" && impl != "neonpfor") {
      continue;
    }

    auto makeSpec = (impl == "fastpfor") ? makeFastPFoRDelta : (impl == "naive") ? makeNaiveDelta : makeNeonPFoRDelta;
    auto spec = makeSpec();

    benchmarkDeltaOperation(spec, "1", "encode", spec.encodeDelta1);
    benchmarkDeltaOperation(spec, "1", "decode", spec.decodeDelta1);
    benchmarkDeltaOperation(spec, "4", "encode", spec.encodeDelta4);
    benchmarkDeltaOperation(spec, "4", "decode", spec.decodeDelta4);

    std::cout << std::endl;
  }
}

} // namespace Testing
} // namespace NeonPForLib

#endif /* NEON_PFOR_DELTA_TEST_H_ */