#ifndef NEON_PFOR_BITPACK_TEST_H_
#define NEON_PFOR_BITPACK_TEST_H_

#include "../vendor/fastpfor-bitpack.h"
#include "bitpack.h"
#include "test_common.h"

#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

namespace NeonPForLib {
namespace Testing {

// Error checking types
enum class PermutationErrorCode {
  MULTIPLE_OUTPUT_BITS,
  MULTIPLE_INPUT_BITS,
  MISSING_OUTPUT_BIT,
  WRONG_OUTPUT_BIT,
  PROCEDURE_EXCEPTION
};

struct PermutationError {
  PermutationErrorCode code;
  std::string message;
};

struct ExtractResult {
  std::vector<int32_t> permutation;
  std::vector<PermutationError> errors;
};

// ──────────────────────────────────────────────────────────────────────────────
// ImplSpec – pointer-based forward/inverse
// ──────────────────────────────────────────────────────────────────────────────
struct ImplSpec {
  std::string name; // "fastpfor" | "neonpfor"
  uint32_t bitLength;
  uint32_t intsPerBlock;
  uint32_t width; // input/output width in bits

  std::function<void(const uint8_t* in, uint8_t* out)> forward;
  std::function<void(const uint8_t* in, uint8_t* out)> inverse;
};

// ──────────────────────────────────────────────────────────────────────────────
// Factories
// ──────────────────────────────────────────────────────────────────────────────
static inline ImplSpec makeNeonPFoR(uint32_t bits, uint32_t n) {
  ImplSpec s;
  s.name = "neonpfor";
  s.bitLength = n * 8;
  s.intsPerBlock = n;
  s.width = 8;

  s.forward = [bits, n](const uint8_t* in, uint8_t* out) { NeonPForLib::pack(in, out, bits, n); };
  s.inverse = [bits, n](const uint8_t* in, uint8_t* out) { NeonPForLib::unpack(in, out, bits, n); };

  return s;
}

static inline ImplSpec makeFastPFoR(uint32_t bits, uint32_t /* n */) {
  ImplSpec s;
  s.name = "fastpfor";
  s.bitLength = 128 * 32; // fastpfor always uses 128, ignore n
  s.intsPerBlock = 128;
  s.width = 32;

  s.forward = [bits](const uint8_t* in, uint8_t* out) {
    const uint32_t* ip = reinterpret_cast<const uint32_t*>(in);
    FastPForLib::simdpack(ip, reinterpret_cast<__m128i*>(out), bits);
  };
  s.inverse = [bits](const uint8_t* in, uint8_t* out) {
    const __m128i* ip = reinterpret_cast<const __m128i*>(in);
    uint32_t* op = reinterpret_cast<uint32_t*>(out);
    FastPForLib::simdunpack(ip, op, bits);
  };

  return s;
}

// ──────────────────────────────────────────────────────────────────────────────
// Error‑checking helpers – adapted to pointer‑signature
// ──────────────────────────────────────────────────────────────────────────────
static inline ExtractResult extractPermutation(uint32_t bitLength,
                                               const std::function<void(const uint8_t*, uint8_t*)>& proc) {
  ExtractResult result;

  const uint32_t byteLength = (bitLength + 7) / 8;

  std::vector<int32_t> permutation(bitLength, -1);

  alignas(64) std::vector<uint8_t> input(byteLength);
  alignas(64) std::vector<uint8_t> output(byteLength);

  for (uint32_t inputBitIndex = 0; inputBitIndex < bitLength; inputBitIndex++) {
    // Clear buffers
    std::fill(output.begin(), output.end(), 0);
    std::fill(input.begin(), input.end(), 0);

    // Set single input bit
    input[inputBitIndex / 8] |= (1u << (inputBitIndex & 7));

    // Apply the procedure and capture the output
    try {
      proc(input.data(), output.data());
    } catch (const std::exception& e) {
      PermutationError error;
      error.code = PermutationErrorCode::PROCEDURE_EXCEPTION;
      error.message = "Procedure threw an exception for bit " + std::to_string(inputBitIndex) + ": " + e.what();
      result.errors.push_back(error);
      continue;
    }

    std::vector<int32_t> setBitIndices;

    // Check each bit in the output
    for (size_t byteIndex = 0; byteIndex < output.size(); byteIndex++) {
      const uint8_t b = output[byteIndex];
      if (b == 0)
        continue;
      for (uint8_t bitPos = 0; bitPos < 8; bitPos++) {
        if (((b >> bitPos) & 1) == 1) {
          setBitIndices.push_back(static_cast<int32_t>(byteIndex * 8 + bitPos));
        }
      }
    }

    if (setBitIndices.size() > 1) {
      std::string errorMsg =
          "Procedure sets multiple output bits for single input bit " + std::to_string(inputBitIndex) + ": ";
      for (size_t i = 0; i < setBitIndices.size(); i++) {
        if (i > 0)
          errorMsg += ", ";
        errorMsg += std::to_string(setBitIndices[i]);
      }
      errorMsg += ".";

      PermutationError error;
      error.code = PermutationErrorCode::MULTIPLE_OUTPUT_BITS;
      error.message = errorMsg;
      result.errors.push_back(error);
      continue;
    }

    if (setBitIndices.size() == 1) {
      int32_t setBitIndex = setBitIndices[0];

      if (permutation[setBitIndex] != -1) {
        std::string errorMsg =
            "Procedure maps multiple input bits to the same output bit " + std::to_string(setBitIndex) +
            ". Input bits: " + std::to_string(permutation[setBitIndex]) + ", " + std::to_string(inputBitIndex) + ".";

        PermutationError error;
        error.code = PermutationErrorCode::MULTIPLE_INPUT_BITS;
        error.message = errorMsg;
        result.errors.push_back(error);
        continue;
      }

      permutation[setBitIndex] = inputBitIndex;
    }
  }

  // If there are any errors, return empty permutation
  if (!result.errors.empty()) {
    result.permutation.clear();
  } else {
    result.permutation = std::move(permutation);
  }

  return result;
}

static inline std::vector<PermutationError> checkBijectivity(const std::vector<int32_t>& forward, // f : A → B
                                                             const std::vector<int32_t>& inverse) // g : B → A
{
  std::vector<PermutationError> errs;
  if (forward.empty() || inverse.empty())
    return errs;

  // Build the reverse map of 'inverse', equal to forward if without error
  std::vector<int32_t> revInv(inverse.size(), -1);
  for (size_t inBit = 0; inBit < inverse.size(); ++inBit) {
    const int32_t outBit = inverse[inBit];
    if (outBit >= 0)
      revInv[inBit] = outBit; // g(outBit) = inBit
  }

  //  Compare f with revInv.  Two error types:
  //    1.  both map the bit but to different positions → "moved" bit
  //    2.  f maps a bit but revInv doesn't → "dropped" bit
  for (size_t outBit = 0; outBit < forward.size(); ++outBit) {
    const int32_t inBit = forward[outBit];
    if (inBit < 0)
      continue; // output bit is always 0

    /* category 1 ─ moved */
    if (inBit < static_cast<int32_t>(revInv.size()) && revInv[inBit] != -1) {
      const int32_t roundTrip = revInv[inBit];
      if (roundTrip != static_cast<int32_t>(outBit)) {
        PermutationError e;
        e.code = PermutationErrorCode::WRONG_OUTPUT_BIT;
        e.message = "Input bit " + std::to_string(inBit) + " moves from output " + std::to_string(outBit) + " to " +
                    std::to_string(roundTrip);
        errs.push_back(std::move(e));
      }
    }

    /* category 2 ─ dropped */
    else {
      PermutationError e;
      e.code = PermutationErrorCode::MISSING_OUTPUT_BIT;
      e.message =
          "Input bit " + std::to_string(inBit) + " is dropped by g∘f; expected at output " + std::to_string(outBit);
      errs.push_back(std::move(e));
    }
  }

  return errs;
}

static inline bool checkForErrors(const ImplSpec& s, uint32_t bits) {
  std::cout << "check " << s.name << " bitpack (k=" << bits << ")...";

  const auto fperm = extractPermutation(s.bitLength, s.forward);
  const auto iperm = extractPermutation(s.bitLength, s.inverse);
  const auto bijectivityErrors = checkBijectivity(fperm.permutation, iperm.permutation);

  const bool ok = fperm.errors.empty() && iperm.errors.empty() && bijectivityErrors.empty();

  if (ok) {
    std::cout << " passed\n";
    const auto base = OUT_DIR + "/" + s.name + "_bitpack_permutation_" + std::to_string(bits);
    std::ofstream bf(base + ".bin", std::ios::binary);
    bf.write(reinterpret_cast<const char*>(fperm.permutation.data()), fperm.permutation.size() * sizeof(int32_t));
  } else {
    std::cout << '\n';
    for (auto&& e : fperm.errors)
      std::cout << "  Forward error: " << e.message << '\n';
    for (auto&& e : iperm.errors)
      std::cout << "  Inverse error: " << e.message << '\n';
    for (auto&& e : bijectivityErrors)
      std::cout << "  Bijectivity error: " << e.message << '\n';
  }

  return ok;
}

// ──────────────────────────────────────────────────────────────────────────────
// Benchmark – pointer calls only
// ──────────────────────────────────────────────────────────────────────────────
static inline void runPass(const ImplSpec& spec, uint32_t k, const char* passName,
                           const std::function<void(const uint8_t*, uint8_t*)>& fn, size_t intsPerBlock, size_t bytesIn,
                           size_t bytesOut, bool isPack) {
  const size_t BUF = std::max(bytesIn, bytesOut);
  alignas(64) std::vector<uint8_t> in(BUF), out(BUF);
  std::iota(in.begin(), in.end(), 0);

  // Warm-up
  for (size_t i = 0; i < 10'000; ++i)
    fn(in.data(), out.data());

  const double baselineNS = MEASURE_BASELINE ? timeEmptyLoop(ITERATIONS_BITPACK) : 0.0;

  const auto t0 = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < ITERATIONS_BITPACK; ++i) {
    fn(in.data(), out.data());
  }
  const auto t1 = std::chrono::high_resolution_clock::now();

  const double nsTotal = std::chrono::duration<double, std::nano>(t1 - t0).count();
  const double nsPerCall = (nsTotal - baselineNS) / ITERATIONS_BITPACK;

  const double intsPerSec = (static_cast<double>(intsPerBlock) * NS_IN_SEC) / nsPerCall;
  const double gbpsIn = static_cast<double>(bytesIn) / nsPerCall;   // 1 Byte/ns = 1 GB/s
  const double gbpsOut = static_cast<double>(bytesOut) / nsPerCall; // 1 Byte/ns = 1 GB/s

  // Format width transformation: pack shows width→k, unpack shows k→width
  std::string widthDisplay;
  if (isPack) {
    widthDisplay = std::to_string(spec.width) + "→" + std::to_string(k);
  } else {
    widthDisplay = std::to_string(k) + "→" + std::to_string(spec.width);
  }

  std::cout << "bench " << std::left << std::setw(8) << spec.name << " " << std::setw(6) << passName
            << " (k=" << widthDisplay << ")"
            << std::string(std::max(0, 7 - static_cast<int>(widthDisplay.length())), ' ') << ": " << std::fixed
            << std::right << std::setw(8) << std::setprecision(1) << intsPerSec / 1e6 << " M int/s, " << std::setw(6)
            << std::setprecision(3) << gbpsIn << " GB/s in, " << std::setw(6) << std::setprecision(3) << gbpsOut
            << " GB/s out\n";
}

static inline void benchmarkPass(const ImplSpec& spec, uint32_t k, bool isPack) {
  // Geometry
  const size_t intsPerBlock = spec.intsPerBlock;
  size_t bytesInForward, bytesOutForward;

  if (spec.name == "neonpfor") {
    bytesInForward = intsPerBlock;            // 1 B per int
    bytesOutForward = (intsPerBlock * k) / 8; // packed size
  } else {                                    // fastpfor
    bytesInForward = intsPerBlock * 4;        // 32-bit ints
    bytesOutForward = (intsPerBlock * k) / 8;
  }

  if (isPack) {
    runPass(spec, k, "pack  ", spec.forward, intsPerBlock, bytesInForward, bytesOutForward, isPack);
  } else {
    runPass(spec, k, "unpack", spec.inverse, intsPerBlock, bytesOutForward, bytesInForward, isPack);
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// Public interface
// ──────────────────────────────────────────────────────────────────────────────
inline bool testBitpack(const std::vector<uint32_t>& ks, const std::vector<std::string>& impls) {
  bool allPassed = true;

  for (const auto& impl : impls) {
    // Skip unsupported implementations
    if (impl != "fastpfor" && impl != "neonpfor") {
      continue;
    }

    uint32_t n = (impl == "neonpfor") ? 256 : 128;
    auto makeSpec = (impl == "fastpfor") ? makeFastPFoR : makeNeonPFoR;

    for (auto k : ks) {
      if (!checkForErrors(makeSpec(k, n), k)) {
        allPassed = false;
      }
    }
  }

  return allPassed;
}

inline void benchmarkBitpack(const std::vector<uint32_t>& ks, const std::vector<std::string>& impls) {
  for (const auto& impl : impls) {
    // Skip unsupported implementations
    if (impl != "fastpfor" && impl != "neonpfor") {
      continue;
    }

    uint32_t n = (impl == "neonpfor") ? 4096 : 128;
    auto makeSpec = (impl == "fastpfor") ? makeFastPFoR : makeNeonPFoR;

    for (auto k : ks) {
      benchmarkPass(makeSpec(k, n), k, true);
    }
    std::cout << std::endl;

    for (auto k : ks) {
      benchmarkPass(makeSpec(k, n), k, false);
    }
    std::cout << std::endl;
  }
}

} // namespace Testing
} // namespace NeonPForLib

#endif /* NEON_PFOR_BITPACK_TEST_H_ */