#include "../vendor/fastpfor-bitpack.h"
#include "bitpack.h"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

// ──────────────────────────────────────────────────────────────────────────────
// Config
// ──────────────────────────────────────────────────────────────────────────────
const std::string OUT_DIR = "data";
constexpr size_t ITERATIONS = 1'000'000; // tight loop
constexpr bool MEASURE_BASELINE = true;  // subtract empty-loop overhead
constexpr double NS_IN_SEC = 1e9;

// ──────────────────────────────────────────────────────────────────────────────
//  Helpers – block geometry
// ──────────────────────────────────────────────────────────────────────────────
static size_t neonPforInts(uint32_t k) {
  switch (k) {
  case 1:
  case 3:
  case 5:
  case 7:
    return 256;
  case 2:
  case 6:
    return 128;
  case 4:
    return 64;
  case 8:
    return 32;
  default:
    return 0;
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// ImplSpec – pointer-based forward/inverse
// ──────────────────────────────────────────────────────────────────────────────
struct ImplSpec {
  std::string name; // "fastpfor" | "neonpfor"
  uint32_t bitLength;

  std::function<void(const uint8_t* in, uint8_t* out)> forward;
  std::function<void(const uint8_t* in, uint8_t* out)> inverse;
};

// ──────────────────────────────────────────────────────────────────────────────
// Factories
// ──────────────────────────────────────────────────────────────────────────────
static ImplSpec makeNeonPFoR(uint32_t bits) {
  ImplSpec s;
  s.name = "neonpfor";
  s.bitLength = 256 * 8;

  s.forward = [bits](const uint8_t* in, uint8_t* out) { NeonPForLib::pack(in, out, bits); };
  s.inverse = [bits](const uint8_t* in, uint8_t* out) { NeonPForLib::unpack(in, out, bits); };

  return s;
}

static ImplSpec makeFastPFoR(uint32_t bits) {
  ImplSpec s;
  s.name = "fastpfor";
  s.bitLength = 128 * 32;

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

static ExtractResult extractPermutation(uint32_t bitLength, const std::function<void(const uint8_t*, uint8_t*)>& proc) {
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

std::vector<PermutationError> checkBijectivity(const std::vector<int32_t>& forward, // f : A → B
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

bool checkForErrors(const ImplSpec& s, uint32_t bits) {
  std::cout << "Checking " << s.name << " (k=" << bits << ")...";

  const auto fperm = extractPermutation(s.bitLength, s.forward);
  const auto iperm = extractPermutation(s.bitLength, s.inverse);
  const auto bijectivityErrors = checkBijectivity(fperm.permutation, iperm.permutation);

  const bool ok = fperm.errors.empty() && iperm.errors.empty() && bijectivityErrors.empty();

  if (ok) {
    std::cout << " passed\n";
    const auto base = OUT_DIR + "/" + s.name + "_permutation_" + std::to_string(bits);
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
// Baseline empty loop timing (chrono)
// ──────────────────────────────────────────────────────────────────────────────
static double timeEmptyLoop(size_t iterations) {
  auto t0 = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < iterations; ++i) {
    __asm__ __volatile__("" ::: "memory");
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::nano>(t1 - t0).count();
}

// ──────────────────────────────────────────────────────────────────────────────
// Benchmark – pointer calls only
// ──────────────────────────────────────────────────────────────────────────────
static void runPass(const ImplSpec& spec, uint32_t k, const char* passName,
                    const std::function<void(const uint8_t*, uint8_t*)>& fn, size_t intsPerBlock, size_t bytesIn,
                    size_t bytesOut) {
  const size_t BUF = std::max(bytesIn, bytesOut);
  alignas(64) std::vector<uint8_t> in(BUF), out(BUF);
  std::iota(in.begin(), in.end(), 0);

  // Warm-up
  for (size_t i = 0; i < 10'000; ++i)
    fn(in.data(), out.data());

  const double baselineNS = MEASURE_BASELINE ? timeEmptyLoop(ITERATIONS) : 0.0;

  const auto t0 = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < ITERATIONS; ++i) {
    fn(in.data(), out.data());
  }
  const auto t1 = std::chrono::high_resolution_clock::now();

  const double nsTotal = std::chrono::duration<double, std::nano>(t1 - t0).count();
  const double nsPerCall = (nsTotal - baselineNS) / ITERATIONS;

  const double intsPerSec = (static_cast<double>(intsPerBlock) * NS_IN_SEC) / nsPerCall;
  const double gbpsIn = static_cast<double>(bytesIn) / nsPerCall;   // 1 Byte/ns = 1 GB/s
  const double gbpsOut = static_cast<double>(bytesOut) / nsPerCall; // 1 Byte/ns = 1 GB/s
  std::cout << "Benchmarking " << std::left << std::setw(8) << spec.name << ' ' << std::setw(6) << passName
            << " (k=" << k << ") : " << std::fixed << std::setprecision(3) << std::right << std::setw(8) << nsPerCall
            << " ns/block, " << std::setw(8) << std::setprecision(1) << intsPerSec / 1e6 << " M int/s, " << std::setw(6)
            << std::setprecision(3) << gbpsIn << " GB/s in, " << std::setw(6) << std::setprecision(3) << gbpsOut
            << " GB/s out\n";
}

static void benchmark(const ImplSpec& spec, uint32_t k) {
  // Geometry
  size_t intsPerBlock, bytesInForward, bytesOutForward;

  if (spec.name == "neonpfor") {
    intsPerBlock = neonPforInts(k);
    bytesInForward = intsPerBlock;            // 1 B per int
    bytesOutForward = (intsPerBlock * k) / 8; // packed size
  } else {                                    // fastpfor
    intsPerBlock = 128;
    bytesInForward = intsPerBlock * 4; // 32-bit ints
    bytesOutForward = (intsPerBlock * k) / 8;
  }

  const size_t bytesInInverse = bytesOutForward;
  const size_t bytesOutInverse = bytesInForward;

  runPass(spec, k, "pack  ", spec.forward, intsPerBlock, bytesInForward, bytesOutForward);
  runPass(spec, k, "unpack", spec.inverse, intsPerBlock, bytesInInverse, bytesOutInverse);
}

// ──────────────────────────────────────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <cmd> <bits> <impl>\n";
    std::cerr << "  cmd : check | benchmark | all\n";
    std::cerr << "  bits : 1 | 2 | ... | 8 | all" << std::endl;
    std::cerr << "  implementation : fastpfor | neonpfor | all" << std::endl;
    return 1;
  }

  std::string cmd = argv[1];
  std::string bitsArg = argv[2];
  std::string implArg = argv[3];

  std::vector<std::string> cmds;
  if (cmd == "all") {
    cmds = {"check", "benchmark"};
  } else {
    cmds = {cmd};
  }

  std::vector<std::string> impls;
  if (implArg == "all") {
    impls = {"fastpfor", "neonpfor"};
  } else {
    impls = {implArg};
  }

  std::vector<uint32_t> ks;
  if (bitsArg == "all")
    for (uint32_t k = 1; k <= 8; ++k)
      ks.push_back(k);
  else
    ks.push_back(static_cast<uint32_t>(std::stoi(bitsArg)));

  for (auto cmd : cmds) {
    for (auto impl : impls) {
      for (auto k : ks) {
        if (impl == "fastpfor") {
          auto spec = makeFastPFoR(k);
          if (cmd == "check")
            checkForErrors(spec, k);
          if (cmd == "benchmark")
            benchmark(spec, k);
        }
        if (impl == "neonpfor") {
          auto spec = makeNeonPFoR(k);
          if (cmd == "check")
            checkForErrors(spec, k);
          if (cmd == "benchmark")
            benchmark(spec, k);
        }
      }
      std::cout << std::endl;
    }
  }
}
