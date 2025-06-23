#include "bitpack_test.h"
#include "delta_test.h"

#include <iostream>
#include <string>
#include <vector>

// ──────────────────────────────────────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <cmd> <operation> <impl>\n";
    std::cerr << "  cmd : check | benchmark | all\n";
    std::cerr << "  operation : bitpack | delta | all\n";
    std::cerr << "  implementation : fastpfor | naive | neonpfor | all" << std::endl;
    return 1;
  }

  std::string cmd = argv[1];
  std::string operation = argv[2];
  std::string implArg = argv[3];

  std::vector<std::string> cmds;
  if (cmd == "all") {
    cmds = {"check", "benchmark"};
  } else {
    cmds = {cmd};
  }

  std::vector<std::string> operations;
  if (operation == "all") {
    operations = {"bitpack", "delta"};
  } else {
    operations = {operation};
  }

  std::vector<std::string> impls;
  if (implArg == "all") {
    impls = {"fastpfor", "naive", "neonpfor"};
  } else {
    impls = {implArg};
  }

  // For bitpack testing, we also need to specify k values (bit widths)
  std::vector<uint32_t> ks;
  for (uint32_t k = 1; k <= 10; ++k) {
    ks.push_back(k);
  }

  for (const auto& currentCmd : cmds) {
    for (const auto& currentOp : operations) {
      std::cout << "=== " << currentCmd << " " << currentOp << " ===" << std::endl;

      if (currentOp == "bitpack") {
        if (currentCmd == "check") {
          bool passed = NeonPForLib::Testing::testBitpack(ks, impls);
          if (!passed) {
            std::cout << "Some bitpack tests failed!" << std::endl;
          }
        } else if (currentCmd == "benchmark") {
          NeonPForLib::Testing::benchmarkBitpack(ks, impls);
        }
      } else if (currentOp == "delta") {
        if (currentCmd == "check") {
          bool passed = NeonPForLib::Testing::testDelta(impls);
          if (!passed) {
            std::cout << "Some delta tests failed!" << std::endl;
          }
        } else if (currentCmd == "benchmark") {
          NeonPForLib::Testing::benchmarkDelta(impls);
        }
      }

      std::cout << std::endl;
    }
  }

  return 0;
}
