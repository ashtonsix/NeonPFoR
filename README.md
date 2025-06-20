# NeonPFoR

> Alternative to FastPFoR, tailored for ARM NEON

## Prerequisites

Tested on freshly provisioned ARM Ubuntu 24 instance (AWS m8g.large). Might not work on other environments.

1. Set up the development environment:

   ```sh
   ./setup.sh
   ```

   This script will install:
   - Build tools (gcc, cmake, ninja, etc.)
   - Clang/LLVM 19 toolchain
   - SIMDe library for cross-platform SIMD
   - Development utilities
   - Optional: zsh with Oh My Zsh

2. Get source code:

   ```sh
   # Clone *and* pull submodules in one go
   git clone --recurse-submodules https://github.com/ashtonsix/neon-pfor

   # (Or, if you already cloned)
   git submodule update --init --recursive
   ```

3. Install the 'clangd' extension for C++ language support

## Running tests and benchmarks

```sh
make && ./build/bitpack-inspect all all all
```

You should see something like:

```txt
Checking fastpfor (k=1)... passed
Checking fastpfor (k=2)... passed
Checking fastpfor (k=3)... passed
Checking fastpfor (k=4)... passed
Checking fastpfor (k=5)... passed
Checking fastpfor (k=6)... passed
Checking fastpfor (k=7)... passed
Checking fastpfor (k=8)... passed

Checking neonpfor (k=1)... passed
Checking neonpfor (k=2)... passed
Checking neonpfor (k=3)... passed
Checking neonpfor (k=4)... passed
Checking neonpfor (k=5)... passed
Checking neonpfor (k=6)... passed
Checking neonpfor (k=7)... passed
Checking neonpfor (k=8)... passed

Benchmarking fastpfor pack   (k=1) :  10236.6 M int/s, 40.946 GB/s in,  1.280 GB/s out
Benchmarking fastpfor pack   (k=2) :  11638.4 M int/s, 46.553 GB/s in,  2.910 GB/s out
Benchmarking fastpfor pack   (k=3) :   9800.5 M int/s, 39.202 GB/s in,  3.675 GB/s out
Benchmarking fastpfor pack   (k=4) :  12665.4 M int/s, 50.662 GB/s in,  6.333 GB/s out
Benchmarking fastpfor pack   (k=5) :   9519.9 M int/s, 38.080 GB/s in,  5.950 GB/s out
Benchmarking fastpfor pack   (k=6) :  10813.2 M int/s, 43.253 GB/s in,  8.110 GB/s out
Benchmarking fastpfor pack   (k=7) :   9203.8 M int/s, 36.815 GB/s in,  8.053 GB/s out
Benchmarking fastpfor pack   (k=8) :  14017.6 M int/s, 56.071 GB/s in, 14.018 GB/s out

Benchmarking fastpfor unpack (k=1) :  12001.4 M int/s,  1.500 GB/s in, 48.005 GB/s out
Benchmarking fastpfor unpack (k=2) :  13338.9 M int/s,  3.335 GB/s in, 53.356 GB/s out
Benchmarking fastpfor unpack (k=3) :  13796.7 M int/s,  5.174 GB/s in, 55.187 GB/s out
Benchmarking fastpfor unpack (k=4) :  14618.0 M int/s,  7.309 GB/s in, 58.472 GB/s out
Benchmarking fastpfor unpack (k=5) :  12823.2 M int/s,  8.014 GB/s in, 51.293 GB/s out
Benchmarking fastpfor unpack (k=6) :  13150.7 M int/s,  9.863 GB/s in, 52.603 GB/s out
Benchmarking fastpfor unpack (k=7) :  12487.3 M int/s, 10.926 GB/s in, 49.949 GB/s out
Benchmarking fastpfor unpack (k=8) :  13721.0 M int/s, 13.721 GB/s in, 54.884 GB/s out

Benchmarking neonpfor pack   (k=1) :  84263.8 M int/s, 84.264 GB/s in, 10.533 GB/s out
Benchmarking neonpfor pack   (k=2) :  88411.3 M int/s, 88.411 GB/s in, 22.103 GB/s out
Benchmarking neonpfor pack   (k=3) :  77173.5 M int/s, 77.173 GB/s in, 28.940 GB/s out
Benchmarking neonpfor pack   (k=4) :  90422.2 M int/s, 90.422 GB/s in, 45.211 GB/s out
Benchmarking neonpfor pack   (k=5) :  58748.5 M int/s, 58.748 GB/s in, 36.718 GB/s out
Benchmarking neonpfor pack   (k=6) :  59027.6 M int/s, 59.028 GB/s in, 44.271 GB/s out
Benchmarking neonpfor pack   (k=7) :  53728.2 M int/s, 53.728 GB/s in, 47.012 GB/s out
Benchmarking neonpfor pack   (k=8) :  69873.3 M int/s, 69.873 GB/s in, 69.873 GB/s out

Benchmarking neonpfor unpack (k=1) :  62674.9 M int/s,  7.834 GB/s in, 62.675 GB/s out
Benchmarking neonpfor unpack (k=2) :  58799.0 M int/s, 14.700 GB/s in, 58.799 GB/s out
Benchmarking neonpfor unpack (k=3) :  57992.2 M int/s, 21.747 GB/s in, 57.992 GB/s out
Benchmarking neonpfor unpack (k=4) :  61975.2 M int/s, 30.988 GB/s in, 61.975 GB/s out
Benchmarking neonpfor unpack (k=5) :  51924.8 M int/s, 32.453 GB/s in, 51.925 GB/s out
Benchmarking neonpfor unpack (k=6) :  54377.3 M int/s, 40.783 GB/s in, 54.377 GB/s out
Benchmarking neonpfor unpack (k=7) :  51400.8 M int/s, 44.976 GB/s in, 51.401 GB/s out
Benchmarking neonpfor unpack (k=8) :  73164.3 M int/s, 73.164 GB/s in, 73.164 GB/s out
```
