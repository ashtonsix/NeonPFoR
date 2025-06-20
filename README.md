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

Benchmarking fastpfor pack   (k=1) :   12.536 ns/block,  10210.5 M int/s, 40.842 GB/s in,  1.276 GB/s out
Benchmarking fastpfor unpack (k=1) :   10.810 ns/block,  11840.9 M int/s,  1.480 GB/s in, 47.364 GB/s out
Benchmarking fastpfor pack   (k=2) :   11.007 ns/block,  11628.8 M int/s, 46.515 GB/s in,  2.907 GB/s out
Benchmarking fastpfor unpack (k=2) :    9.700 ns/block,  13196.3 M int/s,  3.299 GB/s in, 52.785 GB/s out
Benchmarking fastpfor pack   (k=3) :   12.667 ns/block,  10105.1 M int/s, 40.420 GB/s in,  3.789 GB/s out
Benchmarking fastpfor unpack (k=3) :    9.302 ns/block,  13760.5 M int/s,  5.160 GB/s in, 55.042 GB/s out
Benchmarking fastpfor pack   (k=4) :   10.091 ns/block,  12684.9 M int/s, 50.740 GB/s in,  6.342 GB/s out
Benchmarking fastpfor unpack (k=4) :    8.641 ns/block,  14813.0 M int/s,  7.407 GB/s in, 59.252 GB/s out
Benchmarking fastpfor pack   (k=5) :   13.459 ns/block,   9510.3 M int/s, 38.041 GB/s in,  5.944 GB/s out
Benchmarking fastpfor unpack (k=5) :    9.960 ns/block,  12852.0 M int/s,  8.033 GB/s in, 51.408 GB/s out
Benchmarking fastpfor pack   (k=6) :   11.842 ns/block,  10809.1 M int/s, 43.236 GB/s in,  8.107 GB/s out
Benchmarking fastpfor unpack (k=6) :    9.705 ns/block,  13189.2 M int/s,  9.892 GB/s in, 52.757 GB/s out
Benchmarking fastpfor pack   (k=7) :   13.901 ns/block,   9207.7 M int/s, 36.831 GB/s in,  8.057 GB/s out
Benchmarking fastpfor unpack (k=7) :   10.344 ns/block,  12374.2 M int/s, 10.827 GB/s in, 49.497 GB/s out
Benchmarking fastpfor pack   (k=8) :    9.141 ns/block,  14002.9 M int/s, 56.011 GB/s in, 14.003 GB/s out
Benchmarking fastpfor unpack (k=8) :    9.305 ns/block,  13756.1 M int/s, 13.756 GB/s in, 55.025 GB/s out

Benchmarking neonpfor pack   (k=1) :    3.411 ns/block,  75050.7 M int/s, 75.051 GB/s in,  9.381 GB/s out
Benchmarking neonpfor unpack (k=1) :    4.528 ns/block,  56532.8 M int/s,  7.067 GB/s in, 56.533 GB/s out
Benchmarking neonpfor pack   (k=2) :    2.077 ns/block,  61627.0 M int/s, 61.627 GB/s in, 15.407 GB/s out
Benchmarking neonpfor unpack (k=2) :    2.453 ns/block,  52173.3 M int/s, 13.043 GB/s in, 52.173 GB/s out
Benchmarking neonpfor pack   (k=3) :    3.927 ns/block,  65188.6 M int/s, 65.189 GB/s in, 24.446 GB/s out
Benchmarking neonpfor unpack (k=3) :    4.712 ns/block,  54327.9 M int/s, 20.373 GB/s in, 54.328 GB/s out
Benchmarking neonpfor pack   (k=4) :    1.818 ns/block,  35208.9 M int/s, 35.209 GB/s in, 17.604 GB/s out
Benchmarking neonpfor unpack (k=4) :    1.968 ns/block,  32527.3 M int/s, 16.264 GB/s in, 32.527 GB/s out
Benchmarking neonpfor pack   (k=5) :    4.183 ns/block,  61202.2 M int/s, 61.202 GB/s in, 38.251 GB/s out
Benchmarking neonpfor unpack (k=5) :    4.798 ns/block,  53358.4 M int/s, 33.349 GB/s in, 53.358 GB/s out
Benchmarking neonpfor pack   (k=6) :    2.736 ns/block,  46786.6 M int/s, 46.787 GB/s in, 35.090 GB/s out
Benchmarking neonpfor unpack (k=6) :    2.650 ns/block,  48305.4 M int/s, 36.229 GB/s in, 48.305 GB/s out
Benchmarking neonpfor pack   (k=7) :    4.734 ns/block,  54074.8 M int/s, 54.075 GB/s in, 47.315 GB/s out
Benchmarking neonpfor unpack (k=7) :    5.075 ns/block,  50443.9 M int/s, 44.138 GB/s in, 50.444 GB/s out
Benchmarking neonpfor pack   (k=8) :    2.006 ns/block,  15951.9 M int/s, 15.952 GB/s in, 15.952 GB/s out
Benchmarking neonpfor unpack (k=8) :    1.864 ns/block,  17170.8 M int/s, 17.171 GB/s in, 17.171 GB/s out
```
