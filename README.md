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

## Running

For help: `make && ./build/inspect help`

For bitpack benchmarks: `make && ./build/inspect benchmark bitpack all`

Which should yield something like:

```txt
=== benchmark bitpack ===
bench fastpfor pack   (k=32→1) :  10194.6 M int/s, 40.779 GB/s in,  1.274 GB/s out
bench fastpfor pack   (k=32→2) :  11621.9 M int/s, 46.488 GB/s in,  2.905 GB/s out
bench fastpfor pack   (k=32→3) :  10092.2 M int/s, 40.369 GB/s in,  3.785 GB/s out
bench fastpfor pack   (k=32→4) :  12629.6 M int/s, 50.518 GB/s in,  6.315 GB/s out
bench fastpfor pack   (k=32→5) :   9511.3 M int/s, 38.045 GB/s in,  5.945 GB/s out
bench fastpfor pack   (k=32→6) :  10809.3 M int/s, 43.237 GB/s in,  8.107 GB/s out
bench fastpfor pack   (k=32→7) :   9203.8 M int/s, 36.815 GB/s in,  8.053 GB/s out
bench fastpfor pack   (k=32→8) :  13994.3 M int/s, 55.977 GB/s in, 13.994 GB/s out

bench fastpfor unpack (k=1→32) :  12196.7 M int/s,  1.525 GB/s in, 48.787 GB/s out
bench fastpfor unpack (k=2→32) :  13470.2 M int/s,  3.368 GB/s in, 53.881 GB/s out
bench fastpfor unpack (k=3→32) :  13251.6 M int/s,  4.969 GB/s in, 53.006 GB/s out
bench fastpfor unpack (k=4→32) :  14594.3 M int/s,  7.297 GB/s in, 58.377 GB/s out
bench fastpfor unpack (k=5→32) :  12685.7 M int/s,  7.929 GB/s in, 50.743 GB/s out
bench fastpfor unpack (k=6→32) :  13114.3 M int/s,  9.836 GB/s in, 52.457 GB/s out
bench fastpfor unpack (k=7→32) :  12313.7 M int/s, 10.774 GB/s in, 49.255 GB/s out
bench fastpfor unpack (k=8→32) :  14104.3 M int/s, 14.104 GB/s in, 56.417 GB/s out

bench neonpfor pack   (k=8→1)  :  80259.0 M int/s, 80.259 GB/s in, 10.032 GB/s out
bench neonpfor pack   (k=8→2)  :  79545.0 M int/s, 79.545 GB/s in, 19.886 GB/s out
bench neonpfor pack   (k=8→3)  :  68933.6 M int/s, 68.934 GB/s in, 25.850 GB/s out
bench neonpfor pack   (k=8→4)  :  87159.3 M int/s, 87.159 GB/s in, 43.580 GB/s out
bench neonpfor pack   (k=8→5)  :  59482.8 M int/s, 59.483 GB/s in, 37.177 GB/s out
bench neonpfor pack   (k=8→6)  :  59704.5 M int/s, 59.705 GB/s in, 44.778 GB/s out
bench neonpfor pack   (k=8→7)  :  53837.4 M int/s, 53.837 GB/s in, 47.108 GB/s out
bench neonpfor pack   (k=8→8)  :  72358.9 M int/s, 72.359 GB/s in, 72.359 GB/s out

bench neonpfor unpack (k=1→8)  :  64252.3 M int/s,  8.032 GB/s in, 64.252 GB/s out
bench neonpfor unpack (k=2→8)  :  62821.4 M int/s, 15.705 GB/s in, 62.821 GB/s out
bench neonpfor unpack (k=3→8)  :  54753.1 M int/s, 20.532 GB/s in, 54.753 GB/s out
bench neonpfor unpack (k=4→8)  :  66936.0 M int/s, 33.468 GB/s in, 66.936 GB/s out
bench neonpfor unpack (k=5→8)  :  58039.4 M int/s, 36.275 GB/s in, 58.039 GB/s out
bench neonpfor unpack (k=6→8)  :  54134.0 M int/s, 40.601 GB/s in, 54.134 GB/s out
bench neonpfor unpack (k=7→8)  :  48989.6 M int/s, 42.866 GB/s in, 48.990 GB/s out
bench neonpfor unpack (k=8→8)  :  72384.4 M int/s, 72.384 GB/s in, 72.384 GB/s out
```
