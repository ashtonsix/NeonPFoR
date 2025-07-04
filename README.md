# NeonPFoR

> ⚠️ This is a RESEARCH PREVIEW. The API and internal format are not yet stable, and the library is not recommended for production use yet. x86 is not yet supported.
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

bench neonpfor pack   (k=8→1)  :  80866.4 M int/s, 80.866 GB/s in, 10.108 GB/s out
bench neonpfor pack   (k=8→2)  :  96123.8 M int/s, 96.124 GB/s in, 24.031 GB/s out
bench neonpfor pack   (k=8→3)  :  82669.5 M int/s, 82.669 GB/s in, 31.001 GB/s out
bench neonpfor pack   (k=8→4)  :  87948.9 M int/s, 87.949 GB/s in, 43.974 GB/s out
bench neonpfor pack   (k=8→5)  :  60202.8 M int/s, 60.203 GB/s in, 37.627 GB/s out
bench neonpfor pack   (k=8→6)  :  58739.7 M int/s, 58.740 GB/s in, 44.055 GB/s out
bench neonpfor pack   (k=8→7)  :  53785.6 M int/s, 53.786 GB/s in, 47.062 GB/s out
bench neonpfor pack   (k=8→8)  :  73735.8 M int/s, 73.736 GB/s in, 73.736 GB/s out

bench neonpfor unpack (k=1→8)  :  64991.5 M int/s,  8.124 GB/s in, 64.992 GB/s out
bench neonpfor unpack (k=2→8)  :  50427.1 M int/s, 12.607 GB/s in, 50.427 GB/s out
bench neonpfor unpack (k=3→8)  :  59433.5 M int/s, 22.288 GB/s in, 59.434 GB/s out
bench neonpfor unpack (k=4→8)  :  64370.6 M int/s, 32.185 GB/s in, 64.371 GB/s out
bench neonpfor unpack (k=5→8)  :  52209.2 M int/s, 32.631 GB/s in, 52.209 GB/s out
bench neonpfor unpack (k=6→8)  :  52814.0 M int/s, 39.610 GB/s in, 52.814 GB/s out
bench neonpfor unpack (k=7→8)  :  52180.7 M int/s, 45.658 GB/s in, 52.181 GB/s out
bench neonpfor unpack (k=8→8)  :  73742.2 M int/s, 73.742 GB/s in, 73.742 GB/s out
```
