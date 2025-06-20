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
