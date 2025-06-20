# NeonPFoR

> Alternative to FastPFoR, tailored for ARM NEON

## Prerequisites

1. Get source code:

   ```sh
   # Clone *and* pull submodules in one go
   git clone --recurse-submodules https://github.com/ashtonsix/neon-pfor

   # (Or, if you already cloned)
   git submodule update --init --recursive
   ```

2. Open `NeonPFoR.code-workspace` in VSCode or Cursor
3. Click "Reopen in Container", which will configure build toolchain

## Running tests and benchmarks

```sh
make && ./build/bitpack-inspect all all all
```
