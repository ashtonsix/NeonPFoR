# Compiler and flags
CC = clang
CXX = clang++
CFLAGS = -O3 -g -Wall -Wextra
CXXFLAGS = $(CFLAGS) -std=c++17

# Tuning flags for NeonPFoR
CUSTOM_FLAGS = -march=armv9-a -mcpu=neoverse-v2 -mtune=neoverse-v2

# TypeScript runtime
BUN = /home/developer/.bun/bin/bun

# Directories
DATA_DIR = data
BUILD_DIR = build

# SIMDe include path
SIMDE_INCLUDE = -I/usr/local/include/simde

# Input files
FASTPFOR_SRC = vendor/fastpfor-bitpack.cpp
NEONPFOR_SRC = src/bitpack.cpp
MAIN_SRC = src/main.cpp

# Output files
FASTPFOR_OBJ = $(BUILD_DIR)/fastpfor-bitpack.o
NEONPFOR_OBJ = $(BUILD_DIR)/neonpfor-bitpack.o
MAIN_OBJ = $(BUILD_DIR)/main.o
FASTPFOR_ASM = $(BUILD_DIR)/fastpfor-bitpack.s
NEONPFOR_ASM = $(BUILD_DIR)/neonpfor-bitpack.s
EXECUTABLE = $(BUILD_DIR)/bitpack-inspect

# Targets
all: clean setup $(EXECUTABLE) asm

# Create directories
setup:
	mkdir -p $(BUILD_DIR) $(DATA_DIR)

# Compile library
$(FASTPFOR_OBJ): $(FASTPFOR_SRC)
	$(CXX) $(CXXFLAGS) $(SIMDE_INCLUDE) -c $< -o $@

$(NEONPFOR_OBJ): $(NEONPFOR_SRC)
	$(CXX) $(CXXFLAGS) $(SIMDE_INCLUDE) $(CUSTOM_FLAGS) -c $< -o $@

# Compile main program
$(MAIN_OBJ): $(MAIN_SRC)
	$(CXX) $(CXXFLAGS) $(SIMDE_INCLUDE) -c $< -o $@

# Link executable
$(EXECUTABLE): $(FASTPFOR_OBJ) $(NEONPFOR_OBJ) $(MAIN_OBJ) $(INSPECT_OBJ)
	$(CXX) $(CXXFLAGS) $(CUSTOM_FLAGS) $^ -o $@

# Link test executable
$(TEST_EXECUTABLE): # $(INSPECT_TEST_OBJ) $(INSPECT_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Generate assembly for bitpack
asm: $(FASTPFOR_SRC) $(NEONPFOR_SRC)
	$(CXX) $(CXXFLAGS) $(SIMDE_INCLUDE) \
		-march=native -fverbose-asm -g0 \
		-fno-ident -fno-unwind-tables -fno-asynchronous-unwind-tables \
		-S $(FASTPFOR_SRC) -o $(FASTPFOR_ASM)
	$(CXX) $(CXXFLAGS) $(SIMDE_INCLUDE) \
		-fverbose-asm -g0 \
 		-fno-ident -fno-unwind-tables -fno-asynchronous-unwind-tables \
		$(CUSTOM_FLAGS) \
		-S $(NEONPFOR_SRC) -o $(NEONPFOR_ASM)
	sed -i '/^[ \t]*\/\/APP$$/d; /^[ \t]*\/\/NO_APP$$/d' $(FASTPFOR_ASM) $(NEONPFOR_ASM)

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) $(DATA_DIR)

.PHONY: all setup clean asm