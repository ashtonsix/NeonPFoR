CXX = clang++
CXXFLAGS = -O3 -g -Wall -Wextra -std=c++17 -I/usr/local/include/simde
NEON_FLAGS = -march=armv9-a -mcpu=neoverse-v2 -mtune=neoverse-v2
ASM_FLAGS = -fverbose-asm -g0 -fno-ident -fno-unwind-tables -fno-asynchronous-unwind-tables

BUILD_DIR = build
VENDOR_SRCS = $(wildcard vendor/*.cpp)
NEON_SRCS = $(filter-out %/main.cpp, $(wildcard src/*.cpp))
MAIN_SRC = src/main.cpp

VENDOR_OBJS = $(VENDOR_SRCS:vendor/%.cpp=$(BUILD_DIR)/%.o)
NEON_OBJS = $(NEON_SRCS:src/%.cpp=$(BUILD_DIR)/%.o)
MAIN_OBJ = $(BUILD_DIR)/main.o
EXECUTABLE = $(BUILD_DIR)/inspect

all: clean $(EXECUTABLE) asm

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR) data

$(BUILD_DIR)/%.o: vendor/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: src/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(NEON_FLAGS) -c $< -o $@

$(MAIN_OBJ): $(MAIN_SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(EXECUTABLE): $(VENDOR_OBJS) $(NEON_OBJS) $(MAIN_OBJ)
	$(CXX) $(CXXFLAGS) $(NEON_FLAGS) $^ -o $@

asm: $(VENDOR_SRCS:vendor/%.cpp=$(BUILD_DIR)/%.s) $(NEON_SRCS:src/%.cpp=$(BUILD_DIR)/%.s)
	sed -i '/^[ \t]*\/\/APP$$/d; /^[ \t]*\/\/NO_APP$$/d' $^

$(BUILD_DIR)/%.s: vendor/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(ASM_FLAGS) -march=native -S $< -o $@

$(BUILD_DIR)/%.s: src/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(ASM_FLAGS) $(NEON_FLAGS) -S $< -o $@

clean:
	rm -rf $(BUILD_DIR) data

.PHONY: all asm clean