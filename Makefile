CXX = clang++
CXXFLAGS = -O3 -g -Wall -Wextra -std=c++17 -I/usr/local/include/simde
ASM_FLAGS = -fverbose-asm -g0 -fno-ident -fno-unwind-tables -fno-asynchronous-unwind-tables

BUILD_DIR = build
ALL_SRCS = $(wildcard vendor/*.cpp) $(filter-out %/main.cpp, $(wildcard src/*.cpp))
MAIN_SRC = src/main.cpp

ALL_OBJS = $(ALL_SRCS:vendor/%.cpp=$(BUILD_DIR)/%.o)
ALL_OBJS := $(ALL_OBJS:src/%.cpp=$(BUILD_DIR)/%.o)
MAIN_OBJ = $(BUILD_DIR)/main.o
EXECUTABLE = $(BUILD_DIR)/inspect

all: clean $(EXECUTABLE) asm

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR) data

$(BUILD_DIR)/%.o: vendor/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: src/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(MAIN_OBJ): $(MAIN_SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(EXECUTABLE): $(ALL_OBJS) $(MAIN_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@

ASM_FILES = $(ALL_SRCS:vendor/%.cpp=$(BUILD_DIR)/%.s)
ASM_FILES := $(ASM_FILES:src/%.cpp=$(BUILD_DIR)/%.s)

asm: $(ASM_FILES)
	sed -i '/^[ \t]*\/\/APP$$/d; /^[ \t]*\/\/NO_APP$$/d' $^

$(BUILD_DIR)/%.s: vendor/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(ASM_FLAGS) -S $< -o $@

$(BUILD_DIR)/%.s: src/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(ASM_FLAGS) -S $< -o $@

clean:
	rm -rf $(BUILD_DIR) data

.PHONY: all asm clean