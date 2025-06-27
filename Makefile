CXXFLAGS = -O3 \
	-g \
	-Wall \
	-Wextra \
	-std=c++17 \
	-I/usr/local/include/simde \
	-march=armv9.2-a \
	-mllvm -inlinehint-threshold=10000 # Stop LLVM ignoring inline hints
ASM_FLAGS = -fverbose-asm \
	-g0 \
	-fno-ident \
	-fno-unwind-tables \
	-fno-asynchronous-unwind-tables
MCA_FLAGS = -mcpu=neoverse-v2 \
	-timeline \
	-resource-pressure \
	-scheduler-stats \
	-all-stats

BUILD_DIR = build
ALL_SRCS = $(wildcard vendor/*.cpp) $(filter-out %/main.cpp, $(wildcard src/*.cpp))
MAIN_SRC = src/main.cpp

ALL_OBJS = $(ALL_SRCS:vendor/%.cpp=$(BUILD_DIR)/%.o)
ALL_OBJS := $(ALL_OBJS:src/%.cpp=$(BUILD_DIR)/%.o)
MAIN_OBJ = $(BUILD_DIR)/main.o
EXECUTABLE = $(BUILD_DIR)/inspect
ASM_FILES = $(ALL_SRCS:vendor/%.cpp=$(BUILD_DIR)/%.s)
ASM_FILES := $(ASM_FILES:src/%.cpp=$(BUILD_DIR)/%.s)
MCA_FILES = $(ASM_FILES:.s=.mca)

all: clean $(EXECUTABLE) asm # mca

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR) data

$(BUILD_DIR)/%.o: vendor/%.cpp | $(BUILD_DIR)
	clang++ $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: src/%.cpp | $(BUILD_DIR)
	clang++ $(CXXFLAGS) -c $< -o $@

$(MAIN_OBJ): $(MAIN_SRC) | $(BUILD_DIR)
	clang++ $(CXXFLAGS) -c $< -o $@

$(EXECUTABLE): $(ALL_OBJS) $(MAIN_OBJ)
	clang++ $(CXXFLAGS) $^ -o $@

asm: $(ASM_FILES)
	sed -i '/^[ \t]*\/\/APP$$/d; /^[ \t]*\/\/NO_APP$$/d' $^

$(BUILD_DIR)/%.s: vendor/%.cpp | $(BUILD_DIR)
	clang++ $(CXXFLAGS) $(ASM_FLAGS) -S $< -o $@

$(BUILD_DIR)/%.s: src/%.cpp | $(BUILD_DIR)
	clang++ $(CXXFLAGS) $(ASM_FLAGS) -S $< -o $@

# TODO: split each asm file into pieces, and generate MCA per-piece
mca: $(MCA_FILES)

$(BUILD_DIR)/%.mca: $(BUILD_DIR)/%.s
	llvm-mca $(MCA_FLAGS) $< > $@

clean:
	rm -rf $(BUILD_DIR) data

.PHONY: all asm mca clean