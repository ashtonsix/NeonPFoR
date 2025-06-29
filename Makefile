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
VENDOR_SRCS = $(wildcard vendor/*/*.cpp)
SRC_SRCS = $(wildcard src/*.cpp)
MAIN_SRC = test/main.cpp
ALL_SRCS = $(VENDOR_SRCS) $(SRC_SRCS)

# Generate object file names preserving directory structure to avoid conflicts
VENDOR_OBJS = $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(VENDOR_SRCS))
SRC_OBJS = $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(SRC_SRCS))
ALL_OBJS = $(VENDOR_OBJS) $(SRC_OBJS)
MAIN_OBJ = $(BUILD_DIR)/test/main.o
EXECUTABLE = $(BUILD_DIR)/inspect

ASM_FILES = $(patsubst %.cpp,$(BUILD_DIR)/%.s,$(VENDOR_SRCS)) $(patsubst %.cpp,$(BUILD_DIR)/%.s,$(SRC_SRCS))
MCA_FILES = $(ASM_FILES:.s=.mca)

all: clean $(EXECUTABLE) asm # mca

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR) $(BUILD_DIR)/vendor/fastpfor $(BUILD_DIR)/vendor/naive $(BUILD_DIR)/src $(BUILD_DIR)/test data

$(BUILD_DIR)/vendor/%.o: vendor/%.cpp | $(BUILD_DIR)
	clang++ $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/src/%.o: src/%.cpp | $(BUILD_DIR)
	clang++ $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/test/%.o: test/%.cpp | $(BUILD_DIR)
	clang++ $(CXXFLAGS) -c $< -o $@

$(EXECUTABLE): $(ALL_OBJS) $(MAIN_OBJ)
	clang++ $(CXXFLAGS) $^ -o $@

asm: $(ASM_FILES)
	sed -i '/^[ \t]*\/\/APP$$/d; /^[ \t]*\/\/NO_APP$$/d' $^

$(BUILD_DIR)/vendor/%.s: vendor/%.cpp | $(BUILD_DIR)
	clang++ $(CXXFLAGS) $(ASM_FLAGS) -S $< -o $@

$(BUILD_DIR)/src/%.s: src/%.cpp | $(BUILD_DIR)
	clang++ $(CXXFLAGS) $(ASM_FLAGS) -S $< -o $@

$(BUILD_DIR)/test/%.s: test/%.cpp | $(BUILD_DIR)
	clang++ $(CXXFLAGS) $(ASM_FLAGS) -S $< -o $@

# TODO: split each asm file into pieces, and generate MCA per-piece
mca: $(MCA_FILES)

$(BUILD_DIR)/%.mca: $(BUILD_DIR)/%.s
	llvm-mca $(MCA_FLAGS) $< > $@

clean:
	rm -rf $(BUILD_DIR) data

.PHONY: all asm mca clean