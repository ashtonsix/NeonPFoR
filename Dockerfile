FROM ubuntu:24.04

# To avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install required packages for building and development
RUN apt-get update && apt-get install -y \
  build-essential \
  wget \
  git \
  cmake \
  ninja-build \
  python3 \
  python3-pip \
  lsb-release \
  software-properties-common \
  gnupg \
  apt-transport-https \
  ca-certificates \
  curl \
  unzip

# Add LLVM repository
RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
  add-apt-repository "deb http://apt.llvm.org/noble/ llvm-toolchain-noble-19 main"

# Update and install Clang/LLVM 19
RUN apt-get update && \
  apt-get install -y \
  clang-19 \
  llvm-19 \
  lldb-19 \
  lld-19 \
  clangd-19 \
  clang-tools-19 \
  llvm-19-dev

# Set up alternatives for easier access
RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-19 100 && \
  update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-19 100 && \
  update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-19 100 && \
  update-alternatives --install /usr/bin/lldb lldb /usr/bin/lldb-19 100 && \
  update-alternatives --install /usr/bin/llvm-mca llvm-mca /usr/bin/llvm-mca-19 100 && \
  update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-19 100

# Install SIMDe (header-only library)
RUN git clone --depth 1 https://github.com/simd-everywhere/simde.git /usr/local/include/simde && \
  echo "✓ SIMDe installed in /usr/local/include/simde"

# Create a working directory
WORKDIR /neon-pfor

# Install zsh
RUN apt-get update && apt-get install -y zsh

# Set up a non-root user for development
RUN useradd -m developer && \
  chown -R developer:developer /neon-pfor

# Make the setup script executable
RUN chmod +x ./setup.sh 2>/dev/null || true

USER developer

# Install Oh My Zsh
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

# Set zsh as default shell for developer
USER root
RUN chsh -s /usr/bin/zsh developer

USER developer
# Set the default command
CMD ["/usr/bin/zsh"]