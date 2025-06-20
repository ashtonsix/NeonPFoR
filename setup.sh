#!/bin/bash

# NeonPFoR Development Environment Setup Script

set -e  # Exit on any error

echo "🚀 Setting up NeonPFoR development environment..."

# Update package list
echo "📦 Updating package list..."
sudo apt-get update -qq

# Install required packages for building and development
echo "🔧 Installing build tools and dependencies..."
sudo apt-get install -y -qq \
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

# Add LLVM repository (using modern approach)
echo "🔑 Adding LLVM repository..."
wget -q -O /tmp/llvm-snapshot.gpg.key https://apt.llvm.org/llvm-snapshot.gpg.key
sudo gpg --dearmor < /tmp/llvm-snapshot.gpg.key | sudo tee /usr/share/keyrings/llvm-snapshot.gpg > /dev/null 2>&1
echo "deb [signed-by=/usr/share/keyrings/llvm-snapshot.gpg] http://apt.llvm.org/noble/ llvm-toolchain-noble-19 main" | sudo tee /etc/apt/sources.list.d/llvm.list > /dev/null
rm /tmp/llvm-snapshot.gpg.key

# Update and install Clang/LLVM 19
echo "⚡ Installing Clang/LLVM 19 toolchain..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
  clang-19 \
  llvm-19 \
  lldb-19 \
  lld-19 \
  clangd-19 \
  clang-tools-19 \
  llvm-19-dev

# Set up alternatives for easier access
echo "🔗 Setting up command alternatives..."
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-19 100 > /dev/null 2>&1
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-19 100 > /dev/null 2>&1
sudo update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-19 100 > /dev/null 2>&1
sudo update-alternatives --install /usr/bin/lldb lldb /usr/bin/lldb-19 100 > /dev/null 2>&1
sudo update-alternatives --install /usr/bin/llvm-mca llvm-mca /usr/bin/llvm-mca-19 100 > /dev/null 2>&1
sudo update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-19 100 > /dev/null 2>&1

# Install SIMDe (header-only library)
echo "📚 Installing SIMDe library..."
if [ ! -d "/usr/local/include/simde" ]; then
  sudo git clone -q --depth 1 https://github.com/simd-everywhere/simde.git /usr/local/include/simde
  echo "✓ SIMDe installed in /usr/local/include/simde"
else
  echo "✓ SIMDe already installed"
fi

# Install zsh (optional, for better terminal experience)
echo "🐚 Installing zsh..."
sudo apt-get install -y -qq zsh

# Offer to install Oh My Zsh for the current user
echo "🎨 Would you like to install Oh My Zsh for better terminal experience? (y/n)"
read -r install_omz
if [[ $install_omz =~ ^[Yy]$ ]]; then
  if [ ! -d "$HOME/.oh-my-zsh" ]; then
    sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended > /dev/null 2>&1
    echo "✓ Oh My Zsh installed"
  else
    echo "✓ Oh My Zsh already installed"
  fi
  
  echo "🔄 Would you like to set zsh as your default shell? (y/n)"
  read -r set_zsh
  if [[ $set_zsh =~ ^[Yy]$ ]]; then
    sudo chsh -s /usr/bin/zsh $USER > /dev/null 2>&1
    echo "✓ Zsh set as default shell (restart terminal to take effect)"
  fi
fi

echo ""
echo "🎉 NeonPFoR development environment setup complete!"
echo ""
echo "📋 Installed components:"
echo "  • Build tools (gcc, cmake, ninja, etc.)"
echo "  • Clang/LLVM 19 toolchain"
echo "  • SIMDe library"
echo "  • Development utilities"
echo ""
echo "🔍 You can verify the installation with:"
echo "  clang --version"
echo "  llvm-config --version"
echo "  ls /usr/local/include/simde"
echo ""
echo "💡 VS Code users: Install the 'clangd' extension for C++ language support" 