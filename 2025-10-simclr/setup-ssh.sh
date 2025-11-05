#!/bin/bash
set -euo pipefail

# Project-specific setup script for 2025-10-simclr
# Run this script as the 'giacomo' user after running setup-ssh-root.sh as root
#
# Assumptions:
# - Python, PyTorch, and CUDA are pre-installed (for ML workloads)
# - Working directory is /workspace, which is retained when the VM is paused
# - SSH keys are configured (via ForwardAgent or local setup) for git clone
#
# This script will:
# - Clone the playground-ai repository
# - Set up a Python virtual environment using uv with system site packages
# - Install project dependencies
# Note: The virtual environment will be automatically activated by direnv
#       (configured in fish-config.fish) when you cd into the project directory

REPO_URL="git@github.com:giacomoran/playground-ai.git"
REPO_DIR="/workspace/playground-ai"
PROJECT_DIR="$REPO_DIR/2025-10-simclr"

echo "=== Creating workspace directory ==="
if [ ! -d "/workspace" ]; then
    sudo mkdir -p /workspace
    sudo chown giacomo:giacomo /workspace
    echo "Created /workspace directory"
else
    echo "/workspace directory already exists"
fi

echo ""
echo "=== Cloning repository ==="
if [ -d "$REPO_DIR" ]; then
    echo "Repository already exists at $REPO_DIR, pulling latest changes..."
    cd "$REPO_DIR"
    git pull || echo "Warning: git pull failed, continuing anyway"
else
    git clone "$REPO_URL" "$REPO_DIR"
    echo "Repository cloned to $REPO_DIR"
fi

echo ""
echo "=== Installing uv ==="
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for current session
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
    echo "Installed uv"
else
    echo "uv already installed"
fi

echo ""
echo "=== Setting up Python environment ==="
cd "$PROJECT_DIR"

# Create virtual environment with system site packages (to use pre-installed PyTorch/CUDA)
echo "Creating virtual environment with system site packages..."
uv venv --system-site-packages --python "$(which python)"

# Install project dependencies
echo "Installing project dependencies..."
uv pip install -e .

echo ""
echo "=== Setup complete ==="
echo "Virtual environment created at: $PROJECT_DIR/.venv"
echo "Note: The virtual environment will be automatically activated by direnv when you cd into the project directory"
