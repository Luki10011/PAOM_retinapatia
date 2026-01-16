#!/usr/bin/env bash
set -e  # Exit immediately if a command fails
set -u  # Treat unset variables as an error

# -------------------------------
# Activate virtual environment
# -------------------------------
VENV_DIR="./.venv"

if [ -d "$VENV_DIR" ]; then
    echo "Activating virtual environment..."
    # Bash activation
    source "$VENV_DIR/bin/activate"
else
    echo "Error: .venv not found. Please create it first:"
    echo "       python -m venv $VENV_DIR"
    exit 1
fi

# -------------------------------
# Detect installer
# -------------------------------
if command -v uv >/dev/null 2>&1; then
    INSTALLER="uv pip"
    echo "Using 'uv' as installer"
else
    INSTALLER="pip"
    echo "'uv' not found, falling back to pip"
fi

# -------------------------------
# Detect NVIDIA GPU
# -------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_PRESENT=true
    echo "NVIDIA GPU detected, installing standard torch"
else
    GPU_PRESENT=false
    echo "No NVIDIA GPU detected, installing CPU-only torch"
fi

# -------------------------------
# Install PyTorch
# -------------------------------
if [ "$GPU_PRESENT" = true ]; then
    $INSTALLER install torch torchvision
else
    $INSTALLER install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

$INSTALLER install -r requirements.txt 
