#!/bin/bash
# OpenOOD Working Environment Setup
# This configuration has been tested and works correctly.

set -e

ENV_PATH="${1:-~/conda_envs/openood}"

echo "=== OpenOOD Environment Setup ==="
echo "Creating environment at: $ENV_PATH"
echo ""

# Create fresh conda environment
conda create -p "$ENV_PATH" python=3.10 -y

# Activate (for script continuation)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"

echo "Installing PyTorch with CUDA 11.8..."
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118

echo "Installing OpenOOD..."
pip install -e .

echo "Downgrading numpy for imgaug compatibility..."
pip install numpy==1.26.4

echo "Installing additional dependencies..."
pip install statsmodels timm foolbox libmr

echo ""
echo "=== Verifying Installation ==="
python -c "
import torch
import torchvision
import numpy as np
import imgaug
print(f'PyTorch: {torch.__version__}')
print(f'Torchvision: {torchvision.__version__}')
print(f'NumPy: {np.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print('All imports successful!')
"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To activate this environment:"
echo "  conda activate $ENV_PATH"
echo ""
echo "To run benchmarks:"
echo "  bash scripts/benchmark/download_all.sh"
echo "  bash scripts/benchmark/run_all_benchmarks.sh"
