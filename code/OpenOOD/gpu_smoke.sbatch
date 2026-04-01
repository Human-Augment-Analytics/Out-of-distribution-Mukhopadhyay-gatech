#!/bin/bash
#SBATCH -J gpu_smoke
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:05:00
#SBATCH --requeue
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
mkdir -p logs

hostname
nvidia-smi || true

python - <<'EOF'
import torch
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
EOF
