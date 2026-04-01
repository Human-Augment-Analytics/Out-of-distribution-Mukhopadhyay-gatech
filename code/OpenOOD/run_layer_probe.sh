#!/bin/bash
#SBATCH --job-name=layer_probe
#SBATCH -p ice-gpu,coc-gpu
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=layer_probe_%j.log

cd /home/hice1/tkasturi3/OpenOOD
source /home/hice1/tkasturi3/scratch/openood_env310/bin/activate

DATASET=${1:-cifar100}
SEED=${2:-0}

echo "Running layer probe diagnostic: dataset=${DATASET}, seed=${SEED}"

python scripts/layer_probe_diagnostic.py \
    --dataset "${DATASET}" \
    --seed "${SEED}"
