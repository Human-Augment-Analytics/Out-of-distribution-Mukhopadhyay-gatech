#!/bin/bash
#SBATCH --job-name=in1k_ood
#SBATCH --output=logs/imagenet1k_%A_%a.out
#SBATCH --error=logs/imagenet1k_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=128G
#SBATCH --partition=inferno
#SBATCH --array=0-21

# ICE PACE ImageNet-1K Benchmark Job Array

mkdir -p logs

module load anaconda3/2023.03
conda activate ~/scratch/conda_venvs/openood

cd ~/scratch/openood/src/OpenOOD

METHODS=(
    "msp" "odin" "ebo" "mls" "react" "dice" "ash" "knn"
    "vim" "gen" "gradnorm" "mds" "rmds" "klm" "gram"
    "nnguide" "rankfeat" "she" "scale" "fdbd" "relation" "vra"
)

METHOD=${METHODS[$SLURM_ARRAY_TASK_ID]}
CHECKPOINT_ROOT="./results/imagenet_res50_v1.5"

echo "Running ImageNet-1K with $METHOD at $(date)"

python scripts/eval_ood.py \
    --root $CHECKPOINT_ROOT \
    --id-data imagenet \
    --postprocessor $METHOD \
    --save-csv \
    --save-score

echo "Completed $METHOD at $(date)"
