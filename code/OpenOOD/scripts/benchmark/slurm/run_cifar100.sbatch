#!/bin/bash
#SBATCH --job-name=cifar100_ood
#SBATCH --output=logs/cifar100_%A_%a.out
#SBATCH --error=logs/cifar100_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=32G
#SBATCH --partition=inferno
#SBATCH --array=0-21

# ICE PACE CIFAR-100 Benchmark Job Array

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
CHECKPOINT_ROOT="./results/cifar100_res18_v1.5"

echo "Running CIFAR-100 with $METHOD at $(date)"

python scripts/eval_ood.py \
    --root $CHECKPOINT_ROOT \
    --id-data cifar100 \
    --postprocessor $METHOD \
    --save-csv \
    --save-score

echo "Completed $METHOD at $(date)"
