#!/bin/bash
#SBATCH --job-name=in200_ood
#SBATCH --output=logs/imagenet200_%A_%a.out
#SBATCH --error=logs/imagenet200_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=64G
#SBATCH --partition=inferno
#SBATCH --array=0-21

# ICE PACE ImageNet-200 Benchmark Job Array

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
CHECKPOINT_ROOT="./results/imagenet200_res18_v1.5"

echo "Running ImageNet-200 with $METHOD at $(date)"

python scripts/eval_ood.py \
    --root $CHECKPOINT_ROOT \
    --id-data imagenet200 \
    --postprocessor $METHOD \
    --save-csv \
    --save-score

echo "Completed $METHOD at $(date)"
