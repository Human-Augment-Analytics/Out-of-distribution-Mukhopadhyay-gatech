#!/bin/bash
#SBATCH --job-name=kernel_attn_ood
#SBATCH -p ice-gpu,coc-gpu
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=kernel_attn_%j.log

cd /home/hice1/tkasturi3/OpenOOD
source /home/hice1/tkasturi3/scratch/openood_env310/bin/activate

DATASET=${1:-cifar100}        # pass as first arg, default cifar100
SEED=${2:-0}                  # pass as second arg, default seed 0
RFF_DIM=${3:-2048}
SIGMA_SCALE=${4:-0.5}         # pass as fourth arg, default 0.5
SCORE_MODE=${5:-per_class_threshold}  # pass as fifth arg: max or per_class_threshold
ALPHA=${6:-0.05}              # pass as sixth arg, default 0.05
KERNEL_WEIGHTED=${7:-""}      # pass "--kernel-weighted" as seventh arg to enable
DUAL_HEAD=${8:-""}            # pass "--dual-head" as eighth arg to enable
HEAD_ALPHA=${9:-0.5}          # weight for far head (used only with --dual-head)

echo "Running Kernel-Attention OOD: dataset=${DATASET}, seed=${SEED}, D=${RFF_DIM}, sigma_scale=${SIGMA_SCALE}, score_mode=${SCORE_MODE}, alpha=${ALPHA}"

python scripts/eval_ood_kernel_attention.py \
    --dataset "${DATASET}" \
    --seed "${SEED}" \
    --rff-dim "${RFF_DIM}" \
    --sigma-scale "${SIGMA_SCALE}" \
    --score-mode "${SCORE_MODE}" \
    --alpha "${ALPHA}" \
    --head-alpha "${HEAD_ALPHA}" \
    --auto-sigma \
    ${KERNEL_WEIGHTED} \
    ${DUAL_HEAD}
