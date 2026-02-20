#!/bin/bash
# Train LogitNorm on CIFAR-10 - A Top Leaderboard Method
# LogitNorm achieves ~91% near-OOD AUROC vs ~87% for baseline CrossEntropy
#
# Usage:
#   bash scripts/benchmark/train_logitnorm_cifar10.sh
#
# This trains a ResNet-18 with LogitNorm training objective,
# then evaluates with multiple OOD postprocessors.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== Training LogitNorm on CIFAR-10 ==="
echo "Project root: $PROJECT_ROOT"
echo ""

# Check if data exists
if [ ! -d "./data/images_classic/cifar10" ]; then
    echo "ERROR: CIFAR-10 dataset not found at ./data/images_classic/cifar10"
    echo "Please run: bash scripts/benchmark/download_all.sh"
    exit 1
fi

# Training seeds (run 3 seeds for statistical significance)
SEEDS=(0 1 2)

# Train LogitNorm for each seed
for SEED in "${SEEDS[@]}"; do
    RESULT_DIR="./results/cifar10_resnet18_32x32_logitnorm_e100_lr0.1_alpha0.04_default/s${SEED}"

    if [ -f "$RESULT_DIR/best.ckpt" ]; then
        echo "Seed $SEED: Checkpoint already exists at $RESULT_DIR/best.ckpt, skipping training"
        continue
    fi

    echo ""
    echo "=== Training LogitNorm (seed=$SEED) ==="
    echo "Output: $RESULT_DIR"
    echo ""

    PYTHONPATH="$PROJECT_ROOT":$PYTHONPATH python main.py \
        --config configs/datasets/cifar10/cifar10.yml \
        configs/networks/resnet18_32x32.yml \
        configs/pipelines/train/train_logitnorm.yml \
        configs/preprocessors/base_preprocessor.yml \
        --seed $SEED
done

echo ""
echo "=== Training Complete ==="
echo ""

# Evaluate with multiple postprocessors
CHECKPOINT_ROOT="./results/cifar10_resnet18_32x32_logitnorm_e100_lr0.1_alpha0.04_default"

# Verify checkpoint exists
if [ ! -d "$CHECKPOINT_ROOT" ]; then
    echo "ERROR: Training output not found at $CHECKPOINT_ROOT"
    exit 1
fi

echo "=== Evaluating LogitNorm with Multiple Postprocessors ==="
echo "Checkpoint root: $CHECKPOINT_ROOT"
echo ""

# Test with multiple postprocessors (LogitNorm typically uses MSP or EBO)
POSTPROCESSORS=("msp" "ebo" "mls" "odin" "knn" "vim")

for PP in "${POSTPROCESSORS[@]}"; do
    echo ""
    echo "--- Evaluating with postprocessor: $PP ---"

    python scripts/eval_ood.py \
        --id-data cifar10 \
        --root "$CHECKPOINT_ROOT" \
        --postprocessor "$PP" \
        --save-score --save-csv
done

echo ""
echo "=== LogitNorm Evaluation Complete ==="
echo ""
echo "Results saved to: $CHECKPOINT_ROOT/*/ood/"
echo ""
echo "Compare your results with the OpenOOD leaderboard:"
echo "  https://zjysteven.github.io/OpenOOD/"
echo ""
echo "Expected near-OOD AUROC for LogitNorm + MSP: ~91%"
echo "(vs ~87% for baseline CrossEntropy + MSP)"
