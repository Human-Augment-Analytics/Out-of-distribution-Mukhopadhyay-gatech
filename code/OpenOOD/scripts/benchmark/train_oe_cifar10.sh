#!/bin/bash
# Train OE (Outlier Exposure) on CIFAR-10 - A Top Leaderboard Method
# OE achieves ~94% near-OOD AUROC - one of the best methods
#
# IMPORTANT: OE requires auxiliary OOD data (TinyImageNet-597) during training.
# Make sure you've downloaded the full benchmark data:
#   bash scripts/benchmark/download_all.sh
#
# Usage:
#   bash scripts/benchmark/train_oe_cifar10.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== Training OE (Outlier Exposure) on CIFAR-10 ==="
echo "Project root: $PROJECT_ROOT"
echo ""

# Check if data exists
if [ ! -d "./data/images_classic/cifar10" ]; then
    echo "ERROR: CIFAR-10 dataset not found"
    echo "Please run: bash scripts/benchmark/download_all.sh"
    exit 1
fi

# Check if OE auxiliary data exists (TinyImageNet subset)
if [ ! -f "./data/benchmark_imglist/cifar10/train_tin597.txt" ]; then
    echo "ERROR: OE auxiliary data (TinyImageNet-597) not found"
    echo "Please run: bash scripts/benchmark/download_all.sh"
    exit 1
fi

# Training seeds
SEEDS=(0 1 2)

# Train OE for each seed
for SEED in "${SEEDS[@]}"; do
    RESULT_DIR="./results/cifar10_oe_resnet18_32x32_oe_e100_lr0.1_default/s${SEED}"

    if [ -f "$RESULT_DIR/best.ckpt" ]; then
        echo "Seed $SEED: Checkpoint already exists, skipping training"
        continue
    fi

    echo ""
    echo "=== Training OE (seed=$SEED) ==="
    echo ""

    PYTHONPATH="$PROJECT_ROOT":$PYTHONPATH python main.py \
        --config configs/datasets/cifar10/cifar10.yml \
        configs/datasets/cifar10/cifar10_oe.yml \
        configs/networks/resnet18_32x32.yml \
        configs/pipelines/train/baseline.yml \
        configs/pipelines/train/train_oe.yml \
        configs/preprocessors/base_preprocessor.yml \
        --seed $SEED
done

echo ""
echo "=== OE Training Complete ==="
echo ""

# Find the checkpoint directory
CHECKPOINT_ROOT="./results/cifar10_oe_resnet18_32x32_oe_e100_lr0.1_default"

if [ ! -d "$CHECKPOINT_ROOT" ]; then
    # Try alternate naming convention
    CHECKPOINT_ROOT=$(ls -d ./results/cifar10_oe_* 2>/dev/null | head -1)
fi

if [ ! -d "$CHECKPOINT_ROOT" ]; then
    echo "ERROR: Training output not found"
    exit 1
fi

echo "=== Evaluating OE ==="
echo "Checkpoint root: $CHECKPOINT_ROOT"
echo ""

# OE is typically evaluated with MSP, EBO, or MLS
POSTPROCESSORS=("msp" "ebo" "mls")

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
echo "=== OE Evaluation Complete ==="
echo ""
echo "Results saved to: $CHECKPOINT_ROOT/*/ood/"
echo ""
echo "Expected near-OOD AUROC for OE + MSP: ~94%"
