#!/bin/bash
# Train Multiple Top Leaderboard Methods on CIFAR-10
#
# Methods included:
#   1. LogitNorm - ~91% near-OOD AUROC (simple, no extra data needed)
#   2. OE (Outlier Exposure) - ~94% AUROC (needs auxiliary OOD data)
#   3. RotPred - ~94% AUROC (rotation prediction auxiliary task)
#
# Usage:
#   bash scripts/benchmark/train_top_methods_cifar10.sh [method]
#
# Examples:
#   bash scripts/benchmark/train_top_methods_cifar10.sh          # Train all methods
#   bash scripts/benchmark/train_top_methods_cifar10.sh logitnorm  # Train only LogitNorm
#   bash scripts/benchmark/train_top_methods_cifar10.sh oe         # Train only OE
#   bash scripts/benchmark/train_top_methods_cifar10.sh rotpred    # Train only RotPred

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

METHOD="${1:-all}"

echo "=== Training Top CIFAR-10 Leaderboard Methods ==="
echo "Method: $METHOD"
echo "Project root: $PROJECT_ROOT"
echo ""

# Check if data exists
if [ ! -d "./data/images_classic/cifar10" ]; then
    echo "ERROR: CIFAR-10 dataset not found"
    echo "Please run: bash scripts/benchmark/download_all.sh"
    exit 1
fi

SEEDS=(0)  # Single seed for faster iteration; change to (0 1 2) for full benchmarks

train_logitnorm() {
    echo ""
    echo "=========================================="
    echo "  Training LogitNorm"
    echo "=========================================="
    echo ""

    for SEED in "${SEEDS[@]}"; do
        RESULT_DIR="./results/cifar10_resnet18_32x32_logitnorm_e100_lr0.1_alpha0.04_default/s${SEED}"

        if [ -f "$RESULT_DIR/best.ckpt" ]; then
            echo "LogitNorm seed $SEED: Already trained, skipping"
            continue
        fi

        echo "Training LogitNorm (seed=$SEED)..."
        PYTHONPATH="$PROJECT_ROOT":$PYTHONPATH python main.py \
            --config configs/datasets/cifar10/cifar10.yml \
            configs/networks/resnet18_32x32.yml \
            configs/pipelines/train/train_logitnorm.yml \
            configs/preprocessors/base_preprocessor.yml \
            --seed $SEED
    done

    # Evaluate
    CHECKPOINT="./results/cifar10_resnet18_32x32_logitnorm_e100_lr0.1_alpha0.04_default"
    if [ -d "$CHECKPOINT" ]; then
        echo ""
        echo "Evaluating LogitNorm..."
        python scripts/eval_ood.py \
            --id-data cifar10 \
            --root "$CHECKPOINT" \
            --postprocessor msp \
            --save-score --save-csv
    fi
}

train_oe() {
    echo ""
    echo "=========================================="
    echo "  Training OE (Outlier Exposure)"
    echo "=========================================="
    echo ""

    # Check auxiliary data
    if [ ! -f "./data/benchmark_imglist/cifar10/train_tin597.txt" ]; then
        echo "WARNING: OE auxiliary data not found, skipping OE training"
        echo "To enable OE, run: bash scripts/benchmark/download_all.sh"
        return
    fi

    for SEED in "${SEEDS[@]}"; do
        RESULT_DIR="./results/cifar10_oe_resnet18_32x32_oe_e100_lr0.1_default/s${SEED}"

        if [ -f "$RESULT_DIR/best.ckpt" ]; then
            echo "OE seed $SEED: Already trained, skipping"
            continue
        fi

        echo "Training OE (seed=$SEED)..."
        PYTHONPATH="$PROJECT_ROOT":$PYTHONPATH python main.py \
            --config configs/datasets/cifar10/cifar10.yml \
            configs/datasets/cifar10/cifar10_oe.yml \
            configs/networks/resnet18_32x32.yml \
            configs/pipelines/train/baseline.yml \
            configs/pipelines/train/train_oe.yml \
            configs/preprocessors/base_preprocessor.yml \
            --seed $SEED
    done

    # Evaluate - find checkpoint directory
    CHECKPOINT=$(ls -d ./results/cifar10_oe_* 2>/dev/null | head -1)
    if [ -d "$CHECKPOINT" ]; then
        echo ""
        echo "Evaluating OE..."
        python scripts/eval_ood.py \
            --id-data cifar10 \
            --root "$CHECKPOINT" \
            --postprocessor msp \
            --save-score --save-csv
    fi
}

train_rotpred() {
    echo ""
    echo "=========================================="
    echo "  Training RotPred (Rotation Prediction)"
    echo "=========================================="
    echo ""

    for SEED in "${SEEDS[@]}"; do
        RESULT_DIR="./results/cifar10_rot_net_rotpred_e100_lr0.1_default/s${SEED}"

        if [ -f "$RESULT_DIR/best.ckpt" ]; then
            echo "RotPred seed $SEED: Already trained, skipping"
            continue
        fi

        echo "Training RotPred (seed=$SEED)..."
        PYTHONPATH="$PROJECT_ROOT":$PYTHONPATH python main.py \
            --config configs/datasets/cifar10/cifar10.yml \
            configs/networks/rot_net.yml \
            configs/pipelines/train/baseline.yml \
            configs/preprocessors/base_preprocessor.yml \
            --trainer.name rotpred \
            --seed $SEED
    done

    # Evaluate with rotpred postprocessor
    CHECKPOINT="./results/cifar10_rot_net_rotpred_e100_lr0.1_default"
    if [ -d "$CHECKPOINT" ]; then
        echo ""
        echo "Evaluating RotPred..."
        python scripts/eval_ood.py \
            --id-data cifar10 \
            --root "$CHECKPOINT" \
            --postprocessor rotpred \
            --save-score --save-csv
    fi
}

# Run selected method(s)
case "$METHOD" in
    logitnorm)
        train_logitnorm
        ;;
    oe)
        train_oe
        ;;
    rotpred)
        train_rotpred
        ;;
    all)
        train_logitnorm
        train_oe
        train_rotpred
        ;;
    *)
        echo "Unknown method: $METHOD"
        echo "Available: logitnorm, oe, rotpred, all"
        exit 1
        ;;
esac

echo ""
echo "=== Training Complete ==="
echo ""
echo "Results saved in ./results/"
echo ""
echo "Compare with OpenOOD leaderboard: https://zjysteven.github.io/OpenOOD/"
echo ""
echo "Expected near-OOD AUROC (CIFAR-10):"
echo "  - LogitNorm + MSP: ~91%"
echo "  - OE + MSP:        ~94%"
echo "  - RotPred:         ~94%"
echo "  - Baseline (CE):   ~87%"
