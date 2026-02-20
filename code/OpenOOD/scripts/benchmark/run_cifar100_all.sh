#!/bin/bash
# Run ALL post-hoc OOD detection methods on CIFAR-100

set -e

CHECKPOINT_ROOT="./results/cifar100_res18_v1.5"
OUTPUT_DIR="./benchmark_results/cifar100"
mkdir -p $OUTPUT_DIR

echo "=== Running CIFAR-100 OOD Detection Benchmark ==="
echo "Checkpoint: $CHECKPOINT_ROOT"
echo "Results will be saved to: $OUTPUT_DIR"
echo ""

# Post-hoc methods
POSTHOC_METHODS=(
    "msp"
    "odin"
    "ebo"
    "mls"
    "react"
    "dice"
    "ash"
    "knn"
    "vim"
    "gen"
    "gradnorm"
    "mds"
    "rmds"
    "klm"
    "gram"
    "nnguide"
    "rankfeat"
    "she"
    "scale"
    "fdbd"
    "relation"
    "vra"
)

for method in "${POSTHOC_METHODS[@]}"; do
    echo ""
    echo ">>> Running $method on CIFAR-100..."
    python scripts/eval_ood.py \
        --root $CHECKPOINT_ROOT \
        --id-data cifar100 \
        --postprocessor $method \
        --save-csv \
        --save-score \
        2>&1 | tee "$OUTPUT_DIR/${method}_log.txt" || echo "Warning: $method may have failed"
done

echo ""
echo "=== CIFAR-100 Benchmark Complete ==="
echo "Results saved to: $OUTPUT_DIR"
