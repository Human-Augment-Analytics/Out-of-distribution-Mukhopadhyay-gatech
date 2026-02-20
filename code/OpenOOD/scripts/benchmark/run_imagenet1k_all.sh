#!/bin/bash
# Run ALL post-hoc OOD detection methods on ImageNet-1K
# NOTE: ImageNet-1K requires more memory and time

set -e

CHECKPOINT_ROOT="./results/imagenet_res50_v1.5"
OUTPUT_DIR="./benchmark_results/imagenet1k"
mkdir -p $OUTPUT_DIR

echo "=== Running ImageNet-1K OOD Detection Benchmark ==="
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
    echo ">>> Running $method on ImageNet-1K..."
    python scripts/eval_ood.py \
        --root $CHECKPOINT_ROOT \
        --id-data imagenet \
        --postprocessor $method \
        --save-csv \
        --save-score \
        2>&1 | tee "$OUTPUT_DIR/${method}_log.txt" || echo "Warning: $method may have failed"
done

echo ""
echo "=== ImageNet-1K Benchmark Complete ==="
echo "Results saved to: $OUTPUT_DIR"
