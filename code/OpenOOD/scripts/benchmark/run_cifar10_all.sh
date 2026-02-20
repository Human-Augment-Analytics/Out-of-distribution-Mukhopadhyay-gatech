#!/bin/bash
# Run ALL post-hoc OOD detection methods on CIFAR-10
# These methods work with the standard pre-trained checkpoint

set -e

CHECKPOINT_ROOT="./results/cifar10_resnet18_32x32_base_e100_lr0.1_default"
# Alternative if using downloaded v1.5 checkpoint:
# CHECKPOINT_ROOT="./results/cifar10_res18_v1.5"

OUTPUT_DIR="./benchmark_results/cifar10"
mkdir -p $OUTPUT_DIR

echo "=== Running CIFAR-10 OOD Detection Benchmark ==="
echo "Checkpoint: $CHECKPOINT_ROOT"
echo "Results will be saved to: $OUTPUT_DIR"
echo ""

# Post-hoc methods that work with standard checkpoints
POSTHOC_METHODS=(
    "msp"           # Maximum Softmax Probability
    "odin"          # ODIN
    "ebo"           # Energy-Based OOD
    "mls"           # Maximum Logit Score
    "react"         # ReAct
    "dice"          # DICE
    "ash"           # ASH
    "knn"           # KNN-based
    "vim"           # VIM
    "gen"           # GEN
    "gradnorm"      # GradNorm
    "mds"           # Mahalanobis Distance Score
    "rmds"          # Relative MDS
    "klm"           # KL Matching
    "gram"          # Gram Matrices
    "nnguide"       # Neural Network Guided
    "rankfeat"      # RankFeat
    "she"           # SHE
    "scale"         # Scale
    "fdbd"          # FDBD
    "relation"      # Relation
    "vra"           # VRA
)

for method in "${POSTHOC_METHODS[@]}"; do
    echo ""
    echo ">>> Running $method on CIFAR-10..."
    python scripts/eval_ood.py \
        --root $CHECKPOINT_ROOT \
        --id-data cifar10 \
        --postprocessor $method \
        --save-csv \
        --save-score \
        2>&1 | tee "$OUTPUT_DIR/${method}_log.txt" || echo "Warning: $method may have failed"
done

echo ""
echo "=== CIFAR-10 Benchmark Complete ==="
echo "Results saved to: $OUTPUT_DIR"
