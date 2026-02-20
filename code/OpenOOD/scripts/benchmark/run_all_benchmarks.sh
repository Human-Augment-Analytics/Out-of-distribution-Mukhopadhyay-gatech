#!/bin/bash
# Master script to run ALL OpenOOD v1.5 benchmarks
# This runs all post-hoc methods on all datasets

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

echo "=========================================="
echo "OpenOOD v1.5 Full Benchmark Suite"
echo "=========================================="
echo ""
echo "This will run all post-hoc OOD detection methods on:"
echo "  - CIFAR-10"
echo "  - CIFAR-100"
echo "  - ImageNet-200"
echo "  - ImageNet-1K (optional, requires large storage)"
echo ""

# Create results directory
mkdir -p ./benchmark_results

# Run CIFAR-10
echo "=========================================="
echo "Starting CIFAR-10 Benchmark..."
echo "=========================================="
bash scripts/benchmark/run_cifar10_all.sh

# Run CIFAR-100
echo "=========================================="
echo "Starting CIFAR-100 Benchmark..."
echo "=========================================="
bash scripts/benchmark/run_cifar100_all.sh

# Run ImageNet-200
echo "=========================================="
echo "Starting ImageNet-200 Benchmark..."
echo "=========================================="
bash scripts/benchmark/run_imagenet200_all.sh

# Optionally run ImageNet-1K (uncomment if you have the data)
# echo "=========================================="
# echo "Starting ImageNet-1K Benchmark..."
# echo "=========================================="
# bash scripts/benchmark/run_imagenet1k_all.sh

echo ""
echo "=========================================="
echo "ALL BENCHMARKS COMPLETE"
echo "=========================================="
echo ""
echo "Results saved to: ./benchmark_results/"
echo ""
echo "To aggregate results, run:"
echo "  python scripts/benchmark/aggregate_results.py"
