#!/bin/bash
# Download all datasets and checkpoints for OpenOOD v1.5 benchmarks

set -e

echo "=== Downloading ALL OpenOOD v1.5 Datasets and Checkpoints ==="

# Download all v1.5 benchmark datasets (CIFAR-10, CIFAR-100, ImageNet-200, ImageNet-1K)
echo "Downloading datasets..."
python scripts/download/download.py \
    --contents datasets \
    --datasets ood_v1.5 \
    --save_dir ./data ./results

# Download all v1.5 checkpoints
echo "Downloading checkpoints..."
python scripts/download/download.py \
    --contents checkpoints \
    --checkpoints ood_v1.5 \
    --save_dir ./data ./results

echo "=== Download Complete ==="
echo ""
echo "Datasets downloaded to: ./data/"
echo "Checkpoints downloaded to: ./results/"
echo ""
echo "Expected structure:"
echo "  ./data/benchmark_imglist/     - image lists"
echo "  ./data/images_classic/        - CIFAR-10, CIFAR-100, TIN, MNIST, SVHN, etc."
echo "  ./data/images_largescale/     - ImageNet-related datasets"
echo "  ./results/cifar10_res18_v1.5/ - CIFAR-10 checkpoint"
echo "  ./results/cifar100_res18_v1.5/ - CIFAR-100 checkpoint"
echo "  ./results/imagenet200_res18_v1.5/ - ImageNet-200 checkpoint"
echo "  ./results/imagenet_res50_v1.5/ - ImageNet-1K checkpoint"
