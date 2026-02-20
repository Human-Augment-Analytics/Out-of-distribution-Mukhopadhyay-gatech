#!/bin/bash
# Submit all OpenOOD benchmark jobs to ICE PACE SLURM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs

echo "=== Submitting OpenOOD Benchmark Jobs to SLURM ==="
echo ""

# Submit download job first (if needed)
# JOB_DOWNLOAD=$(sbatch --parsable download_all.sbatch)
# echo "Submitted download job: $JOB_DOWNLOAD"

# Submit CIFAR-10 jobs
JOB_CIFAR10=$(sbatch --parsable run_cifar10.sbatch)
echo "Submitted CIFAR-10 job array: $JOB_CIFAR10"

# Submit CIFAR-100 jobs
JOB_CIFAR100=$(sbatch --parsable run_cifar100.sbatch)
echo "Submitted CIFAR-100 job array: $JOB_CIFAR100"

# Submit ImageNet-200 jobs
JOB_IN200=$(sbatch --parsable run_imagenet200.sbatch)
echo "Submitted ImageNet-200 job array: $JOB_IN200"

# Optional: Submit ImageNet-1K jobs (uncomment if you have the data)
# JOB_IN1K=$(sbatch --parsable run_imagenet1k.sbatch)
# echo "Submitted ImageNet-1K job array: $JOB_IN1K"

echo ""
echo "=== All jobs submitted ==="
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Check logs in: logs/"
echo ""
echo "Each job array will run 22 methods in parallel."
echo "Expected total jobs: 66 (3 datasets x 22 methods)"
