#!/bin/bash
#SBATCH --job-name=cifar100umap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=intel
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err

module load python/3.10
module load cuda/12.9

source ~/scratch/.venv/bin/activate

PYTHONPATH='.':$PYTHONPATH \
python main.py \
    --config configs/datasets/cifar100/cifar100.yml \
    configs/datasets/cifar100/cifar100_ood.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/rff.yml \
    --num_workers 8 \
    --network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt' \
    --mark 50032428711122222 \
    --merge_option merge