# OpenOOD v1.5 Benchmark Scripts

Scripts to reproduce all OpenOOD v1.5 benchmark results.

## Quick Start (Local)

```bash
# 1. Download all datasets and checkpoints (~50GB for CIFAR, ~250GB with ImageNet)
bash scripts/benchmark/download_all.sh

# 2. Run all benchmarks
bash scripts/benchmark/run_all_benchmarks.sh

# 3. Aggregate results
python scripts/benchmark/aggregate_results.py
```

## ICE PACE Cluster (SLURM)

```bash
# 1. Submit download job (run once)
sbatch scripts/benchmark/slurm/download_all.sbatch

# 2. Submit all benchmark jobs (22 methods x 4 datasets = 88 jobs)
bash scripts/benchmark/slurm/submit_all.sh

# 3. Monitor jobs
squeue -u $USER

# 4. After completion, aggregate results
python scripts/benchmark/aggregate_results.py
```

## Individual Dataset Benchmarks

```bash
# CIFAR-10 only
bash scripts/benchmark/run_cifar10_all.sh

# CIFAR-100 only
bash scripts/benchmark/run_cifar100_all.sh

# ImageNet-200 only
bash scripts/benchmark/run_imagenet200_all.sh

# ImageNet-1K only (requires large storage)
bash scripts/benchmark/run_imagenet1k_all.sh
```

## Post-hoc Methods Included (22 total)

| Method | Description |
|--------|-------------|
| msp | Maximum Softmax Probability |
| odin | ODIN |
| ebo | Energy-Based OOD |
| mls | Maximum Logit Score |
| react | ReAct |
| dice | DICE |
| ash | ASH |
| knn | KNN-based |
| vim | VIM |
| gen | GEN |
| gradnorm | GradNorm |
| mds | Mahalanobis Distance Score |
| rmds | Relative MDS |
| klm | KL Matching |
| gram | Gram Matrices |
| nnguide | Neural Network Guided |
| rankfeat | RankFeat |
| she | SHE |
| scale | Scale |
| fdbd | FDBD |
| relation | Relation |
| vra | VRA |

## Datasets & OOD Splits

### CIFAR-10
- **Near-OOD**: CIFAR-100, TinyImageNet
- **Far-OOD**: MNIST, SVHN, Texture, Places365

### CIFAR-100
- **Near-OOD**: CIFAR-10, TinyImageNet
- **Far-OOD**: MNIST, SVHN, Texture, Places365

### ImageNet-200
- **Near-OOD**: SSB-hard, NINCO
- **Far-OOD**: iNaturalist, Texture, OpenImage-O

### ImageNet-1K
- **Near-OOD**: SSB-hard, NINCO
- **Far-OOD**: iNaturalist, Texture, OpenImage-O

## Storage Requirements

| Dataset | Size |
|---------|------|
| CIFAR-10 benchmark | ~5 GB |
| CIFAR-100 benchmark | ~5 GB (overlaps) |
| ImageNet-200 benchmark | ~50 GB |
| ImageNet-1K benchmark | ~200 GB |
| All checkpoints | ~5 GB |

## Working Environment (Tested)

```bash
conda create -p ~/conda_envs/openood python=3.10 -y
conda activate ~/conda_envs/openood

pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
pip install -e .
pip install numpy==1.26.4 statsmodels timm foolbox libmr
```
