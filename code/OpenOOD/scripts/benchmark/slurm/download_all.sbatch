#!/bin/bash
#SBATCH --job-name=openood_download
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=inferno

# ICE PACE Download Job for OpenOOD datasets

mkdir -p logs

# Load modules
module load anaconda3/2023.03
conda activate ~/scratch/conda_venvs/openood

cd ~/scratch/openood/src/OpenOOD

echo "Starting download at $(date)"

# Download all v1.5 datasets and checkpoints
python scripts/download/download.py \
    --contents datasets checkpoints \
    --datasets ood_v1.5 \
    --checkpoints ood_v1.5 \
    --save_dir ./data ./results

echo "Download complete at $(date)"
