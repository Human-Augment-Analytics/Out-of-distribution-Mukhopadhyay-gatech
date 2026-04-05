#!/bin/bash
#SBATCH --job-name=sweep-imagenet
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4G
#SBATCH --time=4:00:00
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err

module load python/3.11
module load cuda/11.8


source ~/OpenOOD_Test/Out-of-distribution-Mukhopadhyay-gatech/code/.venv/bin/activate

