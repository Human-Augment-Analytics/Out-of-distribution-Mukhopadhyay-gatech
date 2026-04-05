#!/bin/bash
#SBATCH --job-name=imagenet-sweep-max
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err

module load python/3.9
module load cuda/12.8

source ~/OpenOOD_Test/Out-of-distribution-Mukhopadhyay-gatech/code/.venv/bin/activate

PYTHONPATH='.':$PYTHONPATH \
python main.py \
    --config configs/datasets/imagenet/imagenet.yml \
    configs/datasets/imagenet/imagenet_ood.yml \
    configs/networks/resnet50.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/rff.yml \
    --num_workers 4 \
    --ood_dataset.image_size 256 \
    --dataset.test.batch_size 256 \
    --dataset.val.batch_size 256 \
    --network.pretrained True \
    --network.checkpoint 'results/pretrained_weights/resnet50_imagenet1k_v1.pth' \
    --mark 1

############################################
# we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood_imagenet.py

# available architectures:
# resnet50
# ood
# python scripts/eval_ood_imagenet.py \
#    --tvs-pretrained \
#    --arch resnet50 \
#    --postprocessor gram \
#    --save-score --save-csv #--fsood

# # full-spectrum ood
# python scripts/eval_ood_imagenet.py \
#    --tvs-pretrained \
#    --arch resnet50 \
#    --postprocessor gram \
#    --save-score --save-csv --fsood
