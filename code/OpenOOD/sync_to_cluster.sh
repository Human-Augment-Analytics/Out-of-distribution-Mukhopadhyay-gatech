#!/bin/bash
# Sync local OpenOOD changes to the PACE ICE cluster.
# Usage:
#   ./sync_to_cluster.sh          # sync files to cluster
#   ./sync_to_cluster.sh --dry-run  # preview what would be synced, no transfer

REMOTE="tkasturi3@login-ice.pace.gatech.edu"
REMOTE_DIR="/home/hice1/tkasturi3/OpenOOD"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

DRY=""
if [[ "$1" == "--dry-run" ]]; then
    DRY="-n"
    echo "DRY RUN — no files will be transferred"
fi

rsync -azv ${DRY} \
    --exclude '.git/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.DS_Store' \
    --exclude '*.out' \
    --exclude '*.log' \
    --exclude '--*' \
    --exclude 'logs/' \
    --exclude 'results/checkpoints/' \
    --exclude 'openood/openood/' \
    --exclude 'openood/configs/' \
    --exclude 'openood/scripts/' \
    --exclude 'openood/results/' \
    --exclude 'openood/tests/' \
    --exclude 'openood/tools/' \
    --exclude 'openood/logs/' \
    --exclude 'openood/.github/' \
    --exclude 'openood/main.py' \
    --exclude 'openood/setup.py' \
    --exclude 'openood/pyproject.toml' \
    --exclude 'openood/poetry.lock' \
    --exclude 'openood/imglist_generator.py' \
    --exclude 'openood/path_check.sh' \
    --exclude 'openood/easy_dev.ipynb' \
    --exclude 'openood/eval_kernel_attention.sh' \
    --exclude 'openood/gpu_smoke.sbatch' \
    --exclude 'openood/.gitignore' \
    --exclude 'openood/.pre-commit-config.yaml' \
    --exclude 'openood/CODE_OF_CONDUCT.md' \
    --exclude 'openood/CONTRIBUTING.md' \
    --exclude 'openood/LICENSE' \
    --exclude 'openood/README.md' \
    --exclude 'openood/codespell_ignored.txt' \
    "${LOCAL_DIR}/" \
    "${REMOTE}:${REMOTE_DIR}/"
