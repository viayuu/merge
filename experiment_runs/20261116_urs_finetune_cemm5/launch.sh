#!/usr/bin/env bash
set -euo pipefail
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/merge
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
srun --jobid=1475 --overlap --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:2 --immediate=10 \
    python experiment_runs/20261116_urs_finetune_cemm5/run_queue.py "$@"
