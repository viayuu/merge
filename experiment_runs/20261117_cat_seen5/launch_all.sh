#!/usr/bin/env bash
set -euo pipefail
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/merge
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
exec srun --jobid=1465 --overlap -N1 -n1 -c4 --gres=gpu:2 --immediate=10 \
    bash experiment_runs/20261117_cat_seen5/run_on_node.sh
