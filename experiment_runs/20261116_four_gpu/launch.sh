#!/bin/bash
set -e
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/merge
exec python -u experiment_runs/20261116_four_gpu/run_queue.py "$@" >> experiment_runs/20261116_four_gpu/queue.log 2>&1
