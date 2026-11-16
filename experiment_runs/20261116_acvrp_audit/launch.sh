#!/bin/bash
set -e
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/merge
srun --jobid=1475 --overlap -N1 -n1 -c4 --immediate=10 "$CONDA_PREFIX/bin/python" -u experiment_runs/20261116_acvrp_audit/run_node.py gpu05 > experiment_runs/20261116_acvrp_audit/gpu05.log 2>&1 &
pid4090=$!
srun --jobid=1465 --overlap -N1 -n1 -c4 --immediate=10 "$CONDA_PREFIX/bin/python" -u experiment_runs/20261116_acvrp_audit/run_node.py gpu03 > experiment_runs/20261116_acvrp_audit/gpu03.log 2>&1 &
pid3090=$!
wait "$pid4090"
wait "$pid3090"
