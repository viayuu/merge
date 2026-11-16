#!/bin/bash
set -e
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/merge
exec srun --jobid=1475 --overlap -N1 -n1 -c4 --immediate=10 "$CONDA_PREFIX/bin/python" -u experiment_runs/20261116_acvrp_audit/run_advance_node.py gpu05 > experiment_runs/20261116_acvrp_audit/gpu05_advance.log 2>&1
