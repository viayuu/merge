#!/usr/bin/env bash
set -euo pipefail
node=${1:?Usage: launch_node.sh gpu05|gpu03 calibration|scan|confirm}
phase=${2:?Usage: launch_node.sh gpu05|gpu03 calibration|scan|confirm}
case "$node" in
    gpu05) allocation=1475 ;;
    gpu03) allocation=1465 ;;
    *) echo "Unknown node: $node" >&2; exit 1 ;;
esac
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/merge
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
exec srun --jobid="$allocation" --overlap -N1 -n1 -c4 --gres=gpu:2 --immediate=10 \
    python experiment_runs/20261117_ties_seen5/run_evaluations.py \
    --node "$node" --phase "$phase"
