#!/usr/bin/env bash
set -e
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
export CUDA_VISIBLE_DEVICES="$1"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
shift
exec python /public/home/shiys/merge/experiment_runs/20261116_four_gpu/gpu_test.py "$@"
