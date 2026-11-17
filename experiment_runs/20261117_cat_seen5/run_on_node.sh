#!/usr/bin/env bash
set -euo pipefail
cd /public/home/shiys/merge
run_dir=experiment_runs/20261117_cat_seen5
CUDA_VISIBLE_DEVICES=0 python "$run_dir/build_models.py" --seed 3407 > "$run_dir/build_3407.log" 2>&1 &
cat_build_left=$!
CUDA_VISIBLE_DEVICES=1 python "$run_dir/build_models.py" --seed 3408 > "$run_dir/build_3408.log" 2>&1 &
cat_build_right=$!
wait "$cat_build_left"
wait "$cat_build_right"
CUDA_VISIBLE_DEVICES=0 python "$run_dir/build_models.py" --seed 3409 > "$run_dir/build_3409.log" 2>&1
python "$run_dir/run_queue.py"
python "$run_dir/analyze.py"
