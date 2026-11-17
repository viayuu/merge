#!/usr/bin/env bash
set -euo pipefail
cd /public/home/shiys/merge
run_dir=experiment_runs/20261117_lot_samples_seen5
if [[ "${1:-}" == smoke ]]; then
    CUDA_VISIBLE_DEVICES=0 python "$run_dir/build_models.py" --seed 3407 \
        --ids lot_n64_a1p2_s3407 > "$run_dir/smoke_build_64.log" 2>&1 &
    lot_left=$!
    CUDA_VISIBLE_DEVICES=1 python "$run_dir/build_models.py" --seed 3407 \
        --ids lot_n1000_a1p2_s3407 > "$run_dir/smoke_build_1000.log" 2>&1 &
    lot_right=$!
    wait "$lot_left"
    wait "$lot_right"
    python "$run_dir/run_queue.py" \
        --ids lot_n64_a1p2_s3407,lot_n1000_a1p2_s3407 --episodes 2
else
    CUDA_VISIBLE_DEVICES=0 python "$run_dir/build_models.py" --seed 3407 > "$run_dir/build_3407.log" 2>&1 &
    lot_left=$!
    CUDA_VISIBLE_DEVICES=1 python "$run_dir/build_models.py" --seed 3408 > "$run_dir/build_3408.log" 2>&1 &
    lot_right=$!
    wait "$lot_left"
    wait "$lot_right"
    CUDA_VISIBLE_DEVICES=0 python "$run_dir/build_models.py" --seed 3409 > "$run_dir/build_3409.log" 2>&1
    python "$run_dir/run_queue.py"
    python "$run_dir/analyze.py"
    python "$run_dir/plot_results.py"
fi
