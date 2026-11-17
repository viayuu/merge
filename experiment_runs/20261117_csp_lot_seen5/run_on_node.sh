#!/usr/bin/env bash
set -euo pipefail
cd /public/home/shiys/merge
run_dir=experiment_runs/20261117_csp_lot_seen5
if [[ "${1:-}" == smoke ]]; then
    CUDA_VISIBLE_DEVICES=0 python "$run_dir/build_models.py" --seed 3407 \
        --ids csp_n64_a1p4_r5_s3407,csp_n1000_a1p2_r5_s3407 > "$run_dir/smoke_build.log" 2>&1
    python "$run_dir/run_queue.py" \
        --ids csp_n64_a1p4_r5_s3407,csp_n1000_a1p2_r5_s3407 --episodes 2
else
    # Initialize the shared base SVD before concurrent builders, also on a fresh run.
    CUDA_VISIBLE_DEVICES=0 python "$run_dir/build_models.py" --seed 3407 \
        --ids csp_n64_a1p4_r5_s3407 > "$run_dir/initialize.log" 2>&1
    CUDA_VISIBLE_DEVICES=0 python "$run_dir/build_models.py" --seed 3407 > "$run_dir/build_3407.log" 2>&1 &
    csp_left=$!
    (
        CUDA_VISIBLE_DEVICES=1 python "$run_dir/build_models.py" --seed 3408 > "$run_dir/build_3408.log" 2>&1
        CUDA_VISIBLE_DEVICES=1 python "$run_dir/build_models.py" --seed 3409 > "$run_dir/build_3409.log" 2>&1
    ) &
    csp_right=$!
    wait "$csp_left"
    wait "$csp_right"
    python "$run_dir/run_queue.py"
    python "$run_dir/analyze.py"
    python "$run_dir/plot_results.py"
fi
