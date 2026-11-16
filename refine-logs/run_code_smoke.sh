#!/usr/bin/env bash
source /public/home/shiys/miniconda3/etc/profile.d/conda.sh
conda activate easynco
cd /public/home/shiys/merge
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python merge_lab/evaluate.py \
  --checkpoints URS/pretrained/unified_checkpoint_500.pt checkpoints/average_2.pt \
  --out-dir results/code_smoke \
  --problem_set cvrp,cvrptw --test_episodes 2 --disable_aug --cuda -1 \
  > refine-logs/code_smoke.log 2>&1
smoke_exit=$?
printf '%s\n' "$smoke_exit" > refine-logs/code_smoke.exit
exit "$smoke_exit"
